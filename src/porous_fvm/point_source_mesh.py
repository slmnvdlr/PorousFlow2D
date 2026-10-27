from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .config import MeshConfig
from .geometry import FiveSpotGeometry
from .mesh import MeshError, StructuredMesh


@dataclass(frozen=True)
class PointSourceDiagnostics:
    injector_level: float
    producer_level: float
    injector_min_radius: float
    injector_max_radius: float
    producer_min_radius: float
    producer_max_radius: float
    wall_normal_velocity_error: float


@dataclass(frozen=True)
class CoordinateQuality:
    internal_max_deviation_deg: float
    internal_rms_deviation_deg: float
    boundary_max_deviation_deg: float
    boundary_rms_deviation_deg: float
    boundary_sample_count: int


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class PeriodicPointPair:
    """Complex potential of the reflected five-spot point pair.

    The infinite lattice of images is summed with the Jacobi theta series.
    Its four straight symmetry boundaries are streamlines to roundoff.
    """

    def __init__(self, geometry: FiveSpotGeometry) -> None:
        self.geometry = geometry
        self.swapped = geometry.lx > geometry.ly
        self.a = min(geometry.lx, geometry.ly)
        self.b = max(geometry.lx, geometry.ly)
        self.q = float(np.exp(-np.pi * self.b / self.a))
        orders: list[int] = []
        for order in range(32):
            orders.append(order)
            if order >= 5 and self.q ** ((order + 0.5) ** 2) < 1.0e-16:
                break
        self.orders = np.asarray(orders, dtype=float)
        self.harmonics = 2.0 * self.orders + 1.0
        self.coefficients = (
            (-1.0) ** self.orders
            * self.q ** ((self.orders + 0.5) ** 2)
        )
        self.sink = self.a + 1j * self.b
        self.scale = np.pi / (2.0 * self.a)

    def _complex_coordinate(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.shape[-1] != 2:
            raise ValueError("points must end with x,y coordinates")
        if self.swapped:
            return values[..., 1] + 1j * values[..., 0]
        return values[..., 0] + 1j * values[..., 1]

    def _theta(self, argument: np.ndarray) -> np.ndarray:
        flat = np.asarray(argument, dtype=complex).reshape(-1)
        values = np.sum(
            self.coefficients[:, None]
            * np.sin(self.harmonics[:, None] * flat[None, :]),
            axis=0,
        )
        return values.reshape(np.asarray(argument).shape)

    def _theta_derivative(self, argument: np.ndarray) -> np.ndarray:
        flat = np.asarray(argument, dtype=complex).reshape(-1)
        values = np.sum(
            (self.coefficients * self.harmonics)[:, None]
            * np.cos(self.harmonics[:, None] * flat[None, :]),
            axis=0,
        )
        return values.reshape(np.asarray(argument).shape)

    def potential(self, points: np.ndarray) -> np.ndarray:
        z = self._complex_coordinate(points)
        source_theta = self._theta(self.scale * z)
        sink_theta = self._theta(self.scale * (z - self.sink))
        return (
            np.log(np.abs(source_theta))
            - np.log(np.abs(sink_theta))
            - self.scale * np.imag(z)
        )

    def complex_potential(self, points: np.ndarray) -> np.ndarray:
        """Return a locally continuous branch; unwrap along input curves."""
        z = self._complex_coordinate(points)
        source_theta = self._theta(self.scale * z)
        sink_theta = self._theta(self.scale * (z - self.sink))
        return (
            np.log(source_theta)
            - np.log(sink_theta)
            + 1j * self.scale * z
        )

    def gradient(self, points: np.ndarray) -> np.ndarray:
        z = self._complex_coordinate(points)
        source_argument = self.scale * z
        sink_argument = self.scale * (z - self.sink)
        derivative = self.scale * (
            self._theta_derivative(source_argument) / self._theta(source_argument)
            - self._theta_derivative(sink_argument) / self._theta(sink_argument)
            + 1j
        )
        gradient_prime = np.stack((np.real(derivative), -np.imag(derivative)), axis=-1)
        if self.swapped:
            return gradient_prime[..., ::-1]
        return gradient_prime


def _nominal_circle_points(
    geometry: FiveSpotGeometry, radius: float, producer: bool, count: int = 4097
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 0.5 * np.pi, count)
    if producer:
        points = np.column_stack(
            (
                geometry.lx - radius * np.sin(theta),
                geometry.ly - radius * np.cos(theta),
            )
        )
    else:
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    return theta, points


def _radial_point(
    geometry: FiveSpotGeometry, theta: float, radius: float, producer: bool
) -> np.ndarray:
    if producer:
        return np.array(
            [
                geometry.lx - radius * np.sin(theta),
                geometry.ly - radius * np.cos(theta),
            ]
        )
    return np.array([radius * np.cos(theta), radius * np.sin(theta)])


def _radial_limit(
    geometry: FiveSpotGeometry, theta: float, producer: bool
) -> float:
    sine = max(float(np.sin(theta)), 1.0e-14)
    cosine = max(float(np.cos(theta)), 1.0e-14)
    if producer:
        return min(geometry.lx / sine, geometry.ly / cosine)
    return min(geometry.lx / cosine, geometry.ly / sine)


def _isobar(
    field: PeriodicPointPair,
    geometry: FiveSpotGeometry,
    requested_radius: float,
    producer: bool,
    sample_count: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    theta_circle, circle = _nominal_circle_points(
        geometry, requested_radius, producer
    )
    circle_values = field.potential(circle)
    # The first contour encountered from the reservoir is tangent to the
    # nominal circle and never enters it.
    level = float(np.min(circle_values) if producer else np.max(circle_values))
    theta = np.linspace(0.0, 0.5 * np.pi, sample_count)
    radii = np.empty_like(theta)
    for index, angle in enumerate(theta):
        def residual(radius: float) -> float:
            point = _radial_point(geometry, angle, radius, producer)
            return float(field.potential(point) - level)

        lower = requested_radius
        lower_value = residual(lower)
        if abs(lower_value) <= 2.0e-13:
            radii[index] = lower
            continue
        limit = 0.999999 * _radial_limit(geometry, angle, producer)
        upper = min(limit, max(1.05 * lower, lower + 1.0e-10 * max(geometry.lx, geometry.ly)))
        upper_value = residual(upper)
        while lower_value * upper_value > 0.0 and upper < limit:
            upper = min(limit, 1.35 * upper)
            upper_value = residual(upper)
        if lower_value * upper_value > 0.0:
            raise MeshError(
                "could not locate the finite-radius isobar at "
                f"theta={np.degrees(angle):.4g}°"
            )
        radii[index] = brentq(residual, lower, upper, xtol=1.0e-12, rtol=1.0e-12)
    points = np.vstack(
        [_radial_point(geometry, angle, radius, producer) for angle, radius in zip(theta, radii)]
    )
    return level, theta, points


def _uniform_streamfunction_starts(
    field: PeriodicPointPair,
    theta: np.ndarray,
    contour: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    complex_values = field.complex_potential(contour)
    psi = np.unwrap(np.imag(complex_values))
    direction = np.sign(psi[-1] - psi[0])
    if direction == 0.0:
        raise MeshError("the injector isobar has zero stream-function range")
    monotone = direction * psi
    if np.min(np.diff(monotone)) < -1.0e-8:
        raise MeshError("stream function is not monotone along the injector isobar")
    monotone = np.maximum.accumulate(monotone)
    keep = np.concatenate(([True], np.diff(monotone) > 1.0e-13))
    targets = np.linspace(monotone[0], monotone[-1], count + 1)
    target_theta = np.interp(targets, monotone[keep], theta[keep])
    x = np.interp(target_theta, theta, contour[:, 0])
    y = np.interp(target_theta, theta, contour[:, 1])
    return np.column_stack((x, y)), direction * targets


def _merge_required_levels(
    levels: np.ndarray, required: list[float]
) -> np.ndarray:
    result = np.asarray(levels, dtype=float).copy()
    spacing = float(np.median(np.diff(result)))
    clustered: list[float] = []
    for value in sorted(required):
        if clustered and abs(value - clustered[-1]) < 0.05 * spacing:
            clustered[-1] = 0.5 * (clustered[-1] + value)
        else:
            clustered.append(float(value))
    used: set[int] = set()
    for value in clustered:
        if value <= result[0] or value >= result[-1]:
            continue
        candidates = np.argsort(np.abs(result[1:-1] - value)) + 1
        index = next((int(item) for item in candidates if int(item) not in used), None)
        if index is None:
            raise MeshError("not enough isobar levels for the outer corners")
        result[index] = value
        used.add(index)
    result.sort()
    if np.any(np.diff(result) <= 1.0e-12 * max(1.0, abs(result[-1] - result[0]))):
        raise MeshError("outer-corner potentials produced duplicate isobars")
    return result


def _wall_path(
    field: PeriodicPointPair,
    start: np.ndarray,
    corner: np.ndarray,
    end: np.ndarray,
    levels: np.ndarray,
    samples_per_segment: int,
) -> np.ndarray:
    first = np.linspace(start, corner, samples_per_segment, endpoint=False)
    second = np.linspace(corner, end, samples_per_segment)
    points = np.vstack((first, second))
    potential = field.potential(points)
    scale = max(1.0, abs(levels[-1] - levels[0]))
    reversal = float(np.min(np.diff(potential)))
    if reversal < -2.0e-9 * scale:
        raise MeshError(
            f"potential reverses along a no-flow boundary ({reversal:.3g})"
        )
    potential = np.maximum.accumulate(potential)
    keep = np.concatenate(([True], np.diff(potential) > 1.0e-13 * scale))
    return np.column_stack(
        (
            np.interp(levels, potential[keep], points[keep, 0]),
            np.interp(levels, potential[keep], points[keep, 1]),
        )
    )


def _correct_to_level(
    field: PeriodicPointPair, point: np.ndarray, level: float
) -> np.ndarray:
    result = np.asarray(point, dtype=float).copy()
    for _ in range(4):
        value = float(field.potential(result)) - level
        gradient = np.asarray(field.gradient(result), dtype=float)
        norm2 = float(np.dot(gradient, gradient))
        if norm2 <= 1.0e-28:
            break
        result -= value * gradient / norm2
    return result


def _trace_streamline(
    field: PeriodicPointPair,
    start: np.ndarray,
    levels: np.ndarray,
    domain_scale: float,
) -> np.ndarray:
    def rhs(_potential: float, state: np.ndarray) -> np.ndarray:
        gradient = np.asarray(field.gradient(state), dtype=float)
        norm2 = float(np.dot(gradient, gradient))
        if not np.isfinite(norm2) or norm2 <= 1.0e-28:
            raise MeshError("a streamline reached a stagnation point")
        return gradient / norm2

    solution = solve_ivp(
        rhs,
        (float(levels[0]), float(levels[-1])),
        np.asarray(start, dtype=float),
        t_eval=levels,
        rtol=2.0e-9,
        atol=2.0e-11 * domain_scale,
        max_step=(levels[-1] - levels[0]) / max(240, 6 * len(levels)),
    )
    if not solution.success or solution.y.shape[1] != len(levels):
        raise MeshError(f"streamline integration failed: {solution.message}")
    result = solution.y.T
    result[0] = _correct_to_level(field, result[0], float(levels[0]))
    result[-1] = _correct_to_level(field, result[-1], float(levels[-1]))
    return result


def coordinate_quality(nodes: np.ndarray) -> CoordinateQuality:
    points = np.asarray(nodes, dtype=float)

    def deviation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
        cosine = np.divide(
            np.sum(first * second, axis=-1),
            denominator,
            out=np.full_like(denominator, np.nan),
            where=denominator > 1.0e-30,
        )
        return np.degrees(np.arcsin(np.clip(np.abs(cosine), 0.0, 1.0)))

    along = np.gradient(points, axis=0)
    across = np.gradient(points, axis=1)
    internal = deviation(along[1:-1, 1:-1], across[1:-1, 1:-1]).reshape(-1)
    internal = internal[np.isfinite(internal)]
    boundary_values: list[np.ndarray] = []
    straight_limit = np.cos(np.deg2rad(10.0))
    for edge in (0, -1):
        previous = points[1:-1, edge] - points[:-2, edge]
        following = points[2:, edge] - points[1:-1, edge]
        denominator = np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
        turn_cosine = np.divide(
            np.sum(previous * following, axis=1),
            denominator,
            out=np.full(len(previous), -1.0),
            where=denominator > 1.0e-30,
        )
        regular = turn_cosine >= straight_limit
        tangent = points[2:, edge] - points[:-2, edge]
        inward = (
            points[1:-1, 1] - points[1:-1, 0]
            if edge == 0
            else points[1:-1, -1] - points[1:-1, -2]
        )
        values = deviation(tangent, inward)
        boundary_values.append(values[regular & np.isfinite(values)])
    boundary = np.concatenate(boundary_values)
    return CoordinateQuality(
        internal_max_deviation_deg=float(np.max(internal)),
        internal_rms_deviation_deg=float(np.sqrt(np.mean(internal**2))),
        boundary_max_deviation_deg=float(np.max(boundary)),
        boundary_rms_deviation_deg=float(np.sqrt(np.mean(boundary**2))),
        boundary_sample_count=int(boundary.size),
    )


def _wall_velocity_error(
    field: PeriodicPointPair, geometry: FiveSpotGeometry
) -> float:
    fraction = np.linspace(0.01, 0.99, 101)
    bottom = np.column_stack((fraction * geometry.lx, np.zeros_like(fraction)))
    top = np.column_stack((fraction * geometry.lx, np.full_like(fraction, geometry.ly)))
    left = np.column_stack((np.zeros_like(fraction), fraction * geometry.ly))
    right = np.column_stack((np.full_like(fraction, geometry.lx), fraction * geometry.ly))
    errors = np.concatenate(
        (
            np.abs(field.gradient(bottom)[:, 1]),
            np.abs(field.gradient(top)[:, 1]),
            np.abs(field.gradient(left)[:, 0]),
            np.abs(field.gradient(right)[:, 0]),
        )
    )
    return float(np.max(errors))


def build_point_source_isobar_mesh(
    geometry: FiveSpotGeometry, config: MeshConfig
) -> StructuredMesh:
    """Build a phi-psi grid for the periodic point source/sink solution."""
    field = PeriodicPointPair(geometry)
    contour_samples = max(2001, 80 * config.ny + 1)
    injector_level, injector_theta, injector_contour = _isobar(
        field,
        geometry,
        geometry.injector_radius,
        producer=False,
        sample_count=contour_samples,
    )
    producer_level, _producer_theta, producer_contour = _isobar(
        field,
        geometry,
        geometry.producer_radius,
        producer=True,
        sample_count=contour_samples,
    )
    if not injector_level < producer_level:
        raise MeshError(
            "the selected injector and producer isobars overlap; reduce the well radii"
        )

    starts, psi_levels = _uniform_streamfunction_starts(
        field, injector_theta, injector_contour, config.ny
    )
    phi_levels = np.linspace(injector_level, producer_level, config.nx + 1)
    corner_a = float(field.potential(np.array([geometry.lx, 0.0])))
    corner_b = float(field.potential(np.array([0.0, geometry.ly])))
    phi_levels = _merge_required_levels(phi_levels, [corner_a, corner_b])

    nodes = np.empty((config.nx + 1, config.ny + 1, 2), dtype=float)
    nodes[:, 0] = _wall_path(
        field,
        injector_contour[0],
        np.array([geometry.lx, 0.0]),
        producer_contour[0],
        phi_levels,
        max(2001, 50 * config.nx + 1),
    )
    nodes[:, -1] = _wall_path(
        field,
        injector_contour[-1],
        np.array([0.0, geometry.ly]),
        producer_contour[-1],
        phi_levels,
        max(2001, 50 * config.nx + 1),
    )
    domain_scale = max(geometry.lx, geometry.ly)
    for j in range(1, config.ny):
        nodes[:, j] = _trace_streamline(
            field, starts[j], phi_levels, domain_scale
        )

    # Use the accurately sampled injector contour for the first row.  The
    # integrated curves already end on the producer level; correct once more
    # to remove accumulated ODE error without imposing a circular boundary.
    nodes[0] = starts
    for j in range(1, config.ny):
        nodes[-1, j] = _correct_to_level(
            field, nodes[-1, j], producer_level
        )

    mesh = StructuredMesh(nodes, geometry.thickness)
    quality = mesh.quality()
    if quality.invalid_cells:
        raise MeshError(f"point-source grid contains {quality.invalid_cells} inverted cells")
    if quality.minimum_angle_deg < config.minimum_cell_angle_deg:
        raise MeshError(
            f"minimum cell angle is {quality.minimum_angle_deg:.3g}°, "
            f"required {config.minimum_cell_angle_deg:.3g}°"
        )
    coordinates = coordinate_quality(nodes)
    injector_radius = np.linalg.norm(nodes[0], axis=1)
    producer_radius = np.linalg.norm(
        nodes[-1] - geometry.producer_center, axis=1
    )
    diagnostics = PointSourceDiagnostics(
        injector_level=injector_level,
        producer_level=producer_level,
        injector_min_radius=float(np.min(injector_radius)),
        injector_max_radius=float(np.max(injector_radius)),
        producer_min_radius=float(np.min(producer_radius)),
        producer_max_radius=float(np.max(producer_radius)),
        wall_normal_velocity_error=_wall_velocity_error(field, geometry),
    )
    radius_tolerance = 2.0e-7 * domain_scale
    if diagnostics.injector_min_radius < geometry.injector_radius - radius_tolerance:
        raise MeshError("injector isobar entered the nominal well radius")
    if diagnostics.producer_min_radius < geometry.producer_radius - radius_tolerance:
        raise MeshError("producer isobar entered the nominal well radius")

    mesh.point_source_diagnostics = diagnostics
    mesh.potential_levels = phi_levels
    mesh.streamfunction_levels = psi_levels
    mesh.coordinate_mode = "periodic_point_source_sink"
    mesh.internal_orthogonality_max_deg = coordinates.internal_max_deviation_deg
    mesh.internal_orthogonality_rms_deg = coordinates.internal_rms_deviation_deg
    mesh.boundary_incidence_max_deg = coordinates.boundary_max_deviation_deg
    mesh.boundary_incidence_rms_deg = coordinates.boundary_rms_deviation_deg
    return mesh
