from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MeshConfig
from .flow import FlowResult
from .geometry import FiveSpotGeometry, symmetric_edge_levels, transfinite_nodes
from .mesh import MeshError, StructuredMesh, rectangular_mesh


@dataclass
class _FieldSampler:
    centers: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    geometry: FiveSpotGeometry

    def __post_init__(self) -> None:
        self._points = self.centers.reshape(-1, 2)
        self._velocity = self.velocity.reshape(-1, 2)
        self._pressure = self.pressure.reshape(-1)
        self._nbx = max(8, self.centers.shape[0])
        self._nby = max(8, self.centers.shape[1])
        self._bins: dict[tuple[int, int], list[int]] = {}
        for index, point in enumerate(self._points):
            key = self._key(point)
            self._bins.setdefault(key, []).append(index)

    def _key(self, point: np.ndarray) -> tuple[int, int]:
        bx = int(np.clip(point[0] / self.geometry.lx * self._nbx, 0, self._nbx - 1))
        by = int(np.clip(point[1] / self.geometry.ly * self._nby, 0, self._nby - 1))
        return bx, by

    def _candidates(self, point: np.ndarray) -> np.ndarray:
        bx, by = self._key(point)
        found: list[int] = []
        for radius in range(0, 5):
            found.clear()
            for ix in range(max(0, bx - radius), min(self._nbx, bx + radius + 1)):
                for iy in range(max(0, by - radius), min(self._nby, by + radius + 1)):
                    found.extend(self._bins.get((ix, iy), ()))
            if len(found) >= 6:
                break
        if not found:
            return np.arange(len(self._points))
        return np.asarray(found, dtype=int)

    def sample(self, point: np.ndarray) -> tuple[np.ndarray, float]:
        ids = self._candidates(point)
        delta = self._points[ids] - point
        distance2 = np.einsum("ij,ij->i", delta, delta)
        take = ids[np.argsort(distance2)[: min(8, len(ids))]]
        d2 = np.sum((self._points[take] - point) ** 2, axis=1)
        weights = 1.0 / np.maximum(d2, 1.0e-20)
        weights /= np.sum(weights)
        velocity = np.sum(self._velocity[take] * weights[:, None], axis=0)
        pressure = float(np.sum(self._pressure[take] * weights))
        return velocity, pressure


def _well_refined_arclength_levels(
    count: int,
    refinement: float,
    core_fraction: float,
) -> np.ndarray:
    """Distribute nodes over the whole path while refining both endpoints.

    The constant part of the monitor guarantees non-zero resolution in the
    middle.  The two inverse-distance terms add nodes near the well arcs
    without taking all nodes away from the central part of the reservoir.
    """
    coordinate = np.linspace(0.0, 1.0, 4097)
    strength = 0.02 * refinement
    monitor = (
        1.0
        + strength / (coordinate + core_fraction)
        + strength / (1.0 - coordinate + core_fraction)
    )
    cumulative = np.zeros_like(coordinate)
    cumulative[1:] = np.cumsum(
        0.5 * (monitor[:-1] + monitor[1:]) * np.diff(coordinate)
    )
    cumulative /= cumulative[-1]
    return np.interp(np.linspace(0.0, 1.0, count + 1), cumulative, coordinate)


def _adaptive_pressure_levels(
    cross_sections: np.ndarray,
    sampler: _FieldSampler,
    geometry: FiveSpotGeometry,
    count: int,
    refinement: float,
    injector_pressure: float,
    producer_pressure: float,
) -> np.ndarray:
    """Choose global isobar levels with spatial coverage and well refinement."""
    segment_length = np.linalg.norm(np.diff(cross_sections, axis=0), axis=2)
    path_length = np.vstack(
        [np.zeros(cross_sections.shape[1]), np.cumsum(segment_length, axis=0)]
    )
    total_length = np.maximum(path_length[-1], 1.0e-30)
    mean_arclength = np.mean(path_length / total_length[None, :], axis=1)
    mean_arclength[0] = 0.0
    mean_arclength[-1] = 1.0
    mean_arclength = np.maximum.accumulate(mean_arclength)

    section_pressure = np.empty(cross_sections.shape[:2], dtype=float)
    for i in range(cross_sections.shape[0]):
        for j in range(cross_sections.shape[1]):
            section_pressure[i, j] = sampler.sample(cross_sections[i, j])[1]
    representative_pressure = np.mean(section_pressure, axis=1)
    pressure_coordinate = (injector_pressure - representative_pressure) / (
        injector_pressure - producer_pressure
    )
    pressure_coordinate[0] = 0.0
    pressure_coordinate[-1] = 1.0
    pressure_coordinate = np.maximum.accumulate(np.clip(pressure_coordinate, 0.0, 1.0))

    mean_total_length = float(np.mean(total_length))
    core_fraction = np.clip(
        0.5 * (geometry.injector_radius + geometry.producer_radius) / mean_total_length,
        1.0e-4,
        2.0e-2,
    )
    target_arclength = _well_refined_arclength_levels(count, refinement, core_fraction)
    levels = np.interp(target_arclength, mean_arclength, pressure_coordinate)
    levels[0] = 0.0
    levels[-1] = 1.0
    # A tiny strictly increasing component prevents duplicate isobars if the
    # coarse pilot solution contains an exactly flat pressure interval.
    epsilon = 1.0e-10
    return (1.0 - epsilon) * levels + epsilon * np.linspace(0.0, 1.0, count + 1)


def _symmetric_geometric_levels(count: int, first_step: float) -> np.ndarray:
    """Coordinate levels with equal small end cells and smooth central growth."""
    half = count // 2
    if half < 1:
        return np.linspace(0.0, 1.0, count + 1)
    uniform = 0.5 / half
    target = float(np.clip(first_step, 1.0e-8, uniform))
    if target >= uniform * (1.0 - 1.0e-12):
        left_widths = np.full(half, uniform)
    else:
        def first_width(ratio: float) -> float:
            return 0.5 * (ratio - 1.0) / (ratio**half - 1.0)

        lo, hi = 1.0 + 1.0e-12, 2.0
        while first_width(hi) > target:
            hi *= 1.5
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if first_width(mid) > target:
                lo = mid
            else:
                hi = mid
        ratio = 0.5 * (lo + hi)
        left_widths = target * ratio ** np.arange(half)
        left_widths *= 0.5 / np.sum(left_widths)

    if count % 2 == 0:
        widths = np.concatenate([left_widths, left_widths[::-1]])
    else:
        center = 1.0 - 2.0 * np.sum(left_widths)
        widths = np.concatenate([left_widths, [center], left_widths[::-1]])
    levels = np.concatenate([[0.0], np.cumsum(widths)])
    levels[-1] = 1.0
    return levels


def _elliptic_smooth(nodes: np.ndarray, iterations: int = 180) -> np.ndarray:
    """Laplace smoothing of interior nodes while retaining exact boundaries."""
    result = np.asarray(nodes, dtype=float).copy()
    for _ in range(iterations):
        previous = result.copy()
        result[1:-1, 1:-1] = 0.25 * (
            previous[:-2, 1:-1]
            + previous[2:, 1:-1]
            + previous[1:-1, :-2]
            + previous[1:-1, 2:]
        )
    return result


def _flux_start_points(pilot: StructuredMesh, flow: FlowResult, geometry: FiveSpotGeometry, ny: int) -> np.ndarray:
    segment_flux = np.maximum(flow.xi_flux[0], 0.0)
    total = float(np.sum(segment_flux))
    if total <= 0.0:
        raise MeshError("pilot solution has no inflow through the injector")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_flux) / total))
    fractions = np.linspace(0.0, 1.0, ny + 1)
    eta_pilot = np.linspace(0.0, 1.0, pilot.ny + 1)
    eta = np.interp(fractions, cumulative, eta_pilot)
    return geometry.injector_arc(eta)


def _circle_intersection(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    d = b - a
    offset = a - center
    roots = np.roots([float(np.dot(d, d)), 2.0 * float(np.dot(offset, d)), float(np.dot(offset, offset) - radius**2)])
    valid = [float(root.real) for root in roots if abs(root.imag) < 1.0e-10 and 0.0 <= root.real <= 1.0]
    if valid:
        return a + min(valid) * d
    direction = b - center
    return center + radius * direction / np.linalg.norm(direction)


def _trace_streamline(
    start: np.ndarray,
    sampler: _FieldSampler,
    geometry: FiveSpotGeometry,
    step: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = [np.asarray(start, dtype=float)]
    _, p0 = sampler.sample(points[-1])
    pressures = [p0]
    for _ in range(max_steps):
        current = points[-1]
        velocity, _ = sampler.sample(current)
        speed = float(np.linalg.norm(velocity))
        if speed <= 1.0e-30:
            raise MeshError("streamline reached a near-stagnation region")
        direction = velocity / speed
        midpoint = current + 0.5 * step * direction
        velocity_mid, _ = sampler.sample(midpoint)
        speed_mid = float(np.linalg.norm(velocity_mid))
        if speed_mid <= 1.0e-30:
            raise MeshError("streamline reached a near-stagnation region")
        candidate = current + step * velocity_mid / speed_mid
        distance = np.linalg.norm(candidate - geometry.producer_center)
        if distance <= geometry.producer_radius + step:
            endpoint = _circle_intersection(current, candidate, geometry.producer_center, geometry.producer_radius)
            points.append(endpoint)
            pressures.append(sampler.sample(endpoint)[1])
            break
        trial_step = step
        for _ in range(10):
            if geometry.contains(candidate, tolerance=1.0e-9):
                break
            trial_step *= 0.5
            candidate = current + trial_step * velocity_mid / speed_mid
        else:
            raise MeshError("streamline left the calculation domain")
        points.append(candidate)
        pressures.append(sampler.sample(candidate)[1])
    else:
        raise MeshError("streamline did not reach the producer")
    return np.asarray(points), np.asarray(pressures)


def _resample_by_pressure(
    points: np.ndarray,
    pressures: np.ndarray,
    levels: np.ndarray,
    injector_pressure: float,
    producer_pressure: float,
) -> np.ndarray:
    coordinate = (injector_pressure - pressures) / (injector_pressure - producer_pressure)
    coordinate[0] = 0.0
    coordinate[-1] = 1.0
    coordinate = np.maximum.accumulate(np.clip(coordinate, 0.0, 1.0))
    keep = np.concatenate(([True], np.diff(coordinate) > 1.0e-11))
    coordinate = coordinate[keep]
    points = points[keep]
    if coordinate[-1] < 1.0:
        coordinate = np.append(coordinate, 1.0)
        points = np.vstack([points, points[-1]])
    return np.column_stack(
        [np.interp(levels, coordinate, points[:, 0]), np.interp(levels, coordinate, points[:, 1])]
    )


def build_flow_aligned_mesh(
    geometry: FiveSpotGeometry,
    pilot: StructuredMesh | None,
    pilot_flow: FlowResult | None,
    config: MeshConfig,
    injector_pressure: float,
    producer_pressure: float,
) -> StructuredMesh:
    """Build a smooth, boundary-fitted grid for the five-spot cell.

    A single structured block cannot remain both exactly flow-aligned and
    shape-regular close to a finite-radius well.  The former construction
    prioritised noisy interpolated streamlines and produced crossed, severely
    skewed cells.  This construction prioritises a valid full-domain mesh:
    it is boundary-fitted and smoothly graded at wells.
    The pilot flow remains solved and available for diagnostics.
    """
    if config.generator == "point_source_isobars":
        from .point_source_mesh import build_point_source_isobar_mesh

        return build_point_source_isobar_mesh(geometry, config)
    if config.generator == "uniform_cartesian":
        return rectangular_mesh(
            geometry.lx,
            geometry.ly,
            config.nx,
            config.ny,
            geometry.thickness,
        )

    del pilot, pilot_flow, injector_pressure, producer_pressure
    mean_side = 0.5 * (geometry.lx + geometry.ly)
    arc_step = 0.5 * np.pi * max(geometry.injector_radius, geometry.producer_radius) / config.ny
    first_step = arc_step / mean_side / max(config.pressure_bias, 1.0e-6)
    xi_levels = _symmetric_geometric_levels(config.nx, first_step)
    eta_levels = symmetric_edge_levels(
        config.ny, 0.20 * config.streamline_edge_fraction
    )
    nodes = transfinite_nodes(
        geometry,
        config.nx,
        config.ny,
        xi_levels=xi_levels,
        eta_levels=eta_levels,
    )
    result = StructuredMesh(nodes, geometry.thickness)
    result.assert_valid(minimum_angle_deg=40.0)
    return result
