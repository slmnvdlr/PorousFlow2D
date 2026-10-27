from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg

try:
    import pyamg
except ImportError:  # pragma: no cover - exercised by installation diagnostics
    pyamg = None

from .config import BoundaryConfig, NumericsConfig
from .fields import MaterialProperties
from .mesh import StructuredMesh


class LinearSolveError(RuntimeError):
    pass


_AMG_REBUILD_MOBILITY_CHANGE = 0.5
_AMG_REBUILD_ITERATIONS = 40


@dataclass(frozen=True)
class CornerPointWellModel:
    """Pressure-controlled quarter wells in opposite corner cells."""

    injector_radius: float
    producer_radius: float
    skin: float = 0.0
    injector_corner_skin: float = 0.0
    producer_corner_skin: float = 0.0
    correction_mode: str = "none"
    uncorrected_rate: float = 0.0
    target_rate: float = 0.0
    added_total_resistance: float = 0.0


@dataclass(frozen=True)
class FlowResult:
    pressure: np.ndarray
    xi_flux: np.ndarray
    eta_flux: np.ndarray
    cell_velocity: np.ndarray | None
    iterations: int
    residual_norm: float
    mass_balance_error: float
    injector_rate: float
    producer_rate: float
    well_source: np.ndarray
    injector_well_index: float = 0.0
    producer_well_index: float = 0.0
    injector_equivalent_radius: float = 0.0
    producer_equivalent_radius: float = 0.0


@dataclass
class PressureWorkspace:
    """Reusable geometry, sparse pattern, and AMG state for pressure solves.

    Only the mobility-dependent matrix values change during a simulation.  The
    mesh geometry, permeability projections, CSR sparsity pattern, and index
    locations are constructed once.  Reusing those objects is exact: every
    solve still receives the current transmissibilities and well indices.
    """

    mesh: StructuredMesh
    material: MaterialProperties
    boundary: BoundaryConfig
    numerics: NumericsConfig
    corner_wells: CornerPointWellModel | None = None
    matrix: object = field(init=False, repr=False)
    _diag_positions: np.ndarray = field(init=False, repr=False)
    _xi_forward_positions: np.ndarray = field(init=False, repr=False)
    _xi_backward_positions: np.ndarray = field(init=False, repr=False)
    _eta_forward_positions: np.ndarray = field(init=False, repr=False)
    _eta_backward_positions: np.ndarray = field(init=False, repr=False)
    _xi_boundary_left: np.ndarray = field(init=False, repr=False)
    _xi_boundary_right: np.ndarray = field(init=False, repr=False)
    _xi_left_geometry: np.ndarray = field(init=False, repr=False)
    _xi_right_geometry: np.ndarray = field(init=False, repr=False)
    _eta_bottom_geometry: np.ndarray = field(init=False, repr=False)
    _eta_top_geometry: np.ndarray = field(init=False, repr=False)
    _amg_solver: object | None = field(default=None, init=False, repr=False)
    _amg_mobility: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_iterations: int = field(default=0, init=False, repr=False)
    amg_rebuilds: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if pyamg is None:
            raise LinearSolveError(
                "PyAMG не установлен. Выполните: python -m pip install pyamg"
            )
        self._build_geometry()
        self._build_sparse_pattern()

    def _build_geometry(self) -> None:
        mesh = self.mesh
        centers = mesh.cell_centers
        permeability = self.material.permeability
        nx, ny = mesh.shape

        left_normal = mesh.xi_face_normals[0]
        left_area = mesh.xi_face_areas[0]
        left_face = mesh.xi_face_centers[0]
        left_distance = np.einsum(
            "ij,ij->i", centers[0] - left_face, left_normal
        )
        self._xi_boundary_left = (
            left_area
            * _normal_permeability(permeability[0], left_normal)
            / np.maximum(left_distance, 1.0e-30)
        )

        right_normal = mesh.xi_face_normals[nx]
        right_area = mesh.xi_face_areas[nx]
        right_face = mesh.xi_face_centers[nx]
        right_distance = np.einsum(
            "ij,ij->i", right_face - centers[-1], right_normal
        )
        self._xi_boundary_right = (
            right_area
            * _normal_permeability(permeability[-1], right_normal)
            / np.maximum(right_distance, 1.0e-30)
        )

        if nx > 1:
            normal = mesh.xi_face_normals[1:nx]
            area = mesh.xi_face_areas[1:nx]
            face = mesh.xi_face_centers[1:nx]
            left_distance = np.einsum(
                "ijk,ijk->ij", face - centers[:-1], normal
            )
            right_distance = np.einsum(
                "ijk,ijk->ij", centers[1:] - face, normal
            )
            self._xi_left_geometry = (
                area
                * _normal_permeability(permeability[:-1], normal)
                / np.maximum(left_distance, 1.0e-30)
            )
            self._xi_right_geometry = (
                area
                * _normal_permeability(permeability[1:], normal)
                / np.maximum(right_distance, 1.0e-30)
            )
        else:
            self._xi_left_geometry = np.empty((0, ny), dtype=float)
            self._xi_right_geometry = np.empty((0, ny), dtype=float)

        if ny > 1:
            normal = mesh.eta_face_normals[:, 1:ny]
            area = mesh.eta_face_areas[:, 1:ny]
            face = mesh.eta_face_centers[:, 1:ny]
            bottom_distance = np.einsum(
                "ijk,ijk->ij", face - centers[:, :-1], normal
            )
            top_distance = np.einsum(
                "ijk,ijk->ij", centers[:, 1:] - face, normal
            )
            self._eta_bottom_geometry = (
                area
                * _normal_permeability(permeability[:, :-1], normal)
                / np.maximum(bottom_distance, 1.0e-30)
            )
            self._eta_top_geometry = (
                area
                * _normal_permeability(permeability[:, 1:], normal)
                / np.maximum(top_distance, 1.0e-30)
            )
        else:
            self._eta_bottom_geometry = np.empty((nx, 0), dtype=float)
            self._eta_top_geometry = np.empty((nx, 0), dtype=float)

    def transmissibilities(
        self, total_mobility: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nx, ny = self.mesh.shape
        txi = np.zeros((nx + 1, ny), dtype=float)
        teta = np.zeros((nx, ny + 1), dtype=float)
        if self.corner_wells is None:
            txi[0] = self._xi_boundary_left * total_mobility[0]
            txi[nx] = self._xi_boundary_right * total_mobility[-1]
        if nx > 1:
            left = self._xi_left_geometry * total_mobility[:-1]
            right = self._xi_right_geometry * total_mobility[1:]
            txi[1:nx] = left * right / np.maximum(left + right, 1.0e-300)
        if ny > 1:
            bottom = self._eta_bottom_geometry * total_mobility[:, :-1]
            top = self._eta_top_geometry * total_mobility[:, 1:]
            teta[:, 1:ny] = bottom * top / np.maximum(bottom + top, 1.0e-300)
        return txi, teta

    def _build_sparse_pattern(self) -> None:
        nx, ny = self.mesh.shape
        shape = (nx, ny)
        pattern = _sparse_pressure_matrix(
            np.ones((nx + 1, ny), dtype=float),
            np.ones((nx, ny + 1), dtype=float),
            np.ones(shape, dtype=float),
        )
        self.matrix = pattern
        cell_indices = np.arange(nx * ny, dtype=np.int64).reshape(shape)

        def positions(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
            result = np.empty(rows.size, dtype=np.int64)
            for index, (row, column) in enumerate(
                zip(rows.ravel(), columns.ravel(), strict=True)
            ):
                start = pattern.indptr[row]
                stop = pattern.indptr[row + 1]
                local = np.flatnonzero(pattern.indices[start:stop] == column)
                if local.size != 1:
                    raise LinearSolveError("invalid cached pressure-matrix pattern")
                result[index] = start + int(local[0])
            return result.reshape(rows.shape)

        self._diag_positions = positions(cell_indices, cell_indices)
        self._xi_forward_positions = positions(
            cell_indices[:-1], cell_indices[1:]
        )
        self._xi_backward_positions = positions(
            cell_indices[1:], cell_indices[:-1]
        )
        self._eta_forward_positions = positions(
            cell_indices[:, :-1], cell_indices[:, 1:]
        )
        self._eta_backward_positions = positions(
            cell_indices[:, 1:], cell_indices[:, :-1]
        )

    def update_matrix(
        self,
        txi: np.ndarray,
        teta: np.ndarray,
        well_diagonal: np.ndarray,
    ) -> None:
        data = self.matrix.data
        data.fill(0.0)
        data[self._diag_positions] = (
            _diagonal(txi, teta) + well_diagonal
        )
        xi_values = -txi[1:-1]
        data[self._xi_forward_positions] = xi_values
        data[self._xi_backward_positions] = xi_values
        eta_values = -teta[:, 1:-1]
        data[self._eta_forward_positions] = eta_values
        data[self._eta_backward_positions] = eta_values

    def _mobility_change(self, total_mobility: np.ndarray) -> float:
        if self._amg_mobility is None:
            return np.inf
        denominator = np.maximum(np.abs(self._amg_mobility), 1.0e-300)
        return float(
            np.max(np.abs(total_mobility - self._amg_mobility) / denominator)
        )

    def _ensure_amg(self, total_mobility: np.ndarray) -> None:
        rebuild = (
            self._amg_solver is None
            or self._last_iterations > _AMG_REBUILD_ITERATIONS
            or self._mobility_change(total_mobility) > _AMG_REBUILD_MOBILITY_CHANGE
        )
        if not rebuild:
            return
        self._amg_solver = pyamg.smoothed_aggregation_solver(
            self.matrix.copy(),
            symmetry="symmetric",
        )
        self._amg_mobility = np.asarray(total_mobility, dtype=float).copy()
        self.amg_rebuilds += 1

    def solve(
        self,
        right_hand_side: np.ndarray,
        initial_pressure: np.ndarray,
        total_mobility: np.ndarray,
        *,
        relative_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
    ) -> tuple[np.ndarray, int, float]:
        self._ensure_amg(total_mobility)
        preconditioner = self._amg_solver.aspreconditioner(cycle="V")
        iterations = 0

        def count_iteration(_value: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            self.matrix,
            right_hand_side.ravel(),
            x0=np.asarray(initial_pressure, dtype=float).ravel(),
            rtol=(
                self.numerics.pressure_rtol
                if relative_tolerance is None
                else relative_tolerance
            ),
            atol=(
                self.numerics.pressure_atol
                if absolute_tolerance is None
                else absolute_tolerance
            ),
            maxiter=self.numerics.pressure_max_iterations,
            M=preconditioner,
            callback=count_iteration,
        )
        if info != 0 or not np.all(np.isfinite(solution)):
            reason = (
                f"не достигнута сходимость за {info} итераций"
                if info > 0
                else "численный сбой CG"
            )
            raise LinearSolveError(f"PyAMG/CG: {reason}")
        pressure = solution.reshape(self.mesh.shape)
        residual = float(
            np.linalg.norm((self.matrix @ solution - right_hand_side.ravel()))
        )
        self._last_iterations = iterations
        return pressure, iterations, residual


def _normal_permeability(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...ij,...j->...", n, k, n)


def _transmissibilities(
    mesh: StructuredMesh,
    material: MaterialProperties,
    total_mobility: np.ndarray,
    pressure_xi_boundaries: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = mesh.shape
    centers = mesh.cell_centers
    k = material.permeability
    txi = np.zeros((nx + 1, ny), dtype=float)
    teta = np.zeros((nx, ny + 1), dtype=float)

    for i in range(nx + 1):
        n = mesh.xi_face_normals[i]
        area = mesh.xi_face_areas[i]
        face = mesh.xi_face_centers[i]
        if i == 0:
            if not pressure_xi_boundaries:
                continue
            distance = np.einsum("ij,ij->i", centers[0] - face, n)
            kn = _normal_permeability(k[0], n)
            txi[i] = area * total_mobility[0] * kn / np.maximum(distance, 1.0e-30)
        elif i == nx:
            if not pressure_xi_boundaries:
                continue
            distance = np.einsum("ij,ij->i", face - centers[-1], n)
            kn = _normal_permeability(k[-1], n)
            txi[i] = area * total_mobility[-1] * kn / np.maximum(distance, 1.0e-30)
        else:
            dl = np.einsum("ij,ij->i", face - centers[i - 1], n)
            dr = np.einsum("ij,ij->i", centers[i] - face, n)
            kl = _normal_permeability(k[i - 1], n)
            kr = _normal_permeability(k[i], n)
            gl = area * total_mobility[i - 1] * kl / np.maximum(dl, 1.0e-30)
            gr = area * total_mobility[i] * kr / np.maximum(dr, 1.0e-30)
            txi[i] = gl * gr / np.maximum(gl + gr, 1.0e-300)

    for j in range(1, ny):
        n = mesh.eta_face_normals[:, j]
        area = mesh.eta_face_areas[:, j]
        face = mesh.eta_face_centers[:, j]
        db = np.einsum("ij,ij->i", face - centers[:, j - 1], n)
        dt = np.einsum("ij,ij->i", centers[:, j] - face, n)
        kb = _normal_permeability(k[:, j - 1], n)
        kt = _normal_permeability(k[:, j], n)
        gb = area * total_mobility[:, j - 1] * kb / np.maximum(db, 1.0e-30)
        gt = area * total_mobility[:, j] * kt / np.maximum(dt, 1.0e-30)
        teta[:, j] = gb * gt / np.maximum(gb + gt, 1.0e-300)
    return txi, teta


def quarter_well_productivity(
    cell_dx: float,
    cell_dy: float,
    thickness: float,
    permeability: np.ndarray,
    total_mobility: float,
    well_radius: float,
    skin: float = 0.0,
) -> tuple[float, float]:
    """Return quarter-well productivity and Peaceman equivalent radius.

    Reflection across the two no-flow sides turns the corner cell into a
    ``2*dx`` by ``2*dy`` cell with a full well at its centre.  The Peaceman
    index of that reflected cell is then divided by four, hence the sector
    angle ``pi/2`` in the numerator below.
    """
    tensor = np.asarray(permeability, dtype=float)
    if tensor.shape != (2, 2):
        raise ValueError("corner-well permeability must be a 2x2 tensor")
    scale = max(abs(float(tensor[0, 0])), abs(float(tensor[1, 1])), 1.0e-300)
    if abs(float(tensor[0, 1])) > 1.0e-10 * scale or abs(float(tensor[1, 0])) > 1.0e-10 * scale:
        raise ValueError(
            "Для аналитической формулы точечной скважины главные оси "
            "проницаемости должны совпадать с осями равномерной сетки"
        )
    kx = float(tensor[0, 0])
    ky = float(tensor[1, 1])
    if min(cell_dx, cell_dy, thickness, kx, ky, total_mobility, well_radius) <= 0.0:
        raise ValueError("corner-well dimensions and properties must be positive")
    ratio_yx = ky / kx
    ratio_xy = kx / ky
    reflected_dx = 2.0 * cell_dx
    reflected_dy = 2.0 * cell_dy
    equivalent_radius = 0.28 * np.sqrt(
        np.sqrt(ratio_yx) * reflected_dx**2
        + np.sqrt(ratio_xy) * reflected_dy**2
    ) / (ratio_yx**0.25 + ratio_xy**0.25)
    denominator = np.log(equivalent_radius / well_radius) + skin
    if denominator <= 0.0:
        raise ValueError(
            "Радиус скважины должен быть меньше эквивалентного радиуса "
            "угловой ячейки: "
            f"r_w={well_radius:.6g} м, r_e={equivalent_radius:.6g} м"
        )
    productivity = (
        0.5
        * np.pi
        * thickness
        * np.sqrt(kx * ky)
        * total_mobility
        / denominator
    )
    return float(productivity), float(equivalent_radius)


def _matvec(p: np.ndarray, txi: np.ndarray, teta: np.ndarray) -> np.ndarray:
    result = np.zeros_like(p)
    result[0] += txi[0] * p[0]
    result[-1] += txi[-1] * p[-1]
    delta = p[:-1] - p[1:]
    result[:-1] += txi[1:-1] * delta
    result[1:] -= txi[1:-1] * delta
    delta = p[:, :-1] - p[:, 1:]
    result[:, :-1] += teta[:, 1:-1] * delta
    result[:, 1:] -= teta[:, 1:-1] * delta
    return result


def _diagonal(txi: np.ndarray, teta: np.ndarray) -> np.ndarray:
    return txi[:-1] + txi[1:] + teta[:, :-1] + teta[:, 1:]


def _sparse_pressure_matrix(
    txi: np.ndarray,
    teta: np.ndarray,
    well_diagonal: np.ndarray,
):
    """Assemble the symmetric five-point pressure operator in CSR format."""
    nx, ny = well_diagonal.shape
    indices = np.arange(nx * ny, dtype=np.int64).reshape(nx, ny)
    rows = [indices.ravel()]
    columns = [indices.ravel()]
    values = [(_diagonal(txi, teta) + well_diagonal).ravel()]

    xi_values = txi[1:-1].ravel()
    rows.extend((indices[:-1].ravel(), indices[1:].ravel()))
    columns.extend((indices[1:].ravel(), indices[:-1].ravel()))
    values.extend((-xi_values, -xi_values))

    eta_values = teta[:, 1:-1].ravel()
    rows.extend((indices[:, :-1].ravel(), indices[:, 1:].ravel()))
    columns.extend((indices[:, 1:].ravel(), indices[:, :-1].ravel()))
    values.extend((-eta_values, -eta_values))

    return coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(nx * ny, nx * ny),
    ).tocsr()


def _pcg(matvec, b: np.ndarray, x: np.ndarray, diagonal: np.ndarray, numerics: NumericsConfig):
    r = b - matvec(x)
    z = r / np.maximum(diagonal, 1.0e-300)
    direction = z.copy()
    rz = float(np.sum(r * z))
    initial = float(np.linalg.norm(r.ravel()))
    tolerance = max(numerics.pressure_atol, numerics.pressure_rtol * max(initial, np.linalg.norm(b.ravel())))
    if initial <= tolerance:
        return x, 0, initial
    for iteration in range(1, numerics.pressure_max_iterations + 1):
        ad = matvec(direction)
        denominator = float(np.sum(direction * ad))
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise LinearSolveError("pressure operator is not positive definite")
        alpha = rz / denominator
        x = x + alpha * direction
        r = r - alpha * ad
        norm = float(np.linalg.norm(r.ravel()))
        if norm <= tolerance:
            return x, iteration, norm
        z = r / np.maximum(diagonal, 1.0e-300)
        rz_new = float(np.sum(r * z))
        direction = z + (rz_new / rz) * direction
        rz = rz_new
    raise LinearSolveError(f"pressure solve did not converge in {numerics.pressure_max_iterations} iterations")


def _cell_velocity(mesh: StructuredMesh, qxi: np.ndarray, qeta: np.ndarray) -> np.ndarray:
    velocity = np.zeros(mesh.shape + (2,), dtype=float)
    for i in range(mesh.nx):
        for j in range(mesh.ny):
            normals = np.stack(
                [
                    -mesh.xi_face_normals[i, j],
                    mesh.xi_face_normals[i + 1, j],
                    -mesh.eta_face_normals[i, j],
                    mesh.eta_face_normals[i, j + 1],
                ]
            )
            normal_velocities = np.array(
                [
                    -qxi[i, j] / mesh.xi_face_areas[i, j],
                    qxi[i + 1, j] / mesh.xi_face_areas[i + 1, j],
                    -qeta[i, j] / mesh.eta_face_areas[i, j],
                    qeta[i, j + 1] / mesh.eta_face_areas[i, j + 1],
                ]
            )
            velocity[i, j], *_ = np.linalg.lstsq(normals, normal_velocities, rcond=None)
    return velocity


def solve_pressure(
    mesh: StructuredMesh,
    material: MaterialProperties,
    total_mobility: np.ndarray,
    boundary: BoundaryConfig,
    numerics: NumericsConfig,
    initial_pressure: np.ndarray | None = None,
    reconstruct_velocity: bool = True,
    corner_wells: CornerPointWellModel | None = None,
    workspace: PressureWorkspace | None = None,
) -> FlowResult:
    if total_mobility.shape != mesh.shape or np.any(total_mobility <= 0.0):
        raise ValueError("total mobility must be positive and cell-centred")
    if workspace is None:
        workspace = PressureWorkspace(
            mesh,
            material,
            boundary,
            numerics,
            corner_wells,
        )
    elif (
        workspace.mesh is not mesh
        or workspace.material is not material
        or workspace.corner_wells != corner_wells
    ):
        raise ValueError("pressure workspace does not match the current problem")
    txi, teta = workspace.transmissibilities(total_mobility)
    b = np.zeros(mesh.shape, dtype=float)
    well_diagonal = np.zeros(mesh.shape, dtype=float)
    injector_well_index = producer_well_index = 0.0
    injector_equivalent_radius = producer_equivalent_radius = 0.0
    if corner_wells is None:
        b[0] += txi[0] * boundary.injector_pressure
        b[-1] += txi[-1] * boundary.producer_pressure
    else:
        injector_dx = float(np.linalg.norm(mesh.nodes[1, 0] - mesh.nodes[0, 0]))
        injector_dy = float(np.linalg.norm(mesh.nodes[0, 1] - mesh.nodes[0, 0]))
        producer_dx = float(np.linalg.norm(mesh.nodes[-1, -1] - mesh.nodes[-2, -1]))
        producer_dy = float(np.linalg.norm(mesh.nodes[-1, -1] - mesh.nodes[-1, -2]))
        injector_well_index, injector_equivalent_radius = quarter_well_productivity(
            injector_dx,
            injector_dy,
            mesh.thickness,
            material.permeability[0, 0],
            float(total_mobility[0, 0]),
            corner_wells.injector_radius,
            corner_wells.skin + corner_wells.injector_corner_skin,
        )
        producer_well_index, producer_equivalent_radius = quarter_well_productivity(
            producer_dx,
            producer_dy,
            mesh.thickness,
            material.permeability[-1, -1],
            float(total_mobility[-1, -1]),
            corner_wells.producer_radius,
            corner_wells.skin + corner_wells.producer_corner_skin,
        )
        well_diagonal[0, 0] += injector_well_index
        well_diagonal[-1, -1] += producer_well_index
        b[0, 0] += injector_well_index * boundary.injector_pressure
        b[-1, -1] += producer_well_index * boundary.producer_pressure

    operator = lambda p: _matvec(p, txi, teta) + well_diagonal * p
    workspace.update_matrix(txi, teta, well_diagonal)
    if initial_pressure is None:
        centers = mesh.cell_centers
        lx = max(float(np.max(mesh.nodes[..., 0]) - np.min(mesh.nodes[..., 0])), 1.0e-30)
        ly = max(float(np.max(mesh.nodes[..., 1]) - np.min(mesh.nodes[..., 1])), 1.0e-30)
        fraction = 0.5 * (centers[..., 0] / lx + centers[..., 1] / ly)
        initial_pressure = boundary.injector_pressure + fraction * (
            boundary.producer_pressure - boundary.injector_pressure
        )
    pressure, iterations, residual = workspace.solve(
        b,
        np.asarray(initial_pressure, dtype=float),
        total_mobility,
    )

    total_iterations = iterations
    for refinement in range(6):
        qxi = np.zeros_like(txi)
        qxi[1:-1] = txi[1:-1] * (pressure[:-1] - pressure[1:])
        qeta = np.zeros_like(teta)
        qeta[:, 1:-1] = teta[:, 1:-1] * (pressure[:, :-1] - pressure[:, 1:])
        well_source = np.zeros(mesh.shape, dtype=float)
        if corner_wells is None:
            qxi[0] = txi[0] * (boundary.injector_pressure - pressure[0])
            qxi[-1] = txi[-1] * (pressure[-1] - boundary.producer_pressure)
            injector_rate = float(np.sum(qxi[0]))
            producer_rate = float(np.sum(qxi[-1]))
        else:
            injector_rate = injector_well_index * (
                boundary.injector_pressure - float(pressure[0, 0])
            )
            producer_rate = producer_well_index * (
                float(pressure[-1, -1]) - boundary.producer_pressure
            )
            well_source[0, 0] = injector_rate
            well_source[-1, -1] = -producer_rate
        imbalance = (
            qxi[1:]
            - qxi[:-1]
            + qeta[:, 1:]
            - qeta[:, :-1]
            - well_source
        )
        scale = max(abs(injector_rate), abs(producer_rate), 1.0e-300)
        mass_error = float(np.sum(np.abs(imbalance)) / scale)
        if mass_error <= 1.0e-8:
            break
        if refinement == 5:
            raise LinearSolveError(
                f"pressure fluxes are not locally conservative: mass error={mass_error:.3e}"
            )
        pressure, added_iterations, residual = workspace.solve(
            b,
            pressure,
            total_mobility,
            relative_tolerance=min(
                numerics.pressure_rtol,
                1.0e-10 * 10.0**(-refinement),
            ),
            absolute_tolerance=min(
                numerics.pressure_atol,
                1.0e-15 * 10.0**(-refinement),
            ),
        )
        total_iterations += added_iterations
    return FlowResult(
        pressure=pressure,
        xi_flux=qxi,
        eta_flux=qeta,
        cell_velocity=_cell_velocity(mesh, qxi, qeta) if reconstruct_velocity else None,
        iterations=total_iterations,
        residual_norm=residual,
        mass_balance_error=mass_error,
        injector_rate=injector_rate,
        producer_rate=producer_rate,
        well_source=well_source,
        injector_well_index=injector_well_index,
        producer_well_index=producer_well_index,
        injector_equivalent_radius=injector_equivalent_radius,
        producer_equivalent_radius=producer_equivalent_radius,
    )
