from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except ImportError:  # Numba is an optional acceleration dependency.
    njit = None

from .config import BoundaryConfig, NumericsConfig, PhaseConfig
from .fields import MaterialProperties
from .flow import FlowResult
from .mesh import StructuredMesh
from .relperm import PowerLawRelativePermeability


class SaturationError(RuntimeError):
    pass


class _OrderedSweepUnavailable(RuntimeError):
    """The fixed-flow transport graph cannot be traversed in pressure order."""


@dataclass(frozen=True)
class TransportStep:
    saturation: np.ndarray
    xi_water_flux: np.ndarray
    eta_water_flux: np.ndarray
    well_water_source: np.ndarray
    injector_water_rate: float
    producer_water_rate: float
    water_volume_before: float
    water_volume_after: float
    boundary_water_rate: float
    balance_error: float
    nonlinear_iterations: int = 0


@dataclass
class TransportOrderCache:
    """Validated downstream cell order reused while flow dependencies agree."""

    order: np.ndarray | None = None
    hits: int = 0
    rebuilds: int = 0

    @staticmethod
    def _is_valid(order: np.ndarray, flow: FlowResult) -> bool:
        nx, ny = flow.pressure.shape
        if order.shape != (nx * ny,):
            return False
        rank = np.empty(nx * ny, dtype=np.int64)
        rank[order] = np.arange(order.size, dtype=np.int64)
        rank = rank.reshape((nx, ny))
        scale = max(
            float(np.max(np.abs(flow.xi_flux))),
            float(np.max(np.abs(flow.eta_flux))),
            1.0e-300,
        )
        tolerance = 1.0e-13 * scale
        xi = flow.xi_flux[1:nx]
        positive = xi > tolerance
        negative = xi < -tolerance
        if np.any(rank[:-1][positive] >= rank[1:][positive]):
            return False
        if np.any(rank[1:][negative] >= rank[:-1][negative]):
            return False
        eta = flow.eta_flux[:, 1:ny]
        positive = eta > tolerance
        negative = eta < -tolerance
        if np.any(rank[:, :-1][positive] >= rank[:, 1:][positive]):
            return False
        if np.any(rank[:, 1:][negative] >= rank[:, :-1][negative]):
            return False
        return True

    def get(self, flow: FlowResult) -> np.ndarray:
        if self.order is not None and self._is_valid(self.order, flow):
            self.hits += 1
            return self.order
        candidate = np.argsort(-flow.pressure.ravel(), kind="stable")
        if not self._is_valid(candidate, flow):
            raise _OrderedSweepUnavailable("pressure order does not match flow graph")
        self.order = candidate
        self.rebuilds += 1
        return candidate


def water_fluxes(
    saturation: np.ndarray,
    flow: FlowResult,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
    boundary: BoundaryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    fw = relperm.fractional_flow(saturation, water, oil)
    qxi = flow.xi_flux
    qeta = flow.eta_flux
    nx, ny = saturation.shape
    qw_xi = np.zeros_like(qxi)
    qw_eta = np.zeros_like(qeta)

    qw_xi[0] = np.where(qxi[0] >= 0.0, qxi[0] * boundary.injected_water_fraction, qxi[0] * fw[0])
    qw_xi[nx] = np.where(qxi[nx] >= 0.0, qxi[nx] * fw[-1], qxi[nx] * fw[-1])
    if nx > 1:
        donor = np.where(qxi[1:nx] >= 0.0, fw[:-1], fw[1:])
        qw_xi[1:nx] = qxi[1:nx] * donor
    if ny > 1:
        donor = np.where(qeta[:, 1:ny] >= 0.0, fw[:, :-1], fw[:, 1:])
        qw_eta[:, 1:ny] = qeta[:, 1:ny] * donor
    return qw_xi, qw_eta


def well_water_sources(
    saturation: np.ndarray,
    flow: FlowResult,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
    boundary: BoundaryConfig,
) -> np.ndarray:
    """Water source per cell; positive injects and negative produces."""
    total_source = flow.well_source
    if not np.any(total_source):
        return np.zeros_like(saturation, dtype=float)
    fractional_flow = relperm.fractional_flow(saturation, water, oil)
    return np.where(
        total_source >= 0.0,
        total_source * boundary.injected_water_fraction,
        total_source * fractional_flow,
    )


def water_well_rates(
    saturation: np.ndarray,
    flow: FlowResult,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
    boundary: BoundaryConfig,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    qw_xi, qw_eta = water_fluxes(saturation, flow, relperm, water, oil, boundary)
    qw_source = well_water_sources(saturation, flow, relperm, water, oil, boundary)
    injector_rate = float(np.sum(qw_xi[0]) + np.sum(np.maximum(qw_source, 0.0)))
    producer_rate = float(np.sum(qw_xi[-1]) + np.sum(np.maximum(-qw_source, 0.0)))
    return injector_rate, producer_rate, qw_xi, qw_eta, qw_source


def stable_time_step(
    mesh: StructuredMesh,
    material: MaterialProperties,
    flow: FlowResult,
    derivative_bound: float,
    target_cfl: float,
) -> float:
    outward = (
        np.maximum(-flow.xi_flux[:-1], 0.0)
        + np.maximum(flow.xi_flux[1:], 0.0)
        + np.maximum(-flow.eta_flux[:, :-1], 0.0)
        + np.maximum(flow.eta_flux[:, 1:], 0.0)
        + np.maximum(-flow.well_source, 0.0)
    )
    pore_volume = material.porosity * mesh.cell_volumes
    local = np.divide(
        target_cfl * pore_volume,
        derivative_bound * outward,
        out=np.full_like(pore_volume, np.inf),
        where=outward > 0.0,
    )
    return float(np.min(local))


def advance_saturation(
    mesh: StructuredMesh,
    material: MaterialProperties,
    saturation: np.ndarray,
    flow: FlowResult,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
    boundary: BoundaryConfig,
    numerics: NumericsConfig,
    dt: float,
) -> TransportStep:
    qw_in, qw_out, qw_xi, qw_eta, qw_source = water_well_rates(
        saturation, flow, relperm, water, oil, boundary
    )
    net_out = (
        qw_xi[1:]
        - qw_xi[:-1]
        + qw_eta[:, 1:]
        - qw_eta[:, :-1]
        - qw_source
    )
    pore_volume = material.porosity * mesh.cell_volumes
    water_before = float(np.sum(pore_volume * saturation))
    updated = saturation - dt * net_out / pore_volume
    lower = relperm.swc
    upper = 1.0 - relperm.sor
    violation = max(float(np.max(lower - updated)), float(np.max(updated - upper)), 0.0)
    if violation > numerics.saturation_tolerance:
        raise SaturationError(
            f"saturation left [{lower}, {upper}] by {violation:.3e}; reduce the time step"
        )
    updated = np.clip(updated, lower, upper)
    water_after = float(np.sum(pore_volume * updated))
    boundary_rate = qw_in - qw_out
    expected = water_before + dt * boundary_rate
    balance_error = abs(water_after - expected) / max(abs(expected), abs(water_before), 1.0e-30)
    return TransportStep(
        saturation=updated,
        xi_water_flux=qw_xi,
        eta_water_flux=qw_eta,
        well_water_source=qw_source,
        injector_water_rate=qw_in,
        producer_water_rate=qw_out,
        water_volume_before=water_before,
        water_volume_after=water_after,
        boundary_water_rate=boundary_rate,
        balance_error=balance_error,
    )


def _fractional_flow_derivative(
    saturation: np.ndarray,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
) -> np.ndarray:
    interval = relperm.mobile_interval
    epsilon = 1.0e-7 * interval
    lower = np.maximum(saturation - epsilon, relperm.swc)
    upper = np.minimum(saturation + epsilon, 1.0 - relperm.sor)
    denominator = upper - lower
    difference = relperm.fractional_flow(upper, water, oil) - relperm.fractional_flow(
        lower, water, oil
    )
    return np.divide(
        difference,
        denominator,
        out=np.zeros_like(saturation, dtype=float),
        where=denominator > 0.0,
    )


def _water_flux_derivative_action(
    direction: np.ndarray,
    derivative: np.ndarray,
    flow: FlowResult,
) -> tuple[np.ndarray, np.ndarray]:
    qxi = flow.xi_flux
    qeta = flow.eta_flux
    nx, ny = direction.shape
    action_xi = np.zeros_like(qxi)
    action_eta = np.zeros_like(qeta)

    action_xi[0] = np.where(
        qxi[0] >= 0.0,
        0.0,
        qxi[0] * derivative[0] * direction[0],
    )
    action_xi[nx] = qxi[nx] * derivative[-1] * direction[-1]
    if nx > 1:
        donor = np.where(
            qxi[1:nx] >= 0.0,
            derivative[:-1] * direction[:-1],
            derivative[1:] * direction[1:],
        )
        action_xi[1:nx] = qxi[1:nx] * donor
    if ny > 1:
        donor = np.where(
            qeta[:, 1:ny] >= 0.0,
            derivative[:, :-1] * direction[:, :-1],
            derivative[:, 1:] * direction[:, 1:],
        )
        action_eta[:, 1:ny] = qeta[:, 1:ny] * donor
    return action_xi, action_eta


def _outward_total_rate(flow: FlowResult) -> np.ndarray:
    return (
        np.maximum(-flow.xi_flux[:-1], 0.0)
        + np.maximum(flow.xi_flux[1:], 0.0)
        + np.maximum(-flow.eta_flux[:, :-1], 0.0)
        + np.maximum(flow.eta_flux[:, 1:], 0.0)
        + np.maximum(-flow.well_source, 0.0)
    )


def _bicgstab(
    operator,
    right_hand_side: np.ndarray,
    diagonal: np.ndarray,
    relative_tolerance: float,
    max_iterations: int,
) -> np.ndarray:
    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    shadow = residual.copy()
    direction = np.zeros_like(residual)
    operator_direction = np.zeros_like(residual)
    rho_previous = alpha = omega = 1.0
    norm_rhs = max(float(np.linalg.norm(right_hand_side.ravel())), 1.0e-300)
    if float(np.linalg.norm(residual.ravel())) <= relative_tolerance * norm_rhs:
        return solution

    for _iteration in range(max_iterations):
        rho = float(np.sum(shadow * residual))
        if not np.isfinite(rho) or abs(rho) <= 1.0e-300:
            raise SaturationError("implicit transport linear solve broke down")
        beta = (rho / rho_previous) * (alpha / omega)
        direction = residual + beta * (direction - omega * operator_direction)
        preconditioned_direction = direction / np.maximum(diagonal, 1.0e-300)
        operator_direction = operator(preconditioned_direction)
        denominator = float(np.sum(shadow * operator_direction))
        if not np.isfinite(denominator) or abs(denominator) <= 1.0e-300:
            raise SaturationError("implicit transport linear solve broke down")
        alpha = rho / denominator
        intermediate = residual - alpha * operator_direction
        if float(np.linalg.norm(intermediate.ravel())) <= relative_tolerance * norm_rhs:
            return solution + alpha * preconditioned_direction

        preconditioned_intermediate = intermediate / np.maximum(diagonal, 1.0e-300)
        operator_intermediate = operator(preconditioned_intermediate)
        denominator = float(np.sum(operator_intermediate * operator_intermediate))
        if not np.isfinite(denominator) or denominator <= 1.0e-300:
            raise SaturationError("implicit transport linear solve broke down")
        omega = float(np.sum(operator_intermediate * intermediate)) / denominator
        if not np.isfinite(omega) or abs(omega) <= 1.0e-300:
            raise SaturationError("implicit transport linear solve broke down")
        solution += alpha * preconditioned_direction + omega * preconditioned_intermediate
        residual = intermediate - omega * operator_intermediate
        if float(np.linalg.norm(residual.ravel())) <= relative_tolerance * norm_rhs:
            return solution
        rho_previous = rho
    raise SaturationError("implicit transport linear solve did not converge")


def _fractional_flow_scalar(
    saturation: float,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
) -> float:
    effective = min(
        1.0,
        max(0.0, (saturation - relperm.swc) / relperm.mobile_interval),
    )
    water_mobility = relperm.krw_end * effective**relperm.nw / water.viscosity
    oil_mobility = relperm.kro_end * (1.0 - effective) ** relperm.no / oil.viscosity
    total = water_mobility + oil_mobility
    return water_mobility / total if total > 0.0 else 0.0


def _fractional_flow_derivative_scalar(
    saturation: float,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
) -> float:
    """Analytic fractional-flow derivative with endpoint safeguards."""
    interval = relperm.mobile_interval
    effective = min(1.0, max(0.0, (saturation - relperm.swc) / interval))
    if effective <= 1.0e-8 or effective >= 1.0 - 1.0e-8:
        increment = 1.0e-7 * interval
        left = max(relperm.swc, saturation - increment)
        right = min(1.0 - relperm.sor, saturation + increment)
        if right <= left:
            return 0.0
        return (
            _fractional_flow_scalar(right, relperm, water, oil)
            - _fractional_flow_scalar(left, relperm, water, oil)
        ) / (right - left)

    water_scale = relperm.krw_end / water.viscosity
    oil_scale = relperm.kro_end / oil.viscosity
    water_mobility = water_scale * effective**relperm.nw
    oil_mobility = oil_scale * (1.0 - effective) ** relperm.no
    water_derivative = (
        water_scale * relperm.nw * effective ** (relperm.nw - 1.0)
    )
    oil_derivative = (
        -oil_scale
        * relperm.no
        * (1.0 - effective) ** (relperm.no - 1.0)
    )
    total = water_mobility + oil_mobility
    if total <= 0.0:
        return 0.0
    derivative_effective = (
        water_derivative * oil_mobility
        - water_mobility * oil_derivative
    ) / total**2
    return derivative_effective / interval


if njit is not None:

    @njit(cache=False)
    def _numba_fractional_flow(
        saturation,
        swc,
        mobile_interval,
        krw_end,
        kro_end,
        nw,
        no,
        water_viscosity,
        oil_viscosity,
    ):
        effective = min(1.0, max(0.0, (saturation - swc) / mobile_interval))
        water_mobility = krw_end * effective**nw / water_viscosity
        oil_mobility = kro_end * (1.0 - effective) ** no / oil_viscosity
        total = water_mobility + oil_mobility
        return water_mobility / total if total > 0.0 else 0.0

    @njit(cache=False)
    def _numba_fractional_flow_derivative(
        saturation,
        swc,
        mobile_interval,
        krw_end,
        kro_end,
        nw,
        no,
        water_viscosity,
        oil_viscosity,
    ):
        effective = min(1.0, max(0.0, (saturation - swc) / mobile_interval))
        if effective <= 1.0e-8 or effective >= 1.0 - 1.0e-8:
            increment = 1.0e-7 * mobile_interval
            left = max(swc, saturation - increment)
            right = min(swc + mobile_interval, saturation + increment)
            if right <= left:
                return 0.0
            return (
                _numba_fractional_flow(
                    right,
                    swc,
                    mobile_interval,
                    krw_end,
                    kro_end,
                    nw,
                    no,
                    water_viscosity,
                    oil_viscosity,
                )
                - _numba_fractional_flow(
                    left,
                    swc,
                    mobile_interval,
                    krw_end,
                    kro_end,
                    nw,
                    no,
                    water_viscosity,
                    oil_viscosity,
                )
            ) / (right - left)
        water_scale = krw_end / water_viscosity
        oil_scale = kro_end / oil_viscosity
        water_mobility = water_scale * effective**nw
        oil_mobility = oil_scale * (1.0 - effective) ** no
        water_derivative = water_scale * nw * effective ** (nw - 1.0)
        oil_derivative = -oil_scale * no * (1.0 - effective) ** (no - 1.0)
        total = water_mobility + oil_mobility
        if total <= 0.0:
            return 0.0
        return (
            (water_derivative * oil_mobility - water_mobility * oil_derivative)
            / total**2
            / mobile_interval
        )

    @njit(cache=False)
    def _numba_solve_ordered_cell(
        old_saturation,
        storage,
        outgoing_rate,
        incoming_water_rate,
        swc,
        mobile_interval,
        krw_end,
        kro_end,
        nw,
        no,
        water_viscosity,
        oil_viscosity,
        relative_tolerance,
    ):
        lower = swc
        upper = swc + mobile_interval

        def residual(value):
            return (
                storage * (value - old_saturation)
                + outgoing_rate
                * _numba_fractional_flow(
                    value,
                    swc,
                    mobile_interval,
                    krw_end,
                    kro_end,
                    nw,
                    no,
                    water_viscosity,
                    oil_viscosity,
                )
                - incoming_water_rate
            )

        lower_residual = residual(lower)
        upper_residual = residual(upper)
        scale = max(
            storage * mobile_interval,
            outgoing_rate,
            incoming_water_rate,
            1.0e-300,
        )
        tolerance = min(relative_tolerance, 1.0e-13) * scale
        if lower_residual >= -tolerance:
            return lower, 0, 0
        if upper_residual <= tolerance:
            return upper, 0, 0
        if lower_residual > 0.0 or upper_residual < 0.0:
            return old_saturation, 0, 1

        left = lower
        right = upper
        value = min(right, max(left, old_saturation))
        for iterations in range(1, 25):
            value_residual = residual(value)
            if abs(value_residual) <= tolerance:
                return value, iterations, 0
            if value_residual < 0.0:
                left = value
            else:
                right = value
            if right - left <= 2.0e-13 * mobile_interval:
                return 0.5 * (left + right), iterations, 0
            derivative = storage + outgoing_rate * _numba_fractional_flow_derivative(
                value,
                swc,
                mobile_interval,
                krw_end,
                kro_end,
                nw,
                no,
                water_viscosity,
                oil_viscosity,
            )
            candidate = value - value_residual / derivative if derivative > 0.0 else np.nan
            if not np.isfinite(candidate) or candidate <= left or candidate >= right:
                candidate = 0.5 * (left + right)
            value = candidate

        for bisection_iteration in range(1, 49):
            value = 0.5 * (left + right)
            value_residual = residual(value)
            if abs(value_residual) <= tolerance:
                return value, 24 + bisection_iteration, 0
            if value_residual < 0.0:
                left = value
            else:
                right = value
        return 0.5 * (left + right), 72, 0

    @njit(cache=False)
    def _numba_ordered_sweep(
        old,
        pore_volume,
        qxi,
        qeta,
        source,
        order,
        injected_water_fraction,
        dt,
        swc,
        mobile_interval,
        krw_end,
        kro_end,
        nw,
        no,
        water_viscosity,
        oil_viscosity,
        relative_tolerance,
    ):
        nx, ny = old.shape
        updated = old.copy()
        solved = np.zeros(old.shape, dtype=np.bool_)
        maximum_flux = max(
            np.max(np.abs(qxi)),
            np.max(np.abs(qeta)),
            np.max(np.abs(source)),
            1.0e-300,
        )
        dependency_tolerance = 1.0e-13 * maximum_flux
        maximum_iterations = 0

        for flat_index in order:
            i = flat_index // ny
            j = flat_index % ny
            incoming_water = 0.0
            outgoing = 0.0

            left_flux = qxi[i, j]
            if left_flux > 0.0:
                if i == 0:
                    incoming_water += left_flux * injected_water_fraction
                else:
                    if not solved[i - 1, j] and left_flux > dependency_tolerance:
                        return updated, maximum_iterations, 1
                    incoming_water += left_flux * _numba_fractional_flow(
                        updated[i - 1, j], swc, mobile_interval, krw_end,
                        kro_end, nw, no, water_viscosity, oil_viscosity,
                    )
            else:
                outgoing += -left_flux

            right_flux = qxi[i + 1, j]
            if right_flux < 0.0:
                if i == nx - 1:
                    return updated, maximum_iterations, 1
                rate = -right_flux
                if not solved[i + 1, j] and rate > dependency_tolerance:
                    return updated, maximum_iterations, 1
                incoming_water += rate * _numba_fractional_flow(
                    updated[i + 1, j], swc, mobile_interval, krw_end,
                    kro_end, nw, no, water_viscosity, oil_viscosity,
                )
            else:
                outgoing += right_flux

            bottom_flux = qeta[i, j]
            if bottom_flux > 0.0:
                if j == 0:
                    return updated, maximum_iterations, 1
                if not solved[i, j - 1] and bottom_flux > dependency_tolerance:
                    return updated, maximum_iterations, 1
                incoming_water += bottom_flux * _numba_fractional_flow(
                    updated[i, j - 1], swc, mobile_interval, krw_end,
                    kro_end, nw, no, water_viscosity, oil_viscosity,
                )
            else:
                outgoing += -bottom_flux

            top_flux = qeta[i, j + 1]
            if top_flux < 0.0:
                if j == ny - 1:
                    return updated, maximum_iterations, 1
                rate = -top_flux
                if not solved[i, j + 1] and rate > dependency_tolerance:
                    return updated, maximum_iterations, 1
                incoming_water += rate * _numba_fractional_flow(
                    updated[i, j + 1], swc, mobile_interval, krw_end,
                    kro_end, nw, no, water_viscosity, oil_viscosity,
                )
            else:
                outgoing += top_flux

            cell_source = source[i, j]
            if cell_source > 0.0:
                incoming_water += cell_source * injected_water_fraction
            else:
                outgoing += -cell_source

            value, iterations, status = _numba_solve_ordered_cell(
                old[i, j],
                pore_volume[i, j] / dt,
                outgoing,
                incoming_water,
                swc,
                mobile_interval,
                krw_end,
                kro_end,
                nw,
                no,
                water_viscosity,
                oil_viscosity,
                relative_tolerance,
            )
            if status != 0:
                return updated, maximum_iterations, status
            updated[i, j] = value
            maximum_iterations = max(maximum_iterations, iterations)
            solved[i, j] = True
        return updated, maximum_iterations, 0

else:
    _numba_ordered_sweep = None


def _solve_ordered_cell(
    old_saturation: float,
    storage: float,
    outgoing_rate: float,
    incoming_water_rate: float,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
    relative_tolerance: float,
) -> tuple[float, int]:
    """Solve one monotone backward-Euler balance by safeguarded Newton."""
    lower = relperm.swc
    upper = 1.0 - relperm.sor

    def residual(value: float) -> float:
        return (
            storage * (value - old_saturation)
            + outgoing_rate * _fractional_flow_scalar(value, relperm, water, oil)
            - incoming_water_rate
        )

    lower_residual = residual(lower)
    upper_residual = residual(upper)
    scale = max(storage * relperm.mobile_interval, outgoing_rate, incoming_water_rate, 1.0e-300)
    # A sweep has one scalar residual per cell.  Keeping every row tighter
    # than the global nonlinear tolerance prevents their conservative balance
    # defects from accumulating over large meshes.
    tolerance = min(relative_tolerance, 1.0e-13) * scale
    if lower_residual >= -tolerance:
        return lower, 0
    if upper_residual <= tolerance:
        return upper, 0
    if lower_residual > 0.0 or upper_residual < 0.0:
        raise _OrderedSweepUnavailable("cell balance is not bracketed")

    left = lower
    right = upper
    value = min(right, max(left, old_saturation))
    for iterations in range(1, 25):
        value_residual = residual(value)
        if abs(value_residual) <= tolerance:
            return value, iterations
        if value_residual < 0.0:
            left = value
        else:
            right = value
        if right - left <= 2.0e-13 * relperm.mobile_interval:
            return 0.5 * (left + right), iterations

        derivative = storage + outgoing_rate * _fractional_flow_derivative_scalar(
            value, relperm, water, oil
        )
        candidate = value - value_residual / derivative if derivative > 0.0 else np.nan
        if not np.isfinite(candidate) or candidate <= left or candidate >= right:
            candidate = 0.5 * (left + right)
        value = candidate

    for bisection_iteration in range(1, 49):
        value = 0.5 * (left + right)
        value_residual = residual(value)
        if abs(value_residual) <= tolerance:
            return value, 24 + bisection_iteration
        if value_residual < 0.0:
            left = value
        else:
            right = value
    return 0.5 * (left + right), 72


def _advance_saturation_implicit_ordered(
    mesh: StructuredMesh,
    material: MaterialProperties,
    saturation: np.ndarray,
    flow: FlowResult,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
    boundary: BoundaryConfig,
    numerics: NumericsConfig,
    dt: float,
    order_cache: TransportOrderCache | None = None,
) -> TransportStep:
    """Backward Euler solved by one downstream sweep of the acyclic flow graph.

    Without gravity or capillary pressure every nonzero face flux goes from a
    larger pressure to a smaller pressure.  First-order upwinding therefore
    makes the implicit transport Jacobian triangular after cells are ordered by
    decreasing pressure.  Each row reduces to one monotone scalar equation.
    """
    old = np.asarray(saturation, dtype=float)
    pore_volume = material.porosity * mesh.cell_volumes
    qxi = flow.xi_flux
    qeta = flow.eta_flux
    source = flow.well_source
    nx, ny = mesh.shape
    order = (
        order_cache.get(flow)
        if order_cache is not None
        else np.argsort(-flow.pressure.ravel(), kind="stable")
    )
    use_numba = numerics.transport_acceleration in {"auto", "numba"} and (
        _numba_ordered_sweep is not None
    )
    if numerics.transport_acceleration == "numba" and _numba_ordered_sweep is None:
        raise SaturationError(
            "Выбран режим Numba, но пакет numba не установлен. "
            "Установите porous-fvm[jit] или выберите «Без Numba»."
        )

    if use_numba:
        updated, maximum_scalar_iterations, status = _numba_ordered_sweep(
            old,
            pore_volume,
            qxi,
            qeta,
            source,
            order,
            boundary.injected_water_fraction,
            dt,
            relperm.swc,
            relperm.mobile_interval,
            relperm.krw_end,
            relperm.kro_end,
            relperm.nw,
            relperm.no,
            water.viscosity,
            oil.viscosity,
            numerics.transport_rtol,
        )
        if status != 0:
            raise _OrderedSweepUnavailable(
                "Numba ordered transport graph or cell balance is invalid"
            )
    else:
        updated = old.copy()
        solved = np.zeros(mesh.shape, dtype=bool)
        storage = pore_volume / dt
        outgoing = _outward_total_rate(flow)
        maximum_flux = max(
            float(np.max(np.abs(qxi))),
            float(np.max(np.abs(qeta))),
            float(np.max(np.abs(source))),
            1.0e-300,
        )
        dependency_tolerance = 1.0e-13 * maximum_flux
        maximum_scalar_iterations = 0

        def upstream_fraction(i: int, j: int, rate: float) -> float:
            if not solved[i, j] and rate > dependency_tolerance:
                raise _OrderedSweepUnavailable("flow graph is not ordered by pressure")
            return _fractional_flow_scalar(float(updated[i, j]), relperm, water, oil)

        for flat_index in order:
            i = int(flat_index) // ny
            j = int(flat_index) % ny
            incoming_water = 0.0

            left_flux = float(qxi[i, j])
            if left_flux > 0.0:
                if i == 0:
                    incoming_water += left_flux * boundary.injected_water_fraction
                else:
                    incoming_water += left_flux * upstream_fraction(i - 1, j, left_flux)

            right_flux = float(qxi[i + 1, j])
            if right_flux < 0.0:
                if i == nx - 1:
                    raise _OrderedSweepUnavailable(
                        "unsupported inflow through producer boundary"
                    )
                rate = -right_flux
                incoming_water += rate * upstream_fraction(i + 1, j, rate)

            bottom_flux = float(qeta[i, j])
            if bottom_flux > 0.0:
                if j == 0:
                    raise _OrderedSweepUnavailable(
                        "unsupported inflow through lower boundary"
                    )
                incoming_water += bottom_flux * upstream_fraction(i, j - 1, bottom_flux)

            top_flux = float(qeta[i, j + 1])
            if top_flux < 0.0:
                if j == ny - 1:
                    raise _OrderedSweepUnavailable(
                        "unsupported inflow through upper boundary"
                    )
                rate = -top_flux
                incoming_water += rate * upstream_fraction(i, j + 1, rate)

            cell_source = float(source[i, j])
            if cell_source > 0.0:
                incoming_water += cell_source * boundary.injected_water_fraction

            updated[i, j], scalar_iterations = _solve_ordered_cell(
                float(old[i, j]),
                float(storage[i, j]),
                float(outgoing[i, j]),
                incoming_water,
                relperm,
                water,
                oil,
                numerics.transport_rtol,
            )
            maximum_scalar_iterations = max(
                maximum_scalar_iterations, scalar_iterations
            )
            solved[i, j] = True

    qw_in, qw_out, qw_xi, qw_eta, qw_source = water_well_rates(
        updated, flow, relperm, water, oil, boundary
    )
    water_before = float(np.sum(pore_volume * old))
    water_after = float(np.sum(pore_volume * updated))
    boundary_rate = qw_in - qw_out
    expected = water_before + dt * boundary_rate
    balance_error = abs(water_after - expected) / max(
        abs(expected), abs(water_before), 1.0e-30
    )
    if balance_error > max(1.0e-10, 10.0 * numerics.transport_rtol):
        raise _OrderedSweepUnavailable(
            f"ordered transport balance error is too large: {balance_error:.3e}"
        )
    return TransportStep(
        saturation=updated,
        xi_water_flux=qw_xi,
        eta_water_flux=qw_eta,
        well_water_source=qw_source,
        injector_water_rate=qw_in,
        producer_water_rate=qw_out,
        water_volume_before=water_before,
        water_volume_after=water_after,
        boundary_water_rate=boundary_rate,
        balance_error=balance_error,
        nonlinear_iterations=maximum_scalar_iterations,
    )


def _advance_saturation_implicit_newton(
    mesh: StructuredMesh,
    material: MaterialProperties,
    saturation: np.ndarray,
    flow: FlowResult,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
    boundary: BoundaryConfig,
    numerics: NumericsConfig,
    dt: float,
) -> TransportStep:
    """Backward-Euler finite-volume transport with nonlinear upwind fluxes."""
    pore_volume = material.porosity * mesh.cell_volumes
    old = np.asarray(saturation, dtype=float)
    lower = relperm.swc
    upper = 1.0 - relperm.sor
    current = old.copy()
    outward = _outward_total_rate(flow)

    def residual(
        value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        qw_xi, qw_eta = water_fluxes(value, flow, relperm, water, oil, boundary)
        qw_source = well_water_sources(value, flow, relperm, water, oil, boundary)
        net_out = (
            qw_xi[1:]
            - qw_xi[:-1]
            + qw_eta[:, 1:]
            - qw_eta[:, :-1]
            - qw_source
        )
        return pore_volume * (value - old) / dt + net_out, qw_xi, qw_eta, qw_source

    nonlinear_iterations = 0
    for nonlinear_iterations in range(1, numerics.transport_max_iterations + 1):
        value_residual, qw_xi, qw_eta, qw_source = residual(current)
        derivative = _fractional_flow_derivative(current, relperm, water, oil)
        scale = pore_volume / dt + outward * np.maximum(derivative, 1.0)
        residual_norm = float(
            np.max(np.abs(value_residual) / np.maximum(scale, 1.0e-300))
        )
        if residual_norm <= numerics.transport_rtol:
            break

        def jacobian_action(direction: np.ndarray) -> np.ndarray:
            action_xi, action_eta = _water_flux_derivative_action(direction, derivative, flow)
            flux_action = (
                action_xi[1:]
                - action_xi[:-1]
                + action_eta[:, 1:]
                - action_eta[:, :-1]
            )
            source_action = np.where(
                flow.well_source < 0.0,
                flow.well_source * derivative * direction,
                0.0,
            )
            return pore_volume * direction / dt + flux_action - source_action

        diagonal = pore_volume / dt + outward * derivative
        correction = _bicgstab(
            jacobian_action,
            -value_residual,
            diagonal,
            relative_tolerance=min(1.0e-6, 0.1 * numerics.transport_rtol / max(residual_norm, 1.0e-30)),
            max_iterations=max(80, 4 * int(np.sqrt(mesh.cell_count))),
        )

        bound_tolerance = 1.0e-12 * max(1.0, relperm.mobile_interval)
        correction[(current <= lower + bound_tolerance) & (correction < 0.0)] = 0.0
        correction[(current >= upper - bound_tolerance) & (correction > 0.0)] = 0.0

        step_length = 1.0
        positive = correction > 0.0
        negative = correction < 0.0
        if np.any(positive):
            step_length = min(
                step_length,
                0.999 * float(np.min((upper - current[positive]) / correction[positive])),
            )
        if np.any(negative):
            step_length = min(
                step_length,
                0.999 * float(np.min((lower - current[negative]) / correction[negative])),
            )
        step_length = max(step_length, 1.0e-8)
        accepted = False
        for _line_search in range(14):
            candidate = np.clip(current + step_length * correction, lower, upper)
            candidate_residual, _, _, _ = residual(candidate)
            candidate_norm = float(
                np.max(np.abs(candidate_residual) / np.maximum(scale, 1.0e-300))
            )
            if candidate_norm < residual_norm:
                current = candidate
                accepted = True
                break
            step_length *= 0.5
        if not accepted:
            raise SaturationError("implicit transport Newton line search failed")
    else:
        raise SaturationError(
            f"implicit transport did not converge in {numerics.transport_max_iterations} iterations "
            f"(scaled residual={residual_norm:.3e})"
        )

    value_residual, qw_xi, qw_eta, qw_source = residual(current)
    final_scale = pore_volume / dt + outward
    if float(np.max(np.abs(value_residual) / np.maximum(final_scale, 1.0e-300))) > 10.0 * numerics.transport_rtol:
        raise SaturationError("implicit transport residual is too large")
    water_before = float(np.sum(pore_volume * old))
    water_after = float(np.sum(pore_volume * current))
    qw_in = float(np.sum(qw_xi[0]) + np.sum(np.maximum(qw_source, 0.0)))
    qw_out = float(np.sum(qw_xi[-1]) + np.sum(np.maximum(-qw_source, 0.0)))
    boundary_rate = qw_in - qw_out
    expected = water_before + dt * boundary_rate
    balance_error = abs(water_after - expected) / max(abs(expected), abs(water_before), 1.0e-30)
    return TransportStep(
        saturation=current,
        xi_water_flux=qw_xi,
        eta_water_flux=qw_eta,
        well_water_source=qw_source,
        injector_water_rate=qw_in,
        producer_water_rate=qw_out,
        water_volume_before=water_before,
        water_volume_after=water_after,
        boundary_water_rate=boundary_rate,
        balance_error=balance_error,
        nonlinear_iterations=nonlinear_iterations,
    )


def advance_saturation_implicit(
    mesh: StructuredMesh,
    material: MaterialProperties,
    saturation: np.ndarray,
    flow: FlowResult,
    relperm: PowerLawRelativePermeability,
    water: PhaseConfig,
    oil: PhaseConfig,
    boundary: BoundaryConfig,
    numerics: NumericsConfig,
    dt: float,
    order_cache: TransportOrderCache | None = None,
) -> TransportStep:
    """Backward-Euler upwind transport with an ordered-sweep fast path."""
    try:
        return _advance_saturation_implicit_ordered(
            mesh,
            material,
            saturation,
            flow,
            relperm,
            water,
            oil,
            boundary,
            numerics,
            dt,
            order_cache,
        )
    except _OrderedSweepUnavailable:
        return _advance_saturation_implicit_newton(
            mesh,
            material,
            saturation,
            flow,
            relperm,
            water,
            oil,
            boundary,
            numerics,
            dt,
        )
