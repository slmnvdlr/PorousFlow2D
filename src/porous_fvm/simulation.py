from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .config import (
    CaseConfig,
    MUSKAT_CORNER_CORRECTION,
    UNIFORM_POINT_WELL_FORMULATION,
)
from .fields import MaterialProperties, material_properties
from .flow import CornerPointWellModel, FlowResult, PressureWorkspace, solve_pressure
from .flow_mesh import build_flow_aligned_mesh
from .geometry import FiveSpotGeometry
from .mesh import MeshQuality, StructuredMesh, rectangular_mesh
from .relperm import PowerLawRelativePermeability
from .transport import (
    SaturationError,
    TransportOrderCache,
    advance_saturation,
    advance_saturation_implicit,
    stable_time_step,
    water_well_rates,
)


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class PreparedCase:
    geometry: FiveSpotGeometry
    pilot_mesh: StructuredMesh
    mesh: StructuredMesh
    material: MaterialProperties
    relperm: PowerLawRelativePermeability
    initial_flow: FlowResult
    corner_wells: CornerPointWellModel | None
    pressure_workspace: PressureWorkspace


@dataclass(frozen=True)
class SimulationResult:
    case_name: str
    formulation: str
    transport_scheme: str
    mesh: StructuredMesh
    material: MaterialProperties
    times: np.ndarray
    saturation: np.ndarray
    pressure: np.ndarray
    xi_total_flux: np.ndarray
    eta_total_flux: np.ndarray
    mass_balance_error: np.ndarray
    pressure_iterations: np.ndarray
    diagnostic_times: np.ndarray
    injector_liquid_rate: np.ndarray
    producer_liquid_rate: np.ndarray
    injector_water_rate: np.ndarray
    producer_water_rate: np.ndarray
    injector_water_fraction: np.ndarray
    producer_water_cut: np.ndarray
    completed: bool

    @property
    def final_saturation(self) -> np.ndarray:
        return self.saturation[-1]


def muskat_quarter_five_spot_rate(
    side_length: float,
    thickness: float,
    permeability: float,
    total_mobility: float,
    well_radius: float,
    pressure_drop: float,
) -> float:
    """Return Muskat's rate for one quarter five-spot periodic element.

    Muskat's full-well expression is divided by four because the rectangular
    computational element contains one quarter of the injector and one quarter
    of the producer.
    """
    values = (
        side_length,
        thickness,
        permeability,
        total_mobility,
        well_radius,
        pressure_drop,
    )
    if min(values) <= 0.0:
        raise ValueError("Muskat rate parameters must be positive")
    spacing = np.sqrt(2.0) * side_length
    denominator = np.log(spacing / well_radius) - 0.6188
    if denominator <= 0.0:
        raise ValueError(
            "Формула Маскета требует ln(sqrt(2)*a/r_w) - 0.6188 > 0"
        )
    return float(
        0.25
        * np.pi
        * permeability
        * thickness
        * total_mobility
        * pressure_drop
        / denominator
    )


def _constant_value(values: np.ndarray, name: str) -> float:
    array = np.asarray(values, dtype=float)
    reference = float(array.flat[0])
    scale = max(abs(reference), 1.0e-300)
    if float(np.max(np.abs(array - reference))) > 1.0e-10 * scale:
        raise ValueError(
            f"Поправка Маскета применима только к однородному полю: {name} неоднородно"
        )
    return reference


def _calibrate_muskat_corner_wells(
    case: CaseConfig,
    material: MaterialProperties,
    total_mobility: np.ndarray,
    base_model: CornerPointWellModel,
    uncorrected_flow: FlowResult,
) -> CornerPointWellModel:
    """Convert the missing five-spot resistance into two corner skin terms."""
    geometry = case.geometry
    length_scale = max(geometry.lx, geometry.ly, 1.0)
    if abs(geometry.lx - geometry.ly) > 1.0e-10 * length_scale:
        raise ValueError(
            "Поправка Маскета реализована для квадратного элемента пятиточки: Lx должно равняться Ly"
        )
    radius_scale = max(
        geometry.injector_radius,
        geometry.producer_radius,
        1.0e-300,
    )
    if (
        abs(geometry.injector_radius - geometry.producer_radius)
        > 1.0e-10 * radius_scale
    ):
        raise ValueError(
            "Поправка Маскета требует одинаковых радиусов нагнетательной и добывающей скважин"
        )

    permeability = np.asarray(material.permeability, dtype=float)
    kx = _constant_value(permeability[..., 0, 0], "kx")
    ky = _constant_value(permeability[..., 1, 1], "ky")
    off_diagonal = max(
        float(np.max(np.abs(permeability[..., 0, 1]))),
        float(np.max(np.abs(permeability[..., 1, 0]))),
    )
    permeability_scale = max(abs(kx), abs(ky), 1.0e-300)
    if abs(kx - ky) > 1.0e-10 * permeability_scale or off_diagonal > 1.0e-10 * permeability_scale:
        raise ValueError(
            "Поправка Маскета реализована для однородной изотропной проницаемости"
        )
    mobility = _constant_value(total_mobility, "начальная полная подвижность")
    pressure_drop = (
        case.boundary.injector_pressure - case.boundary.producer_pressure
    )
    target_rate = muskat_quarter_five_spot_rate(
        geometry.lx,
        geometry.thickness,
        kx,
        mobility,
        geometry.injector_radius,
        pressure_drop,
    )
    uncorrected_rate = 0.5 * (
        uncorrected_flow.injector_rate + uncorrected_flow.producer_rate
    )
    if uncorrected_rate <= 0.0:
        raise ValueError("Исходный дебит угловых скважин должен быть положительным")

    added_total_resistance = pressure_drop * (
        1.0 / target_rate - 1.0 / uncorrected_rate
    )
    resistance_scale = pressure_drop / target_rate
    if added_total_resistance < -1.0e-10 * resistance_scale:
        raise ValueError(
            "Исходная сетка уже имеет большее сопротивление, чем формула Маскета; "
            "положительную угловую поправку добавить нельзя"
        )
    added_total_resistance = max(0.0, added_total_resistance)

    # For a quarter well WI = A*lambda/(D+s).  Half of the missing total
    # resistance is assigned to each symmetric well and converted to the
    # dimensionless denominator increment (corner skin).
    geometric_factor = 0.5 * np.pi * geometry.thickness * kx
    corner_skin = (
        geometric_factor * mobility * 0.5 * added_total_resistance
    )
    return replace(
        base_model,
        injector_corner_skin=float(corner_skin),
        producer_corner_skin=float(corner_skin),
        correction_mode=MUSKAT_CORNER_CORRECTION,
        uncorrected_rate=float(uncorrected_rate),
        target_rate=float(target_rate),
        added_total_resistance=float(added_total_resistance),
    )


def prepare_case(case: CaseConfig, progress: ProgressCallback | None = None) -> PreparedCase:
    notify = progress or (lambda _fraction, _message: None)
    geometry = FiveSpotGeometry.from_config(case.geometry)
    relperm = PowerLawRelativePermeability.from_config(case.relperm)
    if case.formulation == UNIFORM_POINT_WELL_FORMULATION:
        notify(0.05, "building uniform Cartesian mesh")
        mesh = rectangular_mesh(
            geometry.lx,
            geometry.ly,
            case.mesh.nx,
            case.mesh.ny,
            geometry.thickness,
        )
        corner_wells = CornerPointWellModel(
            geometry.injector_radius,
            geometry.producer_radius,
        )
        mesh.coordinate_mode = "uniform Cartesian with corner quarter-wells"
    else:
        notify(0.05, "building boundary-fitted structured mesh")
        mesh = build_flow_aligned_mesh(
            geometry,
            None,  # legacy arguments retained by the mesh-dispatch interface
            None,
            case.mesh,
            case.boundary.injector_pressure,
            case.boundary.producer_pressure,
        )
        corner_wells = None
    material = material_properties(case.medium, mesh, case.source_path.parent)
    sw0 = np.full(mesh.shape, case.initial_water_saturation)
    _, _, mobility = relperm.mobilities(sw0, case.water, case.oil)
    pressure_workspace = PressureWorkspace(
        mesh,
        material,
        case.boundary,
        case.numerics,
        corner_wells,
    )
    notify(0.15, "solving pressure on the final mesh")
    initial_flow = solve_pressure(
        mesh,
        material,
        mobility,
        case.boundary,
        case.numerics,
        corner_wells=corner_wells,
        workspace=pressure_workspace,
    )
    if (
        corner_wells is not None
        and case.well_model.corner_correction == MUSKAT_CORNER_CORRECTION
    ):
        corner_wells = _calibrate_muskat_corner_wells(
            case,
            material,
            mobility,
            corner_wells,
            initial_flow,
        )
        notify(0.17, "applying Muskat corner-well resistance correction")
        pressure_workspace = PressureWorkspace(
            mesh,
            material,
            case.boundary,
            case.numerics,
            corner_wells,
        )
        initial_flow = solve_pressure(
            mesh,
            material,
            mobility,
            case.boundary,
            case.numerics,
            initial_pressure=initial_flow.pressure,
            corner_wells=corner_wells,
            workspace=pressure_workspace,
        )
    return PreparedCase(
        geometry=geometry,
        pilot_mesh=mesh,
        mesh=mesh,
        material=material,
        relperm=relperm,
        initial_flow=initial_flow,
        corner_wells=corner_wells,
        pressure_workspace=pressure_workspace,
    )


def run_case(
    case: CaseConfig,
    progress: ProgressCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> SimulationResult:
    notify = progress or (lambda _fraction, _message: None)
    stop_requested = should_stop or (lambda: False)
    prepared = prepare_case(case, notify)
    mesh = prepared.mesh
    saturation = np.full(mesh.shape, case.initial_water_saturation)
    derivative_bound = prepared.relperm.max_fractional_flow_derivative(case.water, case.oil)
    t = case.time.start
    store_field_frames = case.output.store_field_frames
    next_save = t if store_field_frames else np.inf
    history_every = case.output.well_history_every
    next_history = t if history_every is not None else np.inf
    next_dt = case.time.initial_step
    minimum_dt = max(1.0e-12 * (case.time.end - case.time.start), 1.0e-9)
    pressure_guess = prepared.initial_flow.pressure
    times: list[float] = []
    saturation_frames: list[np.ndarray] = []
    pressure_frames: list[np.ndarray] = []
    xi_flux_frames: list[np.ndarray] = []
    eta_flux_frames: list[np.ndarray] = []
    balance_frames: list[float] = []
    iteration_frames: list[int] = []
    diagnostic_times: list[float] = []
    injector_liquid_rate: list[float] = []
    producer_liquid_rate: list[float] = []
    injector_water_rate: list[float] = []
    producer_water_rate: list[float] = []
    injector_water_fraction: list[float] = []
    producer_water_cut: list[float] = []
    last_balance = 0.0
    stopped = False
    transport_order_cache = TransportOrderCache()

    while t < case.time.end - 1.0e-12:
        if stop_requested():
            stopped = True
            break
        _, _, mobility = prepared.relperm.mobilities(saturation, case.water, case.oil)
        flow = solve_pressure(
            mesh,
            prepared.material,
            mobility,
            case.boundary,
            case.numerics,
            initial_pressure=pressure_guess,
            reconstruct_velocity=False,
            corner_wells=prepared.corner_wells,
            workspace=prepared.pressure_workspace,
        )
        pressure_guess = flow.pressure
        if store_field_frames and (t >= next_save - 1.0e-10 or not times):
            times.append(t)
            saturation_frames.append(saturation.copy())
            pressure_frames.append(flow.pressure.copy())
            xi_flux_frames.append(flow.xi_flux.copy())
            eta_flux_frames.append(flow.eta_flux.copy())
            balance_frames.append(last_balance)
            iteration_frames.append(flow.iterations)
            next_save += case.time.save_every

        if history_every is not None and (
            t >= next_history - 1.0e-10 or not diagnostic_times
        ):
            qw_in, qw_out, _qw_xi, _qw_eta, _qw_source = water_well_rates(
                saturation,
                flow,
                prepared.relperm,
                case.water,
                case.oil,
                case.boundary,
            )
            q_in = flow.injector_rate
            q_out = flow.producer_rate
            diagnostic_times.append(t)
            injector_liquid_rate.append(q_in)
            producer_liquid_rate.append(q_out)
            injector_water_rate.append(qw_in)
            producer_water_rate.append(qw_out)
            injector_water_fraction.append(
                qw_in / q_in if abs(q_in) > 1.0e-300 else np.nan
            )
            producer_water_cut.append(
                qw_out / q_out if abs(q_out) > 1.0e-300 else np.nan
            )
            next_history += history_every

        if case.numerics.transport_scheme == "explicit":
            dt = min(
                case.time.initial_step,
                stable_time_step(
                    mesh,
                    prepared.material,
                    flow,
                    derivative_bound,
                    case.time.target_cfl,
                ),
                case.time.end - t,
            )
        else:
            dt = min(next_dt, case.time.initial_step, case.time.end - t)
        if store_field_frames and next_save > t + 1.0e-12:
            dt = min(dt, next_save - t)
        if history_every is not None and next_history > t + 1.0e-12:
            dt = min(dt, next_history - t)
        if not np.isfinite(dt) or dt <= 0.0:
            raise RuntimeError("failed to select a positive time step")
        if case.numerics.transport_scheme == "explicit":
            for retry in range(13):
                try:
                    step = advance_saturation(
                        mesh,
                        prepared.material,
                        saturation,
                        flow,
                        prepared.relperm,
                        case.water,
                        case.oil,
                        case.boundary,
                        case.numerics,
                        dt,
                    )
                    break
                except SaturationError:
                    if retry == 12:
                        raise
                    dt *= 0.5
                    notify(
                        0.20 + 0.78 * (t - case.time.start) / (case.time.end - case.time.start),
                        f"retrying explicit transport at time={t:.6g} s with dt={dt:.3g} s",
                    )
            saturation_change = float(np.max(np.abs(step.saturation - saturation)))
        else:
            for retry in range(30):
                try:
                    step = advance_saturation_implicit(
                        mesh,
                        prepared.material,
                        saturation,
                        flow,
                        prepared.relperm,
                        case.water,
                        case.oil,
                        case.boundary,
                        case.numerics,
                        dt,
                        transport_order_cache,
                    )
                    saturation_change = float(np.max(np.abs(step.saturation - saturation)))
                    if (
                        saturation_change
                        > case.numerics.max_saturation_change * (1.0 + 1.0e-8)
                        and dt > minimum_dt
                    ):
                        reduction = max(
                            0.1,
                            0.8 * case.numerics.max_saturation_change / saturation_change,
                        )
                        dt = max(minimum_dt, dt * reduction)
                        notify(
                            0.20
                            + 0.78
                            * (t - case.time.start)
                            / (case.time.end - case.time.start),
                            f"refining implicit step at time={t:.6g} s: dt={dt:.3g} s",
                        )
                        continue
                    break
                except SaturationError as exc:
                    if retry == 29 or dt <= minimum_dt:
                        raise
                    dt = max(minimum_dt, 0.5 * dt)
                    notify(
                        0.20 + 0.78 * (t - case.time.start) / (case.time.end - case.time.start),
                        f"retrying implicit transport at time={t:.6g} s with dt={dt:.3g} s: {exc}",
                    )
        if history_every is None:
            q_in = flow.injector_rate
            q_out = flow.producer_rate
            qw_in = step.injector_water_rate
            qw_out = step.producer_water_rate
            diagnostic_times.append(t)
            injector_liquid_rate.append(q_in)
            producer_liquid_rate.append(q_out)
            injector_water_rate.append(qw_in)
            producer_water_rate.append(qw_out)
            injector_water_fraction.append(
                qw_in / q_in if abs(q_in) > 1.0e-300 else np.nan
            )
            producer_water_cut.append(
                qw_out / q_out if abs(q_out) > 1.0e-300 else np.nan
            )
        saturation = step.saturation
        last_balance = step.balance_error
        t += dt
        if case.numerics.transport_scheme == "implicit":
            if saturation_change < 0.35 * case.numerics.max_saturation_change:
                growth = 1.8
            elif saturation_change < 0.70 * case.numerics.max_saturation_change:
                growth = 1.35
            else:
                growth = 1.10
            next_dt = min(case.time.initial_step, dt * growth)
        fraction = 0.20 + 0.78 * (t - case.time.start) / (case.time.end - case.time.start)
        notify(
            min(fraction, 0.98),
            f"time={t:.6g} s, dt={dt:.3g} s, method={case.numerics.transport_scheme}, "
            f"iterations={step.nonlinear_iterations}, "
            f"max_dS={saturation_change:.3g}",
        )

    _, _, mobility = prepared.relperm.mobilities(saturation, case.water, case.oil)
    final_flow = solve_pressure(
        mesh,
        prepared.material,
        mobility,
        case.boundary,
        case.numerics,
        initial_pressure=pressure_guess,
        reconstruct_velocity=False,
        corner_wells=prepared.corner_wells,
        workspace=prepared.pressure_workspace,
    )
    final_qw_in, final_qw_out, _final_qw_xi, _final_qw_eta, _final_qw_source = water_well_rates(
        saturation,
        final_flow,
        prepared.relperm,
        case.water,
        case.oil,
        case.boundary,
    )
    if not diagnostic_times or abs(diagnostic_times[-1] - t) > 1.0e-10:
        q_in = final_flow.injector_rate
        q_out = final_flow.producer_rate
        qw_in = final_qw_in
        qw_out = final_qw_out
        diagnostic_times.append(t)
        injector_liquid_rate.append(q_in)
        producer_liquid_rate.append(q_out)
        injector_water_rate.append(qw_in)
        producer_water_rate.append(qw_out)
        injector_water_fraction.append(qw_in / q_in if abs(q_in) > 1.0e-300 else np.nan)
        producer_water_cut.append(qw_out / q_out if abs(q_out) > 1.0e-300 else np.nan)
    if not times or abs(times[-1] - t) > 1.0e-10:
        times.append(t)
        saturation_frames.append(saturation.copy())
        pressure_frames.append(final_flow.pressure.copy())
        xi_flux_frames.append(final_flow.xi_flux.copy())
        eta_flux_frames.append(final_flow.eta_flux.copy())
        balance_frames.append(last_balance)
        iteration_frames.append(final_flow.iterations)
    completed = not stopped and t >= case.time.end - 1.0e-12
    notify(1.0 if completed else min(0.98, 0.20 + 0.78 * (t - case.time.start) / (case.time.end - case.time.start)), "completed" if completed else "stopped")
    return SimulationResult(
        case_name=case.name,
        formulation=case.formulation,
        transport_scheme=case.numerics.transport_scheme,
        mesh=mesh,
        material=prepared.material,
        times=np.asarray(times),
        saturation=np.asarray(saturation_frames),
        pressure=np.asarray(pressure_frames),
        xi_total_flux=np.asarray(xi_flux_frames),
        eta_total_flux=np.asarray(eta_flux_frames),
        mass_balance_error=np.asarray(balance_frames),
        pressure_iterations=np.asarray(iteration_frames),
        diagnostic_times=np.asarray(diagnostic_times),
        injector_liquid_rate=np.asarray(injector_liquid_rate),
        producer_liquid_rate=np.asarray(producer_liquid_rate),
        injector_water_rate=np.asarray(injector_water_rate),
        producer_water_rate=np.asarray(producer_water_rate),
        injector_water_fraction=np.asarray(injector_water_fraction),
        producer_water_cut=np.asarray(producer_water_cut),
        completed=completed,
    )


def case_summary(case: CaseConfig, prepared: PreparedCase) -> dict[str, float | int | str]:
    quality: MeshQuality = prepared.mesh.quality()
    summary: dict[str, float | int | str] = {
        "case": case.name,
        "formulation": case.formulation,
        "mesh_generator": case.mesh.generator,
        "cells": prepared.mesh.cell_count,
        "minimum_cell_area_m2": quality.minimum_area,
        "minimum_angle_deg": quality.minimum_angle_deg,
        "maximum_aspect_ratio": quality.maximum_aspect_ratio,
        "pressure_mass_balance_error": prepared.initial_flow.mass_balance_error,
        "pressure_iterations": prepared.initial_flow.iterations,
        "pressure_solver": "CG + PyAMG smoothed aggregation",
        "amg_rebuilds": prepared.pressure_workspace.amg_rebuilds,
    }
    if prepared.corner_wells is not None:
        well_model = prepared.corner_wells
        summary.update(
            {
                "well_model": (
                    "Peaceman quarter-well with Muskat resistance correction"
                    if well_model.correction_mode == MUSKAT_CORNER_CORRECTION
                    else "Peaceman quarter-well (uncorrected)"
                ),
                "corner_correction": well_model.correction_mode,
                "injector_equivalent_radius_m": prepared.initial_flow.injector_equivalent_radius,
                "producer_equivalent_radius_m": prepared.initial_flow.producer_equivalent_radius,
                "injector_well_index_m3_pa_s": prepared.initial_flow.injector_well_index,
                "producer_well_index_m3_pa_s": prepared.initial_flow.producer_well_index,
                "initial_injector_rate_m3_s": prepared.initial_flow.injector_rate,
                "initial_producer_rate_m3_s": prepared.initial_flow.producer_rate,
                "injector_corner_skin": well_model.injector_corner_skin,
                "producer_corner_skin": well_model.producer_corner_skin,
                "uncorrected_initial_rate_m3_s": well_model.uncorrected_rate,
                "muskat_target_rate_m3_s": well_model.target_rate,
                "added_total_resistance_pa_s_m3": well_model.added_total_resistance,
            }
        )
    if hasattr(prepared.mesh, "smoothing_iterations"):
        summary["mesh_smoothing_iterations"] = int(
            prepared.mesh.smoothing_iterations
        )
    if hasattr(prepared.mesh, "conformal_weight"):
        summary["mesh_conformal_weight"] = float(prepared.mesh.conformal_weight)
    if hasattr(prepared.mesh, "point_source_diagnostics"):
        diagnostics = prepared.mesh.point_source_diagnostics
        summary["injector_isobar_min_radius_m"] = float(
            diagnostics.injector_min_radius
        )
        summary["injector_isobar_max_radius_m"] = float(
            diagnostics.injector_max_radius
        )
        summary["producer_isobar_min_radius_m"] = float(
            diagnostics.producer_min_radius
        )
        summary["producer_isobar_max_radius_m"] = float(
            diagnostics.producer_max_radius
        )
        summary["wall_normal_velocity_error"] = float(
            diagnostics.wall_normal_velocity_error
        )
    for attribute in (
        "coordinate_mode",
        "internal_orthogonality_max_deg",
        "internal_orthogonality_rms_deg",
        "boundary_incidence_max_deg",
        "boundary_incidence_rms_deg",
    ):
        if hasattr(prepared.mesh, attribute):
            summary[attribute] = getattr(prepared.mesh, attribute)
    return summary
