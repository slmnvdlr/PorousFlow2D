from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .fields import MaterialProperties
from .mesh import StructuredMesh
from .simulation import SimulationResult


def _safe_path_component(value: str) -> str:
    result = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value)).strip(" .")
    return result or "calculation"


def create_run_output_path(
    project_path: str | Path,
    result: SimulationResult,
    timestamp: datetime | None = None,
) -> Path:
    """Reserve a unique directory containing every file from one run."""
    project = Path(project_path).resolve()
    output_root = project.parent / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    formulation = (
        "uniform"
        if result.formulation == "uniform_point_wells"
        else "hydrodynamic"
    )
    status = "completed" if result.completed else "partial"
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base_name = _safe_path_component(
        f"{result.case_name}-{formulation}-{status}-{stamp}"
    )
    for suffix in ("", *(f"-{index:02d}" for index in range(1, 100))):
        directory = output_root / f"{base_name}{suffix}"
        try:
            directory.mkdir(exist_ok=False)
            return directory / "result.npz"
        except FileExistsError:
            continue
    raise RuntimeError("cannot reserve a unique result directory")


def _cumulative_volume(time_s: np.ndarray, rate_m3_s: np.ndarray) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    rate = np.asarray(rate_m3_s, dtype=float)
    result = np.zeros_like(time)
    if len(time) > 1:
        increments = 0.5 * (rate[1:] + rate[:-1]) * np.diff(time)
        result[1:] = np.cumsum(increments)
    return result


def _save_well_history_xlsx(result: SimulationResult, path: Path) -> None:
    try:
        import xlsxwriter  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Excel output requires XlsxWriter; reinstall the project dependencies"
        ) from exc

    time = np.asarray(result.diagnostic_times, dtype=float)
    q_in = np.asarray(result.injector_liquid_rate, dtype=float)
    q_liquid = np.asarray(result.producer_liquid_rate, dtype=float)
    qw_in = np.asarray(result.injector_water_rate, dtype=float)
    qw_out = np.asarray(result.producer_water_rate, dtype=float)
    qo_out = q_liquid - qw_out
    seconds_per_day = 86400.0
    table_data = np.column_stack(
        [
            time,
            time / seconds_per_day,
            q_in * seconds_per_day,
            q_liquid * seconds_per_day,
            qw_in * seconds_per_day,
            qw_out * seconds_per_day,
            qo_out * seconds_per_day,
            result.injector_water_fraction,
            result.producer_water_cut,
            _cumulative_volume(time, q_liquid),
            _cumulative_volume(time, qw_out),
            _cumulative_volume(time, qo_out),
        ]
    )
    headers = (
        "Время, с",
        "Время, сут",
        "Закачка жидкости, м³/сут",
        "Дебит жидкости, м³/сут",
        "Закачка воды, м³/сут",
        "Дебит воды, м³/сут",
        "Дебит нефти, м³/сут",
        "Доля воды в закачке",
        "Обводнённость",
        "Накопленная жидкость, м³",
        "Накопленная вода, м³",
        "Накопленная нефть, м³",
    )

    workbook = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})
    workbook.set_properties(
        {
            "title": f"Скважинные функции: {result.case_name}",
            "subject": "Результаты двухфазной фильтрации",
        }
    )
    title_format = workbook.add_format(
        {"font_name": "Arial", "font_size": 14, "bold": True, "font_color": "#17262B"}
    )
    note_format = workbook.add_format(
        {"font_name": "Arial", "font_size": 10, "italic": True, "font_color": "#52666D"}
    )
    number_format = workbook.add_format(
        {"font_name": "Arial", "font_size": 10, "num_format": "0.000000E+00"}
    )
    percent_format = workbook.add_format(
        {"font_name": "Arial", "font_size": 10, "num_format": "0.00%"}
    )
    header_format = workbook.add_format(
        {
            "font_name": "Arial",
            "font_size": 10,
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#275DAD",
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "border": 0,
        }
    )

    wells = workbook.add_worksheet("Скважины")
    wells.hide_gridlines(2)
    wells.write(0, 0, f"Скважинные функции: {result.case_name}", title_format)
    wells.write(1, 0, "Все дебиты положительны; накопленные объёмы вычислены методом трапеций.", note_format)
    wells.add_table(
        2,
        0,
        2 + len(table_data),
        len(headers) - 1,
        {
            "name": "WellHistory",
            "style": "Table Style Medium 2",
            "columns": [
                {
                    "header": header,
                    "header_format": header_format,
                    "format": percent_format if index in (7, 8) else number_format,
                }
                for index, header in enumerate(headers)
            ],
            "data": table_data.tolist(),
        },
    )
    wells.freeze_panes(3, 2)
    wells.set_column(0, 1, 13)
    wells.set_column(2, 6, 22)
    wells.set_column(7, 8, 19, percent_format)
    wells.set_column(9, 11, 24)

    last_row = 2 + len(table_data)
    rates = workbook.add_chart({"type": "line"})
    for column, name, color in (
        (3, "Жидкость", "#275DAD"),
        (5, "Вода", "#5EB3E3"),
        (6, "Нефть", "#914F32"),
    ):
        rates.add_series(
            {
                "name": name,
                "categories": ["Скважины", 3, 1, last_row, 1],
                "values": ["Скважины", 3, column, last_row, column],
                "line": {"color": color, "width": 2},
            }
        )
    rates.set_title(
        {"name": "Дебиты добывающей скважины", "name_font": {"name": "Arial", "size": 12}}
    )
    rates.set_x_axis(
        {"name": "Время, сут", "name_font": {"name": "Arial"}, "num_font": {"name": "Arial"}}
    )
    rates.set_y_axis(
        {
            "name": "Дебит, м³/сут",
            "name_font": {"name": "Arial"},
            "num_font": {"name": "Arial"},
            "major_gridlines": {"visible": True},
        }
    )
    rates.set_legend({"position": "top", "font": {"name": "Arial"}})
    rates.set_style(10)
    wells.insert_chart("N3", rates, {"x_scale": 1.25, "y_scale": 1.15})

    water_cut = workbook.add_chart({"type": "line"})
    water_cut.add_series(
        {
            "name": "Обводнённость",
            "categories": ["Скважины", 3, 1, last_row, 1],
            "values": ["Скважины", 3, 8, last_row, 8],
            "line": {"color": "#2F68B3", "width": 2},
        }
    )
    water_cut.set_title(
        {"name": "Обводнённость добывающей скважины", "name_font": {"name": "Arial", "size": 12}}
    )
    water_cut.set_x_axis(
        {"name": "Время, сут", "name_font": {"name": "Arial"}, "num_font": {"name": "Arial"}}
    )
    water_cut.set_y_axis(
        {
            "name": "Обводнённость",
            "name_font": {"name": "Arial"},
            "num_font": {"name": "Arial"},
            "min": 0,
            "max": 1,
            "num_format": "0%",
        }
    )
    water_cut.set_legend({"none": True})
    water_cut.set_style(10)
    wells.insert_chart("N21", water_cut, {"x_scale": 1.25, "y_scale": 1.15})

    diagnostics = workbook.add_worksheet("Диагностика")
    diagnostics.hide_gridlines(2)
    diagnostics.write(0, 0, "Диагностика расчёта", title_format)
    diagnostics.write(
        1,
        0,
        f"Сетка {result.mesh.nx}×{result.mesh.ny}; схема переноса: {result.transport_scheme}",
        note_format,
    )
    diagnostic_data = np.column_stack(
        [
            result.times,
            result.times / seconds_per_day,
            result.mass_balance_error,
            result.pressure_iterations,
            np.min(result.saturation, axis=(1, 2)),
            np.max(result.saturation, axis=(1, 2)),
            np.min(result.pressure, axis=(1, 2)),
            np.max(result.pressure, axis=(1, 2)),
        ]
    )
    diagnostic_headers = (
        "Время, с",
        "Время, сут",
        "Невязка баланса",
        "Итерации давления",
        "min Sw",
        "max Sw",
        "min p, Па",
        "max p, Па",
    )
    diagnostics.add_table(
        2,
        0,
        2 + len(diagnostic_data),
        len(diagnostic_headers) - 1,
        {
            "name": "RunDiagnostics",
            "style": "Table Style Medium 2",
            "columns": [
                {"header": header, "header_format": header_format, "format": number_format}
                for header in diagnostic_headers
            ],
            "data": diagnostic_data.tolist(),
        },
    )
    diagnostics.freeze_panes(3, 2)
    diagnostics.set_column(0, 1, 13)
    diagnostics.set_column(2, 7, 23)
    workbook.close()


def _arrays(result: SimulationResult) -> dict[str, np.ndarray]:
    return {
        "formulation": np.asarray(result.formulation.encode("ascii")),
        "transport_scheme": np.asarray(result.transport_scheme.encode("ascii")),
        "mesh_nodes": result.mesh.nodes,
        "mesh_thickness": np.asarray(result.mesh.thickness),
        "cell_centers": result.mesh.cell_centers,
        "cell_volumes": result.mesh.cell_volumes,
        "permeability": result.material.permeability,
        "porosity": result.material.porosity,
        "time": result.times,
        "water_saturation": result.saturation,
        "pressure": result.pressure,
        "xi_total_flux": result.xi_total_flux,
        "eta_total_flux": result.eta_total_flux,
        "mass_balance_error": result.mass_balance_error,
        "pressure_iterations": result.pressure_iterations,
        "diagnostic_time": result.diagnostic_times,
        "injector_liquid_rate": result.injector_liquid_rate,
        "producer_liquid_rate": result.producer_liquid_rate,
        "injector_water_rate": result.injector_water_rate,
        "producer_water_rate": result.producer_water_rate,
        "injector_water_fraction": result.injector_water_fraction,
        "producer_water_cut": result.producer_water_cut,
        "completed": np.asarray(result.completed),
    }


def save_results(
    result: SimulationResult,
    path: str | Path,
    source_project: str | Path | None = None,
) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = _arrays(result)
    if destination.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError("HDF5 output requires h5py; use .npz or install the hdf5 extra") from exc
        with h5py.File(destination, "w") as file:
            file.attrs["schema_version"] = 2
            file.attrs["case_name"] = result.case_name
            file.attrs["units"] = "SI"
            for name, values in arrays.items():
                options = {"compression": "gzip", "shuffle": True} if values.ndim > 0 else {}
                file.create_dataset(name, data=values, **options)
    else:
        if destination.suffix.lower() != ".npz":
            destination = destination.with_suffix(".npz")
        np.savez_compressed(destination, schema_version=np.array(2), case_name=np.array(result.case_name), **arrays)

    history_path = destination.with_suffix(".wells.xlsx")
    _save_well_history_xlsx(result, history_path)

    if source_project is not None:
        source = Path(source_project).resolve()
        if source.is_file():
            copied_input = destination.parent / f"input{source.suffix.lower()}"
            if source != copied_input.resolve():
                shutil.copy2(source, copied_input)

    summary = {
        "schema_version": 2,
        "case": result.case_name,
        "formulation": result.formulation,
        "transport_scheme": result.transport_scheme,
        "status": "completed" if result.completed else "stopped",
        "frames": int(len(result.times)),
        "final_time_s": float(result.times[-1]),
        "cells": int(result.mesh.cell_count),
        "mesh_nx": int(result.mesh.nx),
        "mesh_ny": int(result.mesh.ny),
        "saved_field_frames": int(len(result.times)),
        "well_history_points": int(len(result.diagnostic_times)),
        "water_saturation_min": float(np.min(result.final_saturation)),
        "water_saturation_max": float(np.max(result.final_saturation)),
        "maximum_mass_balance_error": float(np.max(result.mass_balance_error)),
        "final_injector_liquid_rate_m3_s": float(result.injector_liquid_rate[-1]),
        "final_producer_liquid_rate_m3_s": float(result.producer_liquid_rate[-1]),
        "final_injector_water_fraction": float(result.injector_water_fraction[-1]),
        "final_producer_water_cut": float(result.producer_water_cut[-1]),
        "cumulative_produced_liquid_m3": float(
            _cumulative_volume(result.diagnostic_times, result.producer_liquid_rate)[-1]
        ),
        "cumulative_produced_water_m3": float(
            _cumulative_volume(result.diagnostic_times, result.producer_water_rate)[-1]
        ),
        "well_history_file": history_path.name,
    }
    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def load_results(path: str | Path) -> SimulationResult:
    source = Path(path).resolve()
    if source.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError("HDF5 input requires h5py; install the hdf5 extra") from exc
        with h5py.File(source, "r") as file:
            arrays = {name: np.asarray(dataset) for name, dataset in file.items()}
            case_name = str(file.attrs.get("case_name", source.stem))
    else:
        with np.load(source) as file:
            arrays = {name: np.asarray(file[name]) for name in file.files}
        case_name = str(arrays.get("case_name", np.asarray(source.stem)).item())

    thickness = float(np.asarray(arrays.get("mesh_thickness", 1.0)).item())
    mesh = StructuredMesh(arrays["mesh_nodes"], thickness)
    material = MaterialProperties(arrays["permeability"], arrays["porosity"])
    material.validate(mesh.shape)
    raw_scheme = np.asarray(arrays.get("transport_scheme", b"explicit")).item()
    transport_scheme = (
        raw_scheme.decode("ascii") if isinstance(raw_scheme, bytes) else str(raw_scheme)
    )
    raw_formulation = np.asarray(
        arrays.get("formulation", b"hydrodynamic_finite_wells")
    ).item()
    formulation = (
        raw_formulation.decode("ascii")
        if isinstance(raw_formulation, bytes)
        else str(raw_formulation)
    )
    return SimulationResult(
        case_name=case_name,
        formulation=formulation,
        transport_scheme=transport_scheme,
        mesh=mesh,
        material=material,
        times=arrays["time"],
        saturation=arrays["water_saturation"],
        pressure=arrays["pressure"],
        xi_total_flux=arrays["xi_total_flux"],
        eta_total_flux=arrays["eta_total_flux"],
        mass_balance_error=arrays["mass_balance_error"],
        pressure_iterations=arrays["pressure_iterations"],
        diagnostic_times=arrays["diagnostic_time"],
        injector_liquid_rate=arrays["injector_liquid_rate"],
        producer_liquid_rate=arrays["producer_liquid_rate"],
        injector_water_rate=arrays["injector_water_rate"],
        producer_water_rate=arrays["producer_water_rate"],
        injector_water_fraction=arrays["injector_water_fraction"],
        producer_water_cut=arrays["producer_water_cut"],
        completed=bool(np.asarray(arrays.get("completed", True)).item()),
    )
