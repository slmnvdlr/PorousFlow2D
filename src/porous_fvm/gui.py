from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .config import (
    BoundaryConfig,
    ConfigurationError,
    GeometryConfig,
    HYDRODYNAMIC_FORMULATION,
    MUSKAT_CORNER_CORRECTION,
    MeshConfig,
    NO_CORNER_CORRECTION,
    NumericsConfig,
    PhaseConfig,
    RelPermConfig,
    TimeConfig,
    UNIFORM_POINT_WELL_FORMULATION,
    WellModelConfig,
    default_case,
    load_case,
)
from .geometry import FiveSpotGeometry
from .io import create_run_output_path, load_results, save_results
from .simulation import case_summary, prepare_case, run_case


BROWN_BLUE_POSITIONS = np.array([0.00, 0.18, 0.34, 0.50, 0.67, 0.83, 1.00])
BROWN_BLUE_RGB = np.array(
    [
        [145, 79, 50],
        [198, 154, 107],
        [249, 220, 195],
        [236, 249, 252],
        [142, 214, 239],
        [94, 179, 227],
        [47, 104, 179],
    ],
    dtype=float,
)

TIME_END_UNITS_SECONDS = {"с": 1.0, "ч": 3600.0, "сут": 86400.0}


def _cell_values_to_nodes(values: np.ndarray) -> np.ndarray:
    """Average cell values at the surrounding structured-grid nodes."""
    cell_values = np.asarray(values, dtype=float)
    if cell_values.ndim != 2:
        raise ValueError("field values must be a two-dimensional cell array")
    nx, ny = cell_values.shape
    node_sum = np.zeros((nx + 1, ny + 1), dtype=float)
    node_count = np.zeros((nx + 1, ny + 1), dtype=float)
    for i_slice, j_slice in (
        (np.s_[:-1], np.s_[:-1]),
        (np.s_[1:], np.s_[:-1]),
        (np.s_[1:], np.s_[1:]),
        (np.s_[:-1], np.s_[1:]),
    ):
        node_sum[i_slice, j_slice] += cell_values
        node_count[i_slice, j_slice] += 1.0
    return node_sum / node_count


def _contour_segments(
    nodes: np.ndarray,
    node_values: np.ndarray,
    level: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return linear contour segments on a curvilinear structured grid."""
    points = np.asarray(nodes, dtype=float)
    values = np.asarray(node_values, dtype=float)
    if points.shape[:2] != values.shape or points.shape[-1] != 2:
        raise ValueError("mesh nodes and nodal field have incompatible shapes")

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    nx, ny = values.shape[0] - 1, values.shape[1] - 1
    for i in range(nx):
        for j in range(ny):
            corners = np.array(
                [
                    points[i, j],
                    points[i + 1, j],
                    points[i + 1, j + 1],
                    points[i, j + 1],
                ]
            )
            corner_values = np.array(
                [
                    values[i, j],
                    values[i + 1, j],
                    values[i + 1, j + 1],
                    values[i, j + 1],
                ]
            )
            center = np.mean(corners, axis=0)
            center_value = float(np.mean(corner_values))
            for k in range(4):
                triangle_points = (center, corners[k], corners[(k + 1) % 4])
                triangle_values = (
                    center_value,
                    float(corner_values[k]),
                    float(corner_values[(k + 1) % 4]),
                )
                crossings: list[np.ndarray] = []
                for edge in ((0, 1), (1, 2), (2, 0)):
                    first, second = edge
                    first_above = triangle_values[first] >= level
                    second_above = triangle_values[second] >= level
                    if first_above == second_above:
                        continue
                    denominator = triangle_values[second] - triangle_values[first]
                    fraction = (level - triangle_values[first]) / denominator
                    crossings.append(
                        triangle_points[first]
                        + fraction * (triangle_points[second] - triangle_points[first])
                    )
                if len(crossings) == 2:
                    segments.append((crossings[0], crossings[1]))
    return segments


class Application(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PorousFlow 2D")
        self.geometry("1240x760")
        self.minsize(920, 620)
        self.events: queue.Queue = queue.Queue()
        default_path = Path(__file__).resolve().parents[2] / "examples" / "five_spot.json"
        self.case_path = tk.StringVar(value=str(default_path) if default_path.exists() else "")
        self._last_project_path = default_path if default_path.exists() else Path.cwd()
        self.status = tk.StringVar(value="Загрузите проект или измените параметры области")
        self.inputs: dict[str, tk.StringVar] = {}
        self.input_widgets: dict[str, ttk.Entry] = {}
        self.input_labels: dict[str, ttk.Label] = {}
        self._busy = False
        self._active_target = ""
        self._stop_event = threading.Event()
        self._applied_case = None
        self._parameters_dirty = False
        self._loading_form = False
        self._result = None
        self._animation_job = None
        self._new_project_unsaved = False
        self._time_end_unit_previous = "с"
        self._hydrodynamic_generator_choice = "ТФКП: точечный источник–сток"
        self._build()
        self.after(50, self._load_project_silent)
        self.after(100, self._poll)

    def _build(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Проект:").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.case_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(toolbar, text="Новый…", command=self._new_project).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Выбрать проект…", command=self._browse).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Загрузить проект", command=self._load_project).pack(side=tk.LEFT, padx=(6, 0))
        self.apply_button = ttk.Button(
            toolbar,
            text="Применить параметры",
            command=self._apply_parameters,
            state=tk.DISABLED,
        )
        self.apply_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Сохранить как…", command=self._save_project).pack(side=tk.LEFT, padx=(6, 0))

        parameter_bar = ttk.Frame(self, padding=(10, 0, 10, 5))
        parameter_bar.pack(fill=tk.X)
        self.parameter_state = tk.StringVar(value="Параметры ещё не загружены")
        self.parameter_state_label = tk.Label(parameter_bar, textvariable=self.parameter_state, anchor="w")
        self.parameter_state_label.pack(fill=tk.X)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        center = ttk.Frame(body, padding=8)
        actions = ttk.Frame(center)
        actions.pack(fill=tk.X, pady=(0, 8))
        self.preview_button = ttk.Button(actions, text="Показать область", command=self._preview)
        self.preview_button.pack(side=tk.LEFT)
        self.mesh_button = ttk.Button(actions, text="Построить сетку", command=self._validate)
        self.mesh_button.pack(side=tk.LEFT, padx=(8, 0))
        self.run_button = ttk.Button(actions, text="6  Запустить расчёт", command=self._run)
        self.run_button.pack(side=tk.LEFT, padx=(8, 0))
        self.stop_button = ttk.Button(actions, text="Остановить", command=self._stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        viewer = ttk.Frame(center)
        viewer.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(viewer, text="Открыть результаты…", command=self._open_results).pack(
            side=tk.LEFT, padx=(0, 18)
        )
        ttk.Label(viewer, text="Поле:").pack(side=tk.LEFT)
        self.field_choice = tk.StringVar(value="Водонасыщенность")
        self.field_select = ttk.Combobox(
            viewer,
            textvariable=self.field_choice,
            state="readonly",
            width=21,
            values=("Водонасыщенность", "Давление", "Скважинные функции"),
        )
        self.field_select.pack(side=tk.LEFT)
        self.field_select.bind("<<ComboboxSelected>>", lambda _event: self._render_result())
        ttk.Label(viewer, text="Вид:").pack(side=tk.LEFT, padx=(18, 5))
        self.view_choice = tk.StringVar(value="Заливка")
        self.view_select = ttk.Combobox(
            viewer,
            textvariable=self.view_choice,
            state="readonly",
            width=18,
            values=("Заливка", "Заливка + сетка", "Изолинии"),
        )
        self.view_select.pack(side=tk.LEFT)
        self.view_select.bind("<<ComboboxSelected>>", lambda _event: self._render_result())
        self.play_button = ttk.Button(viewer, text="▶", width=3, command=self._toggle_animation)
        self.play_button.pack(side=tk.LEFT, padx=(6, 0))

        run_state = ttk.Frame(center)
        run_state.pack(fill=tk.X, pady=(0, 8))
        self.run_progress = ttk.Progressbar(run_state, maximum=100)
        self.run_progress.pack(fill=tk.X)
        self.run_info = tk.StringVar(value="Расчёт не запущен")
        ttk.Label(run_state, textvariable=self.run_info, anchor=tk.W).pack(fill=tk.X, pady=(3, 0))

        timeline = ttk.Frame(center)
        timeline.pack(fill=tk.X, pady=(0, 8))
        self.time_scale = ttk.Scale(timeline, from_=0, to=1, command=self._on_time_change)
        self.time_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.time_label = ttk.Label(timeline, text="t = —", width=20, anchor=tk.E)
        self.time_label.pack(side=tk.LEFT, padx=(8, 0))
        self.canvas = tk.Canvas(
            center,
            background="#f6f8f8",
            highlightthickness=1,
            highlightbackground="#c8d1d3",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        body.add(center, weight=5)

        inspector = ttk.Frame(body, padding=10)
        formulation_group = ttk.LabelFrame(
            inspector,
            text="Постановка задачи",
            padding=8,
        )
        formulation_group.pack(fill=tk.X, pady=(0, 8))
        self._formulation_input(formulation_group)
        self.formulation_help = tk.StringVar()
        ttk.Label(
            formulation_group,
            textvariable=self.formulation_help,
            wraplength=270,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(7, 0))

        notebook = ttk.Notebook(inspector)
        notebook.pack(fill=tk.X)
        geometry_tab = ttk.Frame(notebook, padding=10)
        mesh_tab = ttk.Frame(notebook, padding=10)
        model_tab = ttk.Frame(notebook, padding=10)
        wells_tab = ttk.Frame(notebook, padding=10)
        solver_tab = ttk.Frame(notebook, padding=10)
        notebook.add(geometry_tab, text="1  Область")
        notebook.add(mesh_tab, text="2  Сетка")
        notebook.add(model_tab, text="3  Модель")
        notebook.add(wells_tab, text="4  ОФП и скважины")
        notebook.add(solver_tab, text="5  Расчёт")
        self._input(geometry_tab, 0, "lx", "Размер X", "м")
        self._input(geometry_tab, 1, "ly", "Размер Y", "м")
        self._input(geometry_tab, 2, "thickness", "Толщина", "м")
        self._input(geometry_tab, 3, "injector_radius", "Радиус нагнетателя", "м")
        self._input(geometry_tab, 4, "producer_radius", "Радиус добывающей", "м")
        ttk.Label(
            geometry_tab,
            text="Скважины находятся точно в нижней левой и верхней правой вершинах.",
            wraplength=245,
        ).grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        self._mesh_generator_input(mesh_tab, 0)
        self._input(mesh_tab, 1, "nx", "Продольных", "ячеек")
        self._input(mesh_tab, 2, "ny", "Поперечных", "ячеек")
        self._input(mesh_tab, 3, "pressure_bias", "Сгущение (гладкая)", "коэф.")
        self._input(mesh_tab, 4, "streamline_edge_fraction", "У краёв (гладкая)", "0…1")
        self._input(mesh_tab, 5, "minimum_cell_angle_deg", "Мин. угол ячейки", "°")
        self.mesh_help_text = tk.StringVar()
        self.mesh_help_label = ttk.Label(
            mesh_tab,
            textvariable=self.mesh_help_text,
            wraplength=245,
            justify=tk.LEFT,
        )
        self.mesh_help_label.grid(
            row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0)
        )
        self._update_mesh_help()

        self._input(model_tab, 0, "project_name", "Название проекта", "—")
        self._material_mode_input(model_tab, 1)
        self._input(model_tab, 2, "permeability", "Проницаемость", "м²")
        self._input(model_tab, 3, "porosity", "Пористость", "0…1")
        ttk.Separator(model_tab).grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        self._input(model_tab, 5, "water_density", "Плотность воды", "кг/м³")
        self._input(model_tab, 6, "water_viscosity", "Вязкость воды", "Па·с")
        self._input(model_tab, 7, "oil_density", "Плотность нефти", "кг/м³")
        self._input(model_tab, 8, "oil_viscosity", "Вязкость нефти", "Па·с")
        self._input(model_tab, 9, "initial_water_saturation", "Начальная Sw", "0…1")
        self.material_help = tk.StringVar()
        ttk.Label(
            model_tab,
            textvariable=self.material_help,
            wraplength=245,
            justify=tk.LEFT,
        ).grid(row=10, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        self._input(wells_tab, 0, "swc", "Связанная вода Swc", "0…1")
        self._input(wells_tab, 1, "sor", "Остаточная нефть Sor", "0…1")
        self._input(wells_tab, 2, "krw_end", "Конечная krw", "0…1")
        self._input(wells_tab, 3, "kro_end", "Конечная kro", "0…1")
        self._input(wells_tab, 4, "nw", "Степень воды nw", "—")
        self._input(wells_tab, 5, "no", "Степень нефти no", "—")
        ttk.Separator(wells_tab).grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        self._input(wells_tab, 7, "injector_pressure", "Давление нагнетателя", "Па")
        self._input(wells_tab, 8, "producer_pressure", "Давление добывающей", "Па")
        self._input(wells_tab, 9, "injected_water_fraction", "Доля воды в закачке", "0…1")
        self._corner_correction_input(wells_tab, 10)
        ttk.Label(
            wells_tab,
            text=(
                "Se=(Sw−Swc)/(1−Swc−Sor)\n"
                "krw=krw_end·Se^nw,  kro=kro_end·(1−Se)^no\n"
                "Поправка Маскета добавляет недостающее сопротивление "
                "угловых скважин; для гидродинамической сетки не применяется."
            ),
            wraplength=245,
            justify=tk.LEFT,
        ).grid(row=11, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        self._input(solver_tab, 0, "time_start", "Начальное время", "с")
        self._time_end_input(solver_tab, 1)
        self._input(solver_tab, 2, "initial_step", "Макс. шаг", "с")
        self._input(solver_tab, 3, "save_every", "Интервал кадров", "с")
        self._scheme_input(solver_tab, 4)
        self._acceleration_input(solver_tab, 5)
        self._input(solver_tab, 6, "target_cfl", "Целевой CFL (явный)", "—")
        self._input(solver_tab, 7, "max_saturation_change", "Ограничение ΔS", "1 = выкл.")
        self._input(solver_tab, 8, "transport_rtol", "Отн. точность S", "—")
        self._input(solver_tab, 9, "transport_max_iterations", "Макс. итераций S", "—")
        self._input(solver_tab, 10, "pressure_rtol", "Отн. точность p", "—")
        self._input(solver_tab, 11, "pressure_atol", "Абс. точность p", "—")
        self._input(solver_tab, 12, "pressure_max_iterations", "Макс. итераций p", "—")
        self._input(solver_tab, 13, "saturation_tolerance", "Допуск границ S", "—")
        ttk.Label(
            solver_tab,
            text=(
                "Неявный режим принимает сошедшийся шаг; ΔS=1 отключает дополнительное "
                "дробление. Меньшее ΔS можно использовать для исследования временной "
                "сходимости. Явный режим выбирает шаг по CFL.\n"
                "Ускорение: «Автоматически» использует Numba, если он установлен; "
                "«Без Numba» выполняет тот же алгоритм на Python. Распараллеливание "
                "не применяется."
            ),
            wraplength=245,
        ).grid(row=14, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        ttk.Separator(inspector).pack(fill=tk.X, pady=10)
        ttk.Label(inspector, text="Диагностика", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.details = tk.Text(inspector, width=34, height=18, relief=tk.FLAT, wrap=tk.WORD)
        self.details.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        body.add(inspector, weight=2)

        footer = ttk.Frame(self, padding=8)
        footer.pack(fill=tk.X)
        self.progress = ttk.Progressbar(footer, maximum=100)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(footer, textvariable=self.status).pack(side=tk.LEFT, padx=(10, 0))

    def _input(self, parent: ttk.Frame, row: int, key: str, label: str, unit: str) -> None:
        variable = tk.StringVar()
        self.inputs[key] = variable
        variable.trace_add("write", self._form_changed)
        label_widget = ttk.Label(parent, text=label)
        self.input_labels[key] = label_widget
        label_widget.grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=12)
        self.input_widgets[key] = entry
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 5), pady=4)
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w", pady=4)
        parent.columnconfigure(1, weight=1)

    def _time_end_input(self, parent: ttk.Frame, row: int) -> None:
        """Create the final-time field with a user-selectable display unit."""
        value = tk.StringVar()
        unit = tk.StringVar(value="с")
        self.inputs["time_end"] = value
        self.inputs["time_end_unit"] = unit
        value.trace_add("write", self._form_changed)
        unit.trace_add("write", self._form_changed)
        label_widget = ttk.Label(parent, text="Конечное время")
        self.input_labels["time_end"] = label_widget
        label_widget.grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=value, width=12)
        self.input_widgets["time_end"] = entry
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 5), pady=4)
        selector = ttk.Combobox(
            parent,
            textvariable=unit,
            state="readonly",
            values=tuple(TIME_END_UNITS_SECONDS),
            width=5,
        )
        selector.grid(row=row, column=2, sticky="w", pady=4)
        selector.bind("<<ComboboxSelected>>", self._time_end_unit_changed)
        parent.columnconfigure(1, weight=1)

    def _time_end_unit_changed(self, _event=None) -> None:
        """Change only the display unit, preserving the physical final time."""
        old_unit = self._time_end_unit_previous
        new_unit = self.inputs["time_end_unit"].get()
        try:
            old_scale = TIME_END_UNITS_SECONDS[old_unit]
            new_scale = TIME_END_UNITS_SECONDS[new_unit]
            displayed = float(self.inputs["time_end"].get())
        except (KeyError, ValueError):
            self._time_end_unit_previous = new_unit
            return
        self._time_end_unit_previous = new_unit
        self.inputs["time_end"].set(
            self._format_input_value(displayed * old_scale / new_scale)
        )

    def _formulation_input(self, parent: ttk.Frame) -> None:
        variable = tk.StringVar()
        self.inputs["formulation"] = variable
        variable.trace_add("write", self._form_changed)
        variable.trace_add("write", self._formulation_changed)
        self.formulation_select = ttk.Combobox(
            parent,
            textvariable=variable,
            state="readonly",
            values=(
                "Гидродинамическая сетка",
                "Равномерная сетка, точечные скважины",
            ),
            width=37,
        )
        self.formulation_select.pack(fill=tk.X)

    def _material_mode_input(self, parent: ttk.Frame, row: int) -> None:
        variable = tk.StringVar()
        self.inputs["material_mode"] = variable
        variable.trace_add("write", self._form_changed)
        variable.trace_add("write", self._material_mode_changed)
        ttk.Label(parent, text="Поля среды").grid(row=row, column=0, sticky="w", pady=4)
        self.material_mode_select = ttk.Combobox(
            parent,
            textvariable=variable,
            state="readonly",
            values=(
                "Однородная среда",
                "Сохранить поля из файла",
            ),
            width=29,
        )
        self.material_mode_select.grid(
            row=row, column=1, sticky="ew", padx=(8, 5), pady=4
        )
        ttk.Label(parent, text="—").grid(row=row, column=2, sticky="w", pady=4)
        parent.columnconfigure(1, weight=1)

    def _corner_correction_input(self, parent: ttk.Frame, row: int) -> None:
        variable = tk.StringVar()
        self.inputs["corner_correction"] = variable
        variable.trace_add("write", self._form_changed)
        ttk.Label(parent, text="Угловая поправка").grid(
            row=row, column=0, sticky="w", pady=4
        )
        self.corner_correction_select = ttk.Combobox(
            parent,
            textvariable=variable,
            state="readonly",
            values=("Без поправки", "По Маскету"),
            width=19,
        )
        self.corner_correction_select.grid(
            row=row, column=1, sticky="ew", padx=(8, 5), pady=4
        )
        ttk.Label(parent, text="—").grid(row=row, column=2, sticky="w", pady=4)
        parent.columnconfigure(1, weight=1)

    def _material_mode_changed(self, *_args) -> None:
        if "permeability" not in self.input_widgets:
            return
        homogeneous = self.inputs["material_mode"].get() == "Однородная среда"
        state = tk.NORMAL if homogeneous else tk.DISABLED
        self.input_widgets["permeability"].configure(state=state)
        self.input_widgets["porosity"].configure(state=state)
        if hasattr(self, "material_help"):
            if homogeneous:
                text = (
                    "Введите скалярную изотропную проницаемость и постоянную "
                    "пористость. Они будут заданы во всех ячейках."
                )
            else:
                text = (
                    "Неоднородные зоны или внешние поля проницаемости и пористости "
                    "будут скопированы из загруженного проекта без изменений."
                )
            self.material_help.set(text)

    def _scheme_input(self, parent: ttk.Frame, row: int) -> None:
        variable = tk.StringVar()
        self.inputs["transport_scheme"] = variable
        variable.trace_add("write", self._form_changed)
        ttk.Label(parent, text="Метод переноса").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            parent,
            textvariable=variable,
            state="readonly",
            values=("Неявный", "Явный"),
            width=12,
        ).grid(row=row, column=1, sticky="ew", padx=(8, 5), pady=4)
        ttk.Label(parent, text="—").grid(row=row, column=2, sticky="w", pady=4)
        parent.columnconfigure(1, weight=1)

    def _acceleration_input(self, parent: ttk.Frame, row: int) -> None:
        variable = tk.StringVar()
        self.inputs["transport_acceleration"] = variable
        variable.trace_add("write", self._form_changed)
        ttk.Label(parent, text="Ускорение переноса").grid(
            row=row, column=0, sticky="w", pady=4
        )
        ttk.Combobox(
            parent,
            textvariable=variable,
            state="readonly",
            values=("Автоматически", "Numba", "Без Numba"),
            width=15,
        ).grid(row=row, column=1, sticky="ew", padx=(8, 5), pady=4)
        ttk.Label(parent, text="—").grid(row=row, column=2, sticky="w", pady=4)
        parent.columnconfigure(1, weight=1)

    def _mesh_generator_input(self, parent: ttk.Frame, row: int) -> None:
        variable = tk.StringVar()
        self.inputs["mesh_generator"] = variable
        variable.trace_add("write", self._form_changed)
        variable.trace_add("write", self._mesh_generator_changed)
        ttk.Label(parent, text="Построение").grid(row=row, column=0, sticky="w", pady=4)
        self.mesh_generator_select = ttk.Combobox(
            parent,
            textvariable=variable,
            state="readonly",
            values=(
                "Гладкая (не φ–ψ)",
                "ТФКП: точечный источник–сток",
                "Равномерная прямоугольная",
            ),
            width=29,
        )
        self.mesh_generator_select.grid(row=row, column=1, sticky="ew", padx=(8, 5), pady=4)
        ttk.Label(parent, text="—").grid(row=row, column=2, sticky="w", pady=4)
        parent.columnconfigure(1, weight=1)

    def _mesh_generator_changed(self, *_args) -> None:
        if hasattr(self, "input_widgets") and "pressure_bias" in self.input_widgets:
            mode = self.inputs["mesh_generator"].get()
            uniform_mode = mode == "Равномерная прямоугольная"
            point_mode = (
                mode == "ТФКП: точечный источник–сток"
            )
            if not uniform_mode and mode:
                self._hydrodynamic_generator_choice = mode
            optional_state = tk.DISABLED if point_mode or uniform_mode else tk.NORMAL
            self.input_widgets["pressure_bias"].configure(state=optional_state)
            self.input_widgets["streamline_edge_fraction"].configure(
                state=optional_state
            )
            self.input_widgets["minimum_cell_angle_deg"].configure(
                state=tk.DISABLED if uniform_mode else tk.NORMAL
            )
        self._update_mesh_help()

    def _formulation_changed(self, *_args) -> None:
        if not hasattr(self, "mesh_generator_select"):
            return
        uniform = (
            self.inputs["formulation"].get()
            == "Равномерная сетка, точечные скважины"
        )
        if uniform:
            current = self.inputs["mesh_generator"].get()
            if current and current != "Равномерная прямоугольная":
                self._hydrodynamic_generator_choice = current
            self.inputs["mesh_generator"].set("Равномерная прямоугольная")
            self.mesh_generator_select.configure(
                values=("Равномерная прямоугольная",),
                state=tk.DISABLED,
            )
            if "nx" in self.input_labels:
                self.input_labels["nx"].configure(text="Ячеек по X")
                self.input_labels["ny"].configure(text="Ячеек по Y")
            help_text = (
                "Полный прямоугольник с непроницаемыми внешними границами. "
                "В нижней левой и верхней правой угловых ячейках находятся "
                "точечные четверти скважин. Приток связывает давление забоя "
                "с давлением ячейки аналитическим индексом скважины."
            )
            if hasattr(self, "corner_correction_select"):
                self.corner_correction_select.configure(state="readonly")
        else:
            if self.inputs["mesh_generator"].get() == "Равномерная прямоугольная":
                self.inputs["mesh_generator"].set(
                    self._hydrodynamic_generator_choice
                )
            self.mesh_generator_select.configure(
                values=(
                    "Гладкая (не φ–ψ)",
                    "ТФКП: точечный источник–сток",
                ),
                state="readonly",
            )
            if "nx" in self.input_labels:
                self.input_labels["nx"].configure(text="Вдоль потока")
                self.input_labels["ny"].configure(text="Поперёк потока")
            help_text = (
                "Область ограничена конечными изобарами скважин. Сетка следует "
                "линиям тока и изобарам; скважины входят в границу области."
            )
            if hasattr(self, "corner_correction_select"):
                self.corner_correction_select.configure(state=tk.DISABLED)
        if hasattr(self, "formulation_help"):
            self.formulation_help.set(help_text)
        self._mesh_generator_changed()

    def _update_mesh_help(self) -> None:
        if not hasattr(self, "mesh_help_text"):
            return
        selected = self.inputs.get("mesh_generator")
        mode = selected.get() if selected is not None else ""
        if mode == "Равномерная прямоугольная":
            method = (
                "Равномерная прямоугольная сетка полного размера Lx × Ly. "
                "Радиусы скважин не вырезаются из области: они используются "
                "только в логарифме аналитической формулы притока."
            )
            common = (
                "Числа ячеек задаются независимо по X и Y. Коэффициенты "
                "сгущения и минимальный угол здесь не применяются."
            )
        elif mode == "ТФКП: точечный источник–сток":
            method = (
                "Используется периодическое решение с точечным источником и стоком. "
                "Границы скважин — первые изобары, касающиеся окружностей заданного "
                "радиуса. Равный шаг потенциала автоматически даёт логарифмическое "
                "сгущение у скважин; линии тока равномерны по функции тока. "
                "Дополнительное сгущение изобар и линий у краёв отключено."
            )
            common = (
                "Вдоль потока — число ячеек от нагнетательной к добывающей скважине. "
                "Поперёк потока — число каналов между линиями тока.\n"
                "Мин. угол ячейки — только контроль качества."
            )
        else:
            method = (
                "Гладкая: трансфинитная сетка, точно повторяющая контур. Она не обязана "
                "совпадать с линиями тока; оба коэффициента сгущения управляют только "
                "размерами и формой ячеек. Поперечное сгущение применяется мягче, "
                "чтобы не ухудшать подобие ячеек. Сгущение изобар и линии у краёв "
                "действуют только в этом режиме."
            )
            common = (
                "Вдоль потока — число ячеек между скважинами. Поперёк потока — "
                "число каналов. Мин. угол ячейки — только контроль качества."
            )
        self.mesh_help_text.set(f"{method}\n\n{common}")

    def _form_changed(self, *_args) -> None:
        if self._loading_form:
            return
        self._parameters_dirty = True
        self.parameter_state.set("● Есть неприменённые изменения")
        self.parameter_state_label.configure(fg="#b45309")
        if not self._busy:
            self.apply_button.configure(state=tk.NORMAL)

    def _set_parameters_clean(self, message: str = "✓ Параметры применены") -> None:
        self._parameters_dirty = False
        self.parameter_state.set(message)
        self.parameter_state_label.configure(fg="#16784b")
        if not self._busy:
            self.apply_button.configure(state=tk.DISABLED)

    def _apply_parameters(self) -> None:
        if self._busy:
            return
        try:
            self._applied_case = self._case_from_form()
            if self._new_project_unsaved:
                self._parameters_dirty = False
                self.parameter_state.set("✓ Параметры проверены · файл ещё не сохранён")
                self.parameter_state_label.configure(fg="#b45309")
                self.apply_button.configure(state=tk.DISABLED)
                self.status.set(
                    "Параметры корректны; сохраните новый входной файл"
                )
            else:
                self._set_parameters_clean()
                self.status.set(
                    "Параметры проверены и применены к следующему запуску"
                )
        except Exception as exc:
            messagebox.showerror("Некорректные параметры", str(exc))

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл проекта",
            filetypes=[("Проект", "*.json *.yaml *.yml"), ("Все файлы", "*.*")],
        )
        if path:
            if self._is_result_selection(path):
                self._open_results_path(path)
                return
            self.case_path.set(path)
            self._load_project()

    def _new_project(self) -> None:
        if self._busy:
            return
        try:
            case = default_case()
            self.case_path.set("")
            self._applied_case = case
            self._populate_form(case)
            self._new_project_unsaved = True
            self._parameters_dirty = True
            self.parameter_state.set("● Новый проект ещё не сохранён")
            self.parameter_state_label.configure(fg="#b45309")
            self.apply_button.configure(state=tk.NORMAL)
            self.status.set(
                "Новый проект: задайте параметры, примените и сохраните JSON"
            )
            self._preview()
        except Exception as exc:
            messagebox.showerror("Ошибка нового проекта", str(exc))

    def _open_results(self) -> None:
        path = filedialog.askopenfilename(
            title="Откройте результаты расчёта",
            filetypes=[
                ("Результаты расчёта", "*.npz *.h5 *.hdf5 *.summary.json"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        self._open_results_path(path)

    @staticmethod
    def _is_result_selection(path: str | Path) -> bool:
        source = Path(path)
        return source.suffix.lower() in {".npz", ".h5", ".hdf5"} or source.name.lower().endswith(
            ".summary.json"
        )

    @staticmethod
    def _resolve_result_path(path: str | Path) -> Path:
        source = Path(path).resolve()
        if not source.name.lower().endswith(".summary.json"):
            return source
        base_name = source.name[: -len(".summary.json")]
        for suffix in (".npz", ".h5", ".hdf5"):
            candidate = source.with_name(base_name + suffix)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Выбран файл сводки, а соответствующий файл полей NPZ/HDF5 рядом не найден. "
            "Для просмотра нужен файл с расширением .npz, .h5 или .hdf5."
        )

    def _open_results_path(self, path: str | Path) -> bool:
        try:
            source = self._resolve_result_path(path)
            self._set_result(load_results(source))
            self.status.set(f"Открыты результаты: {source.name}")
            self.run_info.set(f"Открыт файл результатов: {source.name}")
            return True
        except Exception as exc:
            messagebox.showerror("Ошибка результатов", str(exc))
            return False

    def _load_project_silent(self) -> None:
        if not self.case_path.get().strip():
            self._new_project()
            return
        try:
            self._load_project(show_error=False)
        except Exception:
            pass

    def _load_project(self, show_error: bool = True) -> None:
        selected_path = Path(self.case_path.get()).expanduser()
        if self._is_result_selection(selected_path):
            opened = self._open_results_path(selected_path)
            self.case_path.set(str(self._last_project_path))
            if opened:
                self.status.set(
                    "Открыт файл результатов; путь проекта оставлен без изменения"
                )
            return
        try:
            case = load_case(selected_path)
            self._applied_case = case
            self._populate_form(case)
            self._last_project_path = selected_path.resolve()
            self._new_project_unsaved = False
            self._set_parameters_clean("✓ Параметры загружены и применены")
            self.status.set("Параметры проекта загружены")
            self._preview()
        except Exception as exc:
            if show_error:
                messagebox.showerror("Ошибка проекта", str(exc))
            else:
                raise

    @staticmethod
    def _material_form_values(case) -> tuple[str, str, str]:
        permeability = case.medium.get("permeability", {})
        porosity = case.medium.get("porosity", {})
        homogeneous = (
            permeability.get("type") == "constant"
            and permeability.get("representation", "isotropic") == "isotropic"
            and "value_m2" in permeability
            and porosity.get("type") == "constant"
            and "value" in porosity
        )
        if homogeneous:
            return (
                "Однородная среда",
                Application._format_input_value(permeability["value_m2"]),
                Application._format_input_value(porosity["value"]),
            )
        return "Сохранить поля из файла", "", ""

    def _populate_form(self, case) -> None:
        material_mode, permeability, porosity = self._material_form_values(case)
        displayed_end, displayed_end_unit = self._display_time_end(case.time.end)
        values = {
            "formulation": (
                "Равномерная сетка, точечные скважины"
                if case.formulation == UNIFORM_POINT_WELL_FORMULATION
                else "Гидродинамическая сетка"
            ),
            "project_name": case.name,
            "lx": case.geometry.lx,
            "ly": case.geometry.ly,
            "thickness": case.geometry.thickness,
            "injector_radius": case.geometry.injector_radius,
            "producer_radius": case.geometry.producer_radius,
            "mesh_generator": (
                "Равномерная прямоугольная"
                if case.mesh.generator == "uniform_cartesian"
                else (
                    "ТФКП: точечный источник–сток"
                    if case.mesh.generator == "point_source_isobars"
                    else "Гладкая (не φ–ψ)"
                )
            ),
            "nx": case.mesh.nx,
            "ny": case.mesh.ny,
            "pressure_bias": case.mesh.pressure_bias,
            "streamline_edge_fraction": case.mesh.streamline_edge_fraction,
            "minimum_cell_angle_deg": case.mesh.minimum_cell_angle_deg,
            "material_mode": material_mode,
            "permeability": permeability,
            "porosity": porosity,
            "water_density": case.water.density,
            "water_viscosity": case.water.viscosity,
            "oil_density": case.oil.density,
            "oil_viscosity": case.oil.viscosity,
            "initial_water_saturation": case.initial_water_saturation,
            "swc": case.relperm.swc,
            "sor": case.relperm.sor,
            "krw_end": case.relperm.krw_end,
            "kro_end": case.relperm.kro_end,
            "nw": case.relperm.nw,
            "no": case.relperm.no,
            "injector_pressure": case.boundary.injector_pressure,
            "producer_pressure": case.boundary.producer_pressure,
            "injected_water_fraction": case.boundary.injected_water_fraction,
            "corner_correction": (
                "По Маскету"
                if case.well_model.corner_correction == MUSKAT_CORNER_CORRECTION
                else "Без поправки"
            ),
            "time_start": case.time.start,
            "time_end": displayed_end,
            "time_end_unit": displayed_end_unit,
            "initial_step": case.time.initial_step,
            "save_every": case.time.save_every,
            "transport_scheme": (
                "Неявный"
                if case.numerics.transport_scheme == "implicit"
                else "Явный"
            ),
            "transport_acceleration": {
                "auto": "Автоматически",
                "numba": "Numba",
                "python": "Без Numba",
            }[case.numerics.transport_acceleration],
            "target_cfl": case.time.target_cfl,
            "max_saturation_change": case.numerics.max_saturation_change,
            "transport_rtol": case.numerics.transport_rtol,
            "transport_max_iterations": case.numerics.transport_max_iterations,
            "pressure_rtol": case.numerics.pressure_rtol,
            "pressure_atol": case.numerics.pressure_atol,
            "pressure_max_iterations": case.numerics.pressure_max_iterations,
            "saturation_tolerance": case.numerics.saturation_tolerance,
        }
        self._loading_form = True
        try:
            for key, value in values.items():
                self.inputs[key].set(self._format_input_value(value))
        finally:
            self._loading_form = False
        self._time_end_unit_previous = displayed_end_unit
        self._material_mode_changed()

    @staticmethod
    def _format_input_value(value) -> str:
        return value if isinstance(value, str) else f"{value:g}"

    @staticmethod
    def _display_time_end(seconds: float) -> tuple[float, str]:
        """Choose a readable unit without changing the stored SI value."""
        value = float(seconds)
        if value >= TIME_END_UNITS_SECONDS["сут"]:
            return value / TIME_END_UNITS_SECONDS["сут"], "сут"
        if value >= TIME_END_UNITS_SECONDS["ч"]:
            return value / TIME_END_UNITS_SECONDS["ч"], "ч"
        return value, "с"

    @staticmethod
    def _time_end_seconds(displayed_value: str | float, unit: str) -> float:
        """Convert the final time shown in the GUI to the solver SI value."""
        try:
            scale = TIME_END_UNITS_SECONDS[unit]
        except KeyError as exc:
            raise ConfigurationError(f"неизвестная единица времени: {unit}") from exc
        return float(displayed_value) * scale

    def _case_from_form(self):
        base = self._applied_case
        if base is None:
            path = self.case_path.get().strip()
            if not path:
                raise ConfigurationError("Сначала создайте или загрузите проект")
            base = load_case(path)
        project_name = self.inputs["project_name"].get().strip()
        if not project_name:
            raise ConfigurationError("название проекта не может быть пустым")
        formulation = (
            UNIFORM_POINT_WELL_FORMULATION
            if self.inputs["formulation"].get()
            == "Равномерная сетка, точечные скважины"
            else HYDRODYNAMIC_FORMULATION
        )
        geometry = GeometryConfig(
            lx=float(self.inputs["lx"].get()),
            ly=float(self.inputs["ly"].get()),
            thickness=float(self.inputs["thickness"].get()),
            injector_radius=float(self.inputs["injector_radius"].get()),
            producer_radius=float(self.inputs["producer_radius"].get()),
        )
        mesh = MeshConfig(
            nx=int(self.inputs["nx"].get()),
            ny=int(self.inputs["ny"].get()),
            pilot_nx=base.mesh.pilot_nx,
            pilot_ny=base.mesh.pilot_ny,
            pressure_bias=float(self.inputs["pressure_bias"].get()),
            max_trace_steps=base.mesh.max_trace_steps,
            generator={
                "ТФКП: точечный источник–сток": "point_source_isobars",
                "Гладкая (не φ–ψ)": "body_fitted_structured",
                "Равномерная прямоугольная": "uniform_cartesian",
            }[self.inputs["mesh_generator"].get()],
            streamline_edge_fraction=float(
                self.inputs["streamline_edge_fraction"].get()
            ),
            minimum_cell_angle_deg=float(
                self.inputs["minimum_cell_angle_deg"].get()
            ),
        )
        time = TimeConfig(
            start=float(self.inputs["time_start"].get()),
            end=self._time_end_seconds(
                self.inputs["time_end"].get(),
                self.inputs["time_end_unit"].get(),
            ),
            initial_step=float(self.inputs["initial_step"].get()),
            target_cfl=float(self.inputs["target_cfl"].get()),
            save_every=float(self.inputs["save_every"].get()),
        )
        numerics = NumericsConfig(
            pressure_rtol=float(self.inputs["pressure_rtol"].get()),
            pressure_atol=float(self.inputs["pressure_atol"].get()),
            pressure_max_iterations=int(self.inputs["pressure_max_iterations"].get()),
            saturation_tolerance=float(self.inputs["saturation_tolerance"].get()),
            transport_rtol=float(self.inputs["transport_rtol"].get()),
            transport_max_iterations=int(self.inputs["transport_max_iterations"].get()),
            max_saturation_change=float(self.inputs["max_saturation_change"].get()),
            transport_scheme=(
                "implicit" if self.inputs["transport_scheme"].get() == "Неявный" else "explicit"
            ),
            transport_acceleration={
                "Автоматически": "auto",
                "Numba": "numba",
                "Без Numba": "python",
            }[self.inputs["transport_acceleration"].get()],
        )
        if self.inputs["material_mode"].get() == "Однородная среда":
            permeability = float(self.inputs["permeability"].get())
            porosity = float(self.inputs["porosity"].get())
            if permeability <= 0.0:
                raise ConfigurationError("проницаемость должна быть положительной")
            if not 0.0 < porosity <= 1.0:
                raise ConfigurationError("пористость должна быть в интервале (0, 1]")
            medium = {
                "permeability": {
                    "type": "constant",
                    "representation": "isotropic",
                    "value_m2": permeability,
                },
                "porosity": {"type": "constant", "value": porosity},
            }
        else:
            medium = base.medium
        water = PhaseConfig(
            density=float(self.inputs["water_density"].get()),
            viscosity=float(self.inputs["water_viscosity"].get()),
        )
        oil = PhaseConfig(
            density=float(self.inputs["oil_density"].get()),
            viscosity=float(self.inputs["oil_viscosity"].get()),
        )
        relperm = RelPermConfig(
            swc=float(self.inputs["swc"].get()),
            sor=float(self.inputs["sor"].get()),
            krw_end=float(self.inputs["krw_end"].get()),
            kro_end=float(self.inputs["kro_end"].get()),
            nw=float(self.inputs["nw"].get()),
            no=float(self.inputs["no"].get()),
        )
        boundary = BoundaryConfig(
            injector_pressure=float(self.inputs["injector_pressure"].get()),
            producer_pressure=float(self.inputs["producer_pressure"].get()),
            injected_water_fraction=float(
                self.inputs["injected_water_fraction"].get()
            ),
        )
        well_model = WellModelConfig(
            corner_correction=(
                MUSKAT_CORNER_CORRECTION
                if self.inputs["corner_correction"].get() == "По Маскету"
                else NO_CORNER_CORRECTION
            )
        )
        return replace(
            base,
            name=project_name,
            formulation=formulation,
            geometry=geometry,
            mesh=mesh,
            medium=medium,
            water=water,
            oil=oil,
            relperm=relperm,
            initial_water_saturation=float(
                self.inputs["initial_water_saturation"].get()
            ),
            boundary=boundary,
            time=time,
            numerics=numerics,
            well_model=well_model,
        )

    @staticmethod
    def _project_mapping(case) -> dict:
        return {
            "schema_version": 1,
            "project": {"name": case.name, "unit_system": "SI"},
            "formulation": case.formulation,
            "well_model": {
                "corner_correction": case.well_model.corner_correction,
            },
            "geometry": {
                "type": (
                    "five_spot_point_wells"
                    if case.formulation == UNIFORM_POINT_WELL_FORMULATION
                    else "five_spot_cell"
                ),
                "size_m": [case.geometry.lx, case.geometry.ly],
                "thickness_m": case.geometry.thickness,
                "wells": [
                    {"id": "injector", "corner": "bottom_left", "radius_m": case.geometry.injector_radius},
                    {"id": "producer", "corner": "top_right", "radius_m": case.geometry.producer_radius},
                ],
            },
            "mesh": {
                "generator": case.mesh.generator,
                "cells": (
                    {"x": case.mesh.nx, "y": case.mesh.ny}
                    if case.formulation == UNIFORM_POINT_WELL_FORMULATION
                    else {
                        "along_streamline": case.mesh.nx,
                        "across_streamlines": case.mesh.ny,
                    }
                ),
                "pilot_cells": {
                    "along_streamline": case.mesh.pilot_nx,
                    "across_streamlines": case.mesh.pilot_ny,
                },
                "pressure_bias": case.mesh.pressure_bias,
                "max_trace_steps": case.mesh.max_trace_steps,
                "streamline_edge_fraction": case.mesh.streamline_edge_fraction,
                "minimum_cell_angle_deg": case.mesh.minimum_cell_angle_deg,
            },
            "medium": case.medium,
            "phases": {
                "wetting": {
                    "id": "water",
                    "density_kg_m3": case.water.density,
                    "dynamic_viscosity_pa_s": case.water.viscosity,
                },
                "nonwetting": {
                    "id": "oil",
                    "density_kg_m3": case.oil.density,
                    "dynamic_viscosity_pa_s": case.oil.viscosity,
                },
            },
            "relative_permeability": {
                "model": "power_law",
                "saturation": {
                    "variable": "water",
                    "irreducible_wetting": case.relperm.swc,
                    "residual_nonwetting": case.relperm.sor,
                    "outside_physical_range": "clamp",
                },
                "wetting": {"endpoint": case.relperm.krw_end, "exponent": case.relperm.nw},
                "nonwetting": {"endpoint": case.relperm.kro_end, "exponent": case.relperm.no},
            },
            "initial_conditions": {"water_saturation": case.initial_water_saturation},
            "flow_effects": {
                "gravity_enabled": False,
                "capillary_pressure": {"enabled": False},
            },
            "boundary_conditions": {
                "flow": {
                    "well_injector": {"type": "pressure", "value_pa": case.boundary.injector_pressure},
                    "well_producer": {"type": "pressure", "value_pa": case.boundary.producer_pressure},
                    "symmetry_path_a": {"type": "no_flow"},
                    "symmetry_path_b": {"type": "no_flow"},
                },
                "phase_composition": {
                    "well_injector": {
                        "type": "injected_fraction",
                        "water_fraction": case.boundary.injected_water_fraction,
                    },
                    "well_producer": {"type": "advective_outflow"},
                    "symmetry_path_a": {"type": "no_flow"},
                    "symmetry_path_b": {"type": "no_flow"},
                },
            },
            "time": {
                "start_s": case.time.start,
                "end_s": case.time.end,
                "initial_step_s": case.time.initial_step,
                "adaptive": True,
                "target_cfl": case.time.target_cfl,
                "save_every_s": case.time.save_every,
            },
            "numerics": {
                "pressure": {
                    "relative_tolerance": case.numerics.pressure_rtol,
                    "absolute_tolerance": case.numerics.pressure_atol,
                    "max_iterations": case.numerics.pressure_max_iterations,
                },
                "transport": {
                    "scheme": case.numerics.transport_scheme,
                    "coupling": (
                        "sequential_implicit"
                        if case.numerics.transport_scheme == "implicit"
                        else "impes"
                    ),
                    "time_discretization": (
                        "backward_euler"
                        if case.numerics.transport_scheme == "implicit"
                        else "forward_euler"
                    ),
                    "fractional_flow_scheme": "upwind",
                    "enforce_saturation_bounds": True,
                    "saturation_tolerance": case.numerics.saturation_tolerance,
                    "relative_tolerance": case.numerics.transport_rtol,
                    "max_iterations": case.numerics.transport_max_iterations,
                    "max_saturation_change": case.numerics.max_saturation_change,
                    "acceleration": case.numerics.transport_acceleration,
                },
            },
            "output": {
                "store_field_frames": case.output.store_field_frames,
                "well_history_every_s": case.output.well_history_every,
                "store_cell_velocity": True,
                "compression": "gzip",
            },
        }

    @staticmethod
    def _rebase_medium_files(
        medium: dict,
        source_directory: Path,
        target_directory: Path,
    ) -> dict:
        """Keep relative NPZ field references valid after Save As."""
        result = deepcopy(medium)
        for key in ("permeability", "porosity"):
            specification = result.get(key, {})
            if specification.get("type") != "spatial_field" or "file" not in specification:
                continue
            field_path = Path(specification["file"])
            if not field_path.is_absolute():
                field_path = (source_directory / field_path).resolve()
            try:
                stored_path = Path(os.path.relpath(field_path, target_directory))
            except ValueError:
                stored_path = field_path
            specification["file"] = stored_path.as_posix()
        return result

    def _save_project(self) -> None:
        try:
            case = self._case_from_form()
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON project", "*.json")],
                initialfile=f"{case.name}.json",
            )
            if not path:
                return
            output_path = Path(path).resolve()
            mapping = self._project_mapping(case)
            mapping["medium"] = self._rebase_medium_files(
                mapping["medium"],
                case.source_path.parent,
                output_path.parent,
            )
            output_path.write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.case_path.set(str(output_path))
            self._applied_case = load_case(output_path)
            self._last_project_path = output_path
            self._new_project_unsaved = False
            self._set_parameters_clean("✓ Параметры сохранены в файл и применены")
            self.status.set(f"Проект сохранён: {output_path.name}")
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", str(exc))

    def _preview(self) -> None:
        try:
            case = self._case_from_form()
            self._draw_geometry(
                FiveSpotGeometry.from_config(case.geometry),
                case.formulation,
            )
            self._show_details(
                {
                    "Проект": case.name,
                    "Постановка": (
                        "равномерная сетка, угловые точечные четверти скважин"
                        if case.formulation == UNIFORM_POINT_WELL_FORMULATION
                        else "гидродинамическая сетка, конечные границы скважин"
                    ),
                    "Размер": f"{case.geometry.lx:g} × {case.geometry.ly:g} м",
                    "Толщина": f"{case.geometry.thickness:g} м",
                    "Нагнетатель": f"r = {case.geometry.injector_radius:g} м",
                    "Добывающая": f"r = {case.geometry.producer_radius:g} м",
                    "Рабочая сетка": f"{case.mesh.nx} × {case.mesh.ny}",
                    "Проницаемость": self._medium_summary(case),
                    "ОФП": (
                        f"Swc={case.relperm.swc:g}, Sor={case.relperm.sor:g}; "
                        f"nw={case.relperm.nw:g}, no={case.relperm.no:g}"
                    ),
                    "Давления скважин": (
                        f"{case.boundary.injector_pressure:g} / "
                        f"{case.boundary.producer_pressure:g} Па"
                    ),
                    "Угловая поправка": (
                        "Маскет"
                        if case.well_model.corner_correction == MUSKAT_CORNER_CORRECTION
                        else "нет"
                    ),
                    "Построение": (
                        "равномерная прямоугольная"
                        if case.mesh.generator == "uniform_cartesian"
                        else (
                            "ТФКП: точечный источник–сток"
                            if case.mesh.generator == "point_source_isobars"
                            else "гладкое body-fitted (не φ–ψ)"
                        )
                    ),
                    "Интервал времени": f"{case.time.start:g} … {case.time.end:g} с",
                    "Макс. шаг": f"{case.time.initial_step:g} с",
                    "Интервал кадров": f"{case.time.save_every:g} с",
                    "Метод переноса": "неявный" if case.numerics.transport_scheme == "implicit" else "явный",
                    "Целевой CFL": f"{case.time.target_cfl:g}",
                    "Ограничение ΔS": f"{case.numerics.max_saturation_change:g}",
                }
            )
            self.status.set("Предварительный просмотр области")
        except (ConfigurationError, ValueError) as exc:
            messagebox.showerror("Некорректная геометрия", str(exc))

    @staticmethod
    def _medium_summary(case) -> str:
        mode, permeability, porosity = Application._material_form_values(case)
        if mode == "Однородная среда":
            return f"k={permeability} м², φ={porosity}"
        return "неоднородное поле из проекта"

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        state = tk.DISABLED if value else tk.NORMAL
        for button in (self.preview_button, self.mesh_button, self.run_button):
            button.configure(state=state)
        self.apply_button.configure(
            state=tk.DISABLED if value or not self._parameters_dirty else tk.NORMAL
        )
        can_stop = value and self._active_target == "run" and not self._stop_event.is_set()
        self.stop_button.configure(state=tk.NORMAL if can_stop else tk.DISABLED)

    def _start(self, target: str) -> None:
        if self._busy:
            return
        if self._parameters_dirty:
            messagebox.showwarning(
                "Параметры не применены",
                "В полях есть изменения. Нажмите «Применить параметры», "
                "чтобы программа проверила и запомнила их перед запуском.",
            )
            return
        case = self._applied_case
        if case is None:
            messagebox.showwarning(
                "Параметры не применены",
                "Сначала загрузите проект или нажмите «Применить параметры».",
            )
            return
        if target == "run" and self._new_project_unsaved:
            messagebox.showwarning(
                "Проект не сохранён",
                "Перед расчётом сохраните новый входной файл кнопкой «Сохранить как…».",
            )
            return
        self._active_target = target
        self._stop_event.clear()
        self.progress["value"] = 0
        self.run_progress["value"] = 0
        self.run_info.set("Подготовка расчёта…" if target == "run" else "Построение и проверка сетки…")
        self._set_busy(True)
        threading.Thread(target=self._worker, args=(target, case), daemon=True).start()

    def _validate(self) -> None:
        self._start("validate")

    def _run(self) -> None:
        self._start("run")

    def _stop(self) -> None:
        if not self._busy or self._active_target != "run":
            return
        self._stop_event.set()
        self.stop_button.configure(state=tk.DISABLED)
        self.run_info.set("Запрошена остановка — завершается текущий шаг…")
        self.status.set("Останавливаю расчёт после текущего шага")

    def _worker(self, target: str, case) -> None:
        started = time.perf_counter()

        def callback(fraction: float, message: str) -> None:
            elapsed = time.perf_counter() - started
            if 0.01 < fraction < 1.0:
                remaining = elapsed * (1.0 - fraction) / fraction
            else:
                remaining = 0.0
            self.events.put(("progress", fraction, message, elapsed, remaining))

        try:
            if target == "validate":
                prepared = prepare_case(case, callback)
                elapsed = time.perf_counter() - started
                self.events.put(("prepared", prepared, case_summary(case, prepared), elapsed))
            else:
                result = run_case(case, callback, self._stop_event.is_set)
                project_path = Path(self.case_path.get()).resolve()
                output = create_run_output_path(project_path, result)
                save_results(result, output, source_project=project_path)
                elapsed = time.perf_counter() - started
                self.events.put(("result", result, output, elapsed))
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self.events.put(("error", str(exc), elapsed))

    def _poll(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    fraction, message, elapsed, remaining = event[1:]
                    percent = 100.0 * fraction
                    self.progress["value"] = percent
                    self.run_progress["value"] = percent
                    readable = self._progress_message(message)
                    self.status.set(readable)
                    if self._stop_event.is_set() and self._active_target == "run":
                        self.run_info.set(
                            f"Остановка после текущего шага · прошло {self._format_duration(elapsed)}"
                        )
                    else:
                        self.run_info.set(
                            f"{percent:.1f}% · {readable} · прошло {self._format_duration(elapsed)} · "
                            f"осталось примерно {self._format_duration(remaining)}"
                        )
                elif kind == "prepared":
                    self._draw_mesh(event[1].mesh)
                    self._show_details(event[2])
                    self.status.set("Сетка построена и прошла проверку")
                    self.run_progress["value"] = 100
                    self.run_info.set(f"Сетка готова за {self._format_duration(event[3])}")
                    self._active_target = ""
                    self._set_busy(False)
                elif kind == "result":
                    self._set_result(event[1])
                    if event[1].completed:
                        self.progress["value"] = 100
                        self.run_progress["value"] = 100
                        self.status.set(f"Готово: {event[2].parent}")
                        self.run_info.set(
                            f"Расчёт завершён за {self._format_duration(event[3])}; "
                            f"файлы сохранены в {event[2].parent.name}"
                        )
                    else:
                        self.status.set(f"Расчёт остановлен: {event[2].parent}")
                        self.run_info.set(
                            f"Остановлено на t = {self._format_time(float(event[1].times[-1]))} · "
                            f"прошло {self._format_duration(event[3])} · промежуточный результат открыт"
                        )
                    self._active_target = ""
                    self._set_busy(False)
                elif kind == "error":
                    self.status.set("Ошибка")
                    self.run_info.set(f"Ошибка через {self._format_duration(event[2])}")
                    self._active_target = ""
                    self._set_busy(False)
                    messagebox.showerror("Ошибка расчёта", event[1])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0.0:
            return "—"
        total = int(round(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours} ч {minutes:02d} мин"
        if minutes:
            return f"{minutes} мин {secs:02d} с"
        return f"{secs} с"

    @staticmethod
    def _progress_message(message: str) -> str:
        translations = {
            "building pilot mesh": "строится пилотная сетка",
            "solving pilot pressure field": "решается пилотное поле давления",
            "building boundary-fitted structured mesh": "строится рабочая сетка",
            "building uniform Cartesian mesh": "строится равномерная прямоугольная сетка",
            "solving pressure on the final mesh": "решается давление на рабочей сетке",
            "completed": "расчёт завершён",
            "stopped": "расчёт остановлен",
        }
        if message in translations:
            return translations[message]
        if message.startswith("time="):
            return "шаг по времени: " + message.replace("time=", "t=")
        if message.startswith("retrying implicit transport"):
            return "уменьшается неявный шаг: уточняется нелинейное решение"
        if message.startswith("refining implicit step"):
            return "уменьшается неявный шаг для точности фронта"
        if message.startswith("retrying explicit transport"):
            return "уменьшается явный шаг для выполнения условия CFL"
        return message

    def _transform(self, points: np.ndarray):
        width = max(self.canvas.winfo_width(), 200)
        height = max(self.canvas.winfo_height(), 200)
        flat = np.asarray(points).reshape(-1, 2)
        xmin, ymin = np.min(flat, axis=0)
        xmax, ymax = np.max(flat, axis=0)
        scale = 0.88 * min(width / max(xmax - xmin, 1.0e-30), height / max(ymax - ymin, 1.0e-30))
        ox = 0.5 * (width - scale * (xmax - xmin))
        oy = 0.5 * (height - scale * (ymax - ymin))
        return lambda point: (
            ox + scale * (point[0] - xmin),
            height - oy - scale * (point[1] - ymin),
        )

    def _draw_geometry(
        self,
        geometry: FiveSpotGeometry,
        formulation: str = HYDRODYNAMIC_FORMULATION,
    ) -> None:
        self.canvas.delete("all")
        t = np.linspace(0.0, 1.0, 100)
        if formulation == UNIFORM_POINT_WELL_FORMULATION:
            outline = np.array(
                [
                    [0.0, 0.0],
                    [geometry.lx, 0.0],
                    [geometry.lx, geometry.ly],
                    [0.0, geometry.ly],
                ]
            )
            transform = self._transform(outline)
            coords = [value for point in outline for value in transform(point)]
            self.canvas.create_polygon(
                *coords,
                fill="#edf2f3",
                outline="#223238",
                width=2,
            )
            inlet = [
                value
                for point in geometry.injector_arc(t)
                for value in transform(point)
            ]
            outlet = [
                value
                for point in geometry.producer_arc(t)
                for value in transform(point)
            ]
            self.canvas.create_line(*inlet, fill="#cf5c36", width=4, dash=(5, 3))
            self.canvas.create_line(*outlet, fill="#275dad", width=4, dash=(5, 3))
            self.canvas.create_text(
                *transform(geometry.injector_arc(0.5)),
                text="  ¼ нагнетательной скважины\n  в угловой ячейке",
                anchor=tk.W,
            )
            self.canvas.create_text(
                *transform(geometry.producer_arc(0.5)),
                text="¼ добывающей скважины  \nв угловой ячейке  ",
                anchor=tk.E,
            )
            return
        outline = np.vstack(
            [
                geometry.boundary_a(t),
                geometry.producer_arc(t)[1:],
                geometry.boundary_b(t)[::-1][1:],
                geometry.injector_arc(t)[::-1][1:],
            ]
        )
        transform = self._transform(outline)
        coords = [value for point in outline for value in transform(point)]
        self.canvas.create_polygon(*coords, fill="#edf2f3", outline="#223238", width=2)
        inlet = [value for point in geometry.injector_arc(t) for value in transform(point)]
        outlet = [value for point in geometry.producer_arc(t) for value in transform(point)]
        self.canvas.create_line(*inlet, fill="#cf5c36", width=4)
        self.canvas.create_line(*outlet, fill="#275dad", width=4)
        self.canvas.create_text(*transform(geometry.injector_arc(0.5)), text="  нагнетатель", anchor=tk.W)
        self.canvas.create_text(*transform(geometry.producer_arc(0.5)), text="добывающая  ", anchor=tk.E)

    def _draw_mesh(self, mesh) -> None:
        self.canvas.delete("all")
        transform = self._transform(mesh.nodes)
        self._draw_mesh_lines(mesh, transform)

    def _draw_mesh_lines(self, mesh, transform, color: str = "#6c8790", width: int = 1) -> None:
        for i in range(mesh.nx + 1):
            coords = [value for point in mesh.nodes[i] for value in transform(point)]
            self.canvas.create_line(*coords, fill=color, width=width)
        for j in range(mesh.ny + 1):
            coords = [value for point in mesh.nodes[:, j] for value in transform(point)]
            self.canvas.create_line(*coords, fill=color, width=width)

    def _draw_mesh_boundary(self, mesh, transform) -> None:
        for boundary in (
            mesh.nodes[:, 0],
            mesh.nodes[-1, :],
            mesh.nodes[::-1, -1],
            mesh.nodes[0, ::-1],
        ):
            coords = [value for point in boundary for value in transform(point)]
            self.canvas.create_line(*coords, fill="#52666d", width=2)

    def _set_result(self, result) -> None:
        self._stop_animation()
        self._result = result
        frame_count = len(result.times)
        self.time_scale.configure(from_=0, to=max(frame_count - 1, 1))
        self.time_scale.set(0 if result.completed else max(frame_count - 1, 0))
        self.field_choice.set("Водонасыщенность")
        self._show_details(
            {
                "Расчёт": result.case_name,
                "Постановка": (
                    "равномерная сетка, точечные четверти скважин"
                    if result.formulation == UNIFORM_POINT_WELL_FORMULATION
                    else "гидродинамическая сетка, конечные скважины"
                ),
                "Статус": "завершён" if result.completed else "остановлен; данные неполные",
                "Метод переноса": (
                    "неявный" if result.transport_scheme == "implicit" else "явный"
                ),
                "Сохранено кадров": frame_count,
                "Конечное время": self._format_time(float(result.times[-1])),
                "Дебит добывающей": f"{result.producer_liquid_rate[-1] * 86400.0:.6g} м³/сут",
                "Дебит воды": f"{result.producer_water_rate[-1] * 86400.0:.6g} м³/сут",
                "Обводнённость добывающей": f"{100.0 * result.producer_water_cut[-1]:.3f} %",
                "Макс. невязка баланса": f"{np.max(result.mass_balance_error):.3e}",
            }
        )
        self._render_result()

    @staticmethod
    def _format_time(value: float) -> str:
        if abs(value) >= 2.0 * 86400.0:
            return f"{value / 86400.0:.4g} сут"
        if abs(value) >= 2.0 * 3600.0:
            return f"{value / 3600.0:.4g} ч"
        return f"{value:.6g} с"

    def _on_time_change(self, _value=None) -> None:
        if self._result is not None:
            self._render_result()

    def _frame_index(self) -> int:
        if self._result is None:
            return 0
        return int(np.clip(round(float(self.time_scale.get())), 0, len(self._result.times) - 1))

    def _toggle_animation(self) -> None:
        if self._result is None:
            return
        if self.field_choice.get() == "Скважинные функции":
            self.field_choice.set("Водонасыщенность")
        if self._animation_job is not None:
            self._stop_animation()
            return
        if self._frame_index() >= len(self._result.times) - 1:
            self.time_scale.set(0)
        self.play_button.configure(text="■")
        self._animation_job = self.after(180, self._advance_animation)

    def _advance_animation(self) -> None:
        self._animation_job = None
        if self._result is None:
            self._stop_animation()
            return
        index = self._frame_index()
        if index >= len(self._result.times) - 1:
            self._stop_animation()
            return
        self.time_scale.set(index + 1)
        self._render_result()
        self._animation_job = self.after(180, self._advance_animation)

    def _stop_animation(self) -> None:
        if self._animation_job is not None:
            self.after_cancel(self._animation_job)
            self._animation_job = None
        if hasattr(self, "play_button"):
            self.play_button.configure(text="▶")

    def _render_result(self) -> None:
        result = self._result
        if result is None:
            return
        field = self.field_choice.get()
        if field == "Скважинные функции":
            self.view_select.configure(state=tk.DISABLED)
            self._stop_animation()
            self.time_label.configure(text="весь интервал")
            self._draw_well_functions(result)
            return
        self.view_select.configure(state="readonly")
        index = self._frame_index()
        time = float(result.times[index])
        self.time_label.configure(text=f"t = {self._format_time(time)}")
        if field == "Давление":
            values = result.pressure[index]
            limits = (float(np.min(result.pressure)), float(np.max(result.pressure)))
            title = "Давление, Па"
        else:
            values = result.saturation[index]
            limits = (0.0, 1.0)
            title = "Водонасыщенность"
        self._draw_scalar_field(result.mesh, values, limits, title, time)

    @staticmethod
    def _brown_blue(value: float) -> str:
        value = float(np.clip(value, 0.0, 1.0))
        rgb = np.array(
            [np.interp(value, BROWN_BLUE_POSITIONS, BROWN_BLUE_RGB[:, channel]) for channel in range(3)]
        )
        return "#{:02x}{:02x}{:02x}".format(*np.rint(rgb).astype(int))

    def _draw_scalar_field(self, mesh, values, limits, title: str, time: float) -> None:
        self.canvas.delete("all")
        lo, hi = limits
        transform = self._transform(mesh.nodes)
        view = self.view_choice.get()
        if view in {"Заливка", "Заливка + сетка"}:
            self._draw_filled_cells(mesh, values, limits, transform)
            if view == "Заливка + сетка":
                self._draw_mesh_lines(mesh, transform, color="#536d76")
            else:
                self._draw_mesh_boundary(mesh, transform)
        else:
            self._draw_isolines(mesh, values, limits, transform)
        self.canvas.create_text(
            12,
            12,
            anchor=tk.NW,
            text=f"{title};  t = {self._format_time(time)}\nдиапазон шкалы: {lo:.5g} … {hi:.5g}",
            fill="#17262b",
            font=("Segoe UI", 10, "bold"),
        )
        self._draw_colorbar(
            lo,
            hi,
            "Sw" if title == "Водонасыщенность" else "p, Па",
        )

    def _draw_filled_cells(
        self,
        mesh,
        values,
        limits,
        transform,
        band_edges: np.ndarray | None = None,
    ) -> None:
        lo, hi = limits
        span = max(hi - lo, 1.0e-15)
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                field_value = float(values[i, j])
                if band_edges is None:
                    fraction = (field_value - lo) / span
                else:
                    band = int(np.searchsorted(band_edges, field_value, side="right") - 1)
                    band = int(np.clip(band, 0, len(band_edges) - 2))
                    midpoint = 0.5 * (band_edges[band] + band_edges[band + 1])
                    fraction = (midpoint - lo) / span
                vertices = [
                    mesh.nodes[i, j],
                    mesh.nodes[i + 1, j],
                    mesh.nodes[i + 1, j + 1],
                    mesh.nodes[i, j + 1],
                ]
                coords = [value for point in vertices for value in transform(point)]
                self.canvas.create_polygon(
                    *coords,
                    fill=self._brown_blue(fraction),
                    outline="",
                )

    def _draw_isolines(self, mesh, values, limits, transform) -> None:
        lo, hi = limits
        band_edges = np.linspace(lo, hi, 12)
        self._draw_filled_cells(mesh, values, limits, transform, band_edges)
        nodal_values = _cell_values_to_nodes(values)
        for level in band_edges[1:-1]:
            color = "#25383f"
            screen_segments = []
            for first, second in _contour_segments(
                mesh.nodes, nodal_values, float(level)
            ):
                x0, y0 = transform(first)
                x1, y1 = transform(second)
                self.canvas.create_line(x0, y0, x1, y1, fill=color, width=1)
                screen_segments.append((x0, y0, x1, y1))
            if screen_segments:
                x0, y0, x1, y1 = max(
                    screen_segments,
                    key=lambda segment: (segment[2] - segment[0]) ** 2
                    + (segment[3] - segment[1]) ** 2,
                )
                x, y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                label = f"{level:.3g}"
                half_width = 3.2 * len(label) + 2.0
                self.canvas.create_rectangle(
                    x - half_width,
                    y - 7,
                    x + half_width,
                    y + 7,
                    fill="#f7fafb",
                    outline=color,
                )
                self.canvas.create_text(
                    x,
                    y,
                    text=label,
                    fill=color,
                    font=("Segoe UI", 8),
                )
        self._draw_mesh_boundary(mesh, transform)

    def _draw_colorbar(self, low: float, high: float, label: str) -> None:
        width = max(self.canvas.winfo_width(), 200)
        height = max(self.canvas.winfo_height(), 200)
        x0, x1 = width - 42, width - 22
        bar_height = int(min(420, max(180, 0.58 * height), max(height - 110, 80)))
        y0 = int(0.5 * (height - bar_height))
        y1 = y0 + bar_height

        # One horizontal line per screen pixel gives a visually continuous
        # gradient. Canvas rectangles leave sub-pixel seams on scaled displays.
        for y in range(y0, y1 + 1):
            fraction = (y1 - y) / max(y1 - y0, 1)
            self.canvas.create_line(
                x0,
                y,
                x1,
                y,
                fill=self._brown_blue(fraction),
                width=1,
            )
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#52666d")
        for fraction in np.linspace(0.0, 1.0, 6):
            y = y1 - fraction * (y1 - y0)
            value = low + fraction * (high - low)
            self.canvas.create_line(x0 - 4, y, x0, y, fill="#52666d")
            self.canvas.create_text(
                x0 - 7,
                y,
                anchor=tk.E,
                text=f"{value:.4g}",
                fill="#17262b",
            )
        if label:
            self.canvas.create_text(
                (x0 + x1) / 2,
                y0 - 16,
                text=label,
                fill="#17262b",
                font=("Segoe UI", 9, "bold"),
            )

    def _draw_well_functions(self, result) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 500)
        height = max(self.canvas.winfo_height(), 400)
        left, right = 72.0, width - 24.0
        upper_top, upper_bottom = 48.0, 0.49 * height
        lower_top, lower_bottom = 0.61 * height, height - 46.0
        time = np.asarray(result.diagnostic_times, dtype=float)
        if time[-1] >= 2.0 * 86400.0:
            x = time / 86400.0
            x_label = "Время, сут"
        elif time[-1] >= 2.0 * 3600.0:
            x = time / 3600.0
            x_label = "Время, ч"
        else:
            x = time
            x_label = "Время, с"
        q_liquid = np.asarray(result.producer_liquid_rate) * 86400.0
        q_water = np.asarray(result.producer_water_rate) * 86400.0
        water_cut = 100.0 * np.asarray(result.producer_water_cut)

        self._draw_chart(
            x,
            ((q_liquid, "#1f5f99", "жидкость"), (q_water, "#28a6a1", "вода")),
            (left, upper_top, right, upper_bottom),
            "Добывающая скважина: дебит, м³/сут",
            include_zero=True,
        )
        self._draw_chart(
            x,
            ((water_cut, "#cf5c36", "обводнённость"),),
            (left, lower_top, right, lower_bottom),
            "Обводнённость добывающей, %",
            fixed_y=(0.0, 100.0),
        )
        self.canvas.create_text((left + right) / 2.0, height - 14.0, text=x_label, fill="#17262b")

    def _draw_chart(
        self,
        x: np.ndarray,
        series,
        box: tuple[float, float, float, float],
        title: str,
        include_zero: bool = False,
        fixed_y: tuple[float, float] | None = None,
    ) -> None:
        left, top, right, bottom = box
        self.canvas.create_rectangle(left, top, right, bottom, outline="#9caaae")
        self.canvas.create_text(left, top - 18, anchor=tk.W, text=title, fill="#17262b", font=("Segoe UI", 10, "bold"))
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmax <= xmin:
            xmax = xmin + 1.0
        if fixed_y is None:
            finite = np.concatenate([np.asarray(values)[np.isfinite(values)] for values, _, _ in series])
            ymin, ymax = float(np.min(finite)), float(np.max(finite))
            if include_zero:
                ymin, ymax = min(ymin, 0.0), max(ymax, 0.0)
            margin = 0.06 * max(ymax - ymin, abs(ymax), 1.0)
            ymin, ymax = ymin - margin, ymax + margin
        else:
            ymin, ymax = fixed_y

        def point(x_value, y_value):
            px = left + (float(x_value) - xmin) / (xmax - xmin) * (right - left)
            py = bottom - (float(y_value) - ymin) / max(ymax - ymin, 1.0e-30) * (bottom - top)
            return px, py

        self.canvas.create_text(left - 8, top, anchor=tk.E, text=f"{ymax:.4g}", fill="#52666d")
        self.canvas.create_text(left - 8, bottom, anchor=tk.E, text=f"{ymin:.4g}", fill="#52666d")
        self.canvas.create_text(left, bottom + 13, anchor=tk.W, text=f"{xmin:.4g}", fill="#52666d")
        self.canvas.create_text(right, bottom + 13, anchor=tk.E, text=f"{xmax:.4g}", fill="#52666d")
        legend_x = right
        for values, color, label in reversed(series):
            mask = np.isfinite(x) & np.isfinite(values)
            plot_x = x[mask]
            plot_y = np.asarray(values)[mask]
            if plot_x.size == 1:
                px, py = point(plot_x[0], plot_y[0])
                self.canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill=color, outline=color)
            elif plot_x.size > 1:
                coords = [coordinate for xv, yv in zip(plot_x, plot_y) for coordinate in point(xv, yv)]
                self.canvas.create_line(*coords, fill=color, width=2)
            label_width = 7.2 * len(label) + 27.0
            legend_x -= label_width
            self.canvas.create_line(legend_x, top - 17, legend_x + 16, top - 17, fill=color, width=3)
            self.canvas.create_text(legend_x + 20, top - 17, anchor=tk.W, text=label, fill="#52666d")

    def _show_details(self, summary: dict) -> None:
        self.details.delete("1.0", tk.END)
        for key, value in summary.items():
            self.details.insert(tk.END, f"{key}: {value}\n")


def main() -> None:
    Application().mainloop()


if __name__ == "__main__":
    main()
