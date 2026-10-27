from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when a project file is physically or structurally invalid."""


HYDRODYNAMIC_FORMULATION = "hydrodynamic_finite_wells"
UNIFORM_POINT_WELL_FORMULATION = "uniform_point_wells"
NO_CORNER_CORRECTION = "none"
MUSKAT_CORNER_CORRECTION = "muskat"


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not value > 0.0:
        raise ConfigurationError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class GeometryConfig:
    lx: float
    ly: float
    thickness: float
    injector_radius: float
    producer_radius: float

    def __post_init__(self) -> None:
        for name in ("lx", "ly", "thickness", "injector_radius", "producer_radius"):
            _positive(name, getattr(self, name))
        if self.injector_radius >= min(self.lx, self.ly):
            raise ConfigurationError("injector radius is too large for the rectangle")
        if self.producer_radius >= min(self.lx, self.ly):
            raise ConfigurationError("producer radius is too large for the rectangle")


@dataclass(frozen=True)
class MeshConfig:
    nx: int = 60
    ny: int = 40
    pilot_nx: int = 30
    pilot_ny: int = 24
    pressure_bias: float = 1.0
    max_trace_steps: int = 12000
    generator: str = "body_fitted_structured"
    streamline_edge_fraction: float = 0.4
    minimum_cell_angle_deg: float = 2.0

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.pilot_nx, self.pilot_ny) < 4:
            raise ConfigurationError("each mesh direction needs at least four cells")
        _positive("pressure_bias", self.pressure_bias)
        if self.generator not in {
            "body_fitted_structured",
            "point_source_isobars",
            "uniform_cartesian",
        }:
            raise ConfigurationError(
                "mesh generator must be 'body_fitted_structured', "
                "'point_source_isobars', or 'uniform_cartesian'"
            )
        if not 0.0 <= self.streamline_edge_fraction <= 1.0:
            raise ConfigurationError("streamline edge fraction must be in [0, 1]")
        if not 0.0 < self.minimum_cell_angle_deg < 90.0:
            raise ConfigurationError("minimum cell angle must be in (0, 90) degrees")


@dataclass(frozen=True)
class PhaseConfig:
    density: float
    viscosity: float

    def __post_init__(self) -> None:
        _positive("phase density", self.density)
        _positive("phase viscosity", self.viscosity)


@dataclass(frozen=True)
class RelPermConfig:
    swc: float
    sor: float
    krw_end: float
    kro_end: float
    nw: float
    no: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.swc < 1.0:
            raise ConfigurationError("irreducible wetting saturation must be in [0, 1)")
        if not 0.0 <= self.sor < 1.0:
            raise ConfigurationError("residual nonwetting saturation must be in [0, 1)")
        if self.swc + self.sor >= 1.0:
            raise ConfigurationError("irreducible and residual saturations must sum to less than one")
        if not 0.0 < self.krw_end <= 1.0 or not 0.0 < self.kro_end <= 1.0:
            raise ConfigurationError("relative-permeability endpoints must be in (0, 1]")
        _positive("wetting exponent", self.nw)
        _positive("nonwetting exponent", self.no)


@dataclass(frozen=True)
class BoundaryConfig:
    injector_pressure: float
    producer_pressure: float
    injected_water_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.injector_pressure <= self.producer_pressure:
            raise ConfigurationError("injector pressure must exceed producer pressure")
        if not 0.0 <= self.injected_water_fraction <= 1.0:
            raise ConfigurationError("injected water fraction must be in [0, 1]")


@dataclass(frozen=True)
class TimeConfig:
    start: float
    end: float
    initial_step: float
    target_cfl: float
    save_every: float

    def __post_init__(self) -> None:
        if self.start < 0.0 or self.end <= self.start:
            raise ConfigurationError("time interval must satisfy 0 <= start < end")
        _positive("initial time step", self.initial_step)
        _positive("save interval", self.save_every)
        if not 0.0 < self.target_cfl <= 1.0:
            raise ConfigurationError("target CFL must be in (0, 1]")


@dataclass(frozen=True)
class OutputConfig:
    """Controls field snapshots independently from scalar well histories."""

    store_field_frames: bool = True
    well_history_every: float | None = None

    def __post_init__(self) -> None:
        if self.well_history_every is not None:
            _positive("well-history interval", self.well_history_every)


@dataclass(frozen=True)
class WellModelConfig:
    """Controls the pressure connection of corner point wells."""

    corner_correction: str = NO_CORNER_CORRECTION

    def __post_init__(self) -> None:
        if self.corner_correction not in {
            NO_CORNER_CORRECTION,
            MUSKAT_CORNER_CORRECTION,
        }:
            raise ConfigurationError(
                "corner well correction must be 'none' or 'muskat'"
            )


@dataclass(frozen=True)
class NumericsConfig:
    pressure_rtol: float = 1.0e-9
    pressure_atol: float = 1.0e-12
    pressure_max_iterations: int = 4000
    saturation_tolerance: float = 1.0e-10
    transport_rtol: float = 1.0e-8
    transport_max_iterations: int = 20
    max_saturation_change: float = 1.0
    transport_scheme: str = "implicit"
    transport_acceleration: str = "auto"

    def __post_init__(self) -> None:
        _positive("pressure relative tolerance", self.pressure_rtol)
        _positive("pressure absolute tolerance", self.pressure_atol)
        _positive("saturation tolerance", self.saturation_tolerance)
        _positive("transport relative tolerance", self.transport_rtol)
        if self.pressure_max_iterations < 1:
            raise ConfigurationError("pressure maximum iterations must be at least one")
        if self.transport_max_iterations < 1:
            raise ConfigurationError("transport maximum iterations must be at least one")
        if not 0.0 < self.max_saturation_change <= 1.0:
            raise ConfigurationError("maximum saturation change must be in (0, 1]")
        if self.transport_scheme not in {"implicit", "explicit"}:
            raise ConfigurationError("transport scheme must be 'implicit' or 'explicit'")
        if self.transport_acceleration not in {"auto", "numba", "python"}:
            raise ConfigurationError(
                "transport acceleration must be 'auto', 'numba', or 'python'"
            )


@dataclass(frozen=True)
class CaseConfig:
    name: str
    source_path: Path
    formulation: str
    geometry: GeometryConfig
    mesh: MeshConfig
    medium: dict[str, Any]
    water: PhaseConfig
    oil: PhaseConfig
    relperm: RelPermConfig
    initial_water_saturation: float
    boundary: BoundaryConfig
    time: TimeConfig
    numerics: NumericsConfig
    output: OutputConfig
    well_model: WellModelConfig = field(default_factory=WellModelConfig)

    def __post_init__(self) -> None:
        if self.formulation not in {
            HYDRODYNAMIC_FORMULATION,
            UNIFORM_POINT_WELL_FORMULATION,
        }:
            raise ConfigurationError(f"unknown formulation '{self.formulation}'")
        if (
            self.formulation == UNIFORM_POINT_WELL_FORMULATION
            and self.mesh.generator != "uniform_cartesian"
        ):
            raise ConfigurationError(
                "uniform point-well formulation requires mesh.generator=uniform_cartesian"
            )
        if (
            self.formulation == HYDRODYNAMIC_FORMULATION
            and self.mesh.generator == "uniform_cartesian"
        ):
            raise ConfigurationError(
                "hydrodynamic finite-well formulation cannot use uniform_cartesian"
            )
        if not self.relperm.swc <= self.initial_water_saturation <= 1.0 - self.relperm.sor:
            raise ConfigurationError("initial water saturation is outside the physical range")


def default_case(source_path: str | Path | None = None) -> CaseConfig:
    """Return a self-contained, physically valid template for a new GUI project.

    The template deliberately does not read an example file.  This keeps the
    desktop application usable after it has been compiled and copied to a
    computer on which the source tree is absent.
    """
    source = (
        Path(source_path).resolve()
        if source_path is not None
        else (Path.cwd() / "new_five_spot_project.json").resolve()
    )
    return CaseConfig(
        name="new_five_spot_project",
        source_path=source,
        formulation=HYDRODYNAMIC_FORMULATION,
        geometry=GeometryConfig(
            lx=100.0,
            ly=100.0,
            thickness=10.0,
            injector_radius=0.1,
            producer_radius=0.1,
        ),
        mesh=MeshConfig(
            nx=60,
            ny=40,
            pilot_nx=30,
            pilot_ny=24,
            generator="point_source_isobars",
        ),
        medium={
            "permeability": {
                "type": "constant",
                "representation": "isotropic",
                "value_m2": 1.0e-12,
            },
            "porosity": {"type": "constant", "value": 0.2},
        },
        water=PhaseConfig(density=1000.0, viscosity=1.0e-3),
        oil=PhaseConfig(density=800.0, viscosity=5.0e-3),
        relperm=RelPermConfig(
            swc=0.2,
            sor=0.2,
            krw_end=1.0,
            kro_end=1.0,
            nw=2.0,
            no=2.0,
        ),
        initial_water_saturation=0.2,
        boundary=BoundaryConfig(
            injector_pressure=20.0e6,
            producer_pressure=10.0e6,
            injected_water_fraction=1.0,
        ),
        time=TimeConfig(
            start=0.0,
            end=86400.0,
            initial_step=600.0,
            target_cfl=0.5,
            save_every=3600.0,
        ),
        numerics=NumericsConfig(),
        output=OutputConfig(),
        well_model=WellModelConfig(MUSKAT_CORNER_CORRECTION),
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigurationError(
                "YAML support requires PyYAML; install the project with the 'yaml' extra"
            ) from exc
        value = yaml.safe_load(text)
        if not isinstance(value, dict):
            raise ConfigurationError("the project root must be a mapping")
        return value
    raise ConfigurationError("project files must use .json, .yaml or .yml")


def _well_by_id(wells: list[dict[str, Any]], well_id: str) -> dict[str, Any]:
    for well in wells:
        if well.get("id") == well_id:
            return well
    raise ConfigurationError(f"missing well '{well_id}'")


def load_case(path: str | Path) -> CaseConfig:
    source = Path(path).resolve()
    raw = _read_mapping(source)
    if int(raw.get("schema_version", -1)) != 1:
        raise ConfigurationError(
            "Файл проекта должен иметь schema_version = 1. "
            "Если это результат расчёта, откройте файл NPZ/HDF5 кнопкой "
            "«Открыть результаты…»."
        )

    geometry = raw["geometry"]
    geometry_type = str(geometry.get("type", "five_spot_cell"))
    if geometry_type not in {"five_spot_cell", "five_spot_point_wells"}:
        raise ConfigurationError(
            "geometry.type must be five_spot_cell or five_spot_point_wells"
        )
    inferred_formulation = (
        UNIFORM_POINT_WELL_FORMULATION
        if geometry_type == "five_spot_point_wells"
        else HYDRODYNAMIC_FORMULATION
    )
    formulation = str(raw.get("formulation", inferred_formulation))
    lx, ly = map(float, geometry["size_m"])
    wells = geometry["wells"]
    injector = _well_by_id(wells, "injector")
    producer = _well_by_id(wells, "producer")
    if injector.get("corner") != "bottom_left" or producer.get("corner") != "top_right":
        raise ConfigurationError("version 0.1 expects injector at bottom_left and producer at top_right")

    mesh_raw = raw.get("mesh", {})
    cells = mesh_raw.get("cells", {})
    pilot = mesh_raw.get("pilot_cells", {})
    geom_cfg = GeometryConfig(
        lx=lx,
        ly=ly,
        thickness=float(geometry["thickness_m"]),
        injector_radius=float(injector["radius_m"]),
        producer_radius=float(producer["radius_m"]),
    )
    default_generator = (
        "uniform_cartesian"
        if formulation == UNIFORM_POINT_WELL_FORMULATION
        else "body_fitted_structured"
    )
    generator = str(mesh_raw.get("generator", default_generator))
    # Migrate project files created by both superseded conformal generators.
    if generator in {"schwarzian_conformal", "conformal_laplace"}:
        generator = "point_source_isobars"
    mesh_cfg = MeshConfig(
        nx=int(cells.get("x", cells.get("along_streamline", 60))),
        ny=int(cells.get("y", cells.get("across_streamlines", 40))),
        pilot_nx=int(pilot.get("along_streamline", 30)),
        pilot_ny=int(pilot.get("across_streamlines", 24)),
        pressure_bias=float(mesh_raw.get("pressure_bias", 1.0)),
        max_trace_steps=int(mesh_raw.get("max_trace_steps", 12000)),
        generator=generator,
        streamline_edge_fraction=float(mesh_raw.get("streamline_edge_fraction", 0.4)),
        minimum_cell_angle_deg=float(mesh_raw.get("minimum_cell_angle_deg", 2.0)),
    )

    phases = raw["phases"]
    water_raw = phases["wetting"]
    oil_raw = phases["nonwetting"]
    water = PhaseConfig(float(water_raw["density_kg_m3"]), float(water_raw["dynamic_viscosity_pa_s"]))
    oil = PhaseConfig(float(oil_raw["density_kg_m3"]), float(oil_raw["dynamic_viscosity_pa_s"]))

    rp = raw["relative_permeability"]
    if rp.get("model") != "power_law":
        raise ConfigurationError("only power_law relative permeability is supported")
    sat = rp["saturation"]
    if sat.get("variable") != "water":
        raise ConfigurationError("the saturation variable must be water")
    relperm = RelPermConfig(
        swc=float(sat["irreducible_wetting"]),
        sor=float(sat["residual_nonwetting"]),
        krw_end=float(rp["wetting"]["endpoint"]),
        kro_end=float(rp["nonwetting"]["endpoint"]),
        nw=float(rp["wetting"]["exponent"]),
        no=float(rp["nonwetting"]["exponent"]),
    )

    flow_bc = raw["boundary_conditions"]["flow"]
    phase_bc = raw["boundary_conditions"]["phase_composition"]["well_injector"]
    boundary = BoundaryConfig(
        injector_pressure=float(flow_bc["well_injector"]["value_pa"]),
        producer_pressure=float(flow_bc["well_producer"]["value_pa"]),
        injected_water_fraction=float(phase_bc["water_fraction"]),
    )

    tr = raw["time"]
    time = TimeConfig(
        start=float(tr["start_s"]),
        end=float(tr["end_s"]),
        initial_step=float(tr["initial_step_s"]),
        target_cfl=float(tr.get("target_cfl", 0.8)),
        save_every=float(tr["save_every_s"]),
    )
    nr = raw.get("numerics", {})
    pressure = nr.get("pressure", {})
    transport = nr.get("transport", {})
    scheme_value = transport.get("scheme")
    if scheme_value is None:
        scheme_value = "explicit" if transport.get("coupling") == "impes" else "implicit"
    scheme = str(scheme_value).lower()
    numerics = NumericsConfig(
        pressure_rtol=float(pressure.get("relative_tolerance", 1.0e-9)),
        pressure_atol=float(pressure.get("absolute_tolerance", 1.0e-12)),
        pressure_max_iterations=int(pressure.get("max_iterations", 4000)),
        saturation_tolerance=float(transport.get("saturation_tolerance", 1.0e-10)),
        transport_rtol=float(transport.get("relative_tolerance", 1.0e-8)),
        transport_max_iterations=int(transport.get("max_iterations", 20)),
        max_saturation_change=float(transport.get("max_saturation_change", 1.0)),
        transport_scheme=scheme,
        transport_acceleration=str(transport.get("acceleration", "auto")).lower(),
    )
    output_raw = raw.get("output", {})
    history_interval_raw = output_raw.get("well_history_every_s")
    output = OutputConfig(
        store_field_frames=bool(output_raw.get("store_field_frames", True)),
        well_history_every=(
            None
            if history_interval_raw is None
            else float(history_interval_raw)
        ),
    )
    well_model_raw = raw.get("well_model", {})
    well_model = WellModelConfig(
        corner_correction=str(
            well_model_raw.get("corner_correction", NO_CORNER_CORRECTION)
        ).lower()
    )
    return CaseConfig(
        name=str(raw.get("project", {}).get("name", source.stem)),
        source_path=source,
        formulation=formulation,
        geometry=geom_cfg,
        mesh=mesh_cfg,
        medium=dict(raw["medium"]),
        water=water,
        oil=oil,
        relperm=relperm,
        initial_water_saturation=float(raw["initial_conditions"]["water_saturation"]),
        boundary=boundary,
        time=time,
        numerics=numerics,
        output=output,
        well_model=well_model,
    )
