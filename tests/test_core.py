from __future__ import annotations

import json
import unittest
import zipfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from porous_fvm.config import (
    ConfigurationError,
    MUSKAT_CORNER_CORRECTION,
    MeshConfig,
    NO_CORNER_CORRECTION,
    OutputConfig,
    UNIFORM_POINT_WELL_FORMULATION,
    default_case,
    load_case,
)
from porous_fvm.point_source_mesh import (
    build_point_source_isobar_mesh,
    coordinate_quality,
)
from porous_fvm.flow import _transmissibilities, quarter_well_productivity, solve_pressure
from porous_fvm.fields import material_properties
from porous_fvm.flow_mesh import _well_refined_arclength_levels, build_flow_aligned_mesh
from porous_fvm.geometry import (
    FiveSpotGeometry,
    symmetric_edge_levels,
    transfinite_nodes,
)
from porous_fvm.gui import Application, _cell_values_to_nodes, _contour_segments
from porous_fvm.io import create_run_output_path, load_results, save_results
from porous_fvm.mesh import StructuredMesh
from porous_fvm.relperm import PowerLawRelativePermeability
from porous_fvm.simulation import (
    muskat_quarter_five_spot_rate,
    prepare_case,
    run_case,
)
from porous_fvm.transport import (
    TransportOrderCache,
    _advance_saturation_implicit_newton,
    advance_saturation,
    advance_saturation_implicit,
    stable_time_step,
)


ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = load_case(ROOT / "examples" / "five_spot.json")
        self.geometry = FiveSpotGeometry.from_config(self.case.geometry)

    def test_well_arcs_are_centered_at_rectangle_vertices(self) -> None:
        inlet = self.geometry.injector_arc(np.linspace(0.0, 1.0, 9))
        outlet = self.geometry.producer_arc(np.linspace(0.0, 1.0, 9))
        self.assertTrue(np.allclose(np.linalg.norm(inlet, axis=1), self.geometry.injector_radius))
        self.assertTrue(
            np.allclose(
                np.linalg.norm(outlet - self.geometry.producer_center, axis=1),
                self.geometry.producer_radius,
            )
        )

    def test_pilot_mesh_has_positive_cells(self) -> None:
        mesh = StructuredMesh(transfinite_nodes(self.geometry, 14, 10), self.geometry.thickness)
        self.assertEqual(mesh.quality().invalid_cells, 0)

    def test_well_refinement_preserves_central_resolution(self) -> None:
        levels = _well_refined_arclength_levels(50, refinement=4.0, core_fraction=1.0e-3)
        central_nodes = np.count_nonzero((levels >= 0.25) & (levels <= 0.75))
        self.assertGreaterEqual(central_nodes, 15)
        self.assertLess(float(np.max(np.diff(levels))), 0.04)
        self.assertLess(float(levels[1] - levels[0]), float(levels[25] - levels[24]))

    def test_edge_fraction_clusters_transverse_lines_at_both_sides(self) -> None:
        uniform = symmetric_edge_levels(24, 0.0)
        clustered = symmetric_edge_levels(24, 1.0)
        self.assertTrue(np.all(np.diff(clustered) > 0.0))
        self.assertTrue(np.allclose(clustered, 1.0 - clustered[::-1]))
        self.assertLess(
            float(clustered[1] - clustered[0]),
            0.25 * float(uniform[1] - uniform[0]),
        )
        self.assertGreater(
            float(clustered[12] - clustered[11]),
            float(uniform[12] - uniform[11]),
        )

    def test_coordinate_quality_detects_boundary_incidence_error(self) -> None:
        xx, yy = np.meshgrid(
            np.linspace(0.0, 1.0, 9), np.linspace(0.0, 1.0, 7), indexing="ij"
        )
        orthogonal = np.stack([xx, yy], axis=-1)
        exact = coordinate_quality(orthogonal)
        self.assertLess(exact.boundary_max_deviation_deg, 1.0e-10)
        skewed = orthogonal.copy()
        skewed[..., 0] += 0.25 * yy
        distorted = coordinate_quality(skewed)
        self.assertGreater(distorted.boundary_max_deviation_deg, 10.0)


class RelativePermeabilityTests(unittest.TestCase):
    def test_power_law_endpoints_and_monotonicity(self) -> None:
        case = load_case(ROOT / "examples" / "five_spot.json")
        model = PowerLawRelativePermeability.from_config(case.relperm)
        sw = np.linspace(model.swc, 1.0 - model.sor, 101)
        krw, kro = model.evaluate(sw)
        self.assertAlmostEqual(float(krw[0]), 0.0)
        self.assertAlmostEqual(float(krw[-1]), model.krw_end)
        self.assertAlmostEqual(float(kro[0]), model.kro_end)
        self.assertAlmostEqual(float(kro[-1]), 0.0)
        self.assertTrue(np.all(np.diff(krw) >= 0.0))
        self.assertTrue(np.all(np.diff(kro) <= 0.0))


class MaterialFieldTests(unittest.TestCase):
    def test_regular_grid_npz_is_sampled_on_the_calculation_mesh(self) -> None:
        case = load_case(ROOT / "examples" / "five_spot.json")
        geometry = FiveSpotGeometry.from_config(case.geometry)
        mesh = StructuredMesh(transfinite_nodes(geometry, 8, 6), geometry.thickness)
        with TemporaryDirectory() as directory:
            np.savez(
                Path(directory) / "fields.npz",
                x_edges=np.array([0.0, 50.0, 100.0]),
                y_edges=np.array([0.0, 50.0, 100.0]),
                permeability=np.array([[1.0e-12, 2.0e-12], [3.0e-12, 4.0e-12]]),
                porosity=np.array([[0.15, 0.20], [0.25, 0.30]]),
            )
            medium = {
                "permeability": {"type": "spatial_field", "file": "fields.npz"},
                "porosity": {"type": "spatial_field", "file": "fields.npz"},
            }
            material = material_properties(medium, mesh, Path(directory))
            self.assertEqual(material.permeability.shape, mesh.shape + (2, 2))
            self.assertEqual(material.porosity.shape, mesh.shape)
            self.assertGreater(len(np.unique(material.porosity)), 1)
            self.assertTrue(np.all(material.permeability[..., 0, 0] > 0.0))


class ProjectEditingTests(unittest.TestCase):
    def test_default_case_does_not_require_an_example_file(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "new_project.json"
            case = default_case(source)
        self.assertEqual(case.source_path, source.resolve())
        self.assertEqual(case.time.end, 86400.0)
        self.assertEqual(case.mesh.generator, "point_source_isobars")

    def test_gui_final_time_units_convert_to_solver_seconds(self) -> None:
        self.assertEqual(Application._display_time_end(172800.0), (2.0, "сут"))
        self.assertEqual(Application._display_time_end(7200.0), (2.0, "ч"))
        self.assertEqual(Application._time_end_seconds("2", "сут"), 172800.0)
        self.assertEqual(Application._time_end_seconds("1.5", "ч"), 5400.0)

    def test_removed_approximate_generators_are_rejected(self) -> None:
        for name in (
            "analytic_phi_psi",
            "schwarzian_corrected",
            "schwarzian_strict",
            "schwarzian_conformal",
            "conformal_laplace",
        ):
            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                MeshConfig(generator=name)

    def test_demo_uses_point_source_isobar_generator(self) -> None:
        case = load_case(ROOT / "examples" / "five_spot_demo.json")
        self.assertEqual(case.mesh.generator, "point_source_isobars")

    def test_point_source_name_uses_analytic_builder(self) -> None:
        case = load_case(ROOT / "examples" / "five_spot.json")
        geometry = FiveSpotGeometry.from_config(case.geometry)
        analytical = replace(case.mesh, generator="point_source_isobars")
        sentinel = object()
        with patch(
            "porous_fvm.point_source_mesh.build_point_source_isobar_mesh",
            return_value=sentinel,
        ) as builder:
            result = build_flow_aligned_mesh(
                geometry, None, None, analytical, 2.0, 1.0  # type: ignore[arg-type]
            )
        self.assertIs(result, sentinel)
        builder.assert_called_once_with(geometry, analytical)

    def test_point_source_mesh_is_valid_for_small_and_large_wells(self) -> None:
        config = MeshConfig(
            nx=20,
            ny=16,
            pilot_nx=10,
            pilot_ny=8,
            generator="point_source_isobars",
        )
        for radius in (0.1, 10.0):
            with self.subTest(radius=radius):
                geometry = FiveSpotGeometry(100.0, 100.0, 10.0, radius, radius)
                mesh = build_point_source_isobar_mesh(geometry, config)
                quality = mesh.quality()
                self.assertEqual(quality.invalid_cells, 0)
                self.assertGreater(quality.minimum_angle_deg, 35.0)
                self.assertLess(mesh.boundary_incidence_max_deg, 10.0)
                diagnostics = mesh.point_source_diagnostics
                self.assertAlmostEqual(
                    diagnostics.injector_min_radius, radius, places=6
                )
                self.assertAlmostEqual(
                    diagnostics.producer_min_radius, radius, places=6
                )
                self.assertTrue(
                    np.allclose(
                        np.diff(mesh.potential_levels),
                        np.diff(mesh.potential_levels)[0],
                    )
                )

    def test_point_source_mesh_has_no_artificial_edge_grading(self) -> None:
        geometry = FiveSpotGeometry(100.0, 100.0, 10.0, 1.0, 1.0)
        base = MeshConfig(
            nx=12,
            ny=10,
            pilot_nx=8,
            pilot_ny=8,
            generator="point_source_isobars",
            streamline_edge_fraction=0.0,
        )
        first = build_point_source_isobar_mesh(geometry, base)
        second = build_point_source_isobar_mesh(
            geometry, replace(base, streamline_edge_fraction=1.0)
        )
        self.assertTrue(np.allclose(first.nodes, second.nodes))

    def test_uniform_formulation_uses_full_rectangle_and_quarter_wells(self) -> None:
        case = load_case(ROOT / "examples" / "five_spot_uniform.json")
        self.assertEqual(case.formulation, UNIFORM_POINT_WELL_FORMULATION)
        prepared = prepare_case(case)
        mesh = prepared.mesh
        flow = prepared.initial_flow
        self.assertTrue(np.allclose(mesh.nodes[:, 0, 1], 0.0))
        self.assertTrue(np.allclose(mesh.nodes[0, :, 0], 0.0))
        self.assertAlmostEqual(float(np.max(mesh.nodes[..., 0])), case.geometry.lx)
        self.assertAlmostEqual(float(np.max(mesh.nodes[..., 1])), case.geometry.ly)
        self.assertTrue(np.allclose(flow.xi_flux[0], 0.0))
        self.assertTrue(np.allclose(flow.xi_flux[-1], 0.0))
        self.assertTrue(np.allclose(flow.eta_flux[:, 0], 0.0))
        self.assertTrue(np.allclose(flow.eta_flux[:, -1], 0.0))
        self.assertGreater(flow.injector_rate, 0.0)
        self.assertGreater(flow.producer_rate, 0.0)
        self.assertAlmostEqual(
            flow.injector_rate,
            flow.producer_rate,
            delta=1.0e-7 * flow.injector_rate,
        )
        self.assertAlmostEqual(float(np.sum(flow.well_source)), 0.0, places=12)

    def test_quarter_well_index_is_one_fourth_of_reflected_full_well(self) -> None:
        dx = dy = 2.5
        permeability = np.eye(2) * 1.0e-12
        mobility = 500.0
        productivity, equivalent_radius = quarter_well_productivity(
            dx,
            dy,
            10.0,
            permeability,
            mobility,
            0.1,
        )
        expected_radius = 0.28 * np.sqrt(2.0) * dx
        expected = (
            0.5
            * np.pi
            * 10.0
            * 1.0e-12
            * mobility
            / np.log(expected_radius / 0.1)
        )
        self.assertAlmostEqual(equivalent_radius, expected_radius)
        self.assertAlmostEqual(productivity, expected)

    def test_muskat_corner_correction_adds_resistance_and_matches_target(self) -> None:
        case = load_case(ROOT / "examples" / "five_spot_uniform.json")
        case = replace(case, mesh=replace(case.mesh, nx=24, ny=24))
        corrected = prepare_case(case)
        self.assertEqual(
            corrected.corner_wells.correction_mode,
            MUSKAT_CORNER_CORRECTION,
        )
        self.assertGreater(corrected.corner_wells.injector_corner_skin, 0.0)
        self.assertGreater(corrected.corner_wells.added_total_resistance, 0.0)
        self.assertAlmostEqual(
            corrected.initial_flow.producer_rate,
            corrected.corner_wells.target_rate,
            delta=1.0e-8 * corrected.corner_wells.target_rate,
        )

        uncorrected_case = replace(
            case,
            well_model=replace(
                case.well_model,
                corner_correction=NO_CORNER_CORRECTION,
            ),
        )
        uncorrected = prepare_case(uncorrected_case)
        self.assertGreater(
            uncorrected.initial_flow.producer_rate,
            corrected.initial_flow.producer_rate,
        )

    def test_muskat_formula_reproduces_reference_quarter_rate(self) -> None:
        rate = muskat_quarter_five_spot_rate(
            side_length=100.0,
            thickness=10.0,
            permeability=1.0e-12,
            total_mobility=1.0e3,
            well_radius=0.1,
            pressure_drop=10.0e6,
        )
        self.assertAlmostEqual(rate * 86400.0, 1022.6524916782646, places=8)

    def test_gui_mapping_preserves_edited_geometry(self) -> None:
        case = load_case(ROOT / "examples" / "five_spot.json")
        edited = replace(
            case,
            name="created_in_gui",
            geometry=replace(case.geometry, lx=125.0, injector_radius=0.15),
            mesh=replace(case.mesh, generator="point_source_isobars"),
            medium={
                "permeability": {
                    "type": "constant",
                    "representation": "isotropic",
                    "value_m2": 2.5e-13,
                },
                "porosity": {"type": "constant", "value": 0.27},
            },
            water=replace(case.water, density=998.0, viscosity=8.9e-4),
            oil=replace(case.oil, density=810.0, viscosity=7.0e-3),
            relperm=replace(
                case.relperm,
                swc=0.15,
                sor=0.25,
                krw_end=0.8,
                kro_end=0.9,
                nw=2.5,
                no=3.0,
            ),
            initial_water_saturation=0.18,
            boundary=replace(
                case.boundary,
                injector_pressure=22.0e6,
                producer_pressure=9.0e6,
                injected_water_fraction=0.95,
            ),
            time=replace(case.time, end=7200.0, initial_step=120.0, save_every=240.0),
            numerics=replace(
                case.numerics,
                pressure_rtol=1.0e-8,
                pressure_max_iterations=2500,
                transport_acceleration="python",
            ),
        )
        mapping = Application._project_mapping(edited)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "edited.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            loaded = load_case(path)
        self.assertEqual(loaded.name, "created_in_gui")
        self.assertEqual(loaded.geometry.lx, 125.0)
        self.assertEqual(loaded.geometry.injector_radius, 0.15)
        self.assertEqual(loaded.mesh.generator, "point_source_isobars")
        self.assertEqual(loaded.mesh.minimum_cell_angle_deg, 2.0)
        self.assertEqual(loaded.time.end, 7200.0)
        self.assertEqual(loaded.time.initial_step, 120.0)
        self.assertEqual(loaded.time.save_every, 240.0)
        self.assertEqual(loaded.numerics.pressure_rtol, 1.0e-8)
        self.assertEqual(loaded.numerics.pressure_max_iterations, 2500)
        self.assertEqual(loaded.numerics.transport_scheme, "implicit")
        self.assertEqual(loaded.numerics.transport_acceleration, "python")
        self.assertEqual(loaded.medium["permeability"]["value_m2"], 2.5e-13)
        self.assertEqual(loaded.medium["porosity"]["value"], 0.27)
        self.assertEqual(loaded.water.density, 998.0)
        self.assertEqual(loaded.water.viscosity, 8.9e-4)
        self.assertEqual(loaded.oil.density, 810.0)
        self.assertEqual(loaded.oil.viscosity, 7.0e-3)
        self.assertEqual(loaded.relperm.swc, 0.15)
        self.assertEqual(loaded.relperm.sor, 0.25)
        self.assertEqual(loaded.relperm.krw_end, 0.8)
        self.assertEqual(loaded.relperm.kro_end, 0.9)
        self.assertEqual(loaded.relperm.nw, 2.5)
        self.assertEqual(loaded.relperm.no, 3.0)
        self.assertEqual(loaded.initial_water_saturation, 0.18)
        self.assertEqual(loaded.boundary.injector_pressure, 22.0e6)
        self.assertEqual(loaded.boundary.producer_pressure, 9.0e6)
        self.assertEqual(loaded.boundary.injected_water_fraction, 0.95)

    def test_gui_recognizes_editable_and_preserved_material_fields(self) -> None:
        case = load_case(ROOT / "examples" / "five_spot.json")
        self.assertEqual(
            Application._material_form_values(case),
            ("Однородная среда", "1e-12", "0.2"),
        )
        heterogeneous = replace(
            case,
            medium={
                "permeability": {
                    "type": "zones",
                    "default": {
                        "representation": "isotropic",
                        "value_m2": 1.0e-12,
                    },
                    "zones": [],
                },
                "porosity": {"type": "constant", "value": 0.2},
            },
        )
        self.assertEqual(
            Application._material_form_values(heterogeneous),
            ("Сохранить поля из файла", "", ""),
        )

    def test_gui_save_as_rebases_external_material_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            medium = {
                "permeability": {"type": "spatial_field", "file": "fields.npz"},
                "porosity": {"type": "spatial_field", "file": "fields.npz"},
            }
            rebased = Application._rebase_medium_files(medium, source, target)
            for key in ("permeability", "porosity"):
                resolved = (target / rebased[key]["file"]).resolve()
                self.assertEqual(resolved, (source / "fields.npz").resolve())
            self.assertEqual(medium["permeability"]["file"], "fields.npz")

    def test_gui_round_trip_preserves_uniform_point_well_formulation(self) -> None:
        case = load_case(ROOT / "examples" / "five_spot_uniform.json")
        mapping = Application._project_mapping(case)
        self.assertEqual(mapping["geometry"]["type"], "five_spot_point_wells")
        self.assertEqual(mapping["mesh"]["generator"], "uniform_cartesian")
        self.assertEqual(
            mapping["well_model"]["corner_correction"],
            MUSKAT_CORNER_CORRECTION,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "uniform.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            loaded = load_case(path)
        self.assertEqual(loaded.formulation, UNIFORM_POINT_WELL_FORMULATION)
        self.assertEqual(loaded.mesh.generator, "uniform_cartesian")
        self.assertEqual(
            loaded.well_model.corner_correction,
            MUSKAT_CORNER_CORRECTION,
        )

    def test_brown_blue_palette_endpoints(self) -> None:
        self.assertEqual(Application._brown_blue(0.0), "#914f32")
        self.assertEqual(Application._brown_blue(1.0), "#2f68b3")

    def test_gui_formats_numeric_and_choice_parameters(self) -> None:
        self.assertEqual(Application._format_input_value(0.05), "0.05")
        self.assertEqual(Application._format_input_value("Неявный"), "Неявный")

    def test_gui_recognizes_results_selected_in_project_control(self) -> None:
        self.assertTrue(Application._is_result_selection("calculation.npz"))
        self.assertTrue(Application._is_result_selection("calculation.h5"))
        self.assertTrue(Application._is_result_selection("calculation.summary.json"))
        self.assertFalse(Application._is_result_selection("five_spot.json"))

    def test_result_summary_is_redirected_to_companion_field_file(self) -> None:
        with TemporaryDirectory() as directory:
            summary = Path(directory) / "run.summary.json"
            fields = Path(directory) / "run.npz"
            summary.write_text("{}", encoding="utf-8")
            fields.touch()
            self.assertEqual(Application._resolve_result_path(summary), fields.resolve())

    def test_field_viewer_builds_contours_on_curvilinear_cells(self) -> None:
        xx, yy = np.meshgrid(
            np.linspace(0.0, 2.0, 3), np.linspace(0.0, 1.0, 3), indexing="ij"
        )
        nodes = np.stack([xx + 0.2 * yy, yy], axis=-1)
        cell_values = np.array([[0.0, 0.0], [1.0, 1.0]])
        nodal_values = _cell_values_to_nodes(cell_values)
        segments = _contour_segments(nodes, nodal_values, 0.5)
        self.assertGreater(len(segments), 0)
        for first, second in segments:
            self.assertTrue(np.all(np.isfinite(first)))
            self.assertTrue(np.all(np.isfinite(second)))


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        case = load_case(ROOT / "examples" / "five_spot.json")
        cls.case = replace(case, mesh=replace(case.mesh, nx=16, ny=12, pilot_nx=12, pilot_ny=10))
        cls.prepared = prepare_case(cls.case)

    def test_body_fitted_mesh_covers_domain_and_has_regular_cells(self) -> None:
        mesh = self.prepared.mesh
        geometry = self.prepared.geometry
        quality = mesh.quality()
        self.assertEqual(quality.invalid_cells, 0)
        self.assertGreaterEqual(quality.minimum_angle_deg, 40.0)
        self.assertLess(quality.maximum_aspect_ratio, 20.0)
        self.assertTrue(np.allclose(np.linalg.norm(mesh.nodes[0], axis=1), geometry.injector_radius))
        self.assertTrue(
            np.allclose(
                np.linalg.norm(mesh.nodes[-1] - geometry.producer_center, axis=1),
                geometry.producer_radius,
            )
        )
        self.assertAlmostEqual(float(np.min(mesh.nodes[:, :, 0])), 0.0)
        self.assertAlmostEqual(float(np.max(mesh.nodes[:, :, 0])), geometry.lx)
        self.assertAlmostEqual(float(np.min(mesh.nodes[:, :, 1])), 0.0)
        self.assertAlmostEqual(float(np.max(mesh.nodes[:, :, 1])), geometry.ly)

    def test_pressure_conserves_total_flow(self) -> None:
        flow = self.prepared.initial_flow
        self.assertLess(flow.mass_balance_error, 1.0e-5)
        self.assertGreaterEqual(float(np.min(flow.pressure)), self.case.boundary.producer_pressure)
        self.assertLessEqual(float(np.max(flow.pressure)), self.case.boundary.injector_pressure)
        self.assertAlmostEqual(
            float(np.sum(flow.xi_flux[0])),
            float(np.sum(flow.xi_flux[-1])),
            delta=1.0e-5 * abs(float(np.sum(flow.xi_flux[0]))),
        )

    def test_large_uniform_grid_uses_conservative_pyamg_pressure_solve(self) -> None:
        case = load_case(ROOT / "examples" / "five_spot_uniform.json")
        case = replace(case, mesh=replace(case.mesh, nx=65, ny=65))
        prepared = prepare_case(case)
        self.assertGreater(prepared.initial_flow.iterations, 0)
        self.assertLess(prepared.initial_flow.iterations, 50)
        self.assertLess(prepared.initial_flow.mass_balance_error, 1.0e-8)
        self.assertTrue(np.all(np.isfinite(prepared.initial_flow.pressure)))

    def test_cached_transmissibilities_match_reference_formula(self) -> None:
        saturation = np.full(self.prepared.mesh.shape, self.case.initial_water_saturation)
        saturation[:4] = min(0.6, 1.0 - self.case.relperm.sor)
        _, _, mobility = self.prepared.relperm.mobilities(
            saturation, self.case.water, self.case.oil
        )
        cached = self.prepared.pressure_workspace.transmissibilities(mobility)
        reference = _transmissibilities(
            self.prepared.mesh,
            self.prepared.material,
            mobility,
            pressure_xi_boundaries=self.prepared.corner_wells is None,
        )
        self.assertTrue(np.allclose(cached[0], reference[0], rtol=2.0e-15, atol=0.0))
        self.assertTrue(np.allclose(cached[1], reference[1], rtol=2.0e-15, atol=0.0))

    def test_loose_requested_pressure_tolerance_is_refined_for_transport(self) -> None:
        saturation = np.full(self.prepared.mesh.shape, self.case.initial_water_saturation)
        saturation[:3] = min(0.65, 1.0 - self.case.relperm.sor)
        _, _, mobility = self.prepared.relperm.mobilities(
            saturation, self.case.water, self.case.oil
        )
        loose = replace(
            self.case.numerics,
            pressure_rtol=1.0e-5,
            pressure_atol=1.0e-15,
        )
        flow = solve_pressure(
            self.prepared.mesh,
            self.prepared.material,
            mobility,
            self.case.boundary,
            loose,
            initial_pressure=self.prepared.initial_flow.pressure,
            reconstruct_velocity=False,
        )
        self.assertLessEqual(flow.mass_balance_error, 1.0e-8)

    def test_transport_step_is_conservative_and_bounded(self) -> None:
        saturation = np.full(self.prepared.mesh.shape, self.case.initial_water_saturation)
        _, _, mobility = self.prepared.relperm.mobilities(saturation, self.case.water, self.case.oil)
        flow = solve_pressure(
            self.prepared.mesh,
            self.prepared.material,
            mobility,
            self.case.boundary,
            self.case.numerics,
            reconstruct_velocity=False,
        )
        derivative = self.prepared.relperm.max_fractional_flow_derivative(self.case.water, self.case.oil)
        dt = 0.5 * stable_time_step(
            self.prepared.mesh,
            self.prepared.material,
            flow,
            derivative,
            self.case.time.target_cfl,
        )
        step = advance_saturation(
            self.prepared.mesh,
            self.prepared.material,
            saturation,
            flow,
            self.prepared.relperm,
            self.case.water,
            self.case.oil,
            self.case.boundary,
            self.case.numerics,
            dt,
        )
        self.assertLess(step.balance_error, 1.0e-12)
        self.assertGreaterEqual(float(np.min(step.saturation)), self.case.relperm.swc)
        self.assertLessEqual(float(np.max(step.saturation)), 1.0 - self.case.relperm.sor)

    def test_ordered_implicit_sweep_matches_newton_reference(self) -> None:
        saturation = np.full(self.prepared.mesh.shape, self.case.initial_water_saturation)
        ordered = advance_saturation_implicit(
            self.prepared.mesh,
            self.prepared.material,
            saturation,
            self.prepared.initial_flow,
            self.prepared.relperm,
            self.case.water,
            self.case.oil,
            self.case.boundary,
            self.case.numerics,
            30.0,
        )
        newton = _advance_saturation_implicit_newton(
            self.prepared.mesh,
            self.prepared.material,
            saturation,
            self.prepared.initial_flow,
            self.prepared.relperm,
            self.case.water,
            self.case.oil,
            self.case.boundary,
            self.case.numerics,
            30.0,
        )
        self.assertTrue(np.allclose(ordered.saturation, newton.saturation, atol=1.0e-11))
        self.assertLess(ordered.balance_error, 1.0e-12)

    def test_transport_order_is_reused_only_after_validation(self) -> None:
        cache = TransportOrderCache()
        first = cache.get(self.prepared.initial_flow)
        second = cache.get(self.prepared.initial_flow)
        self.assertIs(first, second)
        self.assertEqual(cache.rebuilds, 1)
        self.assertEqual(cache.hits, 1)

    def test_numba_and_python_ordered_sweeps_agree(self) -> None:
        saturation = np.full(self.prepared.mesh.shape, self.case.initial_water_saturation)
        arguments = (
            self.prepared.mesh,
            self.prepared.material,
            saturation,
            self.prepared.initial_flow,
            self.prepared.relperm,
            self.case.water,
            self.case.oil,
            self.case.boundary,
        )
        python_step = advance_saturation_implicit(
            *arguments,
            replace(self.case.numerics, transport_acceleration="python"),
            30.0,
        )
        numba_step = advance_saturation_implicit(
            *arguments,
            replace(self.case.numerics, transport_acceleration="numba"),
            30.0,
        )
        self.assertTrue(
            np.allclose(numba_step.saturation, python_step.saturation, rtol=0.0, atol=2.0e-14)
        )
        self.assertAlmostEqual(numba_step.balance_error, python_step.balance_error, places=14)

    def test_complete_short_implicit_run(self) -> None:
        short_case = replace(
            self.case,
            time=replace(self.case.time, end=30.0, initial_step=10.0, save_every=15.0),
        )
        result = run_case(short_case)
        self.assertTrue(result.completed)
        self.assertAlmostEqual(float(result.times[-1]), 30.0)
        self.assertLess(float(np.max(result.mass_balance_error)), 1.0e-12)
        self.assertGreater(float(np.max(result.final_saturation)), self.case.relperm.swc)
        self.assertAlmostEqual(float(result.diagnostic_times[-1]), 30.0)
        self.assertEqual(result.diagnostic_times.shape, result.producer_liquid_rate.shape)
        self.assertEqual(result.diagnostic_times.shape, result.producer_water_cut.shape)
        self.assertTrue(np.all(result.producer_liquid_rate > 0.0))
        self.assertTrue(np.all((result.producer_water_cut >= 0.0) & (result.producer_water_cut <= 1.0)))

        with TemporaryDirectory() as directory:
            source_project = Path(directory) / "source.json"
            source_project.write_text('{"schema_version": 1}', encoding="utf-8")
            path = save_results(
                result,
                Path(directory) / "run" / "roundtrip.npz",
                source_project=source_project,
            )
            workbook_path = path.with_suffix(".wells.xlsx")
            self.assertTrue(workbook_path.exists())
            self.assertTrue(path.with_suffix(".summary.json").exists())
            self.assertEqual(
                (path.parent / "input.json").read_text(encoding="utf-8"),
                '{"schema_version": 1}',
            )
            with zipfile.ZipFile(workbook_path) as workbook:
                self.assertIn("xl/charts/chart1.xml", workbook.namelist())
                self.assertIn("xl/charts/chart2.xml", workbook.namelist())
                workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")
                self.assertIn("Скважины", workbook_xml)
                self.assertIn("Диагностика", workbook_xml)
            restored = load_results(path)
        self.assertTrue(np.allclose(restored.times, result.times))
        self.assertTrue(np.allclose(restored.pressure, result.pressure))
        self.assertTrue(np.allclose(restored.saturation, result.saturation))
        self.assertTrue(np.allclose(restored.producer_liquid_rate, result.producer_liquid_rate))
        self.assertTrue(np.allclose(restored.producer_water_cut, result.producer_water_cut))
        self.assertTrue(restored.completed)
        self.assertEqual(restored.transport_scheme, "implicit")

    def test_well_history_interval_is_independent_of_field_frames(self) -> None:
        sampled_case = replace(
            self.case,
            time=replace(self.case.time, end=30.0, initial_step=10.0, save_every=30.0),
            output=OutputConfig(store_field_frames=False, well_history_every=10.0),
        )
        result = run_case(sampled_case)

        self.assertTrue(result.completed)
        self.assertTrue(np.allclose(result.diagnostic_times, [0.0, 10.0, 20.0, 30.0]))
        self.assertTrue(np.allclose(result.times, [30.0]))
        self.assertEqual(result.saturation.shape[0], 1)
        self.assertEqual(result.pressure.shape[0], 1)
        self.assertEqual(result.diagnostic_times.shape, result.producer_liquid_rate.shape)
        self.assertEqual(result.diagnostic_times.shape, result.producer_water_cut.shape)

    def test_each_run_gets_a_unique_output_directory(self) -> None:
        short_case = replace(
            self.case,
            time=replace(self.case.time, end=10.0, initial_step=10.0, save_every=10.0),
        )
        result = run_case(short_case)
        with TemporaryDirectory() as directory:
            project = Path(directory) / "case.json"
            project.write_text("{}", encoding="utf-8")
            timestamp = datetime(2026, 10, 2, 12, 30, 45)
            first = create_run_output_path(project, result, timestamp)
            second = create_run_output_path(project, result, timestamp)
            self.assertNotEqual(first.parent, second.parent)
            self.assertEqual(first.name, "result.npz")
            self.assertEqual(first.parent.parent, Path(directory) / "results")

    def test_explicit_transport_remains_available(self) -> None:
        explicit_case = replace(
            self.case,
            time=replace(self.case.time, end=0.05, initial_step=0.05, save_every=0.05),
            numerics=replace(self.case.numerics, transport_scheme="explicit"),
        )
        result = run_case(explicit_case)
        self.assertTrue(result.completed)
        self.assertEqual(result.transport_scheme, "explicit")
        self.assertAlmostEqual(float(result.times[-1]), explicit_case.time.end)
        self.assertLess(float(np.max(result.mass_balance_error)), 1.0e-12)
        self.assertGreater(float(np.max(result.final_saturation)), explicit_case.relperm.swc)

    def test_stopped_run_returns_and_saves_viewable_partial_result(self) -> None:
        interrupted_case = replace(
            self.case,
            time=replace(self.case.time, end=300.0, initial_step=10.0, save_every=50.0),
        )
        stop = {"requested": False}

        def request_stop_after_first_step(_fraction: float, message: str) -> None:
            if message.startswith("time="):
                stop["requested"] = True

        result = run_case(
            interrupted_case,
            progress=request_stop_after_first_step,
            should_stop=lambda: stop["requested"],
        )
        self.assertFalse(result.completed)
        self.assertGreater(float(result.times[-1]), interrupted_case.time.start)
        self.assertLess(float(result.times[-1]), interrupted_case.time.end)
        self.assertEqual(result.saturation.shape[0], len(result.times))
        self.assertEqual(result.pressure.shape[0], len(result.times))
        self.assertGreaterEqual(len(result.diagnostic_times), 1)

        with TemporaryDirectory() as directory:
            path = save_results(result, Path(directory) / "partial.npz")
            restored = load_results(path)
            summary = json.loads(path.with_suffix(".summary.json").read_text(encoding="utf-8"))
        self.assertFalse(restored.completed)
        self.assertEqual(summary["status"], "stopped")
        self.assertTrue(np.allclose(restored.saturation, result.saturation))


if __name__ == "__main__":
    unittest.main()
