from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConfigurationError
from .mesh import StructuredMesh


@dataclass(frozen=True)
class MaterialProperties:
    permeability: np.ndarray
    porosity: np.ndarray

    def validate(self, shape: tuple[int, int]) -> None:
        if self.permeability.shape != shape + (2, 2):
            raise ConfigurationError("permeability field has an invalid shape")
        if self.porosity.shape != shape:
            raise ConfigurationError("porosity field has an invalid shape")
        if not np.all(np.isfinite(self.permeability)) or not np.all(np.isfinite(self.porosity)):
            raise ConfigurationError("material fields contain non-finite values")
        if np.any(self.porosity <= 0.0) or np.any(self.porosity > 1.0):
            raise ConfigurationError("porosity must be in (0, 1]")
        if np.any(np.linalg.eigvalsh(self.permeability) <= 0.0):
            raise ConfigurationError("permeability tensors must be positive definite")


def _constant_tensor(spec: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    representation = spec.get("representation", "isotropic")
    if representation == "isotropic":
        value = float(spec.get("value_m2", spec.get("value", 0.0)))
        base = np.array([[value, 0.0], [0.0, value]])
    elif representation == "diagonal":
        base = np.array([[float(spec["kx_m2"]), 0.0], [0.0, float(spec["ky_m2"])]])
    elif representation == "symmetric_tensor_2d":
        base = np.array(
            [[float(spec["kxx_m2"]), float(spec["kxy_m2"])],
             [float(spec["kxy_m2"]), float(spec["kyy_m2"])]],
        )
    else:
        raise ConfigurationError(f"unknown permeability representation '{representation}'")
    return np.broadcast_to(base, shape + (2, 2)).copy()


def _apply_rectangular_zones(result, centers, zones, value_builder) -> None:
    for zone in zones:
        bounds = zone.get("bounds_m")
        if not bounds or len(bounds) != 4:
            raise ConfigurationError("version 0.1 zones require bounds_m=[xmin,ymin,xmax,ymax]")
        xmin, ymin, xmax, ymax = map(float, bounds)
        mask = (
            (centers[..., 0] >= xmin)
            & (centers[..., 0] <= xmax)
            & (centers[..., 1] >= ymin)
            & (centers[..., 1] <= ymax)
        )
        result[mask] = value_builder(zone)


def _sample_regular_grid(values, x_edges, y_edges, points):
    ix = np.searchsorted(x_edges, points[..., 0], side="right") - 1
    iy = np.searchsorted(y_edges, points[..., 1], side="right") - 1
    if np.any(ix < 0) or np.any(ix >= len(x_edges) - 1) or np.any(iy < 0) or np.any(iy >= len(y_edges) - 1):
        raise ConfigurationError("the spatial material field does not cover the calculation domain")
    return values[iy, ix]


def _from_npz(spec, centers, source_dir: Path, key: str):
    path = (source_dir / spec["file"]).resolve()
    with np.load(path) as data:
        x_edges = np.asarray(data[spec.get("x_edges", "x_edges")], dtype=float)
        y_edges = np.asarray(data[spec.get("y_edges", "y_edges")], dtype=float)
        values = np.asarray(data[spec.get("dataset", key)], dtype=float)
    return _sample_regular_grid(values, x_edges, y_edges, centers)


def material_properties(medium: dict[str, Any], mesh: StructuredMesh, source_dir: Path) -> MaterialProperties:
    shape = mesh.shape
    centers = mesh.cell_centers
    p_spec = medium.get("permeability")
    phi_spec = medium.get("porosity")
    if not isinstance(p_spec, dict) or not isinstance(phi_spec, dict):
        raise ConfigurationError("medium must define permeability and porosity objects")

    p_type = p_spec.get("type", "constant")
    if p_type == "constant":
        permeability = _constant_tensor(p_spec, shape)
    elif p_type == "zones":
        permeability = _constant_tensor(dict(p_spec["default"]), shape)
        _apply_rectangular_zones(
            permeability,
            centers,
            p_spec.get("zones", []),
            lambda zone: _constant_tensor(zone, (1, 1))[0, 0],
        )
    elif p_type == "spatial_field":
        raw = _from_npz(p_spec, centers, source_dir, "permeability")
        permeability = np.zeros(shape + (2, 2), dtype=float)
        if raw.shape == shape:
            permeability[..., 0, 0] = raw
            permeability[..., 1, 1] = raw
        elif raw.shape == shape + (3,):
            permeability[..., 0, 0] = raw[..., 0]
            permeability[..., 0, 1] = permeability[..., 1, 0] = raw[..., 1]
            permeability[..., 1, 1] = raw[..., 2]
        else:
            raise ConfigurationError("spatial permeability must be scalar or [...,3]")
    else:
        raise ConfigurationError(f"unsupported permeability field type '{p_type}'")

    phi_type = phi_spec.get("type", "constant")
    if phi_type == "constant":
        porosity = np.full(shape, float(phi_spec["value"]))
    elif phi_type == "zones":
        porosity = np.full(shape, float(phi_spec["default"]))
        _apply_rectangular_zones(porosity, centers, phi_spec.get("zones", []), lambda zone: float(zone["value"]))
    elif phi_type == "spatial_field":
        porosity = _from_npz(phi_spec, centers, source_dir, "porosity")
    else:
        raise ConfigurationError(f"unsupported porosity field type '{phi_type}'")

    result = MaterialProperties(permeability, porosity)
    result.validate(shape)
    return result
