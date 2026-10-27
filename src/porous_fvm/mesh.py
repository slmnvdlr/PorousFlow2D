from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MeshError(ValueError):
    pass


@dataclass(frozen=True)
class MeshQuality:
    minimum_area: float
    minimum_angle_deg: float
    maximum_aspect_ratio: float
    invalid_cells: int


@dataclass
class StructuredMesh:
    nodes: np.ndarray
    thickness: float

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        if self.nodes.ndim != 3 or self.nodes.shape[2] != 2:
            raise MeshError("nodes must have shape (nx+1, ny+1, 2)")
        if min(self.nodes.shape[:2]) < 2:
            raise MeshError("mesh needs at least one cell")
        if self.thickness <= 0.0:
            raise MeshError("mesh thickness must be positive")
        self._build_metrics()

    @property
    def nx(self) -> int:
        return self.nodes.shape[0] - 1

    @property
    def ny(self) -> int:
        return self.nodes.shape[1] - 1

    @property
    def shape(self) -> tuple[int, int]:
        return self.nx, self.ny

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny

    def flatten(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values).reshape(self.cell_count)

    def unflatten(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values).reshape(self.shape)

    def _build_metrics(self) -> None:
        n = self.nodes
        self.v00 = n[:-1, :-1]
        self.v10 = n[1:, :-1]
        self.v11 = n[1:, 1:]
        self.v01 = n[:-1, 1:]
        self.cell_centers = 0.25 * (self.v00 + self.v10 + self.v11 + self.v01)
        cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        signed_area = 0.5 * (
            cross(self.v00, self.v10)
            + cross(self.v10, self.v11)
            + cross(self.v11, self.v01)
            + cross(self.v01, self.v00)
        )
        self.signed_areas = signed_area
        self.cell_areas = np.abs(signed_area)
        self.cell_volumes = self.cell_areas * self.thickness

        self.xi_face_centers = 0.5 * (n[:, :-1] + n[:, 1:])
        xi_tangent = n[:, 1:] - n[:, :-1]
        xi_length = np.linalg.norm(xi_tangent, axis=2)
        self.xi_face_areas = xi_length * self.thickness
        self.xi_face_normals = np.stack([xi_tangent[..., 1], -xi_tangent[..., 0]], axis=2) / xi_length[..., None]

        self.eta_face_centers = 0.5 * (n[:-1, :] + n[1:, :])
        eta_tangent = n[1:, :] - n[:-1, :]
        eta_length = np.linalg.norm(eta_tangent, axis=2)
        self.eta_face_areas = eta_length * self.thickness
        self.eta_face_normals = np.stack([-eta_tangent[..., 1], eta_tangent[..., 0]], axis=2) / eta_length[..., None]

    def quality(self) -> MeshQuality:
        vertices = [self.v00, self.v10, self.v11, self.v01]
        edges = [vertices[(k + 1) % 4] - vertices[k] for k in range(4)]
        lengths = np.stack([np.linalg.norm(edge, axis=2) for edge in edges])
        aspect = np.max(lengths, axis=0) / np.maximum(np.min(lengths, axis=0), 1.0e-30)
        angles = []
        for k in range(4):
            a = vertices[(k - 1) % 4] - vertices[k]
            b = vertices[(k + 1) % 4] - vertices[k]
            cosine = np.sum(a * b, axis=2) / np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1.0e-30)
            angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        min_angle = np.min(np.stack(angles), axis=0)
        invalid = int(np.count_nonzero(self.signed_areas <= 0.0))
        return MeshQuality(
            minimum_area=float(np.min(self.cell_areas)),
            minimum_angle_deg=float(np.min(min_angle)),
            maximum_aspect_ratio=float(np.max(aspect)),
            invalid_cells=invalid,
        )

    def assert_valid(self, minimum_angle_deg: float = 2.0) -> None:
        quality = self.quality()
        if quality.invalid_cells:
            raise MeshError(f"mesh contains {quality.invalid_cells} inverted cells")
        if quality.minimum_angle_deg < minimum_angle_deg:
            raise MeshError(f"minimum cell angle is {quality.minimum_angle_deg:.3g} degrees")


def rectangular_mesh(lx: float, ly: float, nx: int, ny: int, thickness: float = 1.0) -> StructuredMesh:
    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return StructuredMesh(np.stack([xx, yy], axis=2), thickness)
