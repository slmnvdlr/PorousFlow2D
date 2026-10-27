from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeometryConfig


@dataclass(frozen=True)
class FiveSpotGeometry:
    """A symmetry cell with quarter-well cut-outs at opposite vertices."""

    lx: float
    ly: float
    thickness: float
    injector_radius: float
    producer_radius: float

    @classmethod
    def from_config(cls, config: GeometryConfig) -> "FiveSpotGeometry":
        return cls(
            config.lx,
            config.ly,
            config.thickness,
            config.injector_radius,
            config.producer_radius,
        )

    @property
    def injector_center(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    @property
    def producer_center(self) -> np.ndarray:
        return np.array([self.lx, self.ly])

    def injector_arc(self, eta: np.ndarray | float) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        theta = 0.5 * np.pi * eta
        return np.stack(
            [self.injector_radius * np.cos(theta), self.injector_radius * np.sin(theta)],
            axis=-1,
        )

    def producer_arc(self, eta: np.ndarray | float) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        theta = -0.5 * np.pi * (1.0 + eta)
        return np.stack(
            [self.lx + self.producer_radius * np.cos(theta),
             self.ly + self.producer_radius * np.sin(theta)],
            axis=-1,
        )

    @staticmethod
    def _polyline(points: np.ndarray, t: np.ndarray | float) -> np.ndarray:
        values = np.asarray(t, dtype=float)
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        target = np.clip(values, 0.0, 1.0) * cumulative[-1]
        flat = target.ravel()
        result = np.empty((flat.size, 2), dtype=float)
        for n, distance in enumerate(flat):
            index = min(np.searchsorted(cumulative, distance, side="right") - 1, len(lengths) - 1)
            index = max(index, 0)
            local = (distance - cumulative[index]) / lengths[index]
            result[n] = points[index] + local * segments[index]
        return result.reshape(values.shape + (2,))

    def boundary_a(self, xi: np.ndarray | float) -> np.ndarray:
        """Bottom then right no-flow path, from injector to producer."""
        return self._polyline(
            np.array([
                [self.injector_radius, 0.0],
                [self.lx, 0.0],
                [self.lx, self.ly - self.producer_radius],
            ]),
            xi,
        )

    def boundary_b(self, xi: np.ndarray | float) -> np.ndarray:
        """Left then top no-flow path, from injector to producer."""
        return self._polyline(
            np.array([
                [0.0, self.injector_radius],
                [0.0, self.ly],
                [self.lx - self.producer_radius, self.ly],
            ]),
            xi,
        )

    def contains(self, point: np.ndarray, tolerance: float = 0.0) -> bool:
        x, y = np.asarray(point, dtype=float)
        if x < -tolerance or x > self.lx + tolerance or y < -tolerance or y > self.ly + tolerance:
            return False
        if np.linalg.norm(np.array([x, y]) - self.injector_center) < self.injector_radius - tolerance:
            return False
        if np.linalg.norm(np.array([x, y]) - self.producer_center) < self.producer_radius - tolerance:
            return False
        return True


def symmetric_edge_levels(count: int, edge_fraction: float) -> np.ndarray:
    """Symmetric transverse levels, clustered towards both outer paths.

    ``edge_fraction=0`` is uniform.  Increasing it to one progressively
    allocates more streamline channels next to the two no-flow boundaries
    without changing their total number.
    """
    coordinate = np.linspace(0.0, 1.0, count + 1)
    fraction = float(np.clip(edge_fraction, 0.0, 1.0))
    if fraction <= 1.0e-12:
        return coordinate
    strength = 2.5 * fraction
    return 0.5 * (
        1.0
        + np.tanh(strength * (2.0 * coordinate - 1.0))
        / np.tanh(strength)
    )


def transfinite_nodes(
    geometry: FiveSpotGeometry,
    nx: int,
    ny: int,
    xi_levels: np.ndarray | None = None,
    eta_levels: np.ndarray | None = None,
) -> np.ndarray:
    """Build a body-fitted quadrilateral grid by transfinite interpolation."""
    xi = np.linspace(0.0, 1.0, nx + 1) if xi_levels is None else np.asarray(xi_levels, dtype=float)
    eta = np.linspace(0.0, 1.0, ny + 1) if eta_levels is None else np.asarray(eta_levels, dtype=float)
    if xi.shape != (nx + 1,) or eta.shape != (ny + 1,):
        raise ValueError("grid-coordinate arrays must match the requested cell counts")
    if xi[0] != 0.0 or xi[-1] != 1.0 or eta[0] != 0.0 or eta[-1] != 1.0:
        raise ValueError("grid-coordinate arrays must start at 0 and end at 1")
    if np.any(np.diff(xi) <= 0.0) or np.any(np.diff(eta) <= 0.0):
        raise ValueError("grid-coordinate arrays must be strictly increasing")
    c0 = geometry.injector_arc(eta)
    c1 = geometry.producer_arc(eta)
    d0 = geometry.boundary_a(xi)
    d1 = geometry.boundary_b(xi)
    p00, p01 = c0[0], c0[-1]
    p10, p11 = c1[0], c1[-1]
    nodes = np.empty((nx + 1, ny + 1, 2), dtype=float)
    for i, a in enumerate(xi):
        for j, b in enumerate(eta):
            blend_edges = (1.0 - a) * c0[j] + a * c1[j] + (1.0 - b) * d0[i] + b * d1[i]
            blend_corners = (
                (1.0 - a) * (1.0 - b) * p00
                + (1.0 - a) * b * p01
                + a * (1.0 - b) * p10
                + a * b * p11
            )
            nodes[i, j] = blend_edges - blend_corners
    return nodes
