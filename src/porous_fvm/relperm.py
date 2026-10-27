from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PhaseConfig, RelPermConfig


@dataclass(frozen=True)
class PowerLawRelativePermeability:
    swc: float
    sor: float
    krw_end: float
    kro_end: float
    nw: float
    no: float

    @classmethod
    def from_config(cls, config: RelPermConfig) -> "PowerLawRelativePermeability":
        return cls(config.swc, config.sor, config.krw_end, config.kro_end, config.nw, config.no)

    @property
    def mobile_interval(self) -> float:
        return 1.0 - self.swc - self.sor

    def effective_saturation(self, sw: np.ndarray | float) -> np.ndarray:
        return np.clip((np.asarray(sw, dtype=float) - self.swc) / self.mobile_interval, 0.0, 1.0)

    def evaluate(self, sw: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        se = self.effective_saturation(sw)
        return self.krw_end * se**self.nw, self.kro_end * (1.0 - se) ** self.no

    def mobilities(
        self,
        sw: np.ndarray | float,
        water: PhaseConfig,
        oil: PhaseConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        krw, kro = self.evaluate(sw)
        lw = krw / water.viscosity
        lo = kro / oil.viscosity
        return lw, lo, lw + lo

    def fractional_flow(self, sw: np.ndarray | float, water: PhaseConfig, oil: PhaseConfig) -> np.ndarray:
        lw, _, lt = self.mobilities(sw, water, oil)
        return np.divide(lw, lt, out=np.zeros_like(lw, dtype=float), where=lt > 0.0)

    def max_fractional_flow_derivative(self, water: PhaseConfig, oil: PhaseConfig) -> float:
        sw = np.linspace(self.swc, 1.0 - self.sor, 4001)
        fw = self.fractional_flow(sw, water, oil)
        derivative = np.gradient(fw, sw)
        return max(float(np.max(np.abs(derivative))), 1.0)
