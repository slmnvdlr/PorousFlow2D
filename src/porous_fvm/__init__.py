"""PorousFlow 2D numerical core."""

from .config import CaseConfig, load_case
from .simulation import SimulationResult, run_case

__all__ = ["CaseConfig", "SimulationResult", "load_case", "run_case"]
__version__ = "0.1.0"
