"""Reusable lottery bias analysis package extracted from omegaXiv validation contributions."""

from .analysis import AnalyzerConfig, LotteryBiasAnalyzer
from .data import (
    SplitData,
    assign_regime_ids,
    feasible_changepoints,
    load_lottery_draws,
    prepare_lottery_dataframe,
    sample_iid_draws,
    temporal_split,
)
from .sympy_checks import compute_sympy_invariants

__all__ = [
    "LotteryBiasAnalyzer",
    "AnalyzerConfig",
    "SplitData",
    "assign_regime_ids",
    "feasible_changepoints",
    "load_lottery_draws",
    "prepare_lottery_dataframe",
    "sample_iid_draws",
    "temporal_split",
    "compute_sympy_invariants",
]

__version__ = "0.1.0"
