"""Statistical machinery: moments, PSR/DSR, bootstrap, PBO, Reality Check and SPA."""

from __future__ import annotations

from .bootstrap import (
    BootstrapResult,
    bootstrap_distribution,
    optimal_block_length,
    permutation_null,
)
from .dsr import (
    DSRResult,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_trials,
    effective_number_of_trials,
    expected_max_sharpe,
    null_sharpe_std,
    sharpe_required_for_dsr,
)
from .moments import (
    MomentSummary,
    annualize_sharpe,
    deannualize_sharpe,
    is_effectively_constant,
    kurtosis,
    max_drawdown,
    mean_return,
    sharpe_ratio,
    skewness,
    summarize,
    volatility,
)
from .pbo import (
    DegradationResult,
    PBOResult,
    contiguous_blocks,
    performance_degradation,
    probability_of_backtest_overfitting,
)
from .psr import (
    PSRResult,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_standard_error,
)
from .reality_check import (
    RealityCheckResult,
    reality_check,
)

__all__ = [
    "BootstrapResult",
    "DSRResult",
    "DegradationResult",
    "MomentSummary",
    "PBOResult",
    "PSRResult",
    "RealityCheckResult",
    "annualize_sharpe",
    "bootstrap_distribution",
    "contiguous_blocks",
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_trials",
    "effective_number_of_trials",
    "expected_max_sharpe",
    "is_effectively_constant",
    "kurtosis",
    "max_drawdown",
    "mean_return",
    "min_track_record_length",
    "null_sharpe_std",
    "optimal_block_length",
    "performance_degradation",
    "permutation_null",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "reality_check",
    "sharpe_ratio",
    "sharpe_required_for_dsr",
    "sharpe_standard_error",
    "skewness",
    "summarize",
    "volatility",
]
