"""Statistical machinery: moments, PSR/DSR, bootstrap, PBO, Reality Check, haircut."""

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
from .psr import (
    PSRResult,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_standard_error,
)

__all__ = [
    "BootstrapResult",
    "DSRResult",
    "MomentSummary",
    "PSRResult",
    "annualize_sharpe",
    "bootstrap_distribution",
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
    "permutation_null",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "sharpe_standard_error",
    "skewness",
    "summarize",
    "volatility",
]
