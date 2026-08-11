"""luckdetector — how much of your backtest is luck?

Quick start
-----------
>>> import numpy as np
>>> from luckdetector import ReturnSeries, summarize
>>> rng = np.random.default_rng(0)
>>> series = ReturnSeries(rng.normal(0.0005, 0.01, 1260))
>>> round(summarize(series).sharpe_annual, 2)
0.22

Note what just happened: the *true* annualised Sharpe of that process is
``0.0005 / 0.01 * sqrt(252) = 0.79``, but five years of data reported 0.22. The
estimate is that noisy. Everything in this package exists because of that gap.
"""

from __future__ import annotations

from .exceptions import (
    DataValidationError,
    DegenerateSeriesError,
    InsufficientDataError,
    LuckDetectorError,
)
from .io import load_returns_csv, load_trials_csv, returns_from_prices
from .report import assess
from .stats import (
    MomentSummary,
    annualize_sharpe,
    deannualize_sharpe,
    kurtosis,
    max_drawdown,
    mean_return,
    sharpe_ratio,
    skewness,
    summarize,
    volatility,
)
from .types import ReturnSeries, TestResult, TrialMatrix, Verdict

__version__ = "0.1.0"

__all__ = [
    "DataValidationError",
    "DegenerateSeriesError",
    "InsufficientDataError",
    "LuckDetectorError",
    "MomentSummary",
    "ReturnSeries",
    "TestResult",
    "TrialMatrix",
    "Verdict",
    "__version__",
    "annualize_sharpe",
    "assess",
    "deannualize_sharpe",
    "kurtosis",
    "load_returns_csv",
    "load_trials_csv",
    "max_drawdown",
    "mean_return",
    "returns_from_prices",
    "sharpe_ratio",
    "skewness",
    "summarize",
    "volatility",
]
