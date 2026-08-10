"""Return-series moments: Sharpe, volatility, higher moments, drawdown.

This module is pedantic about one distinction: **per-period** versus
**annualised** statistics. Annualising a Sharpe ratio means multiplying by
``sqrt(periods_per_year)``; every formula in :mod:`luckdetector.stats.psr` and
:mod:`luckdetector.stats.dsr` requires the *per-period* value. Silently mixing
the two inflates significance by a factor of ~16 on daily data, which is the
single most common bug in backtest statistics code.

Convention notes
----------------
* Skewness and kurtosis use the **population** (biased) estimators by default,
  matching Bailey & López de Prado's PSR/DSR papers. Pass ``bias=False`` for the
  sample-corrected versions.
* Kurtosis is reported **raw**, not excess: a Gaussian has kurtosis 3.0.
* Volatility uses ``ddof=1`` (sample standard deviation) by default.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats as sps

from ..exceptions import DegenerateSeriesError
from ..types import FloatArray, ReturnSeries

__all__ = [
    "MomentSummary",
    "annualize_return",
    "annualize_sharpe",
    "as_return_series",
    "deannualize_sharpe",
    "excess_returns",
    "is_effectively_constant",
    "kurtosis",
    "max_drawdown",
    "mean_return",
    "sharpe_ratio",
    "skewness",
    "summarize",
    "volatility",
]

#: A standard deviation smaller than this *relative to the scale of the data* is
#: floating-point residue, not risk. ``np.std`` of 100 identical values returns
#: ~2e-19 rather than exactly 0.0, which would otherwise yield a Sharpe ratio of
#: 4.6e15 — a nonsense number that propagates silently into every downstream test.
_RELATIVE_ZERO = 1e-12


def is_effectively_constant(
    values: ReturnSeries | FloatArray,
    *,
    rtol: float = _RELATIVE_ZERO,
) -> bool:
    """Is this series constant to within floating-point noise?

    Exact equality against zero is the wrong test. Subtracting the mean from 100
    identical values leaves residue on the order of 1e-19, which is enough to make
    downstream ratios and spectral estimates produce confident nonsense. Every
    routine in this package that divides by a dispersion measure should gate on
    this first.
    """
    arr = np.asarray(
        values.values if isinstance(values, ReturnSeries) else values, dtype=np.float64
    )
    sd = float(np.std(arr))
    scale = max(float(np.mean(np.abs(arr))), float(np.finfo(np.float64).tiny))
    return not math.isfinite(sd) or sd <= rtol * scale


def as_return_series(
    data: ReturnSeries | Iterable[float] | FloatArray,
    periods_per_year: int = 252,
) -> ReturnSeries:
    """Accept either a :class:`ReturnSeries` or a raw array, return the former.

    Lets every public function take the convenient form without each one
    re-implementing validation.
    """
    if isinstance(data, ReturnSeries):
        return data
    return ReturnSeries(
        values=np.asarray(data, dtype=np.float64), periods_per_year=periods_per_year
    )


def excess_returns(series: ReturnSeries, risk_free_rate: float = 0.0) -> FloatArray:
    """Subtract a risk-free rate from returns.

    ``risk_free_rate`` is quoted **annually** and converted to per-period by
    simple division — accurate enough at the rates and frequencies involved, and
    it keeps the CLI interface intuitive ("--rf 0.04").
    """
    if risk_free_rate == 0.0:
        return series.values
    return series.values - risk_free_rate / series.periods_per_year


def mean_return(
    data: ReturnSeries | FloatArray,
    *,
    annualized: bool = False,
    geometric: bool = False,
) -> float:
    """Average periodic return, optionally annualised and/or geometric."""
    series = as_return_series(data)
    if geometric:
        growth = float(np.prod(1.0 + series.values))
        if growth <= 0.0:
            raise DegenerateSeriesError(
                "Cumulative wealth hit zero or below; geometric mean is undefined."
            )
        per_period = float(growth ** (1.0 / series.n_periods) - 1.0)
    else:
        per_period = float(np.mean(series.values))
    if not annualized:
        return per_period
    return annualize_return(per_period, series.periods_per_year, geometric=geometric)


def annualize_return(per_period: float, periods_per_year: int, *, geometric: bool = False) -> float:
    """Scale a per-period return up to annual terms."""
    if geometric:
        return (1.0 + per_period) ** periods_per_year - 1.0
    return per_period * periods_per_year


def volatility(
    data: ReturnSeries | FloatArray,
    *,
    annualized: bool = False,
    ddof: int = 1,
) -> float:
    """Standard deviation of returns."""
    series = as_return_series(data)
    sd = float(np.std(series.values, ddof=ddof))
    if not annualized:
        return sd
    return sd * math.sqrt(series.periods_per_year)


def sharpe_ratio(
    data: ReturnSeries | FloatArray,
    *,
    risk_free_rate: float = 0.0,
    annualized: bool = True,
    ddof: int = 1,
) -> float:
    """Sharpe ratio.

    Parameters
    ----------
    risk_free_rate:
        Annual rate, converted internally to per-period.
    annualized:
        ``True`` (default) multiplies by ``sqrt(periods_per_year)`` — the number
        you would quote to a human. **Pass ``False`` when feeding PSR or DSR.**

    Raises
    ------
    DegenerateSeriesError
        If the return series has zero volatility, making the ratio undefined.
    """
    series = as_return_series(data)
    excess = excess_returns(series, risk_free_rate)
    sd = float(np.std(excess, ddof=ddof))
    if is_effectively_constant(excess):
        raise DegenerateSeriesError(
            f"Cannot compute a Sharpe ratio for '{series.name}': the return series has "
            f"effectively zero volatility (sd={sd:.3g}). A constant return stream has "
            "no risk to adjust for."
        )
    sr = float(np.mean(excess)) / sd
    return annualize_sharpe(sr, series.periods_per_year) if annualized else sr


def annualize_sharpe(per_period_sharpe: float, periods_per_year: int) -> float:
    """``SR_annual = SR_period * sqrt(periods_per_year)`` (iid assumption)."""
    return per_period_sharpe * math.sqrt(periods_per_year)


def deannualize_sharpe(annual_sharpe: float, periods_per_year: int) -> float:
    """Inverse of :func:`annualize_sharpe`; use before calling PSR/DSR."""
    return annual_sharpe / math.sqrt(periods_per_year)


def skewness(data: ReturnSeries | FloatArray, *, bias: bool = True) -> float:
    """Third standardised moment.

    Negative skew — many small gains, occasional large losses — *reduces* the
    credibility of a given Sharpe ratio, which is precisely what PSR formalises.
    """
    series = as_return_series(data)
    return float(sps.skew(series.values, bias=bias))


def kurtosis(data: ReturnSeries | FloatArray, *, bias: bool = True) -> float:
    """Fourth standardised moment, **raw** (Gaussian = 3.0, not 0.0)."""
    series = as_return_series(data)
    return float(sps.kurtosis(series.values, fisher=False, bias=bias))


def max_drawdown(data: ReturnSeries | FloatArray) -> float:
    """Largest peak-to-trough decline in the wealth index, as a negative fraction."""
    series = as_return_series(data)
    wealth = series.cumulative()
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_peak - 1.0
    return float(np.min(drawdowns))


@dataclass(frozen=True)
class MomentSummary:
    """Everything the downstream tests need from a return series, computed once."""

    name: str
    n_periods: int
    years: float
    periods_per_year: int
    mean_return_annual: float
    volatility_annual: float
    sharpe_annual: float
    sharpe_per_period: float
    skewness: float
    kurtosis: float
    max_drawdown: float
    total_return: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(
    data: ReturnSeries | FloatArray,
    *,
    risk_free_rate: float = 0.0,
) -> MomentSummary:
    """Compute the full moment summary for a return series."""
    series = as_return_series(data)
    sr_period = sharpe_ratio(series, risk_free_rate=risk_free_rate, annualized=False)
    return MomentSummary(
        name=series.name,
        n_periods=series.n_periods,
        years=series.years,
        periods_per_year=series.periods_per_year,
        mean_return_annual=mean_return(series, annualized=True),
        volatility_annual=volatility(series, annualized=True),
        sharpe_annual=annualize_sharpe(sr_period, series.periods_per_year),
        sharpe_per_period=sr_period,
        skewness=skewness(series),
        kurtosis=kurtosis(series),
        max_drawdown=max_drawdown(series),
        total_return=series.total_return(),
    )
