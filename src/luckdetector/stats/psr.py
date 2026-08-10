"""Probabilistic Sharpe Ratio and Minimum Track Record Length.

Reference
---------
Bailey, D. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.*
Journal of Risk 15(2).

The problem
-----------
An estimated Sharpe ratio is a random variable. Quoting it without its standard
error is like quoting a poll result without a margin of error. Worse, the usual
standard error assumes normally distributed returns, and trading strategies are
routinely negatively skewed and fat-tailed — precisely the shape that makes a
Sharpe ratio *less* trustworthy than it looks.

PSR answers: **given this track record's length and its actual higher moments,
what is the probability that the true Sharpe ratio exceeds some benchmark?**

.. math::

    \\widehat{PSR}(SR^*) = \\Phi\\left[
        \\frac{(\\hat{SR} - SR^*)\\sqrt{n-1}}
             {\\sqrt{1 - \\gamma_3 \\hat{SR} + \\frac{\\gamma_4 - 1}{4}\\hat{SR}^2}}
    \\right]

where :math:`\\hat{SR}` is the **per-period** Sharpe, :math:`\\gamma_3` the skew and
:math:`\\gamma_4` the *raw* kurtosis (3 for a Gaussian).

Reading the correction term: negative skew makes ``-γ₃·ŜR`` positive, inflating
the standard error and lowering PSR. Fat tails do the same through the kurtosis
term. Both encode the same intuition — a strategy that earns steadily and then
occasionally detonates needs a longer record to prove itself.

Units
-----
Every function here takes and returns **annualised** Sharpe ratios at its public
boundary and converts internally, because "which units is this Sharpe in?" is the
single most common source of wrong answers in this literature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from scipy import stats as sps

from ..exceptions import DegenerateSeriesError, InsufficientDataError
from ..types import FloatArray, ReturnSeries
from .moments import (
    annualize_sharpe,
    as_return_series,
    deannualize_sharpe,
    kurtosis,
    sharpe_ratio,
    skewness,
)

__all__ = [
    "PSRResult",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
    "sharpe_standard_error",
    "sharpe_variance_factor",
]

GAUSSIAN_KURTOSIS = 3.0


def sharpe_variance_factor(
    sharpe_per_period: float,
    *,
    skew: float = 0.0,
    kurt: float = GAUSSIAN_KURTOSIS,
) -> float:
    """The non-normality correction :math:`1 - \\gamma_3 SR + \\frac{\\gamma_4-1}{4}SR^2`.

    For Gaussian returns this collapses to :math:`1 + SR^2/2`, recovering Lo's
    (2002) classical result. Values above 1 mean the track record is *less*
    informative than a Gaussian one of the same length.

    Raises
    ------
    DegenerateSeriesError
        If the factor is non-positive, which makes the standard error imaginary.
        This happens only for extreme moment combinations (very large positive
        skew with a large Sharpe) and signals that the asymptotic approximation
        has broken down rather than that the strategy is exceptional.
    """
    factor = 1.0 - skew * sharpe_per_period + 0.25 * (kurt - 1.0) * sharpe_per_period**2
    if factor <= 0.0:
        raise DegenerateSeriesError(
            f"Sharpe variance factor is non-positive ({factor:.4g}) for "
            f"SR={sharpe_per_period:.4g}, skew={skew:.4g}, kurtosis={kurt:.4g}. "
            "The PSR asymptotic approximation does not apply to this combination."
        )
    return float(factor)


def sharpe_standard_error(
    sharpe_per_period: float,
    n_periods: int,
    *,
    skew: float = 0.0,
    kurt: float = GAUSSIAN_KURTOSIS,
) -> float:
    """Standard error of a **per-period** Sharpe ratio estimate.

    Uses ``n - 1`` in the denominator, following Bailey & López de Prado. The
    difference from ``n`` is immaterial at any realistic track-record length.
    """
    if n_periods < 2:
        raise InsufficientDataError(f"Need at least 2 observations, got {n_periods}.")
    factor = sharpe_variance_factor(sharpe_per_period, skew=skew, kurt=kurt)
    return math.sqrt(factor / (n_periods - 1))


@dataclass(frozen=True)
class PSRResult:
    """Probabilistic Sharpe Ratio, with every input needed to reproduce it."""

    psr: float
    sharpe_annual: float
    sharpe_per_period: float
    benchmark_annual: float
    benchmark_per_period: float
    n_periods: int
    periods_per_year: int
    skewness: float
    kurtosis: float
    standard_error_per_period: float

    def __float__(self) -> float:
        return self.psr

    @property
    def standard_error_annual(self) -> float:
        """Standard error expressed in annualised Sharpe units."""
        return annualize_sharpe(self.standard_error_per_period, self.periods_per_year)

    @property
    def interpretation(self) -> str:
        return (
            f"Given {self.n_periods:,} observations with skew {self.skewness:.2f} and "
            f"kurtosis {self.kurtosis:.2f}, there is a {self.psr:.1%} probability that the "
            f"true Sharpe ratio exceeds {self.benchmark_annual:.2f}."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "psr": self.psr,
            "sharpe_annual": self.sharpe_annual,
            "benchmark_annual": self.benchmark_annual,
            "n_periods": self.n_periods,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "standard_error_annual": self.standard_error_annual,
        }


def probabilistic_sharpe_ratio(
    data: ReturnSeries | FloatArray,
    *,
    benchmark_annual_sharpe: float = 0.0,
    risk_free_rate: float = 0.0,
) -> PSRResult:
    """Probability that the true Sharpe ratio exceeds ``benchmark_annual_sharpe``.

    Parameters
    ----------
    data:
        The track record.
    benchmark_annual_sharpe:
        The hurdle, in **annualised** units. Zero asks the weakest possible
        question ("is there any edge at all?"). The Deflated Sharpe Ratio in
        :mod:`luckdetector.stats.dsr` asks the sharp version of this question by
        setting the hurdle to what the *best of N random trials* would achieve.

    Returns
    -------
    PSRResult
        ``float(result)`` gives the probability directly.

    Notes
    -----
    A PSR of 0.95 is the conventional threshold for "this track record is long
    enough to be credible" — it corresponds to a one-sided 95% confidence test.
    """
    series = as_return_series(data)
    sr_period = sharpe_ratio(series, risk_free_rate=risk_free_rate, annualized=False)
    benchmark_period = deannualize_sharpe(benchmark_annual_sharpe, series.periods_per_year)
    g3 = skewness(series)
    g4 = kurtosis(series)

    se = sharpe_standard_error(sr_period, series.n_periods, skew=g3, kurt=g4)
    psr = float(sps.norm.cdf((sr_period - benchmark_period) / se))

    return PSRResult(
        psr=psr,
        sharpe_annual=annualize_sharpe(sr_period, series.periods_per_year),
        sharpe_per_period=sr_period,
        benchmark_annual=benchmark_annual_sharpe,
        benchmark_per_period=benchmark_period,
        n_periods=series.n_periods,
        periods_per_year=series.periods_per_year,
        skewness=g3,
        kurtosis=g4,
        standard_error_per_period=se,
    )


def min_track_record_length(
    data: ReturnSeries | FloatArray,
    *,
    benchmark_annual_sharpe: float = 0.0,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
) -> float:
    """How many observations are needed before this Sharpe would be significant.

    .. math::

        n^* = 1 + \\left[1 - \\gamma_3 \\hat{SR} + \\frac{\\gamma_4-1}{4}\\hat{SR}^2\\right]
              \\left(\\frac{Z_\\alpha}{\\hat{SR} - SR^*}\\right)^2

    Returns the required number of **periods**. Divide by ``periods_per_year`` for
    years. The answer is routinely humbling: a strategy with an annualised Sharpe
    of 0.5 needs roughly a decade of daily data to clear a zero benchmark at 95%
    confidence.

    Raises
    ------
    DegenerateSeriesError
        If the observed Sharpe does not exceed the benchmark, in which case no
        amount of additional data would establish significance at the *observed*
        performance level.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence}.")

    series = as_return_series(data)
    sr_period = sharpe_ratio(series, risk_free_rate=risk_free_rate, annualized=False)
    benchmark_period = deannualize_sharpe(benchmark_annual_sharpe, series.periods_per_year)
    excess = sr_period - benchmark_period
    if excess <= 0.0:
        raise DegenerateSeriesError(
            f"Observed Sharpe ({annualize_sharpe(sr_period, series.periods_per_year):.3f} "
            f"annualised) does not exceed the benchmark ({benchmark_annual_sharpe:.3f}). "
            "No track record length makes this significant at the observed performance."
        )

    factor = sharpe_variance_factor(sr_period, skew=skewness(series), kurt=kurtosis(series))
    z = float(sps.norm.ppf(confidence))
    return 1.0 + factor * (z / excess) ** 2
