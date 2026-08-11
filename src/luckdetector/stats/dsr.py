"""Deflated Sharpe Ratio: the Sharpe you'd expect from the best of N random trials.

Reference
---------
Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality.* Journal of Portfolio
Management 40(5).

The idea
--------
PSR asks whether a track record beats a benchmark. DSR asks the only benchmark
that matters after a search: **what would the winner of N coin-flipping contests
have scored?** If you tried 200 strategy variants and kept the best, the honest
hurdle is not zero — it is the expected maximum of 200 zero-edge Sharpe ratios.

The expected maximum of N iid Sharpe estimates follows from extreme value theory
(the Gumbel limit):

.. math::

    E[\\max_N \\hat{SR}] \\approx \\sigma_{SR}\\left[(1-\\gamma)
        \\Phi^{-1}\\!\\left(1 - \\tfrac{1}{N}\\right)
        + \\gamma\\,\\Phi^{-1}\\!\\left(1 - \\tfrac{1}{N e}\\right)\\right]

with :math:`\\gamma` the Euler–Mascheroni constant and :math:`\\sigma_{SR}` the
cross-sectional standard deviation of the trial Sharpe ratios.

DSR is then simply ``PSR`` evaluated against that hurdle.

Why ``N`` must be the *effective* number of trials
--------------------------------------------------
Two hundred variants of one moving-average rule are not two hundred independent
bets; they are closer to a handful. Counting them as 200 sets the hurdle far too
high and would clear genuinely good strategies as "not proven". :func:`effective_number_of_trials`
discounts the count by the correlation structure of the trials themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats as sps
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from ..exceptions import InsufficientDataError
from ..types import FloatArray, ReturnSeries, TrialMatrix
from .moments import annualize_sharpe, as_return_series, deannualize_sharpe, sharpe_ratio
from .psr import PSRResult, probabilistic_sharpe_ratio

__all__ = [
    "DSR_THRESHOLD",
    "EULER_MASCHERONI",
    "DSRResult",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_trials",
    "effective_number_of_trials",
    "expected_max_sharpe",
    "null_sharpe_std",
    "sharpe_required_for_dsr",
]

#: Euler–Mascheroni constant, from the Gumbel limit of the maximum.
EULER_MASCHERONI = 0.5772156649015329

#: The bar :attr:`DSRResult.passed` applies. Conventional — DSR is a probability
#: and 0.95 is the usual one-sided level — but named so it can be argued with, and
#: because the number it gates is far more demanding than the same figure applied
#: to a naive PSR. On SPY: naive PSR 0.9764 clears it, DSR 0.7692 does not.
DSR_THRESHOLD = 0.95

TrialCountMethod = Literal["independent", "equicorrelated", "cluster"]


def expected_max_sharpe(n_trials: float, sharpe_std: float) -> float:
    """Expected maximum of ``n_trials`` iid zero-mean Sharpe estimates.

    Parameters
    ----------
    n_trials:
        Number of (effective, independent) trials. May be fractional, since
        :func:`effective_number_of_trials` returns a discounted count.
    sharpe_std:
        Cross-sectional standard deviation of the trial Sharpe ratios, in the
        **same units** as the value you intend to compare against.

    Notes
    -----
    Grows like ``sqrt(2 ln N)`` — slowly, but relentlessly. Doubling the number of
    trials you ran raises the hurdle much less than most people fear, while going
    from 1 trial to 200 raises it enormously.
    """
    if n_trials < 1:
        raise InsufficientDataError(f"n_trials must be at least 1, got {n_trials}.")
    if sharpe_std < 0:
        raise ValueError(f"sharpe_std must be non-negative, got {sharpe_std}.")
    if n_trials == 1:
        return 0.0

    q1 = float(sps.norm.ppf(1.0 - 1.0 / n_trials))
    q2 = float(sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return sharpe_std * ((1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2)


def null_sharpe_std(n_periods: int) -> float:
    """Per-period standard deviation of a Sharpe estimate under the zero-edge null.

    Under :math:`H_0: SR = 0` with Gaussian returns the variance factor is 1, so
    the standard error is simply :math:`1/\\sqrt{n-1}`. Used as the default
    dispersion when the caller knows how many strategies were tried but did not
    keep their individual return streams.
    """
    if n_periods < 2:
        raise InsufficientDataError(f"Need at least 2 observations, got {n_periods}.")
    return 1.0 / math.sqrt(n_periods - 1)


def sharpe_required_for_dsr(result: DSRResult, *, confidence: float = DSR_THRESHOLD) -> float:
    """The annualised Sharpe this record needed in order to clear ``confidence``.

    The inverse of the Deflated Sharpe Ratio, and the number the report layer
    actually draws. It exists because the obvious thing to plot — the expected
    maximum of noise — is **not** the bar DSR applies, and plotting it alone
    argues the opposite of the test's own conclusion.

    On SPY the expected maximum of noise is 0.31 and the winner posted 0.49, so a
    figure marking only that hurdle shows the winner comfortably clearing it while
    the DSR of 0.769 calls it luck. The reconciliation is that the hurdle is a
    *point*, and the winner's Sharpe is an *estimate* with a standard error of
    0.247: being 0.18 above the hurdle is only 0.74 standard errors above it, and
    0.95 confidence needs 1.645. So the Sharpe actually required is

    .. math::

        SR_{req} = \\mathbb{E}[\\max SR_{noise}] + \\Phi^{-1}(c)\\,\\hat\\sigma(\\widehat{SR})

    which on SPY is **0.715** — a bar the winner misses by 0.22, and one that no
    variant in the family reaches.

    The standard error is a property of the record alone, not of the benchmark it
    is compared against, so this inversion is exact rather than an approximation:
    substituting the result back into :func:`deflated_sharpe_ratio` returns
    ``confidence`` to floating-point precision.

    Parameters
    ----------
    result:
        A computed :class:`DSRResult`.
    confidence:
        The bar to invert, defaulting to :data:`DSR_THRESHOLD`.

    Returns
    -------
    float
        Annualised Sharpe required. Compare against ``result.sharpe_annual``:
        the record passes exactly when it is at or above this number.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence}.")
    quantile = float(sps.norm.ppf(confidence))
    return result.expected_max_sharpe_annual + quantile * result.psr_result.standard_error_annual


def effective_number_of_trials(
    correlation: FloatArray,
    *,
    method: TrialCountMethod = "cluster",
    cluster_threshold: float = 0.5,
) -> float:
    """Discount a raw trial count by how correlated the trials actually are.

    Parameters
    ----------
    correlation:
        Square cross-trial correlation matrix.
    method:
        ``"independent"``
            No discount; returns ``N``. Use when trials are genuinely unrelated.
        ``"equicorrelated"``
            :math:`N_{eff} = N / (1 + (N-1)\\bar{\\rho})` using the mean off-diagonal
            correlation. Cheap, closed-form, and exact when correlations are uniform.
        ``"cluster"`` (default)
            Hierarchical clustering on the correlation distance
            :math:`d = \\sqrt{(1-\\rho)/2}`, counting clusters. Handles the realistic
            case of several tight families of near-duplicate strategies.
    cluster_threshold:
        Correlation above which two trials are treated as the same bet. The
        default of 0.5 is deliberately conservative.

    Returns
    -------
    float
        A value in ``[1, N]``.
    """
    corr = np.asarray(correlation, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"correlation must be a square matrix, got shape {corr.shape}.")
    n = corr.shape[0]
    if n < 2:
        return float(n)

    if method == "independent":
        return float(n)

    if method == "equicorrelated":
        off_diagonal = corr[~np.eye(n, dtype=bool)]
        rho = float(np.clip(np.mean(off_diagonal), 0.0, 1.0))
        return float(np.clip(n / (1.0 + (n - 1) * rho), 1.0, n))

    if method == "cluster":
        # Clip first: sampling noise can push a correlation marginally outside
        # [-1, 1], which would make the distance imaginary.
        safe = np.clip(corr, -1.0, 1.0)
        distance = np.sqrt(np.maximum((1.0 - safe) / 2.0, 0.0))
        np.fill_diagonal(distance, 0.0)
        distance = (distance + distance.T) / 2.0  # enforce exact symmetry
        linkage = hierarchy.linkage(squareform(distance, checks=False), method="average")
        cut = math.sqrt(max((1.0 - cluster_threshold) / 2.0, 0.0))
        labels = hierarchy.fcluster(linkage, t=cut, criterion="distance")
        return float(len(np.unique(labels)))

    raise ValueError(f"Unknown method {method!r}.")


@dataclass(frozen=True)
class DSRResult:
    """Deflated Sharpe Ratio and the selection-bias hurdle it was measured against."""

    dsr: float
    psr_at_zero: float
    sharpe_annual: float
    expected_max_sharpe_annual: float
    n_trials: int
    n_effective_trials: float
    trial_sharpe_std_annual: float
    psr_result: PSRResult

    def __float__(self) -> float:
        return self.dsr

    @property
    def passed(self) -> bool:
        """Does the strategy survive its own search, at :data:`DSR_THRESHOLD`?"""
        return self.dsr >= DSR_THRESHOLD

    @property
    def interpretation(self) -> str:
        return (
            f"Across {self.n_trials:,} trials ({self.n_effective_trials:.1f} effectively "
            f"independent), the best of pure noise would be expected to post an annualised "
            f"Sharpe of {self.expected_max_sharpe_annual:.2f}. This strategy posted "
            f"{self.sharpe_annual:.2f}, giving a {self.dsr:.1%} probability that its edge "
            f"is real rather than the product of the search."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "dsr": self.dsr,
            "psr_at_zero": self.psr_at_zero,
            "sharpe_annual": self.sharpe_annual,
            "expected_max_sharpe_annual": self.expected_max_sharpe_annual,
            "n_trials": self.n_trials,
            "n_effective_trials": self.n_effective_trials,
            "trial_sharpe_std_annual": self.trial_sharpe_std_annual,
        }


def deflated_sharpe_ratio(
    data: ReturnSeries | FloatArray,
    *,
    n_trials: int,
    trial_sharpe_std_annual: float | None = None,
    n_effective_trials: float | None = None,
    risk_free_rate: float = 0.0,
) -> DSRResult:
    """Deflate a single track record by the number of strategies that were tried.

    Parameters
    ----------
    n_trials:
        How many strategy variants were evaluated before this one was selected.
        **This is the number people lie to themselves about.** It includes every
        parameter setting tried, every universe swapped in, every date range
        nudged — not just the configurations that made it into a spreadsheet.
    trial_sharpe_std_annual:
        Cross-sectional dispersion of the trial Sharpes, annualised. If omitted,
        defaults to the zero-edge null dispersion :math:`1/\\sqrt{n-1}`, which is
        the right assumption when the trials were genuinely uninformative.
    n_effective_trials:
        Override the independent-trial count directly. Prefer
        :func:`deflated_sharpe_ratio_from_trials` when you still have the trial
        return streams, since it measures this rather than assuming it.
    """
    if n_trials < 1:
        raise InsufficientDataError(f"n_trials must be at least 1, got {n_trials}.")

    series = as_return_series(data)
    ppy = series.periods_per_year

    if trial_sharpe_std_annual is None:
        std_period = null_sharpe_std(series.n_periods)
    else:
        std_period = deannualize_sharpe(trial_sharpe_std_annual, ppy)

    effective = float(n_trials) if n_effective_trials is None else n_effective_trials
    hurdle_period = expected_max_sharpe(effective, std_period)
    hurdle_annual = annualize_sharpe(hurdle_period, ppy)

    psr_result = probabilistic_sharpe_ratio(
        series,
        benchmark_annual_sharpe=hurdle_annual,
        risk_free_rate=risk_free_rate,
    )
    psr_zero = probabilistic_sharpe_ratio(series, risk_free_rate=risk_free_rate).psr

    return DSRResult(
        dsr=psr_result.psr,
        psr_at_zero=psr_zero,
        sharpe_annual=psr_result.sharpe_annual,
        expected_max_sharpe_annual=hurdle_annual,
        n_trials=n_trials,
        n_effective_trials=effective,
        trial_sharpe_std_annual=annualize_sharpe(std_period, ppy),
        psr_result=psr_result,
    )


def deflated_sharpe_ratio_from_trials(
    trials: TrialMatrix,
    *,
    index: int | None = None,
    method: TrialCountMethod = "cluster",
    cluster_threshold: float = 0.5,
    risk_free_rate: float = 0.0,
) -> DSRResult:
    """Deflate the winning trial using dispersion and correlation measured from the family.

    This is the strong form. Rather than asking the user how many trials they ran
    and how dispersed those trials were, it measures both from the trial matrix —
    removing the two numbers most vulnerable to wishful thinking.

    Parameters
    ----------
    index:
        Which trial to judge. Defaults to the one with the highest Sharpe, which
        is what a backtester would have selected.
    """
    sharpes_period = np.array(
        [
            sharpe_ratio(trials.trial(i), risk_free_rate=risk_free_rate, annualized=False)
            for i in range(trials.n_trials)
        ]
    )
    chosen = int(np.argmax(sharpes_period)) if index is None else index

    effective = effective_number_of_trials(
        trials.correlation(), method=method, cluster_threshold=cluster_threshold
    )
    std_period = float(np.std(sharpes_period, ddof=1))

    return deflated_sharpe_ratio(
        trials.trial(chosen),
        n_trials=trials.n_trials,
        trial_sharpe_std_annual=annualize_sharpe(std_period, trials.periods_per_year),
        n_effective_trials=effective,
        risk_free_rate=risk_free_rate,
    )
