"""White's Reality Check and Hansen's SPA: does the *best* of the family beat a benchmark?

References
----------
White, H. (2000). *A Reality Check for Data Snooping.* Econometrica 68(5).
Hansen, P. R. (2005). *A Test for Superior Predictive Ability.* Journal of Business
& Economic Statistics 23(4).

The question
------------
PSR asks whether one track record is long enough to trust. DSR asks whether the
winner beats the best of ``N`` coin flips. PBO asks whether selection survives
into a hold-out. This module asks the question a sceptic asks first:

    **Is the best strategy in this family actually better than the benchmark, once
    you account for the fact that it is the best of many?**

With ``f[k, t]`` the period-``t`` return of strategy ``k`` minus the benchmark's,
the null is that *no* strategy has a positive expected differential:
:math:`\\max_k E[f_k] \\le 0`. The test statistic is the largest standardised
average outperformance in the family, and the null distribution comes from a
stationary bootstrap of the whole ``(K, T)`` panel at once.

Why the benchmark should not be zero
------------------------------------
Set the benchmark to zero and the question becomes "did any of these make money",
which almost any strategy family passes in a bull market. Set it to buy-and-hold
and the question becomes "was any of this worth doing", which is the one that
matters. Both are available through the ``benchmark`` parameter; the SPY result
this package reports uses buy-and-hold, because 0 of 157 mined variants beat it
and a test that cannot see that is not worth running.

Why the resamples must be shared across strategies
--------------------------------------------------
Every strategy is resampled on the **same** bootstrap index vector. This is the
single easiest way to get RC and SPA wrong, because resampling each strategy
independently destroys the cross-sectional dependence that gives ``max_k`` its
meaning.

The direction of that error is worth stating precisely, because it is the
opposite of what one might assume. A mined family is mostly *positively*
correlated — 157 moving-average rules over one price series are close to a single
bet. Independent resampling prices them as 157 separate bets, so the maximum of
the null is drawn far higher than it should be and the p-value comes back **too
large**: the test loses power rather than manufacturing significance. Measured on
20 exact duplicates of one strategy, where the correct answer is known by
construction because the family contains a single bet: shared indices return
p = 0.002, matching the single-strategy p-value exactly, while independent
resampling returns 0.020. ``test_duplicating_a_strategy_changes_nothing`` pins
the first half of that down without a tolerance.

How it stays fast
-----------------
The resampled mean of strategy ``k`` under replicate ``b`` depends on the
resampled indices only through *how many times each period was drawn*. So the
``(B, T)`` count matrix ``C`` is built once and every resampled mean falls out of
a single matrix product, ``f @ C.T / T``. For the SPY grid — ``K = 157``,
``T = 4173``, ``B = 1000`` — that is one 1.3 GFLOP matmul against a 33 MB matrix.
Materialising ``f[:, idx]`` for each replicate instead would need 5 GB.

Hansen's two improvements over White
------------------------------------
**Studentisation.** RC compares raw average outperformance across strategies, so
a high-variance strategy dominates the maximum on noise alone. SPA divides each
by :math:`\\hat{\\omega}_k`, the bootstrap standard deviation of its own
statistic, putting every strategy on the same footing.

**Dropping hopeless strategies from the recentring.** This is the one that
changes conclusions. Under RC, *every* strategy is recentred on its own sample
mean, so a strategy that lost 90% of the time still contributes a mean-zero draw
to the maximum of the null distribution. Pile in enough of them and the null
inflates until nothing can reject it. SPA recentres a hopeless strategy on zero
instead, leaving its bootstrap statistic as far below the maximum as its sample
performance was:

======================  ================================================
recentring ``g_k``      threshold
======================  ================================================
``lower``               ``mean(f_k) * 1{mean(f_k) >= 0}``
``consistent``          ``mean(f_k) * 1{mean(f_k) >= -c_k}``
``upper``               ``mean(f_k)`` — identical to RC's recentring
======================  ================================================

with :math:`c_k = \\sqrt{\\hat{\\omega}_k^2 \\cdot 2 \\ln \\ln T / T}`. Since
``lower`` drops a superset of what ``consistent`` drops, and ``upper`` drops
nothing, the recentred bootstrap statistics are ordered replicate by replicate
and therefore ``p_lower <= p_consistent <= p_upper`` holds exactly, not
approximately. ``consistent`` is the one to quote; ``lower`` is deliberately
liberal and ``upper`` deliberately conservative, and they are reported so the
gap between them is visible rather than hidden.

Two spec corrections, recorded rather than quietly fixed
--------------------------------------------------------
The Phase 6 brief got the sign wrong twice, in both cases on a point that is
decidable by argument rather than by opinion.

**"Adding garbage strategies will lower RC's p-value."** It cannot. Extra
strategies only enlarge the set the bootstrap maximum is taken over, so ``V*_b``
is weakly larger for every single replicate while the observed statistic is
unchanged — the p-value therefore **rises**, monotonically. Measured: adding 100
hopeless strategies to a family of 10 moves RC from 0.011 to 0.140, carrying it
across any conventional threshold, while SPA's ``consistent`` p-value stays at
0.013 to the last bit. Losing power is Hansen's actual complaint about RC, and
the contrast survives the correction intact — it is sharper for it.

**"Independent resampling produces p-values far too small."** Backwards, for the
positively correlated families that mining produces; see above. The structural
instruction — share one index set — is right, and only the stated direction of
the failure was wrong.

Both are asserted in the tests, so a future reader meets the measurement rather
than the claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import DataValidationError, InsufficientDataError
from ..types import BoolArray, FloatArray, IntArray, ReturnSeries, TrialMatrix
from .bootstrap import optimal_block_length, stationary_indices

__all__ = [
    "DEFAULT_N_RESAMPLES",
    "MIN_PERIODS",
    "SIGNIFICANCE_LEVEL",
    "RealityCheckResult",
    "reality_check",
]

#: Bootstrap replicates. 1,000 is White's own choice and resolves p-values to
#: about a thousandth, which is finer than the sampling error in any of the
#: inputs.
DEFAULT_N_RESAMPLES = 1000

#: Below this many periods the studentisation threshold ``2 ln ln T`` and the
#: block-length estimator are both being asked to work on nothing. RC on 9
#: observations is not a conservative test, it is a meaningless one.
MIN_PERIODS = 10

#: The bar :attr:`RealityCheckResult.passed` applies to the SPA ``consistent``
#: p-value. Unlike :data:`luckdetector.stats.pbo.PBO_THRESHOLD` this one *is* a
#: convention, but it is still named so it can be argued with.
SIGNIFICANCE_LEVEL = 0.05

#: A bootstrap standard deviation below this multiple of the statistic's own
#: scale is floating-point residue, not sampling variation — the same reasoning
#: as :func:`luckdetector.stats.moments.is_effectively_constant`. It fires when a
#: strategy is *identical* to the benchmark, which makes its studentised
#: statistic 0/0.
_RELATIVE_ZERO = 1e-12


def _benchmark_values(
    benchmark: float | FloatArray | ReturnSeries, n_periods: int
) -> tuple[FloatArray, str]:
    """Coerce the benchmark to a length-``T`` array of returns, plus a label."""
    if isinstance(benchmark, ReturnSeries):
        values = benchmark.values
        name = benchmark.name
    elif isinstance(benchmark, (int, float, np.floating, np.integer)):
        constant = float(benchmark)
        label = "zero" if constant == 0.0 else f"a constant {constant:.4g} per period"
        return np.full(n_periods, constant, dtype=np.float64), label
    else:
        values = np.asarray(benchmark, dtype=np.float64).ravel()
        name = "benchmark"

    if values.size != n_periods:
        raise DataValidationError(
            f"Benchmark has {values.size} periods but the trials have {n_periods}. "
            "The two must be aligned on the same calendar — a benchmark offset by even "
            "one period compares each strategy against the wrong day."
        )
    if not np.all(np.isfinite(values)):
        raise DataValidationError("Benchmark contains non-finite values (NaN or inf).")
    return np.asarray(values, dtype=np.float64), name


def _automatic_block_length(differentials: FloatArray) -> float:
    """Median Politis–White block length across the strategies' differentials.

    One block length has to serve every strategy, because they share a single set
    of resampling indices. The median rather than the mean because a single
    rarely-trading variant can return a block length several times the typical
    one, and there is no reason to let it set the dependence assumption for the
    whole family.

    Each row is divided by its peak absolute value first. Politis–White depends
    only on the autocorrelation structure, so it is invariant to that rescaling
    (the numerator and denominator of every term scale together); the point is to
    keep the row inside the validation domain of :class:`ReturnSeries`, which is
    written for *returns* and rejects anything below -100%. A difference of two
    returns is entitled to be smaller than that.
    """
    lengths: list[float] = []
    for row in differentials:
        peak = float(np.max(np.abs(row)))
        scaled = row / peak if peak > 0.0 else row
        lengths.append(optimal_block_length(scaled, method="stationary"))
    return float(np.median(lengths))


def _resample_count_matrix(indices: IntArray, n_periods: int) -> FloatArray:
    """``(B, T)`` matrix counting how often each period appears in each replicate.

    This is the whole vectorisation. A resampled mean is a weighted sum of the
    original observations with integer weights, so once the weights are known the
    resampling is a matrix product and no replicate is ever materialised.
    """
    n_resamples = int(indices.shape[0])
    offsets = np.arange(n_resamples, dtype=np.int64) * n_periods
    flat = (indices + offsets[:, None]).ravel()
    counts = np.bincount(flat, minlength=n_resamples * n_periods)
    return np.asarray(counts.reshape(n_resamples, n_periods), dtype=np.float64)


def _bootstrap_p_value(null: FloatArray, observed: float) -> float:
    """Share of the null at or above ``observed``, counting the observed value in.

    The ``+1`` in numerator and denominator is the same convention as
    :meth:`luckdetector.stats.bootstrap.BootstrapResult.p_value`: it prevents a
    p-value of exactly zero, which would claim more resolution than ``B`` draws
    can support.
    """
    return float((int(np.sum(null >= observed)) + 1) / (null.size + 1))


def _spa_null(
    centred: FloatArray,
    shift: FloatArray,
    omega: FloatArray,
    degenerate: BoolArray,
) -> FloatArray:
    """``(B,)`` null distribution of the SPA statistic for one recentring rule.

    ``centred`` is ``sqrt(T) * (mean(f*_k,b) - mean(f_k))``; ``shift`` is
    ``sqrt(T) * (mean(f_k) - g_k)``, which is zero for a strategy that keeps its
    own mean and strongly negative for one that has been recentred on zero.
    """
    statistic = (centred + shift[:, None]) / omega[:, None]
    statistic = np.where(degenerate[:, None], -np.inf, statistic)
    return np.asarray(np.maximum(np.max(statistic, axis=0), 0.0), dtype=np.float64)


@dataclass(frozen=True)
class RealityCheckResult:
    """White's Reality Check and Hansen's SPA, computed from one shared bootstrap.

    Both tests need the same resampled panel, and SPA's studentisation is
    estimated from those very replicates, so running them separately would cost
    twice as much and produce two inconsistent nulls. They are reported together.

    **A large p-value is the common outcome and it is not a bug.** The null is
    that nothing in the family beats the benchmark; failing to reject it means the
    data gave no reason to believe otherwise, which is exactly what an honest
    mined family usually shows.
    """

    p_reality_check: float
    p_lower: float
    p_consistent: float
    p_upper: float
    statistic_reality_check: float
    statistic_spa: float
    best_trial: int
    best_label: str
    best_trial_studentised: int
    mean_outperformance: FloatArray
    omega: FloatArray
    n_trials: int
    n_periods: int
    periods_per_year: int
    n_resamples: int
    block_length: float
    benchmark_name: str
    n_recentred_lower: int
    n_recentred_consistent: int
    n_degenerate: int

    def __float__(self) -> float:
        """The number to quote: SPA's ``consistent`` p-value."""
        return self.p_consistent

    @property
    def passed(self) -> bool:
        """Did the family beat the benchmark by more than data snooping explains?

        Polarity matches :class:`luckdetector.types.TestResult`: ``True`` means the
        evidence is consistent with genuine skill.
        """
        return self.p_consistent < SIGNIFICANCE_LEVEL

    @property
    def best_outperformance_annual(self) -> float:
        """Best strategy's average outperformance, annualised, in return units."""
        return float(self.mean_outperformance[self.best_trial]) * self.periods_per_year

    @property
    def n_beating_benchmark(self) -> int:
        """How many strategies out-performed the benchmark on average at all."""
        return int(np.sum(self.mean_outperformance > 0.0))

    @property
    def interpretation(self) -> str:
        if self.n_beating_benchmark == 0:
            lead = (
                f"Not one of the {self.n_trials:,} strategies beat {self.benchmark_name} on "
                f"average, so there is nothing for the test to fail to explain"
            )
        else:
            lead = (
                f"{self.n_beating_benchmark:,} of {self.n_trials:,} strategies beat "
                f"{self.benchmark_name} on average, the best of them ('{self.best_label}') by "
                f"{self.best_outperformance_annual:.2%} a year"
            )
        verdict = (
            "beyond what selecting the best of the family explains"
            if self.passed
            else "well within what selecting the best of the family explains"
        )
        return (
            f"{lead}. Against {self.benchmark_name}, White's Reality Check returns "
            f"p = {self.p_reality_check:.4f} and Hansen's SPA p = {self.p_consistent:.4f} "
            f"(bracketed by {self.p_lower:.4f} and {self.p_upper:.4f}) over "
            f"{self.n_resamples:,} stationary-bootstrap resamples with a mean block length "
            f"of {self.block_length:.1f} periods — {verdict}."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "p_reality_check": self.p_reality_check,
            "p_lower": self.p_lower,
            "p_consistent": self.p_consistent,
            "p_upper": self.p_upper,
            "statistic_reality_check": self.statistic_reality_check,
            "statistic_spa": self.statistic_spa,
            "best_trial": self.best_trial,
            "best_label": self.best_label,
            "best_outperformance_annual": self.best_outperformance_annual,
            "n_beating_benchmark": self.n_beating_benchmark,
            "n_trials": self.n_trials,
            "n_periods": self.n_periods,
            "n_resamples": self.n_resamples,
            "block_length": self.block_length,
            "benchmark": self.benchmark_name,
            "n_recentred_lower": self.n_recentred_lower,
            "n_recentred_consistent": self.n_recentred_consistent,
            "n_degenerate": self.n_degenerate,
        }


def reality_check(
    trials: TrialMatrix,
    benchmark: float | FloatArray | ReturnSeries = 0.0,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    block_length: float | None = None,
    seed: int | np.random.Generator | None = 0,
) -> RealityCheckResult:
    """Test whether the best strategy in a family beats a benchmark.

    Runs White's Reality Check and all three variants of Hansen's SPA against one
    shared set of stationary-bootstrap resamples.

    Parameters
    ----------
    trials:
        Every strategy that was tried. As with PBO, keeping only the winner makes
        this test impossible: the whole point is to price the maximum over the
        family, and a family of one has no maximum to price.
    benchmark:
        A constant per-period return (``0.0``, the default), or a length-``T``
        array, or a :class:`~luckdetector.types.ReturnSeries` aligned with the
        trials. **Zero is the soft test.** Passing ``MiningResult.buy_and_hold``
        asks whether the search beat doing nothing, which is the question that
        actually discriminates.
    n_resamples:
        Bootstrap replicates ``B``. Resolves p-values to about ``1 / B``.
    block_length:
        Mean block length for the stationary bootstrap. Left as ``None`` it is
        estimated by the Politis–White rule from the loss differentials
        themselves, which is the honest default — a hand-picked block length is
        one more researcher degree of freedom.
    seed:
        An int, a ``Generator``, or ``None``. Defaults to 0 so results reproduce
        without the caller having to think about it.

    Returns
    -------
    RealityCheckResult
        Small p-values mean the family's best beat the benchmark by more than
        data snooping accounts for. Quote ``p_consistent``.

    Raises
    ------
    InsufficientDataError
        If the sample is shorter than :data:`MIN_PERIODS`.
    DataValidationError
        If the benchmark is misaligned with the trials or non-finite.
    ValueError
        If ``n_resamples`` or an explicit ``block_length`` is not positive.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n_trials, n_periods = trials.shape

    if n_periods < MIN_PERIODS:
        raise InsufficientDataError(
            f"Reality Check needs at least {MIN_PERIODS} periods, got {n_periods}. Below "
            "that the block-length estimate and the 2*ln(ln(T)) recentring threshold are "
            "both meaningless."
        )
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be at least 2, got {n_resamples}.")

    benchmark_values, benchmark_name = _benchmark_values(benchmark, n_periods)

    # f[k, t]: how much strategy k beat the benchmark by in period t.
    differentials = trials.values - benchmark_values
    mean_outperformance = np.asarray(differentials.mean(axis=1), dtype=np.float64)
    root_t = math.sqrt(n_periods)

    if block_length is None:
        block_length = _automatic_block_length(differentials)
    elif block_length <= 0.0:
        raise ValueError(f"block_length must be positive, got {block_length}.")

    # One index set for every strategy. Resampling them independently would break
    # the cross-sectional dependence and shrink the null distribution — see the
    # module docstring.
    indices = stationary_indices(n_periods, block_length, n_resamples, rng)
    counts = _resample_count_matrix(indices, n_periods)
    resampled_means = (differentials @ counts.T) / n_periods
    centred = root_t * (resampled_means - mean_outperformance[:, None])

    # ------------------------------------------------------------ Reality Check
    observed_rc = root_t * mean_outperformance
    statistic_rc = float(np.max(observed_rc))
    p_reality_check = _bootstrap_p_value(np.max(centred, axis=0), statistic_rc)

    # -------------------------------------------------------------------- SPA
    # omega_k is the sampling standard deviation of sqrt(T) * mean(f_k), taken
    # from the replicates themselves so it inherits the same serial dependence
    # the block bootstrap was chosen to preserve.
    omega = np.asarray(np.std(centred, axis=1, ddof=1), dtype=np.float64)
    scale = root_t * np.mean(np.abs(differentials), axis=1)
    degenerate: BoolArray = np.asarray(omega <= _RELATIVE_ZERO * scale, dtype=np.bool_)
    safe_omega = np.where(degenerate, 1.0, omega)

    studentised = np.where(degenerate, -np.inf, observed_rc / safe_omega)
    statistic_spa = float(max(0.0, float(np.max(studentised))))

    # Hansen's threshold: a strategy this far below zero cannot plausibly be a
    # contender, so recentring it on its own mean only inflates the null.
    cutoff = -np.sqrt(np.square(omega) * (2.0 * math.log(math.log(n_periods))) / n_periods)
    keep_lower: BoolArray = np.asarray(mean_outperformance >= 0.0, dtype=np.bool_)
    keep_consistent: BoolArray = np.asarray(mean_outperformance >= cutoff, dtype=np.bool_)

    # shift = sqrt(T) * (mean(f_k) - g_k): zero when the strategy keeps its own
    # mean, strongly negative when it has been recentred on zero instead.
    zeros = np.zeros(n_trials, dtype=np.float64)
    shift_lower = np.where(keep_lower, zeros, observed_rc)
    shift_consistent = np.where(keep_consistent, zeros, observed_rc)

    p_lower = _bootstrap_p_value(
        _spa_null(centred, shift_lower, safe_omega, degenerate), statistic_spa
    )
    p_consistent = _bootstrap_p_value(
        _spa_null(centred, shift_consistent, safe_omega, degenerate), statistic_spa
    )
    p_upper = _bootstrap_p_value(_spa_null(centred, zeros, safe_omega, degenerate), statistic_spa)

    best_trial = int(np.argmax(mean_outperformance))
    return RealityCheckResult(
        p_reality_check=p_reality_check,
        p_lower=p_lower,
        p_consistent=p_consistent,
        p_upper=p_upper,
        statistic_reality_check=statistic_rc,
        statistic_spa=statistic_spa,
        best_trial=best_trial,
        best_label=trials.labels[best_trial],
        best_trial_studentised=int(np.argmax(studentised)),
        mean_outperformance=mean_outperformance,
        omega=omega,
        n_trials=n_trials,
        n_periods=n_periods,
        periods_per_year=trials.periods_per_year,
        n_resamples=int(n_resamples),
        block_length=float(block_length),
        benchmark_name=benchmark_name,
        n_recentred_lower=int(np.sum(~keep_lower)),
        n_recentred_consistent=int(np.sum(~keep_consistent)),
        n_degenerate=int(np.sum(degenerate)),
    )
