"""Probability of Backtest Overfitting, via Combinatorially Symmetric Cross-Validation.

Reference
---------
Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). *The Probability of
Backtest Overfitting.* Journal of Computational Finance 20(4).

The idea
--------
DSR asks whether the winner's Sharpe ratio is larger than the best of ``N`` coin
flips. PBO asks a sharper and more practical question: **when you pick the
in-sample winner, does it stay a winner?**

Take the ``T`` periods of a trial matrix, cut them into ``S`` contiguous blocks,
and choose half the blocks as in-sample and the complementary half as
out-of-sample. Select whichever trial looked best in-sample, then look up where
that same trial ranks out-of-sample. Repeat over every one of the
:math:`\\binom{S}{S/2}` ways of splitting. If selection carries no information,
the winner's out-of-sample rank is uniform, so it lands in the bottom half half
the time:

.. math::

    \\bar{\\omega}_c = \\frac{r_c}{N+1}, \\qquad
    \\lambda_c = \\ln\\frac{\\bar{\\omega}_c}{1 - \\bar{\\omega}_c}, \\qquad
    \\mathrm{PBO} = P(\\lambda_c \\le 0)

**PBO = 0.5 is the noise baseline, not a passing grade.** Above 0.5 is worse than
noise: in-sample selection is actively anti-predictive, which is the signature of
a search that has fitted the sample's idiosyncrasies rather than any structure.

Why the blocks must be contiguous
---------------------------------
Randomly assigning individual periods to folds would scatter each block of
volatility clustering, each trend, and each regime across *both* sides of the
split. In-sample and out-of-sample would then be near-duplicates of one another,
the winner would trivially persist, and PBO would come back reassuringly low for
a strategy with no out-of-sample life at all. Contiguity is what makes the two
halves genuinely different samples.

Why "combinatorially symmetric"
-------------------------------
A single train/test split gives one number that depends entirely on where the
cut happened to fall — the classic way to fool yourself is to try a few cut
points and report the flattering one. Enumerating *all* balanced splits removes
that degree of freedom, and because every split appears alongside its own
complement, in-sample and out-of-sample are drawn from identically distributed
sets. No split is privileged.

Why the performance metric is fixed to the Sharpe ratio
-------------------------------------------------------
``S = 16`` means 12,870 splits, and a Python loop that slices and re-scores
157 trials for each of them is minutes of work. But the Sharpe ratio of any union
of blocks is a function of nothing more than the per-block counts, sums and sums
of squares, so all 12,870 splits collapse into two matrix products against a
``(n_splits, S)`` membership matrix. That is the whole reason this module runs in
well under a second, and it is only available for metrics that decompose over
blocks — hence no pluggable metric callback.

The sums are taken about each trial's full-sample mean. Variance from the raw
``E[x^2] - E[x]^2`` identity loses precision through catastrophic cancellation;
centring first makes the subtracted term smaller than the retained one by a
factor of the subsample length, so the identity becomes numerically harmless.

A warning about the degradation slope
-------------------------------------
It is tempting to read a negative in-sample/out-of-sample regression slope as
proof of overfitting. **It is not**, and this module measures the reason rather
than assuming it away. Within one dataset the two halves *partition a fixed
total*: whatever the in-sample half takes, the out-of-sample half must give back,
so :math:`\\bar{r}_{OOS} \\approx \\bar{r}_{total} - \\bar{r}_{IS}` and the slope is
mechanically pinned near :math:`-1` before any strategy selection happens at all.
Measured on a *fixed* trial with no selection whatsoever, the slope of this
package's own estimator is -0.999.

So the null for :class:`DegradationResult` is roughly -1, not 0, and the
statistic barely separates a mined family from pure noise. It is computed and
kept because the in-sample/out-of-sample scatter is worth *seeing*, and because
Bailey et al. report it — not because it should carry weight in a verdict.
:attr:`PBOResult.probability_of_loss` and
:attr:`PBOResult.dominance_fraction` discriminate; the slope does not.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import DegenerateSeriesError, InsufficientDataError
from ..types import BoolArray, FloatArray, IntArray, TrialMatrix
from .moments import annualize_sharpe, is_effectively_constant

__all__ = [
    "DEFAULT_MAX_COMBINATIONS",
    "DEFAULT_N_BLOCKS",
    "MIN_N_BLOCKS",
    "PBO_NOISE_BASELINE",
    "PBO_THRESHOLD",
    "DegradationResult",
    "PBOResult",
    "contiguous_blocks",
    "performance_degradation",
    "probability_of_backtest_overfitting",
]

#: Bailey et al.'s recommended block count: 12,870 splits, each half of the sample.
DEFAULT_N_BLOCKS = 16

#: Below this, there are too few distinct splits for the logit distribution to mean
#: anything: ``S = 2`` yields two splits and a PBO that can only be 0, 0.5 or 1.
MIN_N_BLOCKS = 4

#: PBO under pure noise. Selection that carries no information lands the in-sample
#: winner in the bottom half of the out-of-sample ranking exactly half the time.
PBO_NOISE_BASELINE = 0.5

#: The bar :attr:`PBOResult.passed` applies. **This is a judgement call, not a
#: convention from the literature** — Bailey et al. propose no cutoff. It is a named
#: constant precisely so that it can be argued with rather than buried in a
#: comparison. 0.2 says: the in-sample winner must survive into the top half of the
#: out-of-sample field in at least four splits out of five.
PBO_THRESHOLD = 0.2

#: Enumerating every balanced split becomes expensive past ``S = 18``
#: (``C(20, 10) = 184,756``). Beyond this many, splits are sampled at random and
#: :attr:`PBOResult.exhaustive` is set to ``False``.
DEFAULT_MAX_COMBINATIONS = 20_000

#: A subsample standard deviation below this multiple of the trial's own average
#: absolute return is floating-point residue rather than risk. Same reasoning as
#: :func:`luckdetector.stats.moments.is_effectively_constant`, applied per split.
_SUBSAMPLE_RELATIVE_ZERO = 1e-12


def contiguous_blocks(n_periods: int, n_blocks: int) -> list[tuple[int, int]]:
    """Cut ``n_periods`` into ``n_blocks`` contiguous ``[start, stop)`` ranges.

    Sizes differ by at most one period, with the longer blocks first, so no
    observation is discarded when ``n_blocks`` does not divide ``n_periods``.
    Bailey et al. assume exactly equal blocks; trimming the remainder would throw
    away up to ``S - 1`` periods for no benefit, and since the Sharpe ratio is a
    per-period statistic, blocks differing by a single observation change nothing.

    Raises
    ------
    ValueError
        If ``n_blocks`` is odd or smaller than :data:`MIN_N_BLOCKS`.
    InsufficientDataError
        If the sample cannot give every block at least two observations.
    """
    if n_blocks % 2 != 0:
        raise ValueError(
            f"n_blocks must be even so that in-sample and out-of-sample halves are the "
            f"same size, got {n_blocks}. CSCV is defined on balanced splits; an odd "
            "block count would systematically favour one side."
        )
    if n_blocks < MIN_N_BLOCKS:
        raise ValueError(
            f"n_blocks must be at least {MIN_N_BLOCKS}, got {n_blocks}. Fewer blocks give "
            f"too few distinct splits for P(lambda <= 0) to be estimable."
        )
    if n_periods < 2 * n_blocks:
        raise InsufficientDataError(
            f"Cannot cut {n_periods} periods into {n_blocks} blocks: every block needs at "
            f"least 2 observations, so this requires {2 * n_blocks} periods. Use fewer "
            "blocks or a longer sample."
        )

    base, remainder = divmod(n_periods, n_blocks)
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(n_blocks):
        stop = start + base + (1 if i < remainder else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def _membership_matrix(
    n_blocks: int,
    *,
    max_combinations: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, bool]:
    """In-sample indicator matrix of shape ``(n_splits, n_blocks)``.

    Exhaustive when :math:`\\binom{S}{S/2}` is affordable, otherwise a random
    sample of distinct balanced splits.
    """
    half = n_blocks // 2
    total = math.comb(n_blocks, half)

    if total <= max_combinations:
        chosen = np.array(list(itertools.combinations(range(n_blocks), half)), dtype=np.int64)
        exhaustive = True
    else:
        # Draw distinct subsets by rejection, keyed on a bitmask. Cheaper than
        # materialising all C(S, S/2) rows purely to discard most of them.
        seen: set[int] = set()
        rows: list[IntArray] = []
        while len(rows) < max_combinations:
            draw = rng.permutation(n_blocks)[:half]
            key = int(np.sum(np.left_shift(1, draw, dtype=np.int64)))
            if key not in seen:
                seen.add(key)
                rows.append(np.asarray(np.sort(draw), dtype=np.int64))
        chosen = np.array(rows, dtype=np.int64)
        exhaustive = False

    membership = np.zeros((chosen.shape[0], n_blocks), dtype=np.float64)
    np.put_along_axis(membership, chosen, 1.0, axis=1)
    return membership, exhaustive


def _block_statistics(
    values: FloatArray, blocks: Sequence[tuple[int, int]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Per-block count, sum and sum of squares of each trial's centred returns.

    Returns ``(counts, sums, sums_of_squares, centre)`` with shapes ``(S,)``,
    ``(N, S)``, ``(N, S)`` and ``(N, 1)``. Adding ``centre`` back recovers the
    uncentred mean; the variance is unaffected by the shift.
    """
    centre = np.asarray(np.mean(values, axis=1, keepdims=True), dtype=np.float64)
    centred = values - centre
    counts = np.array([stop - start for start, stop in blocks], dtype=np.float64)
    sums = np.column_stack([centred[:, start:stop].sum(axis=1) for start, stop in blocks])
    squares = np.column_stack(
        [np.square(centred[:, start:stop]).sum(axis=1) for start, stop in blocks]
    )
    return (
        counts,
        np.asarray(sums, dtype=np.float64),
        np.asarray(squares, dtype=np.float64),
        centre,
    )


def _subsample_sharpe(
    counts: FloatArray,
    sums: FloatArray,
    squares: FloatArray,
    centre: FloatArray,
    scale: FloatArray,
    membership: FloatArray,
) -> tuple[FloatArray, BoolArray]:
    """Per-period Sharpe of every trial on every subsample, plus a degeneracy mask.

    Shapes: ``membership`` is ``(n_splits, S)``; the result is ``(N, n_splits)``.
    A subsample with no dispersion has no risk to adjust for, so its Sharpe ratio
    does not exist; those entries come back as ``-inf`` — never selected in-sample,
    ranked last out-of-sample — and are counted rather than silently absorbed.
    """
    n = membership @ counts
    total = sums @ membership.T
    total_squares = squares @ membership.T

    mean = total / n
    variance = (total_squares - n * np.square(mean)) / (n - 1.0)
    sd = np.sqrt(np.maximum(variance, 0.0))

    degenerate: BoolArray = np.asarray(sd <= _SUBSAMPLE_RELATIVE_ZERO * scale, dtype=np.bool_)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (mean + centre) / sd
    return np.asarray(np.where(degenerate, -np.inf, sharpe), dtype=np.float64), degenerate


@dataclass(frozen=True)
class DegradationResult:
    """OLS of out-of-sample Sharpe on in-sample Sharpe, across splits.

    **Read the slope against a null of about -1, not 0.** The two halves of each
    split partition a fixed total, which forces a negative relationship before any
    selection takes place — measured at -0.999 on a fixed trial with no selection
    at all. A negative slope here is therefore the *expected* result under pure
    noise and is not evidence of overfitting; see the module docstring.

    Kept for the scatter plot and for comparability with Bailey et al., not as a
    load-bearing statistic.
    """

    slope: float
    intercept: float
    r_squared: float
    n_points: int


def performance_degradation(
    in_sample: FloatArray, out_of_sample: FloatArray
) -> DegradationResult:
    """Regress out-of-sample performance on in-sample performance.

    Non-finite pairs are dropped before fitting, so degenerate splits neither
    contribute to nor poison the fit.

    Raises
    ------
    InsufficientDataError
        If fewer than three usable pairs survive.
    DegenerateSeriesError
        If the in-sample values have no dispersion, leaving the slope undefined.
    """
    x = np.asarray(in_sample, dtype=np.float64).ravel()
    y = np.asarray(out_of_sample, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"Got {x.size} in-sample and {y.size} out-of-sample values.")

    usable = np.isfinite(x) & np.isfinite(y)
    x, y = x[usable], y[usable]
    if x.size < 3:
        raise InsufficientDataError(
            f"Need at least 3 finite (in-sample, out-of-sample) pairs to fit a slope, got {x.size}."
        )

    x_centred = x - x.mean()
    y_centred = y - y.mean()
    sxx = float(x_centred @ x_centred)
    if sxx <= 0.0 or is_effectively_constant(x):
        raise DegenerateSeriesError(
            "In-sample Sharpe ratios are identical across splits, so the degradation "
            "slope is undefined. This usually means every trial is the same strategy."
        )

    sxy = float(x_centred @ y_centred)
    syy = float(y_centred @ y_centred)
    slope = sxy / sxx
    return DegradationResult(
        slope=slope,
        intercept=float(y.mean() - slope * x.mean()),
        r_squared=0.0 if syy <= 0.0 else float(sxy * sxy / (sxx * syy)),
        n_points=int(x.size),
    )


def _dominance_fraction(selected: FloatArray, benchmark: FloatArray) -> float:
    """Share of the pooled support where the selected CDF sits at or above the benchmark's.

    A CDF that lies higher everywhere describes a distribution with its mass lower
    down — worse. So a value of 1.0 means picking a trial *at random* first-order
    stochastically dominates picking the in-sample winner, and 0.0 means the
    reverse.
    """
    grid = np.unique(np.concatenate([selected, benchmark]))
    grid = grid[np.isfinite(grid)]
    if grid.size == 0:
        return float("nan")
    cdf_selected = np.searchsorted(np.sort(selected), grid, side="right") / selected.size
    cdf_benchmark = np.searchsorted(np.sort(benchmark), grid, side="right") / benchmark.size
    return float(np.mean(cdf_selected >= cdf_benchmark))


@dataclass(frozen=True)
class PBOResult:
    """Probability of backtest overfitting, with the evidence behind it.

    The per-split arrays are kept so the report layer can draw the logit histogram
    and the in-sample/out-of-sample scatter without re-running the cross-validation.
    """

    pbo: float
    n_trials: int
    n_blocks: int
    n_splits: int
    exhaustive: bool
    logits: FloatArray
    relative_ranks: FloatArray
    selected_trials: IntArray
    is_sharpe_selected: FloatArray
    oos_sharpe_selected: FloatArray
    oos_sharpe_random: FloatArray
    degradation: DegradationResult
    probability_of_loss: float
    dominance_fraction: float
    n_degenerate_subsamples: int

    def __float__(self) -> float:
        return self.pbo

    @property
    def passed(self) -> bool:
        """Does the in-sample winner persist often enough to clear :data:`PBO_THRESHOLD`?"""
        return self.pbo < PBO_THRESHOLD

    @property
    def worse_than_noise(self) -> bool:
        """Is selection actively anti-predictive — worse than choosing at random?"""
        return self.pbo > PBO_NOISE_BASELINE

    @property
    def median_oos_sharpe_selected(self) -> float:
        """Median annualised out-of-sample Sharpe of the in-sample winner."""
        finite = self.oos_sharpe_selected[np.isfinite(self.oos_sharpe_selected)]
        return float(np.median(finite)) if finite.size else float("nan")

    @property
    def interpretation(self) -> str:
        verdict = (
            "worse than choosing a strategy at random"
            if self.worse_than_noise
            else "no better than a coin flip"
            if self.pbo >= PBO_THRESHOLD
            else "largely persistent"
        )
        return (
            f"Across {self.n_splits:,} balanced splits of {self.n_blocks} contiguous blocks, "
            f"the best of {self.n_trials:,} trials in-sample fell into the bottom half of the "
            f"field out-of-sample {self.pbo:.1%} of the time — {verdict}. Its median "
            f"out-of-sample Sharpe was {self.median_oos_sharpe_selected:.2f}, and it lost "
            f"money out-of-sample in {self.probability_of_loss:.1%} of splits."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pbo": self.pbo,
            "n_trials": self.n_trials,
            "n_blocks": self.n_blocks,
            "n_splits": self.n_splits,
            "exhaustive": self.exhaustive,
            "degradation_slope": self.degradation.slope,
            "degradation_intercept": self.degradation.intercept,
            "degradation_r_squared": self.degradation.r_squared,
            "probability_of_loss": self.probability_of_loss,
            "dominance_fraction": self.dominance_fraction,
            "median_oos_sharpe_selected": self.median_oos_sharpe_selected,
            "n_degenerate_subsamples": self.n_degenerate_subsamples,
        }


def probability_of_backtest_overfitting(
    trials: TrialMatrix,
    *,
    n_blocks: int = DEFAULT_N_BLOCKS,
    risk_free_rate: float = 0.0,
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
    seed: int | np.random.Generator | None = 0,
) -> PBOResult:
    """Estimate the probability that in-sample selection does not survive out-of-sample.

    Parameters
    ----------
    trials:
        Every strategy that was tried, not just the winner. Discarding the losers
        makes this statistic impossible to compute — the whole question is where
        the winner sits *within the field* on data it was not chosen on.
    n_blocks:
        Number of contiguous blocks ``S``. Must be even and at least
        :data:`MIN_N_BLOCKS`. The default of 16 gives 12,870 splits.
    risk_free_rate:
        Annual rate, converted internally to per-period. A constant shift, so it
        moves every trial's Sharpe together and rarely changes the ranking.
    max_combinations:
        Cap on the number of splits. Below the cap every balanced split is used and
        the result is exactly reproducible; above it, splits are sampled and
        :attr:`PBOResult.exhaustive` is ``False``.
    seed:
        An int, a ``Generator``, or ``None``. Used only for the random-trial
        comparison and for sampling splits when the cap binds; PBO itself is
        deterministic whenever the enumeration is exhaustive. Defaults to 0 so
        results reproduce without the caller having to think about it.

    Returns
    -------
    PBOResult
        ``pbo`` near 0.5 means selection is uninformative; above 0.5 means it is
        anti-predictive.

    Raises
    ------
    DegenerateSeriesError
        If any trial has effectively zero volatility over the full sample.
    InsufficientDataError
        If the sample is too short to be cut into ``n_blocks`` blocks.
    ValueError
        If ``n_blocks`` is odd or below :data:`MIN_N_BLOCKS`.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    values = trials.values - risk_free_rate / trials.periods_per_year
    for i in range(trials.n_trials):
        if is_effectively_constant(values[i]):
            raise DegenerateSeriesError(
                f"Trial {i} ('{trials.labels[i]}') has effectively zero volatility over the "
                "full sample, so it has no Sharpe ratio to rank. Drop never-trading variants "
                "before measuring PBO."
            )

    blocks = contiguous_blocks(trials.n_periods, n_blocks)
    membership, exhaustive = _membership_matrix(
        n_blocks, max_combinations=max_combinations, rng=rng
    )
    n_splits = membership.shape[0]

    counts, sums, squares, centre = _block_statistics(values, blocks)
    scale = np.asarray(np.mean(np.abs(values), axis=1, keepdims=True), dtype=np.float64)

    is_sharpe, is_degenerate = _subsample_sharpe(
        counts, sums, squares, centre, scale, membership
    )
    oos_sharpe, oos_degenerate = _subsample_sharpe(
        counts, sums, squares, centre, scale, 1.0 - membership
    )

    columns = np.arange(n_splits)
    selected = np.argmax(is_sharpe, axis=0)
    oos_of_selected = oos_sharpe[selected, columns]

    # Mid-rank of the selected trial among all trials out-of-sample, 1 = worst.
    # The +1 in the denominator keeps omega strictly inside (0, 1) so the logit
    # stays finite even when the winner ranks first or last.
    fewer = np.sum(oos_sharpe < oos_of_selected, axis=0)
    tied = np.sum(oos_sharpe == oos_of_selected, axis=0)
    ranks = np.asarray(fewer + 0.5 * tied + 0.5, dtype=np.float64)
    omega = ranks / (trials.n_trials + 1.0)
    logits = np.asarray(np.log(omega / (1.0 - omega)), dtype=np.float64)

    is_of_selected = is_sharpe[selected, columns]
    random_pick = rng.integers(0, trials.n_trials, size=n_splits)
    oos_of_random = oos_sharpe[random_pick, columns]

    # Annualise via the package's own definition rather than restating sqrt(f) here,
    # but apply it as a scalar factor so 12,870 splits stay a vector operation.
    factor = annualize_sharpe(1.0, trials.periods_per_year)
    is_annual = is_of_selected * factor
    oos_annual = oos_of_selected * factor
    random_annual = oos_of_random * factor

    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)),
        n_trials=trials.n_trials,
        n_blocks=n_blocks,
        n_splits=n_splits,
        exhaustive=exhaustive,
        logits=logits,
        relative_ranks=omega,
        selected_trials=np.asarray(selected, dtype=np.int64),
        is_sharpe_selected=is_annual,
        oos_sharpe_selected=oos_annual,
        oos_sharpe_random=random_annual,
        degradation=performance_degradation(is_annual, oos_annual),
        probability_of_loss=float(np.mean(oos_of_selected < 0.0)),
        dominance_fraction=_dominance_fraction(oos_annual, random_annual),
        n_degenerate_subsamples=int(np.sum(is_degenerate) + np.sum(oos_degenerate)),
    )
