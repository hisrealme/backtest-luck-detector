"""Resampling engines: iid, circular block, and stationary bootstraps.

Why not just the iid bootstrap
------------------------------
Financial returns are not independent. Volatility clusters, momentum and
mean-reversion persist, and drawdowns arrive in runs. An iid bootstrap destroys
all of that, producing a null distribution that is far too narrow — which makes
mediocre strategies look significant. Block methods resample *runs* of
consecutive observations instead, preserving dependence up to the block length.

The three methods
-----------------
``iid``
    Draw ``n`` observations with replacement. Correct only for genuinely
    independent data. Included as a baseline and for testing.
``circular``
    Circular block bootstrap (Politis & Romano 1992). Fixed block length ``b``,
    wrapping at the end of the sample so every observation is equally likely to
    appear — unlike the naive moving-block bootstrap, which under-weights the
    edges.
``stationary``
    Stationary bootstrap (Politis & Romano 1994). Block lengths are geometric
    with mean ``b``, which makes the resampled series genuinely stationary rather
    than merely block-wise stationary. **This is the method White's Reality Check
    and Hansen's SPA require**, so it is the default here.

Block length selection
----------------------
:func:`optimal_block_length` implements the automatic rule of Politis & White
(2004). Choosing ``b`` badly matters: too short destroys the dependence you were
trying to preserve, too long and every resample is a near-copy of the original.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..exceptions import InsufficientDataError
from ..types import FloatArray, IntArray, ReturnSeries
from .moments import as_return_series, is_effectively_constant

__all__ = [
    "BootstrapMethod",
    "BootstrapResult",
    "bootstrap_distribution",
    "bootstrap_indices",
    "circular_block_indices",
    "iid_indices",
    "optimal_block_length",
    "permutation_null",
    "stationary_indices",
]

BootstrapMethod = Literal["iid", "circular", "stationary"]


def _check(n: int, size: int) -> None:
    if n < 2:
        raise InsufficientDataError(f"Need at least 2 observations to resample, got {n}.")
    if size < 1:
        raise ValueError(f"size must be positive, got {size}.")


def iid_indices(n: int, size: int, rng: np.random.Generator) -> IntArray:
    """``(size, n)`` matrix of iid bootstrap indices."""
    _check(n, size)
    return np.asarray(rng.integers(0, n, size=(size, n)), dtype=np.int64)


def circular_block_indices(
    n: int,
    block_length: int,
    size: int,
    rng: np.random.Generator,
) -> IntArray:
    """``(size, n)`` indices for the circular block bootstrap.

    Blocks wrap around the end of the sample, so every observation appears in
    exactly ``block_length`` blocks and none is under-represented.
    """
    _check(n, size)
    if block_length < 1:
        raise ValueError(f"block_length must be at least 1, got {block_length}.")
    block_length = min(block_length, n)

    n_blocks = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=(size, n_blocks))
    offsets = np.arange(block_length)
    # (size, n_blocks, block_length) -> flatten to a series, wrapping modulo n
    drawn = (starts[:, :, None] + offsets[None, None, :]) % n
    return np.asarray(drawn.reshape(size, -1)[:, :n], dtype=np.int64)


def stationary_indices(
    n: int,
    mean_block_length: float,
    size: int,
    rng: np.random.Generator,
) -> IntArray:
    """``(size, n)`` indices for the stationary bootstrap (Politis & Romano 1994).

    At each step, with probability ``p = 1 / mean_block_length`` jump to a fresh
    uniformly random position; otherwise advance one step, wrapping at the end.
    Block lengths are therefore geometric with mean ``mean_block_length``, and the
    resampled series is stationary.
    """
    _check(n, size)
    if mean_block_length <= 0:
        raise ValueError(f"mean_block_length must be positive, got {mean_block_length}.")

    p = min(1.0, 1.0 / mean_block_length)
    indices = np.empty((size, n), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n, size=size)

    jump = rng.random((size, n - 1)) < p
    fresh = rng.integers(0, n, size=(size, n - 1))
    for t in range(1, n):
        advanced = (indices[:, t - 1] + 1) % n
        indices[:, t] = np.where(jump[:, t - 1], fresh[:, t - 1], advanced)
    return indices


def bootstrap_indices(
    n: int,
    size: int,
    rng: np.random.Generator,
    *,
    method: BootstrapMethod = "stationary",
    block_length: float | None = None,
    values: FloatArray | None = None,
) -> IntArray:
    """Dispatch to the requested resampling scheme.

    If ``block_length`` is omitted for a block method, it is chosen automatically
    from ``values`` via :func:`optimal_block_length`.
    """
    if method == "iid":
        return iid_indices(n, size, rng)

    if block_length is None:
        if values is None:
            raise ValueError(
                f"method={method!r} needs a block_length, or `values` from which to "
                "estimate one automatically."
            )
        block_length = optimal_block_length(values, method=method)

    if method == "circular":
        return circular_block_indices(n, max(1, round(block_length)), size, rng)
    if method == "stationary":
        return stationary_indices(n, block_length, size, rng)
    raise ValueError(f"Unknown bootstrap method {method!r}.")


def _flat_top_kernel(s: FloatArray) -> FloatArray:
    """Politis & White's trapezoidal kernel: 1 on [0, ½], tapering to 0 at 1."""
    abs_s = np.abs(s)
    return np.where(abs_s <= 0.5, 1.0, np.where(abs_s <= 1.0, 2.0 * (1.0 - abs_s), 0.0))


def optimal_block_length(
    values: ReturnSeries | FloatArray,
    *,
    method: BootstrapMethod = "stationary",
) -> float:
    """Automatic block-length selection, Politis & White (2004).

    Estimates the optimal expected block length by balancing the bias from
    truncating dependence against the variance from having few distinct blocks:

    .. math:: b_{opt} = \\left(\\frac{2 \\hat{G}^2}{\\hat{D}}\\right)^{1/3} n^{1/3}

    The correlogram lag ``m`` is chosen as the smallest lag beyond which the
    sample autocorrelations stay inside the :math:`2\\sqrt{\\log_{10}(n)/n}` band —
    i.e. the point where dependence becomes statistically indistinguishable from
    zero.

    Returns
    -------
    float
        Expected block length, clipped to ``[1, min(3*sqrt(n), n/3)]``.
    """
    series = as_return_series(values)
    n = series.n_periods
    # A constant series has no dependence structure to estimate. Without this
    # guard the whole calculation runs on ~1e-19 of floating-point residue and
    # returns a confident, meaningless block length (23.1 for 100 identical values).
    if n < 8 or is_effectively_constant(series):
        return 1.0

    x = series.values - series.values.mean()
    max_lag = int(min(math.ceil(math.sqrt(n)) + 20, n - 1))
    denom = float(np.dot(x, x)) / n
    if denom <= 0.0:
        return 1.0
    acov = np.array([float(np.dot(x[: n - k], x[k:])) / n for k in range(max_lag + 1)])
    acf = acov / denom

    # Correlogram rule: first lag m past which K_n successive autocorrelations
    # all fall inside the significance band.
    band = 2.0 * math.sqrt(math.log10(n) / n)
    window = max(5, int(math.sqrt(math.log10(n))))
    m = 0
    for lag in range(1, max_lag + 1):
        upper = min(lag + window, max_lag)
        if np.all(np.abs(acf[lag:upper]) < band):
            m = lag - 1
            break
    else:
        m = max_lag

    big_m = min(2 * max(m, 1), max_lag)
    lags = np.arange(-big_m, big_m + 1)
    weights = _flat_top_kernel(lags / big_m)
    two_sided = acov[np.abs(lags)]

    g_hat = float(np.sum(weights * np.abs(lags) * two_sided))
    spectral = float(np.sum(weights * two_sided))
    d_hat = 2.0 * spectral**2 if method == "stationary" else (4.0 / 3.0) * spectral**2
    if d_hat <= 0.0 or g_hat == 0.0:
        return 1.0

    b_opt = ((2.0 * g_hat**2) / d_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_max = min(3.0 * math.sqrt(n), n / 3.0)
    return float(np.clip(b_opt, 1.0, b_max))


@dataclass(frozen=True)
class BootstrapResult:
    """A bootstrap null distribution and the observed statistic it brackets."""

    observed: float
    distribution: FloatArray
    method: BootstrapMethod
    block_length: float | None

    @property
    def n_resamples(self) -> int:
        return int(self.distribution.size)

    def p_value(self, *, alternative: Literal["greater", "less"] = "greater") -> float:
        """Bootstrap p-value, with the observed value counted in.

        Adding one to numerator and denominator is not a rounding quirk — it
        prevents a p-value of exactly zero, which would claim more precision than
        ``n_resamples`` draws can support.
        """
        if alternative == "greater":
            extreme = int(np.sum(self.distribution >= self.observed))
        else:
            extreme = int(np.sum(self.distribution <= self.observed))
        return (extreme + 1) / (self.n_resamples + 1)

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Percentile interval of the bootstrap distribution."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie between 0 and 1, got {level}.")
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(self.distribution, [tail, 1.0 - tail])
        return float(low), float(high)


def bootstrap_distribution(
    values: ReturnSeries | FloatArray,
    statistic: Callable[[FloatArray], float],
    *,
    n_resamples: int = 1000,
    method: BootstrapMethod = "stationary",
    block_length: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> BootstrapResult:
    """Resample ``values`` and evaluate ``statistic`` on each resample.

    Parameters
    ----------
    statistic:
        Any function from a 1-D array to a float — ``np.mean``, a Sharpe ratio,
        a drawdown. Called once per resample.
    seed:
        An int, a ``Generator``, or ``None``. Passing an int makes the whole
        procedure byte-for-byte reproducible.
    """
    series = as_return_series(values)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    if method != "iid" and block_length is None:
        block_length = optimal_block_length(series, method=method)

    indices = bootstrap_indices(
        series.n_periods,
        n_resamples,
        rng,
        method=method,
        block_length=block_length,
        values=series.values,
    )
    resampled = series.values[indices]
    distribution = np.array([statistic(row) for row in resampled], dtype=np.float64)

    return BootstrapResult(
        observed=float(statistic(series.values)),
        distribution=distribution,
        method=method,
        block_length=None if method == "iid" else block_length,
    )


def permutation_null(
    values: ReturnSeries | FloatArray,
    statistic: Callable[[FloatArray], float],
    *,
    n_permutations: int = 1000,
    seed: int | np.random.Generator | None = None,
) -> BootstrapResult:
    """Null distribution from shuffling, which destroys timing but keeps the marginals.

    The right null for "does this strategy's *timing* add anything?" — a permuted
    series holds exactly the same returns in a different order, so any edge that
    survives shuffling was never about timing in the first place.
    """
    series = as_return_series(values)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    distribution = np.array(
        [statistic(rng.permutation(series.values)) for _ in range(n_permutations)],
        dtype=np.float64,
    )
    return BootstrapResult(
        observed=float(statistic(series.values)),
        distribution=distribution,
        method="iid",
        block_length=None,
    )
