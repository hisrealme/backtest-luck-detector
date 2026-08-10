"""Signal families and their parameter grids.

The point of this module is *not* to find alpha. It is to manufacture the exact
situation the rest of the package exists to diagnose: a few hundred plausible,
highly correlated strategy variants, from which someone will inevitably keep the
best one.

Look-ahead bias
---------------
Every function here returns positions aligned so that ``position[t]`` is decided
using prices up to and including ``prices[t]``, and earns ``returns[t]``, which is
the move from ``prices[t]`` to ``prices[t+1]``. Positions are therefore length
``T-1`` for ``T`` prices. Getting this off by one is the classic way to invent an
edge that does not exist, so :func:`luckdetector.mining.signals` is tested against
a deliberately constructed look-ahead trap.

During an indicator's warm-up period the position is flat, not the first defined
value carried backwards — the strategy genuinely could not have traded yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import DataValidationError, InsufficientDataError
from ..types import FloatArray

__all__ = [
    "SignalSet",
    "default_grid",
    "donchian_breakout",
    "moving_average_crossover",
    "rsi_reversion",
    "time_series_momentum",
]


@dataclass(frozen=True)
class SignalSet:
    """A family of strategy variants: labels plus their position matrix.

    ``positions`` has shape ``(n_variants, T-1)`` for ``T`` input prices, holding
    values in ``{-1, 0, +1}``.
    """

    labels: list[str]
    positions: FloatArray

    def __post_init__(self) -> None:
        if len(self.labels) != self.positions.shape[0]:
            raise DataValidationError(
                f"{len(self.labels)} labels for {self.positions.shape[0]} variants."
            )

    @property
    def n_variants(self) -> int:
        return int(self.positions.shape[0])

    def __len__(self) -> int:
        return self.n_variants


def _validate(prices: FloatArray, needed: int) -> FloatArray:
    arr = np.asarray(prices, dtype=np.float64).ravel()
    if arr.size < needed + 2:
        raise InsufficientDataError(
            f"Need at least {needed + 2} prices for this indicator, got {arr.size}."
        )
    if np.any(arr <= 0.0):
        raise DataValidationError("Price series contains non-positive values.")
    return arr


def _rolling_mean(x: FloatArray, window: int) -> FloatArray:
    """Rolling mean with NaN warm-up, same length as ``x``."""
    out = np.full(x.size, np.nan)
    if window > x.size:
        return out
    cumulative = np.concatenate([[0.0], np.cumsum(x)])
    out[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return out


def _rolling_extreme(x: FloatArray, window: int, *, maximum: bool) -> FloatArray:
    out = np.full(x.size, np.nan)
    if window > x.size:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(x, window)
    out[window - 1 :] = windows.max(axis=1) if maximum else windows.min(axis=1)
    return out


def _to_positions(indicator_signal: FloatArray) -> FloatArray:
    """Drop the final signal (nothing left to earn) and flatten NaN warm-up to 0."""
    positions = indicator_signal[:-1].astype(np.float64)
    return np.nan_to_num(positions, nan=0.0)


def moving_average_crossover(
    prices: FloatArray,
    fast_windows: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80),
    slow_windows: tuple[int, ...] = (50, 75, 100, 125, 150, 175, 200, 225, 250, 300),
) -> SignalSet:
    """Long when the fast moving average is above the slow one, short otherwise.

    The canonical over-fitted strategy family: dozens of parameter pairs, all
    tracking the same underlying trend signal, all mutually correlated.
    """
    arr = _validate(prices, max(slow_windows))
    labels: list[str] = []
    rows: list[FloatArray] = []
    means = {w: _rolling_mean(arr, w) for w in set(fast_windows) | set(slow_windows)}

    for fast in fast_windows:
        for slow in slow_windows:
            if fast >= slow:
                continue
            signal = np.sign(means[fast] - means[slow])
            signal[np.isnan(means[slow])] = np.nan
            labels.append(f"MA({fast},{slow})")
            rows.append(_to_positions(signal))
    return SignalSet(labels, np.array(rows))


def time_series_momentum(
    prices: FloatArray,
    lookbacks: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 60, 90, 120, 150, 180, 210, 250),
) -> SignalSet:
    """Long if the trailing return over ``lookback`` periods was positive."""
    arr = _validate(prices, max(lookbacks))
    labels: list[str] = []
    rows: list[FloatArray] = []
    for lookback in lookbacks:
        signal = np.full(arr.size, np.nan)
        signal[lookback:] = np.sign(arr[lookback:] / arr[:-lookback] - 1.0)
        labels.append(f"MOM({lookback})")
        rows.append(_to_positions(signal))
    return SignalSet(labels, np.array(rows))


def rsi_reversion(
    prices: FloatArray,
    windows: tuple[int, ...] = (5, 7, 10, 14, 21, 28),
    thresholds: tuple[int, ...] = (15, 20, 25, 30, 35, 40),
) -> SignalSet:
    """Buy oversold, sell overbought, using a simple-average RSI.

    Uses a simple rolling mean of gains and losses rather than Wilder smoothing.
    The distinction is immaterial here: the goal is a family of plausible rules,
    not a faithful reproduction of any particular trading platform.
    """
    arr = _validate(prices, max(windows) + 1)
    deltas = np.diff(arr, prepend=arr[0])
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)

    labels: list[str] = []
    rows: list[FloatArray] = []
    for window in windows:
        avg_gain = _rolling_mean(gains, window)
        avg_loss = _rolling_mean(losses, window)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.inf)
            rsi = 100.0 - 100.0 / (1.0 + rs)
        rsi[np.isnan(avg_gain)] = np.nan

        for low in thresholds:
            high = 100 - low
            signal = np.where(rsi < low, 1.0, np.where(rsi > high, -1.0, 0.0))
            signal[np.isnan(rsi)] = np.nan
            labels.append(f"RSI({window},{low})")
            rows.append(_to_positions(signal))
    return SignalSet(labels, np.array(rows))


def donchian_breakout(
    prices: FloatArray,
    lookbacks: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 55, 70, 90, 120, 150, 200),
) -> SignalSet:
    """Long on a new ``lookback``-period high, short on a new low, else hold flat."""
    arr = _validate(prices, max(lookbacks))
    labels: list[str] = []
    rows: list[FloatArray] = []
    for lookback in lookbacks:
        highs = _rolling_extreme(arr, lookback, maximum=True)
        lows = _rolling_extreme(arr, lookback, maximum=False)
        signal = np.where(arr >= highs, 1.0, np.where(arr <= lows, -1.0, 0.0))
        signal[np.isnan(highs)] = np.nan
        labels.append(f"BRK({lookback})")
        rows.append(_to_positions(signal))
    return SignalSet(labels, np.array(rows))


def default_grid(prices: FloatArray) -> SignalSet:
    """Every family at its default parameters — 157 heavily correlated variants.

    This is the number that goes into the Deflated Sharpe Ratio as ``n_trials``,
    and the reason the effective-trial clustering in
    :mod:`luckdetector.stats.dsr` matters: these families overlap heavily.
    """
    families = [
        moving_average_crossover(prices),
        time_series_momentum(prices),
        rsi_reversion(prices),
        donchian_breakout(prices),
    ]
    labels = [label for family in families for label in family.labels]
    positions = np.vstack([family.positions for family in families])
    return SignalSet(labels, positions)
