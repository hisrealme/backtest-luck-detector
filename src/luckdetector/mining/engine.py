"""Vectorised backtester: turn a grid of signals into a ``TrialMatrix``.

This is deliberately the least interesting code in the package. It exists only to
manufacture realistic trial matrices for the statistics to judge — it is not a
backtesting framework, and it models exactly one friction (a flat per-turnover
cost in basis points).

What it *is* careful about
--------------------------
* **No look-ahead.** Position ``t`` earns return ``t``, where return ``t`` is the
  move from price ``t`` to price ``t+1``.
* **Costs on turnover, not on holdings.** Charged on ``|Δposition|``, so a
  strategy that flips from long to short pays twice — which is what makes
  high-frequency parameter settings look bad, correctly.
* **Costs charged from the first trade.** Entering the initial position is a
  trade, so turnover at ``t=0`` is ``|position[0]|``, not zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import DataValidationError
from ..types import FloatArray, TrialMatrix
from .signals import SignalSet, default_grid

__all__ = ["MiningResult", "backtest", "mine", "synthetic_prices"]


def synthetic_prices(
    n_periods: int = 3780,
    *,
    seed: int = 0,
    annual_drift: float = 0.07,
    annual_vol: float = 0.16,
    vol_persistence: float = 0.94,
    vol_of_vol: float = 0.25,
    periods_per_year: int = 252,
) -> FloatArray:
    """A price path with realistic volatility clustering, for offline use.

    **This is synthetic data, and the package never pretends otherwise.** Real
    market data requires a network call, which tests must not make; bundling a
    file of genuine prices and calling it a fixture would also raise licensing
    questions. So the offline demo runs on a GARCH-like path: log-volatility
    follows an AR(1), which reproduces the volatility clustering and mild fat
    tails that make block bootstraps necessary in the first place.

    The default of 3780 periods is fifteen years of daily data.
    """
    if n_periods < 2:
        raise DataValidationError(f"n_periods must be at least 2, got {n_periods}.")

    rng = np.random.default_rng(seed)
    drift = annual_drift / periods_per_year
    base_log_vol = np.log(annual_vol / np.sqrt(periods_per_year))

    log_vol = np.empty(n_periods)
    log_vol[0] = base_log_vol
    shocks = rng.normal(0.0, vol_of_vol, n_periods)
    for t in range(1, n_periods):
        log_vol[t] = base_log_vol + vol_persistence * (log_vol[t - 1] - base_log_vol) + shocks[t]

    returns = drift + np.exp(log_vol) * rng.standard_normal(n_periods)
    return np.asarray(100.0 * np.exp(np.cumsum(returns)), dtype=np.float64)


def backtest(
    prices: FloatArray,
    positions: FloatArray,
    *,
    cost_bps: float = 1.0,
) -> FloatArray:
    """Apply a position matrix to a price series, net of turnover costs.

    Parameters
    ----------
    prices:
        Length ``T``.
    positions:
        Shape ``(n_variants, T-1)``, values typically in ``{-1, 0, +1}``.
    cost_bps:
        Cost per unit of turnover, in basis points. ``1.0`` means a full
        long-to-flat trade costs 1bp of notional.

    Returns
    -------
    FloatArray
        Shape ``(n_variants, T-1)`` of net strategy returns.
    """
    price_array = np.asarray(prices, dtype=np.float64).ravel()
    position_array = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    asset_returns = price_array[1:] / price_array[:-1] - 1.0

    if position_array.shape[1] != asset_returns.size:
        raise DataValidationError(
            f"positions has {position_array.shape[1]} columns but {price_array.size} prices "
            f"imply {asset_returns.size} returns. Positions must be length T-1 so that "
            "position t earns return t; anything else is a look-ahead bug."
        )
    if cost_bps < 0:
        raise DataValidationError(f"cost_bps must be non-negative, got {cost_bps}.")

    gross = position_array * asset_returns
    # Turnover at t=0 is the cost of establishing the initial position.
    turnover = np.abs(np.diff(position_array, axis=1, prepend=0.0))
    return np.asarray(gross - turnover * (cost_bps / 10_000.0), dtype=np.float64)


@dataclass(frozen=True)
class MiningResult:
    """Every strategy that was tried, plus the benchmark they should be judged against."""

    trials: TrialMatrix
    buy_and_hold: FloatArray
    cost_bps: float

    @property
    def n_trials(self) -> int:
        return self.trials.n_trials


def mine(
    prices: FloatArray,
    *,
    signals: SignalSet | None = None,
    cost_bps: float = 1.0,
    periods_per_year: int = 252,
) -> MiningResult:
    """Brute-force a grid of strategies over one price series.

    Returns *all* of them, not the winner. Keeping the losers is the entire point:
    :func:`luckdetector.stats.dsr.deflated_sharpe_ratio_from_trials` and the PBO
    machinery both need the full family to say anything meaningful, and a
    backtester who discards the failures has destroyed the evidence required to
    judge the survivor.
    """
    price_array = np.asarray(prices, dtype=np.float64).ravel()
    grid = default_grid(price_array) if signals is None else signals
    returns = backtest(price_array, grid.positions, cost_bps=cost_bps)

    # Drop variants that never traded — an all-flat strategy has zero volatility
    # and would break every ratio downstream.
    traded = np.abs(grid.positions).sum(axis=1) > 0
    if not traded.any():
        raise DataValidationError("No strategy in the grid ever took a position.")

    trials = TrialMatrix(
        values=returns[traded],
        periods_per_year=periods_per_year,
        labels=[label for label, keep in zip(grid.labels, traded, strict=True) if keep],
    )
    buy_and_hold = price_array[1:] / price_array[:-1] - 1.0
    return MiningResult(trials=trials, buy_and_hold=buy_and_hold, cost_bps=cost_bps)
