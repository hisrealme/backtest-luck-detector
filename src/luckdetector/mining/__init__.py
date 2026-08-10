"""Strategy mining: manufacture the trial matrices the statistics need to judge."""

from __future__ import annotations

from .engine import MiningResult, backtest, mine, synthetic_prices
from .signals import (
    SignalSet,
    default_grid,
    donchian_breakout,
    moving_average_crossover,
    rsi_reversion,
    time_series_momentum,
)

__all__ = [
    "MiningResult",
    "SignalSet",
    "backtest",
    "default_grid",
    "donchian_breakout",
    "mine",
    "moving_average_crossover",
    "rsi_reversion",
    "synthetic_prices",
    "time_series_momentum",
]
