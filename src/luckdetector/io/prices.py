"""Real market data, with on-disk caching.

Where synthetic data belongs, and where it does not
---------------------------------------------------
**Unit tests**: synthetic, always. A test suite that reaches the network is
non-deterministic, slow, fails when a vendor has an outage, and silently changes
its answers when a provider revises history. Every test in this package runs
offline against seeded data, and that is a deliberate engineering choice rather
than a limitation.

**Demonstrations and headline results**: real data, always. "We indicted a
moving-average crossover on fifteen years of SPY" is a claim about the world.
"We indicted one on prices we generated ourselves" is a claim about our random
number generator. Only the first is worth reporting.

This module supplies the second half. It is an optional extra
(``pip install -e ".[data]"``) because nothing in the core statistics needs it,
and its tests are marked ``network`` so CI never depends on a vendor being up.

Caching
-------
Downloads are cached as CSV under ``~/.cache/luckdetector`` keyed by symbol and
date range. Re-running a demo therefore hits the network once, and every
subsequent run is reproducible even offline — which matters, because a result you
cannot reproduce next month is not a result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError, InsufficientDataError, LuckDetectorError
from ..types import FloatArray

__all__ = [
    "DEFAULT_CACHE_DIR",
    "Downloader",
    "PriceHistory",
    "cache_path",
    "load_prices",
    "yfinance_downloader",
]

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "luckdetector"

#: ``(symbol, start, end) -> DataFrame`` with a DatetimeIndex and a ``close`` column.
Downloader = Callable[[str, str, str], pd.DataFrame]

MIN_OBSERVATIONS = 300


@dataclass(frozen=True)
class PriceHistory:
    """A validated close-price series with its provenance attached.

    ``source`` records where the data came from — ``"download"``, ``"cache"``, or
    the name of an injected test double. A result whose provenance is unknown is
    not reportable, so this travels with the numbers.
    """

    symbol: str
    dates: pd.DatetimeIndex
    close: FloatArray
    source: str

    def __post_init__(self) -> None:
        if self.close.ndim != 1:
            raise DataValidationError(f"close must be 1-D, got shape {self.close.shape}.")
        if len(self.dates) != self.close.size:
            raise DataValidationError(
                f"{len(self.dates)} dates for {self.close.size} prices; lengths must match."
            )
        if self.close.size < MIN_OBSERVATIONS:
            raise InsufficientDataError(
                f"{self.symbol}: got {self.close.size} observations, need at least "
                f"{MIN_OBSERVATIONS}. The slowest signal in the default grid uses a "
                "300-period window."
            )
        if not np.all(np.isfinite(self.close)):
            raise DataValidationError(f"{self.symbol}: price series contains NaN or inf.")
        if np.any(self.close <= 0.0):
            raise DataValidationError(f"{self.symbol}: price series contains non-positive values.")
        if not self.dates.is_monotonic_increasing:
            raise DataValidationError(f"{self.symbol}: dates are not sorted ascending.")

    @property
    def n_periods(self) -> int:
        return int(self.close.size)

    @property
    def span(self) -> str:
        return f"{self.dates[0]:%Y-%m-%d} to {self.dates[-1]:%Y-%m-%d}"

    def returns(self) -> FloatArray:
        """Simple returns, one shorter than the price series."""
        return np.asarray(self.close[1:] / self.close[:-1] - 1.0, dtype=np.float64)


def yfinance_downloader(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted closes from Yahoo Finance.

    Adjusted rather than raw closes: a strategy backtested on unadjusted prices
    sees a large negative return on every ex-dividend date that never happened.
    """
    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise LuckDetectorError(
            "Downloading prices needs yfinance, which is an optional extra. "
            'Install it with: pip install -e ".[data]"'
        ) from exc

    frame = yfinance.download(
        symbol, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False
    )
    if frame is None or frame.empty:
        raise DataValidationError(
            f"No data returned for {symbol!r} between {start} and {end}. Check the ticker."
        )
    return pd.DataFrame(
        {"close": frame["Close"].astype(float)}, index=pd.DatetimeIndex(frame.index)
    )


def cache_path(symbol: str, start: str, end: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Deterministic cache location for one symbol and date range."""
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in symbol.upper())
    return cache_dir / f"{safe}_{start}_{end}.csv"


def load_prices(
    symbol: str = "SPY",
    *,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    downloader: Downloader | None = None,
    refresh: bool = False,
) -> PriceHistory:
    """Load close prices, downloading only if they are not already cached.

    Parameters
    ----------
    downloader:
        Injected for testability. Defaults to :func:`yfinance_downloader`; tests
        pass a deterministic stand-in so the caching logic can be verified without
        touching the network.
    refresh:
        Bypass the cache and re-download.

    Examples
    --------
    >>> load_prices("SPY", start="2010-01-01")           # doctest: +SKIP
    PriceHistory(symbol='SPY', ..., source='download')
    """
    end = end or date.today().isoformat()
    path = cache_path(symbol, start, end, cache_dir)

    if path.exists() and not refresh:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        source = "cache"
    else:
        fetch = downloader or yfinance_downloader
        frame = fetch(symbol, start, end)
        if "close" not in frame.columns:
            raise DataValidationError(
                f"Downloader returned columns {list(frame.columns)}; expected a 'close' column."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
        source = "download"

    frame = frame.dropna().sort_index()
    return PriceHistory(
        symbol=symbol.upper(),
        dates=pd.DatetimeIndex(frame.index),
        close=np.asarray(frame["close"].to_numpy(), dtype=np.float64),
        source=source,
    )
