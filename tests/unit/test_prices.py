"""Price loading and caching, verified without touching the network.

The downloader is injected, so every branch of the caching and validation logic is
testable offline. Exactly one test in this file is allowed to reach a vendor, and
it is marked ``network`` so CI never runs it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from luckdetector.exceptions import DataValidationError, InsufficientDataError
from luckdetector.io.prices import PriceHistory, cache_path, load_prices
from luckdetector.mining import mine, synthetic_prices


class RecordingDownloader:
    """A stand-in that counts calls, so cache hits are observable."""

    def __init__(self, n_periods: int = 800) -> None:
        self.calls = 0
        self.n_periods = n_periods

    def __call__(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        self.calls += 1
        index = pd.bdate_range(start, periods=self.n_periods)
        return pd.DataFrame(
            {"close": synthetic_prices(self.n_periods, seed=abs(hash(symbol)) % 1000)},
            index=index,
        )


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


class TestPriceHistory:
    def _history(self, n: int = 400, **kw: object) -> PriceHistory:
        defaults: dict[str, object] = {
            "symbol": "TEST",
            "dates": pd.bdate_range("2015-01-01", periods=n),
            "close": synthetic_prices(n, seed=1),
            "source": "test",
        }
        defaults.update(kw)
        return PriceHistory(**defaults)  # type: ignore[arg-type]

    def test_basic_properties(self) -> None:
        history = self._history()
        assert history.n_periods == 400
        assert history.returns().size == 399
        assert "to" in history.span

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(DataValidationError, match="lengths must match"):
            self._history(dates=pd.bdate_range("2015-01-01", periods=399))

    def test_rejects_short_history(self) -> None:
        with pytest.raises(InsufficientDataError, match="at least 300"):
            self._history(n=100)

    def test_rejects_non_positive_prices(self) -> None:
        bad = synthetic_prices(400, seed=1)
        bad[17] = 0.0
        with pytest.raises(DataValidationError, match="non-positive"):
            self._history(close=bad)

    def test_rejects_nan(self) -> None:
        bad = synthetic_prices(400, seed=1)
        bad[17] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            self._history(close=bad)

    def test_rejects_unsorted_dates(self) -> None:
        dates = pd.bdate_range("2015-01-01", periods=400)[::-1]
        with pytest.raises(DataValidationError, match="not sorted"):
            self._history(dates=pd.DatetimeIndex(dates))


class TestCaching:
    def test_first_call_downloads_second_uses_cache(self, cache: Path) -> None:
        downloader = RecordingDownloader()
        first = load_prices(
            "SPY", start="2015-01-01", end="2020-01-01", cache_dir=cache, downloader=downloader
        )
        second = load_prices(
            "SPY", start="2015-01-01", end="2020-01-01", cache_dir=cache, downloader=downloader
        )
        assert downloader.calls == 1
        assert first.source == "download"
        assert second.source == "cache"
        np.testing.assert_allclose(first.close, second.close)

    def test_refresh_forces_a_new_download(self, cache: Path) -> None:
        downloader = RecordingDownloader()
        load_prices(
            "SPY", start="2015-01-01", end="2020-01-01", cache_dir=cache, downloader=downloader
        )
        load_prices(
            "SPY",
            start="2015-01-01",
            end="2020-01-01",
            cache_dir=cache,
            downloader=downloader,
            refresh=True,
        )
        assert downloader.calls == 2

    def test_different_ranges_cache_separately(self, cache: Path) -> None:
        downloader = RecordingDownloader()
        load_prices(
            "SPY", start="2015-01-01", end="2020-01-01", cache_dir=cache, downloader=downloader
        )
        load_prices(
            "SPY", start="2016-01-01", end="2020-01-01", cache_dir=cache, downloader=downloader
        )
        assert downloader.calls == 2
        assert len(list(cache.glob("*.csv"))) == 2

    def test_cache_path_is_filesystem_safe(self) -> None:
        path = cache_path("BRK/B", "2015-01-01", "2020-01-01", Path("/tmp"))
        assert "/" not in path.name
        assert path.name.startswith("BRK_B")

    def test_rejects_downloader_without_close_column(self, cache: Path) -> None:
        def bad(symbol: str, start: str, end: str) -> pd.DataFrame:
            return pd.DataFrame(
                {"price": [1.0, 2.0]}, index=pd.bdate_range("2020-01-01", periods=2)
            )

        with pytest.raises(DataValidationError, match="expected a 'close' column"):
            load_prices("X", start="2015-01-01", end="2020-01-01", cache_dir=cache, downloader=bad)

    def test_loaded_prices_drive_the_miner(self, cache: Path) -> None:
        """End to end: whatever comes out of the loader must be minable."""
        history = load_prices(
            "SPY",
            start="2015-01-01",
            end="2020-01-01",
            cache_dir=cache,
            downloader=RecordingDownloader(1200),
        )
        result = mine(history.close, cost_bps=1.0)
        assert result.n_trials >= 150
        assert result.trials.n_periods == history.n_periods - 1


@pytest.mark.network
def test_real_download_smoke(tmp_path: Path) -> None:
    """The one test allowed to touch a vendor. Never runs in CI."""
    history = load_prices("SPY", start="2015-01-01", end="2020-01-01", cache_dir=tmp_path)
    assert history.n_periods > 1000
    assert history.source == "download"
    assert 50.0 < float(history.close.mean()) < 1000.0
