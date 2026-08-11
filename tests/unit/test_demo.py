"""Demo data resolution: cache, then download, then refuse — and never fall back.

Every test here runs through the injectable downloader already in
``io/prices.py``, so nothing touches the network. The refusal tests are the
important ones: the failure mode being guarded against is a demo that quietly
substitutes a random number generator for a market and then prints the same
confident narrative, which is the exact error this package exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from luckdetector.exceptions import DataValidationError, LuckDetectorError
from luckdetector.report.demo import (
    DEMO_SEED,
    cached_ranges,
    planted_edge_analysis,
    planted_edge_trials,
    resolve_demo_prices,
    run_demo,
)


def write_cache(directory: Path, name: str, *, n: int = 400) -> Path:
    """A cache file in exactly the layout ``load_prices`` writes and reads."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    frame = pd.DataFrame({"close": prices}, index=pd.bdate_range("2010-01-04", periods=n))
    path = directory / name
    frame.to_csv(path)
    return path


def refusing_downloader(symbol: str, start: str, end: str) -> pd.DataFrame:
    raise DataValidationError("no network in tests")


class TestCachedRanges:
    def test_finds_ranges_for_the_symbol(self, tmp_path: Path) -> None:
        write_cache(tmp_path, "SPY_2010-01-01_2020-01-01.csv")
        write_cache(tmp_path, "SPY_2010-01-01_2026-08-10.csv")
        write_cache(tmp_path, "QQQ_2010-01-01_2026-08-10.csv")
        assert cached_ranges("SPY", tmp_path) == [
            ("2010-01-01", "2020-01-01"),
            ("2010-01-01", "2026-08-10"),
        ]

    def test_sorts_so_the_latest_end_is_last(self, tmp_path: Path) -> None:
        write_cache(tmp_path, "SPY_2010-01-01_2026-08-10.csv")
        write_cache(tmp_path, "SPY_2010-01-01_2011-01-01.csv")
        assert cached_ranges("SPY", tmp_path)[-1][1] == "2026-08-10"

    def test_filters_on_start_when_asked(self, tmp_path: Path) -> None:
        write_cache(tmp_path, "SPY_2010-01-01_2026-08-10.csv")
        write_cache(tmp_path, "SPY_2015-01-01_2026-08-10.csv")
        assert cached_ranges("SPY", tmp_path, start="2015-01-01") == [
            ("2015-01-01", "2026-08-10")
        ]

    def test_ignores_files_that_are_not_cache_entries(self, tmp_path: Path) -> None:
        write_cache(tmp_path, "SPY_2010-01-01_2026-08-10.csv")
        (tmp_path / "notes.csv").write_text("not a cache file")
        (tmp_path / "SPY_garbage.csv").write_text("nor this")
        assert len(cached_ranges("SPY", tmp_path)) == 1

    def test_a_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert cached_ranges("SPY", tmp_path / "nope") == []


class TestResolveDemoPrices:
    def test_uses_the_cache_without_downloading(self, tmp_path: Path) -> None:
        write_cache(tmp_path, "SPY_2010-01-01_2026-08-10.csv")
        history = resolve_demo_prices(
            cache_dirs=(tmp_path,), downloader=refusing_downloader
        )
        assert history.source == "cache"

    def test_the_cache_hits_even_though_the_key_contains_todays_date(
        self, tmp_path: Path
    ) -> None:
        """The wrinkle that makes exact-key lookup wrong, pinned as a test.

        ``cache_path`` keys on ``(symbol, start, end)`` and ``end`` defaults to
        today, so a lookup by exact key misses a file written yesterday. The
        cached range here ends in the past and must still resolve, or "use the
        cache if present" would be false on every day but one.
        """
        write_cache(tmp_path, "SPY_2010-01-01_2011-06-30.csv")
        history = resolve_demo_prices(
            cache_dirs=(tmp_path,), downloader=refusing_downloader
        )
        assert history.source == "cache"

    def test_prefers_the_most_recent_cached_range(self, tmp_path: Path) -> None:
        write_cache(tmp_path, "SPY_2010-01-01_2011-01-01.csv", n=350)
        write_cache(tmp_path, "SPY_2010-01-01_2026-08-10.csv", n=500)
        history = resolve_demo_prices(
            cache_dirs=(tmp_path,), downloader=refusing_downloader
        )
        assert history.n_periods == 500

    def test_searches_cache_directories_in_order(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        write_cache(first, "SPY_2010-01-01_2020-01-01.csv", n=400)
        write_cache(second, "SPY_2010-01-01_2026-08-10.csv", n=500)
        history = resolve_demo_prices(
            cache_dirs=(first, second), downloader=refusing_downloader
        )
        assert history.n_periods == 400

    def test_downloads_when_nothing_is_cached(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def downloader(symbol: str, start: str, end: str) -> pd.DataFrame:
            calls.append(symbol)
            rng = np.random.default_rng(1)
            prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 400)))
            return pd.DataFrame(
                {"close": prices}, index=pd.bdate_range("2010-01-04", periods=400)
            )

        history = resolve_demo_prices(cache_dirs=(tmp_path,), downloader=downloader)
        assert history.source == "download"
        assert calls == ["SPY"]

    def test_refuses_loudly_when_the_download_fails(self, tmp_path: Path) -> None:
        """It must fail, not fall back. The message has to name the offline flag."""
        with pytest.raises(LuckDetectorError) as excinfo:
            resolve_demo_prices(cache_dirs=(tmp_path,), downloader=refusing_downloader)
        message = str(excinfo.value)
        assert "luckdet demo --offline" in message
        assert "SYNTHETIC" in message
        assert "will not silently substitute" in message

    def test_refuses_when_downloading_is_disabled(self, tmp_path: Path) -> None:
        with pytest.raises(LuckDetectorError, match="--offline"):
            resolve_demo_prices(cache_dirs=(tmp_path,), allow_download=False)


class TestPlantedEdge:
    def test_the_edge_is_in_the_first_variants_only(self) -> None:
        trials = planted_edge_trials()
        means = trials.values.mean(axis=1)
        assert means[:5].mean() > means[5:].mean()
        assert list(trials.labels[:2]) == ["edge_0", "edge_1"]

    def test_is_reproducible(self) -> None:
        np.testing.assert_array_equal(
            planted_edge_trials(DEMO_SEED).values, planted_edge_trials(DEMO_SEED).values
        )

    def test_is_detected_as_skill(self) -> None:
        """The control half of the demo. If this goes flaky, the effect size moved."""
        analysis = planted_edge_analysis(n_resamples=300)
        assert analysis.label == "LIKELY_SKILL"

    def test_is_labelled_synthetic(self) -> None:
        assert planted_edge_analysis(n_resamples=100).synthetic


class TestRunDemo:
    def test_offline_needs_no_data_and_labels_everything_synthetic(self) -> None:
        result = run_demo(offline=True, n_resamples=100)
        assert result.real.synthetic
        assert result.planted.synthetic
        assert "not a market" in result.real.provenance

    def test_both_halves_are_produced(self, tmp_path: Path) -> None:
        write_cache(tmp_path, "SPY_2010-01-01_2026-08-10.csv", n=600)
        result = run_demo(cache_dirs=(tmp_path,), n_resamples=100)
        assert result.real.n_trials > 1
        assert result.planted.label == "LIKELY_SKILL"
        assert len(result.labels) == 2

    def test_real_half_is_not_labelled_synthetic(self, tmp_path: Path) -> None:
        write_cache(tmp_path, "SPY_2010-01-01_2026-08-10.csv", n=600)
        result = run_demo(cache_dirs=(tmp_path,), n_resamples=100)
        assert not result.real.synthetic
