"""Loader behaviour, especially the failure modes we deliberately refuse to paper over."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from luckdetector.exceptions import DataValidationError
from luckdetector.io import loaders


@pytest.fixture
def returns_csv(tmp_path: Path, rng: np.random.Generator) -> Path:
    dates = pd.bdate_range("2020-01-01", periods=500)
    frame = pd.DataFrame({"date": dates, "strategy": rng.normal(0.0004, 0.01, 500)})
    path = tmp_path / "returns.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def trials_csv(tmp_path: Path, rng: np.random.Generator) -> Path:
    dates = pd.bdate_range("2020-01-01", periods=300)
    frame = pd.DataFrame(
        rng.normal(0, 0.01, (300, 8)),
        columns=[f"ma_{i}" for i in range(8)],
        index=dates,
    )
    frame.index.name = "date"
    path = tmp_path / "trials.csv"
    frame.to_csv(path)
    return path


class TestFrequencyInference:
    @pytest.mark.parametrize(
        ("freq", "expected"),
        [("B", 252), ("W", 52), ("ME", 12), ("QE", 4), ("YE", 1)],
    )
    def test_infers_common_frequencies(self, freq: str, expected: int) -> None:
        index = pd.date_range("2015-01-01", periods=60, freq=freq)
        assert loaders.infer_periods_per_year(index) == expected

    def test_survives_gaps(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=100).delete([10, 11, 40])
        assert loaders.infer_periods_per_year(index) == 252

    def test_rejects_spacing_coarser_than_annual(self) -> None:
        index = pd.DatetimeIndex(["2000-01-01", "2005-01-01", "2010-01-01"])
        with pytest.raises(DataValidationError, match="coarser than annual"):
            loaders.infer_periods_per_year(index)

    def test_needs_enough_timestamps(self) -> None:
        with pytest.raises(DataValidationError, match="at least 3"):
            loaders.infer_periods_per_year(pd.DatetimeIndex(["2020-01-01", "2020-01-02"]))


class TestPricesToReturns:
    def test_simple_returns(self) -> None:
        out = loaders.returns_from_prices(np.array([100.0, 110.0, 99.0]))
        np.testing.assert_allclose(out, [0.10, -0.10])

    def test_log_returns(self) -> None:
        out = loaders.returns_from_prices(np.array([100.0, 110.0]), log=True)
        np.testing.assert_allclose(out, [np.log(1.1)])

    def test_output_is_one_shorter(self, rng: np.random.Generator) -> None:
        prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, 250))
        assert loaders.returns_from_prices(prices).size == 249

    def test_rejects_non_positive_prices(self) -> None:
        with pytest.raises(DataValidationError, match="non-positive"):
            loaders.returns_from_prices(np.array([100.0, 0.0, 90.0]))

    def test_two_dimensional_prices_convert_column_wise(self) -> None:
        prices = np.array([[100.0, 50.0], [110.0, 45.0]])
        np.testing.assert_allclose(loaders.returns_from_prices(prices), [[0.10, -0.10]])

    def test_rejects_three_dimensional_input(self) -> None:
        with pytest.raises(DataValidationError, match="1-D or 2-D"):
            loaders.returns_from_prices(np.ones((2, 2, 2)))


class TestLoadReturns:
    def test_loads_and_infers_frequency(self, returns_csv: Path) -> None:
        series = loaders.load_returns_csv(returns_csv, date_column="date")
        assert series.n_periods == 500
        assert series.periods_per_year == 252
        assert series.name == "strategy"

    def test_requires_column_when_ambiguous(self, tmp_path: Path) -> None:
        path = tmp_path / "two.csv"
        pd.DataFrame({"a": [0.01, 0.02], "b": [0.03, 0.04]}).to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="pass column="):
            loaders.load_returns_csv(path)

    def test_named_column_is_honoured(self, tmp_path: Path) -> None:
        path = tmp_path / "two.csv"
        pd.DataFrame({"a": [0.01, 0.02], "b": [0.03, 0.04]}).to_csv(path, index=False)
        series = loaders.load_returns_csv(path, column="b")
        np.testing.assert_allclose(series.values, [0.03, 0.04])

    def test_unknown_column_raises(self, returns_csv: Path) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            loaders.load_returns_csv(returns_csv, column="nope", date_column="date")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="No such file"):
            loaders.load_returns_csv(tmp_path / "ghost.csv")

    def test_nan_policy_raise_is_default(self, tmp_path: Path) -> None:
        path = tmp_path / "nan.csv"
        pd.DataFrame({"r": [0.01, np.nan, 0.02]}).to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="missing value"):
            loaders.load_returns_csv(path)

    def test_nan_policy_drop(self, tmp_path: Path) -> None:
        path = tmp_path / "nan.csv"
        pd.DataFrame({"r": [0.01, np.nan, 0.02]}).to_csv(path, index=False)
        assert loaders.load_returns_csv(path, nan_policy="drop").n_periods == 2

    def test_catches_percentage_units_bug(self, tmp_path: Path, rng: np.random.Generator) -> None:
        path = tmp_path / "pct.csv"
        pd.DataFrame({"r": rng.normal(0.04, 1.0, 300)}).to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="percentages"):
            loaders.load_returns_csv(path)

    def test_price_column_converted(self, tmp_path: Path, rng: np.random.Generator) -> None:
        prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 400))
        path = tmp_path / "prices.csv"
        pd.DataFrame({"close": prices}).to_csv(path, index=False)
        series = loaders.load_returns_csv(path, are_prices=True)
        assert series.n_periods == 399
        assert abs(float(np.std(series.values)) - 0.01) < 0.003


class TestLoadTrials:
    def test_columns_orientation(self, trials_csv: Path) -> None:
        matrix = loaders.load_trials_csv(trials_csv, date_column="date")
        assert matrix.shape == (8, 300)
        assert matrix.labels[0] == "ma_0"
        assert matrix.periods_per_year == 252

    def test_rows_orientation(self, rng: np.random.Generator) -> None:
        frame = pd.DataFrame(rng.normal(0, 0.01, (5, 120)))
        matrix = loaders.trial_matrix_from_frame(frame, orient="rows", periods_per_year=252)
        assert matrix.shape == (5, 120)

    def test_rejects_non_numeric_only_frame(self) -> None:
        frame = pd.DataFrame({"note": ["a", "b", "c"]})
        with pytest.raises(DataValidationError, match="No numeric columns"):
            loaders.trial_matrix_from_frame(frame)
