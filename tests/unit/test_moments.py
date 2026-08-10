"""Moment calculations, checked against hand computation and scipy/pandas."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from luckdetector.exceptions import DegenerateSeriesError
from luckdetector.stats import moments
from luckdetector.types import ReturnSeries

ReturnFactory = Callable[..., Any]


def _sharpes(trial_values: np.ndarray) -> np.ndarray:
    """Annualised Sharpe of every row in a trial matrix."""
    return np.array([moments.sharpe_ratio(row) for row in trial_values])


class TestSharpe:
    def test_matches_hand_computation(self, flat_returns: ReturnSeries) -> None:
        # values [0, .02, 0, .02]: mean .01, sample sd .011547 -> SR_period = .8660
        assert moments.sharpe_ratio(flat_returns, annualized=False) == pytest.approx(
            0.01 / np.std(flat_returns.values, ddof=1)
        )

    def test_annualisation_is_sqrt_time(self, noise_returns: ReturnSeries) -> None:
        per_period = moments.sharpe_ratio(noise_returns, annualized=False)
        annual = moments.sharpe_ratio(noise_returns, annualized=True)
        assert annual == pytest.approx(per_period * math.sqrt(252))

    def test_annualize_deannualize_round_trip(self) -> None:
        assert moments.deannualize_sharpe(
            moments.annualize_sharpe(0.05, 252), 252
        ) == pytest.approx(0.05)

    def test_matches_pandas_reference(self, noise_returns: ReturnSeries) -> None:
        series = pd.Series(noise_returns.values)
        expected = series.mean() / series.std(ddof=1) * math.sqrt(252)
        assert moments.sharpe_ratio(noise_returns) == pytest.approx(expected)

    def test_risk_free_rate_lowers_sharpe(self, skilled_returns: ReturnSeries) -> None:
        gross = moments.sharpe_ratio(skilled_returns)
        net = moments.sharpe_ratio(skilled_returns, risk_free_rate=0.04)
        assert net < gross

    def test_constant_series_raises(self) -> None:
        # np.std of identical values returns ~2e-19, not 0.0. Without a scale-aware
        # guard this series would report a Sharpe ratio of 4.6e15.
        with pytest.raises(DegenerateSeriesError, match="zero volatility"):
            moments.sharpe_ratio(ReturnSeries(np.full(100, 0.001)))

    def test_all_zero_series_raises(self) -> None:
        with pytest.raises(DegenerateSeriesError):
            moments.sharpe_ratio(ReturnSeries(np.zeros(100)))

    def test_skilled_beats_noise_on_average(self, make_trials: ReturnFactory) -> None:
        # A *single* draw proves nothing — see test_pure_noise_can_look_skilled below.
        # Averaged over 400 independent draws, the planted edge must show up.
        skilled = _sharpes(make_trials(1.0, n_trials=400, offset=100))
        noise = _sharpes(make_trials(0.0, n_trials=400, offset=200))
        assert skilled.mean() > noise.mean()
        assert float(skilled.mean()) == pytest.approx(1.0, abs=0.15)
        assert float(noise.mean()) == pytest.approx(0.0, abs=0.15)

    def test_sharpe_estimate_is_noisy_at_realistic_sample_sizes(
        self, make_trials: ReturnFactory, n_days: int
    ) -> None:
        """Sanity-check the standard error that every downstream test relies on."""
        sharpes = _sharpes(make_trials(0.0, n_trials=800, offset=300))
        expected_se = math.sqrt(252 / n_days)  # ~0.45 annualised over five years
        assert float(sharpes.std()) == pytest.approx(expected_se, rel=0.15)

    def test_pure_noise_can_look_skilled(self, make_trials: ReturnFactory, n_days: int) -> None:
        """The premise of the entire project, asserted as a fact about the data.

        Take 200 five-year backtests with **no edge at all**. The best of them posts an
        annualised Sharpe around 1.15 — a number most people would call a good strategy.

        The expected maximum of N zero-edge Sharpe estimates is approximately
        ``se * Phi^-1(1 - 1/N)``, which is the exact quantity the Deflated Sharpe Ratio
        uses as its benchmark in Phase 2. Averaging over 5 independent batches keeps the
        assertion about the phenomenon rather than about one lucky seed.
        """
        n_trials = 200
        batch_maxima = [
            _sharpes(make_trials(0.0, n_trials=n_trials, offset=400 + b)).max() for b in range(5)
        ]
        expected_max = math.sqrt(252 / n_days) * float(sps.norm.ppf(1 - 1 / n_trials))

        assert expected_max > 1.0  # the headline: pure noise "achieves" Sharpe > 1
        assert float(np.mean(batch_maxima)) == pytest.approx(expected_max, rel=0.25)
        assert min(batch_maxima) > 0.75


class TestHigherMoments:
    def test_skew_matches_scipy(self, noise_returns: ReturnSeries) -> None:
        assert moments.skewness(noise_returns) == pytest.approx(sps.skew(noise_returns.values))

    def test_kurtosis_is_raw_not_excess(self, rng: np.random.Generator) -> None:
        gaussian = ReturnSeries(rng.normal(0, 0.01, 200_000))
        assert moments.kurtosis(gaussian) == pytest.approx(3.0, abs=0.1)

    def test_detects_negative_skew(self, rng: np.random.Generator) -> None:
        # Many small gains, rare large losses: the classic "picking up pennies" shape.
        values = np.where(rng.random(10_000) < 0.98, 0.001, -0.05)
        assert moments.skewness(ReturnSeries(values)) < -3.0

    def test_fat_tails_raise_kurtosis(self, rng: np.random.Generator) -> None:
        fat = ReturnSeries(rng.standard_t(df=4, size=50_000) * 0.005)
        assert moments.kurtosis(fat) > 3.0


class TestDrawdownAndReturns:
    def test_max_drawdown_hand_computed(self) -> None:
        # 1.0 -> 1.1 -> 0.88 -> 0.968: trough is 0.88 against a peak of 1.1 = -20%
        series = ReturnSeries([0.10, -0.20, 0.10])
        assert moments.max_drawdown(series) == pytest.approx(-0.20)

    def test_max_drawdown_is_non_positive(self, noise_returns: ReturnSeries) -> None:
        assert moments.max_drawdown(noise_returns) <= 0.0

    def test_monotonic_series_has_no_drawdown(self) -> None:
        assert moments.max_drawdown(ReturnSeries(np.full(50, 0.01))) == pytest.approx(0.0)

    def test_geometric_mean_below_arithmetic(self, noise_returns: ReturnSeries) -> None:
        assert moments.mean_return(noise_returns, geometric=True) < moments.mean_return(
            noise_returns
        )

    def test_geometric_mean_undefined_after_total_loss(self) -> None:
        with pytest.raises(DegenerateSeriesError, match="geometric"):
            moments.mean_return(ReturnSeries([-1.0, 0.5]), geometric=True)

    def test_geometric_annualisation_compounds(self) -> None:
        assert moments.annualize_return(0.01, 12, geometric=True) == pytest.approx(1.01**12 - 1)

    def test_annualised_volatility(self, noise_returns: ReturnSeries) -> None:
        assert moments.volatility(noise_returns, annualized=True) == pytest.approx(
            0.01 * math.sqrt(252), rel=0.1
        )


class TestSummarize:
    def test_summary_is_internally_consistent(self, skilled_returns: ReturnSeries) -> None:
        summary = moments.summarize(skilled_returns)
        assert summary.sharpe_annual == pytest.approx(summary.sharpe_per_period * math.sqrt(252))
        assert summary.n_periods == 1260
        assert summary.years == pytest.approx(5.0)
        assert summary.name == "skilled"

    def test_as_dict_round_trips(self, noise_returns: ReturnSeries) -> None:
        payload = moments.summarize(noise_returns).as_dict()
        assert payload["periods_per_year"] == 252
        assert set(payload) >= {"sharpe_annual", "skewness", "kurtosis", "max_drawdown"}

    def test_accepts_raw_array(self, rng: np.random.Generator) -> None:
        assert moments.summarize(rng.normal(0, 0.01, 300)).n_periods == 300
