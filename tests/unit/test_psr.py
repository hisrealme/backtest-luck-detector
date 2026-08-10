"""PSR and MinTRL.

The important tests here are not "does the formula match itself" but:

* does the analytic standard error match the *empirical* dispersion of Sharpe
  estimates under simulation (``test_standard_error_matches_simulation``), and
* does MinTRL round-trip through PSR (``test_min_trl_round_trips_through_psr``).

Those two pin the module to reality rather than to a transcription of a paper.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy import stats as sps

from luckdetector.exceptions import DegenerateSeriesError, InsufficientDataError
from luckdetector.stats import moments
from luckdetector.stats.psr import (
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_standard_error,
    sharpe_variance_factor,
)
from luckdetector.types import ReturnSeries

ReturnFactory = Callable[..., Any]


class TestVarianceFactor:
    def test_gaussian_case_reduces_to_lo_2002(self) -> None:
        # For skew=0, kurtosis=3 the factor must collapse to 1 + SR^2/2.
        for sr in (0.0, 0.05, 0.2):
            assert sharpe_variance_factor(sr) == pytest.approx(1.0 + sr**2 / 2)

    def test_negative_skew_inflates_variance(self) -> None:
        assert sharpe_variance_factor(0.1, skew=-1.5) > sharpe_variance_factor(0.1, skew=0.0)

    def test_fat_tails_inflate_variance(self) -> None:
        assert sharpe_variance_factor(0.1, kurt=9.0) > sharpe_variance_factor(0.1, kurt=3.0)

    def test_rejects_impossible_moment_combination(self) -> None:
        with pytest.raises(DegenerateSeriesError, match="non-positive"):
            sharpe_variance_factor(2.0, skew=5.0, kurt=1.0)

    def test_standard_error_shrinks_with_sample_size(self) -> None:
        assert sharpe_standard_error(0.05, 10_000) < sharpe_standard_error(0.05, 100)

    def test_standard_error_needs_two_observations(self) -> None:
        with pytest.raises(InsufficientDataError):
            sharpe_standard_error(0.05, 1)

    def test_standard_error_matches_simulation(self, make_trials: ReturnFactory) -> None:
        """The analytic SE must match the empirical spread of estimated Sharpes."""
        n_days = 756
        values = make_trials(0.8, n_trials=2000, offset=900, n=n_days)
        estimates = np.array([moments.sharpe_ratio(row, annualized=False) for row in values])
        empirical = float(estimates.std(ddof=1))
        analytic = sharpe_standard_error(float(estimates.mean()), n_days)
        assert empirical == pytest.approx(analytic, rel=0.05)


class TestProbabilisticSharpeRatio:
    def test_zero_excess_gives_one_half(self) -> None:
        series = ReturnSeries(np.tile([0.01, -0.01], 500))
        result = probabilistic_sharpe_ratio(series)
        # A perfectly symmetric alternating series has a Sharpe of exactly 0.
        assert result.sharpe_annual == pytest.approx(0.0, abs=1e-12)
        assert result.psr == pytest.approx(0.5)

    def test_bounded_between_zero_and_one(
        self, skilled_returns: ReturnSeries, noise_returns: ReturnSeries
    ) -> None:
        for series in (skilled_returns, noise_returns):
            assert 0.0 <= probabilistic_sharpe_ratio(series).psr <= 1.0

    def test_increases_with_track_record_length(self, make_exact_returns: ReturnFactory) -> None:
        """Same realised Sharpe, more data: strictly more credible."""
        short = probabilistic_sharpe_ratio(make_exact_returns(1.0, offset=11, n=252)).psr
        long = probabilistic_sharpe_ratio(make_exact_returns(1.0, offset=11, n=2520)).psr
        assert long > short

    def test_decreases_as_benchmark_rises(self, skilled_returns: ReturnSeries) -> None:
        low = probabilistic_sharpe_ratio(skilled_returns, benchmark_annual_sharpe=0.0).psr
        high = probabilistic_sharpe_ratio(skilled_returns, benchmark_annual_sharpe=1.5).psr
        assert high < low

    def test_skilled_scores_higher_than_noise(
        self, skilled_returns: ReturnSeries, noise_returns: ReturnSeries
    ) -> None:
        assert (
            probabilistic_sharpe_ratio(skilled_returns).psr
            > probabilistic_sharpe_ratio(noise_returns).psr
        )

    def test_negative_skew_lowers_psr(self, rng: np.random.Generator) -> None:
        """Same Sharpe, opposite skew: the pennies-in-front-of-a-steamroller penalty."""
        # Demean *exactly*: the sample mean of an exponential draw is not 1.0, and
        # the residual leaks into the two series with opposite signs, which would
        # give them different Sharpe ratios and invalidate the comparison.
        base = rng.standard_exponential(4000)
        base -= base.mean()
        positive = ReturnSeries(base * 0.01 + 0.0004)
        negative = ReturnSeries(-base * 0.01 + 0.0004)
        # Construction gives both the same mean and the same volatility...
        assert moments.sharpe_ratio(positive) == pytest.approx(moments.sharpe_ratio(negative))
        # ...but opposite skew, and PSR must penalise the negative one.
        assert moments.skewness(negative) < 0 < moments.skewness(positive)
        assert probabilistic_sharpe_ratio(negative).psr < probabilistic_sharpe_ratio(positive).psr

    def test_benchmark_is_interpreted_as_annualised(self, skilled_returns: ReturnSeries) -> None:
        """A benchmark equal to the observed annualised Sharpe must give PSR = 0.5."""
        observed = moments.sharpe_ratio(skilled_returns)
        result = probabilistic_sharpe_ratio(skilled_returns, benchmark_annual_sharpe=observed)
        assert result.psr == pytest.approx(0.5, abs=1e-9)

    def test_result_is_self_consistent(self, skilled_returns: ReturnSeries) -> None:
        result = probabilistic_sharpe_ratio(skilled_returns, benchmark_annual_sharpe=0.3)
        assert float(result) == result.psr
        assert result.sharpe_annual == pytest.approx(result.sharpe_per_period * math.sqrt(252))
        assert result.standard_error_annual == pytest.approx(
            result.standard_error_per_period * math.sqrt(252)
        )
        assert "probability" in result.interpretation
        assert result.as_dict()["psr"] == result.psr


class TestMinTrackRecordLength:
    def test_round_trips_through_psr(self, skilled_returns: ReturnSeries) -> None:
        """A record of exactly MinTRL length must sit exactly at the confidence level.

        This is the tightest available check: the two formulas are algebraic
        inverses, so any transcription error in either one breaks this.
        """
        required = min_track_record_length(skilled_returns, confidence=0.95)
        sr = moments.sharpe_ratio(skilled_returns, annualized=False)
        se = sharpe_standard_error(
            sr,
            round(required),
            skew=moments.skewness(skilled_returns),
            kurt=moments.kurtosis(skilled_returns),
        )
        assert float(sps.norm.cdf(sr / se)) == pytest.approx(0.95, abs=1e-3)

    def test_higher_sharpe_needs_less_data(self, make_exact_returns: ReturnFactory) -> None:
        weak = make_exact_returns(0.5, offset=21, n=2520)
        strong = make_exact_returns(2.0, offset=22, n=2520)
        assert min_track_record_length(strong) < min_track_record_length(weak)

    def test_mediocre_sharpe_needs_a_decade(self, make_exact_returns: ReturnFactory) -> None:
        """The humbling result, computed exactly.

        For near-Gaussian returns the variance factor is ~1, so
        ``n* ≈ (z / SR_period)^2 = (1.645 / (0.5/sqrt(252)))^2 ≈ 2728`` trading
        days — about 10.8 years to establish that a Sharpe of 0.5 beats zero.
        """
        years = min_track_record_length(make_exact_returns(0.5, offset=23, n=5040)) / 252
        assert years == pytest.approx(10.8, abs=0.5)

    def test_quadratic_in_sharpe(self, make_exact_returns: ReturnFactory) -> None:
        """Halving the Sharpe quadruples the data required."""
        half = min_track_record_length(make_exact_returns(0.5, offset=24, n=5040))
        full = min_track_record_length(make_exact_returns(1.0, offset=24, n=5040))
        assert half / full == pytest.approx(4.0, rel=0.05)

    def test_higher_confidence_needs_more_data(self, skilled_returns: ReturnSeries) -> None:
        assert min_track_record_length(skilled_returns, confidence=0.99) > min_track_record_length(
            skilled_returns, confidence=0.90
        )

    def test_undefined_when_sharpe_below_benchmark(self, noise_returns: ReturnSeries) -> None:
        with pytest.raises(DegenerateSeriesError, match="does not exceed"):
            min_track_record_length(noise_returns, benchmark_annual_sharpe=3.0)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
    def test_rejects_invalid_confidence(self, skilled_returns: ReturnSeries, bad: float) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            min_track_record_length(skilled_returns, confidence=bad)
