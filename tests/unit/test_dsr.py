"""Deflated Sharpe Ratio, effective trial counting, and expected maxima.

The load-bearing test in this file is ``test_matches_monte_carlo_simulation``: the
Gumbel approximation for the expected maximum is the whole basis of DSR, so it is
checked against brute-force simulation rather than against itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from luckdetector.exceptions import InsufficientDataError
from luckdetector.stats import moments
from luckdetector.stats.dsr import (
    DSR_THRESHOLD,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_trials,
    effective_number_of_trials,
    expected_max_sharpe,
    null_sharpe_std,
    sharpe_required_for_dsr,
)
from luckdetector.stats.psr import probabilistic_sharpe_ratio
from luckdetector.types import ReturnSeries, TrialMatrix

ReturnFactory = Callable[..., Any]


class TestExpectedMaxSharpe:
    def test_single_trial_has_no_hurdle(self) -> None:
        assert expected_max_sharpe(1, 0.5) == 0.0

    def test_grows_with_trial_count(self) -> None:
        values = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000)]
        assert values == sorted(values)

    def test_scales_linearly_in_dispersion(self) -> None:
        assert expected_max_sharpe(100, 2.0) == pytest.approx(2 * expected_max_sharpe(100, 1.0))

    def test_matches_monte_carlo_simulation(self, rng: np.random.Generator) -> None:
        """Compare the Gumbel approximation to the brute-force expected maximum."""
        for n_trials in (10, 50, 200, 1000):
            draws = rng.standard_normal((4000, n_trials))
            simulated = float(draws.max(axis=1).mean())
            assert expected_max_sharpe(n_trials, 1.0) == pytest.approx(simulated, rel=0.06)

    def test_grows_like_sqrt_log_n(self) -> None:
        """Doubling trials adds little; the first hundred add a lot."""
        assert expected_max_sharpe(200, 1.0) / expected_max_sharpe(100, 1.0) < 1.15
        assert expected_max_sharpe(100, 1.0) / expected_max_sharpe(2, 1.0) > 2.5

    def test_rejects_zero_trials(self) -> None:
        with pytest.raises(InsufficientDataError):
            expected_max_sharpe(0, 1.0)

    def test_rejects_negative_dispersion(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            expected_max_sharpe(10, -1.0)

    def test_reproduces_the_phase_one_finding(self) -> None:
        """200 zero-edge five-year backtests should imply a best Sharpe near 1.2.

        Phase 1 measured a mean batch maximum of ~1.19 annualised by simulation.
        The analytic hurdle must land in the same place, since this is exactly the
        quantity DSR uses as its benchmark.
        """
        std_period = null_sharpe_std(1260)
        hurdle_annual = expected_max_sharpe(200, std_period) * math.sqrt(252)
        assert hurdle_annual == pytest.approx(1.19, rel=0.15)


class TestEffectiveTrials:
    def _matrix(self, values: np.ndarray) -> np.ndarray:
        return np.corrcoef(values)

    def test_independent_trials_are_not_discounted(self, make_trials: ReturnFactory) -> None:
        corr = self._matrix(make_trials(0.0, n_trials=50, offset=31, n=2000))
        effective = effective_number_of_trials(corr, method="equicorrelated")
        assert effective == pytest.approx(50, rel=0.15)

    def test_identical_trials_collapse_to_one(self, rng: np.random.Generator) -> None:
        row = rng.normal(0, 0.01, 500)
        corr = self._matrix(np.tile(row, (20, 1)) + rng.normal(0, 1e-9, (20, 500)))
        assert effective_number_of_trials(corr, method="cluster") == pytest.approx(1.0)

    def test_cluster_method_recovers_group_count(self, rng: np.random.Generator) -> None:
        """Three tight families of near-duplicates should count as roughly three bets."""
        n_periods = 1000
        blocks = []
        for _ in range(3):
            base = rng.normal(0, 0.01, n_periods)
            for _ in range(10):
                blocks.append(base + rng.normal(0, 0.002, n_periods))
        corr = self._matrix(np.array(blocks))
        assert effective_number_of_trials(corr, method="cluster") == pytest.approx(3, abs=1)

    def test_independent_method_applies_no_discount(self, rng: np.random.Generator) -> None:
        row = rng.normal(0, 0.01, 200)
        corr = self._matrix(np.tile(row, (12, 1)) + rng.normal(0, 1e-9, (12, 200)))
        assert effective_number_of_trials(corr, method="independent") == 12.0

    def test_result_is_bounded_by_trial_count(self, make_trials: ReturnFactory) -> None:
        corr = self._matrix(make_trials(0.0, n_trials=30, offset=32, n=800))
        for method in ("equicorrelated", "cluster", "independent"):
            effective = effective_number_of_trials(corr, method=method)  # type: ignore[arg-type]
            assert 1.0 <= effective <= 30.0

    def test_rejects_non_square_input(self) -> None:
        with pytest.raises(ValueError, match="square"):
            effective_number_of_trials(np.zeros((3, 5)))

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            effective_number_of_trials(np.eye(4), method="magic")  # type: ignore[arg-type]


class TestDeflatedSharpeRatio:
    def test_deflation_always_lowers_the_verdict(self, skilled_returns: ReturnSeries) -> None:
        result = deflated_sharpe_ratio(skilled_returns, n_trials=100)
        assert result.dsr < result.psr_at_zero

    def test_more_trials_means_lower_dsr(self, skilled_returns: ReturnSeries) -> None:
        few = deflated_sharpe_ratio(skilled_returns, n_trials=2).dsr
        many = deflated_sharpe_ratio(skilled_returns, n_trials=5000).dsr
        assert many < few

    def test_single_trial_matches_plain_psr(self, skilled_returns: ReturnSeries) -> None:
        result = deflated_sharpe_ratio(skilled_returns, n_trials=1)
        assert result.dsr == pytest.approx(probabilistic_sharpe_ratio(skilled_returns).psr)

    def test_rejects_zero_trials(self, skilled_returns: ReturnSeries) -> None:
        with pytest.raises(InsufficientDataError):
            deflated_sharpe_ratio(skilled_returns, n_trials=0)

    def test_best_of_pure_noise_is_not_significant(self, noise_trials: TrialMatrix) -> None:
        """The headline behaviour: mine 200 zero-edge strategies, keep the winner.

        Its raw Sharpe looks respectable and its undeflated PSR is high. DSR must
        refuse to call it skill.
        """
        result = deflated_sharpe_ratio_from_trials(noise_trials)
        assert result.sharpe_annual > 0.8  # the winner looks good...
        assert result.psr_at_zero > 0.95  # ...and naive PSR is fooled...
        assert not result.passed  # ...but DSR is not.
        assert result.dsr < 0.95

    def test_genuine_edge_survives_deflation(self, make_returns: ReturnFactory) -> None:
        """A real, strong, long-lived edge found in few trials must still pass."""
        series = ReturnSeries(make_returns(1.5, offset=41, n=5040), periods_per_year=252)
        result = deflated_sharpe_ratio(series, n_trials=20)
        assert result.passed
        assert result.dsr > 0.95

    def test_from_trials_selects_the_winner(self, noise_trials: TrialMatrix) -> None:
        best = max(
            moments.sharpe_ratio(noise_trials.trial(i)) for i in range(noise_trials.n_trials)
        )
        assert deflated_sharpe_ratio_from_trials(noise_trials).sharpe_annual == pytest.approx(best)

    def test_from_trials_honours_explicit_index(self, noise_trials: TrialMatrix) -> None:
        result = deflated_sharpe_ratio_from_trials(noise_trials, index=7)
        assert result.sharpe_annual == pytest.approx(moments.sharpe_ratio(noise_trials.trial(7)))

    def test_correlated_trials_get_a_lower_hurdle(self, rng: np.random.Generator) -> None:
        """Fifty variants of one idea must not be punished as fifty independent bets."""
        base = rng.normal(0.0004, 0.01, 1500)
        duplicates = TrialMatrix(
            np.array([base + rng.normal(0, 0.0005, 1500) for _ in range(50)]),
            periods_per_year=252,
        )
        clustered = deflated_sharpe_ratio_from_trials(duplicates, method="cluster")
        naive = deflated_sharpe_ratio_from_trials(duplicates, method="independent")
        assert clustered.n_effective_trials < naive.n_effective_trials
        assert clustered.expected_max_sharpe_annual < naive.expected_max_sharpe_annual
        assert clustered.dsr > naive.dsr

    def test_result_is_self_describing(self, noise_trials: TrialMatrix) -> None:
        result = deflated_sharpe_ratio_from_trials(noise_trials)
        assert float(result) == result.dsr
        assert "effectively" in result.interpretation
        assert result.as_dict()["n_trials"] == 200
        assert result.psr_result.benchmark_annual == pytest.approx(
            result.expected_max_sharpe_annual
        )


class TestSharpeRequiredForDSR:
    """Inverting DSR — the number the report draws, and why it is not the hurdle.

    The Phase 8 brief specified a figure marking the *expected maximum of noise*
    against the winner. That is the wrong bar: on SPY the winner clears it by
    0.18 and is still called luck, because the hurdle is a point while the
    winner's Sharpe is an estimate with a standard error of 0.247. This function
    is the bar the test actually applies, and these tests pin the relationship
    between the two.
    """

    def test_round_trips_through_the_statistic(self, noise_trials: TrialMatrix) -> None:
        """Substituting the answer back in returns the confidence exactly."""
        from scipy import stats as sps

        result = deflated_sharpe_ratio_from_trials(noise_trials)
        required = sharpe_required_for_dsr(result)
        recovered = float(
            sps.norm.cdf(
                (required - result.expected_max_sharpe_annual)
                / result.psr_result.standard_error_annual
            )
        )
        assert recovered == pytest.approx(DSR_THRESHOLD, abs=1e-12)

    def test_is_strictly_above_the_expected_maximum_of_noise(
        self, noise_trials: TrialMatrix
    ) -> None:
        """The gap between the two is the standard error the figure has to show."""
        result = deflated_sharpe_ratio_from_trials(noise_trials)
        assert sharpe_required_for_dsr(result) > result.expected_max_sharpe_annual

    def test_agrees_with_passed_in_both_directions(self, noise_trials: TrialMatrix) -> None:
        """Clearing the drawn bar and passing DSR must be the same event."""
        result = deflated_sharpe_ratio_from_trials(noise_trials)
        clears = result.sharpe_annual >= sharpe_required_for_dsr(result)
        assert clears == result.passed

    def test_a_lower_confidence_is_a_lower_bar(self, noise_trials: TrialMatrix) -> None:
        result = deflated_sharpe_ratio_from_trials(noise_trials)
        assert sharpe_required_for_dsr(result, confidence=0.80) < sharpe_required_for_dsr(
            result, confidence=0.95
        )

    def test_fifty_percent_confidence_is_exactly_the_expected_maximum(
        self, noise_trials: TrialMatrix
    ) -> None:
        """A sanity anchor: the median of the null sits on the hurdle itself."""
        result = deflated_sharpe_ratio_from_trials(noise_trials)
        assert sharpe_required_for_dsr(result, confidence=0.5) == pytest.approx(
            result.expected_max_sharpe_annual
        )

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_impossible_confidence(
        self, noise_trials: TrialMatrix, confidence: float
    ) -> None:
        result = deflated_sharpe_ratio_from_trials(noise_trials)
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            sharpe_required_for_dsr(result, confidence=confidence)
