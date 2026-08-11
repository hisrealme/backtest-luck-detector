"""PBO via CSCV: block splitting, subsample scoring, and the overfitting probability.

Two tests in this file are load-bearing.

``test_subsample_sharpe_matches_brute_force`` checks the vectorised scorer against
literally concatenating the blocks and calling ``mean/std``. The block-sufficient-
statistics trick is what makes 12,870 splits fast, and a subtle error in it would
change every number this module produces while leaving all the behavioural tests
looking plausible.

``TestDegradationNull`` pins down the fact that the in-sample/out-of-sample slope is
mechanically about -1 under *no* selection. That is a property of partitioning a
fixed total, not a finding about strategies, and it is asserted here so nobody
later reads a negative slope as evidence of overfitting.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from luckdetector.exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
)
from luckdetector.stats.pbo import (
    MIN_N_BLOCKS,
    PBO_THRESHOLD,
    contiguous_blocks,
    performance_degradation,
    probability_of_backtest_overfitting,
)
from luckdetector.types import TrialMatrix

ReturnFactory = Callable[..., Any]

N_PERIODS = 1260
DAILY_VOL = 0.01


def noise_matrix(rng: np.random.Generator, n_trials: int = 50) -> TrialMatrix:
    """A family of trials with no edge whatsoever."""
    return TrialMatrix(rng.normal(0.0, DAILY_VOL, (n_trials, N_PERIODS)), periods_per_year=252)


def planted_edge_matrix(
    rng: np.random.Generator,
    *,
    n_trials: int = 50,
    n_good: int = 10,
    annual_sharpe: float = 2.0,
) -> TrialMatrix:
    """Mostly noise, with ``n_good`` trials carrying a genuine, persistent edge."""
    values = rng.normal(0.0, DAILY_VOL, (n_trials, N_PERIODS))
    values[:n_good] += (annual_sharpe / np.sqrt(252)) * DAILY_VOL
    return TrialMatrix(values, periods_per_year=252)


class TestContiguousBlocks:
    def test_blocks_tile_the_sample_exactly(self) -> None:
        blocks = contiguous_blocks(1260, 16)
        assert blocks[0][0] == 0
        assert blocks[-1][1] == 1260
        assert all(a[1] == b[0] for a, b in itertools.pairwise(blocks))

    def test_no_observation_is_discarded_on_an_awkward_length(self) -> None:
        """4,173 periods into 16 blocks: trimming would throw away 13 days."""
        blocks = contiguous_blocks(4173, 16)
        assert sum(stop - start for start, stop in blocks) == 4173

    def test_block_sizes_differ_by_at_most_one(self) -> None:
        sizes = [stop - start for start, stop in contiguous_blocks(1000, 16)]
        assert max(sizes) - min(sizes) <= 1

    def test_rejects_odd_block_count(self) -> None:
        with pytest.raises(ValueError, match="even"):
            contiguous_blocks(1260, 15)

    def test_rejects_too_few_blocks(self) -> None:
        with pytest.raises(ValueError, match=f"at least {MIN_N_BLOCKS}"):
            contiguous_blocks(1260, 2)

    def test_rejects_sample_shorter_than_block_count(self) -> None:
        """S larger than T — every block would be empty."""
        with pytest.raises(InsufficientDataError, match="at least 2 observations"):
            contiguous_blocks(10, 16)


class TestSubsampleScoring:
    def test_subsample_sharpe_matches_brute_force(self, rng: np.random.Generator) -> None:
        """The vectorised scorer must equal naive concatenate-and-score, exactly.

        Uses a period count that is *not* divisible by the block count, since that
        is where an off-by-one in the block bounds would hide.
        """
        from luckdetector.stats.pbo import (
            _block_statistics,
            _membership_matrix,
            _subsample_sharpe,
        )

        values = rng.normal(0.0003, DAILY_VOL, (9, 613))
        blocks = contiguous_blocks(613, 8)
        membership, _ = _membership_matrix(
            8, max_combinations=10_000, rng=np.random.default_rng(0)
        )
        counts, sums, squares, centre = _block_statistics(values, blocks)
        scale = np.mean(np.abs(values), axis=1, keepdims=True)
        fast, _ = _subsample_sharpe(counts, sums, squares, centre, scale, membership)

        for split in range(membership.shape[0]):
            periods = np.concatenate(
                [np.arange(*blocks[b]) for b in np.flatnonzero(membership[split])]
            )
            sub = values[:, periods]
            expected = sub.mean(axis=1) / sub.std(axis=1, ddof=1)
            assert fast[:, split] == pytest.approx(expected, abs=1e-12)

    def test_in_sample_and_out_of_sample_are_complementary(
        self, rng: np.random.Generator
    ) -> None:
        """Every period belongs to exactly one side of each split."""
        from luckdetector.stats.pbo import _membership_matrix

        membership, _ = _membership_matrix(8, max_combinations=10_000, rng=rng)
        assert np.all(membership.sum(axis=1) == 4)
        assert np.all((membership + (1.0 - membership)) == 1.0)

    def test_enumeration_is_exhaustive_and_distinct(self, rng: np.random.Generator) -> None:
        from luckdetector.stats.pbo import _membership_matrix

        membership, exhaustive = _membership_matrix(16, max_combinations=100_000, rng=rng)
        assert exhaustive
        assert membership.shape == (12_870, 16)
        assert len(np.unique(membership, axis=0)) == 12_870

    def test_falls_back_to_sampling_when_the_cap_binds(self, rng: np.random.Generator) -> None:
        from luckdetector.stats.pbo import _membership_matrix

        membership, exhaustive = _membership_matrix(16, max_combinations=500, rng=rng)
        assert not exhaustive
        assert membership.shape == (500, 16)
        assert len(np.unique(membership, axis=0)) == 500  # sampled without replacement


class TestPBOUnderTheNull:
    def test_pure_noise_gives_pbo_near_one_half(self, rng: np.random.Generator) -> None:
        """The defining property, checked as an *ensemble* average.

        A single dataset is not enough. PBO computed on one trial matrix has a
        standard deviation of roughly 0.15 around 0.5 — the 12,870 splits are
        heavily dependent, so they carry far less information than their count
        suggests. Asserting on one draw would give a test that passes or fails on
        the seed. Averaging over independent datasets tests the actual claim.
        """
        values = [
            probability_of_backtest_overfitting(noise_matrix(rng), n_blocks=16).pbo
            for _ in range(30)
        ]
        assert float(np.mean(values)) == pytest.approx(0.5, abs=0.08)

    def test_single_dataset_dispersion_is_wide(self, rng: np.random.Generator) -> None:
        """Documents *why* the test above averages: one draw is genuinely noisy."""
        values = [
            probability_of_backtest_overfitting(noise_matrix(rng), n_blocks=16).pbo
            for _ in range(30)
        ]
        assert float(np.std(values)) > 0.05

    def test_noise_does_not_pass(self, rng: np.random.Generator) -> None:
        result = probability_of_backtest_overfitting(noise_matrix(rng), n_blocks=16)
        assert not result.passed


class TestPBOWithGenuineEdge:
    def test_planted_edge_gives_low_pbo(self, rng: np.random.Generator) -> None:
        """A real, persistent edge in a subset of trials must survive selection."""
        result = probability_of_backtest_overfitting(planted_edge_matrix(rng), n_blocks=16)
        assert result.pbo < 0.1
        assert result.passed

    def test_single_good_trial_among_noise_is_found(self, rng: np.random.Generator) -> None:
        result = probability_of_backtest_overfitting(
            planted_edge_matrix(rng, n_good=1, annual_sharpe=2.5), n_blocks=16
        )
        assert result.pbo < PBO_THRESHOLD
        assert result.probability_of_loss < 0.1

    def test_edge_wins_on_the_discriminating_statistics(self, rng: np.random.Generator) -> None:
        """The three statistics that actually separate signal from noise."""
        noise = probability_of_backtest_overfitting(noise_matrix(rng), n_blocks=16)
        edge = probability_of_backtest_overfitting(planted_edge_matrix(rng), n_blocks=16)
        assert edge.pbo < noise.pbo
        assert edge.probability_of_loss < noise.probability_of_loss
        assert edge.dominance_fraction < noise.dominance_fraction

    def test_selected_trial_is_one_of_the_good_ones(self, rng: np.random.Generator) -> None:
        result = probability_of_backtest_overfitting(
            planted_edge_matrix(rng, n_good=10), n_blocks=16
        )
        assert float(np.mean(result.selected_trials < 10)) > 0.9


class TestDegradationNull:
    """The in-sample/out-of-sample slope is mechanically negative. Do not 'fix' it."""

    def test_fixed_trial_slope_is_minus_one(self, rng: np.random.Generator) -> None:
        """With no selection at all, the slope is -1 by construction.

        The two halves of a split partition a fixed total, so a period of good
        performance counted in-sample is a period unavailable out-of-sample. This
        is arithmetic, not a statement about strategies — which is exactly why a
        negative slope is not evidence of overfitting.
        """
        from luckdetector.stats.pbo import (
            _block_statistics,
            _membership_matrix,
            _subsample_sharpe,
        )

        values = rng.normal(0.0, DAILY_VOL, (5, N_PERIODS))
        blocks = contiguous_blocks(N_PERIODS, 8)
        membership, _ = _membership_matrix(8, max_combinations=10_000, rng=rng)
        counts, sums, squares, centre = _block_statistics(values, blocks)
        scale = np.mean(np.abs(values), axis=1, keepdims=True)
        in_sample, _ = _subsample_sharpe(counts, sums, squares, centre, scale, membership)
        out_sample, _ = _subsample_sharpe(
            counts, sums, squares, centre, scale, 1.0 - membership
        )

        for trial in range(values.shape[0]):
            slope = performance_degradation(in_sample[trial], out_sample[trial]).slope
            assert slope == pytest.approx(-1.0, abs=0.05)

    def test_slope_does_not_separate_noise_from_edge(self, rng: np.random.Generator) -> None:
        """Both are strongly negative, so the statistic carries little information."""
        noise = probability_of_backtest_overfitting(noise_matrix(rng), n_blocks=16)
        edge = probability_of_backtest_overfitting(planted_edge_matrix(rng), n_blocks=16)
        assert noise.degradation.slope < 0.0
        assert edge.degradation.slope < 0.0


class TestPerformanceDegradation:
    def test_recovers_a_known_slope(self) -> None:
        x = np.linspace(-1.0, 1.0, 50)
        result = performance_degradation(x, 2.0 * x + 3.0)
        assert result.slope == pytest.approx(2.0)
        assert result.intercept == pytest.approx(3.0)
        assert result.r_squared == pytest.approx(1.0)
        assert result.n_points == 50

    def test_ignores_non_finite_pairs(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0, np.inf])
        y = np.array([0.0, 1.0, 2.0, 3.0, 1.0])
        assert performance_degradation(x, y).n_points == 4

    def test_flat_response_has_zero_slope_and_no_fit(self) -> None:
        result = performance_degradation(np.linspace(0, 1, 20), np.full(20, 0.5))
        assert result.slope == pytest.approx(0.0)
        assert result.r_squared == pytest.approx(0.0)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="in-sample"):
            performance_degradation(np.zeros(5), np.zeros(4))

    def test_rejects_too_few_points(self) -> None:
        with pytest.raises(InsufficientDataError, match="at least 3"):
            performance_degradation(np.zeros(2), np.zeros(2))

    def test_rejects_constant_predictor(self) -> None:
        with pytest.raises(DegenerateSeriesError, match="identical"):
            performance_degradation(np.ones(10), np.linspace(0, 1, 10))


class TestValidationAndEdges:
    def test_rejects_a_constant_trial(self, rng: np.random.Generator) -> None:
        """A never-trading variant has no Sharpe ratio to rank."""
        values = rng.normal(0.0, DAILY_VOL, (6, N_PERIODS))
        values[3] = 0.0
        trials = TrialMatrix(values, periods_per_year=252, labels=[f"v{i}" for i in range(6)])
        with pytest.raises(DegenerateSeriesError, match="v3"):
            probability_of_backtest_overfitting(trials, n_blocks=8)

    def test_rejects_odd_block_count(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(noise_matrix(rng, 5), n_blocks=7)

    def test_rejects_more_blocks_than_periods(self, rng: np.random.Generator) -> None:
        short = TrialMatrix(rng.normal(0.0, DAILY_VOL, (5, 20)), periods_per_year=252)
        with pytest.raises(InsufficientDataError):
            probability_of_backtest_overfitting(short, n_blocks=16)

    def test_single_trial_is_rejected_at_construction(self, rng: np.random.Generator) -> None:
        """PBO is meaningless for one trial; the type refuses to build the input."""
        with pytest.raises(InsufficientDataError, match="at least 2 trials"):
            TrialMatrix(rng.normal(0.0, DAILY_VOL, (1, N_PERIODS)), periods_per_year=252)

    def test_two_trials_still_works(self, rng: np.random.Generator) -> None:
        result = probability_of_backtest_overfitting(noise_matrix(rng, 2), n_blocks=8)
        assert 0.0 <= result.pbo <= 1.0

    def test_logits_are_always_finite(self, rng: np.random.Generator) -> None:
        """The ``N + 1`` denominator keeps omega strictly inside (0, 1)."""
        result = probability_of_backtest_overfitting(planted_edge_matrix(rng), n_blocks=16)
        assert np.all(np.isfinite(result.logits))
        assert np.all(result.relative_ranks > 0.0)
        assert np.all(result.relative_ranks < 1.0)

    def test_pbo_is_the_share_of_non_positive_logits(self, rng: np.random.Generator) -> None:
        result = probability_of_backtest_overfitting(noise_matrix(rng), n_blocks=16)
        assert result.pbo == pytest.approx(float(np.mean(result.logits <= 0.0)))

    def test_risk_free_rate_shifts_but_does_not_break(self, rng: np.random.Generator) -> None:
        trials = noise_matrix(rng)
        result = probability_of_backtest_overfitting(trials, n_blocks=8, risk_free_rate=0.04)
        assert 0.0 <= result.pbo <= 1.0
        assert np.all(result.oos_sharpe_selected < 100.0)


class TestReproducibility:
    def test_same_seed_gives_identical_output(self, rng: np.random.Generator) -> None:
        trials = noise_matrix(rng)
        first = probability_of_backtest_overfitting(trials, n_blocks=16, seed=7)
        second = probability_of_backtest_overfitting(trials, n_blocks=16, seed=7)
        assert first.pbo == second.pbo
        assert np.array_equal(first.logits, second.logits)
        assert np.array_equal(first.oos_sharpe_random, second.oos_sharpe_random)

    def test_pbo_itself_is_seed_independent_when_exhaustive(
        self, rng: np.random.Generator
    ) -> None:
        """Only the random-trial benchmark uses the rng; PBO is deterministic."""
        trials = noise_matrix(rng)
        first = probability_of_backtest_overfitting(trials, n_blocks=16, seed=1)
        second = probability_of_backtest_overfitting(trials, n_blocks=16, seed=2)
        assert first.pbo == second.pbo
        assert not np.array_equal(first.oos_sharpe_random, second.oos_sharpe_random)

    def test_accepts_a_generator(self, rng: np.random.Generator) -> None:
        trials = noise_matrix(rng)
        result = probability_of_backtest_overfitting(
            trials, n_blocks=8, seed=np.random.default_rng(3)
        )
        assert 0.0 <= result.pbo <= 1.0


class TestResultShape:
    def test_result_is_self_describing(self, rng: np.random.Generator) -> None:
        result = probability_of_backtest_overfitting(noise_matrix(rng), n_blocks=16)
        assert float(result) == result.pbo
        assert result.n_splits == 12_870
        assert result.exhaustive
        assert result.n_trials == 50
        assert "balanced splits" in result.interpretation
        assert result.as_dict()["n_blocks"] == 16

    def test_per_split_arrays_are_aligned(self, rng: np.random.Generator) -> None:
        result = probability_of_backtest_overfitting(noise_matrix(rng), n_blocks=8)
        for array in (
            result.logits,
            result.relative_ranks,
            result.selected_trials,
            result.is_sharpe_selected,
            result.oos_sharpe_selected,
            result.oos_sharpe_random,
        ):
            assert array.shape == (result.n_splits,)

    def test_sharpes_are_reported_annualised(self, rng: np.random.Generator) -> None:
        """Package-wide convention: public numbers are annualised."""
        result = probability_of_backtest_overfitting(
            planted_edge_matrix(rng, annual_sharpe=2.0), n_blocks=16
        )
        assert 0.5 < float(np.median(result.oos_sharpe_selected)) < 8.0

    def test_worse_than_noise_flag(self, rng: np.random.Generator) -> None:
        result = probability_of_backtest_overfitting(planted_edge_matrix(rng), n_blocks=16)
        assert not result.worse_than_noise

    def test_clean_data_has_no_degenerate_subsamples(self, rng: np.random.Generator) -> None:
        result = probability_of_backtest_overfitting(noise_matrix(rng), n_blocks=16)
        assert result.n_degenerate_subsamples == 0
