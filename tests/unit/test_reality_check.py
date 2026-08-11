"""White's Reality Check and Hansen's SPA: size, power, and two corrections to the spec.

Four tests in this file are load-bearing.

``test_resampled_means_match_brute_force`` checks the count-matrix trick against
literally indexing ``f[:, idx]`` and taking the mean. Every number this module
produces flows through that matmul, and an error in it would leave all the
behavioural tests looking plausible.

``TestSizeCalibration`` is the point of the phase. A test of superior predictive
ability is worth nothing unless its p-values are uniform when nothing is
superior, and that is checked with a KS test over 200 independent null datasets
rather than by eyeballing a single number.

``test_duplicating_a_strategy_changes_nothing`` pins down the shared-resample
requirement without a tolerance. Twenty copies of one strategy are one bet, so
they must be priced as one bet. This holds only when every strategy is resampled
on the same index vector.

``TestGarbageStrategies`` measures the argument for SPA over RC — and corrects the
Phase 6 brief, which predicted that adding garbage strategies would *lower* RC's
p-value. It raises it, necessarily, and these tests assert the direction that is
true rather than the one that was written down.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy import stats as sps

from luckdetector.exceptions import DataValidationError, InsufficientDataError
from luckdetector.stats.bootstrap import stationary_indices
from luckdetector.stats.reality_check import (
    MIN_PERIODS,
    SIGNIFICANCE_LEVEL,
    RealityCheckResult,
    reality_check,
)
from luckdetector.types import ReturnSeries, TrialMatrix

ReturnFactory = Callable[..., Any]

N_PERIODS = 1000
DAILY_VOL = 0.01

#: Independent null datasets behind the size calibration. The Phase 6 acceptance
#: criterion asks for at least 200.
N_NULL_DATASETS = 200

#: Seed for the multi-dataset loops. They cannot take the ``rng`` fixture, since
#: some of them are cached across tests, so they own a stream keyed like
#: ``conftest.generator`` — derived from one base seed by an offset, never shared,
#: so no result depends on the order pytest happens to run things in.
BASE_SEED = 20260811


def stream(offset: int) -> np.random.Generator:
    """A generator whose output depends only on ``offset``, never on call order."""
    return np.random.default_rng(BASE_SEED + offset)


def per_period_mean(annual_sharpe: float, vol: float = DAILY_VOL) -> float:
    """Per-period mean return implied by a planted annualised Sharpe ratio."""
    return annual_sharpe / np.sqrt(252) * vol


def noise_matrix(
    rng: np.random.Generator, n_trials: int = 20, n_periods: int = N_PERIODS
) -> TrialMatrix:
    """A family with exactly zero expected outperformance.

    This is the *least favourable configuration* — the boundary of the null, where
    every strategy has zero expected differential. Any configuration further
    inside the null gives conservative p-values, so this is the only configuration
    a size calibration can legitimately use.
    """
    return TrialMatrix(rng.normal(0.0, DAILY_VOL, (n_trials, n_periods)), periods_per_year=252)


def buried_edge_matrix(
    make_exact: ReturnFactory,
    rng: np.random.Generator,
    *,
    good_sharpe: float = 1.5,
    n_noise: int = 9,
    n_garbage: int = 0,
    garbage_sharpe: float = -4.0,
    offset: int = 101,
) -> TrialMatrix:
    """One genuinely good strategy, some noise, and optionally a pile of garbage.

    The good strategy comes from ``make_exact_returns`` rather than a planted
    draw. A *planted* annual Sharpe of 1.2 came back realised at **-0.43** on one
    of the seeds tried while writing these tests — which is the subject of this
    project, but a poor foundation for a power test that needs to know its own
    input.
    """
    rows = [np.asarray(make_exact(good_sharpe, offset=offset, n=N_PERIODS))[None, :]]
    if n_noise:
        rows.append(rng.normal(0.0, DAILY_VOL, (n_noise, N_PERIODS)))
    if n_garbage:
        rows.append(rng.normal(per_period_mean(garbage_sharpe), DAILY_VOL, (n_garbage, N_PERIODS)))
    return TrialMatrix(np.vstack(rows), periods_per_year=252)


class TestVectorisation:
    def test_resampled_means_match_brute_force(self, rng: np.random.Generator) -> None:
        """``f @ C.T / T`` must equal materialising every replicate, exactly.

        The count matrix is the only reason this runs on 157 x 4,173 x 1,000
        without 5 GB of intermediates, and it is the kind of trick that fails
        silently.
        """
        from luckdetector.stats.reality_check import _resample_count_matrix

        values = rng.normal(0.0003, DAILY_VOL, (7, 137))
        indices = stationary_indices(137, 5.0, 40, rng)
        counts = _resample_count_matrix(indices, 137)

        fast = (values @ counts.T) / 137
        for b in range(indices.shape[0]):
            assert fast[:, b] == pytest.approx(values[:, indices[b]].mean(axis=1), abs=1e-15)

    def test_every_replicate_draws_exactly_t_observations(self, rng: np.random.Generator) -> None:
        from luckdetector.stats.reality_check import _resample_count_matrix

        counts = _resample_count_matrix(stationary_indices(137, 5.0, 40, rng), 137)
        assert np.all(counts.sum(axis=1) == 137)

    def test_full_scale_problem_is_tractable(self, rng: np.random.Generator) -> None:
        """The SPY shape: 157 strategies, 4,173 periods, 1,000 resamples."""
        trials = TrialMatrix(rng.normal(0.0, DAILY_VOL, (157, 4173)), periods_per_year=252)
        result = reality_check(trials, 0.0, n_resamples=1000, block_length=5.0)
        assert 0.0 < result.p_consistent <= 1.0


class TestSharedResamples:
    """One index set for the whole panel. The easiest thing to get wrong."""

    def test_duplicating_a_strategy_changes_nothing(
        self, make_exact_returns: ReturnFactory
    ) -> None:
        """Twenty copies of one strategy are one bet, and must be priced as one.

        No tolerance is needed. With shared indices the twenty rows resample
        identically, so the maximum over ``k`` *is* the single strategy's
        statistic and every p-value is bit-for-bit the two-copy answer. Resample
        each row independently and this fails immediately.
        """
        single = np.asarray(make_exact_returns(1.6, offset=201, n=N_PERIODS))
        two = TrialMatrix(np.tile(single, (2, 1)), periods_per_year=252)
        twenty = TrialMatrix(np.tile(single, (20, 1)), periods_per_year=252)

        pair = reality_check(two, 0.0, n_resamples=1000, seed=3)
        many = reality_check(twenty, 0.0, n_resamples=1000, seed=3)

        assert many.p_reality_check == pair.p_reality_check
        assert many.p_consistent == pair.p_consistent
        assert many.p_upper == pair.p_upper

    def test_independent_resampling_would_over_penalise(
        self, make_exact_returns: ReturnFactory
    ) -> None:
        """The counterfactual, measured — and it corrects the Phase 6 brief.

        The brief said independent resampling produces p-values "far too small".
        For a positively correlated family it does the opposite: it prices twenty
        duplicates as twenty independent bets, drawing the null maximum far too
        high. The failure mode is lost power, not false significance.
        """
        from luckdetector.stats.reality_check import _resample_count_matrix

        single = np.asarray(make_exact_returns(1.6, offset=201, n=N_PERIODS))
        trials = TrialMatrix(np.tile(single, (20, 1)), periods_per_year=252)
        shared = reality_check(trials, 0.0, n_resamples=2000, block_length=2.0, seed=3)

        rng = np.random.default_rng(3)
        means = trials.values.mean(axis=1)
        root_t = np.sqrt(N_PERIODS)
        null = np.empty((trials.n_trials, 2000))
        for k in range(trials.n_trials):
            counts = _resample_count_matrix(
                stationary_indices(N_PERIODS, 2.0, 2000, rng), N_PERIODS
            )
            null[k] = root_t * ((trials.values[k] @ counts.T) / N_PERIODS - means[k])
        independent = (np.sum(np.max(null, axis=0) >= root_t * means.max()) + 1) / 2001

        assert independent > 3.0 * shared.p_reality_check


@pytest.fixture(scope="module")
def null_p_values() -> dict[str, np.ndarray]:
    """p-values from ``N_NULL_DATASETS`` independent datasets on the null boundary.

    Computed once and shared: five tests interrogate the same sample, and
    recomputing it per test would buy nothing but runtime.
    """
    rng = stream(601)
    collected: dict[str, list[float]] = {"rc": [], "lower": [], "consistent": [], "upper": []}
    for _ in range(N_NULL_DATASETS):
        result = reality_check(
            noise_matrix(rng, n_trials=20, n_periods=500), 0.0, n_resamples=500, seed=rng
        )
        collected["rc"].append(result.p_reality_check)
        collected["lower"].append(result.p_lower)
        collected["consistent"].append(result.p_consistent)
        collected["upper"].append(result.p_upper)
    return {key: np.array(value) for key, value in collected.items()}


class TestSizeCalibration:
    """p-values must be Uniform(0,1) when nothing beats the benchmark."""

    def test_reality_check_p_values_are_uniform(
        self, null_p_values: dict[str, np.ndarray]
    ) -> None:
        assert sps.kstest(null_p_values["rc"], "uniform").pvalue > 0.01

    def test_spa_consistent_p_values_are_uniform(
        self, null_p_values: dict[str, np.ndarray]
    ) -> None:
        assert sps.kstest(null_p_values["consistent"], "uniform").pvalue > 0.01

    def test_spa_upper_p_values_are_uniform(self, null_p_values: dict[str, np.ndarray]) -> None:
        assert sps.kstest(null_p_values["upper"], "uniform").pvalue > 0.01

    def test_rejection_rate_matches_the_nominal_level(
        self, null_p_values: dict[str, np.ndarray]
    ) -> None:
        """Calibration as a practitioner states it: a 5% test rejects 5% of the time."""
        for key in ("rc", "consistent", "upper"):
            rejected = float(np.mean(null_p_values[key] < SIGNIFICANCE_LEVEL))
            assert rejected == pytest.approx(SIGNIFICANCE_LEVEL, abs=0.04)

    def test_lower_bound_is_deliberately_liberal(
        self, null_p_values: dict[str, np.ndarray]
    ) -> None:
        """``p_lower`` is *not* calibrated, by design, and this records it.

        The lower recentring drops every strategy with a negative sample mean —
        about half the family under the null — which deflates the null
        distribution and over-rejects. It is reported as a bound on
        ``p_consistent``, never as a number to quote on its own.
        """
        assert sps.kstest(null_p_values["lower"], "uniform").pvalue < 0.01
        assert float(np.mean(null_p_values["lower"])) < 0.45
        assert float(np.mean(null_p_values["lower"] < SIGNIFICANCE_LEVEL)) > SIGNIFICANCE_LEVEL


class TestPower:
    def test_a_genuine_edge_is_detected(
        self, make_exact_returns: ReturnFactory, rng: np.random.Generator
    ) -> None:
        trials = buried_edge_matrix(make_exact_returns, rng, good_sharpe=2.0)
        result = reality_check(trials, 0.0, n_resamples=1000, seed=11)
        assert result.p_consistent < SIGNIFICANCE_LEVEL
        assert result.passed
        assert result.best_trial == 0

    def test_spa_is_never_less_powerful_than_reality_check(
        self, make_exact_returns: ReturnFactory
    ) -> None:
        """The Phase 6 acceptance criterion, over 40 datasets rather than one.

        SPA's advantage comes from studentising and from refusing to let hopeless
        strategies inflate the null. On a family where a real edge is buried among
        garbage, neither can hurt.
        """
        rng = stream(602)
        rc_values: list[float] = []
        spa_values: list[float] = []
        for i in range(40):
            trials = buried_edge_matrix(
                make_exact_returns, rng, good_sharpe=1.0, n_garbage=100, offset=700 + i
            )
            result = reality_check(trials, 0.0, n_resamples=500, seed=rng)
            rc_values.append(result.p_reality_check)
            spa_values.append(result.p_consistent)

        rc_p = np.array(rc_values)
        spa_p = np.array(spa_values)
        assert np.all(spa_p <= rc_p)
        assert float(np.mean(spa_p < SIGNIFICANCE_LEVEL)) >= float(
            np.mean(rc_p < SIGNIFICANCE_LEVEL)
        )


@pytest.fixture
def garbage_pair(
    make_exact_returns: ReturnFactory,
) -> tuple[RealityCheckResult, RealityCheckResult]:
    """The same family with and without 100 hopeless strategies bolted on.

    The block length is pinned so both runs draw *identical* resampling indices.
    The only difference between them is then the 100 extra strategies, which makes
    this a controlled experiment rather than two runs that happen to differ.
    """
    lean = buried_edge_matrix(make_exact_returns, stream(603), n_garbage=0)
    padded = buried_edge_matrix(make_exact_returns, stream(603), n_garbage=100)
    return (
        reality_check(lean, 0.0, n_resamples=1000, block_length=2.0, seed=11),
        reality_check(padded, 0.0, n_resamples=1000, block_length=2.0, seed=11),
    )


class TestGarbageStrategies:
    """Hopeless strategies must not be able to change a verdict about a good one."""

    def test_the_observed_statistic_is_untouched(
        self, garbage_pair: tuple[RealityCheckResult, RealityCheckResult]
    ) -> None:
        """Garbage never wins, so it cannot move the thing being tested."""
        lean, padded = garbage_pair
        assert padded.statistic_reality_check == lean.statistic_reality_check
        assert padded.statistic_spa == lean.statistic_spa
        assert padded.n_trials == lean.n_trials + 100

    def test_garbage_strategies_cripple_reality_check(
        self, garbage_pair: tuple[RealityCheckResult, RealityCheckResult]
    ) -> None:
        """Corrects the Phase 6 brief, which predicted the opposite direction.

        RC recentres every strategy on its own mean, so each hopeless strategy
        still contributes a mean-zero draw to the maximum of the null. A hundred
        of them carry RC from significant to not, on otherwise unchanged data.
        """
        lean, padded = garbage_pair
        assert lean.p_reality_check < SIGNIFICANCE_LEVEL
        assert padded.p_reality_check > lean.p_reality_check
        assert padded.p_reality_check > SIGNIFICANCE_LEVEL

    def test_garbage_strategies_leave_spa_unmoved(
        self, garbage_pair: tuple[RealityCheckResult, RealityCheckResult]
    ) -> None:
        """The whole argument for SPA, and it holds to the last bit."""
        lean, padded = garbage_pair
        assert padded.p_consistent == lean.p_consistent
        assert padded.p_consistent < SIGNIFICANCE_LEVEL
        # Every one of the 100 additions was recentred on zero, and the verdicts on
        # the original ten are untouched — so the count rises by exactly 100.
        assert padded.n_recentred_consistent - lean.n_recentred_consistent == 100

    def test_upper_recentring_fails_in_exactly_the_same_way(
        self, garbage_pair: tuple[RealityCheckResult, RealityCheckResult]
    ) -> None:
        """Confirms the mechanism, not just the outcome.

        ``upper`` *is* RC's recentring, studentised. If the diagnosis is right it
        has to degrade alongside RC while ``consistent`` holds — and it does.
        """
        lean, padded = garbage_pair
        assert padded.p_upper > lean.p_upper
        assert padded.p_upper > SIGNIFICANCE_LEVEL

    def test_more_strategies_can_only_raise_the_reality_check_p_value(
        self, make_exact_returns: ReturnFactory, rng: np.random.Generator
    ) -> None:
        """Monotonicity, which is why the brief's prediction was impossible.

        A maximum taken over a superset is weakly larger for every replicate, and
        here the observed statistic is fixed by the first strategy.
        """
        base = buried_edge_matrix(make_exact_returns, rng, good_sharpe=1.2, n_noise=4)
        previous = 0.0
        for extra in (0, 10, 40, 120):
            values = (
                np.vstack([base.values, rng.normal(0.0, DAILY_VOL, (extra, N_PERIODS))])
                if extra
                else base.values
            )
            padded = TrialMatrix(values, periods_per_year=252)
            current = reality_check(
                padded, 0.0, n_resamples=500, block_length=2.0, seed=11
            ).p_reality_check
            assert current >= previous
            previous = current


class TestPValueOrdering:
    def test_lower_le_consistent_le_upper_on_every_dataset(self) -> None:
        """Holds by construction, so it is asserted without tolerance.

        ``lower`` recentres a superset of what ``consistent`` recentres and
        ``upper`` recentres nothing, so replicate by replicate the three null
        statistics are ordered and the p-values inherit that order.
        """
        rng = stream(604)
        for _ in range(15):
            result = reality_check(noise_matrix(rng), 0.0, n_resamples=300, seed=rng)
            assert result.p_lower <= result.p_consistent <= result.p_upper

    def test_ordering_survives_a_planted_edge(
        self, make_exact_returns: ReturnFactory, rng: np.random.Generator
    ) -> None:
        trials = buried_edge_matrix(make_exact_returns, rng, n_garbage=50)
        result = reality_check(trials, 0.0, n_resamples=500, seed=11)
        assert result.p_lower <= result.p_consistent <= result.p_upper

    def test_upper_is_the_studentised_reality_check(self, rng: np.random.Generator) -> None:
        """Same recentring, so the two agree closely on a homoscedastic family."""
        result = reality_check(noise_matrix(rng), 0.0, n_resamples=1000, seed=11)
        assert result.p_upper == pytest.approx(result.p_reality_check, abs=0.15)


class TestBenchmark:
    def test_beating_zero_is_not_beating_the_benchmark(
        self, make_exact_returns: ReturnFactory
    ) -> None:
        """The SPY finding in miniature, and the reason the parameter exists.

        Thirty watered-down versions of a rising benchmark all make good money —
        annualised Sharpes of 1.2 to 2.3 — so against zero the family looks
        excellent and the test says so. Against the very thing they were diluting,
        not one of the thirty beats it and the p-value is 1.0. Same data, same
        machinery, opposite verdict: the benchmark *is* the question being asked.
        """
        benchmark = np.asarray(make_exact_returns(2.2, offset=605, n=N_PERIODS))
        family = benchmark * 0.6 + stream(605).normal(0.0, DAILY_VOL * 0.4, (30, N_PERIODS))
        trials = TrialMatrix(family, periods_per_year=252)

        against_zero = reality_check(trials, 0.0, n_resamples=1000, seed=5)
        against_benchmark = reality_check(trials, benchmark, n_resamples=1000, seed=5)

        assert against_zero.passed
        assert against_zero.n_beating_benchmark == 30
        assert not against_benchmark.passed
        assert against_benchmark.p_consistent == 1.0
        assert against_benchmark.n_beating_benchmark == 0

    def test_accepts_a_return_series_and_keeps_its_name(self, rng: np.random.Generator) -> None:
        benchmark = rng.normal(per_period_mean(0.5), DAILY_VOL, N_PERIODS)
        result = reality_check(
            noise_matrix(rng),
            ReturnSeries(benchmark, 252, "buy-and-hold"),
            n_resamples=200,
            seed=5,
        )
        assert result.benchmark_name == "buy-and-hold"
        assert "buy-and-hold" in result.interpretation

    def test_zero_benchmark_is_named_readably(self, rng: np.random.Generator) -> None:
        assert reality_check(noise_matrix(rng), 0.0, n_resamples=50).benchmark_name == "zero"

    def test_constant_benchmark_is_named_readably(self, rng: np.random.Generator) -> None:
        result = reality_check(noise_matrix(rng), 0.0004, n_resamples=50)
        assert result.benchmark_name == "a constant 0.0004 per period"

    def test_a_higher_hurdle_can_only_hurt(
        self, make_exact_returns: ReturnFactory, rng: np.random.Generator
    ) -> None:
        trials = buried_edge_matrix(make_exact_returns, rng, good_sharpe=1.5)
        easy = reality_check(trials, 0.0, n_resamples=500, block_length=2.0, seed=5)
        hard = reality_check(trials, 0.001, n_resamples=500, block_length=2.0, seed=5)
        assert hard.p_consistent > easy.p_consistent

    def test_rejects_a_misaligned_benchmark(self, rng: np.random.Generator) -> None:
        with pytest.raises(DataValidationError, match="aligned"):
            reality_check(noise_matrix(rng), np.zeros(N_PERIODS - 1), n_resamples=50)

    def test_rejects_a_non_finite_benchmark(self, rng: np.random.Generator) -> None:
        benchmark = np.zeros(N_PERIODS)
        benchmark[17] = np.nan
        with pytest.raises(DataValidationError, match="non-finite"):
            reality_check(noise_matrix(rng), benchmark, n_resamples=50)


class TestDegenerateAndEdgeCases:
    def test_a_strategy_identical_to_the_benchmark_is_excluded(
        self, rng: np.random.Generator
    ) -> None:
        """Its studentised statistic is 0/0, so it is counted rather than absorbed."""
        benchmark = rng.normal(per_period_mean(0.4), DAILY_VOL, N_PERIODS)
        values = rng.normal(0.0, DAILY_VOL, (5, N_PERIODS))
        values[2] = benchmark
        trials = TrialMatrix(values, periods_per_year=252)

        result = reality_check(trials, benchmark, n_resamples=200, seed=5)
        assert result.n_degenerate == 1
        assert np.isfinite(result.p_consistent)
        assert result.best_trial_studentised != 2

    def test_a_family_identical_to_the_benchmark_cannot_reject(
        self, rng: np.random.Generator
    ) -> None:
        benchmark = rng.normal(0.0, DAILY_VOL, N_PERIODS)
        trials = TrialMatrix(np.tile(benchmark, (3, 1)), periods_per_year=252)
        result = reality_check(trials, benchmark, n_resamples=200, seed=5)
        assert result.n_degenerate == 3
        assert result.statistic_spa == 0.0
        assert result.p_consistent == 1.0
        assert not result.passed

    def test_a_family_that_loses_everywhere_cannot_reject(self, rng: np.random.Generator) -> None:
        losers = rng.normal(per_period_mean(-3.0), DAILY_VOL, (10, N_PERIODS))
        result = reality_check(TrialMatrix(losers, periods_per_year=252), 0.0, n_resamples=500)
        assert result.statistic_spa == 0.0
        assert result.p_consistent == 1.0
        assert result.n_beating_benchmark == 0
        assert "nothing for the test to fail to explain" in result.interpretation

    def test_rejects_too_short_a_sample(self, rng: np.random.Generator) -> None:
        short = TrialMatrix(rng.normal(0.0, DAILY_VOL, (4, MIN_PERIODS - 1)), periods_per_year=252)
        with pytest.raises(InsufficientDataError, match=f"at least {MIN_PERIODS} periods"):
            reality_check(short, 0.0)

    def test_rejects_too_few_resamples(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="n_resamples"):
            reality_check(noise_matrix(rng), 0.0, n_resamples=1)

    def test_rejects_a_non_positive_block_length(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="block_length"):
            reality_check(noise_matrix(rng), 0.0, n_resamples=50, block_length=0.0)

    def test_two_trials_is_enough(self, rng: np.random.Generator) -> None:
        result = reality_check(noise_matrix(rng, n_trials=2), 0.0, n_resamples=200)
        assert 0.0 < result.p_consistent <= 1.0


class TestBlockLength:
    def test_automatic_length_is_one_on_independent_data(self, rng: np.random.Generator) -> None:
        assert reality_check(noise_matrix(rng), 0.0, n_resamples=100).block_length < 2.0

    def test_automatic_length_grows_with_serial_dependence(
        self, rng: np.random.Generator
    ) -> None:
        """An AR(1) family needs longer blocks, and the Politis-White rule finds them."""
        shocks = rng.normal(0.0, DAILY_VOL, (10, N_PERIODS))
        values = shocks.copy()
        for t in range(1, N_PERIODS):
            values[:, t] = 0.7 * values[:, t - 1] + shocks[:, t]
        result = reality_check(TrialMatrix(values, periods_per_year=252), 0.0, n_resamples=100)
        assert result.block_length > 3.0

    def test_automatic_length_is_invariant_to_scale(self, rng: np.random.Generator) -> None:
        """Politis-White reads the autocorrelation structure, which rescaling preserves.

        This is what licenses the module dividing each row by its peak before
        handing it to an estimator whose input type validates *returns*: a
        difference of two returns is entitled to fall below -100%, and a returns
        validator would reject it.
        """
        values = rng.normal(0.0, DAILY_VOL, (6, N_PERIODS))
        plain = reality_check(TrialMatrix(values, periods_per_year=252), 0.0, n_resamples=50)
        scaled = reality_check(
            TrialMatrix(values * 0.37, periods_per_year=252), 0.0, n_resamples=50
        )
        assert scaled.block_length == pytest.approx(plain.block_length)

    def test_explicit_block_length_is_used_and_recorded(self, rng: np.random.Generator) -> None:
        result = reality_check(noise_matrix(rng), 0.0, n_resamples=100, block_length=25.0)
        assert result.block_length == 25.0

    def test_survives_a_strategy_matching_the_benchmark(self, rng: np.random.Generator) -> None:
        """A zero differential row has no autocorrelation structure to estimate."""
        benchmark = rng.normal(0.0, DAILY_VOL, N_PERIODS)
        values = rng.normal(0.0, DAILY_VOL, (4, N_PERIODS))
        values[1] = benchmark
        result = reality_check(
            TrialMatrix(values, periods_per_year=252), benchmark, n_resamples=100
        )
        assert result.block_length >= 1.0


class TestReproducibility:
    def test_same_seed_gives_identical_output(self, rng: np.random.Generator) -> None:
        trials = noise_matrix(rng)
        first = reality_check(trials, 0.0, n_resamples=300, seed=7)
        second = reality_check(trials, 0.0, n_resamples=300, seed=7)
        assert first.as_dict() == second.as_dict()
        assert np.array_equal(first.omega, second.omega)

    def test_different_seeds_move_the_p_value(self, rng: np.random.Generator) -> None:
        trials = noise_matrix(rng)
        first = reality_check(trials, 0.0, n_resamples=300, seed=1)
        second = reality_check(trials, 0.0, n_resamples=300, seed=2)
        assert first.p_reality_check != second.p_reality_check
        assert first.statistic_reality_check == second.statistic_reality_check

    def test_accepts_a_generator(self, rng: np.random.Generator) -> None:
        result = reality_check(
            noise_matrix(rng), 0.0, n_resamples=200, seed=np.random.default_rng(3)
        )
        assert 0.0 < result.p_consistent <= 1.0


class TestResultShape:
    def test_float_is_the_number_to_quote(self, rng: np.random.Generator) -> None:
        result = reality_check(noise_matrix(rng), 0.0, n_resamples=200)
        assert float(result) == result.p_consistent

    def test_p_values_never_reach_zero(
        self, make_exact_returns: ReturnFactory, rng: np.random.Generator
    ) -> None:
        """``(extreme + 1) / (B + 1)`` claims only the resolution ``B`` supports."""
        trials = buried_edge_matrix(make_exact_returns, rng, good_sharpe=4.0)
        result = reality_check(trials, 0.0, n_resamples=200, seed=11)
        assert result.p_consistent == pytest.approx(1.0 / 201.0)

    def test_result_is_self_describing(self, rng: np.random.Generator) -> None:
        result = reality_check(noise_matrix(rng), 0.0, n_resamples=200, seed=11)
        assert result.n_trials == 20
        assert result.n_periods == N_PERIODS
        assert result.periods_per_year == 252
        assert result.n_resamples == 200
        assert result.mean_outperformance.shape == (20,)
        assert result.omega.shape == (20,)
        assert result.as_dict()["benchmark"] == "zero"
        assert "Reality Check" in result.interpretation

    def test_best_label_comes_from_the_trial_matrix(self, rng: np.random.Generator) -> None:
        values = rng.normal(0.0, DAILY_VOL, (4, N_PERIODS))
        values[2] += per_period_mean(3.0)
        trials = TrialMatrix(values, periods_per_year=252, labels=["a", "b", "MA(80,250)", "d"])
        result = reality_check(trials, 0.0, n_resamples=200, seed=11)
        assert result.best_trial == 2
        assert result.best_label == "MA(80,250)"
        assert "MA(80,250)" in result.interpretation

    def test_annualised_outperformance_is_reported_in_return_units(
        self, make_exact_returns: ReturnFactory, rng: np.random.Generator
    ) -> None:
        """Package-wide convention: public numbers are annualised."""
        trials = buried_edge_matrix(make_exact_returns, rng, good_sharpe=1.5)
        result = reality_check(trials, 0.0, n_resamples=200, seed=11)
        expected = float(result.mean_outperformance[result.best_trial]) * 252
        assert result.best_outperformance_annual == pytest.approx(expected)
        assert 0.0 < result.best_outperformance_annual < 1.0
