"""Resampling engines.

The tests that matter are the ones that would fail if a block method silently
degenerated into an iid one:

* ``test_block_methods_preserve_autocorrelation`` — resamples of an AR(1) series
  must retain dependence that the iid bootstrap destroys.
* ``test_geometric_block_lengths`` — the stationary bootstrap's runs must be
  geometrically distributed with the requested mean.
* ``test_longer_memory_gets_longer_blocks`` — automatic selection must react to
  the actual persistence in the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from luckdetector.exceptions import InsufficientDataError
from luckdetector.stats.bootstrap import (
    bootstrap_distribution,
    bootstrap_indices,
    circular_block_indices,
    iid_indices,
    optimal_block_length,
    permutation_null,
    stationary_indices,
)
from luckdetector.types import ReturnSeries


def ar1(n: int, phi: float, rng: np.random.Generator, scale: float = 0.01) -> np.ndarray:
    """An AR(1) series with autocorrelation ``phi`` — dependence with a known answer."""
    noise = rng.normal(0, scale, n)
    out = np.empty(n)
    out[0] = noise[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + noise[t]
    return out


def lag1_autocorr(x: np.ndarray) -> float:
    centred = x - x.mean()
    return float(np.dot(centred[:-1], centred[1:]) / np.dot(centred, centred))


class TestIndexGenerators:
    @pytest.mark.parametrize("maker", ["iid", "circular", "stationary"])
    def test_shape_and_range(self, maker: str, rng: np.random.Generator) -> None:
        n, size = 100, 25
        if maker == "iid":
            idx = iid_indices(n, size, rng)
        elif maker == "circular":
            idx = circular_block_indices(n, 10, size, rng)
        else:
            idx = stationary_indices(n, 10.0, size, rng)
        assert idx.shape == (size, n)
        assert idx.min() >= 0
        assert idx.max() < n

    def test_seeded_runs_are_identical(self) -> None:
        a = stationary_indices(50, 5.0, 10, np.random.default_rng(3))
        b = stationary_indices(50, 5.0, 10, np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = stationary_indices(50, 5.0, 10, np.random.default_rng(3))
        b = stationary_indices(50, 5.0, 10, np.random.default_rng(4))
        assert not np.array_equal(a, b)

    def test_circular_blocks_are_consecutive(self, rng: np.random.Generator) -> None:
        """Within a block, indices must advance by exactly one (modulo n)."""
        n, block = 60, 6
        idx = circular_block_indices(n, block, 40, rng)
        for row in idx:
            for start in range(0, n - block, block):
                chunk = row[start : start + block]
                steps = (np.diff(chunk) - 1) % n
                assert np.all(steps == 0)

    def test_block_length_of_one_is_iid_like(self, rng: np.random.Generator) -> None:
        idx = circular_block_indices(200, 1, 50, rng)
        assert idx.shape == (50, 200)

    def test_geometric_block_lengths(self, rng: np.random.Generator) -> None:
        """Stationary bootstrap runs must average the requested block length."""
        n, mean_block = 5000, 20.0
        idx = stationary_indices(n, mean_block, 20, rng)
        breaks = ((np.diff(idx, axis=1) - 1) % n) != 0
        n_breaks = int(breaks.sum())
        observed_mean_run = idx.size / (n_breaks + idx.shape[0])
        assert observed_mean_run == pytest.approx(mean_block, rel=0.15)

    def test_rejects_degenerate_inputs(self, rng: np.random.Generator) -> None:
        with pytest.raises(InsufficientDataError):
            iid_indices(1, 10, rng)
        with pytest.raises(ValueError, match="size must be positive"):
            iid_indices(10, 0, rng)
        with pytest.raises(ValueError, match="block_length"):
            circular_block_indices(10, 0, 5, rng)
        with pytest.raises(ValueError, match="mean_block_length"):
            stationary_indices(10, 0.0, 5, rng)

    def test_dispatch_requires_a_block_length_or_data(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="needs a block_length"):
            bootstrap_indices(100, 10, rng, method="stationary")

    def test_dispatch_rejects_unknown_method(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="Unknown bootstrap method"):
            bootstrap_indices(100, 10, rng, method="quantum", block_length=5)  # type: ignore[arg-type]


class TestDependencePreservation:
    def test_block_methods_preserve_autocorrelation(self, rng: np.random.Generator) -> None:
        """The whole reason block bootstraps exist, asserted directly."""
        series = ar1(4000, phi=0.6, rng=rng)
        original = lag1_autocorr(series)
        assert original == pytest.approx(0.6, abs=0.05)

        def mean_resampled_acf(idx: np.ndarray) -> float:
            return float(np.mean([lag1_autocorr(series[row]) for row in idx]))

        iid = mean_resampled_acf(iid_indices(4000, 40, rng))
        circular = mean_resampled_acf(circular_block_indices(4000, 50, 40, rng))
        stationary = mean_resampled_acf(stationary_indices(4000, 50.0, 40, rng))

        assert abs(iid) < 0.05  # iid destroys the dependence entirely
        assert circular > 0.45  # block methods retain most of it
        assert stationary > 0.45

    def test_iid_bootstrap_preserves_the_mean(self, rng: np.random.Generator) -> None:
        values = rng.normal(0.0005, 0.01, 2000)
        idx = iid_indices(2000, 400, rng)
        assert float(np.mean(values[idx])) == pytest.approx(float(values.mean()), abs=3e-4)


class TestOptimalBlockLength:
    def test_independent_data_gets_short_blocks(self, rng: np.random.Generator) -> None:
        assert optimal_block_length(rng.normal(0, 0.01, 3000)) < 8.0

    def test_longer_memory_gets_longer_blocks(self, rng: np.random.Generator) -> None:
        weak = optimal_block_length(ar1(3000, phi=0.1, rng=rng))
        strong = optimal_block_length(ar1(3000, phi=0.8, rng=rng))
        assert strong > weak

    def test_bounded_and_positive(self, rng: np.random.Generator) -> None:
        n = 3000
        b = optimal_block_length(ar1(n, phi=0.7, rng=rng))
        assert 1.0 <= b <= min(3 * np.sqrt(n), n / 3)

    def test_circular_variant_differs_from_stationary(self, rng: np.random.Generator) -> None:
        series = ar1(3000, phi=0.5, rng=rng)
        assert optimal_block_length(series, method="circular") != optimal_block_length(
            series, method="stationary"
        )

    def test_tiny_samples_return_one(self, rng: np.random.Generator) -> None:
        assert optimal_block_length(rng.normal(0, 0.01, 5)) == 1.0

    def test_constant_series_returns_one(self) -> None:
        assert optimal_block_length(np.full(100, 0.001)) == 1.0


class TestBootstrapDistribution:
    def test_distribution_brackets_the_observed_mean(self, rng: np.random.Generator) -> None:
        values = ReturnSeries(rng.normal(0.0005, 0.01, 1500))
        result = bootstrap_distribution(values, np.mean, n_resamples=500, seed=1)
        low, high = result.confidence_interval(0.95)
        assert low < result.observed < high
        assert result.n_resamples == 500

    def test_seeded_reproducibility(self, rng: np.random.Generator) -> None:
        values = ReturnSeries(rng.normal(0, 0.01, 500))
        a = bootstrap_distribution(values, np.mean, n_resamples=100, seed=7)
        b = bootstrap_distribution(values, np.mean, n_resamples=100, seed=7)
        np.testing.assert_allclose(a.distribution, b.distribution)

    def test_block_length_chosen_automatically(self, rng: np.random.Generator) -> None:
        values = ReturnSeries(ar1(2000, phi=0.7, rng=rng))
        result = bootstrap_distribution(values, np.mean, n_resamples=50, seed=2)
        assert result.block_length is not None
        assert result.block_length > 1.0

    def test_iid_method_reports_no_block_length(self, rng: np.random.Generator) -> None:
        values = ReturnSeries(rng.normal(0, 0.01, 400))
        result = bootstrap_distribution(values, np.mean, n_resamples=50, method="iid", seed=2)
        assert result.block_length is None

    def test_p_value_never_reaches_zero(self, rng: np.random.Generator) -> None:
        """A p-value of exactly 0 would overstate what 100 resamples can resolve."""
        values = ReturnSeries(rng.normal(0.05, 0.001, 300))
        result = bootstrap_distribution(values, np.mean, n_resamples=100, seed=5)
        assert result.p_value(alternative="greater") > 0.0

    def test_p_value_direction(self, rng: np.random.Generator) -> None:
        values = ReturnSeries(rng.normal(0.0, 0.01, 800))
        result = bootstrap_distribution(values, np.mean, n_resamples=400, seed=6)
        # The observed mean sits in the middle of its own bootstrap distribution.
        assert 0.3 < result.p_value(alternative="greater") < 0.7

    def test_rejects_invalid_confidence_level(self, rng: np.random.Generator) -> None:
        values = ReturnSeries(rng.normal(0, 0.01, 200))
        result = bootstrap_distribution(values, np.mean, n_resamples=50, seed=1)
        with pytest.raises(ValueError, match="between 0 and 1"):
            result.confidence_interval(1.5)


class TestPermutationNull:
    def test_shuffling_destroys_timing_but_keeps_returns(self, rng: np.random.Generator) -> None:
        """A momentum-style edge must vanish under permutation; the marginals must not."""
        values = ReturnSeries(ar1(2000, phi=0.5, rng=rng))
        result = permutation_null(values, lag1_autocorr, n_permutations=200, seed=4)
        assert result.observed > 0.4
        assert abs(float(result.distribution.mean())) < 0.05
        assert result.p_value() < 0.01

    def test_mean_is_invariant_under_permutation(self, rng: np.random.Generator) -> None:
        values = ReturnSeries(rng.normal(0.0005, 0.01, 500))
        result = permutation_null(values, np.mean, n_permutations=50, seed=4)
        np.testing.assert_allclose(result.distribution, result.observed)
