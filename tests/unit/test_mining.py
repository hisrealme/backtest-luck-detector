"""Signal grids and the vectorised backtester.

The single most important test in this file is ``test_no_look_ahead_bias``. A
one-period misalignment between signal and return manufactures an edge out of
nothing, and it would flow silently into every statistic downstream — the package
would confidently pronounce a bug to be skill.
"""

from __future__ import annotations

import numpy as np
import pytest

from luckdetector.exceptions import DataValidationError, InsufficientDataError
from luckdetector.mining import signals as sig
from luckdetector.mining.engine import backtest, mine, synthetic_prices
from luckdetector.stats import sharpe_ratio


@pytest.fixture
def prices() -> np.ndarray:
    return synthetic_prices(1500, seed=11)


class TestSyntheticPrices:
    def test_shape_and_positivity(self) -> None:
        p = synthetic_prices(500, seed=1)
        assert p.size == 500
        assert np.all(p > 0)

    def test_reproducible(self) -> None:
        np.testing.assert_allclose(synthetic_prices(300, seed=2), synthetic_prices(300, seed=2))

    def test_seeds_differ(self) -> None:
        assert not np.allclose(synthetic_prices(300, seed=2), synthetic_prices(300, seed=3))

    def test_has_volatility_clustering(self) -> None:
        """Squared returns must be autocorrelated — the property that motivates block bootstraps."""
        returns = np.diff(np.log(synthetic_prices(4000, seed=5)))
        squared = returns**2 - np.mean(returns**2)
        acf1 = float(np.dot(squared[:-1], squared[1:]) / np.dot(squared, squared))
        assert acf1 > 0.1

    def test_rejects_degenerate_length(self) -> None:
        with pytest.raises(DataValidationError):
            synthetic_prices(1)


class TestSignalGrids:
    def test_positions_are_one_shorter_than_prices(self, prices: np.ndarray) -> None:
        grid = sig.default_grid(prices)
        assert grid.positions.shape[1] == prices.size - 1

    def test_positions_are_ternary(self, prices: np.ndarray) -> None:
        grid = sig.default_grid(prices)
        assert set(np.unique(grid.positions)) <= {-1.0, 0.0, 1.0}

    def test_default_grid_is_large(self, prices: np.ndarray) -> None:
        assert len(sig.default_grid(prices)) >= 150

    def test_labels_match_variants(self, prices: np.ndarray) -> None:
        grid = sig.default_grid(prices)
        assert len(grid.labels) == grid.n_variants
        assert len(set(grid.labels)) == len(grid.labels)  # no duplicates

    def test_warmup_is_flat(self, prices: np.ndarray) -> None:
        """A strategy cannot trade before its indicator exists."""
        grid = sig.moving_average_crossover(prices, fast_windows=(10,), slow_windows=(200,))
        assert np.all(grid.positions[0, :199] == 0.0)

    def test_crossover_direction(self) -> None:
        """A monotonically rising price series must leave a crossover rule long."""
        rising = np.linspace(100.0, 200.0, 400)
        grid = sig.moving_average_crossover(rising, fast_windows=(5,), slow_windows=(50,))
        assert np.all(grid.positions[0, 60:] == 1.0)

    def test_momentum_direction(self) -> None:
        falling = np.linspace(200.0, 100.0, 400)
        grid = sig.time_series_momentum(falling, lookbacks=(20,))
        assert np.all(grid.positions[0, 25:] == -1.0)

    def test_breakout_goes_long_at_new_highs(self) -> None:
        rising = np.linspace(100.0, 200.0, 300)
        grid = sig.donchian_breakout(rising, lookbacks=(20,))
        assert np.all(grid.positions[0, 25:] == 1.0)

    def test_rsi_produces_both_signs(self, prices: np.ndarray) -> None:
        grid = sig.rsi_reversion(prices)
        assert grid.positions.min() == -1.0
        assert grid.positions.max() == 1.0

    def test_fast_slower_than_slow_pairs_are_skipped(self, prices: np.ndarray) -> None:
        grid = sig.moving_average_crossover(prices, fast_windows=(50,), slow_windows=(50, 100))
        assert grid.labels == ["MA(50,100)"]

    def test_rejects_short_series(self) -> None:
        with pytest.raises(InsufficientDataError):
            sig.time_series_momentum(np.linspace(100, 110, 10), lookbacks=(250,))

    def test_rejects_non_positive_prices(self) -> None:
        with pytest.raises(DataValidationError, match="non-positive"):
            sig.time_series_momentum(np.array([100.0, 0.0] + [100.0] * 300))

    def test_signalset_rejects_label_mismatch(self) -> None:
        with pytest.raises(DataValidationError, match="labels"):
            sig.SignalSet(["a"], np.zeros((3, 10)))


class TestBacktest:
    def test_always_long_reproduces_buy_and_hold(self, prices: np.ndarray) -> None:
        positions = np.ones((1, prices.size - 1))
        net = backtest(prices, positions, cost_bps=0.0)
        expected = prices[1:] / prices[:-1] - 1.0
        np.testing.assert_allclose(net[0], expected)

    def test_short_is_the_negative_of_long(self, prices: np.ndarray) -> None:
        long = backtest(prices, np.ones((1, prices.size - 1)), cost_bps=0.0)
        short = backtest(prices, -np.ones((1, prices.size - 1)), cost_bps=0.0)
        np.testing.assert_allclose(long, -short)

    def test_flat_earns_nothing(self, prices: np.ndarray) -> None:
        net = backtest(prices, np.zeros((1, prices.size - 1)), cost_bps=5.0)
        np.testing.assert_allclose(net, 0.0)

    def test_costs_reduce_returns(self, prices: np.ndarray) -> None:
        rng = np.random.default_rng(0)
        positions = rng.choice([-1.0, 0.0, 1.0], size=(3, prices.size - 1))
        free = backtest(prices, positions, cost_bps=0.0)
        costly = backtest(prices, positions, cost_bps=10.0)
        assert costly.sum() < free.sum()

    def test_entry_is_charged(self, prices: np.ndarray) -> None:
        """Establishing the first position is a trade and must be billed."""
        positions = np.ones((1, prices.size - 1))
        free = backtest(prices, positions, cost_bps=0.0)
        costly = backtest(prices, positions, cost_bps=100.0)
        assert costly[0, 0] == pytest.approx(free[0, 0] - 0.01)

    def test_flip_costs_double(self, prices: np.ndarray) -> None:
        positions = np.ones((1, prices.size - 1))
        positions[0, 10:] = -1.0
        net_free = backtest(prices, positions, cost_bps=0.0)
        net_costly = backtest(prices, positions, cost_bps=100.0)
        charged = net_free[0, 10] - net_costly[0, 10]
        assert charged == pytest.approx(0.02)  # |−1 − 1| = 2 units of turnover

    def test_rejects_misaligned_positions(self, prices: np.ndarray) -> None:
        with pytest.raises(DataValidationError, match="look-ahead"):
            backtest(prices, np.ones((1, prices.size)))

    def test_rejects_negative_costs(self, prices: np.ndarray) -> None:
        with pytest.raises(DataValidationError, match="non-negative"):
            backtest(prices, np.ones((1, prices.size - 1)), cost_bps=-1.0)

    def test_no_look_ahead_bias(self) -> None:
        """A perfect-foresight signal must be detectably different from a legal one.

        Build a signal that knows tomorrow's return. If the engine's alignment is
        correct, feeding it as a *position* array earns a spectacular Sharpe —
        confirming the engine applies position t to return t, and therefore that
        an ordinary indicator (which only sees prices up to t) cannot cheat.
        """
        prices = synthetic_prices(2000, seed=21)
        future_returns = prices[1:] / prices[:-1] - 1.0

        oracle = np.sign(future_returns).reshape(1, -1)
        legal = np.sign(np.concatenate([[0.0], future_returns[:-1]])).reshape(1, -1)

        oracle_sharpe = sharpe_ratio(backtest(prices, oracle, cost_bps=0.0)[0])
        legal_sharpe = sharpe_ratio(backtest(prices, legal, cost_bps=0.0)[0])

        assert oracle_sharpe > 10.0  # foresight is worth a lot
        assert legal_sharpe < 2.0  # yesterday's sign is worth little
        assert oracle_sharpe > 5 * abs(legal_sharpe)


class TestMine:
    def test_produces_a_trial_matrix(self, prices: np.ndarray) -> None:
        result = mine(prices)
        assert result.n_trials >= 150
        assert result.trials.n_periods == prices.size - 1
        assert result.buy_and_hold.size == prices.size - 1

    def test_trials_are_heavily_correlated(self, prices: np.ndarray) -> None:
        """Variants of a handful of ideas are not independent bets, and it shows."""
        corr = mine(prices).trials.correlation()
        off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
        assert float(np.mean(np.abs(off_diagonal))) > 0.15

    def test_labels_survive(self, prices: np.ndarray) -> None:
        labels = mine(prices).trials.labels
        assert any(label.startswith("MA(") for label in labels)
        assert any(label.startswith("BRK(") for label in labels)

    def test_costs_flow_through(self, prices: np.ndarray) -> None:
        cheap = mine(prices, cost_bps=0.0).trials.values.sum()
        dear = mine(prices, cost_bps=20.0).trials.values.sum()
        assert dear < cheap

    def test_rejects_a_grid_that_never_trades(self, prices: np.ndarray) -> None:
        flat = sig.SignalSet(["flat_a", "flat_b"], np.zeros((2, prices.size - 1)))
        with pytest.raises(DataValidationError, match="ever took a position"):
            mine(prices, signals=flat)
