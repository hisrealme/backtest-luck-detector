"""Shared fixtures. Everything is seeded; no test touches the network.

Each fixture owns its **own** generator, derived from ``SEED`` by an offset. Sharing
one generator across fixtures makes the data depend on the order pytest happens to
instantiate them, so the same test can pass or fail depending on which other fixtures
it requests. That is exactly the kind of irreproducibility this project is about.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from luckdetector.types import ReturnSeries, TrialMatrix

SEED = 20260810

#: Five years of daily data — the length of a plausible real track record.
N_DAYS = 1260
DAILY_VOL = 0.01


def generator(offset: int) -> np.random.Generator:
    """A generator whose stream depends only on ``offset``, never on call order."""
    return np.random.default_rng(SEED + offset)


def gaussian_returns(
    annual_sharpe: float,
    *,
    offset: int,
    n: int = N_DAYS,
    vol: float = DAILY_VOL,
) -> np.ndarray:
    """Daily returns with a *planted* annualised Sharpe ratio."""
    per_period_sharpe = annual_sharpe / np.sqrt(252)
    return generator(offset).normal(per_period_sharpe * vol, vol, n)


def gaussian_trials(
    annual_sharpe: float,
    *,
    n_trials: int,
    offset: int,
    n: int = N_DAYS,
    vol: float = DAILY_VOL,
) -> np.ndarray:
    """``(n_trials, n)`` of independent return streams, all with the same planted edge.

    Drawn from a **single** generator stream rather than one generator per row.
    Seeding each row separately with consecutive seeds produces streams that are
    reproducible but measurably under-dispersed, which quietly weakens any test of
    extreme-value behaviour — precisely the behaviour the Deflated Sharpe Ratio
    depends on.
    """
    per_period_sharpe = annual_sharpe / np.sqrt(252)
    return generator(offset).normal(per_period_sharpe * vol, vol, (n_trials, n))


@pytest.fixture
def rng() -> np.random.Generator:
    return generator(0)


@pytest.fixture
def make_trials() -> Callable[..., np.ndarray]:
    """Factory fixture: a matrix of independent return streams with a planted Sharpe."""
    return gaussian_trials


@pytest.fixture
def make_returns() -> Callable[..., np.ndarray]:
    """Factory fixture: build return streams with a planted Sharpe, reproducibly.

    Exposed as a fixture rather than a plain import so tests never depend on
    ``sys.path`` gymnastics to reach ``conftest``.
    """
    return gaussian_returns


@pytest.fixture
def n_days() -> int:
    return N_DAYS


@pytest.fixture
def flat_returns() -> ReturnSeries:
    """A hand-checkable series: mean 0.01, so the per-period Sharpe is 0.01 / sd."""
    return ReturnSeries(values=np.array([0.0, 0.02, 0.0, 0.02]), periods_per_year=4)


@pytest.fixture
def noise_returns() -> ReturnSeries:
    """Five years of daily returns with no edge whatsoever."""
    return ReturnSeries(gaussian_returns(0.0, offset=1), periods_per_year=252, name="noise")


@pytest.fixture
def skilled_returns() -> ReturnSeries:
    """Five years of daily returns with a genuine 1.0 annualised Sharpe."""
    return ReturnSeries(gaussian_returns(1.0, offset=2), periods_per_year=252, name="skilled")


@pytest.fixture
def noise_trials() -> TrialMatrix:
    """200 zero-edge strategies over five years — the canonical 'all luck' input."""
    values = generator(3).normal(0.0, DAILY_VOL, (200, N_DAYS))
    return TrialMatrix(
        values=values,
        periods_per_year=252,
        labels=[f"noise_{i}" for i in range(200)],
    )
