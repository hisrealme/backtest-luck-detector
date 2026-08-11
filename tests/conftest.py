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

from luckdetector.exceptions import DataValidationError
from luckdetector.mining import mine, synthetic_prices
from luckdetector.report.analysis import Analysis, analyse, analyse_mined
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


def exact_sharpe_returns(
    annual_sharpe: float,
    *,
    offset: int,
    n: int = N_DAYS,
    vol: float = DAILY_VOL,
) -> np.ndarray:
    """Returns whose **realised** annualised Sharpe is exactly ``annual_sharpe``.

    ``gaussian_returns`` plants a Sharpe in the *population*; what comes back in
    any single draw can be wildly different — a planted 0.5 has realised -0.11 at
    one of the seeds used here. That randomness is the subject of this project, but
    it makes a poor foundation for a test that needs to know the input Sharpe.

    Standardising the draw to zero mean and unit sample standard deviation, then
    rescaling, fixes the realised moments exactly while keeping a realistic shape.
    """
    z = generator(offset).standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return vol * z + (annual_sharpe / np.sqrt(252)) * vol


@pytest.fixture
def rng() -> np.random.Generator:
    return generator(0)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Make any attempted download fail immediately, as it would with no network.

    Tests that exercise the *refusal* path need the download to be unavailable
    deterministically. Relying on ``yfinance`` not being installed would make the
    test pass for the wrong reason on a machine that has the ``[data]`` extra —
    and pass by making a real network call on one that does not.
    """

    def unavailable(symbol: str, start: str, end: str) -> object:
        raise DataValidationError(f"no network available for {symbol!r} in tests")

    monkeypatch.setattr("luckdetector.io.prices.yfinance_downloader", unavailable)
    return monkeypatch


@pytest.fixture
def make_exact_returns() -> Callable[..., np.ndarray]:
    """Factory fixture: returns with an exactly known realised Sharpe ratio."""
    return exact_sharpe_returns


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


# --------------------------------------------------------------- report layer

#: Both report fixtures are session-scoped. ``Analysis`` is frozen and every test
#: that consumes one only reads from it, so re-mining a grid per test would buy
#: nothing but wall-clock.


@pytest.fixture(scope="session")
def luck_analysis() -> Analysis:
    """A **mined** family with no real edge: the report's LIKELY_LUCK path.

    Mined rather than assembled from independent noise, and that is not
    incidental. The geometry the report has to draw only exists in a
    *correlated* family: 157 variants of four ideas collapse to about ten
    effectively independent trials, which drops the expected-max-of-noise hurdle
    well below the observed maximum and puts the winner above it while the
    Deflated Sharpe Ratio still says luck. A matrix of independent trials does
    the opposite — the winner lands *below* the expected max — and would hide
    the very problem ``test_analysis.TestTheHurdleThatIsActuallyApplied``
    exists to pin.

    ``synthetic=False`` is a statement about the *report flag*, which is what
    these tests exercise, not a claim about the prices. They are generated, and
    no number from this fixture is ever reported.
    """
    result = mine(synthetic_prices(900, seed=3), cost_bps=1.0)
    return analyse_mined(
        result,
        title="Mined grid, no real edge",
        provenance="seeded synthetic path, 900 periods",
        n_blocks=8,
        n_resamples=200,
        seed=1,
    )


@pytest.fixture(scope="session")
def edge_analysis() -> Analysis:
    """A family where a few variants really do have an edge, generously sized.

    Matches the configuration ``report.demo`` uses and Phase 7 measured: the
    verdict layer needs a large effect before it will say skill, and a fixture
    built on a realistic one would be flaky rather than informative. See
    ``test_verdict.test_detection_rate_at_a_realistic_edge_is_poor``.
    """
    values = generator(42).normal(0.0, DAILY_VOL, (50, 2520))
    values[:5] += (3.0 / np.sqrt(252)) * DAILY_VOL
    trials = TrialMatrix(
        values, periods_per_year=252, labels=[f"variant_{i}" for i in range(50)]
    )
    return analyse(
        trials,
        title="Planted edge",
        provenance="seeded edge",
        synthetic=True,
        n_blocks=8,
        n_resamples=200,
        seed=3,
    )
