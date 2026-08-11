"""The end-to-end demonstration: real data indicted, planted edge acquitted.

Both halves are required. A tool that only ever says "luck" has not been shown to
be a detector — it has been shown to be a pessimist, and the two are
indistinguishable from one example. So ``luckdet demo`` runs the machinery twice:
once on real SPY prices, where it should return ``LIKELY_LUCK``, and once on a
family with a genuine edge planted in it, where it should return
``LIKELY_SKILL``.

The planted edge is deliberately large — 5 of 50 variants at an annualised Sharpe
of 3.0 over ten years — and that is not a flattering choice, it is a measured
one. Phase 7 established that the verdict layer calls a *realistic* edge (10 of
50 at Sharpe 2.0 over five years) skill only 20% of the time. A demo built on
that effect size would be flaky, and papering over the flakiness with a lucky
seed would be worse than either. The configuration used here is the one measured
to return ``LIKELY_SKILL`` on all 25 datasets tried; the weakness itself is
reported in the caveats of every generated report.

Where the demo's data comes from
--------------------------------
**Cache, then download, then refuse.** The cache is consulted first so the demo
is reproducible and works offline once it has run once; a download is attempted
only if nothing is cached; and if neither is available the command *fails*, with
a message pointing at ``--offline``. It never quietly falls back to synthetic
data, because a demo that silently swaps the market for a random number generator
while printing the same confident narrative is precisely the failure mode this
package exists to catch.

One wrinkle worth recording, because it is invisible until it bites:
:func:`~luckdetector.io.prices.cache_path` keys the cache on ``(symbol, start,
end)`` and ``end`` defaults to *today*, so an exact-key lookup misses every day
after the file was written — the cached ``SPY_2010-01-01_2026-08-10.csv`` is a
miss on 2026-08-11 and every day thereafter. Resolution therefore matches on
symbol and start and takes the range with the latest end date, which is what
"use the cache if present" has to mean in practice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..exceptions import LuckDetectorError
from ..io.prices import DEFAULT_CACHE_DIR, Downloader, PriceHistory, load_prices
from ..mining.engine import mine, synthetic_prices
from ..types import TrialMatrix
from .analysis import Analysis, analyse, analyse_mined

__all__ = [
    "DEMO_SEED",
    "DEMO_SYMBOL",
    "DemoResult",
    "cached_ranges",
    "planted_edge_analysis",
    "resolve_demo_prices",
    "run_demo",
]

DEMO_SYMBOL = "SPY"
DEMO_START = "2010-01-01"
DEMO_SEED = 20260811

#: Searched in order. ``outputs/`` is the project's own scratch directory, which
#: is gitignored and holds the SPY CSV a previous run downloaded; the user cache
#: is where :func:`~luckdetector.io.prices.load_prices` puts things by default.
DEMO_CACHE_DIRS: tuple[Path, ...] = (Path("outputs"), DEFAULT_CACHE_DIR)

#: The planted-edge configuration, measured in Phase 7 to return LIKELY_SKILL on
#: all 25 datasets tried. Do not weaken these without re-measuring: at 10 of 50
#: with Sharpe 2.0 over five years the detection rate is 20%.
EDGE_N_TRIALS = 50
EDGE_N_GOOD = 5
EDGE_SHARPE = 3.0
EDGE_N_PERIODS = 2520
EDGE_DAILY_VOL = 0.01

_CACHE_NAME = re.compile(
    r"^(?P<symbol>.+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.csv$"
)


@dataclass(frozen=True)
class DemoResult:
    """Both halves of the demonstration, each a full :class:`Analysis`."""

    real: Analysis
    planted: Analysis

    @property
    def labels(self) -> tuple[str, str]:
        return (self.real.label, self.planted.label)


def cached_ranges(
    symbol: str, cache_dir: Path, *, start: str | None = None
) -> list[tuple[str, str]]:
    """Every ``(start, end)`` range already cached for ``symbol``, latest end last.

    Matching on the filename rather than probing an exact path is what makes
    "use the cache if present" actually true — see the module docstring for why
    the exact-key lookup silently misses.
    """
    if not cache_dir.is_dir():
        return []
    found: list[tuple[str, str]] = []
    for candidate in cache_dir.glob("*.csv"):
        match = _CACHE_NAME.match(candidate.name)
        if match is None or match.group("symbol").upper() != symbol.upper():
            continue
        if start is not None and match.group("start") != start:
            continue
        found.append((match.group("start"), match.group("end")))
    return sorted(found, key=lambda pair: pair[1])


def resolve_demo_prices(
    symbol: str = DEMO_SYMBOL,
    *,
    start: str = DEMO_START,
    cache_dirs: tuple[Path, ...] | None = None,
    downloader: Downloader | None = None,
    allow_download: bool = True,
) -> PriceHistory:
    """Cache, then download, then refuse.

    ``cache_dirs`` defaults to :data:`DEMO_CACHE_DIRS`, resolved at call time
    rather than bound into the signature so the constant can be overridden.

    Raises
    ------
    LuckDetectorError
        When nothing is cached and the download is unavailable or fails. The
        message names ``luckdet demo --offline`` explicitly, because the whole
        point of failing here is that the alternative — quietly substituting
        synthetic data — would produce a confident narrative about a random
        number generator.
    """
    cache_dirs = DEMO_CACHE_DIRS if cache_dirs is None else cache_dirs
    for cache_dir in cache_dirs:
        ranges = cached_ranges(symbol, cache_dir, start=start)
        if ranges:
            cached_start, cached_end = ranges[-1]  # the most recent end date
            return load_prices(symbol, start=cached_start, end=cached_end, cache_dir=cache_dir)

    if not allow_download:
        raise LuckDetectorError(
            f"No cached prices for {symbol} in "
            f"{', '.join(str(d) for d in cache_dirs)}, and downloading is disabled. "
            "Run `luckdet demo --offline` to demonstrate the machinery on a synthetic "
            "price path — every figure and number it produces will be labelled SYNTHETIC."
        )

    target = cache_dirs[0]
    try:
        return load_prices(symbol, start=start, cache_dir=target, downloader=downloader)
    except LuckDetectorError as exc:
        raise LuckDetectorError(
            f"No cached prices for {symbol} in "
            f"{', '.join(str(d) for d in cache_dirs)}, and the download failed: {exc}\n"
            "Run `luckdet demo --offline` to demonstrate the machinery on a synthetic "
            "price path — every figure and number it produces will be labelled SYNTHETIC. "
            "The demo will not silently substitute synthetic data for a market."
        ) from exc


def planted_edge_trials(seed: int = DEMO_SEED) -> TrialMatrix:
    """Mostly noise, with a genuine persistent edge in a handful of variants.

    Drawn from a single seeded stream rather than one generator per row, for the
    reason recorded in ``tests/conftest.py``: seeding each row separately gives
    measurably under-dispersed extremes, which quietly weakens exactly the
    extreme-value behaviour the Deflated Sharpe Ratio depends on.
    """
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, EDGE_DAILY_VOL, (EDGE_N_TRIALS, EDGE_N_PERIODS))
    values[:EDGE_N_GOOD] += (EDGE_SHARPE / np.sqrt(252)) * EDGE_DAILY_VOL
    labels = [f"edge_{i}" if i < EDGE_N_GOOD else f"noise_{i}" for i in range(EDGE_N_TRIALS)]
    return TrialMatrix(values, periods_per_year=252, labels=labels)


def planted_edge_analysis(*, seed: int = DEMO_SEED, n_resamples: int = 1000) -> Analysis:
    """The control: run the same machinery on a family that really does have an edge."""
    trials = planted_edge_trials(seed)
    return analyse(
        trials,
        benchmark=0.0,
        title="Control: a genuine edge, planted and then detected",
        provenance=(
            f"Synthetic control — {EDGE_N_GOOD} of {EDGE_N_TRIALS} variants carry a true "
            f"annualised Sharpe of {EDGE_SHARPE:.1f} over "
            f"{EDGE_N_PERIODS / 252:.0f} years, seed {seed}"
        ),
        synthetic=True,
        n_resamples=n_resamples,
        seed=11,
    )


def real_data_analysis(
    history: PriceHistory,
    *,
    cost_bps: float = 1.0,
    n_resamples: int = 1000,
    synthetic: bool = False,
) -> Analysis:
    """Mine a grid over real prices and judge the winner against buy-and-hold."""
    result = mine(history.close, cost_bps=cost_bps)
    return analyse_mined(
        result,
        title=f"{history.symbol}: is the best of {result.n_trials} strategies real?",
        provenance=f"{history.symbol}, {history.span} ({history.source}), "
        f"{history.n_periods:,} closes, {cost_bps:.1f}bp cost",
        synthetic=synthetic,
        dates=history.dates,
        n_resamples=n_resamples,
    )


def synthetic_demo_analysis(
    *, periods: int = 3780, seed: int = 42, cost_bps: float = 1.0, n_resamples: int = 1000
) -> Analysis:
    """The ``--offline`` substitute for real prices, labelled as what it is."""
    prices = synthetic_prices(periods, seed=seed)
    result = mine(prices, cost_bps=cost_bps)
    return analyse_mined(
        result,
        title=f"SYNTHETIC path: is the best of {result.n_trials} strategies real?",
        provenance=(
            f"Synthetic GARCH-like price path, {periods:,} periods, seed {seed}, "
            f"{cost_bps:.1f}bp cost — not a market"
        ),
        synthetic=True,
        n_resamples=n_resamples,
    )


def run_demo(
    *,
    offline: bool = False,
    symbol: str = DEMO_SYMBOL,
    start: str = DEMO_START,
    cost_bps: float = 1.0,
    cache_dirs: tuple[Path, ...] | None = None,
    downloader: Downloader | None = None,
    n_resamples: int = 1000,
) -> DemoResult:
    """Run both halves of the demonstration.

    Returns
    -------
    DemoResult
        ``real`` is the mined family (SPY unless ``offline``), ``planted`` is the
        control with a genuine edge. On the shipped configuration the labels are
        ``LIKELY_LUCK`` and ``LIKELY_SKILL`` respectively.
    """
    if offline:
        first = synthetic_demo_analysis(cost_bps=cost_bps, n_resamples=n_resamples)
    else:
        history = resolve_demo_prices(
            symbol, start=start, cache_dirs=cache_dirs, downloader=downloader
        )
        first = real_data_analysis(history, cost_bps=cost_bps, n_resamples=n_resamples)
    return DemoResult(real=first, planted=planted_edge_analysis(n_resamples=n_resamples))
