"""One family in, four statistics and a verdict out.

This is the helper the CLI drives. It exists in the library rather than in
``cli.py`` for the reason stated in the Phase 8 brief: **the demo must not invent
numbers.** Every figure printed by ``luckdet demo`` or rendered into the HTML
report comes from this module, which in turn only calls the same public functions
a user would call themselves. If a number appears in the report that cannot be
traced to a function in :mod:`luckdetector.stats`, that is a bug.

Why it takes the whole family
-----------------------------
:func:`analyse` accepts a :class:`~luckdetector.types.TrialMatrix`, never a lone
winner, because three of the four statistics are undefined without the losers.
The winner is *derived* here — the argmax of the trial Sharpes — rather than
supplied, so the report cannot be handed a strategy that was not actually the
best of its search.

The division of labour with :mod:`luckdetector.report.verdict`
--------------------------------------------------------------
``verdict.assess`` deliberately takes computed results and does no statistics.
This module is the layer that computes them. Keeping the two apart is what lets
the rule table be read and argued with independently of how its inputs were
produced, and it is why Phase 7 shipped without any of this wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..mining.engine import MiningResult
from ..stats.dsr import DSRResult, deflated_sharpe_ratio_from_trials, sharpe_required_for_dsr
from ..stats.moments import max_drawdown, sharpe_ratio
from ..stats.pbo import PBOResult, probability_of_backtest_overfitting
from ..stats.psr import PSRResult, probabilistic_sharpe_ratio
from ..stats.reality_check import RealityCheckResult, reality_check
from ..types import FloatArray, ReturnSeries, TrialMatrix, Verdict
from .verdict import assess

__all__ = [
    "Analysis",
    "analyse",
    "analyse_mined",
]

#: CSCV block count used for the report. Phase 5 measured PBO as stable across
#: S in {8, 12, 16, 20}; 16 is the middle of that range and the value every
#: published SPY number in this project was computed at.
DEFAULT_N_BLOCKS = 16

#: Bootstrap replicates for RC/SPA. Phase 6 ran the full SPY grid at this count
#: in 0.17s, so there is no reason to economise.
DEFAULT_N_RESAMPLES = 1000


@dataclass(frozen=True)
class Analysis:
    """Everything the report renders, computed once and passed around by reference.

    The four statistics are kept as their own result objects rather than being
    flattened into floats, because each one carries the inputs needed to
    reproduce it and an ``interpretation`` string the template renders verbatim.
    """

    trials: TrialMatrix
    benchmark_returns: FloatArray
    benchmark_name: str
    trial_sharpes: FloatArray
    winner_index: int
    psr: PSRResult
    dsr: DSRResult
    pbo: PBOResult
    spa: RealityCheckResult
    verdict: Verdict
    title: str
    provenance: str
    synthetic: bool
    cost_bps: float | None = None
    dates: Any = None
    detail: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- winner

    @property
    def winner(self) -> ReturnSeries:
        """The best trial by realised Sharpe — what a backtester would have reported."""
        return self.trials.trial(self.winner_index)

    @property
    def winner_label(self) -> str:
        return str(self.trials.labels[self.winner_index])

    @property
    def winner_sharpe(self) -> float:
        return float(self.trial_sharpes[self.winner_index])

    @property
    def winner_total_return(self) -> float:
        return self.winner.total_return()

    @property
    def winner_max_drawdown(self) -> float:
        return max_drawdown(self.winner)

    # -------------------------------------------------------------- benchmark

    @property
    def benchmark_series(self) -> ReturnSeries:
        return ReturnSeries(
            values=self.benchmark_returns,
            periods_per_year=self.trials.periods_per_year,
            name=self.benchmark_name,
        )

    @property
    def benchmark_sharpe(self) -> float:
        """Annualised Sharpe of the benchmark, or ``nan`` if it never moves.

        A benchmark of literally zero has no volatility, so its Sharpe is
        undefined rather than infinite. The report prints an em dash.
        """
        if np.all(self.benchmark_returns == 0.0):
            return float("nan")
        return sharpe_ratio(self.benchmark_series)

    @property
    def benchmark_total_return(self) -> float:
        return self.benchmark_series.total_return()

    # ------------------------------------------------------------------ sizes

    @property
    def n_trials(self) -> int:
        return self.trials.n_trials

    @property
    def n_periods(self) -> int:
        return self.trials.n_periods

    @property
    def years(self) -> float:
        return self.trials.n_periods / self.trials.periods_per_year

    # ---------------------------------------------------- the deflation hurdle

    @property
    def dsr_hurdle(self) -> float:
        """Annualised Sharpe the winner needed to clear :data:`DSR_THRESHOLD`.

        Not the same thing as ``dsr.expected_max_sharpe_annual``, and the
        difference is the whole reason :func:`~luckdetector.stats.dsr.
        sharpe_required_for_dsr` exists — see its docstring.
        """
        return sharpe_required_for_dsr(self.dsr)

    @property
    def n_trials_above_expected_max(self) -> int:
        """How many variants clear the *expected maximum of noise*.

        On SPY this is 43 of 157. Quoted in the report because a reader looking
        at that hurdle alone would conclude a quarter of the family had an edge.
        """
        return int(np.sum(self.trial_sharpes >= self.dsr.expected_max_sharpe_annual))

    @property
    def n_trials_above_hurdle(self) -> int:
        """How many variants clear the bar DSR actually applies. On SPY: none."""
        return int(np.sum(self.trial_sharpes >= self.dsr_hurdle))

    @property
    def n_beating_benchmark(self) -> int:
        """How many variants out-performed the benchmark on average."""
        return int(self.spa.n_beating_benchmark)

    # ----------------------------------------------------------------- verdict

    @property
    def label(self) -> str:
        return self.verdict.label

    def summary(self) -> dict[str, Any]:
        """A flat, JSON-friendly digest. Every value traced to a library call."""
        return {
            "title": self.title,
            "provenance": self.provenance,
            "synthetic": self.synthetic,
            "benchmark": self.benchmark_name,
            "n_trials": self.n_trials,
            "n_periods": self.n_periods,
            "years": self.years,
            "winner": self.winner_label,
            "winner_sharpe": self.winner_sharpe,
            "winner_total_return": self.winner_total_return,
            "winner_max_drawdown": self.winner_max_drawdown,
            "benchmark_sharpe": self.benchmark_sharpe,
            "benchmark_total_return": self.benchmark_total_return,
            "dsr_hurdle": self.dsr_hurdle,
            "n_trials_above_expected_max": self.n_trials_above_expected_max,
            "n_trials_above_hurdle": self.n_trials_above_hurdle,
            "n_beating_benchmark": self.n_beating_benchmark,
            "label": self.label,
            "psr": self.psr.as_dict(),
            "dsr": self.dsr.as_dict(),
            "pbo": self.pbo.as_dict(),
            "spa": self.spa.as_dict(),
        }


def _trial_sharpes(trials: TrialMatrix) -> FloatArray:
    """Annualised Sharpe of every trial, in one vectorised pass."""
    mean = trials.values.mean(axis=1)
    sd = trials.values.std(axis=1, ddof=1)
    ratio = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0.0)
    return np.asarray(ratio * np.sqrt(trials.periods_per_year), dtype=np.float64)


def analyse(
    trials: TrialMatrix,
    *,
    benchmark: float | FloatArray | ReturnSeries = 0.0,
    title: str = "Backtest luck assessment",
    provenance: str = "",
    synthetic: bool = False,
    cost_bps: float | None = None,
    dates: Any = None,
    n_blocks: int = DEFAULT_N_BLOCKS,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
    risk_free_rate: float = 0.0,
) -> Analysis:
    """Run all four statistics over one family of trials and combine them.

    Parameters
    ----------
    trials:
        Every variant that was tried. The winner is taken as the argmax of the
        realised annualised Sharpe, which is what a backtester reporting a single
        result would have selected.
    benchmark:
        Passed through to :func:`~luckdetector.stats.reality_check.reality_check`.
        A scalar ``0.0`` asks the soft question ("did anything make money"); a
        return series asks whether the search beat the thing you could have done
        instead. On SPY those give p = 0.24 and p = 1.00 respectively. Pass a
        named :class:`~luckdetector.types.ReturnSeries` to control how it is
        described — the name is read back off the Reality Check result so the
        report and the interpretation strings cannot drift apart.
    seed:
        Threaded into PBO's split sampling and the bootstrap. Same seed, same
        report.

    Returns
    -------
    Analysis
        Carrying the four results, the verdict, and the derived quantities the
        figures need.
    """
    sharpes = _trial_sharpes(trials)
    winner_index = int(np.argmax(sharpes))
    winner = trials.trial(winner_index)

    psr_result = probabilistic_sharpe_ratio(winner, risk_free_rate=risk_free_rate)
    dsr_result = deflated_sharpe_ratio_from_trials(
        trials, index=winner_index, risk_free_rate=risk_free_rate
    )
    pbo_result = probability_of_backtest_overfitting(
        trials, n_blocks=n_blocks, risk_free_rate=risk_free_rate, seed=seed
    )
    spa_result = reality_check(trials, benchmark, n_resamples=n_resamples, seed=seed)

    if isinstance(benchmark, ReturnSeries):
        benchmark_values = benchmark.values
    else:
        benchmark_values = np.broadcast_to(
            np.asarray(benchmark, dtype=np.float64), (trials.n_periods,)
        ).astype(np.float64)

    return Analysis(
        trials=trials,
        benchmark_returns=np.ascontiguousarray(benchmark_values),
        # Read back off the Reality Check rather than accepted as a parameter, so
        # the table header can never disagree with the interpretation strings.
        benchmark_name=spa_result.benchmark_name,
        trial_sharpes=sharpes,
        winner_index=winner_index,
        psr=psr_result,
        dsr=dsr_result,
        pbo=pbo_result,
        spa=spa_result,
        verdict=assess(psr=psr_result, dsr=dsr_result, pbo=pbo_result, spa=spa_result),
        title=title,
        provenance=provenance,
        synthetic=synthetic,
        cost_bps=cost_bps,
        dates=dates,
    )


def analyse_mined(
    result: MiningResult,
    *,
    against_buy_and_hold: bool = True,
    **kwargs: Any,
) -> Analysis:
    """:func:`analyse` for the output of :func:`luckdetector.mining.mine`.

    Defaults to judging the family against **buy-and-hold** rather than against
    zero, because that is the comparison a sceptic actually cares about and the
    one the SPY result turns on: 0 of 157 variants beat it. Pass
    ``against_buy_and_hold=False`` for the softer test against zero.
    """
    benchmark: float | ReturnSeries
    if against_buy_and_hold:
        benchmark = ReturnSeries(
            values=result.buy_and_hold,
            periods_per_year=result.trials.periods_per_year,
            name="buy-and-hold",
        )
    else:
        benchmark = 0.0
    kwargs.setdefault("cost_bps", result.cost_bps)
    return analyse(result.trials, benchmark=benchmark, **kwargs)
