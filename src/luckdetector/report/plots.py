"""The two figures, and why the second one is not the figure that was specified.

Both use the object-oriented matplotlib API and never import ``pyplot``. That is
deliberate: ``pyplot`` keeps a global registry of figures, which in a library
means every report leaks a figure until the process exits, and it selects a
backend by inspecting the environment, which in a test suite means the same code
renders differently on a developer's laptop and in CI. Constructing
:class:`~matplotlib.figure.Figure` directly has neither problem.

Figure 1 — cumulative return
----------------------------
The winner against the benchmark, on one pair of axes. On SPY this is the
"+205.6% looks superb until you see +814.3%" picture, and it needs no commentary.

Figure 2 — the deflation hurdle, drawn correctly
------------------------------------------------
The Phase 8 brief specified this as *the distribution of all 157 trial Sharpes,
with the expected-max-of-noise hurdle (0.309) and the winner (0.491) marked on
it*, described as "the whole thesis in one image: the winner sits inside the
noise distribution."

**Drawn literally, that image argues the opposite of the verdict**, for two
reasons that are visible the moment it is plotted:

1. The winner is the *maximum* of the 157 Sharpes by construction, so it sits at
   the extreme right edge of the plotted distribution — at the 100th percentile,
   never inside the mass. There is no arrangement of that histogram in which the
   marker falls among the others.
2. The winner (0.4905) is **above** the expected-max hurdle (0.3086), and so are
   42 other variants — 43 of 157, or 27%. A reader shown one line at 0.309 and
   one at 0.491 reads "clears the bar", while the Deflated Sharpe Ratio of 0.769
   says luck.

Both are real and neither is a drafting problem. The reconciliation is that the
expected maximum of noise is a *point estimate of a hurdle*, while the winner's
Sharpe is an *estimate with a standard error* — 0.247 on SPY. Being 0.18 above
the hurdle is only 0.74 standard errors above it, and 0.95 confidence needs
1.645.

So this figure draws what the test actually does: the sampling distribution of
the winner's Sharpe under the null that its true edge is exactly the
best-of-noise hurdle, with everything at or below the observed winner shaded.
**That shaded area is the Deflated Sharpe Ratio**, readable off the picture, and
the 0.95 bar sits where :func:`~luckdetector.stats.dsr.sharpe_required_for_dsr`
puts it — 0.715 on SPY, to the right of every variant in the family.

The histogram of the 157 trial Sharpes is kept underneath, because the spread of
the family is what sets the hurdle in the first place and dropping it would hide
the input to the test. It is simply no longer asked to carry a claim it cannot
support.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import stats as sps

from ..stats.dsr import DSR_THRESHOLD
from ..types import FloatArray
from .analysis import Analysis

__all__ = [
    "COLOURS",
    "cumulative_return_figure",
    "figure_to_base64",
    "trial_sharpe_figure",
]

#: One palette, defined once, so the two figures and the HTML agree. Chosen to
#: stay distinguishable in greyscale and for the red/green pair to remain
#: separable under the common forms of colour blindness.
COLOURS: dict[str, str] = {
    "winner": "#c0392b",
    "benchmark": "#2c3e50",
    "family": "#95a5a6",
    "null": "#2980b9",
    "hurdle": "#8e44ad",
    "expected_max": "#7f8c8d",
    "synthetic": "#d35400",
}

_FIGSIZE = (9.0, 4.8)
_DPI = 110


@contextmanager
def _paper_type() -> Iterator[None]:
    """Set figure type to match the report's body text, for the duration.

    The report is set in a serif face and the figures sit inches from it, so
    matplotlib's sans-serif default reads as a foreign object dropped into the
    page. DejaVu Serif ships with matplotlib, which means this adds no font
    dependency and cannot resolve differently in CI.

    **The rc dict is written inline rather than hoisted to a module constant, and
    that is not a style choice.** matplotlib 3.11 types ``rc_context``'s parameter
    as a ``dict`` keyed by a ``Literal`` union of all ~300 rcParam names. A
    constant annotated ``dict[str, Any]`` is not assignable to it — ``dict`` is
    invariant in its key type — so hoisting it fails ``mypy --strict`` on every
    Python that resolves matplotlib 3.11, while passing on 3.10, which caps at
    matplotlib 3.10 and ships the looser ``dict[str, Any]`` signature. That is
    exactly how it reached CI green locally and red on four of five jobs.

    Passing the literal directly lets mypy check each key against whatever union
    matplotlib currently declares, which type-checks under both signatures and
    turns a future renamed rcParam into an error here rather than a silently
    ignored setting. Annotating it ``dict[Any, Any]`` would also pass, but by
    switching the check off rather than satisfying it.
    """
    with rc_context(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    ):
        yield


def _new_figure(synthetic: bool) -> tuple[Any, Any]:
    """A figure and its single axes, with the SYNTHETIC stamp applied if needed."""
    figure = Figure(figsize=_FIGSIZE, dpi=_DPI, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    if synthetic:
        # Deliberately loud. The offline path must never produce an image that
        # could be mistaken for a result about a real market.
        axes.text(
            0.5,
            0.5,
            "SYNTHETIC",
            transform=axes.transAxes,
            fontsize=52,
            color=COLOURS["synthetic"],
            alpha=0.18,
            ha="center",
            va="center",
            rotation=24,
            zorder=5,
            fontweight="bold",
        )
    return figure, axes


def _x_axis(analysis: Analysis, n: int) -> tuple[Any, str]:
    """Dates when the analysis carries them, period index otherwise."""
    dates = analysis.dates
    if dates is not None and len(dates) >= n:
        return np.asarray(dates[-n:]), "Date"
    return np.arange(n), "Trading period"


def cumulative_return_figure(analysis: Analysis, *, title: bool = True) -> Any:
    """Cumulative wealth: the winner against the benchmark it was competing with.

    Both lines start at 1.0 and compound the same periods, so the vertical gap at
    any point is the whole of the comparison. Drawn on a log scale because a
    sixteen-year equity curve spanning 1x to 9x compresses the early years into
    invisibility on a linear axis, and the early years are where the winner is
    ahead.

    ``title=False`` drops the heading, which is what the HTML report asks for: a
    figure printed directly above its own numbered caption should not restate the
    caption. The default is ``True`` so the figure still stands on its own when
    it is used outside the report.
    """
    with _paper_type():
        return _cumulative_return_figure(analysis, title=title)


def _cumulative_return_figure(analysis: Analysis, *, title: bool) -> Any:
    figure, axes = _new_figure(analysis.synthetic)

    winner_curve = analysis.winner.cumulative()
    benchmark_curve = analysis.benchmark_series.cumulative()
    x, x_label = _x_axis(analysis, winner_curve.size)

    axes.plot(
        x,
        winner_curve,
        color=COLOURS["winner"],
        linewidth=1.6,
        label=f"{analysis.winner_label}  ({analysis.winner_total_return:+.1%})",
    )
    axes.plot(
        x,
        benchmark_curve,
        color=COLOURS["benchmark"],
        linewidth=1.6,
        label=f"{analysis.benchmark_name}  ({analysis.benchmark_total_return:+.1%})",
    )

    if np.all(benchmark_curve > 0.0) and np.all(winner_curve > 0.0):
        axes.set_yscale("log")

    if title:
        axes.set_title("Cumulative return: the reported winner against doing nothing")
    axes.set_xlabel(x_label)
    axes.set_ylabel("Growth of 1.0 (log scale)")
    axes.legend(loc="upper left", frameon=False)
    axes.grid(True, alpha=0.25, linewidth=0.6)
    return figure


def trial_sharpe_figure(analysis: Analysis, *, title: bool = True) -> Any:
    """The deflation hurdle, drawn as the test applies it.

    See the module docstring for why this is not the figure the Phase 8 brief
    specified. The short version: the winner is the maximum of the family by
    construction and sits above the expected-max-of-noise line, so a plot of
    those two things alone reads as a pass while the Deflated Sharpe Ratio calls
    it luck.

    What is drawn instead:

    * the histogram of every trial's realised Sharpe, which is the *input* to the
      hurdle rather than the test;
    * the sampling distribution of the winner's Sharpe under the null that its
      true edge equals the best-of-noise hurdle, shaded up to the observed
      winner — **the shaded area is the DSR**;
    * the winner, the expected maximum of noise, and the Sharpe the winner would
      have needed to clear 0.95.

    ``title=False`` drops the heading for use directly above a numbered caption.
    """
    with _paper_type():
        return _trial_sharpe_figure(analysis, title=title)


def _trial_sharpe_figure(analysis: Analysis, *, title: bool) -> Any:
    figure, axes = _new_figure(analysis.synthetic)

    sharpes: FloatArray = analysis.trial_sharpes
    winner = analysis.winner_sharpe
    expected_max = analysis.dsr.expected_max_sharpe_annual
    hurdle = analysis.dsr_hurdle
    sigma = analysis.dsr.psr_result.standard_error_annual

    lo = float(min(sharpes.min(), expected_max - 3.2 * sigma))
    hi = float(max(sharpes.max(), hurdle, expected_max + 3.2 * sigma))
    pad = 0.06 * (hi - lo)

    counts, _, _ = axes.hist(
        sharpes,
        bins=min(40, max(10, sharpes.size // 4)),
        density=True,
        color=COLOURS["family"],
        alpha=0.55,
        label=f"{analysis.n_trials} trials actually run",
    )

    grid = np.linspace(lo - pad, hi + pad, 400)
    null = np.asarray(sps.norm.pdf(grid, loc=expected_max, scale=sigma), dtype=np.float64)
    axes.plot(
        grid,
        null,
        color=COLOURS["null"],
        linewidth=1.8,
        label="winner's Sharpe if its true edge were the noise hurdle",
    )
    axes.fill_between(
        grid,
        null,
        where=grid <= winner,
        color=COLOURS["null"],
        alpha=0.22,
        label=f"DSR = {analysis.dsr.dsr:.3f} (this area)",
    )

    axes.axvline(
        expected_max,
        color=COLOURS["expected_max"],
        linestyle=":",
        linewidth=1.6,
        label=f"expected max of noise {expected_max:.3f}",
    )
    axes.axvline(
        winner,
        color=COLOURS["winner"],
        linestyle="-",
        linewidth=1.8,
        label=f"winner {winner:.3f}",
    )
    axes.axvline(
        hurdle,
        color=COLOURS["hurdle"],
        linestyle="--",
        linewidth=1.8,
        label=f"Sharpe needed for DSR {DSR_THRESHOLD:.2f}: {hurdle:.3f}",
    )

    if title:
        verdict = "short of" if winner < hurdle else "clear of"
        axes.set_title(
            f"The winner is {verdict} the bar it has to clear "
            f"(gap {winner - hurdle:+.3f} annualised Sharpe)"
        )
    axes.set_xlabel("Annualised Sharpe ratio")
    axes.set_ylabel("Density")
    axes.set_xlim(lo - pad, hi + pad)
    # Headroom so the six-entry legend clears the tallest bars rather than
    # sitting on top of the data it is describing.
    peak = float(max(np.max(counts), np.max(null)))
    axes.set_ylim(0.0, 1.45 * peak)
    axes.legend(loc="upper left", frameon=False, fontsize=8)
    axes.grid(True, alpha=0.25, linewidth=0.6)
    return figure


def figure_to_base64(figure: Any, *, dpi: int = _DPI) -> str:
    """Render to a base64 PNG payload for inline embedding.

    Returned without the ``data:image/png;base64,`` prefix so the template
    controls the URI. Embedding rather than linking is what makes the report a
    single file that still opens correctly with no network access.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
