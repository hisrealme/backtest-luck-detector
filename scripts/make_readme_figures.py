"""Regenerate the two figures the README opens with.

The README is the only place in the project that shows a figure without a
numbered caption beside it, so both figures are drawn with ``title=True`` — the
default — and that is what the default exists for. Inside the HTML report the
titles are suppressed instead, because there each figure is printed directly
above its own caption and a heading a centimetre above the caption restates it.

The images are committed to ``docs/figures/``. They have to be: ``outputs/`` is
gitignored, so nothing generated there reaches a reader who clones the repo, and
a README that links to an image which is not in the repository shows a broken
image on GitHub. That is the whole reason this script writes where it does.

**The prices are real.** The figures come from the same cached SPY history
``luckdet demo`` resolves, through the same :class:`~luckdetector.report.
analysis.Analysis` object, so the numbers printed on them are the published
numbers rather than a second computation that could drift. Running this without
that cache and without a network will fail loudly, which is the behaviour
``resolve_demo_prices`` is designed for — a README figure quietly drawn from a
random number generator would be the exact error this package exists to catch.

    make figures        # or: python scripts/make_readme_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from luckdetector.exceptions import LuckDetectorError
from luckdetector.report.demo import real_data_analysis, resolve_demo_prices
from luckdetector.report.plots import cumulative_return_figure, trial_sharpe_figure

DESTINATION = Path("docs/figures")

#: Higher than the report's inline DPI. The report embeds its PNGs as base64 and
#: pays for every pixel in file size; the README's are served by GitHub and are
#: read on displays that will happily use the extra resolution.
DPI = 160


def main() -> int:
    try:
        history = resolve_demo_prices()
    except LuckDetectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    analysis = real_data_analysis(history)
    DESTINATION.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("cumulative_return.png", cumulative_return_figure(analysis)),
        ("deflation_hurdle.png", trial_sharpe_figure(analysis)),
    ):
        path = DESTINATION / name
        figure.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"wrote {path}")

    print(
        f"\n{analysis.provenance}\n"
        f"winner {analysis.winner_label} at {analysis.winner_sharpe:.3f}, "
        f"DSR {analysis.dsr.dsr:.4f}, hurdle {analysis.dsr_hurdle:.3f}, "
        f"verdict {analysis.label}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
