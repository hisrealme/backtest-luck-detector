"""One self-contained HTML file: verdict, evidence, figures, caveats.

Self-contained is a requirement rather than a nicety. Both figures are embedded
as base64 PNGs, the CSS is inline, and there is no JavaScript and no CDN — so the
file opens correctly from a USB stick on a machine with no network, which is the
only definition of "reproducible artefact" that survives contact with a
compliance department.

What the template does *not* do
-------------------------------
It derives nothing. Every statistic, threshold and interpretation string is
rendered straight off :class:`~luckdetector.report.analysis.Analysis` and the
:class:`~luckdetector.types.TestResult` objects inside it, in
:data:`~luckdetector.report.verdict.TEST_ORDER`. A template that re-computed even
a percentage would be a second implementation of the statistics with no tests
against it.

``NOT_APPLICABLE`` gets its own treatment
-----------------------------------------
Not a pass, not a failure, and emphatically not greyed out. On the SPY report the
inapplicable row carries the strongest sentence in the whole analysis — *not one
of the 157 strategies beat buy-and-hold* — and a template that styled it as a
muted footnote would bury the best finding in the project under its own
formatting. It is rendered as a callout with its own colour, above the fold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment

from ..stats.dsr import DSR_THRESHOLD
from ..types import TestResult
from .analysis import Analysis
from .plots import COLOURS, cumulative_return_figure, figure_to_base64, trial_sharpe_figure
from .verdict import TEST_TITLES

__all__ = [
    "CAVEATS",
    "render_report",
    "write_report",
]

#: The limitations that travel with every report, each one measured elsewhere in
#: the project rather than offered as a generic disclaimer. Kept in code rather
#: than in the template so they can be asserted on in tests.
CAVEATS: tuple[tuple[str, str], ...] = (
    (
        "The window flatters the benchmark",
        "2010–2026 was a historic bull market, so trend rules that go flat or short "
        "necessarily give up ground. This is evidence about this family of rules over "
        "this window, not a verdict on trend following in general.",
    ),
    (
        "PBO has no purge or embargo",
        "The cross-validation splits the record into contiguous blocks with no gap "
        "between the in-sample and out-of-sample halves, so a rule with a 250-day "
        "lookback is contaminated near each seam. This makes the winner look more "
        "persistent than it is, which means the true probability of overfitting is "
        "likely worse than the figure reported above, not better.",
    ),
    (
        "A luck verdict is weak evidence of absence",
        "Measured across 25 independent datasets, a genuine annualised Sharpe of 2.0 "
        "planted in 10 of 50 variants over five years is called LIKELY_SKILL only 20% "
        "of the time; it takes 5 of 50 at Sharpe 3.0 over ten years to reach 100%. The "
        "Deflated Sharpe Ratio is the binding constraint at every effect size. This "
        "instrument is far better at catching luck than at certifying skill, and the "
        "asymmetry is deliberate.",
    ),
    (
        "The combination rules are invented",
        "The four statistics are each derived from a published result and checked "
        "against simulation. The rule table that combines them into one label is not — "
        "there is no literature on weighing a Deflated Sharpe Ratio against a PBO. "
        "Three of the four thresholds are conventional; the precedence is a judgement. "
        "See docs/METHODS.md §9.",
    ),
)

_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "PASS": ("pass", "PASS"),
    "FAIL": ("fail", "FLAGGED"),
    "NOT_APPLICABLE": ("na", "NOT APPLICABLE"),
}

_VERDICT_STYLE: dict[str, tuple[str, str]] = {
    "LIKELY_LUCK": ("luck", "This backtest is most likely luck."),
    "LIKELY_SKILL": ("skill", "This backtest survived every test that prices the search."),
    "INCONCLUSIVE": ("inconclusive", "The evidence does not settle the question."),
}

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
  :root {
    --ink: #1c2833; --muted: #5d6d7e; --rule: #d5dbdb; --paper: #ffffff;
    --pass: #1e8449; --fail: #c0392b; --na: #b7791f; --wash: #f4f6f7;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 2.2rem 1.2rem 4rem; background: var(--wash); color: var(--ink);
         font: 16px/1.62 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               Helvetica, sans-serif; }
  main { max-width: 980px; margin: 0 auto; background: var(--paper); padding: 2.4rem 2.6rem 3rem;
         border: 1px solid var(--rule); border-radius: 6px; }
  h1 { font-size: 1.85rem; margin: 0 0 .35rem; letter-spacing: -.01em; }
  h2 { font-size: 1.18rem; margin: 2.6rem 0 .8rem; padding-bottom: .35rem;
       border-bottom: 1px solid var(--rule); }
  .provenance { color: var(--muted); font-size: .92rem; margin: 0 0 1.6rem; }
  .synthetic { background: #fdebd0; border: 1px solid #d35400; color: #8c3d00;
               padding: .9rem 1.1rem; border-radius: 4px; margin: 0 0 1.6rem;
               font-weight: 600; letter-spacing: .02em; }
  .banner { padding: 1.3rem 1.5rem; border-radius: 5px; margin: 0 0 1.4rem;
            border-left: 6px solid; }
  .banner .label { font-size: 1.5rem; font-weight: 700; letter-spacing: .01em; }
  .banner .rule { margin-top: .35rem; color: var(--muted); font-size: .95rem; }
  .banner.luck { background: #fdedec; border-color: var(--fail); }
  .banner.luck .label { color: var(--fail); }
  .banner.skill { background: #eafaf1; border-color: var(--pass); }
  .banner.skill .label { color: var(--pass); }
  .banner.inconclusive { background: #fef9e7; border-color: var(--na); }
  .banner.inconclusive .label { color: var(--na); }
  table { width: 100%; border-collapse: collapse; margin: .4rem 0 1.2rem; font-size: .95rem; }
  th, td { text-align: left; padding: .62rem .7rem; border-bottom: 1px solid var(--rule); }
  th { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
  td.num { text-align: right; font-variant-numeric: tabular-nums;
           font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .tag { display: inline-block; padding: .16rem .55rem; border-radius: 3px;
         font-size: .74rem; font-weight: 700; letter-spacing: .05em; }
  .tag.pass { background: #eafaf1; color: var(--pass); }
  .tag.fail { background: #fdedec; color: var(--fail); }
  .tag.na   { background: #fef5e7; color: var(--na); }
  .evidence { margin: 1.4rem 0; padding: 1rem 1.2rem; border-left: 4px solid var(--rule);
              background: #fbfcfc; border-radius: 0 4px 4px 0; }
  .evidence.pass { border-color: var(--pass); }
  .evidence.fail { border-color: var(--fail); }
  .evidence.na { border-color: var(--na); background: #fffdf6; }
  .evidence h3 { margin: 0 0 .3rem; font-size: 1rem; }
  .evidence h3 .tag { margin-left: .5rem; vertical-align: 2px; }
  .evidence p { margin: .35rem 0 0; }
  .evidence.na p { font-size: 1.02rem; }
  figure { margin: 1.6rem 0; }
  figure img { width: 100%; height: auto; display: block; border: 1px solid var(--rule);
               border-radius: 4px; }
  figcaption { color: var(--muted); font-size: .88rem; margin-top: .55rem; }
  .caveat { margin: .9rem 0; }
  .caveat strong { display: block; }
  .caveat span { color: var(--muted); font-size: .93rem; }
  footer { margin-top: 2.6rem; padding-top: 1rem; border-top: 1px solid var(--rule);
           color: var(--muted); font-size: .85rem; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }
</style>
</head>
<body>
<main>
  <h1>{{ title }}</h1>
  <p class="provenance">{{ provenance }}</p>

  {% if synthetic %}
  <div class="synthetic">
    SYNTHETIC DATA — every number and every figure below was computed from a generated
    price path, not from a market. This output demonstrates the machinery; it is not a
    result about any real instrument.
  </div>
  {% endif %}

  <div class="banner {{ verdict_class }}">
    <div class="label">{{ label_text }}</div>
    <div class="rule">{{ verdict_sentence }} {{ rule_sentence }}</div>
  </div>

  <h2>What was tested</h2>
  <table>
    <tbody>
      {% for label, value in facts %}
      <tr><th>{{ label }}</th><td class="num">{{ value }}</td></tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>The evidence</h2>
  <table>
    <thead>
      <tr><th>Test</th><th>Statistic</th><th>Threshold</th><th>Status</th></tr>
    </thead>
    <tbody>
      {% for row in rows %}
      <tr>
        <td>{{ row.title }}</td>
        <td class="num">{{ row.statistic }}</td>
        <td class="num">{{ row.threshold }}</td>
        <td><span class="tag {{ row.css }}">{{ row.badge }}</span></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  {% for row in rows %}
  <div class="evidence {{ row.css }}">
    <h3>{{ row.title }}<span class="tag {{ row.css }}">{{ row.badge }}</span></h3>
    <p>{{ row.interpretation }}</p>
  </div>
  {% endfor %}

  <h2>Figures</h2>
  <figure>
    <img alt="Cumulative return of the winner against the benchmark"
         src="data:image/png;base64,{{ figure_cumulative }}">
    <figcaption>{{ caption_cumulative }}</figcaption>
  </figure>
  <figure>
    <img alt="Trial Sharpe distribution against the deflation hurdle"
         src="data:image/png;base64,{{ figure_sharpes }}">
    <figcaption>{{ caption_sharpes }}</figcaption>
  </figure>

  <h2>Caveats</h2>
  {% for heading, body in caveats %}
  <div class="caveat"><strong>{{ heading }}</strong><span>{{ body }}</span></div>
  {% endfor %}

  <footer>
    Generated by luckdetector {{ version }}. Every number above was produced by a public
    function in <code>luckdetector.stats</code>; nothing in this document was computed by
    the template. Method and thresholds: <code>docs/METHODS.md</code>.
  </footer>
</main>
</body>
</html>
"""


def _format_cells(result: TestResult) -> tuple[str, str]:
    """The statistic and threshold cells for one row.

    An inapplicable test gets an em dash in both, and that is not cosmetic. When
    SPA drops out because nothing beat the benchmark, its ``statistic`` field
    holds :math:`V_{SPA} = \\max(0, \\cdot) = 0` — the *test statistic*, not a
    p-value. Rendering it in a column headed "statistic" next to a 0.05
    threshold produces the line ``p = 0.0000 vs 0.0500``, which reads as the most
    decisive rejection in the document at the exact moment the test found
    nothing to weigh. That is the misreading :class:`TestResult` was given a
    third state to prevent, and a table is the easiest place to reintroduce it.

    There is no comparison to render, so none is rendered. The interpretation
    paragraph below the table carries the p-value of 1.0000 together with the
    reason it means nothing.
    """
    if result.status == "NOT_APPLICABLE":
        return "—", "—"
    if result.p_value is not None:
        return f"p = {result.statistic:.4f}", f"{result.threshold:.4f}"
    return f"{result.statistic:.4f}", f"{result.threshold:.4f}"


def _rows(analysis: Analysis) -> list[dict[str, str]]:
    """One row per test, in ``TEST_ORDER``, derived from nothing."""
    rows: list[dict[str, str]] = []
    for result in analysis.verdict.results:
        css, badge = _STATUS_STYLE[result.status]
        statistic, threshold = _format_cells(result)
        rows.append(
            {
                "name": result.name,
                "title": TEST_TITLES.get(result.name, result.name),
                "statistic": statistic,
                "threshold": threshold,
                "css": css,
                "badge": badge,
                "interpretation": result.interpretation,
            }
        )
    return rows


def _facts(analysis: Analysis) -> list[tuple[str, str]]:
    """The "what was tested" block, formatted in Python rather than in the template.

    Jinja can format numbers, but doing it there puts presentation logic somewhere
    no test can reach it and no type checker can see it. The template loops; it
    does not compute.
    """
    benchmark_sharpe = analysis.benchmark_sharpe
    facts: list[tuple[str, str]] = [
        ("Strategies tried", f"{analysis.n_trials:,}"),
        ("Reported winner", analysis.winner_label),
        ("Winner annualised Sharpe", f"{analysis.winner_sharpe:.3f}"),
        ("Winner total return", f"{analysis.winner_total_return:+.1%}"),
        ("Winner max drawdown", f"{analysis.winner_max_drawdown:.1%}"),
        ("Benchmark", analysis.benchmark_name),
        # A benchmark of literally zero has no volatility, so its Sharpe is
        # undefined rather than infinite; nan is printed as an em dash.
        (
            "Benchmark annualised Sharpe",
            "—" if benchmark_sharpe != benchmark_sharpe else f"{benchmark_sharpe:.3f}",
        ),
        ("Benchmark total return", f"{analysis.benchmark_total_return:+.1%}"),
        (
            "Variants beating the benchmark",
            f"{analysis.n_beating_benchmark} of {analysis.n_trials}",
        ),
        ("Observations", f"{analysis.n_periods:,} ({analysis.years:.1f} years)"),
        ("Expected max Sharpe of noise", f"{analysis.dsr.expected_max_sharpe_annual:.3f}"),
        ("Sharpe needed to clear DSR", f"{analysis.dsr_hurdle:.3f}"),
    ]
    if analysis.cost_bps is not None:
        facts.append(("Transaction cost", f"{analysis.cost_bps:.1f} bp per unit turnover"))
    return facts


def _captions(analysis: Analysis) -> tuple[str, str]:
    """Figure captions, stated in terms of the numbers actually plotted."""
    cumulative = (
        f"{analysis.winner_label} compounded {analysis.winner_total_return:+.1%} over "
        f"{analysis.years:.1f} years against {analysis.benchmark_total_return:+.1%} for "
        f"{analysis.benchmark_name}, with a maximum drawdown of "
        f"{analysis.winner_max_drawdown:.1%}. Log scale."
    )
    gap = analysis.winner_sharpe - analysis.dsr_hurdle
    sharpes = (
        f"The winner is the maximum of the {analysis.n_trials} trials by construction, so it "
        f"sits at the right edge of the grey histogram rather than inside it. It clears the "
        f"expected maximum of noise ({analysis.dsr.expected_max_sharpe_annual:.3f}) — as do "
        f"{analysis.n_trials_above_expected_max} of {analysis.n_trials} variants — but that is "
        f"not the bar the Deflated Sharpe Ratio applies. Allowing for the "
        f"{analysis.dsr.psr_result.standard_error_annual:.3f} standard error on the winner's "
        f"own Sharpe, {DSR_THRESHOLD:.0%} confidence requires {analysis.dsr_hurdle:.3f}; the "
        f"winner posted {analysis.winner_sharpe:.3f}, a gap of {gap:+.3f}. The shaded area is "
        f"the DSR itself, {analysis.dsr.dsr:.4f}."
    )
    return cumulative, sharpes


def render_report(analysis: Analysis, *, version: str | None = None) -> str:
    """Render the full report to a single HTML string.

    Parameters
    ----------
    analysis:
        From :func:`luckdetector.report.analysis.analyse`.
    version:
        Stamped into the footer. Defaults to the installed package version.

    Returns
    -------
    str
        A complete document with both figures embedded. No external assets, no
        JavaScript, no network access required to open it.
    """
    if version is None:
        from .. import __version__

        version = __version__

    verdict_class, verdict_sentence = _VERDICT_STYLE[analysis.verdict.label]
    rule_sentence = analysis.verdict.narrative.split("\n", 1)[0]

    caption_cumulative, caption_sharpes = _captions(analysis)

    context: dict[str, Any] = {
        "title": analysis.title,
        "provenance": analysis.provenance,
        "synthetic": analysis.synthetic,
        "verdict_class": verdict_class,
        "verdict_sentence": verdict_sentence,
        "label_text": analysis.verdict.label.replace("_", " "),
        "rule_sentence": rule_sentence,
        "facts": _facts(analysis),
        "rows": _rows(analysis),
        "figure_cumulative": figure_to_base64(cumulative_return_figure(analysis)),
        "figure_sharpes": figure_to_base64(trial_sharpe_figure(analysis)),
        "caption_cumulative": caption_cumulative,
        "caption_sharpes": caption_sharpes,
        "caveats": CAVEATS,
        "version": version,
        "colours": COLOURS,
    }

    environment = Environment(autoescape=True)
    return environment.from_string(_TEMPLATE).render(**context)


def write_report(analysis: Analysis, path: str | Path, *, version: str | None = None) -> Path:
    """Render and write the report, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(analysis, version=version), encoding="utf-8")
    return destination
