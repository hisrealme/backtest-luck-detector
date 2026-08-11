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
formatting. It gets a numbered subsection of its own like every other test, and
its table row shows no comparison at all rather than a misleading one (§11.3).

Why it is laid out as a paper
-----------------------------
Single column, serif, justified, numbered sections, tables ruled top-mid-bottom
with no verticals, captions above tables and below figures, and near enough to
monochrome. The format is borrowed from a conference paper because the document
makes the same kind of claim: here is what was measured, here is the evidence,
here is what would change the conclusion. A dashboard aesthetic — status
chips, coloured cards, a big red banner — invites the reader to skim to the
verdict and stop, which is the exact behaviour that produces a 0.9764 PSR and a
published strategy. Paper typography is slower on purpose.

Status is therefore carried by weight and small caps rather than by colour, and
exactly one accent is used per document, tied to the verdict. The figures keep
their own palette; everything else is black on white.
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

#: ``css class`` → ``(short badge, prose used in the table cell)``. Status is
#: carried by weight and small caps rather than colour; only ``flagged`` picks up
#: the document's single accent.
_STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    "PASS": ("pass", "passed", "passed"),
    "FAIL": ("fail", "flagged", "flagged"),
    "NOT_APPLICABLE": ("na", "not applicable", "not applicable"),
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
    --ink: #111111; --muted: #55595e; --rule: #111111; --hair: #c9ccd0;
    --paper: #ffffff; --desk: #e9e8e4; --accent: #7d2b2b;
  }
  .v-skill { --accent: #1d5c38; }
  .v-inconclusive { --accent: #7a5c14; }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body {
    margin: 0; padding: 2.5rem 1rem 5rem; background: var(--desk); color: var(--ink);
    font-family: "Libertinus Serif", "Linux Libertine O", "Palatino Linotype",
                 Palatino, "Book Antiqua", Georgia, "Times New Roman", serif;
    font-size: 16px; line-height: 1.52;
  }
  article {
    max-width: 44rem; margin: 0 auto; background: var(--paper);
    padding: 4.2rem 4.4rem 3.4rem; border: 1px solid var(--hair);
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }
  p { margin: 0 0 .72rem; text-align: justify; hyphens: auto; }
  .runhead {
    margin: 0 0 2.2rem; text-align: center; font-size: .7rem; letter-spacing: .16em;
    text-transform: uppercase; color: var(--muted);
  }
  h1 {
    margin: 0 0 .7rem; text-align: center; font-size: 1.72rem; line-height: 1.24;
    font-weight: 700; letter-spacing: -.005em;
  }
  .byline {
    margin: 0 0 1.8rem; text-align: center; font-size: .87rem; color: var(--muted);
  }
  .synthetic {
    margin: 0 0 1.7rem; padding: .55rem .8rem; text-align: center;
    border: 1px solid var(--ink); font-size: .82rem; letter-spacing: .04em;
  }
  .synthetic b { font-weight: 700; letter-spacing: .12em; }
  .abstract {
    margin: 0 0 1.9rem; padding: 0 2.1rem; font-size: .93rem;
  }
  .abstract h2 {
    margin: 0 0 .4rem; text-align: center; font-size: .8rem; letter-spacing: .14em;
    text-transform: uppercase; font-weight: 700; border: 0; padding: 0;
  }
  .verdict {
    display: block; margin: 0 0 .55rem; text-align: center;
    font-size: 1.06rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: var(--accent);
  }
  h2 {
    margin: 1.9rem 0 .6rem; font-size: 1.02rem; font-weight: 700;
    border-bottom: 1px solid var(--hair); padding-bottom: .22rem;
  }
  h2 .n { display: inline-block; min-width: 1.25rem; }
  h3 { margin: 1.1rem 0 .3rem; font-size: .95rem; font-weight: 700; }
  h3 .st {
    font-weight: 700; font-size: .68rem; letter-spacing: .13em; text-transform: uppercase;
    color: var(--muted); margin-left: .5rem; vertical-align: .1em;
  }
  h3 .st.flagged { color: var(--accent); }
  h3 .st.na { color: var(--ink); border-bottom: 1px solid var(--ink); }
  table { width: 100%; border-collapse: collapse; margin: 0 0 1.3rem; font-size: .9rem; }
  caption {
    caption-side: top; text-align: left; font-size: .87rem; margin-bottom: .42rem;
  }
  caption b { font-weight: 700; }
  thead th {
    font-weight: 700; font-size: .72rem; letter-spacing: .09em; text-transform: uppercase;
  }
  th, td { padding: .3rem .5rem; text-align: left; }
  tbody tr th { font-weight: 400; }
  td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
  .toprule td, .toprule th { border-top: 1.4px solid var(--rule); }
  thead tr:last-child th { border-bottom: .7px solid var(--rule); }
  tbody tr:last-child td, tbody tr:last-child th { border-bottom: 1.4px solid var(--rule); }
  tr.flagged th, tr.flagged td { color: var(--accent); }
  figure { margin: 1.4rem 0 1.5rem; }
  figure img { display: block; width: 100%; height: auto; }
  figcaption {
    margin-top: .5rem; font-size: .85rem; text-align: justify; hyphens: auto;
  }
  figcaption b { font-weight: 700; }
  .limit { margin: 0 0 .6rem; font-size: .9rem; }
  .limit b { font-weight: 700; }
  .colophon {
    margin-top: 2.4rem; padding-top: .7rem; border-top: .7px solid var(--hair);
    font-size: .78rem; color: var(--muted); text-align: left;
  }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         font-size: .88em; }
  @media print {
    body { background: #fff; padding: 0; font-size: 10.5pt; }
    article { max-width: none; border: 0; box-shadow: none; padding: 0; }
    figure, table { break-inside: avoid; }
    h2, h3 { break-after: avoid; }
  }
  @media (max-width: 640px) {
    article { padding: 2rem 1.4rem; }
    .abstract { padding: 0; }
  }
</style>
</head>
<body>
<article class="v-{{ verdict_class }}">

  <p class="runhead">{{ runhead }}</p>
  <h1>{{ title }}</h1>
  <p class="byline">{{ provenance }}</p>

  {% if synthetic %}
  <p class="synthetic"><b>SYNTHETIC DATA</b> — generated prices, not a market. Every
     number and figure below describes a random number generator.</p>
  {% endif %}

  <section class="abstract">
    <h2>Abstract</h2>
    <span class="verdict">{{ label_text }}</span>
    <p>{{ abstract }}</p>
  </section>

  <h2><span class="n">1</span> Data and search</h2>
  <p>{{ search_paragraph }}</p>

  <table>
    <caption><b>Table 1.</b> The search, and the record it produced.</caption>
    <tbody>
      {% for label, value in facts %}
      <tr{% if loop.first %} class="toprule"{% endif %}>
        <th>{{ label }}</th><td class="n">{{ value }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <figure>
    <img alt="Cumulative return of the winner against the benchmark"
         src="data:image/png;base64,{{ figure_cumulative }}">
    <figcaption><b>Figure 1.</b> {{ caption_cumulative }}</figcaption>
  </figure>

  <h2><span class="n">2</span> Evidence</h2>
  <p>{{ evidence_paragraph }}</p>

  <table>
    <caption><b>Table 2.</b> The four tests, in the order they are evaluated. A
      flagged test is one that objected; an inapplicable one had nothing to weigh
      and is neither evidence for nor against.</caption>
    <thead>
      <tr class="toprule">
        <th>Test</th><th class="n">Statistic</th><th class="n">Threshold</th><th>Outcome</th>
      </tr>
    </thead>
    <tbody>
      {% for row in rows %}
      <tr{% if row.css == 'fail' %} class="flagged"{% endif %}>
        <th>{{ row.title }}</th>
        <td class="n">{{ row.statistic }}</td>
        <td class="n">{{ row.threshold }}</td>
        <td>{{ row.badge_text }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  {% for row in rows %}
  <h3>2.{{ loop.index }}&nbsp; {{ row.title
      }}<span class="st {{ row.css }}">{{ row.badge }}</span></h3>
  <p>{{ row.interpretation }}</p>
  {% endfor %}

  <h2><span class="n">3</span> The bar the winner had to clear</h2>
  <p>{{ hurdle_paragraph }}</p>

  <figure>
    <img alt="Trial Sharpe distribution against the deflation hurdle"
         src="data:image/png;base64,{{ figure_sharpes }}">
    <figcaption><b>Figure 2.</b> {{ caption_sharpes }}</figcaption>
  </figure>

  <h2><span class="n">4</span> Limitations</h2>
  <p>Each of the following is measured elsewhere in this project rather than
     offered as a generic disclaimer.</p>
  {% for heading, body in caveats %}
  <p class="limit"><b>{{ heading }}.</b> {{ body }}</p>
  {% endfor %}

  <p class="colophon">
    Generated by luckdetector {{ version }}. Every number above was produced by a public
    function in <code>luckdetector.stats</code>; nothing in this document was computed by
    the template. Method, thresholds and the arguments against them:
    <code>docs/METHODS.md</code>. This file is self-contained — both figures are embedded,
    and it uses no external stylesheet, font or script.
  </p>

</article>
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
        css, badge, badge_text = _STATUS_STYLE[result.status]
        statistic, threshold = _format_cells(result)
        rows.append(
            {
                "name": result.name,
                "title": TEST_TITLES.get(result.name, result.name),
                "statistic": statistic,
                "threshold": threshold,
                "css": css,
                "badge": badge,
                "badge_text": badge_text,
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


def _abstract(analysis: Analysis) -> str:
    """The verdict, restated as an abstract would state it.

    Composed here rather than in the template for the same reason as everything
    else: it quotes six numbers, and a template that formatted them would be a
    second implementation with nothing checking it.
    """
    flagged = [TEST_TITLES.get(r.name, r.name) for r in analysis.verdict.flags]
    if flagged:
        objection = (
            f"{_join(flagged)} objected"
            if len(flagged) > 1
            else f"the {flagged[0]} objected"
        )
    else:
        objection = "no test objected"

    beat = (
        f"Not one of them beat {analysis.benchmark_name}"
        if analysis.n_beating_benchmark == 0
        else f"{analysis.n_beating_benchmark} of them beat {analysis.benchmark_name}"
    )
    return (
        f"A grid of {analysis.n_trials:,} strategy variants was searched over "
        f"{analysis.n_periods:,} periods ({analysis.years:.1f} years). The best of them, "
        f"{analysis.winner_label}, posted an annualised Sharpe ratio of "
        f"{analysis.winner_sharpe:.3f} and a total return of "
        f"{analysis.winner_total_return:+.1%} — the number a backtester reporting a single "
        f"result would have shown you. {beat}. Four tests were then applied to that record: "
        f"one that knows only its length, and three that know a search took place. "
        f"{objection[0].upper()}{objection[1:]}."
    )


def _join(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` — the Oxford-free serial form."""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _prose(analysis: Analysis) -> dict[str, str]:
    """The lead paragraph under each numbered section heading."""
    naive = analysis.psr
    search = (
        "Every variant tried is kept, not just the winner: three of the four tests below "
        "are undefined without the losers, and a backtester who discards the failures has "
        "destroyed the evidence needed to judge the survivor. The winner is not nominated "
        "here — it is the highest realised Sharpe ratio in the family, which is what "
        "selecting on a backtest actually means."
    )
    evidence = (
        f"The tests are ordered weakest question first. The Probabilistic Sharpe Ratio asks "
        f"only whether the record is long enough for its own Sharpe ratio to be "
        f"distinguishable from {naive.benchmark_annual:.2f}; it knows nothing about how many "
        f"strategies were tried to find it, and it is a precondition rather than a verdict. "
        f"The three that follow each price the search in a different way — multiplicity, "
        f"selection stability, and the benchmark — so a single objection is not outvoted by "
        f"the others."
    )
    gap = analysis.winner_sharpe - analysis.dsr_hurdle
    sigma = analysis.dsr.psr_result.standard_error_annual
    over_point = analysis.winner_sharpe - analysis.dsr.expected_max_sharpe_annual
    hurdle = (
        f"The obvious bar to draw is the expected maximum of noise, "
        f"{analysis.dsr.expected_max_sharpe_annual:.3f}, and it is the wrong one. That figure "
        f"is a point estimate of a hurdle, while the winner's {analysis.winner_sharpe:.3f} is "
        f"an estimate carrying a standard error of {sigma:.3f}. Clearing the point by "
        f"{over_point:+.3f} is worth only {over_point / sigma:.2f} standard errors, where "
        f"{DSR_THRESHOLD:.0%} confidence needs 1.64. The bar that follows is "
        f"{analysis.dsr_hurdle:.3f}, which the winner "
        f"{'misses' if gap < 0 else 'clears'} by {abs(gap):.3f} and which "
        f"{analysis.n_trials_above_hurdle} of {analysis.n_trials} variants reach."
    )
    return {"search": search, "evidence": evidence, "hurdle": hurdle}


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

    verdict_class, _ = _VERDICT_STYLE[analysis.verdict.label]
    caption_cumulative, caption_sharpes = _captions(analysis)
    prose = _prose(analysis)

    context: dict[str, Any] = {
        "title": analysis.title,
        "runhead": "Backtest luck assessment",
        "provenance": analysis.provenance,
        "synthetic": analysis.synthetic,
        "verdict_class": verdict_class,
        "label_text": analysis.verdict.label.replace("_", " "),
        "abstract": _abstract(analysis),
        "search_paragraph": prose["search"],
        "evidence_paragraph": prose["evidence"],
        "hurdle_paragraph": prose["hurdle"],
        "facts": _facts(analysis),
        "rows": _rows(analysis),
        # Titles off: each figure is printed directly above its own numbered
        # caption, and a heading that restates the caption is noise.
        "figure_cumulative": figure_to_base64(cumulative_return_figure(analysis, title=False)),
        "figure_sharpes": figure_to_base64(trial_sharpe_figure(analysis, title=False)),
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
