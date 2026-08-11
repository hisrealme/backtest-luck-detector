"""Command-line interface.

Deliberately thin: **every number shown here is computed by the library**, and no
statistics live in this file. ``summary`` describes a track record; ``mine``
brute-forces a strategy grid and judges the winner; ``report`` writes the full
HTML assessment; ``demo`` runs the whole thing end to end on real prices and then
on a planted edge.

That rule is why ``report`` and ``demo`` are as short as they are. Anything they
needed that did not already exist — running all four statistics over a family,
resolving the demo's data, inverting the Deflated Sharpe Ratio — was added to the
library with its own tests rather than written inline here, where it would have
been a second implementation of the statistics with nothing checking it.

The ``luckdet`` console script is declared in ``pyproject.toml``, so this module
must remain importable at all times — otherwise a fresh ``pip install -e .``
produces a script that crashes on launch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .exceptions import LuckDetectorError
from .io import load_prices, load_returns_csv
from .mining import mine as mine_grid
from .mining import synthetic_prices
from .report.analysis import Analysis, analyse_mined
from .report.demo import run_demo, synthetic_demo_analysis
from .report.html import write_report
from .stats import (
    deflated_sharpe_ratio_from_trials,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    summarize,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="How much of your backtest is luck?",
)
console = Console()


def _error(exc: LuckDetectorError) -> None:
    """Print an error with the message escaped rather than parsed as markup.

    Rich reads ``[...]`` as a style tag, so an exception whose text contains
    square brackets is silently mangled. The one that matters:
    ``pip install -e ".[data]"`` renders as ``pip install -e "."`` — the
    instruction the user needs, with the part that makes it work removed.
    """
    console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"luckdetector {__version__}")


@app.command()
def summary(
    path: Path = typer.Argument(..., help="CSV or parquet file holding the track record."),
    column: str | None = typer.Option(None, help="Column holding returns (or prices)."),
    date_column: str | None = typer.Option(None, help="Date column, used to infer frequency."),
    periods_per_year: int | None = typer.Option(None, help="Override the inferred frequency."),
    prices: bool = typer.Option(False, help="Treat the column as price levels, not returns."),
    risk_free: float = typer.Option(0.0, help="Annual risk-free rate, e.g. 0.04."),
) -> None:
    """Describe a track record: Sharpe, higher moments, drawdown.

    This is descriptive only. It makes no claim about whether the performance is
    real — that is what ``luckdet report`` is for.
    """
    try:
        series = load_returns_csv(
            path,
            column=column,
            date_column=date_column,
            periods_per_year=periods_per_year,
            are_prices=prices,
        )
        stats = summarize(series, risk_free_rate=risk_free)
    except LuckDetectorError as exc:
        _error(exc)
        raise typer.Exit(code=2) from exc

    table = Table(title=f"{stats.name} — descriptive summary", title_style="bold")
    table.add_column("Statistic")
    table.add_column("Value", justify="right")

    rows: list[tuple[str, str]] = [
        ("Observations", f"{stats.n_periods:,}"),
        ("Years", f"{stats.years:.2f}"),
        ("Annualised return", f"{stats.mean_return_annual:.2%}"),
        ("Annualised volatility", f"{stats.volatility_annual:.2%}"),
        ("Annualised Sharpe", f"{stats.sharpe_annual:.3f}"),
        ("Skewness", f"{stats.skewness:.3f}"),
        ("Kurtosis (raw)", f"{stats.kurtosis:.3f}"),
        ("Max drawdown", f"{stats.max_drawdown:.2%}"),
        ("Total return", f"{stats.total_return:.2%}"),
    ]
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)
    console.print(
        "[dim]Descriptive only — this says nothing about whether the performance is "
        "real. Run `luckdet report` for the Deflated Sharpe Ratio, PBO and the "
        "Reality Check.[/dim]"
    )


@app.command()
def mine(
    symbol: str = typer.Argument("SPY", help="Ticker to download and mine."),
    start: str = typer.Option("2010-01-01", help="First date, YYYY-MM-DD."),
    end: str | None = typer.Option(None, help="Last date; defaults to today."),
    cost_bps: float = typer.Option(1.0, help="Transaction cost per unit of turnover, in bp."),
    synthetic: bool = typer.Option(
        False, help="Skip the download and use a synthetic price path instead."
    ),
    periods: int = typer.Option(3780, help="Length of the synthetic path, if used."),
) -> None:
    """Brute-force a grid of strategies, then judge the winner.

    Downloads real prices by default. ``--synthetic`` runs the same pipeline on a
    generated path, which is useful for a quick offline check but is *not* a
    result worth reporting: it describes a random number generator, not a market.
    """
    try:
        if synthetic:
            prices = synthetic_prices(periods, seed=42)
            provenance = f"synthetic path, {periods} periods, seed 42"
        else:
            history = load_prices(symbol, start=start, end=end)
            prices = history.close
            provenance = f"{history.symbol}, {history.span} ({history.source})"

        result = mine_grid(prices, cost_bps=cost_bps)
        sharpes = np.array([sharpe_ratio(result.trials.trial(i)) for i in range(result.n_trials)])
        best = int(np.argmax(sharpes))
        winner = result.trials.trial(best)
        deflated = deflated_sharpe_ratio_from_trials(result.trials)
        naive = probabilistic_sharpe_ratio(winner)
    except LuckDetectorError as exc:
        _error(exc)
        raise typer.Exit(code=2) from exc

    console.print(f"[dim]{provenance}[/dim]")
    console.print(
        f"Mined [bold]{result.n_trials}[/bold] strategies; "
        f"buy-and-hold Sharpe {sharpe_ratio(result.buy_and_hold):.3f}\n"
    )

    table = Table(title=f"Winner: {result.trials.labels[best]}", title_style="bold")
    table.add_column("Statistic")
    table.add_column("Value", justify="right")
    for label, value in [
        ("Annualised Sharpe", f"{sharpes[best]:.3f}"),
        ("Total return", f"{winner.total_return():.1%}"),
        ("Max drawdown", f"{max_drawdown(winner):.1%}"),
        ("Naive PSR vs zero", f"{naive.psr:.4f}"),
        ("Trials run", f"{deflated.n_trials}"),
        ("Effectively independent", f"{deflated.n_effective_trials:.0f}"),
        ("Expected max Sharpe of noise", f"{deflated.expected_max_sharpe_annual:.3f}"),
        ("Deflated Sharpe Ratio", f"{deflated.dsr:.4f}"),
    ]:
        table.add_row(label, value)
    console.print(table)

    if deflated.passed:
        console.print("[bold green]VERDICT: survives deflation[/bold green]")
    else:
        console.print("[bold red]VERDICT: likely luck[/bold red]")
    console.print(f"[dim]{deflated.interpretation}[/dim]")
    raise typer.Exit(code=0 if deflated.passed else 1)


#: Exit codes. ``mine`` already used 0/1 to encode the verdict so a shell script
#: can branch on it; ``report`` and ``demo`` keep the convention.
_EXIT_BY_LABEL: dict[str, int] = {"LIKELY_SKILL": 0, "INCONCLUSIVE": 0, "LIKELY_LUCK": 1}

_LABEL_STYLE: dict[str, str] = {
    "LIKELY_SKILL": "bold green",
    "INCONCLUSIVE": "bold yellow",
    "LIKELY_LUCK": "bold red",
}


def _print_analysis(analysis: Analysis) -> None:
    """Print one assessment. Every value is read off the Analysis, never derived."""
    if analysis.synthetic:
        console.print(
            "[bold yellow]SYNTHETIC DATA[/bold yellow] — generated prices, not a market. "
            "Every figure below describes a random number generator."
        )
    console.print(f"[dim]{analysis.provenance}[/dim]")

    table = Table(title=analysis.title, title_style="bold")
    table.add_column("Statistic")
    table.add_column("Value", justify="right")
    benchmark_sharpe = analysis.benchmark_sharpe
    rows: list[tuple[str, str]] = [
        ("Strategies tried", f"{analysis.n_trials:,}"),
        ("Reported winner", analysis.winner_label),
        ("Winner annualised Sharpe", f"{analysis.winner_sharpe:.3f}"),
        ("Winner total return", f"{analysis.winner_total_return:+.1%}"),
        ("Winner max drawdown", f"{analysis.winner_max_drawdown:.1%}"),
        (f"Benchmark ({analysis.benchmark_name}) Sharpe", "—"
         if benchmark_sharpe != benchmark_sharpe else f"{benchmark_sharpe:.3f}"),
        ("Variants beating benchmark", f"{analysis.n_beating_benchmark} of {analysis.n_trials}"),
        ("Expected max Sharpe of noise", f"{analysis.dsr.expected_max_sharpe_annual:.3f}"),
        ("Sharpe needed to clear DSR", f"{analysis.dsr_hurdle:.3f}"),
    ]
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)

    for result in analysis.verdict.results:
        console.print(f"  [{result.status}] {result.name.upper()}  {result.interpretation}")

    style = _LABEL_STYLE[analysis.label]
    console.print(f"\n[{style}]VERDICT: {analysis.label.replace('_', ' ')}[/{style}]")


@app.command()
def report(
    symbol: str = typer.Argument("SPY", help="Ticker to download and mine."),
    start: str = typer.Option("2010-01-01", help="First date, YYYY-MM-DD."),
    end: str | None = typer.Option(None, help="Last date; defaults to today."),
    cost_bps: float = typer.Option(1.0, help="Transaction cost per unit of turnover, in bp."),
    output: Path = typer.Option(Path("report.html"), help="Where to write the HTML report."),
    against_zero: bool = typer.Option(
        False, help="Judge against zero instead of buy-and-hold (the softer test)."
    ),
    resamples: int = typer.Option(1000, help="Bootstrap replicates for the Reality Check."),
    synthetic: bool = typer.Option(
        False, help="Skip the download and use a synthetic price path instead."
    ),
    periods: int = typer.Option(3780, help="Length of the synthetic path, if used."),
) -> None:
    """Run all four tests over a mined grid and write a single self-contained HTML file.

    The report embeds both figures as base64 PNGs and uses no external CSS, no
    CDN and no JavaScript, so it opens correctly on a machine with no network.
    """
    try:
        if synthetic:
            analysis = synthetic_demo_analysis(
                periods=periods, cost_bps=cost_bps, n_resamples=resamples
            )
        else:
            history = load_prices(symbol, start=start, end=end)
            mined = mine_grid(history.close, cost_bps=cost_bps)
            analysis = analyse_mined(
                mined,
                against_buy_and_hold=not against_zero,
                title=f"{history.symbol}: is the best of {mined.n_trials} strategies real?",
                provenance=(
                    f"{history.symbol}, {history.span} ({history.source}), "
                    f"{history.n_periods:,} closes, {cost_bps:.1f}bp cost"
                ),
                dates=history.dates,
                n_resamples=resamples,
            )
        destination = write_report(analysis, output)
    except LuckDetectorError as exc:
        _error(exc)
        raise typer.Exit(code=2) from exc

    _print_analysis(analysis)
    console.print(f"\nWrote [bold]{destination}[/bold]")
    raise typer.Exit(code=_EXIT_BY_LABEL[analysis.label])


@app.command()
def demo(
    output: Path = typer.Option(Path("demo_report.html"), help="Where to write the HTML report."),
    offline: bool = typer.Option(
        False, help="Run on a synthetic price path; labels every figure SYNTHETIC."
    ),
    symbol: str = typer.Option("SPY", help="Ticker for the real-data half."),
    start: str = typer.Option("2010-01-01", help="First date, YYYY-MM-DD."),
    cost_bps: float = typer.Option(1.0, help="Transaction cost per unit of turnover, in bp."),
    resamples: int = typer.Option(1000, help="Bootstrap replicates for the Reality Check."),
) -> None:
    """Run the whole thing end to end: real prices indicted, planted edge acquitted.

    Both halves are the demonstration. A tool that only ever says "luck" is
    indistinguishable from a pessimist, so the second half plants a genuine edge
    and checks that the same machinery finds it.

    Data resolution is **cache, then download, then refuse**. If no prices are
    cached and the download is unavailable, the command fails and points at
    ``--offline`` rather than quietly substituting synthetic data.
    """
    try:
        result = run_demo(
            offline=offline,
            symbol=symbol,
            start=start,
            cost_bps=cost_bps,
            n_resamples=resamples,
        )
        destination = write_report(result.real, output)
    except LuckDetectorError as exc:
        _error(exc)
        raise typer.Exit(code=2) from exc

    console.print("[bold]1/2 — the real thing[/bold]\n")
    _print_analysis(result.real)
    console.print("\n[bold]2/2 — the control: an edge that is genuinely there[/bold]\n")
    _print_analysis(result.planted)

    console.print(
        f"\nWrote [bold]{destination}[/bold]  "
        f"[dim](the assessment above, as one self-contained file)[/dim]"
    )
    raise typer.Exit(code=_EXIT_BY_LABEL[result.real.label])


if __name__ == "__main__":  # pragma: no cover
    app()
