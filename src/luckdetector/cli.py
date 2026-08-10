"""Command-line interface.

Deliberately thin: every number shown here is computed by the library, and no
statistics live in this file. ``summary`` describes a track record; ``mine``
brute-forces a strategy grid and judges the winner. The full ``report`` command,
which adds PBO and the Reality Check, arrives in Phase 10.

The ``luckdet`` console script is declared in ``pyproject.toml``, so this module
must remain importable even while the interesting commands are unimplemented —
otherwise a fresh ``pip install -e .`` produces a script that crashes on launch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .exceptions import LuckDetectorError
from .io import load_prices, load_returns_csv
from .mining import mine as mine_grid
from .mining import synthetic_prices
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
    real — that is what the remaining phases of the tool are for.
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
        console.print(f"[bold red]Error:[/bold red] {exc}")
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
        "[dim]Descriptive only. Deflated Sharpe, PBO and Reality Check arrive in "
        "later phases — see docs/BLUEPRINT.md.[/dim]"
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
        console.print(f"[bold red]Error:[/bold red] {exc}")
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


if __name__ == "__main__":  # pragma: no cover
    app()
