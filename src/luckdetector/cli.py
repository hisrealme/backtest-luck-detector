"""Command-line interface.

Deliberately thin: every number shown here is computed by the library, and no
statistics live in this file. Right now it exposes the Phase 1 descriptive
summary; ``report``, ``mine`` and ``demo`` arrive in Phase 10 once the tests they
depend on exist.

The ``luckdet`` console script is declared in ``pyproject.toml``, so this module
must remain importable even while the interesting commands are unimplemented —
otherwise a fresh ``pip install -e .`` produces a script that crashes on launch.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .exceptions import LuckDetectorError
from .io import load_returns_csv
from .stats import summarize

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


if __name__ == "__main__":  # pragma: no cover
    app()
