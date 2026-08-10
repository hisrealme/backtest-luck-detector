"""CLI smoke tests.

The console script is declared in ``pyproject.toml``; if ``luckdetector.cli``
stops importing, a fresh install produces a ``luckdet`` binary that crashes on
launch. These tests exist mainly to catch that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from luckdetector import __version__
from luckdetector.cli import app

runner = CliRunner()


@pytest.fixture
def returns_csv(tmp_path: Path, rng: np.random.Generator) -> Path:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=500),
            "strategy": rng.normal(0.0004, 0.01, 500),
        }
    )
    path = tmp_path / "returns.csv"
    frame.to_csv(path, index=False)
    return path


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_summary_command(returns_csv: Path) -> None:
    result = runner.invoke(app, ["summary", str(returns_csv), "--date-column", "date"])
    assert result.exit_code == 0
    assert "Annualised Sharpe" in result.stdout


def test_summary_reports_errors_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["summary", str(tmp_path / "missing.csv")])
    assert result.exit_code == 2
    assert "Error" in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "How much of your backtest is luck?" in result.stdout


class TestMineCommand:
    def test_synthetic_mode_needs_no_network(self) -> None:
        result = runner.invoke(app, ["mine", "--synthetic", "--periods", "1200"])
        assert "Winner:" in result.stdout
        assert "Deflated Sharpe Ratio" in result.stdout
        assert "VERDICT" in result.stdout

    def test_exit_code_encodes_the_verdict(self) -> None:
        """Exit 1 for 'likely luck', 0 for 'survives' — usable from a shell script."""
        result = runner.invoke(app, ["mine", "--synthetic", "--periods", "1200"])
        assert result.exit_code in (0, 1)

    def test_reports_provenance(self) -> None:
        result = runner.invoke(app, ["mine", "--synthetic", "--periods", "1200"])
        assert "synthetic path" in result.stdout
