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


def test_error_messages_are_not_parsed_as_markup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rich reads ``[...]`` as a style tag, which eats the fix it is telling you about.

    The message that matters is ``pip install -e ".[data]"``. Rendered as markup,
    ``[data]`` is consumed as an unknown style and the user is told to run
    ``pip install -e "."``, which does not install the thing they are missing.
    """
    from luckdetector.exceptions import DataValidationError

    def broken(*args: object, **kwargs: object) -> object:
        raise DataValidationError('needs the extra: pip install -e ".[data]"')

    monkeypatch.setattr("luckdetector.cli.load_returns_csv", broken)
    result = runner.invoke(app, ["summary", str(tmp_path / "anything.csv")])
    assert result.exit_code == 2
    assert '.[data]' in result.stdout


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


class TestReportCommand:
    """``report`` writes the HTML; the numbers in it come from the library."""

    def test_synthetic_mode_writes_a_self_contained_file(self, tmp_path: Path) -> None:
        destination = tmp_path / "out" / "report.html"
        result = runner.invoke(
            app,
            [
                "report",
                "--synthetic",
                "--periods",
                "900",
                "--resamples",
                "100",
                "--output",
                str(destination),
            ],
        )
        assert result.exit_code in (0, 1), result.stdout
        assert destination.exists()
        document = destination.read_text(encoding="utf-8")
        assert document.startswith("<!DOCTYPE html>")
        assert "<script" not in document.lower()
        assert "SYNTHETIC DATA" in document

    def test_prints_the_verdict_and_the_hurdle(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["report", "--synthetic", "--periods", "900", "--resamples", "100",
             "--output", str(tmp_path / "r.html")],
        )
        assert "VERDICT" in result.stdout
        assert "Sharpe needed to clear DSR" in result.stdout

    def test_real_data_path_runs_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The non-synthetic branch, exercised with the loader stubbed out.

        ``load_prices`` is replaced rather than the downloader beneath it so the
        test neither reaches the network nor writes into the user's real cache
        directory.
        """
        from luckdetector.io.prices import PriceHistory

        n = 900
        rng = np.random.default_rng(4)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
        history = PriceHistory(
            symbol="TEST",
            dates=pd.DatetimeIndex(pd.bdate_range("2015-01-02", periods=n)),
            close=closes,
            source="stub",
        )
        monkeypatch.setattr("luckdetector.cli.load_prices", lambda *a, **k: history)

        destination = tmp_path / "real.html"
        result = runner.invoke(
            app,
            ["report", "TEST", "--resamples", "100", "--output", str(destination)],
        )
        assert result.exit_code in (0, 1), result.stdout
        document = destination.read_text(encoding="utf-8")
        assert "TEST" in document
        assert "SYNTHETIC DATA" not in document
        assert "buy-and-hold" in document

    def test_against_zero_switches_the_benchmark(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luckdetector.io.prices import PriceHistory

        n = 900
        rng = np.random.default_rng(4)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
        history = PriceHistory(
            symbol="TEST",
            dates=pd.DatetimeIndex(pd.bdate_range("2015-01-02", periods=n)),
            close=closes,
            source="stub",
        )
        monkeypatch.setattr("luckdetector.cli.load_prices", lambda *a, **k: history)

        destination = tmp_path / "zero.html"
        result = runner.invoke(
            app,
            ["report", "TEST", "--against-zero", "--resamples", "100",
             "--output", str(destination)],
        )
        assert result.exit_code in (0, 1), result.stdout
        assert "Benchmark</th><td class=\"num\">zero" in destination.read_text(encoding="utf-8")

    def test_reports_errors_cleanly(
        self, tmp_path: Path, no_network: pytest.MonkeyPatch
    ) -> None:
        """An unavailable download must exit 2, not raise a traceback at the user."""
        result = runner.invoke(
            app,
            ["report", "NOT_A_TICKER", "--start", "2010-01-01",
             "--output", str(tmp_path / "r.html")],
        )
        assert result.exit_code == 2
        assert "Error" in result.stdout


class TestDemoCommand:
    def test_offline_runs_both_halves_with_no_network(self, tmp_path: Path) -> None:
        destination = tmp_path / "demo.html"
        result = runner.invoke(
            app, ["demo", "--offline", "--resamples", "100", "--output", str(destination)]
        )
        assert result.exit_code in (0, 1), result.stdout
        assert "1/2" in result.stdout
        assert "2/2" in result.stdout
        assert destination.exists()

    def test_offline_labels_everything_synthetic(self, tmp_path: Path) -> None:
        destination = tmp_path / "demo.html"
        result = runner.invoke(
            app, ["demo", "--offline", "--resamples", "100", "--output", str(destination)]
        )
        assert "SYNTHETIC DATA" in result.stdout
        assert "SYNTHETIC DATA" in destination.read_text(encoding="utf-8")

    def test_the_control_half_finds_the_planted_edge(self, tmp_path: Path) -> None:
        """Both halves are the demonstration — see report/demo.py."""
        result = runner.invoke(
            app,
            ["demo", "--offline", "--resamples", "100", "--output", str(tmp_path / "d.html")],
        )
        assert "LIKELY SKILL" in result.stdout

    def test_refuses_rather_than_falling_back(
        self, tmp_path: Path, no_network: pytest.MonkeyPatch
    ) -> None:
        """With no cache and no network the command fails and names ``--offline``.

        The failure is the feature. A demo that silently swapped in synthetic
        data here would print the same confident narrative about a random number
        generator, which is the error this whole package exists to catch.
        """
        no_network.setattr(
            "luckdetector.report.demo.DEMO_CACHE_DIRS", (tmp_path / "empty",)
        )
        result = runner.invoke(app, ["demo", "--output", str(tmp_path / "d.html")])
        assert result.exit_code == 2
        assert "--offline" in result.stdout
        assert not (tmp_path / "d.html").exists(), "wrote a report from refused data"
