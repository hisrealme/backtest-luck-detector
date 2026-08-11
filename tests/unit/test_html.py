"""The HTML report: self-contained, derived from nothing, and honest about NA.

Two properties carry most of the weight here. **Self-contained** is checked by
asserting there is no reference to any external asset — a report that silently
depends on a CDN stops rendering the moment it is opened offline, which is the
one situation it exists for. **Derived from nothing** is checked by asserting
that the statistics in the table are the ones on the result objects, so a
template edit cannot quietly become a second implementation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from luckdetector.report.analysis import Analysis
from luckdetector.report.html import CAVEATS, render_report, write_report


@pytest.fixture(scope="module")
def _cache() -> dict[str, str]:
    return {}


@pytest.fixture
def document(luck_analysis: Analysis, _cache: dict[str, str]) -> str:
    """Rendered once per module — both figures make this the slowest step here."""
    if "luck" not in _cache:
        _cache["luck"] = render_report(luck_analysis)
    return _cache["luck"]


class TestSelfContained:
    def test_is_a_complete_document(self, document: str) -> None:
        assert document.startswith("<!DOCTYPE html>")
        assert document.rstrip().endswith("</html>")

    def test_has_no_external_assets(self, document: str) -> None:
        """No CDN, no stylesheet link, no script tag. It must open with no network."""
        assert "<script" not in document.lower()
        assert "<link" not in document.lower()
        assert not re.search(r'src="https?://', document)
        assert not re.search(r'href="https?://', document)

    def test_both_figures_are_embedded_as_data_uris(self, document: str) -> None:
        embedded = re.findall(r'src="data:image/png;base64,([A-Za-z0-9+/=]+)"', document)
        assert len(embedded) == 2
        assert all(len(payload) > 1000 for payload in embedded)

    def test_styles_are_inline(self, document: str) -> None:
        assert "<style>" in document


class TestEvidenceTable:
    def test_has_one_row_per_test_in_order(self, document: str, luck_analysis: Analysis) -> None:
        titles = [
            "Probabilistic Sharpe Ratio",
            "Deflated Sharpe Ratio",
            "Probability of Backtest Overfitting",
            "Reality Check / SPA",
        ]
        positions = [document.index(title) for title in titles]
        assert positions == sorted(positions), "evidence is not rendered in TEST_ORDER"
        assert len(luck_analysis.verdict.results) == 4

    def test_statistics_are_the_ones_on_the_results(
        self, document: str, luck_analysis: Analysis
    ) -> None:
        """The template renders; it does not compute."""
        for result in luck_analysis.verdict.results:
            if result.status == "NOT_APPLICABLE":
                continue  # covered by TestNotApplicableIsNotBuried
            prefix = "p = " if result.p_value is not None else ""
            assert f"{prefix}{result.statistic:.4f}" in document
            assert f"{result.threshold:.4f}" in document

    def test_every_interpretation_appears_verbatim(
        self, document: str, luck_analysis: Analysis
    ) -> None:
        for result in luck_analysis.verdict.results:
            # The document is HTML-escaped, so compare on a distinctive fragment
            # that contains no escapable characters.
            fragment = result.interpretation.split(",")[0]
            assert fragment in document

    def test_verdict_banner_states_the_label_and_the_rule(
        self, document: str, luck_analysis: Analysis
    ) -> None:
        assert luck_analysis.label.replace("_", " ") in document
        assert "banner luck" in document


@pytest.fixture(scope="module")
def na_document() -> str:
    """A report whose SPA row is NOT_APPLICABLE: nothing beat the benchmark.

    Seed 5 is chosen because on that path not one of the 157 variants beats
    buy-and-hold, which reproduces the SPY case exactly. Asserted rather than
    skipped over — a fixture that quietly skips when its precondition fails
    turns three tests into no tests.
    """
    from luckdetector.mining import mine, synthetic_prices
    from luckdetector.report.analysis import analyse_mined

    result = mine(synthetic_prices(900, seed=5), cost_bps=1.0)
    analysis = analyse_mined(result, n_blocks=8, n_resamples=100, seed=1)
    assert analysis.n_beating_benchmark == 0, "fixture no longer reproduces the SPY case"
    assert analysis.verdict.result("spa").status == "NOT_APPLICABLE"
    return render_report(analysis)


class TestNotApplicableIsNotBuried:
    """``NOT_APPLICABLE`` gets its own treatment, because on SPY it is the finding."""

    def test_is_rendered_as_its_own_status(self, na_document: str) -> None:
        assert "NOT APPLICABLE" in na_document
        assert "evidence na" in na_document

    def test_is_not_styled_as_a_pass_or_a_failure(self, na_document: str) -> None:
        block = na_document[na_document.index("evidence na") :][:400]
        assert "tag pass" not in block
        assert "tag fail" not in block

    def test_carries_the_finding_rather_than_a_p_value(self, na_document: str) -> None:
        assert "Not one of the" in na_document
        assert "empty comparison rather than a rejection" in na_document

    def test_renders_no_comparison_at_all_in_the_table(self, na_document: str) -> None:
        """The regression this row is most likely to acquire, pinned.

        When SPA drops out, ``TestResult.statistic`` holds the SPA *test
        statistic*, which is zero by construction — not a p-value. A row that
        formats it like one prints ``p = 0.0000`` against a 0.05 threshold and
        turns the weakest cell in the report into its most emphatic rejection.
        The row must show no comparison, because there is none to show.
        """
        row = na_document[na_document.index("Reality Check / SPA") :][:400]
        assert "p = 0.0000" not in row
        assert "0.0500" not in row
        assert row.count("—") >= 2


class TestSyntheticLabelling:
    def test_synthetic_reports_say_so_loudly(self, edge_analysis: Analysis) -> None:
        document = render_report(edge_analysis)
        assert "SYNTHETIC DATA" in document
        assert "not from a market" in document

    def test_real_reports_carry_no_synthetic_banner(self, document: str) -> None:
        assert "SYNTHETIC DATA" not in document


class TestCaveats:
    def test_every_caveat_is_rendered(self, document: str) -> None:
        for heading, _ in CAVEATS:
            assert heading in document

    def test_the_measured_limitations_are_present(self, document: str) -> None:
        """Purging gap and detection power, both measured elsewhere in the project."""
        assert "no gap" in document
        assert "20%" in document

    def test_the_invented_rules_are_admitted(self, document: str) -> None:
        assert "The combination rules are invented" in document


class TestWriteReport:
    def test_writes_a_file_and_creates_parents(
        self, luck_analysis: Analysis, tmp_path: Path
    ) -> None:
        destination = write_report(luck_analysis, tmp_path / "nested" / "report.html")
        assert destination.exists()
        assert destination.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_version_is_stamped(self, luck_analysis: Analysis, tmp_path: Path) -> None:
        destination = write_report(luck_analysis, tmp_path / "r.html", version="9.9.9")
        assert "luckdetector 9.9.9" in destination.read_text(encoding="utf-8")

    def test_defaults_to_the_installed_version(self, document: str) -> None:
        from luckdetector import __version__

        assert f"luckdetector {__version__}" in document
