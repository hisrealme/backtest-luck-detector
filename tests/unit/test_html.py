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


def evidence_table(document: str) -> str:
    """Just the Table 2 markup.

    Scoped rather than searched document-wide because the abstract also names the
    tests that objected, so a bare ``document.index("Deflated Sharpe Ratio")``
    finds the prose rather than the row.
    """
    start = document.index("<b>Table 2.")
    return document[start : document.index("</table>", start)]


def section(document: str, heading: str) -> str:
    """One numbered subsection, from its ``<h3>`` to the next heading."""
    start = document.index(f"{heading}<span")
    rest = document[start:]
    end = rest.find("<h3>")
    return rest if end == -1 else rest[:end]


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


class TestPaperLayout:
    """The document is laid out as a paper, and the structure is load-bearing."""

    def test_has_an_abstract_and_numbered_sections(self, document: str) -> None:
        assert "<h2>Abstract</h2>" in document
        for number, heading in enumerate(
            ["Data and search", "Evidence", "The bar the winner had to clear", "Limitations"],
            start=1,
        ):
            assert f'<span class="n">{number}</span> {heading}' in document

    def test_the_abstract_quotes_the_search_and_the_record(
        self, document: str, luck_analysis: Analysis
    ) -> None:
        """It is composed in Python from the analysis, not written by hand."""
        assert f"{luck_analysis.n_trials:,} strategy variants" in document
        assert f"{luck_analysis.winner_sharpe:.3f}" in document
        assert luck_analysis.winner_label in document

    def test_tables_are_captioned_above_and_figures_below(self, document: str) -> None:
        """Paper convention, and the two are not interchangeable."""
        assert "<caption><b>Table 1.</b>" in document
        assert "<b>Table 2.</b>" in document
        assert "<figcaption><b>Figure 1.</b>" in document
        assert "<figcaption><b>Figure 2.</b>" in document

    def test_tables_have_no_vertical_rules(self, document: str) -> None:
        """Booktabs style: horizontal rules only, top, mid and bottom."""
        css = document.split("</style>")[0]
        assert "border-left" not in css
        assert "border-right" not in css
        assert "border-top: 1.4px solid" in css

    def test_carries_no_dashboard_chrome(self, document: str) -> None:
        """The verdict is stated once, in the abstract, not as a coloured banner."""
        assert "banner" not in document
        assert document.count('class="verdict"') == 1


class TestEvidenceTable:
    def test_has_one_row_per_test_in_order(self, document: str, luck_analysis: Analysis) -> None:
        titles = [
            "Probabilistic Sharpe Ratio",
            "Deflated Sharpe Ratio",
            "Probability of Backtest Overfitting",
            "Reality Check / SPA",
        ]
        table = evidence_table(document)
        positions = [table.index(title) for title in titles]
        assert positions == sorted(positions), "evidence is not rendered in TEST_ORDER"
        assert len(luck_analysis.verdict.results) == 4

    def test_each_test_also_gets_its_own_numbered_subsection(self, document: str) -> None:
        """Paper layout: the table is the summary, the subsections are the argument."""
        for index in range(1, 5):
            assert f">2.{index}&nbsp;" in document

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

    def test_abstract_states_the_verdict(self, document: str, luck_analysis: Analysis) -> None:
        assert f'<span class="verdict">{luck_analysis.label.replace("_", " ")}</span>' in document

    def test_the_single_accent_is_keyed_to_the_verdict(self, document: str) -> None:
        """One accent per document, set on the article, so the rest can stay black."""
        assert '<article class="v-luck">' in document
        assert "--accent" in document


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
        assert 'class="st na"' in na_document
        assert "not applicable" in na_document

    def test_gets_a_numbered_subsection_like_every_other_test(self, na_document: str) -> None:
        """Not a footnote, not greyed out — the same structural weight as a pass."""
        block = section(na_document, "Reality Check / SPA")
        assert block.startswith("Reality Check / SPA")
        assert 'class="st na"' in block

    def test_is_not_styled_as_a_pass_or_a_failure(self, na_document: str) -> None:
        block = section(na_document, "Reality Check / SPA")
        assert 'class="st pass"' not in block
        assert 'class="st fail"' not in block

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
        table = evidence_table(na_document)
        row = table[table.index("Reality Check / SPA") :]
        assert "p = 0.0000" not in row
        assert "0.0500" not in row
        assert row.count("—") >= 2


class TestSyntheticLabelling:
    def test_synthetic_reports_say_so_loudly(self, edge_analysis: Analysis) -> None:
        document = render_report(edge_analysis)
        assert "SYNTHETIC DATA" in document
        assert "not a market" in document
        assert 'class="synthetic"' in document

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
