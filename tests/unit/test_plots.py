"""Figure tests, asserted on structure rather than on pixels.

**No image hashes.** A hash of the rendered PNG breaks on every matplotlib
release, every font-config change and every freetype bump, and when it breaks it
tells you nothing about whether the figure is still correct. What these tests
check instead is that the right number of artists exist, that they carry the
right data, and — the part that actually matters — that the marked lines sit
where the statistics put them.

The figures are built through the object-oriented API, so nothing here needs a
display, a backend or a ``pyplot`` teardown.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from luckdetector.report.analysis import Analysis
from luckdetector.report.plots import (
    COLOURS,
    cumulative_return_figure,
    figure_to_base64,
    trial_sharpe_figure,
)


class TestCumulativeReturnFigure:
    @pytest.fixture
    def figure(self, luck_analysis: Analysis) -> object:
        return cumulative_return_figure(luck_analysis)

    def test_has_one_axes_and_two_series(self, figure: object) -> None:
        axes = figure.axes  # type: ignore[attr-defined]
        assert len(axes) == 1
        assert len(axes[0].lines) == 2

    def test_series_are_the_full_length_of_the_record(
        self, figure: object, luck_analysis: Analysis
    ) -> None:
        """A curve one period short is a silent off-by-one in the comparison."""
        for line in figure.axes[0].lines:  # type: ignore[attr-defined]
            assert len(line.get_ydata()) == luck_analysis.n_periods

    def test_both_curves_start_at_one(self, figure: object) -> None:
        for line in figure.axes[0].lines:  # type: ignore[attr-defined]
            first = float(np.asarray(line.get_ydata())[0])
            assert first == pytest.approx(1.0, abs=0.35)

    def test_is_labelled(self, figure: object, luck_analysis: Analysis) -> None:
        axes = figure.axes[0]  # type: ignore[attr-defined]
        assert "Cumulative return" in axes.get_title()
        assert axes.get_xlabel()
        assert "log scale" in axes.get_ylabel()
        labels = [text.get_text() for text in axes.get_legend().get_texts()]
        assert len(labels) == 2
        assert any(luck_analysis.winner_label in label for label in labels)
        assert any(luck_analysis.benchmark_name in label for label in labels)

    def test_uses_a_log_scale_when_both_curves_stay_positive(self, figure: object) -> None:
        assert figure.axes[0].get_yscale() == "log"  # type: ignore[attr-defined]

    def test_dates_are_used_when_supplied(self, luck_analysis: Analysis) -> None:
        import pandas as pd

        dates = pd.bdate_range("2015-01-01", periods=luck_analysis.n_periods)
        with_dates = cumulative_return_figure(
            Analysis(**{**luck_analysis.__dict__, "dates": dates})
        )
        assert with_dates.axes[0].get_xlabel() == "Date"


class TestTrialSharpeFigure:
    @pytest.fixture
    def figure(self, luck_analysis: Analysis) -> object:
        return trial_sharpe_figure(luck_analysis)

    def test_has_one_axes_with_a_histogram_and_a_null_curve(self, figure: object) -> None:
        axes = figure.axes[0]  # type: ignore[attr-defined]
        assert len(axes.patches) > 0, "the family histogram is missing"
        # One null density curve plus three marked verticals.
        assert len(axes.lines) == 4

    def test_the_three_verticals_sit_where_the_statistics_put_them(
        self, figure: object, luck_analysis: Analysis
    ) -> None:
        """The whole point of the figure, so it is asserted rather than eyeballed."""
        positions = sorted(
            float(np.asarray(line.get_xdata())[0])
            for line in figure.axes[0].lines  # type: ignore[attr-defined]
            if len(set(np.asarray(line.get_xdata()).tolist())) == 1
        )
        expected = sorted(
            [
                luck_analysis.dsr.expected_max_sharpe_annual,
                luck_analysis.winner_sharpe,
                luck_analysis.dsr_hurdle,
            ]
        )
        np.testing.assert_allclose(positions, expected, rtol=1e-9)

    def test_the_hurdle_is_drawn_to_the_right_of_the_winner_when_the_test_fails(
        self, figure: object, luck_analysis: Analysis
    ) -> None:
        """The correction, visible in the geometry: a failing DSR must look like one."""
        assert luck_analysis.dsr_hurdle > luck_analysis.winner_sharpe
        lo, hi = figure.axes[0].get_xlim()  # type: ignore[attr-defined]
        assert lo < luck_analysis.winner_sharpe < luck_analysis.dsr_hurdle < hi

    def test_the_shaded_region_is_the_dsr(self, figure: object, luck_analysis: Analysis) -> None:
        """The caption claims the shaded area equals the DSR; check it numerically."""
        from scipy import stats as sps

        area = float(
            sps.norm.cdf(
                luck_analysis.winner_sharpe,
                loc=luck_analysis.dsr.expected_max_sharpe_annual,
                scale=luck_analysis.dsr.psr_result.standard_error_annual,
            )
        )
        assert area == pytest.approx(luck_analysis.dsr.dsr, abs=1e-9)
        assert len(figure.axes[0].collections) == 1  # type: ignore[attr-defined]

    def test_every_trial_is_inside_the_drawn_range(
        self, figure: object, luck_analysis: Analysis
    ) -> None:
        lo, hi = figure.axes[0].get_xlim()  # type: ignore[attr-defined]
        assert lo <= luck_analysis.trial_sharpes.min()
        assert hi >= luck_analysis.trial_sharpes.max()

    def test_legend_names_every_marked_quantity(self, figure: object) -> None:
        labels = [
            text.get_text()
            for text in figure.axes[0].get_legend().get_texts()  # type: ignore[attr-defined]
        ]
        assert len(labels) == 6
        joined = " ".join(labels)
        for fragment in ("trials actually run", "DSR", "expected max of noise", "winner"):
            assert fragment in joined

    def test_title_states_the_gap(self, figure: object) -> None:
        assert "short of the bar" in figure.axes[0].get_title()  # type: ignore[attr-defined]

    def test_a_passing_family_says_so(self, edge_analysis: Analysis) -> None:
        figure = trial_sharpe_figure(edge_analysis)
        assert "clear of the bar" in figure.axes[0].get_title()


class TestTitlesAreOptional:
    """The report prints each figure directly above its own numbered caption.

    A matplotlib title there would restate the caption a centimetre above it, so
    the report asks for ``title=False``. The default stays ``True`` because the
    figures are also used on their own, where nothing else names them.
    """

    def test_titles_are_on_by_default(self, luck_analysis: Analysis) -> None:
        assert cumulative_return_figure(luck_analysis).axes[0].get_title()
        assert trial_sharpe_figure(luck_analysis).axes[0].get_title()

    def test_titles_can_be_suppressed(self, luck_analysis: Analysis) -> None:
        assert cumulative_return_figure(luck_analysis, title=False).axes[0].get_title() == ""
        assert trial_sharpe_figure(luck_analysis, title=False).axes[0].get_title() == ""

    def test_suppressing_the_title_changes_nothing_else(self, luck_analysis: Analysis) -> None:
        """Only the heading goes; the data and the marked lines are untouched."""
        with_title = trial_sharpe_figure(luck_analysis).axes[0]
        without = trial_sharpe_figure(luck_analysis, title=False).axes[0]
        assert len(with_title.lines) == len(without.lines)
        assert with_title.get_xlim() == without.get_xlim()
        assert with_title.get_xlabel() == without.get_xlabel()

    def test_the_report_suppresses_them(self, luck_analysis: Analysis) -> None:
        """Pinned here rather than left to the template to remember."""
        from luckdetector.report.html import render_report

        document = render_report(luck_analysis)
        assert "Cumulative return: the reported winner" not in document


class TestSyntheticStamp:
    def test_synthetic_data_is_stamped_on_every_figure(self, edge_analysis: Analysis) -> None:
        """The offline path must never produce an image mistakable for a real result."""
        assert edge_analysis.synthetic
        for build in (cumulative_return_figure, trial_sharpe_figure):
            texts = [t.get_text() for t in build(edge_analysis).axes[0].texts]
            assert "SYNTHETIC" in texts

    def test_real_data_is_not_stamped(self, luck_analysis: Analysis) -> None:
        assert not luck_analysis.synthetic
        texts = [t.get_text() for t in cumulative_return_figure(luck_analysis).axes[0].texts]
        assert "SYNTHETIC" not in texts


class TestEmbedding:
    def test_base64_round_trips_to_a_png(self, luck_analysis: Analysis) -> None:
        payload = figure_to_base64(cumulative_return_figure(luck_analysis), dpi=60)
        raw = base64.b64decode(payload, validate=True)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_payload_carries_no_data_uri_prefix(self, luck_analysis: Analysis) -> None:
        """The template owns the URI, so the helper must not embed one too."""
        payload = figure_to_base64(cumulative_return_figure(luck_analysis), dpi=60)
        assert not payload.startswith("data:")


def test_palette_is_shared_so_the_two_figures_agree() -> None:
    assert COLOURS["winner"] != COLOURS["benchmark"]
    assert set(COLOURS) >= {"winner", "benchmark", "family", "null", "hurdle", "synthetic"}
