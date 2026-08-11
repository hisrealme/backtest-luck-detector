"""The pipeline helper: four statistics over one family, and the derived quantities.

The tests that matter most here are not the plumbing ones. They are the three
that pin the measurement which contradicted the Phase 8 brief — that the winner
is the maximum of its family by construction, that it can clear the expected
maximum of noise while still being called luck, and that the bar the Deflated
Sharpe Ratio actually applies is a different and higher number. Those live at the
bottom of the file.
"""

from __future__ import annotations

import numpy as np
import pytest

from luckdetector.mining import mine, synthetic_prices
from luckdetector.report.analysis import Analysis, analyse, analyse_mined
from luckdetector.stats.dsr import DSR_THRESHOLD
from luckdetector.types import ReturnSeries, TrialMatrix


class TestAnalyse:
    def test_runs_all_four_statistics(self, luck_analysis: Analysis) -> None:
        """Every test in TEST_ORDER is present, so the report never has a blank row."""
        names = [result.name for result in luck_analysis.verdict.results]
        assert names == ["psr", "dsr", "pbo", "spa"]

    def test_winner_is_derived_not_supplied(self, luck_analysis: Analysis) -> None:
        """The report cannot be handed a strategy that was not the best of its search."""
        assert luck_analysis.winner_index == int(np.argmax(luck_analysis.trial_sharpes))
        assert (
            luck_analysis.winner_label
            == luck_analysis.trials.labels[luck_analysis.winner_index]
        )

    def test_trial_sharpes_match_the_public_function(self, luck_analysis: Analysis) -> None:
        """The vectorised pass agrees with sharpe_ratio() trial by trial."""
        from luckdetector.stats import sharpe_ratio

        expected = [
            sharpe_ratio(luck_analysis.trials.trial(i)) for i in range(luck_analysis.n_trials)
        ]
        np.testing.assert_allclose(luck_analysis.trial_sharpes, expected, rtol=1e-12)

    def test_is_reproducible(self) -> None:
        """Same seed, same verdict — including the two stochastic statistics."""
        values = np.random.default_rng(7).normal(0.0, 0.01, (20, 600))
        trials = TrialMatrix(values, periods_per_year=252)
        first = analyse(trials, n_blocks=8, n_resamples=100, seed=5)
        second = analyse(trials, n_blocks=8, n_resamples=100, seed=5)
        assert first.pbo.pbo == second.pbo.pbo
        assert first.spa.p_consistent == second.spa.p_consistent
        assert first.label == second.label

    def test_a_zero_benchmark_has_no_sharpe(self, edge_analysis: Analysis) -> None:
        """A constant benchmark has no volatility, so its Sharpe is undefined, not huge."""
        assert np.isnan(edge_analysis.benchmark_sharpe)
        assert edge_analysis.benchmark_total_return == pytest.approx(0.0)

    def test_summary_is_json_friendly(self, luck_analysis: Analysis) -> None:
        import json

        payload = luck_analysis.summary()
        assert json.loads(json.dumps(payload))["label"] == luck_analysis.label

    def test_detects_a_planted_edge(self, edge_analysis: Analysis) -> None:
        """The control half of the demo, at the effect size Phase 7 measured."""
        assert edge_analysis.label == "LIKELY_SKILL"
        assert edge_analysis.verdict.n_failed == 0


@pytest.fixture(scope="module")
def mined_analysis() -> Analysis:
    result = mine(synthetic_prices(900, seed=3), cost_bps=1.0)
    return analyse_mined(result, n_blocks=8, n_resamples=100, seed=1)


class TestAnalyseMined:
    """The ``MiningResult`` entry point, which is what both CLI commands use."""

    def test_defaults_to_buy_and_hold(self, mined_analysis: Analysis) -> None:
        """The benchmark a sceptic cares about, not the softer test against zero."""
        assert mined_analysis.benchmark_name == "buy-and-hold"
        assert not np.isnan(mined_analysis.benchmark_sharpe)

    def test_benchmark_name_comes_from_the_reality_check(self, mined_analysis: Analysis) -> None:
        """One source of truth, so the table header cannot contradict the prose."""
        assert mined_analysis.benchmark_name == mined_analysis.spa.benchmark_name
        assert mined_analysis.benchmark_name in mined_analysis.spa.interpretation

    def test_carries_the_cost_through(self, mined_analysis: Analysis) -> None:
        assert mined_analysis.cost_bps == 1.0

    def test_against_zero_is_available(self) -> None:
        result = mine(synthetic_prices(900, seed=3), cost_bps=1.0)
        analysis = analyse_mined(
            result, against_buy_and_hold=False, n_blocks=8, n_resamples=100, seed=1
        )
        assert analysis.benchmark_name == "zero"

    def test_benchmark_series_is_aligned_with_the_trials(self, mined_analysis: Analysis) -> None:
        """One period of misalignment compares every strategy against the wrong day."""
        assert mined_analysis.benchmark_returns.size == mined_analysis.n_periods


class TestNamedBenchmark:
    def test_a_named_series_names_itself_in_the_report(self) -> None:
        values = np.random.default_rng(9).normal(0.0, 0.01, (12, 500))
        trials = TrialMatrix(values, periods_per_year=252)
        benchmark = ReturnSeries(
            np.random.default_rng(10).normal(0.0002, 0.01, 500), name="60/40 portfolio"
        )
        analysis = analyse(trials, benchmark=benchmark, n_blocks=6, n_resamples=80)
        assert analysis.benchmark_name == "60/40 portfolio"


# ------------------------------------------------- the Phase 8 spec correction


class TestTheHurdleThatIsActuallyApplied:
    """Three properties that the brief's figure 2 got backwards, pinned as tests.

    The Phase 8 brief described the trial-Sharpe figure as showing that "the
    winner sits inside the noise distribution", marked against the expected
    maximum of noise. Neither half survives measurement, and both failures are
    structural rather than incidental — so they are asserted here rather than
    left as a note in a document.
    """

    def test_the_winner_is_the_maximum_so_it_cannot_sit_inside_the_family(
        self, luck_analysis: Analysis
    ) -> None:
        """It is the argmax by construction: the 100th percentile, every time.

        No arrangement of the histogram puts the marker among the others, so a
        figure whose caption claims otherwise is claiming something impossible.
        """
        assert luck_analysis.winner_sharpe == pytest.approx(luck_analysis.trial_sharpes.max())
        assert luck_analysis.n_trials_above_expected_max >= 1

    def test_clearing_the_expected_max_of_noise_is_not_clearing_the_bar(
        self, luck_analysis: Analysis
    ) -> None:
        """The measurement that made the specified figure argue the wrong way.

        On SPY the winner posts 0.4905 against an expected-max hurdle of 0.3086
        — visibly *above* it, along with 42 other variants — while the Deflated
        Sharpe Ratio of 0.7692 calls it luck. A plot of those two numbers alone
        therefore reads as a pass at exactly the moment the test says fail.
        """
        assert luck_analysis.dsr.dsr < DSR_THRESHOLD
        assert luck_analysis.winner_sharpe > luck_analysis.dsr.expected_max_sharpe_annual
        assert luck_analysis.winner_sharpe < luck_analysis.dsr_hurdle

    def test_the_real_bar_is_higher_and_nothing_in_the_family_reaches_it(
        self, luck_analysis: Analysis
    ) -> None:
        """The hurdle DSR applies allows for the standard error on the estimate."""
        assert luck_analysis.dsr_hurdle > luck_analysis.dsr.expected_max_sharpe_annual
        assert luck_analysis.n_trials_above_hurdle == 0
        assert luck_analysis.n_trials_above_expected_max > luck_analysis.n_trials_above_hurdle

    def test_the_hurdle_agrees_with_the_verdict_in_both_directions(
        self, luck_analysis: Analysis, edge_analysis: Analysis
    ) -> None:
        """Passing DSR and clearing the drawn bar are the same event, not two."""
        for analysis in (luck_analysis, edge_analysis):
            clears = analysis.winner_sharpe >= analysis.dsr_hurdle
            assert clears == (analysis.dsr.dsr >= DSR_THRESHOLD)

    @pytest.mark.parametrize("seed", [3, 7, 11, 42])
    @pytest.mark.parametrize("periods", [900, 1500, 2520])
    def test_the_misleading_geometry_is_structural_not_a_property_of_spy(
        self, seed: int, periods: int
    ) -> None:
        """Measured across twelve independent mined families, not argued from one.

        The concern with correcting a spec on the strength of a single dataset is
        that the dataset, not the spec, might be the odd one out. It is not: on
        every mined grid tried — four seeds by three sample lengths — the winner
        lands **above** the expected maximum of noise and **below** the Sharpe
        the Deflated Sharpe Ratio actually requires. A figure marking only the
        first of those two lines would show a clean pass on all twelve, while the
        verdict is LIKELY_LUCK on all twelve.

        The mechanism is the correlation a search induces: 157 variants of four
        ideas are worth about ten independent trials, which pulls the expected
        maximum down well below the observed one. Any mined family has it.
        """
        from luckdetector.stats import deflated_sharpe_ratio_from_trials, sharpe_required_for_dsr

        result = mine(synthetic_prices(periods, seed=seed), cost_bps=1.0)
        sharpes = result.trials.values.mean(axis=1) / result.trials.values.std(axis=1, ddof=1)
        winner = int(np.argmax(sharpes))
        dsr = deflated_sharpe_ratio_from_trials(result.trials, index=winner)

        assert dsr.n_effective_trials < result.n_trials / 5, "family was not correlated"
        assert dsr.sharpe_annual > dsr.expected_max_sharpe_annual, "clears the drawn line"
        assert dsr.sharpe_annual < sharpe_required_for_dsr(dsr), "but misses the real bar"
        assert not dsr.passed
