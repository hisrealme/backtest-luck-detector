"""The rule table, at every boundary, plus the test that stops it saying 'no' to everything.

Two tests here matter more than the rest.

``test_a_planted_edge_is_called_skill`` runs the whole pipeline — PSR, DSR, PBO,
RC/SPA — on synthetic data with a genuine, persistent edge and requires the answer
to be LIKELY_SKILL. A luck detector that returns "luck" unconditionally is not a
detector, it is a rejection stamp, and this is the only test that can tell the
difference. BLUEPRINT §7 step 6 calls it out for the same reason.

``test_spa_with_no_candidate_is_not_a_rejection`` covers the case that forced
``NOT_APPLICABLE`` into existence: an SPA p-value of 1.0 obtained because nothing
in the family beat the benchmark is not evidence, in either direction.

The rest walk the rule table. Every one of the five rules has a test that fires
it, and every threshold is tested exactly *at* its boundary rather than
comfortably either side, because an off-by-one in a comparison operator is the
failure mode a rule table actually has.
"""

from __future__ import annotations

import numpy as np
import pytest

from luckdetector.report.verdict import (
    _RULES,
    SELECTION_AWARE_TESTS,
    TEST_ORDER,
    assess,
)
from luckdetector.stats.dsr import (
    DSR_THRESHOLD,
    DSRResult,
    deflated_sharpe_ratio_from_trials,
)
from luckdetector.stats.pbo import (
    PBO_THRESHOLD,
    DegradationResult,
    PBOResult,
    probability_of_backtest_overfitting,
)
from luckdetector.stats.psr import PSR_THRESHOLD, PSRResult, probabilistic_sharpe_ratio
from luckdetector.stats.reality_check import (
    SIGNIFICANCE_LEVEL,
    RealityCheckResult,
    reality_check,
)
from luckdetector.types import TrialMatrix

N_PERIODS = 1260
DAILY_VOL = 0.01


# --------------------------------------------------------------------- factories
# Built by hand rather than computed from data, so a threshold can be probed
# *exactly* at its boundary instead of hunting for a dataset that happens to land
# there. The statistics themselves are tested in their own files.


def make_psr(psr: float) -> PSRResult:
    return PSRResult(
        psr=psr,
        sharpe_annual=0.8,
        sharpe_per_period=0.8 / np.sqrt(252),
        benchmark_annual=0.0,
        benchmark_per_period=0.0,
        n_periods=N_PERIODS,
        periods_per_year=252,
        skewness=-0.1,
        kurtosis=3.4,
        standard_error_per_period=0.028,
    )


def make_dsr(dsr: float) -> DSRResult:
    return DSRResult(
        dsr=dsr,
        psr_at_zero=0.97,
        sharpe_annual=0.8,
        expected_max_sharpe_annual=0.4,
        n_trials=157,
        n_effective_trials=12.0,
        trial_sharpe_std_annual=0.3,
        psr_result=make_psr(0.97),
    )


def make_pbo(pbo: float) -> PBOResult:
    splits = 70
    return PBOResult(
        pbo=pbo,
        n_trials=50,
        n_blocks=8,
        n_splits=splits,
        exhaustive=True,
        logits=np.linspace(-1.0, 1.0, splits),
        relative_ranks=np.linspace(0.1, 0.9, splits),
        selected_trials=np.zeros(splits, dtype=np.int64),
        is_sharpe_selected=np.linspace(0.5, 1.5, splits),
        oos_sharpe_selected=np.linspace(-0.2, 0.4, splits),
        oos_sharpe_random=np.linspace(-0.1, 0.3, splits),
        degradation=DegradationResult(slope=-0.9, intercept=0.1, r_squared=0.4, n_points=splits),
        probability_of_loss=0.3,
        dominance_fraction=0.5,
        n_degenerate_subsamples=0,
    )


def make_spa(p_consistent: float, *, n_beating: int = 3, n_trials: int = 10) -> RealityCheckResult:
    """An SPA result. ``n_beating=0`` reproduces the empty-comparison case."""
    outperformance = np.full(n_trials, -0.001)
    outperformance[:n_beating] = 0.0002
    return RealityCheckResult(
        p_reality_check=p_consistent,
        p_lower=max(0.0, p_consistent - 0.01),
        p_consistent=p_consistent,
        p_upper=min(1.0, p_consistent + 0.01),
        statistic_reality_check=0.02,
        statistic_spa=0.0 if n_beating == 0 else 2.1,
        best_trial=0,
        best_label="MA(80,250)",
        best_trial_studentised=0,
        mean_outperformance=outperformance,
        omega=np.full(n_trials, 0.03),
        n_trials=n_trials,
        n_periods=N_PERIODS,
        periods_per_year=252,
        n_resamples=1000,
        block_length=5.0,
        benchmark_name="buy-and-hold",
        n_recentred_lower=n_trials - n_beating,
        n_recentred_consistent=n_trials - n_beating,
        n_degenerate=0,
    )


def noise_trials(rng: np.random.Generator, n_trials: int = 50) -> TrialMatrix:
    return TrialMatrix(rng.normal(0.0, DAILY_VOL, (n_trials, N_PERIODS)), periods_per_year=252)


def edge_trials(
    rng: np.random.Generator,
    *,
    n_trials: int = 50,
    n_good: int = 5,
    annual_sharpe: float = 3.0,
    n_periods: int = 2520,
) -> TrialMatrix:
    """Mostly noise, with ``n_good`` variants carrying a genuine persistent edge.

    The defaults are deliberately generous — ten years of daily data and a Sharpe
    of 3.0 in 5 of 50 variants — because at more modest effect sizes the verdict
    layer detects a real edge only sometimes. That is a property of the
    instrument, not of these fixtures, and
    ``test_detection_rate_at_a_realistic_edge_is_poor`` measures it rather than
    letting a generous fixture paper over it.
    """
    values = rng.normal(0.0, DAILY_VOL, (n_trials, n_periods))
    values[:n_good] += (annual_sharpe / np.sqrt(252)) * DAILY_VOL
    return TrialMatrix(values, periods_per_year=252)


def full_assessment(trials: TrialMatrix, *, n_resamples: int = 400) -> str:
    """Run every statistic in the package over one family and return the label."""
    winner = int(np.argmax(trials.values.mean(axis=1) / trials.values.std(axis=1, ddof=1)))
    return assess(
        psr=probabilistic_sharpe_ratio(trials.trial(winner)),
        dsr=deflated_sharpe_ratio_from_trials(trials),
        pbo=probability_of_backtest_overfitting(trials, n_blocks=8),
        spa=reality_check(trials, 0.0, n_resamples=n_resamples, seed=11),
    ).label


# ------------------------------------------------------------------ rule table


class TestRuleTable:
    def test_every_rule_is_reachable(self) -> None:
        """Five rules, five names, and a catch-all last so exactly one always fires."""
        names = [rule.name for rule in _RULES]
        assert names == [
            "no-evidence",
            "record-too-short",
            "flagged",
            "psr-only",
            "clean",
        ]
        assert len(set(names)) == len(names)

    def test_no_evidence_at_all(self) -> None:
        verdict = assess()
        assert verdict.label == "INCONCLUSIVE"
        assert verdict.results == []
        assert "No statistics were supplied" in verdict.narrative

    def test_only_an_inapplicable_test_is_still_no_evidence(self) -> None:
        """A test that could not run does not count as having run."""
        verdict = assess(spa=make_spa(1.0, n_beating=0))
        assert verdict.label == "INCONCLUSIVE"
        assert verdict.n_failed == 0
        assert verdict.applicable == []

    def test_record_too_short_when_nothing_else_objects(self) -> None:
        """A lone PSR flag is 'we cannot tell yet', not 'this is luck'."""
        verdict = assess(
            psr=make_psr(0.60), dsr=make_dsr(0.99), pbo=make_pbo(0.05), spa=make_spa(0.01)
        )
        assert verdict.label == "INCONCLUSIVE"
        assert [r.name for r in verdict.flags] == ["psr"]
        assert "too short" in verdict.narrative

    def test_psr_alone_failing_is_inconclusive(self) -> None:
        assert assess(psr=make_psr(0.60)).label == "INCONCLUSIVE"

    def test_psr_flag_plus_any_other_flag_is_luck(self) -> None:
        """Once something else objects, the short record stops being the story."""
        verdict = assess(psr=make_psr(0.60), pbo=make_pbo(0.84))
        assert verdict.label == "LIKELY_LUCK"
        assert [r.name for r in verdict.flags] == ["psr", "pbo"]

    @pytest.mark.parametrize(
        ("kwargs", "flagged"),
        [
            ({"dsr": make_dsr(0.10)}, "dsr"),
            ({"pbo": make_pbo(0.84)}, "pbo"),
            ({"spa": make_spa(0.43)}, "spa"),
        ],
    )
    def test_a_single_selection_aware_flag_is_enough(
        self, kwargs: dict[str, object], flagged: str
    ) -> None:
        """One objection is not outvoted. The asymmetry is deliberate."""
        verdict = assess(psr=make_psr(0.99), **kwargs)  # type: ignore[arg-type]
        assert verdict.label == "LIKELY_LUCK"
        assert [r.name for r in verdict.flags] == [flagged]

    def test_psr_alone_can_never_certify_skill(self) -> None:
        """The naive test this package exists to debunk cannot deliver a clean bill.

        The SPY winner posts a PSR of 0.9764 and is luck on every selection-aware
        test. A verdict layer that returned LIKELY_SKILL here would reproduce
        exactly the mistake it was built to catch.
        """
        verdict = assess(psr=make_psr(0.9764))
        assert verdict.label == "INCONCLUSIVE"
        assert verdict.n_failed == 0
        assert "prices the search" in verdict.narrative

    def test_a_vacuous_spa_cannot_stand_in_for_a_selection_test(self) -> None:
        verdict = assess(psr=make_psr(0.99), spa=make_spa(1.0, n_beating=0))
        assert verdict.label == "INCONCLUSIVE"

    def test_clean_sweep_is_skill(self) -> None:
        verdict = assess(
            psr=make_psr(0.99), dsr=make_dsr(0.99), pbo=make_pbo(0.05), spa=make_spa(0.01)
        )
        assert verdict.label == "LIKELY_SKILL"
        assert verdict.n_failed == 0

    def test_skill_without_psr_supplied(self) -> None:
        """Declining to ask a question is not the same as failing it.

        The rule table reasons only about evidence it was given. PSR is not
        mandatory for LIKELY_SKILL — DSR carries its own PSR internally — but a
        PSR that *was* supplied and failed does block the verdict.
        """
        assert assess(dsr=make_dsr(0.99), pbo=make_pbo(0.05)).label == "LIKELY_SKILL"
        assert assess(psr=make_psr(0.5), dsr=make_dsr(0.99), pbo=make_pbo(0.05)).label != (
            "LIKELY_SKILL"
        )

    def test_skill_needs_only_one_selection_aware_test(self) -> None:
        for kwargs in ({"dsr": make_dsr(0.99)}, {"pbo": make_pbo(0.05)}, {"spa": make_spa(0.01)}):
            verdict = assess(psr=make_psr(0.99), **kwargs)  # type: ignore[arg-type]
            assert verdict.label == "LIKELY_SKILL"


# ------------------------------------------------------------------- thresholds


class TestThresholdBoundaries:
    """Each comparison probed exactly at its constant, where operators go wrong."""

    def test_psr_passes_exactly_at_its_threshold(self) -> None:
        assert assess(psr=make_psr(PSR_THRESHOLD)).result("psr").passed
        assert not assess(psr=make_psr(np.nextafter(PSR_THRESHOLD, 0.0))).result("psr").passed

    def test_dsr_passes_exactly_at_its_threshold(self) -> None:
        assert assess(dsr=make_dsr(DSR_THRESHOLD)).result("dsr").passed
        assert not assess(dsr=make_dsr(np.nextafter(DSR_THRESHOLD, 0.0))).result("dsr").passed

    def test_pbo_fails_exactly_at_its_threshold(self) -> None:
        """PBO passes *below* the bar, so equality is a failure."""
        assert not assess(pbo=make_pbo(PBO_THRESHOLD)).result("pbo").passed
        assert assess(pbo=make_pbo(np.nextafter(PBO_THRESHOLD, 0.0))).result("pbo").passed

    def test_spa_fails_exactly_at_its_significance_level(self) -> None:
        """A p-value must be strictly below the level to reject."""
        assert not assess(spa=make_spa(SIGNIFICANCE_LEVEL)).result("spa").passed
        assert assess(spa=make_spa(np.nextafter(SIGNIFICANCE_LEVEL, 0.0))).result("spa").passed

    def test_thresholds_are_recorded_on_the_evidence(self) -> None:
        verdict = assess(
            psr=make_psr(0.99), dsr=make_dsr(0.99), pbo=make_pbo(0.05), spa=make_spa(0.01)
        )
        assert verdict.result("psr").threshold == PSR_THRESHOLD
        assert verdict.result("dsr").threshold == DSR_THRESHOLD
        assert verdict.result("pbo").threshold == PBO_THRESHOLD
        assert verdict.result("spa").threshold == SIGNIFICANCE_LEVEL


# -------------------------------------------------------------- not applicable


class TestNotApplicable:
    def test_spa_with_no_candidate_is_not_a_rejection(self) -> None:
        """The SPY case, and the reason the third state exists.

        p = 1.0 obtained because nothing beat the benchmark is an empty
        comparison. Scored as a flag it would be the most decisive rejection in
        the report; scored as a pass it would be nonsense.
        """
        result = assess(spa=make_spa(1.0, n_beating=0, n_trials=157)).result("spa")
        assert result.status == "NOT_APPLICABLE"
        assert not result.flagged
        assert not result.passed
        assert not result.applicable

    def test_the_finding_survives_into_the_narrative(self) -> None:
        """Inapplicable is not silent. The strongest sentence still gets said."""
        verdict = assess(pbo=make_pbo(0.84), spa=make_spa(1.0, n_beating=0, n_trials=157))
        assert "Not one of the 157 strategies beat buy-and-hold" in verdict.narrative
        assert "empty comparison rather than a rejection" in verdict.narrative

    def test_an_inapplicable_spa_does_not_change_the_verdict(self) -> None:
        """It neither rescues nor condemns; the other evidence decides alone."""
        without = assess(psr=make_psr(0.99), pbo=make_pbo(0.84))
        with_vacuous = assess(
            psr=make_psr(0.99), pbo=make_pbo(0.84), spa=make_spa(1.0, n_beating=0)
        )
        assert with_vacuous.label == without.label == "LIKELY_LUCK"
        assert with_vacuous.n_failed == without.n_failed

    def test_one_survivor_is_enough_to_make_spa_applicable(self) -> None:
        assert assess(spa=make_spa(0.4, n_beating=1)).result("spa").status == "FAIL"


# ---------------------------------------------------------------- presentation


class TestEvidenceAndNarrative:
    def test_results_are_reported_in_evaluation_order(self) -> None:
        verdict = assess(
            spa=make_spa(0.01), pbo=make_pbo(0.05), dsr=make_dsr(0.99), psr=make_psr(0.99)
        )
        assert tuple(r.name for r in verdict.results) == TEST_ORDER

    def test_narrative_leads_with_the_label_and_the_reason(self) -> None:
        verdict = assess(psr=make_psr(0.99), pbo=make_pbo(0.84))
        first = verdict.narrative.splitlines()[0]
        assert first.startswith("LIKELY LUCK")
        assert "objected" in first

    def test_narrative_carries_every_interpretation(self) -> None:
        psr, pbo = make_psr(0.99), make_pbo(0.84)
        verdict = assess(psr=psr, pbo=pbo)
        assert psr.interpretation in verdict.narrative
        assert pbo.interpretation in verdict.narrative
        assert "[PASS]" in verdict.narrative
        assert "[FAIL]" in verdict.narrative

    def test_each_result_keeps_the_detail_needed_to_reproduce_it(self) -> None:
        verdict = assess(dsr=make_dsr(0.77), pbo=make_pbo(0.84))
        assert verdict.result("dsr").detail["n_effective_trials"] == 12.0
        assert verdict.result("pbo").detail["n_splits"] == 70

    def test_selection_aware_set_is_what_the_docs_claim(self) -> None:
        assert sorted(SELECTION_AWARE_TESTS) == ["dsr", "pbo", "spa"]
        assert "psr" not in SELECTION_AWARE_TESTS


# ------------------------------------------------------------------ end to end


class TestAgainstRealStatistics:
    """Wired to the actual statistics, not to hand-built results."""

    def test_a_planted_edge_is_called_skill(self, rng: np.random.Generator) -> None:
        """The test that stops this being a machine that says 'no' to everything.

        Five of fifty variants carry a genuine, persistent annualised Sharpe of
        3.0 over ten years. Every statistic in the package is run on that family
        and the verdict must come back LIKELY_SKILL. BLUEPRINT §7 step 6 asks for
        exactly this, and it is the only test that distinguishes a detector from a
        rubber stamp.

        The effect size is large on purpose, and not to flatter the tool: it is
        chosen so the assertion does not ride on the seed. Measured across 25
        independent datasets at this configuration the verdict is LIKELY_SKILL
        every time, with a worst-case DSR of 0.994 against its 0.95 bar. At the
        weaker edge this fixture used to plant, it was 20%.
        """
        trials = edge_trials(rng)
        winner = int(np.argmax(trials.values.mean(axis=1) / trials.values.std(axis=1, ddof=1)))

        verdict = assess(
            psr=probabilistic_sharpe_ratio(trials.trial(winner)),
            dsr=deflated_sharpe_ratio_from_trials(trials),
            pbo=probability_of_backtest_overfitting(trials, n_blocks=8),
            spa=reality_check(trials, 0.0, n_resamples=500, seed=11),
        )
        assert verdict.label == "LIKELY_SKILL"
        assert verdict.n_failed == 0

    def test_detection_rate_at_a_realistic_edge_is_poor(self) -> None:
        """The limitation, measured and pinned rather than left for a reader to find.

        Ten of fifty variants with a real annualised Sharpe of 2.0 over five
        years — a strong edge by any practical standard — is called LIKELY_SKILL
        only about a fifth of the time. DSR is the binding constraint: its median
        over these datasets sits near 0.89 against a 0.95 bar.

        This is a conservative instrument, not a broken one: the winner of a
        50-variant search genuinely does have to clear the expected maximum of 50
        noise trials, and at this sample length it often cannot. But it means a
        LIKELY_LUCK verdict on a real strategy family is weak evidence of absence,
        and the number belongs in the test suite rather than in a footnote.

        Asserted as a range, following the Phase 5 rule about never asserting on a
        single draw.
        """
        labels = []
        for seed in range(20):
            rng = np.random.default_rng(20260900 + seed)
            trials = edge_trials(rng, n_good=10, annual_sharpe=2.0, n_periods=N_PERIODS)
            labels.append(full_assessment(trials, n_resamples=300))

        detected = float(np.mean([label == "LIKELY_SKILL" for label in labels]))
        assert 0.0 < detected < 0.5

    def test_pure_noise_is_called_luck(self, rng: np.random.Generator) -> None:
        """The mirror image, on a family with no edge planted anywhere."""
        trials = noise_trials(rng)
        winner = int(np.argmax(trials.values.mean(axis=1) / trials.values.std(axis=1, ddof=1)))

        verdict = assess(
            psr=probabilistic_sharpe_ratio(trials.trial(winner)),
            dsr=deflated_sharpe_ratio_from_trials(trials),
            pbo=probability_of_backtest_overfitting(trials, n_blocks=8),
            spa=reality_check(trials, 0.0, n_resamples=500, seed=11),
        )
        assert verdict.label == "LIKELY_LUCK"
        assert "dsr" in [r.name for r in verdict.flags]

    def test_noise_is_never_called_skill(self) -> None:
        """The error that would matter most, checked across datasets rather than one."""
        labels = [
            full_assessment(noise_trials(np.random.default_rng(20260950 + seed)), n_resamples=300)
            for seed in range(20)
        ]
        assert "LIKELY_SKILL" not in labels

    def test_a_family_that_loses_to_its_benchmark(self, rng: np.random.Generator) -> None:
        """The shape of the SPY result: SPA has no candidate, the others decide.

        Every variant is a watered-down copy of the benchmark, so none of them
        beats it and SPA drops out as inapplicable — while remaining the source
        of the sharpest sentence in the report.
        """
        benchmark = rng.normal(2.0 / np.sqrt(252) * DAILY_VOL, DAILY_VOL, N_PERIODS)
        trials = TrialMatrix(
            benchmark * 0.5 + rng.normal(0.0, DAILY_VOL * 0.5, (20, N_PERIODS)),
            periods_per_year=252,
        )
        spa = reality_check(trials, benchmark, n_resamples=500, seed=11)

        verdict = assess(pbo=probability_of_backtest_overfitting(trials, n_blocks=8), spa=spa)
        assert spa.n_beating_benchmark == 0
        assert verdict.result("spa").status == "NOT_APPLICABLE"
        assert "had no candidate to price" in verdict.narrative
