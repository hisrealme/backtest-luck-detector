"""Validation behaviour of the core data model.

These tests exist because every downstream statistic assumes clean, aligned,
correctly-scaled input. If validation is loose here, every number in the report
is quietly untrustworthy.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from luckdetector.exceptions import DataValidationError, InsufficientDataError
from luckdetector.types import ReturnSeries, TestResult, TestStatus, TrialMatrix, Verdict


class TestReturnSeries:
    def test_accepts_lists_and_arrays(self) -> None:
        assert ReturnSeries([0.01, -0.01, 0.02]).n_periods == 3
        assert ReturnSeries(np.array([0.01, -0.01])).n_periods == 2

    def test_rejects_nan(self) -> None:
        with pytest.raises(DataValidationError, match="non-finite"):
            ReturnSeries([0.01, np.nan, 0.02])

    def test_rejects_inf(self) -> None:
        with pytest.raises(DataValidationError, match="non-finite"):
            ReturnSeries([0.01, np.inf])

    def test_rejects_two_dimensional(self) -> None:
        with pytest.raises(DataValidationError, match="1-D"):
            ReturnSeries(np.zeros((3, 4)))

    def test_rejects_returns_below_minus_one(self) -> None:
        with pytest.raises(DataValidationError, match="below -100%"):
            ReturnSeries([0.01, -1.5])

    def test_rejects_too_few_observations(self) -> None:
        with pytest.raises(InsufficientDataError):
            ReturnSeries([0.01])

    @pytest.mark.parametrize("bad_ppy", [0, -252, 2.5])
    def test_rejects_bad_frequency(self, bad_ppy: object) -> None:
        with pytest.raises(DataValidationError, match="positive integer"):
            ReturnSeries([0.01, 0.02], periods_per_year=bad_ppy)  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        series = ReturnSeries([0.01, 0.02])
        with pytest.raises(dataclasses.FrozenInstanceError):
            series.periods_per_year = 12  # type: ignore[misc]

    def test_years_and_length(self) -> None:
        series = ReturnSeries(np.zeros(504), periods_per_year=252)
        assert series.years == pytest.approx(2.0)
        assert len(series) == 504

    def test_cumulative_and_total_return(self) -> None:
        series = ReturnSeries([0.10, 0.10])
        assert series.cumulative()[-1] == pytest.approx(1.21)
        assert series.total_return() == pytest.approx(0.21)

    def test_slice_preserves_metadata(self) -> None:
        series = ReturnSeries(np.zeros(100), periods_per_year=12, name="abc")
        sub = series.slice(10, 20)
        assert sub.n_periods == 10
        assert sub.periods_per_year == 12
        assert sub.name == "abc"

    def test_percentage_heuristic(self) -> None:
        assert ReturnSeries([0.5, -0.5, 0.9, -0.9]).looks_like_percentages()
        assert not ReturnSeries([0.005, -0.005, 0.009]).looks_like_percentages()


class TestTrialMatrix:
    def test_shape_and_labels(self) -> None:
        matrix = TrialMatrix(np.zeros((5, 100)))
        assert matrix.shape == (5, 100)
        assert matrix.n_trials == 5
        assert matrix.labels[0] == "trial_0"

    def test_rejects_one_dimensional(self) -> None:
        with pytest.raises(DataValidationError, match="2-D"):
            TrialMatrix(np.zeros(100))

    def test_rejects_single_trial(self) -> None:
        with pytest.raises(InsufficientDataError, match="at least 2 trials"):
            TrialMatrix(np.zeros((1, 100)))

    def test_rejects_too_few_periods(self) -> None:
        with pytest.raises(InsufficientDataError, match="at least 2 periods"):
            TrialMatrix(np.zeros((5, 1)))

    def test_rejects_returns_below_minus_one(self) -> None:
        values = np.zeros((3, 10))
        values[1, 4] = -1.5
        with pytest.raises(DataValidationError, match="below -100%"):
            TrialMatrix(values)

    def test_len_is_trial_count(self) -> None:
        assert len(TrialMatrix(np.zeros((7, 20)))) == 7

    def test_rejects_label_length_mismatch(self) -> None:
        with pytest.raises(DataValidationError, match="labels"):
            TrialMatrix(np.zeros((3, 10)), labels=["a", "b"])

    def test_trial_extraction_round_trips(self, rng: np.random.Generator) -> None:
        values = rng.normal(0, 0.01, (4, 50))
        matrix = TrialMatrix(values, labels=list("abcd"))
        extracted = matrix.trial(2)
        assert extracted.name == "c"
        np.testing.assert_allclose(extracted.values, values[2])

    def test_column_slice_keeps_all_trials(self, rng: np.random.Generator) -> None:
        matrix = TrialMatrix(rng.normal(0, 0.01, (6, 100)))
        block = matrix.columns(20, 60)
        assert block.shape == (6, 40)

    def test_correlation_is_square_and_unit_diagonal(self, rng: np.random.Generator) -> None:
        matrix = TrialMatrix(rng.normal(0, 0.01, (5, 200)))
        corr = matrix.correlation()
        assert corr.shape == (5, 5)
        np.testing.assert_allclose(np.diag(corr), 1.0)


class TestVerdict:
    def _result(self, name: str, status: TestStatus) -> TestResult:
        return TestResult(name=name, statistic=0.5, threshold=0.5, status=status)

    def test_counts_flags(self) -> None:
        verdict = Verdict(
            label="LIKELY_LUCK",
            results=[self._result("dsr", "FAIL"), self._result("pbo", "PASS")],
        )
        assert verdict.n_failed == 1

    def test_inapplicable_is_not_a_failure(self) -> None:
        """The distinction the whole third state exists for.

        A test that had nothing to weigh must not be counted as an objection, or
        the SPY report would claim SPA rejected when SPA never ran.
        """
        verdict = Verdict(
            label="LIKELY_LUCK",
            results=[self._result("dsr", "FAIL"), self._result("spa", "NOT_APPLICABLE")],
        )
        assert verdict.n_failed == 1
        assert [r.name for r in verdict.flags] == ["dsr"]
        assert [r.name for r in verdict.applicable] == ["dsr"]

    def test_status_properties_are_mutually_exclusive(self) -> None:
        for status, expected in (
            ("PASS", (True, False, True)),
            ("FAIL", (False, True, True)),
            ("NOT_APPLICABLE", (False, False, False)),
        ):
            result = self._result("dsr", status)  # type: ignore[arg-type]
            assert (result.passed, result.flagged, result.applicable) == expected

    def test_lookup_by_name(self) -> None:
        verdict = Verdict(label="INCONCLUSIVE", results=[self._result("dsr", "PASS")])
        assert verdict.result("dsr").passed
        with pytest.raises(KeyError):
            verdict.result("missing")
