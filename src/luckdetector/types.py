"""Core data model.

Two input shapes drive everything in this package:

``ReturnSeries``
    One strategy's periodic returns. Enough for PSR / DSR / MinTRL / haircut.

``TrialMatrix``
    *Every* strategy that was tried, aligned on the same calendar. Required by
    CSCV/PBO and by White's Reality Check, both of which reason about the family
    of trials rather than the winner alone.

Both are frozen and validate on construction, so no downstream routine ever has
to re-check shape, finiteness, or alignment.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt

from .exceptions import DataValidationError, InsufficientDataError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

VerdictLabel = Literal["LIKELY_SKILL", "INCONCLUSIVE", "LIKELY_LUCK"]

#: Outcome of one test. ``NOT_APPLICABLE`` is not a polite failure — it means the
#: statistic had nothing to weigh, and it must never be counted as evidence in
#: either direction. See :class:`TestResult`.
TestStatus = Literal["PASS", "FAIL", "NOT_APPLICABLE"]

#: Calendar conventions accepted by ``periods_per_year``.
COMMON_FREQUENCIES: dict[str, int] = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "annual": 1,
}


def _to_float_array(values: Iterable[float] | FloatArray, *, label: str) -> FloatArray:
    """Coerce to a contiguous float64 array, failing loudly on junk."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise DataValidationError(
            f"{label} contains {n_bad} non-finite value(s) (NaN or inf). "
            "Clean or explicitly drop them before constructing this object."
        )
    return np.ascontiguousarray(arr)


def _validate_periods_per_year(ppy: int) -> None:
    if not isinstance(ppy, (int, np.integer)) or ppy <= 0:
        raise DataValidationError(f"periods_per_year must be a positive integer, got {ppy!r}.")


@dataclass(frozen=True)
class ReturnSeries:
    """A single strategy's *simple* periodic returns.

    Parameters
    ----------
    values:
        1-D array of simple returns per period (``0.01`` is +1%). Not prices, not
        cumulative wealth, not percentages.
    periods_per_year:
        Number of return observations in a year: 252 daily, 52 weekly, 12 monthly.
        Used only for annualisation — never inside PSR/DSR, which are per-period.
    name:
        Label used in reports.
    """

    values: FloatArray
    periods_per_year: int = 252
    name: str = "strategy"

    MIN_PERIODS: ClassVar[int] = 2

    def __post_init__(self) -> None:
        arr = _to_float_array(self.values, label="ReturnSeries.values")
        if arr.ndim != 1:
            raise DataValidationError(
                f"ReturnSeries.values must be 1-D, got shape {arr.shape}. "
                "For many strategies at once use TrialMatrix."
            )
        if arr.size < self.MIN_PERIODS:
            raise InsufficientDataError(
                f"ReturnSeries needs at least {self.MIN_PERIODS} observations, got {arr.size}."
            )
        if np.any(arr < -1.0):
            raise DataValidationError(
                "ReturnSeries.values contains returns below -100%, which is impossible for "
                "simple returns. Did you pass log returns, percentages, or prices?"
            )
        _validate_periods_per_year(self.periods_per_year)
        object.__setattr__(self, "values", arr)

    # ------------------------------------------------------------------ size

    @property
    def n_periods(self) -> int:
        """Number of return observations."""
        return int(self.values.size)

    @property
    def years(self) -> float:
        """Length of the track record in years."""
        return self.n_periods / self.periods_per_year

    def __len__(self) -> int:
        return self.n_periods

    # ------------------------------------------------------------ transforms

    def cumulative(self) -> FloatArray:
        """Wealth index starting at 1.0 (length ``n_periods``)."""
        return np.asarray(np.cumprod(1.0 + self.values), dtype=np.float64)

    def total_return(self) -> float:
        """Compounded return over the whole sample."""
        return float(self.cumulative()[-1] - 1.0)

    def slice(self, start: int, stop: int) -> ReturnSeries:
        """Return a sub-period as a new ``ReturnSeries``."""
        return ReturnSeries(
            values=self.values[start:stop],
            periods_per_year=self.periods_per_year,
            name=self.name,
        )

    def looks_like_percentages(self) -> bool:
        """Heuristic guard: returns stated as ``1.5`` instead of ``0.015``.

        A daily strategy with a 40% standard deviation *per period* is far more
        likely to be a units bug than a real track record.
        """
        return bool(np.std(self.values) > 0.4)


@dataclass(frozen=True)
class TrialMatrix:
    """All strategy variants that were tried, aligned on a common calendar.

    Parameters
    ----------
    values:
        Shape ``(n_trials, n_periods)``. Row ``i`` is the return stream of trial ``i``.
    periods_per_year:
        As for :class:`ReturnSeries`.
    labels:
        Human-readable descriptor per trial, e.g. ``"MA(10,50)"``. Auto-generated
        if omitted.
    """

    values: FloatArray
    periods_per_year: int = 252
    labels: Sequence[str] = ()

    MIN_TRIALS: ClassVar[int] = 2

    def __post_init__(self) -> None:
        arr = _to_float_array(self.values, label="TrialMatrix.values")
        if arr.ndim != 2:
            raise DataValidationError(
                f"TrialMatrix.values must be 2-D (n_trials, n_periods), got shape {arr.shape}."
            )
        n_trials, n_periods = arr.shape
        if n_trials < self.MIN_TRIALS:
            raise InsufficientDataError(
                f"TrialMatrix needs at least {self.MIN_TRIALS} trials, got {n_trials}. "
                "With a single trial there is no selection bias to measure."
            )
        if n_periods < 2:
            raise InsufficientDataError(f"TrialMatrix needs at least 2 periods, got {n_periods}.")
        if np.any(arr < -1.0):
            raise DataValidationError(
                "TrialMatrix.values contains returns below -100%. Check units."
            )
        _validate_periods_per_year(self.periods_per_year)

        labels = (
            tuple(self.labels) if len(self.labels) else tuple(f"trial_{i}" for i in range(n_trials))
        )
        if len(labels) != n_trials:
            raise DataValidationError(
                f"Got {len(labels)} labels for {n_trials} trials; lengths must match."
            )

        object.__setattr__(self, "values", arr)
        object.__setattr__(self, "labels", labels)

    # ------------------------------------------------------------------ size

    @property
    def n_trials(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_periods(self) -> int:
        return int(self.values.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_trials, self.n_periods)

    def __len__(self) -> int:
        return self.n_trials

    # ------------------------------------------------------------ transforms

    def trial(self, index: int) -> ReturnSeries:
        """Extract one trial as a :class:`ReturnSeries`."""
        return ReturnSeries(
            values=self.values[index],
            periods_per_year=self.periods_per_year,
            name=self.labels[index],
        )

    def correlation(self) -> FloatArray:
        """Cross-trial correlation matrix, used to estimate *effective* trial count."""
        return np.asarray(np.corrcoef(self.values), dtype=np.float64)

    def columns(self, start: int, stop: int) -> TrialMatrix:
        """Slice a contiguous block of periods across all trials (used by CSCV)."""
        return TrialMatrix(
            values=self.values[:, start:stop],
            periods_per_year=self.periods_per_year,
            labels=self.labels,
        )


@dataclass(frozen=True)
class TestResult:
    """Outcome of a single statistical test, in a form the report can render.

    ``PASS`` always means *the evidence is consistent with genuine skill*,
    regardless of whether the underlying statistic is a p-value or a probability.
    Keeping that polarity uniform is what lets the verdict layer stay simple.

    **The third state is load-bearing.** A test can produce a number without
    producing evidence, and collapsing that into pass/fail misreports it in one
    direction or the other. The case that forced this: on the SPY grid, White's
    Reality Check returns *p = 1.0000* against buy-and-hold — but only because not
    one of the 157 variants beat the benchmark, so the statistic is
    :math:`\\max(0, \\cdot) = 0` and no resample can fall below it. Read as ``FAIL``
    that is the strongest rejection in the report; read as ``PASS`` it is
    nonsense. It is neither. The test never ran, and ``NOT_APPLICABLE`` says so
    while the interpretation string carries what *was* learned.
    """

    __test__: ClassVar[bool] = False  # stop pytest trying to collect this as a test class

    name: str
    statistic: float
    threshold: float
    status: TestStatus
    p_value: float | None = None
    interpretation: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Is this evidence *for* skill? ``NOT_APPLICABLE`` is not."""
        return self.status == "PASS"

    @property
    def flagged(self) -> bool:
        """Did this test actively raise a flag? ``NOT_APPLICABLE`` never does."""
        return self.status == "FAIL"

    @property
    def applicable(self) -> bool:
        """Did the test have anything to weigh?"""
        return self.status != "NOT_APPLICABLE"


@dataclass(frozen=True)
class Verdict:
    """The final call, plus every piece of evidence behind it."""

    label: VerdictLabel
    results: list[TestResult]
    narrative: str = ""

    @property
    def n_failed(self) -> int:
        """Tests that raised a flag. Inapplicable tests are not failures."""
        return sum(1 for r in self.results if r.flagged)

    @property
    def flags(self) -> list[TestResult]:
        """The tests that objected, in the order they were evaluated."""
        return [r for r in self.results if r.flagged]

    @property
    def applicable(self) -> list[TestResult]:
        """The tests that had something to weigh."""
        return [r for r in self.results if r.applicable]

    def result(self, name: str) -> TestResult:
        for r in self.results:
            if r.name == name:
                return r
        raise KeyError(f"No test named {name!r}; have {[r.name for r in self.results]}.")
