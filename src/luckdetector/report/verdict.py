"""The rule table: four statistics in, one defensible answer out.

Why this layer is the weakest one in the package, stated first
--------------------------------------------------------------
Every statistic behind this module is derived from a published result and checked
against simulation. **The rules that combine them are not.** There is no
literature on how to weigh a Deflated Sharpe Ratio against a PBO against an SPA
p-value, because the question is one of judgement rather than of mathematics. The
thresholds are invented, the precedence is invented, and both are stated here as
named constants and an explicit table so that a reader can disagree with a
specific line rather than with a black box.

That is the whole design constraint: **no composite score.** A weighted average of
four statistics would look more sophisticated and would be less honest, because it
would hide which piece of evidence drove the answer and would let two mild
concerns average into a clean bill of health. Instead each test raises its own
flag, the flags are combined by a rule table short enough to read, and the verdict
records which rule fired.

The rules
---------
Evaluated in order, first match wins:

===================  ================  =================================================
rule                 verdict           when
===================  ================  =================================================
``no-evidence``      ``INCONCLUSIVE``  no test could be run at all
``record-too-short`` ``INCONCLUSIVE``  PSR is the *only* test that flagged
``flagged``          ``LIKELY_LUCK``   any selection-aware test flagged
``psr-only``         ``INCONCLUSIVE``  nothing flagged, but only PSR was available
``clean``            ``LIKELY_SKILL``  nothing flagged, and a selection-aware test ran
===================  ================  =================================================

Three of those five deserve their reasoning written down.

**Why a lone PSR flag is inconclusive rather than damning.** PSR failing means the
record is too short to distinguish this Sharpe ratio from its benchmark given the
skew and fat tails. If nothing else objected, the honest report is *we cannot
tell yet* — and :func:`luckdetector.stats.psr.min_track_record_length` says how
much more data would settle it. If something else *did* object, the short record
stops being the interesting fact and the rule below takes over.

**Why PSR alone can never return LIKELY_SKILL.** PSR is exactly the naive test
this package exists to debunk: it knows the length of the record and nothing about
how many strategies were tried to find it. The SPY winner posts a PSR of 0.9764
and is luck on every selection-aware test. A verdict layer that could certify
skill from PSR alone would reproduce the error it was built to catch, so
``LIKELY_SKILL`` requires at least one of DSR, PBO or SPA to have run and passed.

**Why one flag is enough to say LIKELY_LUCK.** The alternative — counting flags,
or averaging them — treats the tests as interchangeable measurements of one
quantity. They are not: DSR prices multiplicity, PBO prices selection stability,
SPA prices the benchmark. Each can be the only one positioned to see a given
failure, so a single objection is not outvoted. The cost of this choice is real
and worth naming: **the layer is asymmetric, and deliberately so.** It is much
easier to be told your backtest is luck than to be told it is skill. For a tool
whose entire purpose is to resist a flattering answer, that is the right way round
— but it means ``LIKELY_LUCK`` is a weaker claim than ``LIKELY_SKILL``, and the
narrative names the specific flag rather than implying a consensus.

A test can produce a number without producing evidence
------------------------------------------------------
``NOT_APPLICABLE`` exists because of a case Phase 6 threw up on real data. Against
buy-and-hold the SPY grid returns an SPA p-value of exactly 1.0000 — not because
the test looked and found nothing, but because not one of the 157 variants beat
the benchmark, so the statistic is :math:`\\max(0, \\cdot) = 0` and no resample can
fall below it. Scored as a failure that is the most decisive rejection in the
report; scored as a pass it is nonsense. It is neither, and an inapplicable test
neither flags nor counts toward the evidence needed for ``LIKELY_SKILL``.

The finding does not disappear, though — it moves into the interpretation string,
where it belongs. "Not one of 157 strategies beat buy-and-hold" is the strongest
sentence in the SPY analysis. It is simply not a *p-value*.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ..stats.dsr import DSR_THRESHOLD, DSRResult
from ..stats.pbo import PBO_THRESHOLD, PBOResult
from ..stats.psr import PSR_THRESHOLD, PSRResult
from ..stats.reality_check import SIGNIFICANCE_LEVEL, RealityCheckResult
from ..types import TestResult, TestStatus, Verdict, VerdictLabel

__all__ = [
    "SELECTION_AWARE_TESTS",
    "TEST_ORDER",
    "assess",
]

TestName = Literal["psr", "dsr", "pbo", "spa"]

#: Evaluation order, which is also the order evidence is reported in: weakest
#: question first, so a reader meets "is this record long enough" before "did the
#: search that found it beat doing nothing".
TEST_ORDER: tuple[TestName, ...] = ("psr", "dsr", "pbo", "spa")

#: The tests that know a search took place. ``LIKELY_SKILL`` requires at least one
#: of these to have run and passed — PSR on its own is the naive test, and
#: certifying skill from it would reproduce the error this package exists to catch.
SELECTION_AWARE_TESTS: frozenset[str] = frozenset({"dsr", "pbo", "spa"})

#: Human titles for the report layer, kept next to the short keys so the two
#: cannot drift apart.
TEST_TITLES: dict[str, str] = {
    "psr": "Probabilistic Sharpe Ratio",
    "dsr": "Deflated Sharpe Ratio",
    "pbo": "Probability of Backtest Overfitting",
    "spa": "Reality Check / SPA",
}


def _status(passed: bool) -> TestStatus:
    return "PASS" if passed else "FAIL"


def _psr_evidence(result: PSRResult) -> TestResult:
    return TestResult(
        name="psr",
        statistic=result.psr,
        threshold=PSR_THRESHOLD,
        status=_status(result.passed),
        interpretation=result.interpretation,
        detail=result.as_dict(),
    )


def _dsr_evidence(result: DSRResult) -> TestResult:
    return TestResult(
        name="dsr",
        statistic=result.dsr,
        threshold=DSR_THRESHOLD,
        status=_status(result.passed),
        interpretation=result.interpretation,
        detail=result.as_dict(),
    )


def _pbo_evidence(result: PBOResult) -> TestResult:
    return TestResult(
        name="pbo",
        statistic=result.pbo,
        threshold=PBO_THRESHOLD,
        status=_status(result.passed),
        interpretation=result.interpretation,
        detail=result.as_dict(),
    )


def _spa_evidence(result: RealityCheckResult) -> TestResult:
    """SPA's evidence, with the empty-comparison case separated out.

    When nothing in the family beat the benchmark there is no maximum to price,
    so the p-value of 1.0 describes an empty comparison rather than a rejection.
    That is recorded as ``NOT_APPLICABLE`` and the finding is moved into the
    interpretation, where it reads as what it is.
    """
    if result.n_beating_benchmark == 0:
        return TestResult(
            name="spa",
            statistic=result.statistic_spa,
            threshold=SIGNIFICANCE_LEVEL,
            status="NOT_APPLICABLE",
            p_value=result.p_consistent,
            interpretation=(
                f"Not one of the {result.n_trials:,} strategies beat "
                f"{result.benchmark_name} on average, so the test had no candidate to "
                f"price and its p-value of {result.p_consistent:.4f} reflects an empty "
                "comparison rather than a rejection. The absence of any candidate is "
                "itself the finding, and a harder one than any p-value here."
            ),
            detail=result.as_dict(),
        )
    return TestResult(
        name="spa",
        statistic=result.p_consistent,
        threshold=SIGNIFICANCE_LEVEL,
        status=_status(result.passed),
        p_value=result.p_consistent,
        interpretation=result.interpretation,
        detail=result.as_dict(),
    )


@dataclass(frozen=True)
class _Evidence:
    """The assembled test results, with the questions the rules need to ask."""

    results: tuple[TestResult, ...]

    @property
    def flag_names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.results if r.flagged)

    @property
    def has_applicable(self) -> bool:
        return any(r.applicable for r in self.results)

    @property
    def has_selection_aware(self) -> bool:
        return any(r.applicable and r.name in SELECTION_AWARE_TESTS for r in self.results)


@dataclass(frozen=True)
class _Rule:
    """One line of the rule table."""

    name: str
    label: VerdictLabel
    reason: str
    applies: Callable[[_Evidence], bool]


#: The rule table. Evaluated in order, first match wins; the last rule is a
#: catch-all so exactly one always fires. Every line is unit-tested at its
#: boundary in ``tests/unit/test_verdict.py``.
_RULES: tuple[_Rule, ...] = (
    _Rule(
        name="no-evidence",
        label="INCONCLUSIVE",
        reason="no test could be run on the evidence supplied",
        applies=lambda e: not e.has_applicable,
    ),
    _Rule(
        name="record-too-short",
        label="INCONCLUSIVE",
        reason=(
            "the track record is too short to judge, and nothing else objected — "
            "min_track_record_length says how much more data would settle it"
        ),
        applies=lambda e: e.flag_names == ("psr",),
    ),
    _Rule(
        name="flagged",
        label="LIKELY_LUCK",
        reason="at least one test that knows a search took place objected",
        applies=lambda e: bool(e.flag_names),
    ),
    _Rule(
        name="psr-only",
        label="INCONCLUSIVE",
        reason=(
            "the record clears the naive test, but no test that prices the search "
            "was available — pass the full trial matrix to get a verdict"
        ),
        applies=lambda e: not e.has_selection_aware,
    ),
    _Rule(
        name="clean",
        label="LIKELY_SKILL",
        reason="every applicable test passed, including at least one that prices the search",
        applies=lambda _: True,
    ),
)


def _narrative(rule: _Rule, evidence: _Evidence) -> str:
    """A plain-text account of the verdict and the evidence under it."""
    headline = rule.label.replace("_", " ")
    lines = [f"{headline} — {rule.reason}.", ""]
    for result in evidence.results:
        title = TEST_TITLES.get(result.name, result.name)
        lines.append(f"  [{result.status}] {title}")
        lines.append(f"      {result.interpretation}")
    if not evidence.results:
        lines.append("  No statistics were supplied.")
    return "\n".join(lines).rstrip()


def assess(
    *,
    psr: PSRResult | None = None,
    dsr: DSRResult | None = None,
    pbo: PBOResult | None = None,
    spa: RealityCheckResult | None = None,
) -> Verdict:
    """Combine whichever statistics were computed into one verdict.

    Takes results rather than raw returns on purpose: no statistics are computed
    here, so the rule table can be read and argued with without also auditing how
    the inputs were produced. Every argument is optional, and the verdict degrades
    honestly as evidence is withheld — a missing test is never treated as a
    passing one.

    Parameters
    ----------
    psr:
        From :func:`luckdetector.stats.probabilistic_sharpe_ratio`. Treated as a
        **precondition**, not as a fourth flag: on its own it can block a verdict
        of skill but can never certify one.
    dsr:
        From :func:`luckdetector.stats.deflated_sharpe_ratio` or
        :func:`~luckdetector.stats.deflated_sharpe_ratio_from_trials`.
    pbo:
        From :func:`luckdetector.stats.probability_of_backtest_overfitting`.
    spa:
        From :func:`luckdetector.stats.reality_check`. Marked ``NOT_APPLICABLE``
        when nothing in the family beat the benchmark, since there is then no
        maximum to price.

    Returns
    -------
    Verdict
        ``label`` is one of ``LIKELY_SKILL``, ``INCONCLUSIVE``, ``LIKELY_LUCK``;
        ``results`` holds one :class:`~luckdetector.types.TestResult` per statistic
        supplied, in :data:`TEST_ORDER`; ``narrative`` explains which rule fired
        and why.

    Notes
    -----
    ``LIKELY_LUCK`` needs only one objection, while ``LIKELY_SKILL`` needs a clean
    sweep *and* at least one selection-aware test. That asymmetry is deliberate —
    see the module docstring — and means the two labels are not equally strong
    claims.
    """
    supplied: list[TestResult] = []
    if psr is not None:
        supplied.append(_psr_evidence(psr))
    if dsr is not None:
        supplied.append(_dsr_evidence(dsr))
    if pbo is not None:
        supplied.append(_pbo_evidence(pbo))
    if spa is not None:
        supplied.append(_spa_evidence(spa))

    order: dict[str, int] = {name: i for i, name in enumerate(TEST_ORDER)}
    supplied.sort(key=lambda r: order[r.name])
    evidence = _Evidence(results=tuple(supplied))

    rule = next(r for r in _RULES if r.applies(evidence))
    return Verdict(
        label=rule.label,
        results=list(evidence.results),
        narrative=_narrative(rule, evidence),
    )
