# Project 1 — The Backtest Luck-Detector

**Blueprint v1.0 — as built, at v0.1.0**

Sections 0–9 are the plan, kept as written so the amendments in §6a stay legible.
Sections 10–12 are the working record that used to live in a separate
`HANDOFF.md`: the decisions already settled, the limitations already measured,
and the two environment traps that have each cost the project a day. That file
was for building; this one is for reading, and keeping two of them meant the
reasoning lived in the wrong place.

---

## 0. The one-sentence pitch

> Given a trading strategy's backtest, `luckdetector` estimates how much of the reported
> performance is explained by chance and selection bias rather than genuine edge — and
> returns a defensible verdict with the statistics to back it up.

## 1. The problem being solved

A backtest is a *maximum*. Nobody publishes the first strategy they tried; they publish the
best of several hundred they tried. The reported Sharpe ratio is therefore an order
statistic, not a sample mean, and the standard t-test is invalid.

Three distinct failure modes get conflated in practice, and this project separates them:

| Failure mode | Question it answers | Test used here |
|---|---|---|
| **Small-sample noise** | Is the track record long enough to distinguish this Sharpe from zero, given fat tails? | PSR, MinTRL |
| **Selection bias** (multiple testing) | Would the best of N random strategies look this good anyway? | DSR, Reality Check, SPA |
| **Overfitting** (in-sample tuning) | Does the parameter choice that looked best in-sample survive out-of-sample? | PBO via CSCV |

A tool that only does one of these is a toy. Doing all three, and reconciling them into one
verdict, is what makes this a portfolio project.

## 2. Design principles

1. **Every number is falsifiable.** Every statistic must be validated against a known null
   or a published reference value in the test suite. If it can't be validated, it doesn't ship.
2. **No network in tests.** Market data is cached; a small bundled CSV backs the test suite so
   CI is deterministic and offline.
3. **Seeded everywhere.** Every stochastic routine takes an explicit `rng` / `seed`. Two runs
   with the same seed produce byte-identical reports.
4. **Library first, CLI second.** All logic lives in importable, typed functions. The CLI is a
   thin shell. Nothing user-facing does statistics.
5. **Honest verdicts.** No black-box composite score. The verdict is a transparent rule table
   where each test can independently flag a problem, and the report says which one fired and why.

## 3. Repository layout

Everything below is built. As of v0.1.0 nothing in this tree is planned.

```
backtest-luck-detector/
├── README.md                     # the shop window: verdict and figures above the fold
├── LICENSE                       # MIT
├── pyproject.toml                # deps, build, ruff/mypy/pytest config
├── Makefile                      # make install / test / lint / demo / figures
├── .gitignore
├── .github/workflows/ci.yml      # lint + typecheck + pytest on 3.10–3.14
│
├── src/luckdetector/
│   ├── __init__.py               # public API re-exports + __version__
│   ├── types.py                  # ReturnSeries, TrialMatrix, TestResult, Verdict
│   ├── exceptions.py
│   │
│   ├── io/
│   │   ├── loaders.py            # CSV/parquet → ReturnSeries / TrialMatrix + validation
│   │   └── prices.py             # price download (yfinance/stooq), on-disk cache, fallback
│   │
│   ├── stats/
│   │   ├── moments.py            # sharpe, skew, kurtosis, annualisation, drawdown
│   │   ├── psr.py                # Probabilistic Sharpe Ratio, MinTRL
│   │   ├── dsr.py                # expected max Sharpe, Deflated Sharpe Ratio, N_eff
│   │   ├── bootstrap.py          # iid / circular-block / stationary bootstrap, permutation
│   │   ├── reality_check.py      # White's RC, Hansen's SPA
│   │   └── pbo.py                # CSCV → PBO, degradation, stochastic dominance
│   │
│   ├── mining/
│   │   ├── signals.py            # parameter grids: MA cross, momentum, RSI, breakout
│   │   └── engine.py             # vectorised backtester → TrialMatrix
│   │
│   ├── report/
│   │   ├── verdict.py            # rule table → Verdict + narrative
│   │   ├── analysis.py           # TrialMatrix → four statistics + Verdict
│   │   ├── plots.py              # matplotlib figures, OO API, never pyplot
│   │   ├── html.py               # Jinja2 → single-file HTML report
│   │   └── demo.py               # data resolution + the two-half demo
│   │
│   └── cli.py                    # typer app: version/summary/mine/report/demo
│
├── tests/
│   └── unit/                     # one file per module; null calibration lives here too
│
├── scripts/
│   └── make_readme_figures.py    # `make figures` — regenerates the two PNGs below
│
├── examples/
│   └── 01_quickstart.ipynb       # SPY end to end, shipped with its outputs
│
└── docs/
    ├── BLUEPRINT.md              # this file
    ├── METHODS.md                # the maths, with citations
    └── figures/
        ├── cumulative_return.png # committed: outputs/ is gitignored, so a
        └── deflation_hurdle.png  # README figure has nowhere else to live
```

## 4. Core data model

```python
@dataclass(frozen=True)
class ReturnSeries:
    values: np.ndarray            # 1-D periodic returns (not prices, not cumulative)
    periods_per_year: int         # 252 daily, 12 monthly, ...
    name: str = "strategy"

@dataclass(frozen=True)
class TrialMatrix:
    values: np.ndarray            # shape (n_trials, n_periods)
    periods_per_year: int
    labels: list[str]             # parameter descriptor per trial
    # invariant: every trial spans the same calendar periods — required by CSCV and RC

@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    p_value: float | None
    threshold: float
    passed: bool
    detail: dict[str, Any]        # everything needed to reproduce the plot

@dataclass(frozen=True)
class Verdict:
    label: Literal["LIKELY_SKILL", "INCONCLUSIVE", "LIKELY_LUCK"]
    results: list[TestResult]
    narrative: str
```

The two entry points map to the two things a user can hand us:

- **A single track record** (one return stream, plus a claimed number of trials) → PSR, DSR,
  MinTRL, Monte Carlo, haircut.
- **A full trial matrix** (all N strategies that were tried) → everything above *plus* PBO,
  Reality Check and SPA, which need the whole family. This is the strictly better input, and
  the mining engine exists so we can always produce one.

## 5. The statistics — specification

Full derivations live in `docs/METHODS.md`. Implementation contract summarised here.

### 5.1 Probabilistic Sharpe Ratio (Bailey & López de Prado 2012)

Probability that the true Sharpe exceeds a benchmark `SR*`, correcting for non-normality:

```
PSR(SR*) = Φ[ (ŜR − SR*) · √(n−1) / √(1 − γ₃·ŜR + ((γ₄−1)/4)·ŜR²) ]
```

`ŜR` is **per-period** (never annualised), `γ₃` skew, `γ₄` **raw** kurtosis (3 for normal).
Negative skew and excess kurtosis both *lower* PSR — the intuition being that a strategy that
makes pennies and loses dollars needs a longer record to prove itself.

**MinTRL** — track record length needed for significance at confidence `α`:

```
n* = 1 + [1 − γ₃·ŜR + ((γ₄−1)/4)·ŜR²] · (z_α / (ŜR − SR*))²
```

### 5.2 Deflated Sharpe Ratio (Bailey & López de Prado 2014)

DSR = PSR evaluated at a benchmark equal to the Sharpe you'd *expect* from the best of N
independent trials with zero true edge:

```
E[max ŜR] ≈ √V · [ (1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]
```

with `γ` the Euler–Mascheroni constant (0.5772…) and `V` the cross-sectional variance of the
trial Sharpes. `N` must be the *effective* number of independent trials: correlated trials
count for less. We estimate `N_eff` by clustering the trial correlation matrix
(hierarchical clustering on `√(0.5(1−ρ))`) and counting clusters.

**Guard rails:** if the user supplies a single track record and no `N`, we require them to
state it, and default to `N=1` with a loud warning in the report — because the whole point of
the project is that `N=1` is almost never true.

### 5.3 Bootstrap engine

- IID bootstrap (baseline).
- Circular block bootstrap, block length `b`.
- **Stationary bootstrap** (Politis & Romano 1994) — geometric block lengths, mean `1/q`.
  This is the one Reality Check and SPA require, since it preserves serial dependence while
  keeping the resample stationary.
- Automatic block-length selection (Politis & White 2004) as a helper.
- Permutation null: shuffle signal relative to returns to break the timing edge while
  preserving both marginal distributions.

### 5.4 Probability of Backtest Overfitting — CSCV (Bailey et al. 2017)

The headline diagnostic. Algorithm:

1. Split the `T` periods into `S` disjoint contiguous submatrices (default `S = 16`).
2. For each of the `C(S, S/2)` ways to choose `S/2` submatrices as **in-sample**, the
   complement is **out-of-sample**.
3. In each split: pick `n* = argmax` of the IS performance across trials. Find that same
   trial's **rank** among all trials OOS; normalise to `ω̄ ∈ (0,1)`.
4. Logit `λ = ln(ω̄ / (1 − ω̄))`.
5. **PBO = P(λ ≤ 0)** — the frequency with which the IS-best strategy lands in the bottom half
   out-of-sample.

Companion outputs from the same machinery, all of which go in the report:
- **Performance degradation:** OLS of OOS Sharpe on IS Sharpe. A negative slope is damning —
  it means better IS performance *predicts worse* OOS performance.
- **Probability of loss:** fraction of splits where the IS-best has OOS Sharpe < 0.
- **Stochastic dominance:** compare the OOS distribution of the IS-best against the OOS
  distribution of a randomly-chosen trial.

Interpretation baseline: under pure noise, PBO → 0.5. PBO > 0.5 means selection is actively
anti-predictive.

### 5.5 White's Reality Check & Hansen's SPA

Null: **no strategy in the family beats the benchmark.**
`f_k,t` = period-t performance of strategy `k` minus benchmark. Statistic:

```
V = max_k √T · f̄_k
```

Resample with the stationary bootstrap, recentre (`f̄*_k − f̄_k`), and take the p-value as the
fraction of bootstrap `V*` exceeding `V`.

**SPA** improves on this in two ways that matter: it studentises by `ω̂_k`, and it drops
strategies that are *very* bad from the recentring (threshold
`ĝ_k = f̄_k · 1{ f̄_k ≥ −√(ω̂²_k · 2·ln ln T / T) }`), so a pile of garbage strategies can no
longer make the good one look significant by dragging down the null. We report
lower / consistent / upper p-values.

### 5.6 Harvey–Liu haircut Sharpe (2015) — **cut**

Originally Phase 7: adjust an observed t-statistic under Bonferroni, Holm and
Benjamini–Hochberg–Yekutieli, then map back to `haircut SR = SR · (t_adj / t_obs)`.

**Dropped as redundant.** It answers the same question as the Deflated Sharpe Ratio —
*how much of this edge survives the number of things you tried* — using a cruder
instrument. DSR prices multiplicity through extreme value theory with a correlation-
discounted effective trial count; Bonferroni assumes independence and ignores the
dispersion of the trials entirely. Building both would mean doing one job twice, with
the weaker method arriving second.

The quotable line it was meant to produce ("your Sharpe of 1.8 is really 0.7") is
already available from DSR, and stated better: *the best of 12 independent noise
trials would have scored 0.31, and you scored 0.49.*

Kept in this document rather than deleted so the reasoning survives the decision. See
§6 for what replaced the phase.

## 6. Phase plan

Each phase ends with green tests and one commit. Nothing moves forward on red.

**Revised after Phase 5** — thirteen entries down to ten, and the plan now ends at
Phase 9 rather than Phase 12. See §6a for what was cut and why.

| # | Phase | Deliverable | Done when |
|---|---|---|---|
| 0 | Scaffolding | repo, packaging, CI, lint/type config | `make test` runs, CI green on first push |
| 1 | Core types + moments | `types.py`, `io/loaders.py`, `stats/moments.py` | Sharpe/skew/kurtosis match pandas & scipy on fixtures; loaders reject bad input |
| 2 | PSR / DSR / MinTRL | `stats/psr.py`, `stats/dsr.py` | Reproduces published worked examples to 3 dp; PSR monotone in n and in skew |
| 3 | Bootstrap | `stats/bootstrap.py` | Resamples preserve mean in expectation; block bootstrap preserves autocorrelation; seeded reproducibility |
| 4 | Mining engine | `mining/` | 500-strategy grid on SPY in < 10 s; vectorised output matches a naive loop |
| 5 | PBO / CSCV | `stats/pbo.py` | PBO ≈ 0.5 on synthetic noise; PBO < 0.1 on synthetic true-edge data |
| 6 | RC / SPA | `stats/reality_check.py` | p-values ~Uniform(0,1) under the null (KS test passes); SPA ≥ RC power on a planted edge |
| 7 | Verdict | `report/verdict.py` | Rule table unit-tested at every boundary; thresholds named as constants and defended in METHODS |
| 8 | Report + CLI | `report/plots.py`, `report/html.py`, `cli.py` | `luckdet demo` runs end-to-end from a clean install; single self-contained HTML; two figures, snapshot-tested |
| 9 | Docs + publish | README, METHODS, notebook | Repo is legible to a stranger in 60 seconds; v0.1.0 tagged and pushed |

**All ten are done.** v0.1.0 is the tag on Phase 9. Phase 9 itself: the two
figures generated from real SPY prices and committed to `docs/figures/`, the
README restructured so the verdict and both figures are above the fold, the
quickstart notebook in `examples/`, the working handoff folded into §10–§12 of
this document and deleted, and section 3 of the generated report fixed after it
was read on a family that passes (`METHODS.md` §12.2).

Power curves for each statistic are folded into `tests/unit/` alongside the
calibration tests that already live there, marked `slow`. They are not a phase.

## 6a. Scope cuts (amendment, after Phase 5)

Two numbered phases were removed, one was merged, and the animation layer was
dropped entirely. The project had reached the point where
additional *statistics* added nothing and additional *legibility* added a lot: a
reader spends about sixty seconds here, and everything that does not survive that
minute was work done for its own sake. Recording the reasoning so it is not
relitigated.

**Cut — the manim animation layer.** Four scenes, a toolchain of ffmpeg, Pango and
LaTeX, rendering necessarily excluded from CI so it would rot unnoticed, to produce a
GIF in a README. The stated goal — "README opens with the demo result and a figure" —
is satisfied by two static matplotlib figures at a small fraction of the cost. This
was the single largest item in the remaining plan and the one with the weakest link
to anything a reader would check.

**Cut — Phase 7, the Harvey–Liu haircut.** Redundant with DSR; see §5.6.

**Merged — the standalone validation suite (old Phase 11).** Null calibration was
never actually deferred to a final phase; it has been written inline from Phase 1
onward, and each existing test is checked against simulation rather than against its
own algebra: the Sharpe standard error against the spread of 2,000 simulated
estimates, the Gumbel expected maximum against brute force for N ∈ {10, 50, 200,
1000}, PBO against a 30-dataset ensemble null, and Phase 6's KS uniformity criterion.
Rebuilding that as a separate tree would be filing, not work. The one genuinely new
piece — documented power curves — becomes a handful of `slow`-marked tests in the
existing unit files.

**Merged — CLI into the report phase.** `luckdet mine` already exists; `demo` is
mostly wiring, and it is the same work as making the report render.

**Kept deliberately, despite being the weakest link:** the verdict layer. A tool that
returns five numbers and no answer is unfinished. But its thresholds are invented
rather than derived, and that should be said out loud in METHODS before a reader says
it first. `PBO_THRESHOLD = 0.2` already sets the precedent: a named constant with an
explicit note that it is a judgement call, not a convention from the literature.

## 7. The demo that sells the project

`luckdet demo` runs this narrative end-to-end, and its output becomes the README hero:

1. Download 15 years of SPY daily data.
2. Brute-force a grid of ~500 moving-average-crossover variants.
3. Report the best one: "Sharpe 1.9, 340% cumulative return." It looks fantastic.
4. Then run the detector on it and show:
   - Deflated Sharpe Ratio ≈ 0.2 → cannot reject that this is the best of 500 coin flips
   - PBO ≈ 0.6 → the IS-best strategy underperforms the median OOS more often than not
   - Reality Check p ≈ 0.7 → the family as a whole beats buy-and-hold no better than chance
5. **Verdict: LIKELY LUCK.**
6. Then repeat on a synthetic strategy with a *real* planted edge and show the tool correctly
   returns **LIKELY SKILL** — proving it isn't just a machine that says "no" to everything.

That last step is the difference between a project that looks rigorous and one that is.

## 8. Non-goals (stated so scope stays fixed)

- Not a backtesting framework. The miner exists only to generate trial matrices.
- No transaction-cost modelling beyond a simple flat bps charge.
- No live trading, no broker integration, no intraday data.
- No ML strategy generation — the point is statistics, not alpha.

## 9. Definition of done

- `pip install -e ".[dev]"` then `make test` passes from a clean clone on 3.10–3.14.
- CI green, coverage ≥ 90% on `src/luckdetector/stats/`.
- Every statistical routine has a null-calibration test in `tests/unit/`, checked
  against simulation rather than against a re-transcription of its own formula.
- README opens with the demo result and a figure, not with installation instructions.
- `docs/METHODS.md` cites every paper implemented, and states plainly where a
  threshold is a judgement call rather than a convention.

## 10. Decisions already settled — do not re-open without a measurement

These were argued out during Phases 1–8 and several of them cost real time to
find. They lived in a working `HANDOFF.md` until v0.1.0; that file has been
deleted and the durable parts are here, because this is where the project's
reasoning already lives. The point of writing them down is that they stop being
re-litigated — but each one is a claim about the code, so the way to overturn one
is to show the measurement, not to prefer a different intuition.

**Units.** Every public function takes and returns **annualised** Sharpe ratios
and converts internally. PSR/DSR maths is per-period. Mixing them inflates
significance by ~16× on daily data and is the classic bug in this literature.

**Synthetic data is for tests, never for headlines.** Tests must not touch the
network. Any *reported result* uses real prices. `synthetic_prices()` is honestly
named and never dressed up as a ticker. **No market data is committed to this
repository** — which is a constraint on the README, the notebook and the report
as much as on the test suite, and §12.1 of `METHODS.md` records how each one
lives within it.

**Keep every trial, never just the winner.** `mine()` returns the whole family.
PBO and Reality Check are impossible without it.

**Validate against reality, not against the paper's algebra.** Every statistic is
checked against simulation or a known null, never against a re-transcription of
its own formula. The Sharpe standard error matches the empirical spread of 2,000
simulated estimates within 5%; the Gumbel expected-maximum matches brute force
within 6%; the CSCV subsample scorer matches naive concatenate-and-score to
1.7e-16.

**Seed everything.** Every stochastic routine takes an explicit `rng`/`seed`.
Same seed, identical output.

**Guard degenerate series.** `moments.is_effectively_constant` exists because
`np.std` of identical values returns ~2e-19, not 0.0. Any new routine that
divides by a dispersion measure must gate on it.

**Test fixtures draw from one seeded stream per batch.** Seeding each row with
consecutive seeds gives measurably under-dispersed extremes. Use `make_trials` /
`make_exact_returns` in `tests/conftest.py`.

**Reduce resampling to a matrix product.** The Sharpe of any union of blocks
depends only on per-block count, sum and sum-of-squares, so 12,870 CSCV splits
collapse into two matmuls against an `(n_splits, S)` membership matrix — the full
SPY grid in 0.04s. The same trick carries the Reality Check's 1,000 replicates.
Do not write a Python loop over resamples.

**Centre before using the** `E[x²] − E[x]²` **identity.** The raw form loses
precision to catastrophic cancellation; centring on the full-sample mean shrinks
the subtracted term by a factor of the subsample length and makes it harmless.

**A negative in-sample/out-of-sample slope means nothing.** The two halves of a
split *partition a fixed total*, so `mean_OOS ≈ mean_total − mean_IS` and the
slope is pinned near −1 before any selection happens — measured at **−0.999** on
a fixed trial with no selection at all. Noise gives −0.82; a planted genuine edge
gives −0.69. It does not discriminate, and
`test_fixed_trial_slope_is_minus_one` exists to stop it being "fixed" into
significance. `probability_of_loss` and `dominance_fraction` are the companions
that do work.

**A single PBO estimate is noisy.** Across 30 independent zero-edge datasets
(N = 50, T = 1260, S = 16): mean 0.505, **sd 0.153**, range 0.23–0.80. The 12,870
splits are heavily dependent. Any test of a null value must average over
datasets, never assert on one draw, or it passes and fails on the seed.

**Reuse one set of bootstrap indices across every strategy** in the Reality
Check. Resampling each independently destroys the cross-sectional dependence that
makes `max_k` meaningful. Note the direction, which the Phase 6 brief had
backwards: a mined family is mostly *positively* correlated, so independent
resampling prices 157 near-duplicates as 157 separate bets and returns p-values
that are too **large**. It costs power rather than manufacturing significance.

## 11. Known limitations, measured

Every item here was measured rather than suspected, and each ships in the caveats
of every generated report. None is a defect to be quietly fixed before release;
three of them are properties of the method that a reader is entitled to know.

- **CSCV has no purge or embargo.** With S = 16 and T = 4,173 each block is ~261
  days, so a 250-day-lookback rule is contaminated near each seam. This makes the
  winner look *more* persistent than it is, so the true PBO is likely worse than
  0.84, not better. A purged variant is the obvious extension, and v0.1.0 ships
  with the gap open and documented rather than with statistical work started in a
  release phase.
- **SPA does not protect against pruning the family.** It defends against
  *padding* with obvious garbage; *deleting* the merely-bad variants after seeing
  the results defeats both tests equally. On the real SPY family against zero,
  pruning 157 → best 16 moves RC 0.242 → 0.068 and SPA 0.411 → 0.065, with
  nothing recentred at any stage. The same honest-book-keeping dependence DSR has
  and PBO does not.
- **"SPA ≥ RC power" is not a universal domination.** It holds in the case SPA
  was designed for — a real edge buried among genuinely bad strategies — and the
  test asserts it there. On SPY against zero it fails (RC 0.24, SPA 0.43) because
  none of the 157 falls below Hansen's cutoff, so only the studentisation is in
  play. Do not quote it as a general property.
- **The verdict layer has poor power on the skill side.** The SPY *luck* verdict
  is robust to any defensible threshold: DSR flags above 0.77, PBO below 0.84,
  and one flag suffices. The skill side is weak. Across 25 independent datasets,
  a genuine annualised Sharpe of 2.0 planted in 10 of 50 variants over five years
  is called LIKELY_SKILL only **20%** of the time; it takes 5 of 50 at Sharpe 3.0
  over ten years to reach 100%. **DSR is the binding constraint at every effect
  size** — PBO returns 0.000 and SPA rejects throughout. So LIKELY_LUCK is weak
  evidence of absence. `test_detection_rate_at_a_realistic_edge_is_poor` pins the
  figure so it cannot drift or be quietly threshold-tuned away. Do not "fix" this
  by lowering `DSR_THRESHOLD`: the conservatism is the instrument working, and
  the honest response is the caveat, not a friendlier bar.
  - Not yet examined: whether the effective-trial count is the real culprit. DSR's
    hurdle rises with the *dispersion* of trial Sharpes, so planting 5 good
    variants scores better than planting 10 of the same quality. That is arguably
    correct, but nobody has looked.
- **Two small gaps in the report layer.** `luckdet report` and `luckdet demo`
  write the *real* half to HTML only — the planted-edge control prints to the
  terminal and is not rendered. And the uncovered lines in `cli.py` are all in
  `mine`, which predates the report phase: its download branch and error handler
  have no test because nothing injects a downloader through that command.
  `report` covers the equivalent path by stubbing `cli.load_prices`; `mine` could
  do the same.

## 12. Environment, and two traps worth knowing about

```bash
cd ~/Documents/"Project 1"           # the quotes matter
source .venv/bin/activate            # Python 3.14
make check                           # ruff + mypy + pytest
```

Python 3.10 is the floor (`zip(strict=True)`, runtime `X | None` in typer
signatures). macOS system Python is 3.9 and will not work. Editable installs need
pip ≥ 21.3.

`outputs/` holds the cached SPY CSV and the saved run summaries. It is
gitignored: real prices stay local and stay available for analysis without
re-downloading, while nothing generated there reaches a reader who clones the
repo. That is why the two README figures are committed to `docs/figures/` and why
the reported numbers live in `README.md` and `docs/METHODS.md`. `luckdet demo`
searches `outputs/` before the user cache, so running it from the repo root picks
up that CSV and needs no network.

**A green local run is necessary and not sufficient — the matrix is the gate.**
CI runs 3.10 through 3.14 and dependencies resolve to different versions per
interpreter, so a single interpreter is structurally incapable of seeing part of
the matrix. This has bitten the project twice. Phase 5 shipped an annotation bug
that passed on numpy 2.2 and failed on 3.14. Phase 8 shipped a `mypy --strict`
failure that passed on the 3.10 job and failed on the other four, because
matplotlib 3.11 dropped Python 3.10 — 3.10 resolves matplotlib 3.10.9 with a
loose signature while 3.11+ resolve 3.11.1 with a strict one. numpy splits the
same way: numpy 2.3 requires 3.11+, so a 3.10 environment never typechecks
against the newer stubs. `METHODS.md` §11.5 has the full account. The rule it
implies: **where a dependency's stubs vary across the supported range, prefer the
form that lets the dependency's own types do the checking over the form that
asserts a type of your own.** And run `make check` locally before pushing, then
wait for the matrix before tagging.

**Agent sessions: enable file deletion for the folder before any git command.**
A sandbox mount can start without delete permission, and git cannot clear
`.git/index.lock` without it — so every git write fails, and worse, even a plain
`git status` creates a lock it then cannot remove, blocking the user's own
commands until they delete it by hand. This cost real time in Phase 5. If a
delete returns "Operation not permitted", request delete access rather than
routing commits back to the user; `git --no-optional-locks status` is the safe
read-only inspection in the meantime. And do the committing — the user should
only ever need to run `git push`.

## 13. References

- Bailey, D. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* J. Risk.
- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* J. Portfolio Management.
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). *The Probability of Backtest Overfitting.* J. Computational Finance.
- White, H. (2000). *A Reality Check for Data Snooping.* Econometrica.
- Hansen, P. R. (2005). *A Test for Superior Predictive Ability.* J. Business & Economic Statistics.
- Harvey, C. & Liu, Y. (2015). *Backtesting.* J. Portfolio Management.
- Politis, D. & Romano, J. (1994). *The Stationary Bootstrap.* JASA.
- Politis, D. & White, H. (2004). *Automatic Block-Length Selection.* Econometric Reviews.
