# Project 1 — The Backtest Luck-Detector

**Blueprint v1.0**

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

```
backtest-luck-detector/
├── README.md                     # the shop window: problem, headline result, usage
├── LICENSE                       # MIT
├── pyproject.toml                # deps, build, ruff/mypy/pytest config
├── Makefile                      # make install / test / lint / demo
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml      # lint + typecheck + pytest on 3.10–3.12
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
│   │   ├── plots.py              # matplotlib figures
│   │   └── html.py               # Jinja2 → single-file HTML report
│   │
│   └── cli.py                    # typer app
│
├── tests/
│   ├── unit/                     # one file per module; null calibration lives here too
│   └── data/                     # small bundled fixtures
│
├── examples/
│   ├── 01_quickstart.ipynb
│   └── 02_mining_demo.py         # mine 500 strategies on SPY, then indict them
│
└── docs/
    ├── BLUEPRINT.md              # this file
    ├── METHODS.md                # the maths, with citations
    └── figures/
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

## 10. References

- Bailey, D. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* J. Risk.
- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* J. Portfolio Management.
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). *The Probability of Backtest Overfitting.* J. Computational Finance.
- White, H. (2000). *A Reality Check for Data Snooping.* Econometrica.
- Hansen, P. R. (2005). *A Test for Superior Predictive Ability.* J. Business & Economic Statistics.
- Harvey, C. & Liu, Y. (2015). *Backtesting.* J. Portfolio Management.
- Politis, D. & Romano, J. (1994). *The Stationary Bootstrap.* JASA.
- Politis, D. & White, H. (2004). *Automatic Block-Length Selection.* Econometric Reviews.
