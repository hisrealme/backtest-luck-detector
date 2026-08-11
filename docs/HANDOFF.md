# Current state and next steps

Written at the end of Phase 5, extended at the end of Phases 6, 7 and 8. Read this
before touching the code — it records the decisions that have already been argued out,
so they don't get re-litigated, and it specifies the next phase precisely enough to
implement from. **The next phase is 9, docs and release; see `BLUEPRINT.md`.**
Sections 4, 5 and 6 are the closed records of Phases 6, 7 and 8, kept because all three
specs turned out to contain errors worth remembering.

---

## 1. Where the project is

| | |
|---|---|
| Phases complete | 0–8 (scaffolding, core model, PSR/DSR, bootstrap, mining, PBO, RC/SPA, verdict, report/CLI) |
| Commits | 25, all authored `hisrealme <315065552+hisrealme@users.noreply.github.com>` (check with `git rev-list --count HEAD`; this line has gone stale twice) |
| Tests | 434 passing, 1 skipped (1 network test deselected), **97%** coverage |
| Gates | `ruff` clean, `mypy --strict` clean |
| CI | GitHub Actions, Python 3.10 / 3.11 / 3.12 / 3.13 / 3.14 — **green on the new repo** |
| Repo | `github.com/hisrealme/backtest-luck-detector`, public |

### Modules that exist

```
src/luckdetector/
├── types.py            ReturnSeries, TrialMatrix, TestResult, Verdict  (frozen, validating)
│                       FloatArray / IntArray / BoolArray dtype aliases
├── exceptions.py       LuckDetectorError hierarchy
├── cli.py              typer app: `version`, `summary`, `mine`
├── io/
│   ├── loaders.py      CSV/parquet → ReturnSeries / TrialMatrix
│   └── prices.py       yfinance + on-disk cache, injectable downloader
├── mining/
│   ├── signals.py      4 families, 157-variant default grid
│   └── engine.py       vectorised backtester, synthetic_prices
└── stats/
    ├── moments.py      Sharpe, skew, kurtosis, drawdown, is_effectively_constant
    ├── psr.py          Probabilistic Sharpe Ratio, MinTRL
    ├── dsr.py          Deflated Sharpe Ratio, expected_max_sharpe, effective trials
    ├── bootstrap.py    iid / circular / stationary, Politis–White block length
    ├── pbo.py          PBO via CSCV, degradation, dominance   ← Phase 5
    └── reality_check.py  White's RC + Hansen's SPA, 3 variants  ← Phase 6
└── report/
    ├── verdict.py      rule table → Verdict + narrative       ← Phase 7
    ├── analysis.py     TrialMatrix → 4 statistics + Verdict   ← Phase 8
    ├── plots.py        2 figures, OO matplotlib, no pyplot    ← Phase 8
    ├── html.py         one self-contained file, Jinja2        ← Phase 8
    └── demo.py         data resolution + the two-half demo    ← Phase 8
```

`docs/METHODS.md` now exists and covers phases 0–8. **Extend it in the same pass as
each new phase**, not afterwards — the reasoning is only cheap to write down while
it is fresh.

---

## 2. Results already established

Run on **SPY, 2010-01-04 to 2026-08-07**, 4,174 trading days / 4,173 returns,
adjusted closes. Reproduce with `luckdet mine SPY --start 2010-01-01`.

```
Buy-and-hold           Sharpe 0.867   +814.3%   maxDD -33.7%
Winner MA(80,250)      Sharpe 0.491   +205.6%   maxDD -40.3%   naive PSR 0.9764
157 trials -> 12 effectively independent
Expected max Sharpe of noise: 0.309
Deflated Sharpe Ratio: 0.7692   -> LIKELY LUCK
```

**Phase 5 added a far sharper verdict.** PBO via CSCV, S = 16, 12,870 splits:

```
PBO                              0.8396   (noise baseline 0.50)
Probability of OOS loss          0.4407
Stochastic dominance fraction    0.973
Median OOS Sharpe, IS winner     0.032
Median OOS Sharpe, random trial  0.189
```

Selection is not merely uninformative here, it is **anti-predictive**: picking the
in-sample winner is worse than picking a variant at random. MA(80,250) — the
full-sample winner that would have been reported as *the* result — is the
in-sample pick in only **1.6%** of splits. Stable across S ∈ {8,12,16,20}
(0.80–0.84) and across costs 0–5bp.

**The finding that still matters most: 0 of 157 strategies beat buy-and-hold.** The
winner's returns minus buy-and-hold's have an annualised Sharpe of **-0.416**.
Twenty-six variants (17%) had a negative Sharpe outright.

Caveat kept in the README and METHODS: 2010–2026 was a historic bull market, so
trend rules that go flat or short necessarily give up ground. This is evidence
about this family over this window, not a verdict on trend following.

---

## 3. Decisions already made — do not re-open without reason

**Units.** Every public function takes and returns **annualised** Sharpe ratios and
converts internally. PSR/DSR maths is per-period. Mixing them inflates significance by
~16x on daily data and is the classic bug in this literature.

**Synthetic data is for tests, never for headlines.** Tests must not touch the
network. Any *reported result* uses real prices. `synthetic_prices()` is honestly
named and never dressed up as a ticker. No market data is committed to the repo.

**Keep every trial, never just the winner.** `mine()` returns the whole family. PBO
and Reality Check are impossible without it.

**Validate against reality, not against the paper's algebra.** Every statistic is
checked against simulation or a known null, not against a re-transcription of its own
formula. Examples: the Sharpe standard error matches the empirical spread of 2,000
simulated estimates within 5%; the Gumbel expected-maximum matches brute-force
simulation within 6%; the CSCV subsample scorer matches naive concatenate-and-score
to 1.7e-16.

**Seed everything.** Every stochastic routine takes an explicit `rng`/`seed`. Same
seed, identical output.

**Guard degenerate series.** `moments.is_effectively_constant` exists because
`np.std` of identical values returns ~2e-19, not 0.0. Any new routine that divides by
a dispersion measure must gate on it.

**Test fixtures draw from one seeded stream per batch.** Seeding each row with
consecutive seeds gives measurably under-dispersed extremes. Use `make_trials` /
`make_exact_returns` in `tests/conftest.py`.

### New in Phase 5 — three that cost real time to find

**Reduce resampling to a matrix product.** The Sharpe of any union of blocks depends
only on per-block count, sum and sum-of-squares, so 12,870 CSCV splits collapse into
two matmuls against a `(n_splits, S)` membership matrix. Full SPY grid: **0.04s**.
The same trick applies to Phase 6 (see §4). Do not write a Python loop over
resamples.

**Centre before using the** `E[x²] − E[x]²` **identity.** Raw form loses precision to
catastrophic cancellation; centring on the full-sample mean shrinks the subtracted
term by a factor of the subsample length and makes it harmless.

**A negative in-sample/out-of-sample slope means nothing.** The Phase 5 spec called
it "damning". It is not. The two halves of a split *partition a fixed total*, so
`mean_OOS ≈ mean_total − mean_IS` and the slope is pinned near −1 before any
selection happens — measured at **−0.999** on a fixed trial with no selection at all.
Noise gives −0.82, a planted genuine edge gives −0.69: it does not discriminate.
`test_fixed_trial_slope_is_minus_one` locks this in. Do not "fix" it into
significance. `probability_of_loss` and `dominance_fraction` are the companions that
actually work.

**A single PBO estimate is noisy.** Across 30 independent zero-edge datasets
(N = 50, T = 1260, S = 16): mean 0.505, **sd 0.153**, range 0.23–0.80. The 12,870
splits are heavily dependent. Any test of a null value must average over datasets,
never assert on one draw — otherwise it passes or fails on the seed.

---

## 4. Phase 6 — Reality Check & SPA — **DONE**

White (2000), *A Reality Check for Data Snooping*, Econometrica 68(5).
Hansen (2005), *A Test for Superior Predictive Ability*, JBES 23(4).

> **Shipped in `stats/reality_check.py`, 47 tests, 100% coverage on the module.**
> The spec below is left as written so the corrections stay legible. **Two of its
> claims had the sign backwards**, both flagged inline and both now asserted in
> tests:
>
> * "adding garbage will *lower* RC's p-value" — it raises it, monotonically;
> * "independent resampling gives p-values far too *small*" — too large, for the
>   positively correlated families mining produces.
>
> **SPY result.** Against buy-and-hold every p-value is exactly 1.0000 (all 157
> variants have a negative mean differential, so `V_SPA = 0`). Against zero,
> RC 0.2428 and SPA 0.4306. Full run in `outputs/spy_rc_spa.txt`; the reasoning
> is in `METHODS.md` §8, the numbers in §9.
>
> **Two limitations found that are worth carrying forward** — see §8.

**Null: no strategy in the family beats the benchmark.**

### Benchmark choice — decided

The headline SPY result uses **buy-and-hold** as the benchmark, not zero. Given that
0 of 157 variants beat it, that is the test a sceptic actually cares about, and it is
the honest one. `MiningResult.buy_and_hold` already carries the series.

Implement `benchmark` as a parameter so zero remains available (it is the softer,
more publishable test and useful for comparison), but the reported number is against
buy-and-hold.

### Reality Check

With `f[k, t]` = strategy `k`'s period-`t` return minus the benchmark's:

```
V = max_k  sqrt(T) * mean(f_k)
```

Resample with the **stationary bootstrap**, recentre, and take

```
V*_b = max_k  sqrt(T) * (mean(f*_k,b) - mean(f_k))
p_RC = fraction of b with V*_b >= V
```

### SPA

Two improvements over RC, both of which matter:

1. **Studentise** by `ω̂_k`, the standard deviation of `sqrt(T) * mean(f_k)`.
   Estimate it from the bootstrap replicates themselves — the sd across `b` of
   `sqrt(T) * (mean(f*_k,b) − mean(f_k))` — so it inherits the same dependence
   structure. Test statistic `V_SPA = max(0, max_k sqrt(T)*mean(f_k)/ω̂_k)`.

2. **Drop hopeless strategies from the recentring**, so a pile of garbage can no
   longer make the good one look significant by dragging the null down. Three
   variants, reported together:

   | | recentring threshold `g_k` |
   |---|---|
   | lower | `mean(f_k) · 1{mean(f_k) ≥ 0}` |
   | consistent | `mean(f_k) · 1{mean(f_k) ≥ −sqrt(ω̂²_k · 2 ln ln T / T)}` |
   | upper | `mean(f_k)` (equivalent to RC's recentring) |

   `p_lower ≤ p_consistent ≤ p_upper` must hold by construction — assert it.

### Implementation notes

**Reuse one set of bootstrap indices across all `k`.** Resampling each strategy
independently destroys the cross-sectional dependence that makes `max_k` meaningful
and will silently produce p-values that are far too small. This is the single easiest
way to get RC/SPA wrong.
> **Correction.** The instruction is right, the direction is not. A mined family is
> mostly *positively* correlated, so independent resampling prices 157 near-duplicate
> rules as 157 separate bets, draws the null maximum too high, and returns p-values
> that are too **large** — it costs power rather than manufacturing significance.
> Measured on 20 exact duplicates, where the family is one bet by construction:
> shared indices give p = 0.002, matching the single-strategy answer exactly;
> independent resampling gives 0.020. See `test_independent_resampling_would_over_penalise`.

**Vectorise via a count matrix, as in Phase 5.** For replicate `b` with index vector
`idx_b`, the resampled mean for strategy `k` is `f[k] @ bincount(idx_b) / T`. So
build `C` of shape `(B, T)` once and compute all resampled means as `f @ C.T / T` →
`(K, B)` in a single matmul. With K=157, T=4173, B=1000 that is ~0.7 GFLOP and a
`(1000, 4173)` float64 matrix (33 MB) — seconds, not minutes. Materialising
`f[:, idx]` directly would be 5 GB.

**Block length** comes from `bootstrap.optimal_block_length` (Politis–White), already
implemented and already guarded against the degenerate-series trap. Do not hand-pick
it.

### Acceptance criteria

- p-values ≈ Uniform(0,1) under the null — KS test passes on ≥ 200 simulated
  null datasets. This is the size calibration and it is the point of the phase.
- SPA has **≥ RC power** on data with a planted edge buried among many bad strategies
  (that is the case SPA was designed for; construct it explicitly).
- `p_lower ≤ p_consistent ≤ p_upper` holds on every dataset tested.
- Adding pure-garbage strategies to a family must **not** lower the SPA p-value,
  but *will* lower RC's — demonstrate both, that contrast is the whole argument
  for SPA.
  > **Correction.** Garbage *raises* RC's p-value, and cannot do anything else: a
  > maximum over a superset is weakly larger for every replicate while the observed
  > statistic is unchanged. That is Hansen's actual complaint about RC and the
  > contrast is sharper for it. Measured with the resampling indices pinned, 10
  > strategies then + 100 garbage: RC 0.011 → 0.140, SPA consistent 0.013 → 0.013
  > (bit-identical), SPA upper 0.014 → 0.171. `upper` is RC's recentring
  > studentised, so its degrading alongside RC confirms the mechanism.
- Runs 157 × 4,173 with B = 1,000 in reasonable time. Vectorise.
  > **Met.** Both SPY benchmarks, B = 1,000, in 0.17s total.

### Files

`src/luckdetector/stats/reality_check.py`, `tests/unit/test_reality_check.py`,
export from `stats/__init__.py`, tick Phase 6 in `README.md`, add the RC/SPA section
to `docs/METHODS.md`.

### Prediction, written down so it can be checked rather than rationalised

Against buy-and-hold I expect **RC and SPA both to fail to reject the null** — p-values
well above 0.1, probably above 0.5 — because no variant beats the benchmark at all.
Against zero I expect RC to be marginal and SPA more conservative. If RC returns a
*significant* p-value against buy-and-hold, suspect the recentring or a shared-index
bug before believing it.

Note the Phase 5 lesson about worthless predictions: "the slope will be negative" was
unfalsifiable. The prediction above is falsifiable — a p-value below 0.1 against
buy-and-hold would genuinely surprise.

> **Outcome: the headline prediction held in its strongest form, the secondary one
> did not.** Against buy-and-hold, p = 1.0000 exactly — "well above 0.5" was right,
> and for the stated reason. Against zero, RC came in at 0.2428, which is not
> "marginal" by any reading; "SPA more conservative" was right (0.4306 > 0.2428) but
> for a reason the prediction did not anticipate — nothing is bad enough relative to
> zero to be recentred, so SPA's second improvement never engages and only the
> studentisation is left. Recorded rather than rounded into a hit.

---

## 5. Phase 7 — verdict aggregation — **DONE**

Shipped in `report/verdict.py`, 35 tests, 100% coverage on the module. Three design
questions were settled before implementation and should not be re-opened casually:

**Worst flag wins.** Five rules, evaluated in order, first match wins:
`no-evidence` → INCONCLUSIVE; `record-too-short` (PSR is the *only* flag) →
INCONCLUSIVE; `flagged` → LIKELY_LUCK; `psr-only` → INCONCLUSIVE; `clean` →
LIKELY_SKILL. No composite score, deliberately: a weighted average would hide which
evidence drove the answer.

**PSR is a precondition, not a fourth flag.** A lone PSR flag means "we cannot tell
yet", and PSR alone can never certify skill — it is the naive test this package
exists to debunk, and the SPY winner clears it at 0.9764 while failing everything
that knows a search took place. LIKELY_SKILL requires at least one of DSR, PBO or
SPA to have run and passed.

**`TestResult` has three states.** `NOT_APPLICABLE` exists because an SPA p-value of
1.0 obtained because *nothing beat the benchmark* is an empty comparison, not a
rejection. It neither flags nor counts toward the evidence LIKELY_SKILL needs.

`PSR_THRESHOLD` and `DSR_THRESHOLD` were named alongside the existing
`PBO_THRESHOLD` and `SIGNIFICANCE_LEVEL`. METHODS §9 states plainly that three of the
four are conventional and the combination logic is invented.

**The finding that matters most from this phase is in §8 open items: detection power
is poor.** A genuine Sharpe of 2.0 is called LIKELY_SKILL 20% of the time. Read that
before designing anything in Phase 8 that claims the tool "detects" edge.

---

## 6. Phase 8 — Report + CLI — **DONE**

> **Shipped in `report/analysis.py`, `report/plots.py`, `report/html.py`,
> `report/demo.py` and two CLI commands. 87 new tests, 100% coverage on all four
> modules bar one branch in `plots.py`.** The spec below is left as written so the
> corrections stay legible. **One of its three "already taken" decisions was wrong**,
> and one further error was introduced while implementing it:
>
> * **Figure 2 argued the opposite of the verdict.** The brief specified the trial
>   Sharpes marked against the *expected maximum of noise* (0.309) and the winner
>   (0.491), captioned "the winner sits inside the noise distribution". The winner is
>   the argmax of the family by construction, so it is at the 100th percentile and
>   can never sit inside it; and it is *above* 0.309, as are 43 of the 157 variants,
>   so the drawn picture reads as a pass while DSR 0.7692 says luck. The bar DSR
>   actually applies allows for the 0.2472 standard error on the winner's own Sharpe:
>   **0.7152**, which the winner misses by 0.2246 and which no variant reaches.
>   Added `stats.sharpe_required_for_dsr` (an exact inversion, not a new judgement)
>   and drew that instead, shading the area that *is* the DSR. **Measured on twelve
>   independently mined families — four seeds × three lengths — the geometry holds on
>   all twelve**, so this is structural to any correlated family rather than a quirk
>   of SPY. `test_the_misleading_geometry_is_structural_not_a_property_of_spy`.
> * **The evidence table rendered the inapplicable SPA row as `p = 0.0000` vs
>   `0.0500`.** `TestResult.statistic` holds *V_SPA* = 0 in that case, not a p-value,
>   so formatting the row like the others turned the one cell with nothing to weigh
>   into the most emphatic rejection in the document — the exact misreading the third
>   state exists to prevent. Inapplicable rows now render an em dash in both cells.
>   `test_renders_no_comparison_at_all_in_the_table`.
>
> The other two decisions — data resolution and report scope — were implemented as
> written. One wrinkle in the first: `cache_path` keys on `(symbol, start, end)` and
> `end` defaults to today, so an exact-key lookup misses a file written yesterday.
> Resolution matches on symbol and start and takes the latest end.
>
> **SPY result.** `luckdet demo` reproduces every published figure exactly —
> MA(80,250) at 0.491, DSR 0.7692, PBO 0.8396, 0 of 157 beating buy-and-hold,
> **LIKELY_LUCK** — then returns **LIKELY_SKILL** on the planted-edge control, in
> 1.3s total. Reasoning in `METHODS.md` §11.

**Deliverable:** `report/plots.py`, `report/html.py`, and the CLI commands that drive
them. Done when `luckdet demo` runs end to end from a clean install, produces one
self-contained HTML file, and the two figures are snapshot-tested.

`report/verdict.py` deliberately takes *computed results* rather than raw returns, so
that no statistics live in user-facing code. Phase 8 is the wiring that was kept out
of Phase 7 on purpose.

### Decisions already taken — implement these, don't re-litigate

**Demo data: cache, then download, then refuse.** `luckdet demo` uses the `outputs/`
cache if present, downloads if not, and if neither is available **fails with a message
pointing at `luckdet demo --offline`**. The offline path runs on `synthetic_prices`
and labels every figure and every number `SYNTHETIC`. This keeps "synthetic data is
for tests, never for headlines" (§3) intact while still working without a network. Do
not quietly fall back to synthetic — the failure has to be loud, and the offline
output has to be unmistakable.

**Two figures.**

1. *Cumulative return, winner vs buy-and-hold.* The "+205.6% looks superb until you
   see +814.3%" picture. One line each, drawdown shading optional.
2. *Distribution of all 157 trial Sharpes*, with the expected-max-of-noise hurdle
   (0.309) and the winner (0.491) marked on it. This is the whole thesis in one
   image: the winner sits inside the noise distribution.
   > **Correction — this is the decision that was wrong, and both halves of it are.**
   > The winner is the maximum of the plotted distribution by construction, so it is
   > never "inside" it; and 0.491 > 0.309, along with 42 other variants, so marking
   > those two lines shows the winner *clearing* the hurdle at the moment DSR calls
   > it luck. The expected maximum is a point estimate; the winner's Sharpe is an
   > estimate with a 0.2472 standard error, and 0.95 confidence needs
   > 0.3086 + 1.645 × 0.2472 = **0.7152**. The figure now draws the null sampling
   > distribution with the area below the winner shaded — that area *is* the DSR,
   > 0.7692 — plus the 0.7152 bar, which nothing in the grid reaches. The histogram
   > stays, since the dispersion of the family is what sets the hurdle, but it no
   > longer carries a claim it cannot support. See `METHODS.md` §11.2.

Snapshot-test them on **structure, not pixels** — number of axes, series lengths,
labels, tick counts. Image hashes break on every matplotlib and font change and will
rot the suite; there is no value in asserting on anti-aliasing.

**HTML: a full report, single file.** Verdict banner; the four-row evidence table with
statistic, threshold and status; both figures inline as base64 so the file is
genuinely self-contained; each test's `interpretation` paragraph; and the caveats —
bull-market window, the CSCV purging gap (§8), and the low detection power (§8). Jinja2
is already a dependency. No external CSS, no CDN, no JavaScript.

### Notes that will save time

**The verdict layer already produces the report's spine.** `Verdict.results` is
ordered by `TEST_ORDER`, each `TestResult` carries `name`, `statistic`, `threshold`,
`status`, `interpretation` and a `detail` dict, and `TEST_TITLES` in
`report/verdict.py` maps short keys to human titles. The template should render that
list rather than re-deriving anything.

**Render `NOT_APPLICABLE` as its own thing.** Not as a pass, not as a failure, not
greyed out into invisibility. On the SPY report it carries the single strongest
sentence in the analysis — "not one of the 157 strategies beat buy-and-hold" — and a
template that styles it as a muted footnote would bury the best finding in the
project.

**The demo needs both halves.** BLUEPRINT §7 step 6: run it on SPY and get
LIKELY_LUCK, then run it on a planted edge and get LIKELY_SKILL. Use the effect size
from `tests/unit/test_verdict.py::edge_trials` — 5 of 50 variants at Sharpe 3.0 over
ten years — because that is the configuration measured to return LIKELY_SKILL on all
25 datasets tried. A weaker edge will make the demo flaky, and §8 explains why.

**`luckdet demo` should not invent numbers.** Everything it prints should come from
the same functions the library exposes. If the demo needs a helper that runs all four
statistics over a `MiningResult`, that helper belongs in the library with a test, not
in `cli.py`.

**Existing CLI:** `version`, `summary`, `mine` work today. `report` and `demo` are the
new ones. `cli.py` has `B008` ignored in ruff config because typer requires calls in
argument defaults.

### Acceptance criteria

- `luckdet demo` from a clean clone: cache/download path produces the SPY narrative
  ending in LIKELY_LUCK, then the planted-edge narrative ending in LIKELY_SKILL.
- `luckdet demo --offline` runs with no network and labels everything SYNTHETIC.
- One HTML file, opens correctly with no network access, no external assets.
- Both figures snapshot-tested on structure; no image-hash assertions.
- No network in tests. The demo's data loading is tested through the injectable
  downloader already in `io/prices.py`.
- `make check` green; coverage stays at or above 90% on `src/luckdetector/stats/`.

### Prediction, written down so it can be checked rather than rationalised

The HTML report will be the easiest phase so far and the plots the fiddliest part of
it — specifically, getting the Sharpe-distribution figure to make the hurdle legible
without it looking like the winner is far from the mass, when the point is that it is
*inside* it. I expect at least one caveat to be discovered while writing the template
that is not yet in METHODS. Phases 5, 6 and 7 each turned up a spec error; if Phase 8
turns up none, that is more likely to mean the spec was vague than that it was right.

> **Outcome: right about which part would be hard, wrong about why; right about the
> spec error, and it found two.** The Sharpe-distribution figure was indeed the
> fiddliest part — but the difficulty was not making the winner's distance from the
> mass legible, it was that the winner is *not* inside the mass and no amount of
> drafting could make it look as though it were. The premise of the prediction was
> the thing that failed. The HTML itself was the easiest part, as forecast. The
> caveat discovered while writing the template was not a new limitation but a
> presentational one — §11.3's `p = 0.0000` row — which is arguably a weaker hit
> than the prediction intended, and is recorded as such rather than rounded up.

---

## 7. Environment

```bash
cd ~/Documents/"Project 1"
source .venv/bin/activate          # Python 3.14
make check                         # ruff + mypy + pytest
```

Python 3.10 is the floor (`zip(strict=True)`, runtime `X | None` in typer signatures).
macOS system Python is 3.9 and will not work. Editable installs need pip ≥ 21.3.

`outputs/` holds the cached SPY CSV and the saved run summaries (`spy_run.txt`,
`spy_pbo.txt`, `spy_rc_spa.txt`, `spy_verdict.txt`, and from Phase 8 `spy_demo.txt`
and `spy_report.html`). It is gitignored — real prices stay local and are available
for analysis without re-downloading. Reported numbers live in `README.md` and
`docs/METHODS.md` so a stranger cloning the repo can still see them.

`luckdet demo` searches `outputs/` before the user cache, so running it from the repo
root picks up that CSV and needs no network.

### Notes for agent sessions — read before running anything

**Enable file deletion for the folder as your first action, before any git command.**
The sandbox mount can start without delete permission. Git cannot clear
`.git/index.lock` without it, so *every* git write fails — and worse, even a plain
`git status` creates a lock it then cannot remove, leaving a stale
`.git/index.lock` that blocks the user's own commands until they delete it by hand.
This cost real time in Phase 5. If a delete returns "Operation not permitted",
request delete access for the folder rather than routing commits to the user. Use
`git --no-optional-locks status` for read-only inspection if in doubt.

**Do the committing.** Stage and commit the work yourself; the user should only ever
need to run `git push`.

**Sandbox `mypy` lags the local stubs.** Phase 5 shipped a real annotation bug — a
bool mask typed as `FloatArray` — that passed on the sandbox's numpy 2.2 and failed
on the local 3.14 environment. **Always tell the user to run `make check` locally
before pushing**, and treat a green sandbox as necessary but not sufficient.

---

## 8. Open items

- ~~Delete the old GitHub repo (`backtest-luck-detector-old`)~~ — **done**. The
  pre-rename commits are off GitHub's servers.
- **Known gap in Phase 5:** CSCV has no purge/embargo between in-sample and
  out-of-sample blocks. With S = 16 and T = 4,173 each block is ~261 days, so a
  250-day-lookback rule is contaminated near each seam. This makes the winner look
  *more* persistent than it is, so the true PBO is likely worse than 0.84. A purged
  variant is the obvious extension.
- **Known gaps in Phase 6**, both measured rather than suspected:
  - **SPA does not protect against pruning the family.** It defends against
    *padding* with obvious garbage; *deleting* the merely-bad variants after seeing
    the results defeats both tests equally. On the real SPY family against zero,
    pruning 157 → best 16 moves RC 0.242 → 0.068 and SPA 0.411 → 0.065, with
    nothing recentred at any stage. Same honest-book-keeping dependence DSR has and
    PBO does not.
  - **"SPA ≥ RC power" is not a universal domination.** It holds in the case SPA
    was designed for (a real edge buried among genuinely bad strategies) and the
    test asserts it there. On SPY against zero it fails — RC 0.24, SPA 0.43 —
    because none of the 157 falls below Hansen's cutoff, so only the studentisation
    is in play. Do not quote it as a general property.
- **Known gap in Phase 7 — the verdict layer has poor power, measured.** The SPY
  *luck* verdict is robust to any defensible threshold: DSR flags above 0.77, PBO
  below 0.84, and one flag suffices. The *skill* side is weak. Across 25
  independent datasets, a genuine annualised Sharpe of 2.0 planted in 10 of 50
  variants over five years is called LIKELY_SKILL only **20%** of the time; it
  takes 5 of 50 at Sharpe 3.0 over ten years to reach 100%. **DSR is the binding
  constraint at every effect size** — PBO returns 0.000 and SPA rejects throughout.
  So LIKELY_LUCK is weak evidence of absence. Full power table in METHODS §9;
  `test_detection_rate_at_a_realistic_edge_is_poor` pins the figure so it cannot
  drift or be quietly threshold-tuned away. Do not "fix" this by lowering
  `DSR_THRESHOLD` — the conservatism is the instrument working, and the honest
  response is the caveat, not a friendlier bar.
  - Worth checking before Phase 8 whether the effective-trial count is the real
    culprit: DSR's hurdle rises with the *dispersion* of trial Sharpes, so planting
    5 good variants scores better than planting 10 of the same quality. That is
    arguably correct, but it has not been examined.
- **Known gaps in Phase 8**, both small:
  - `luckdet report` and `luckdet demo` write the *real* half to HTML only. The
    planted-edge control prints to the terminal and is not rendered. That matches
    the brief ("one self-contained HTML file") but a second file, or both
    assessments in one document, would make the control easier to point at.
  - The remaining uncovered lines in `cli.py` are all in `mine`, which predates
    this phase: its download branch and its error handler have no test because
    nothing injects a downloader through that command. `report` covers the
    equivalent path by stubbing `cli.load_prices`; `mine` could do the same.
- **The phase plan was cut after Phase 5** — see `docs/BLUEPRINT.md` §6 and §6a. The
  Harvey–Liu haircut is dropped as redundant with DSR; the standalone validation
  suite is folded into `tests/unit/` where the calibration tests already live; the
  CLI merges into the report phase; the manim animation layer is dropped entirely.
  What remains after Phase 7: **8 report + CLI, 9 docs and release.**
  Do not resurrect `stats/haircut.py`, `tests/validation/` or `animations/` — each
  was removed for a reason recorded in §6a.
