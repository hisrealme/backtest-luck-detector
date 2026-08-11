# Current state and next steps

Written at the end of Phase 5. Read this before touching the code — it records the
decisions that have already been argued out, so they don't get re-litigated, and it
specifies Phase 6 precisely enough to implement from.

---

## 1. Where the project is

| | |
|---|---|
| Phases complete | 0–5 (scaffolding, core model, PSR/DSR, bootstrap, mining, PBO) |
| Commits | 16, all authored `hisrealme <315065552+hisrealme@users.noreply.github.com>` |
| Tests | 247 passing, 1 skipped (1 network test deselected), **96%** coverage |
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
    └── pbo.py          PBO via CSCV, degradation, dominance   ← Phase 5
```

`docs/METHODS.md` now exists and covers phases 0–5. **Extend it in the same pass as
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

## 4. Phase 6 — Reality Check & SPA (next)

White (2000), *A Reality Check for Data Snooping*, Econometrica 68(5).
Hansen (2005), *A Test for Superior Predictive Ability*, JBES 23(4).

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
- Runs 157 × 4,173 with B = 1,000 in reasonable time. Vectorise.

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

---

## 5. Environment

```bash
cd ~/Documents/"Project 1"
source .venv/bin/activate          # Python 3.14
make check                         # ruff + mypy + pytest
```

Python 3.10 is the floor (`zip(strict=True)`, runtime `X | None` in typer signatures).
macOS system Python is 3.9 and will not work. Editable installs need pip ≥ 21.3.

`outputs/` holds the cached SPY CSV and the saved run summaries (`spy_run.txt`,
`spy_pbo.txt`). It is gitignored — real prices stay local and are available for
analysis without re-downloading. Reported numbers live in `README.md` and
`docs/METHODS.md` so a stranger cloning the repo can still see them.

**Note for agent sessions:** the sandbox mount may start without file-delete
permission, which breaks *all* git write operations (git cannot clear `.git/index.lock`)
and leaves a stale lock behind that blocks the user's own commands too. If `rm`
returns "Operation not permitted", request delete access for the folder first rather
than routing commits to the user. `mypy` in a sandbox may also lag the local
numpy stubs — Phase 5 shipped a real annotation bug that only the local 3.14
environment caught, so **always run `make check` locally before pushing**.

---

## 6. Open items

- Delete the old GitHub repo (`backtest-luck-detector-old`) — that is what actually
  removes the pre-rename commits from GitHub's servers. **Still outstanding.**
- **Known gap in Phase 5:** CSCV has no purge/embargo between in-sample and
  out-of-sample blocks. With S = 16 and T = 4,173 each block is ~261 days, so a
  250-day-lookback rule is contaminated near each seam. This makes the winner look
  *more* persistent than it is, so the true PBO is likely worse than 0.84. A purged
  variant is the obvious extension.
- Phases 7–12 and the manim scenes are specified in `docs/BLUEPRINT.md` (see §6a for
  the animation layer's isolation rules). Animation is Phase 9, not before.
