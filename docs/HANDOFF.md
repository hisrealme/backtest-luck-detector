# Current state and next steps

Written at the end of Phase 4. Read this before touching the code — it records the
decisions that have already been argued out, so they don't get re-litigated, and it
specifies Phase 5 precisely enough to implement from.

---

## 1. Where the project is

| | |
|---|---|
| Phases complete | 0–4 (scaffolding, core model, PSR/DSR, bootstrap, mining) |
| Commits | 13, all authored `hisrealme <315065552+hisrealme@users.noreply.github.com>` |
| Tests | 201 passing (1 network test deselected), **96%** coverage |
| Gates | `ruff` clean, `mypy --strict` clean |
| CI | GitHub Actions, Python 3.10 / 3.11 / 3.12 / 3.13 / 3.14 |
| Repo | `github.com/hisrealme/backtest-luck-detector`, public |

### Modules that exist

```
src/luckdetector/
├── types.py            ReturnSeries, TrialMatrix, TestResult, Verdict  (frozen, validating)
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
    └── bootstrap.py    iid / circular / stationary, Politis–White block length
```

---

## 2. Results already established

Run on **SPY, 2010-01-04 to 2026-08-07**, 4,174 trading days, adjusted closes.
Reproduce with `luckdet mine SPY --start 2010-01-01`.

```
Buy-and-hold           Sharpe 0.867   +814.3%   maxDD -33.7%
Winner MA(80,250)      Sharpe 0.491   +205.6%   maxDD -40.3%   naive PSR 0.9764
157 trials -> 12 effectively independent
Expected max Sharpe of noise: 0.309
Deflated Sharpe Ratio: 0.7692   -> LIKELY LUCK
```

**The finding that matters: 0 of 157 strategies beat buy-and-hold.** The winner's
returns minus buy-and-hold's have an annualised Sharpe of **-0.416**. Twenty-six
variants (17%) had a negative Sharpe outright. Breakouts were worst — median -0.227,
none positive.

Caveat kept in the README: 2010–2026 was a historic bull market, so trend rules that
go flat or short necessarily give up ground. This is evidence about this family over
this window, not a verdict on trend following.

---

## 3. Decisions already made — do not re-open without reason

**Units.** Every public function takes and returns **annualised** Sharpe ratios and
converts internally. PSR/DSR maths is per-period. Mixing them inflates significance by
~16x on daily data and is the classic bug in this literature.

**Synthetic data is for tests, never for headlines.** Tests must not touch the
network — non-determinism, vendor outages, silently revised history. But any *reported
result* uses real prices. `synthetic_prices()` is honestly named and never dressed up
as a ticker. No market data is committed to the repo.

**Keep every trial, never just the winner.** `mine()` returns the whole family. PBO and
Reality Check are impossible without it.

**Validate against reality, not against the paper's algebra.** Every statistic is
checked against simulation or a known null, not against a re-transcription of its own
formula. Existing examples: the Sharpe standard error matches the empirical spread of
2,000 simulated estimates within 5%; the Gumbel expected-maximum matches brute-force
simulation within 6% for N ∈ {10, 50, 200, 1000}; MinTRL round-trips through PSR.

**Seed everything.** Every stochastic routine takes an explicit `rng`/`seed`. Same seed,
identical output.

**Guard degenerate series.** `moments.is_effectively_constant` exists because `np.std`
of identical values returns ~2e-19, not 0.0. That produced a Sharpe of 4.6e15 in Phase 1
and a block length of 23.1 in Phase 3. Any new routine that divides by a dispersion
measure must gate on it.

**Test fixtures draw from one seeded stream per batch.** Seeding each row with
consecutive seeds gives measurably under-dispersed extremes, which silently weakens any
test of maximum-Sharpe behaviour. Use `make_trials` / `make_exact_returns` in
`tests/conftest.py`; the latter fixes the *realised* Sharpe exactly, for tests that need
a known input.

---

## 4. Phase 5 — PBO via CSCV (next)

Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest Overfitting*.

### Algorithm

1. Split the `T` periods of a `TrialMatrix` into `S` disjoint **contiguous** submatrices
   (default `S = 16`). Contiguity matters — random splits destroy serial dependence.
2. For each of the `C(S, S/2)` ways to choose `S/2` blocks as **in-sample**, the
   complement is **out-of-sample**. For `S = 16` that is 12,870 splits.
3. In each split: pick `n* = argmax` of in-sample performance across trials.
4. Find that same trial's **rank** among all trials out-of-sample; normalise to
   `ω̄ ∈ (0, 1)`.
5. Logit `λ = ln(ω̄ / (1 − ω̄))`.
6. **PBO = P(λ ≤ 0)** — how often the in-sample winner lands in the bottom half
   out-of-sample.

### Companion outputs from the same loop

- **Performance degradation**: OLS of OOS Sharpe on IS Sharpe. A *negative slope* is
  damning — better in-sample performance predicting worse out-of-sample.
- **Probability of loss**: fraction of splits where the IS-best has OOS Sharpe < 0.
- **Stochastic dominance**: OOS distribution of the IS-best vs a randomly chosen trial.

### Interpretation baseline

Under pure noise PBO → 0.5. Above 0.5 means selection is actively anti-predictive.

### Acceptance criteria

- PBO ≈ 0.5 on synthetic zero-edge trials
- PBO < 0.1 on synthetic data with a planted true edge
- Handles odd `S`, `S` larger than `T`, and single-trial input with clear errors
- Runs 157 × 4,173 with `S = 16` in reasonable time (12,870 splits — vectorise the
  ranking, don't loop in Python over trials)

### Files

`src/luckdetector/stats/pbo.py`, `tests/unit/test_pbo.py`, export from
`stats/__init__.py`, tick Phase 5 in `README.md`.

---

## 5. Environment

```bash
cd ~/Documents/"Project 1"
source .venv/bin/activate          # Python 3.14
make check                         # ruff + mypy + pytest
```

Python 3.10 is the floor (`zip(strict=True)`, runtime `X | None` in typer signatures).
macOS system Python is 3.9 and will not work. Editable installs need pip ≥ 21.3.

`outputs/` holds the cached SPY CSV and is gitignored — real prices are available
locally for analysis without re-downloading.

---

## 6. Open items

- Delete the old GitHub repo (`backtest-luck-detector-old`) — that is what actually
  removes the pre-rename commits from GitHub's servers.
- Confirm CI is green on the new repo; it has not run there yet.
- `docs/METHODS.md` does not exist yet. It should be written alongside each remaining
  phase: the maths, the intuition, and the questions someone would ask about it.
- Phases 6–12 and the manim scenes are specified in `docs/BLUEPRINT.md` (see §6a for the
  animation layer's isolation rules).
