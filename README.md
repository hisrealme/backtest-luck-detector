# The Backtest Luck-Detector

**How much of your backtest is luck?**

`luckdetector` takes a trading strategy's track record — ideally along with every
variant you tried before settling on it — and estimates how much of the reported
performance is explained by chance and selection bias rather than genuine edge.

> A backtest is a *maximum*, not a sample mean. Nobody publishes the first strategy
> they tried; they publish the best of the several hundred they tried. The reported
> Sharpe ratio is an order statistic, and the ordinary t-test is invalid on it.

## What it measures

Three failure modes get conflated in practice. This tool separates them:

| Failure mode | The question | Test |
|---|---|---|
| **Small-sample noise** | Is the record long enough to distinguish this Sharpe from zero, given fat tails? | Probabilistic Sharpe Ratio, Minimum Track Record Length |
| **Selection bias** | Would the best of N random strategies have looked this good anyway? | Deflated Sharpe Ratio, White's Reality Check, Hansen's SPA, Harvey–Liu haircut |
| **Overfitting** | Does the parameter set that won in-sample survive out-of-sample? | Probability of Backtest Overfitting via CSCV |

## What it does, on real data

Mine 157 moving-average, momentum, RSI and breakout variants over **SPY, 2010-01-04
to 2026-08-07** — 4,174 trading days. Keep the best one, exactly as any backtester
would.

```
$ luckdet mine SPY --start 2010-01-01

SPY, 2010-01-04 to 2026-08-07 (cache)
Mined 157 strategies; buy-and-hold Sharpe 0.867

           Winner: MA(80,250)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Statistic                    ┃  Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Annualised Sharpe            │  0.491 │
│ Total return                 │ 205.6% │
│ Max drawdown                 │ -40.3% │
│ Naive PSR vs zero            │ 0.9764 │
│ Trials run                   │    157 │
│ Effectively independent      │     12 │
│ Expected max Sharpe of noise │  0.309 │
│ Deflated Sharpe Ratio        │ 0.7692 │
└──────────────────────────────┴────────┘
VERDICT: likely luck
```

Two findings, and the second is the one that matters.

**The winner fails deflation.** A Sharpe of 0.491 with a 97.6% probabilistic Sharpe
ratio looks significant. But 157 variants cluster into ~12 independent bets, the best
of pure noise would be expected to score 0.309 anyway, and the honest probability the
edge is real is 76.9% — under the 95% bar.

**Not one of the 157 beat buy-and-hold.** Zero. Buy-and-hold returned 814.3% at a
Sharpe of 0.867 with a *smaller* drawdown (-33.7% against the winner's -40.3%). The
winner's return stream minus buy-and-hold's has an annualised Sharpe of **-0.416**.
The entire exercise destroyed value, and 26 of the variants (17%) posted a negative
Sharpe outright.

That is the whole thesis in one table. A backtester who reported only MA(80,250) would
show you 205% cumulative and a significant t-statistic, while quietly having lost to
doing nothing at all for sixteen years.

<sub>**Caveat, stated because it matters:** 2010–2026 was a historic bull market, and
trend rules that go flat or short necessarily give up ground in a relentless uptrend.
This is evidence about *this* strategy family over *this* period, not a universal claim
about trend following. Run `luckdet mine` on your own symbol and window — the tool has
no stake in the answer.</sub>

Run against 200 strategies with *literally zero* edge planted in any of them, the same
machinery returns a deflated ratio of 0.31, correctly refusing to be impressed by a
winner that posted a 1.12 Sharpe.

## Status

🚧 Under active construction. See [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md) for the full
12-phase build plan and the mathematical specification.

- [x] Phase 0 — repo scaffolding, CI, lint/type gates
- [x] Phase 1 — core data model, loaders, return moments
- [x] Phase 2 — PSR / DSR / MinTRL
- [x] Phase 3 — bootstrap engine
- [x] Phase 4 — strategy mining engine
- [x] Phase 5 — PBO via CSCV
- [ ] Phase 6 — Reality Check / SPA
- [ ] Phase 7 — Harvey–Liu haircut
- [ ] Phase 8 — verdict aggregation
- [ ] Phase 9 — plots + HTML report
- [ ] Phase 10 — CLI *(`luckdet version` and `luckdet summary` work today; `report` / `mine` / `demo` pending)*
- [ ] Phase 11 — statistical validation suite
- [ ] Phase 12 — docs and release

## Install

Requires **Python 3.10 or later**; tested through 3.14. macOS ships 3.9, which is
end-of-life — check with
`python3 --version` and install a current build from
[python.org](https://www.python.org/downloads/) if needed.

```bash
git clone https://github.com/hisrealme/backtest-luck-detector.git
cd backtest-luck-detector
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip     # editable installs need pip >= 21.3
make install                             # pip install -e ".[dev,data]"
make check                               # lint + typecheck + tests
```

## Usage (current)

```python
import numpy as np
from luckdetector import ReturnSeries, TrialMatrix, summarize
from luckdetector.stats import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_trials,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)

returns = ReturnSeries(np.random.default_rng(0).normal(0.0005, 0.01, 1260))

summarize(returns)                                    # descriptive statistics
probabilistic_sharpe_ratio(returns)                   # is the record long enough?
min_track_record_length(returns) / 252                # ...if not, how long?
deflated_sharpe_ratio(returns, n_trials=200)          # would 200 coin flips beat it?
```

If you kept every variant you tried — and you should — hand over the whole family
and let the tool measure the trial count and their correlation itself:

```python
trials = TrialMatrix(all_my_backtests, periods_per_year=252)
result = deflated_sharpe_ratio_from_trials(trials)
print(result.interpretation)
```

Or mine a grid yourself and watch it get indicted:

```python
from luckdetector.mining import mine, synthetic_prices

result = mine(synthetic_prices(3780, seed=42), cost_bps=1.0)   # 157 variants, 15 years
print(deflated_sharpe_ratio_from_trials(result.trials).interpretation)
```

`synthetic_prices` is exactly what it says: a GARCH-like path with realistic
volatility clustering. Nothing here ships fabricated market data dressed up as real
prices. Point `mine()` at your own price series to judge your own backtests.

From the command line:

```bash
luckdet summary returns.csv --date-column date
```

Loading from disk:

```python
from luckdetector import load_returns_csv, load_trials_csv

track_record = load_returns_csv("returns.csv", date_column="date")
every_variant = load_trials_csv("trials.csv", date_column="date")  # the strong input
```

## Design principles

1. **Every number is falsifiable.** Each statistic is validated against a known null
   or a published reference value in the test suite.
2. **No network in tests.** Market data is cached and a bundled fixture backs CI.
3. **Seeded everywhere.** Same seed, byte-identical report.
4. **Library first, CLI second.** No statistics live in user-facing code.
5. **Honest verdicts.** No black-box composite score — a transparent rule table where
   each test can independently raise a flag, and the report says which one fired.

## References

Bailey & López de Prado (2012, 2014); Bailey, Borwein, López de Prado & Zhu (2017);
White (2000); Hansen (2005); Harvey & Liu (2015); Politis & Romano (1994).
Full citations in [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md).

## License

MIT
