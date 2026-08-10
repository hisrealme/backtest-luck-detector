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

## Status

🚧 Under active construction. See [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md) for the full
12-phase build plan and the mathematical specification.

- [x] Phase 0 — repo scaffolding, CI, lint/type gates
- [x] Phase 1 — core data model, loaders, return moments
- [ ] Phase 2 — PSR / DSR / MinTRL
- [ ] Phase 3 — bootstrap engine
- [ ] Phase 4 — strategy mining engine
- [ ] Phase 5 — PBO via CSCV
- [ ] Phase 6 — Reality Check / SPA
- [ ] Phase 7 — Harvey–Liu haircut
- [ ] Phase 8 — verdict aggregation
- [ ] Phase 9 — plots + HTML report
- [ ] Phase 10 — CLI
- [ ] Phase 11 — statistical validation suite
- [ ] Phase 12 — docs and release

## Install

```bash
git clone https://github.com/hisrealme/backtest-luck-detector.git
cd backtest-luck-detector
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev,data]"
make check            # lint + typecheck + tests
```

## Usage (current)

```python
import numpy as np
from luckdetector import ReturnSeries, summarize

returns = ReturnSeries(np.random.default_rng(0).normal(0.0005, 0.01, 1260))
print(summarize(returns))
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
