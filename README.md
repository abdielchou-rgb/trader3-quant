# trader3

> Quantitative trading engine for the China A-share market — dual-mode: standalone engine + embeddable Python library, with an honesty-first data discipline layer (IronGate) that most open-source frameworks don't have.

**All outputs are candidate signals, not investment advice.**

---

## Why another quant framework?

Most retail quant stacks fail not at strategy ideas but at **silent data leakage**, **look-ahead bias**, and **backtests that ignore A-share market microstructure**. trader3 is built around the opposite defaults:

| Default | What trader3 does |
|---|---|
| No look-ahead | Signal on day *t* → effective day *t+1* (`pending_weights`), costs charged on the effective day |
| No survivorship bias | Universe membership resolved `asof_date` at window start, never the current constituent list over historical windows |
| A-share microstructure | Stamp duty on sells only, per-board price limits (main 9.8% / ChiNext & STAR 19.5% / BSE 29%), lot-size (100-share) rounding, suspension handling |
| Data honesty | Synthetic fallback data is forcibly labeled `（合成数据）`; unlabeled synthetic output is **blocked** by gate G7 |
| Reproducibility | Same seed → same result, enforced by tests (truncation-retrain consistency for the deep model, deterministic GP evolution) |

## Features

**Backtesting & validation**
- Vectorized daily backtest on real Qlib-format binary data (numpy loader, no qlib dependency), synthetic fallback clearly labeled
- Walk-Forward Analysis (WFA) with per-window IS/OOS Sharpe, parameter stability, overfit probability — **first-day turnover costs and price-limit constraints included**
- Purged K-Fold CPCV + Deflated Sharpe for OOS distribution estimation
- Brinson (BHB) industry attribution (real path; Barra exposure honestly declared unavailable until a multi-factor library is wired)

**Factor factory & evolution**
- Genetic programming (GP) alpha mining with alphagen-style fitness (signed IC, ICIR, monotonicity, long-short, parsimony penalty)
- Train ≤ 2023 / validate ≥ 2024 enforced time split; forward returns verified forward-looking by regression tests
- **LSTM deep scorer** (`evolve/core/deep_model.py`): torch CPU model with no-lookahead guarantee (truncation-retrain consistency test), deterministic same-seed output, deterministic numpy ridge fallback when torch is missing (explicitly flagged, never labeled as LSTM)
- Six-gate strategy selector (IC / ICIR / monotonicity / long-short / complexity / correlation dedup)

**Portfolio & execution**
- Risk parity / mean-variance / Black-Litterman optimization + regime-aware probability-weighted allocation
- Almgren-Chriss impact model, A-share cost rules, TWAP/VWAP/IS/Adaptive execution algorithms
- Barra-style cross-sectional risk model, risk overlay, kill-switch

**Market state**
- 4-state Gaussian HMM (trending-up / ranging / bearish / high-vol), hmmlearn with built-in EM fallback

**Live pathway (honest status: not yet battle-tested with real money)**
- `ShadowBroker`: wraps any real broker, records orders without dispatching — for backtest→live parallel validation
- **QMT/miniQMT adapter** (`trader3/v2/live/qmt_broker.py`): fail-fast when `xtquant` is missing, explicit `QMT-SIM` labeling in offline simulated mode, A-share lot rounding
- **Shadow reconcile** (`trader3/v2/shadow_reconcile.py`): daily target-vs-shadow-position gap tracking, persisted to `shared_state/shadow/shadow_run.json`
- CTP adapter (futures, guarded behind SIMULATED when vnpy_ctp absent), paper trading with atomic state writes

**Embeddable**
- `from trader3 import Trader3` → 10 tools, each returning a unified `Trader3Response` (summary / key_metrics / charts / caveats / metadata), designed to be cited inside research reports

## Quick start

```bash
pip install -e .
pytest tests/ -q          # 514 passed / 2 skipped baseline
```

```python
from trader3 import Trader3

t3 = Trader3()
r = t3.run_backtest(start_date="2024-01-01", end_date="2024-12-31")
print(r.summary)
r = t3.diagnose_market_regime()
print(r.summary)
```

CLI:

```bash
python -m trader3.cli backtest --start 2024-01-01 --end 2024-12-31
python -m trader3.cli regime
py -3.11 evolve/run_evolution.py --universe csi300 --gen 15 --pop 50 --deep   # GP + LSTM candidates
```

## Data

The engine reads Qlib-format daily binaries (`.day.bin` with open/high/low/close/volume/vwap/amount/factor, close = forward-adjusted) and a `financials.db` SQLite of A-share fundamentals. Point the loader at your data root:

- `T3_QLIB_BIN` / `QLIB_BIN` env var, or
- `trader3.data_provider.QlibDataProvider(data_dir=...)`

Without real data the engine degrades to clearly-labeled synthetic fallback — it never silently fabricates numbers (hardcoded fake indicators were removed during the 2026-08 audit; gate G7 blocks unlabeled synthetic output).

A small ETF sample (`data/etf/*.csv`) is included for the evolve pipeline demo.

## Project layout

```
trader3/            core package — 10 tools behind IronGate gates
  tools/            backtest / optimize / execution / signal / valuation (+ cpcv)
  v2/               event collection, trigger engine, risk chain, paper trading,
                    factor DSL, Barra-style risk model, MoE ensemble, live/
  v2/live/          broker_base + ShadowBroker / CTP / QMT / paper / tiger
evolve/             GP factor factory + LSTM deep scorer
scripts/            market-data incremental update pipeline, baselines
tests/              516 tests (tests/ is isolated via pyproject testpaths)
docs/               baselines, research surveys, audit reports
```

## Quality gates

```bash
py -3.11 -m ruff check .   # all green
py -3.11 -m mypy           # 0 errors on the 8 checked modules
py -3.11 -m pytest tests/ -q
```

CI (`.github/workflows/ci.yml`): ruff + mypy + pytest on push, nightly full backtest matrix on schedule.

## Honest limitations

- The QMT adapter's real-money path (`xtquant`/`xttrader`) is **not yet battle-tested** — only the offline SIM path is covered by tests. Do not wire real capital before terminal integration testing.
- The LSTM scorer achieved IC ≈ 0.001 on real csi300 data in the 2026-09 smoke run — an honest result the selector correctly rejected. Deep-model value needs longer windows / richer features; no claims made.
- WFA nightly cost/limit coverage is first-day-of-window only; intraday granularity is not modeled in the main engine.
- financials.db has no announcement-date column; asof anti-look-ahead depends on `AnnouncementCalendar` coverage.
- ST-list detection relies on the instrument name field (missing names fail closed).

## Related internal docs

`CLAUDE.md` (agent working contract, Chinese), `docs/` (baselines & audit reports, mostly Chinese). The codebase is bilingual: docstrings/summaries in Chinese for the primary user, identifiers and APIs in English.

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

trader3 outputs **candidate signals, not investment advice**. Backtest results — including synthetic-data fallbacks — do not represent real-world performance. Validate with real data, run shadow-mode parallel tracking, and understand the market-microstructure assumptions before any live deployment.
