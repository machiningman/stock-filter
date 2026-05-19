# AGENTS.md — Agent Guide for stock-filter

## Project

Indonesian LQ45 Weekly Stock Screener. Screens stocks using fundamental + technical filters, classifies into 4 categories (Candidate, Watch, Speculative, Avoid), scores 0-100, exports weekly CSV. Screening/reporting only — no auto-trading.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r stock_screener/requirements.txt
```

*(On macOS/Linux, use `.venv/bin/python`)*

## Quick Commands

```bash
# Run the screener
.venv\Scripts\python.exe -m stock_screener.src.main

# Run backtest
.venv\Scripts\python.exe -m stock_screener.src.backtest

# Run backtest diagnostics
.venv\Scripts\python.exe -m stock_screener.src.backtest_diagnostics

# Run all tests
.venv\Scripts\python.exe -m pytest stock_screener/tests/ -v

# Run a single test file
.venv\Scripts\python.exe -m pytest stock_screener/tests/test_scoring.py -v
```

## Directory Map

```
stock_screener/
├── config.yaml          # All thresholds (technical, fundamental, classification, scoring, sectors, data, logging, backtest)
├── requirements.txt     # pyyaml, pandas, numpy, yfinance, tenacity, pytest, pyarrow
├── src/
│   ├── config.py        # YAML config loader with validation, is_bank_sector()
│   ├── data_io.py       # Data loading, validation, yfinance fetch with retry, parquet cache, CSV export
│   ├── pipeline.py      # Composable functions: resampling, filters, scoring, classification, report generation
│   ├── backtest.py              # Backtest engine with forward returns over configurable horizons
│   ├── backtest_diagnostics.py  # Analyzes backtest output: ticker/year/signal breakdowns, ASCII charts
│   └── main.py                  # CLI entry point; orchestrates the full pipeline
├── data/
│   ├── lq45_constituents.csv    # Input: ticker, company_name, sector, effective_period
│   ├── fundamentals_latest.csv  # Input: ticker, sector, financial metrics
│   └── cache/                   # Price cache (parquet + meta.json); auto-created, gitignored, TTL 7 days
├── reports/             # weekly_screening_YYYY-MM-DD.csv, backtest_technical_summary.csv, backtest_technical_trades.csv
└── tests/               # pytest suite: test_config, test_data_io, test_filters, test_scoring,
                         #   test_classification, test_report, test_resample, test_backtest,
                         #   test_backtest_diagnostics, test_integration
plan/                    # Execution plans (active & completed); gitignored, local-only
```

## Golden Principles

1. **Parse data at boundaries** — validate all external data (yfinance, CSV inputs) before use.
2. **Always add tests** — every new feature or bug fix must include tests.
3. **Config-driven, not code-driven** — thresholds live in `config.yaml`. Do not hardcode values.
4. **Bank vs non-bank separation** — detection via `is_bank_sector()` in `config.py` (matches `config.yaml → sectors.bank`). Banks use ROE/PBV/NPL; non-banks use ROE/DER/RevenueGrowth/ProfitGrowth/OCF. Keep filter logic separate.
5. **Scoring uses linear interpolation** — all sub-scores normalized between config-defined min and target values, weighted, then clipped to [0, 100].
6. **No side effects in pure functions** — filters and scoring functions must be deterministic given the same inputs.
7. **Preserve existing test contracts** — if a test exists, do not change its expected behavior without explicit user approval.

## Architecture Notes

- `config.py` loads and validates `config.yaml` — the single source of truth for thresholds.
- `data_io.py` handles yfinance fetches with retry (tenacity), local parquet caching (pyarrow), data validation, and CSV I/O.
- `pipeline.py` provides composable building blocks. `main.py` orchestrates the full pipeline: load data → apply filters → classify → score → export.
- `backtest.py` computes forward returns over configurable horizons for historical analysis.
- `backtest_diagnostics.py` analyzes backtest output: per-ticker/year/signal breakdowns, horizon reversal analysis, extreme observations, and failure reasons.

## Workflow Rules

1. **Explore first** — use the `explore` subagent and Semble MCP to understand existing code before proposing changes.
2. **Plan non-trivial changes** — create or update a plan in `plan/` before implementing multi-file or architectural changes.
3. **Self-review, then dispatch reviewers** — review your own diff, then use all 3 `reviewer` subagents (different models) before marking work as complete.
4. **Update docs with code** — if behavior changes, update relevant docs/comments in the same PR.

## Where to Find More

- **Architecture:** `ARCHITECTURE.md` (module dependencies, data flow, function map, scoring/bank tables)
- **Execution plans:** `plan/` directory (active plans, completed plans, tech debt)
- **Configuration reference:** `stock_screener/config.yaml`
- **Full project docs:** `README.md`
