# ARCHITECTURE.md — LQ45 Weekly Stock Screener

## System Overview

A Python screening pipeline that ingests LQ45 constituents + fundamentals, fetches price data from yfinance, applies fundamental and technical filters, classifies stocks into 4 categories, scores them 0-100, and exports a weekly CSV report. A separate backtest engine evaluates historical filter performance, and a diagnostics module analyzes backtest results.

## Module Dependency Graph

`
main.py ──────┬──► config.py
backtest.py ──┼──► data_io.py
              └──► pipeline.py ──► config.py

backtest_diagnostics.py  (standalone — no internal imports)
`

- `main.py` and `backtest.py` import from `config.py`, `data_io.py`, and `pipeline.py`.
- `pipeline.py` imports only `is_bank_sector()` from `config.py`.
- `data_io.py` has **no internal imports** — it receives `config` as a function parameter.
- `backtest_diagnostics.py` is standalone — imports only stdlib, pandas, numpy.
- All filter and scoring functions are pure (same inputs → same outputs). `generate_report()` and a few functions that log warnings are the exceptions.

## Module Responsibilities

### `config.py` — Configuration Layer (~300 lines)
- Loads and validates `config.yaml` with section-level defaults
- Validates required top-level keys, scoring weights sum, normalization ranges
- Exports `is_bank_sector(sector, config)` for sector detection (case-insensitive match against `sectors.bank` list)
- Exports `get_bank_sectors(config)` — returns lowercased bank sector names
- **No business logic** — only config parsing and validation

### `data_io.py` — Data Access Layer (~560 lines)
- **CSV I/O**: `load_universe()`, `load_fundamentals()` — parse + validate required columns
- **Validation**: `validate_data()` — checks ticker overlap, duplicates, out-of-range values (ROE > 200, DER < 0)
- **Price fetching**: `fetch_prices()`, `fetch_index()` — yfinance with `@retry` (tenacity, 5 attempts, exponential backoff)
- **Caching**: Per-ticker parquet + meta.json in `data/cache/`, TTL-based expiry (default 7 days), atomic writes (temp file + `os.replace`)
- **Column normalization**: Handles yfinance MultiIndex columns (flattens to simple strings)

### `pipeline.py` — Screening Logic Layer (~1,100 lines)
All filter and scoring functions are pure: same inputs → same outputs. `generate_report()` writes CSV as a side effect.

| Function | Purpose |
|----------|---------|
| `resample_to_weekly()` | Daily → weekly OHLCV (Friday-ending) |
| `calculate_sma()` | Simple moving average |
| `calculate_distance_from_sma()` | `(close - sma) / sma` |
| `calculate_relative_strength()` | Stock return vs index return over N weeks |
| `calculate_technical_features()` | Aggregates SMA20, SMA50, distance, RS, rising trend |
| `apply_fundamental_filter()` | Bank vs non-bank hard filter (calls `is_bank_sector()`) |
| `apply_technical_filter()` | 5 technical checks: Close>SMA20, Close>SMA50, SMA20 rising, RS>0, distance<max |
| `calculate_data_completeness()` | 50% fundamental fields + 50% price availability |
| `classify_stock()` | Candidate / Watch / Speculative / Avoid decision tree |
| `normalize_score()` | Linear interpolation to 0-100, supports inverted metrics |
| `calculate_fundamental_score()` | ROE-based |
| `calculate_earnings_momentum_score()` | Net profit growth YoY-based |
| `calculate_valuation_score()` | PBV for banks, PER for non-banks (inverted) |
| `calculate_technical_score()` | Two-sided absolute deviation from target distance |
| `calculate_relative_strength_score()` | RS 13-week score |
| `calculate_final_score()` | Weighted sum of 5 sub-scores, clipped to [0, 100] |
| `generate_report()` | CSV export, sorted by status priority then score descending |

### `main.py` — Screener Orchestrator (~420 lines)
- Entry point: `python -m stock_screener.src.main`
- Pipeline: load config → load universe → load fundamentals → validate → fetch prices → for each stock (resample → filter → classify → score) → export report
- Error handling: per-ticker try/except, graceful degradation (missing fundamentals → empty Series)
- Report date derived from last index weekly bar

### `backtest.py` — Backtest Engine (~585 lines)
- Entry point: `python -m stock_screener.src.backtest`
- Uses separate `data/cache/backtest/` subdirectory to avoid cache collision with screener's 18-month cache
- Fetches extended history (`backtest.history_months`, default 60 months)
- Generates evaluation dates: skips warmup period + last horizon weeks
- Evaluates each ticker at each date: slice data → technical features → technical filter → forward returns
- Outputs: `backtest_technical_trades.csv` (per observation) + `backtest_technical_summary.csv` (aggregated by horizon/pass-fail)
- **Technical-only**: does not apply fundamental filters (answers \"do technical passes predict outperformance?\")
- Warns about survivorship bias (uses current LQ45 constituents for historical analysis)

### `backtest_diagnostics.py` — Backtest Diagnostics (~1,160 lines)
- Entry point: `python -m stock_screener.src.backtest_diagnostics [--trades-csv PATH] [--output-dir PATH] [--top-n N]`
- Reads `backtest_technical_trades.csv` and produces 6 diagnostic analyses + console output with ASCII diverging bar charts
- **Analyses**:
  - `analyze_by_ticker()` — per-ticker excess returns, pass rates, win/loss asymmetry
  - `analyze_by_year()` — per-year breakdown with market regime (bull/bear)
  - `analyze_by_signal()` — signal profile buckets (distance from SMA20 × RS × rising trend)
  - `analyze_4w_vs_13w()` — horizon reversal contingency table (4w winners → 13w losers?)
  - `analyze_extreme_observations()` — best and worst 13w excess return observations
  - `analyze_failure_reasons()` — frequency analysis of filter failure reasons
- **Output**: 6 diagnostic CSVs + console summary with diverging bar charts (Unicode or ASCII-safe fallback)
- **Standalone**: no internal imports; reads backtest output, does not run the backtest itself

## Data Flow

### Screener Pipeline (main.py)

`
lq45_constituents.csv ──┐
                        ├──► validate_data() ──► universe tickers
fundamentals_latest.csv ─┘
                              │
yfinance API ─────────────────┼──► price data (daily, cached)
                              │
                              ▼
                    resample_to_weekly() ──► weekly OHLCV
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        calculate_technical_features()  apply_fundamental_filter()
                    │                   │
                    ▼                   ▼
            apply_technical_filter()   (bank/non-bank rules)
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    classify_stock() ──► Candidate/Watch/Speculative/Avoid
                              │
                    calculate_*_score() ──► 5 sub-scores → final_score
                              │
                    generate_report() ──► weekly_screening_YYYY-MM-DD.csv
`

### Backtest Pipeline (backtest.py)

`
lq45_constituents.csv ──► fetch extended price history (60 months)
                              │
                              ▼
                    resample_to_weekly() ──► weekly OHLCV
                              │
                    for each date, each ticker:
                      slice → technical features → technical filter
                              │
                              ▼
                    forward returns (4w, 13w) vs IHSG
                              │
                    generate_backtest_reports() ──► backtest_technical_*.csv
`

### Diagnostics Pipeline (backtest_diagnostics.py)

`
backtest_technical_trades.csv ──► load_trades_csv()
                                      │
                                      ▼
                    6 analysis functions ──► diagnostic CSVs
                                      │
                                      ▼
                    print_summary() ──► console output with bar charts
`

## Classification Decision Tree

`
data_completeness < min_data_completeness  →  \"Avoid\"
fundamental_pass AND technical_pass         →  \"Candidate\"
fundamental_pass AND NOT technical_pass     →  \"Watch\"
NOT fundamental_pass AND technical_pass     →  \"Speculative\"
NOT fundamental_pass AND NOT technical_pass →  \"Avoid\"
`

## Scoring Components

| Component | Weight | Metric | Normalization |
|-----------|--------|--------|---------------|
| Fundamental Quality | 35% | ROE | 0→25 (higher is better) |
| Earnings Momentum | 20% | Net Profit Growth YoY | -20→30 (higher is better) |
| Valuation | 15% | PBV (banks) / PER (non-banks) | inverted (lower is better) |
| Technical Trend | 20% | Distance from SMA20 | two-sided, target=0.02, max=0.15 |
| Relative Strength | 10% | RS 13-week | -0.10→0.10 (higher is better) |

## Bank vs Non-Bank Separation

Sector detection via `is_bank_sector()` matches against `config.yaml → sectors.bank` (case-insensitive).

| Metric | Bank | Non-Bank |
|--------|------|----------|
| ROE | ≥ 10% | ≥ 10% |
| PBV | ≤ 2.5 | — |
| DER | — | ≤ 1.5 |
| Revenue Growth YoY | — | ≥ 0% |
| Net Profit Growth YoY | ≥ -10% | ≥ -10% |
| Gross NPL | ≤ 3% (optional) | — |
| OCF Positive | — | required (configurable) |

## Cache Strategy

- **Location**: `stock_screener/data/cache/` (screener), `stock_screener/data/cache/backtest/` (backtest)
- **Format**: Parquet + `.meta.json` (timestamp) per ticker
- **TTL**: 7 days (configurable via `data.cache_ttl_days`)
- **Write**: Atomic (temp file → `os.replace`)
- **Read**: Validates columns, empty content, age; cleans up corrupted/expired files
- **Inter-request delay**: 0.5-1.5s random sleep between yfinance requests

## Test Organization

One test file per module under `stock_screener/tests/`:

| Test File | Tests |
|-----------|-------|
| `test_config.py` | Config loading, validation, defaults, `is_bank_sector()` |
| `test_data_io.py` | CSV loading, validation, cache read/write/fetch |
| `test_filters.py` | Fundamental and technical hard filters |
| `test_scoring.py` | All scoring functions, normalization, final score |
| `test_classification.py` | `classify_stock()`, `calculate_data_completeness()` |
| `test_resample.py` | `resample_to_weekly()`, `calculate_technical_features()` |
| `test_report.py` | `generate_report()` output format and sorting |
| `test_backtest.py` | Backtest date generation, forward returns, evaluation |
| `test_backtest_diagnostics.py` | Diagnostics analysis functions, CSV loading, console output |
| `test_integration.py` | End-to-end pipeline integration |
