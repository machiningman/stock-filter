# Indonesian LQ45 Weekly Stock Screener

A Python tool that screens LQ45 stocks using fundamental and technical filters, classifies them into 4 categories, scores 0-100, and exports a weekly CSV report.

Screening/reporting only — no auto-trading, no buy/sell signals.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r stock_screener/requirements.txt
.venv\Scripts\python.exe -m stock_screener.src.main
```

- **Required input files:** `stock_screener/data/lq45_constituents.csv` and `stock_screener/data/fundamentals_latest.csv` (sample data provided; update periodically from IDX).
- **Output:** `stock_screener/reports/weekly_screening_YYYY-MM-DD.csv` (19 columns: ticker, sector, scores, status, technical values, reasons, warnings, review note).
- First run fetches ~18 months of price data from yfinance (~1-2 min for sample; ~5-10 min for full 45 tickers).
- *(On macOS/Linux, use `.venv/bin/python`)*

## Configuration

All thresholds in `stock_screener/config.yaml` — no code changes needed:

- **`technical`** — SMA periods, max distance from SMA, RS weeks
- **`fundamental_bank`** — ROE, PBV, NPL, profit growth (hard filters)
- **`fundamental_non_bank`** — ROE, DER, revenue growth, profit growth (hard filters)
- **`scoring`** — 5 weights + normalization ranges (valuation uses PBV for banks, PER for non-banks)
- **`data`** — cache TTL, price history months

## Classification & Scoring

| Status | Condition |
|--------|-----------|
| Candidate | Fundamental pass + Technical pass |
| Watch | Fundamental pass + Technical fail |
| Speculative | Fundamental fail + Technical pass |
| Avoid | Both fail, or data completeness < 60% |

- **Technical pass requires:** Close > SMA20, Close > SMA50, SMA20 rising, RS > 0, distance from SMA20 < max threshold.
- **Data completeness** = 50% × (required fundamental fields present) + 50% × (price data available).
- **Scoring** (0-100, linear interpolation clipped to range): Fundamental quality (35%), Earnings momentum (20%), Valuation (15%), Technical trend (20%), Relative strength (10%).

## Project Structure

```
stock_screener/
├── config.yaml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_io.py
│   ├── pipeline.py
│   └── main.py
├── data/
├── reports/
└── tests/
    └── __init__.py
```

## Disclaimer

This is a screening/reporting tool, not financial advice. Price data sourced from yfinance (unofficial API) and may contain delays or inaccuracies. No guarantees of accuracy or profitability. Use at your own risk.
