"""
Main orchestration script for the LQ45 Weekly Stock Screener.

Runs the full screening pipeline:
1. Load configuration
2. Load universe & fundamentals CSVs
3. Fetch price data (with caching)
4. For each stock: resample, filter, classify, score
5. Export CSV report

Usage::

    python -m stock_screener.src.main
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import pandas as pd

from stock_screener.src.config import is_bank_sector, load_config
from stock_screener.src.data_io import (
    fetch_index,
    fetch_prices,
    load_fundamentals,
    load_universe,
    validate_data,
)
from stock_screener.src.pipeline import (
    apply_fundamental_filter,
    apply_technical_filter,
    calculate_data_completeness,
    calculate_earnings_momentum_score,
    calculate_final_score,
    calculate_fundamental_score,
    calculate_relative_strength_score,
    calculate_technical_features,
    calculate_technical_score,
    calculate_valuation_score,
    classify_stock,
    generate_report,
    resample_to_weekly,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector-specific required fields for data completeness
# ---------------------------------------------------------------------------

_BANK_REQUIRED_FIELDS = ["roe", "pbv", "net_profit_growth_yoy"]
_NON_BANK_REQUIRED_FIELDS = [
    "roe",
    "der",
    "revenue_growth_yoy",
    "net_profit_growth_yoy",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_script_dir() -> str:
    """Return the directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


def _setup_logging(config: dict) -> None:
    """Configure root logger based on the *logging* section of *config*."""
    log_cfg = config.get("logging", {})
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = log_cfg.get("format", "%(message)s")
    logging.basicConfig(level=level, format=fmt, force=True)


def _build_result_entry(
    stock_row: pd.Series,
    ticker: str,
    tech_features: dict,
    fund_result: dict,
    tech_result: dict,
    status: str,
    fundamentals_row: pd.Series,
    sector: str,
    config: dict,
) -> dict[str, Any]:
    """Calculate all scores and build a complete result dictionary.

    Parameters
    ----------
    stock_row : pd.Series
        Row from the universe DataFrame (contains ``company_name`` etc.).
    ticker : str
        Ticker symbol.
    tech_features : dict
        Output from :func:`~stock_screener.src.pipeline.calculate_technical_features`.
    fund_result : dict
        Output from :func:`~stock_screener.src.pipeline.apply_fundamental_filter`.
    tech_result : dict
        Output from :func:`~stock_screener.src.pipeline.apply_technical_filter`.
    status : str
        Classification result (``"Candidate"``, ``"Watch"``, etc.).
    fundamentals_row : pd.Series
        The stock's fundamentals row.
    sector : str
        Sector name.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict
        Result dictionary with all fields expected by ``generate_report``.
    """
    # --- Sub-scores ---
    fundamental_score = calculate_fundamental_score(fundamentals_row, config)
    earnings_momentum_score = calculate_earnings_momentum_score(
        fundamentals_row, config
    )
    valuation_score = calculate_valuation_score(fundamentals_row, sector, config)
    technical_score_val = calculate_technical_score(tech_features, config)
    rs = tech_features.get("relative_strength_13w", float("nan"))
    rs_score = calculate_relative_strength_score(rs, config)

    sub_scores = {
        "fundamental_quality": fundamental_score,
        "earnings_momentum": earnings_momentum_score,
        "valuation": valuation_score,
        "technical_trend": technical_score_val,
        "relative_strength": rs_score,
    }
    weights = config.get("scoring", {}).get("weights", {})
    final_score = calculate_final_score(sub_scores, weights)

    # --- Combine reasons, warnings, missing flags ---
    all_reasons = fund_result.get("reasons", []) + tech_result.get("reasons", [])
    all_warnings = fund_result.get("warnings", []) + tech_result.get("warnings", [])
    missing_data_flags = fund_result.get("missing_fields", [])

    return {
        "ticker": ticker,
        "company_name": stock_row.get("company_name", ""),
        "sector": sector,
        "final_score": final_score,
        "status": status,
        "fundamental_score": fundamental_score,
        "earnings_momentum_score": earnings_momentum_score,
        "technical_score": technical_score_val,
        "valuation_score": valuation_score,
        "relative_strength_score": rs_score,
        "close": tech_features.get("close", float("nan")),
        "weekly_sma20": tech_features.get("sma_short", float("nan")),
        "weekly_sma50": tech_features.get("sma_long", float("nan")),
        "distance_from_sma20": tech_features.get("distance_from_sma20", float("nan")),
        "relative_strength_13w": tech_features.get(
            "relative_strength_13w", float("nan")
        ),
        "reasons": all_reasons,
        "warnings": all_warnings,
        "missing_data_flags": missing_data_flags,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the LQ45 weekly stock screening pipeline."""
    script_dir = _get_script_dir()
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    config_path = os.path.join(project_root, "config.yaml")

    # --- Load config --------------------------------------------------------
    try:
        config = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        logging.error("Failed to load config: %s", exc)
        sys.exit(1)

    _setup_logging(config)
    logger.info("Starting LQ45 Weekly Stock Screener")

    # --- Determine paths ----------------------------------------------------
    data_dir = os.path.join(project_root, "data")
    report_dir = os.path.join(project_root, "reports")
    cache_dir = os.path.join(data_dir, "cache")

    universe_path = os.path.join(data_dir, "lq45_constituents.csv")
    fundamentals_path = os.path.join(data_dir, "fundamentals_latest.csv")

    # ---------------------------------------------------------------
    # 1. Load universe
    # ---------------------------------------------------------------
    logger.info("Loading LQ45 constituents...")
    if not os.path.exists(universe_path):
        logger.error("Universe CSV not found: %s", universe_path)
        sys.exit(1)

    try:
        universe_df = load_universe(universe_path)
    except ValueError as exc:
        logger.error("Failed to load universe: %s", exc)
        sys.exit(1)

    # ---------------------------------------------------------------
    # 2. Load fundamentals
    # ---------------------------------------------------------------
    logger.info("Loading fundamentals...")
    if not os.path.exists(fundamentals_path):
        logger.warning(
            "Fundamentals CSV not found: %s. "
            "Continuing with empty fundamentals — all stocks will fail "
            "the fundamental filter.",
            fundamentals_path,
        )
        fundamentals_df = pd.DataFrame(
            {"ticker": pd.Series(dtype="str"), "sector": pd.Series(dtype="str")}
        )
    else:
        try:
            fundamentals_df = load_fundamentals(fundamentals_path)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning(
                "Failed to load fundamentals: %s. "
                "Continuing with empty fundamentals.",
                exc,
            )
            fundamentals_df = pd.DataFrame(
                {"ticker": pd.Series(dtype="str"), "sector": pd.Series(dtype="str")}
            )

    # ---------------------------------------------------------------
    # 3. Validate data
    # ---------------------------------------------------------------
    validation = validate_data(universe_df, fundamentals_df, config)
    for w in validation.get("warnings", []):
        logger.warning(w)

    if not validation.get("is_valid", True):
        for e in validation.get("errors", []):
            logger.error(e)
        logger.error("Data validation failed. Exiting.")
        sys.exit(1)

    # Use deduplicated DataFrames from validation
    universe_df = validation.get("universe", universe_df)
    fundamentals_df = validation.get("fundamentals", fundamentals_df)

    tickers = universe_df["ticker"].tolist()
    if not tickers:
        logger.error("Universe is empty after validation.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 4. Fetch price data
    # ---------------------------------------------------------------
    logger.info("Fetching price data...")
    price_data = fetch_prices(tickers, config, cache_dir)
    index_data = fetch_index(config, cache_dir)

    if index_data.empty:
        logger.error(
            "Failed to fetch index data (%s). Exiting.",
            config.get("data", {}).get("index_ticker", "^JKSE"),
        )
        sys.exit(1)

    if all(df.empty for df in price_data.values()):
        logger.error("All price data fetches failed. Exiting.")
        sys.exit(1)

    # --- Resample index to weekly ---
    index_weekly = resample_to_weekly(index_data)
    if index_weekly.empty:
        logger.error("Index weekly data is empty after resampling. Exiting.")
        sys.exit(1)

    # Derive report_date from last index weekly date
    report_date = index_weekly.index[-1].strftime("%Y-%m-%d")

    # ---------------------------------------------------------------
    # 5. Process each stock
    # ---------------------------------------------------------------
    logger.info("Processing %d stocks...", len(tickers))

    # Build a lookup: ticker -> fundamentals row
    fund_lookup: dict[str, pd.Series] = {}
    for _, row in fundamentals_df.iterrows():
        fund_lookup[row["ticker"]] = row

    results: list[dict[str, Any]] = []

    for _, stock_row in universe_df.iterrows():
        ticker = stock_row["ticker"]
        sector = stock_row.get("sector", "")
        company_name = stock_row.get("company_name", "")

        try:
            # --- Get fundamentals for this ticker ---
            if ticker in fund_lookup:
                fundamentals_row = fund_lookup[ticker]
            else:
                fundamentals_row = pd.Series(
                    {"ticker": ticker, "sector": sector}
                )

            # --- Get price data ---
            price_df = price_data.get(ticker, pd.DataFrame())

            # --- Resample daily -> weekly ---
            stock_weekly = resample_to_weekly(price_df)

            # --- Calculate technical features ---
            tech_features = calculate_technical_features(
                stock_weekly, index_weekly, config
            )

            # --- Fundamental filter ---
            fund_result = apply_fundamental_filter(fundamentals_row, config)

            # --- Technical filter ---
            tech_result = apply_technical_filter(tech_features, config)

            # --- Data completeness ---
            has_price_data = not stock_weekly.empty
            required_fields = (
                _BANK_REQUIRED_FIELDS
                if is_bank_sector(sector, config)
                else _NON_BANK_REQUIRED_FIELDS
            )
            completeness = calculate_data_completeness(
                fundamentals_row, required_fields, has_price_data
            )

            # --- Classify ---
            status = classify_stock(fund_result, tech_result, completeness, config)

            # --- Score ---
            entry = _build_result_entry(
                stock_row,
                ticker,
                tech_features,
                fund_result,
                tech_result,
                status,
                fundamentals_row,
                sector,
                config,
            )
            results.append(entry)

        except Exception as exc:
            logger.warning("Error processing ticker '%s': %s", ticker, exc)
            results.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "sector": sector,
                    "final_score": 0.0,
                    "status": "Avoid",
                    "fundamental_score": 0.0,
                    "earnings_momentum_score": 0.0,
                    "technical_score": 0.0,
                    "valuation_score": 0.0,
                    "relative_strength_score": 0.0,
                    "close": float("nan"),
                    "weekly_sma20": float("nan"),
                    "weekly_sma50": float("nan"),
                    "distance_from_sma20": float("nan"),
                    "relative_strength_13w": float("nan"),
                    "reasons": [f"Error: {exc}"],
                    "warnings": ["Unhandled error during processing"],
                    "missing_data_flags": ["processing_error"],
                }
            )

    # ---------------------------------------------------------------
    # 6. Generate report
    # ---------------------------------------------------------------
    logger.info("Generating report...")
    report_path = generate_report(results, report_dir, report_date)
    logger.info("Report written to %s", report_path)

    # ---------------------------------------------------------------
    # 7. Log summary
    # ---------------------------------------------------------------
    status_counts: dict[str, int] = {
        "Candidate": 0,
        "Watch": 0,
        "Speculative": 0,
        "Avoid": 0,
    }
    for r in results:
        s = r.get("status", "Avoid")
        if s in status_counts:
            status_counts[s] += 1

    logger.info(
        "Summary: %d Candidates, %d Watch, %d Speculative, %d Avoid",
        status_counts["Candidate"],
        status_counts["Watch"],
        status_counts["Speculative"],
        status_counts["Avoid"],
    )


if __name__ == "__main__":
    main()
