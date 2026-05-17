"""
Technical-only historical backtest for the LQ45 Stock Screener.

Answers: "When the technical filter passed historically, did those stocks
outperform IHSG over the next 4-13 weeks?"

Usage:
    python -m stock_screener.src.backtest
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from stock_screener.src.config import load_config
from stock_screener.src.data_io import fetch_index, fetch_prices, load_universe
from stock_screener.src.pipeline import (
    apply_technical_filter,
    calculate_technical_features,
    resample_to_weekly,
)

logger = logging.getLogger(__name__)


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


def load_backtest_inputs(
    config: dict,
    data_dir: str,
    cache_dir: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load universe, fetch prices (with extended history), and fetch index.

    Uses a separate ``backtest/`` subdirectory under *cache_dir* to avoid
    cache collisions with the main screener (which may have cached only
    18 months of data).

    Parameters
    ----------
    config : dict
        Validated configuration with a ``backtest`` section.
    data_dir : str
        Path to the data directory (contains universe CSV).
    cache_dir : str
        Path to the main cache directory.

    Returns
    -------
    tuple
        (universe_df, price_data_dict, index_daily_df)
    """
    universe_path = os.path.join(data_dir, "lq45_constituents.csv")
    if not os.path.exists(universe_path):
        raise FileNotFoundError(f"Universe CSV not found: {universe_path}")
    universe_df = load_universe(universe_path)
    tickers = universe_df["ticker"].tolist()

    months_override = config["backtest"]["history_months"]

    # Use separate cache subdir to avoid collision with screener's 18-month cache
    bt_cache_dir = os.path.join(cache_dir, "backtest")

    logger.info(
        "Fetching %d months of price data for %d tickers...",
        months_override,
        len(tickers),
    )
    price_data = fetch_prices(
        tickers, config, bt_cache_dir, months_override=months_override
    )
    index_data = fetch_index(config, bt_cache_dir, months_override=months_override)

    if index_data.empty:
        raise RuntimeError(
            f"Failed to fetch index data ({config['data']['index_ticker']})"
        )

    return universe_df, price_data, index_data


def get_backtest_dates(
    index_weekly: pd.DataFrame,
    min_warmup_weeks: int,
    horizons_weeks: list[int],
) -> list[pd.Timestamp]:
    """
    Generate valid backtest evaluation dates from the index weekly calendar.

    Skips the first ``min_warmup_weeks`` (need SMA50/RS history) and the
    last ``max(horizons_weeks)`` (need forward return data).

    Parameters
    ----------
    index_weekly : pd.DataFrame
        Weekly index OHLCV with a DatetimeIndex of Friday labels.
    min_warmup_weeks : int
        Number of initial weeks to skip for warmup.
    horizons_weeks : list[int]
        Forward return horizons in weeks (e.g., [4, 13]).

    Returns
    -------
    list[pd.Timestamp]
        Sorted list of valid evaluation dates (Fridays).
    """
    all_dates = index_weekly.index.tolist()

    if not horizons_weeks:
        logger.warning("horizons_weeks is empty - no evaluation dates possible")
        return []

    max_horizon = max(horizons_weeks)

    # Need at least warmup + max_horizon + 1 weeks of data
    min_required = min_warmup_weeks + max_horizon + 1
    if len(all_dates) < min_required:
        logger.warning(
            "Insufficient index data: %d weeks available, need at least %d",
            len(all_dates),
            min_required,
        )
        return []

    # Skip first min_warmup_weeks (need warmup history before first eval)
    # Skip last max_horizon weeks (need forward data after last eval)
    start_idx = min_warmup_weeks
    end_idx = len(all_dates) - max_horizon

    valid_dates = all_dates[start_idx:end_idx]
    logger.info(
        "Generated %d backtest dates from %s to %s",
        len(valid_dates),
        valid_dates[0].strftime("%Y-%m-%d") if valid_dates else "N/A",
        valid_dates[-1].strftime("%Y-%m-%d") if valid_dates else "N/A",
    )
    return valid_dates


def get_forward_return(
    weekly_df: pd.DataFrame,
    date: pd.Timestamp,
    horizon_weeks: int,
) -> float:
    """
    Calculate forward return from *date* to *date + horizon_weeks*.

    Uses label-based lookup on the DatetimeIndex. Returns NaN if the
    future bar is not available (delisting, gap, etc.).

    Because of label-based gap handling, the actual measurement window
    may be slightly longer than ``horizon_weeks`` when intermediate bars
    are missing (e.g., holidays, trading halts).

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly OHLCV with a DatetimeIndex.
    date : pd.Timestamp
        Evaluation date (must be in the index).
    horizon_weeks : int
        Forward horizon in weeks.

    Returns
    -------
    float
        Forward return as a decimal (e.g., 0.05 = 5%). NaN if unavailable.
    """
    # Target date: horizon_weeks Fridays after the evaluation date
    # Use pd.DateOffset(weeks=horizon_weeks) to find the approximate target
    # Then find the closest available date in the index
    target_date = date + pd.DateOffset(weeks=horizon_weeks)

    # Guard: date must exist in the index
    if date not in weekly_df.index:
        return float("nan")

    # Find the closest available date >= target_date
    future_dates = weekly_df.index[weekly_df.index >= target_date]
    if len(future_dates) == 0:
        return float("nan")

    future_date = future_dates[0]

    # Guard: reject if the actual gap exceeds the horizon by more than 2 weeks
    # (handles delisting, trading halts, extended data gaps)
    actual_days = (future_date - date).days
    expected_days = horizon_weeks * 7
    if actual_days > expected_days + 14:
        return float("nan")

    close_now = weekly_df.loc[date, "Close"]
    close_future = weekly_df.loc[future_date, "Close"]

    if pd.isna(close_now) or pd.isna(close_future) or abs(close_now) < 1e-8:
        return float("nan")

    return float(close_future / close_now - 1.0)


def _is_within_week_tolerance(date_a: pd.Timestamp, date_b: pd.Timestamp) -> bool:
    """Check if two dates are within 7 calendar days of each other.

    Uses strict ``< 7`` so that consecutive Fridays (exactly 7 days apart)
    are rejected — they represent different calendar weeks.
    """
    return abs((date_a - date_b).days) < 7


def evaluate_ticker_on_date(
    ticker: str,
    stock_row: pd.Series,
    stock_weekly: pd.DataFrame,
    index_weekly: pd.DataFrame,
    date: pd.Timestamp,
    horizons: list[int],
    config: dict,
) -> dict[str, Any] | None:
    """
    Evaluate a single ticker at a single evaluation date.

    Slices data up to *date*, calculates technical features, applies the
    technical filter, and computes forward returns for each horizon.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    stock_row : pd.Series
        Row from universe DataFrame (company_name, sector).
    stock_weekly : pd.DataFrame
        Full weekly OHLCV for this stock.
    index_weekly : pd.DataFrame
        Full weekly OHLCV for the index.
    date : pd.Timestamp
        Evaluation date (Friday from index calendar).
    horizons : list[int]
        Forward return horizons in weeks.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict or None
        Result dict with all columns for the trade-level CSV, or None if
        the observation should be skipped (insufficient data, date mismatch).

    Note
    ----
    Forward returns for both stock and index are measured from
    ``stock_last_date`` (the stock's last available bar) to ensure excess
    return is computed over the same calendar window. If ``stock_last_date``
    differs from the evaluation ``date``, this introduces a small timing
    offset (up to 6 days) in the signal-to-return measurement.
    """
    if stock_weekly.empty:
        return None

    # --- Strict same-week match ---
    # Find stock's last available weekly bar at or before the evaluation date
    stock_dates_at_or_before = stock_weekly.index[stock_weekly.index <= date]
    if len(stock_dates_at_or_before) == 0:
        return None  # Stock has no data on or before this date

    stock_last_date = stock_dates_at_or_before[-1]

    # Check week tolerance
    if not _is_within_week_tolerance(stock_last_date, date):
        return None  # Stock data is stale (>= 7 days before eval date)

    # --- Slice data up to evaluation date (inclusive) ---
    stock_slice = stock_weekly.loc[:stock_last_date]
    index_dates_at_or_before = index_weekly.index[index_weekly.index <= date]
    if len(index_dates_at_or_before) == 0:
        return None
    index_slice = index_weekly.loc[:index_dates_at_or_before[-1]]

    if stock_slice.empty or index_slice.empty:
        return None

    # --- Calculate technical features (uses .iloc[-1] on the slice) ---
    tech_features = calculate_technical_features(stock_slice, index_slice, config)

    # --- Apply technical filter ---
    tech_result = apply_technical_filter(tech_features, config)
    passed = tech_result["passes"]

    # --- Forward returns ---
    # Stock forward return starts from stock_last_date.
    # For the index, resolve the closest index bar <= stock_last_date to avoid
    # KeyError when stock_last_date is not in the index calendar (e.g., holidays).
    forward_returns = {}
    index_returns = {}
    for h in horizons:
        fr = get_forward_return(stock_weekly, stock_last_date, h)
        # Resolve index evaluation date
        idx_dates = index_weekly.index[index_weekly.index <= stock_last_date]
        if len(idx_dates) == 0:
            ir = float("nan")
        else:
            ir = get_forward_return(index_weekly, idx_dates[-1], h)
        forward_returns[h] = fr
        index_returns[h] = ir

    # --- Build result dict ---
    result = {
        "date": date,
        "ticker": ticker,
        "company_name": str(stock_row.get("company_name", "")),
        "sector": str(stock_row.get("sector", "")),
        "passed_technical": passed,
        "close": tech_features.get("close", float("nan")),
        "sma20": tech_features.get("sma_short", float("nan")),
        "sma50": tech_features.get("sma_long", float("nan")),
        "distance_from_sma20": tech_features.get("distance_from_sma20", float("nan")),
        "relative_strength_13w": tech_features.get(
            "relative_strength_13w", float("nan")
        ),
        "sma20_is_rising": tech_features.get("sma_short_is_rising", False),
        "reasons": ";".join(tech_result.get("reasons", [])),
        "warnings": ";".join(tech_result.get("warnings", [])),
    }

    for h in horizons:
        fr = forward_returns[h]
        ir = index_returns[h]
        result[f"forward_return_{h}w"] = fr
        result[f"index_return_{h}w"] = ir
        if pd.isna(fr) or pd.isna(ir):
            result[f"excess_return_{h}w"] = float("nan")
        else:
            result[f"excess_return_{h}w"] = fr - ir

    return result


def generate_backtest_reports(
    trades_df: pd.DataFrame,
    report_dir: str,
    horizons: list[int],
) -> tuple[str, str]:
    """
    Generate trade-level and summary CSV reports.

    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with one row per ticker-date observation.
    report_dir : str
        Directory to write reports into (created if missing).
    horizons : list[int]
        Forward return horizons in weeks.

    Returns
    -------
    tuple
        (trades_csv_path, summary_csv_path)
    """
    os.makedirs(report_dir, exist_ok=True)

    # --- Trade-level CSV ---
    trades_path = os.path.join(report_dir, "backtest_technical_trades.csv")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8")
    logger.info("Trade-level report saved to %s", trades_path)

    # --- Summary CSV ---
    # Guard against empty DataFrame (no columns -> KeyError on "passed_technical")
    if trades_df.empty or "passed_technical" not in trades_df.columns:
        logger.warning("Empty trades DataFrame - writing empty summary")
        summary_df = pd.DataFrame(columns=[
            "horizon_weeks", "passed_technical", "n_observations",
            "n_valid_returns", "avg_forward_return", "median_forward_return",
            "win_rate", "avg_index_return", "avg_excess_return", "excess_win_rate",
        ])
        summary_path = os.path.join(report_dir, "backtest_technical_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")
        logger.info("Summary report saved to %s", summary_path)
        return os.path.abspath(trades_path), os.path.abspath(summary_path)

    summary_rows = []
    for h in horizons:
        for passed_val in [True, False]:
            group = trades_df[trades_df["passed_technical"] == passed_val]
            fr_col = f"forward_return_{h}w"
            ir_col = f"index_return_{h}w"
            er_col = f"excess_return_{h}w"

            fr_valid = group[fr_col].dropna()
            ir_valid = group[ir_col].dropna()
            er_valid = group[er_col].dropna()

            n_obs = len(group)
            n_valid = len(fr_valid)

            summary_rows.append(
                {
                    "horizon_weeks": h,
                    "passed_technical": passed_val,
                    "n_observations": n_obs,
                    "n_valid_returns": n_valid,
                    "avg_forward_return": (
                        float(fr_valid.mean()) if n_valid > 0 else float("nan")
                    ),
                    "median_forward_return": (
                        float(fr_valid.median()) if n_valid > 0 else float("nan")
                    ),
                    "win_rate": (
                        float((fr_valid > 0).mean()) if n_valid > 0 else float("nan")
                    ),
                    "avg_index_return": (
                        float(ir_valid.mean()) if len(ir_valid) > 0 else float("nan")
                    ),
                    "avg_excess_return": (
                        float(er_valid.mean()) if len(er_valid) > 0 else float("nan")
                    ),
                    "excess_win_rate": (
                        float((er_valid > 0).mean())
                        if len(er_valid) > 0
                        else float("nan")
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(report_dir, "backtest_technical_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info("Summary report saved to %s", summary_path)

    return os.path.abspath(trades_path), os.path.abspath(summary_path)


def run_technical_backtest() -> None:
    """
    Run the full technical-only historical backtest.

    Orchestrates: config loading, data fetching, date generation,
    per-ticker-per-date evaluation, and report export.
    """
    script_dir = _get_script_dir()
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    config_path = os.path.join(project_root, "config.yaml")

    # --- Load config ---
    try:
        config = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        logging.error("Failed to load config: %s", exc)
        sys.exit(1)

    _setup_logging(config)
    logger.info("Starting Technical-Only Historical Backtest")
    start_time = time.monotonic()

    # --- Warn about survivorship bias ---
    logger.warning(
        "SURVIVORSHIP BIAS: Using current LQ45 constituents for historical "
        "backtest. Stocks delisted or removed from LQ45 during the backtest "
        "period are excluded. Results represent a best-case scenario."
    )

    # --- Determine paths ---
    data_dir = os.path.join(project_root, "data")
    report_dir = os.path.join(project_root, "reports")
    cache_dir = os.path.join(data_dir, "cache")

    # --- Load inputs ---
    universe_df, price_data, index_data = load_backtest_inputs(
        config, data_dir, cache_dir
    )
    tickers = universe_df["ticker"].tolist()

    # --- Resample to weekly ---
    logger.info("Resampling price data to weekly...")
    index_weekly = resample_to_weekly(index_data)
    if index_weekly.empty:
        logger.error("Index weekly data is empty after resampling. Exiting.")
        sys.exit(1)

    stock_weekly_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = price_data.get(ticker, pd.DataFrame())
        stock_weekly_data[ticker] = resample_to_weekly(df)

    # --- Generate backtest dates ---
    bt_cfg = config["backtest"]
    eval_dates = get_backtest_dates(
        index_weekly,
        bt_cfg["min_warmup_weeks"],
        bt_cfg["horizons_weeks"],
    )
    if not eval_dates:
        logger.error("No valid backtest dates. Exiting.")
        sys.exit(1)

    horizons = bt_cfg["horizons_weeks"]

    # --- Evaluate each ticker at each date ---
    logger.info(
        "Evaluating %d tickers across %d dates...", len(tickers), len(eval_dates)
    )
    all_results: list[dict[str, Any]] = []

    for date in eval_dates:
        for _, stock_row in universe_df.iterrows():
            ticker = stock_row["ticker"]
            stock_weekly = stock_weekly_data.get(ticker, pd.DataFrame())

            try:
                result = evaluate_ticker_on_date(
                    ticker,
                    stock_row,
                    stock_weekly,
                    index_weekly,
                    date,
                    horizons,
                    config,
                )
                if result is not None:
                    all_results.append(result)
            except Exception as exc:
                logger.warning(
                    "Error evaluating %s on %s: %s", ticker, date.strftime("%Y-%m-%d"), exc
                )

    if not all_results:
        logger.warning("No valid observations generated. Exiting.")
        sys.exit(1)

    trades_df = pd.DataFrame(all_results)
    logger.info("Generated %d observations", len(trades_df))

    # --- Generate reports ---
    trades_path, summary_path = generate_backtest_reports(
        trades_df, report_dir, horizons
    )

    # --- Log summary ---
    logger.info("Backtest complete.")
    logger.info("Trade-level CSV: %s", trades_path)
    logger.info("Summary CSV: %s", summary_path)

    # Log key summary stats
    for h in horizons:
        passed = trades_df[trades_df["passed_technical"] == True]
        failed = trades_df[trades_df["passed_technical"] == False]
        er_col = f"excess_return_{h}w"

        passed_er = passed[er_col].dropna()
        failed_er = failed[er_col].dropna()

        logger.info(
            "Horizon %dw: passed=%d (avg_excess=%.4f), failed=%d (avg_excess=%.4f)",
            h,
            len(passed),
            float(passed_er.mean()) if len(passed_er) > 0 else float("nan"),
            len(failed),
            float(failed_er.mean()) if len(failed_er) > 0 else float("nan"),
        )

    elapsed = time.monotonic() - start_time
    logger.info("Total evaluation time: %.2f seconds", elapsed)


if __name__ == "__main__":
    run_technical_backtest()
