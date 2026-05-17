"""
Backtest diagnostics for the LQ45 Stock Screener.

Reads backtest_technical_trades.csv produced by the technical backtest module
and produces diagnostic CSVs and a text-based summary with diverging ASCII bar
charts. The diagnostics explain why the technical filter underperformed the
IHSG benchmark, especially at the 13-week forward return horizon.

Usage:
    python -m stock_screener.src.backtest_diagnostics [--trades-csv PATH] [--output-dir PATH] [--top-n N]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Encoding-safe bar characters
# ---------------------------------------------------------------------------

# Detect whether the console supports Unicode block characters (U+2580+).
# Windows cp1252 cannot encode them, so we fall back to ASCII-safe chars.
try:
    _test_char = "\u2588"
    _test_char.encode(sys.stdout.encoding or "utf-8")
    _BAR_POS = "\u2588"   # full block (right-extending bar)
    _BAR_NEG = "\u2584"   # lower half block (left-extending bar)
    _BAR_BG = "\u2591"    # light shade (background)
except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
    _BAR_POS = "#"
    _BAR_NEG = "@"
    _BAR_BG = "."

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = frozenset({
    "date", "ticker", "passed_technical",
    "forward_return_4w", "forward_return_13w",
    "index_return_4w", "index_return_13w",
    "excess_return_4w", "excess_return_13w",
})

# Signal bucket boundaries (explicit, domain-specific thresholds)
_DISTANCE_BUCKETS = [-np.inf, -0.05, 0.0, 0.05, 0.10, 0.15, np.inf]
_DISTANCE_LABELS = ["<=-5%", "(-5%,0%]", "(0%,5%]", "(5%,10%]", "(10%,15%]", ">15%"]

_RS_BUCKETS = [-np.inf, 0.0, 0.05, 0.10, np.inf]
_RS_LABELS = ["<=0%", "(0%,5%]", "(5%,10%]", ">10%"]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_mean(series: pd.Series) -> float:
    """Return mean of non-NaN values, or NaN if all NaN."""
    valid = series.dropna()
    return float(valid.mean()) if len(valid) > 0 else float("nan")


def _safe_median(series: pd.Series) -> float:
    """Return median of non-NaN values, or NaN if all NaN."""
    valid = series.dropna()
    return float(valid.median()) if len(valid) > 0 else float("nan")


def _safe_std(series: pd.Series) -> float:
    """Return std of non-NaN values, or NaN if fewer than 2 values."""
    valid = series.dropna()
    return float(valid.std()) if len(valid) > 1 else float("nan")


def _safe_win_rate(series: pd.Series) -> float:
    """Return fraction of positive values among non-NaN, or NaN if none."""
    valid = series.dropna()
    return float((valid > 0).mean()) if len(valid) > 0 else float("nan")


def _categorize_reason(reason: str) -> str:
    """Replace numeric values with a generic placeholder for categorization.

    This strips variable-specific numbers from templated reason strings
    so that semantically identical reasons (e.g. 'Distance from SMA20=0.0010'
    and 'Distance from SMA20=0.1500') are grouped together, while preserving
    identifiers like SMA20 (not mangled to SMAX).

    Parameters
    ----------
    reason : str
        A single PASS/FAIL reason string, e.g.
        ``"PASS: |Distance from SMA20|=0.0010 (< 0.1500)"``.

    Returns
    -------
    str
        The reason with numeric values replaced by ``'X'``.
    """
    # Replace numeric values after = signs (e.g. Close=9500.0 -> Close=X)
    result = re.sub(r"=(\d+\.?\d*)", "=X", reason)
    # Replace numeric ranges in parentheses (e.g. (>= 0.1500) -> (...))
    result = re.sub(r"\([^)]*\d[^)]*\)", "(...)", result)
    # Replace standalone decimal numbers, but NOT digits embedded in
    # identifiers like SMA20 (lookbehind prevents matching after [A-Za-z_]).
    result = re.sub(r"(?<![A-Za-z_])\d+\.\d+(?![A-Za-z_])", "X", result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the diagnostics module.

    Supports overriding the input trades CSV path, output directory,
    and the number of extreme observations to include in the report.

    Parameters
    ----------
    argv : list[str] | None
        Argument list. Uses sys.argv[1:] if None.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: trades_csv, output_dir, top_n.
    """
    parser = argparse.ArgumentParser(
        description="Backtest diagnostics for the LQ45 Stock Screener."
    )
    parser.add_argument(
        "--trades-csv",
        type=str,
        default=None,
        help="Path to backtest_technical_trades.csv (default: auto-detect in reports/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for diagnostic CSV output (default: reports/)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of best/worst observations to report (default: 50)",
    )
    args = parser.parse_args(argv)

    # Auto-detect defaults relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    if args.trades_csv is None:
        args.trades_csv = os.path.join(project_root, "reports", "backtest_technical_trades.csv")
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, "reports")

    return args


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_trades_csv(path: str) -> pd.DataFrame:
    """Load and validate the trade-level CSV from the technical backtest.

    Reads the CSV file, validates that all required columns are present,
    converts string-encoded booleans to Python bool type, and ensures the
    DataFrame is non-empty. Optional columns are not required -- analyses
    that depend on them will be skipped with a warning if absent.

    Parameters
    ----------
    path : str
        Path to backtest_technical_trades.csv.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with correct dtypes:
        - passed_technical: bool (converted from "True"/"False" string)
        - sma20_is_rising: bool (converted from "True"/"False" string, if present)
        - date: str (kept as string for label-based operations)
        - numeric columns: float64

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If required columns are missing or CSV is empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trades CSV not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Trades CSV is empty: {path}")

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )

    # Convert string booleans to Python bool.
    # Pandas sometimes auto-converts "True"/"False" in CSV to Python bool;
    # only map when the column is still an object (string) dtype.
    if df["passed_technical"].dtype != bool:
        df["passed_technical"] = df["passed_technical"].map({"True": True, "False": False})
    if "sma20_is_rising" in df.columns and df["sma20_is_rising"].dtype != bool:
        df["sma20_is_rising"] = df["sma20_is_rising"].map({"True": True, "False": False})

    return df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyze_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker breakdown of excess returns, pass rates, and distribution shape.

    Groups observations by ticker and computes pass rate, average excess return
    for passed and failed stocks at both horizons, win rates, and distribution
    statistics (std, median, avg win magnitude vs avg loss magnitude) to detect
    loss asymmetry that averages alone would hide.

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level DataFrame with boolean passed_technical column.

    Returns
    -------
    pd.DataFrame
        One row per ticker with columns:
        ticker, n_observations, n_passed, n_failed, pass_rate,
        avg_excess_4w_passed, avg_excess_4w_failed,
        avg_excess_13w_passed, avg_excess_13w_failed,
        median_excess_13w_passed, std_excess_13w_passed,
        avg_win_13w_passed, avg_loss_13w_passed, win_loss_ratio_13w_passed,
        win_rate_4w_passed, win_rate_4w_failed,
        win_rate_13w_passed, win_rate_13w_failed,
        n_valid_13w_passed.
    """
    rows: list[dict[str, Any]] = []
    for ticker, group in df.groupby("ticker"):
        n_obs = len(group)
        passed = group[group["passed_technical"] == True]
        failed = group[group["passed_technical"] == False]
        n_passed = len(passed)
        n_failed = len(failed)
        pass_rate = n_passed / n_obs if n_obs > 0 else 0.0

        # Excess return averages
        avg_excess_4w_passed = _safe_mean(passed["excess_return_4w"])
        avg_excess_4w_failed = _safe_mean(failed["excess_return_4w"])
        avg_excess_13w_passed = _safe_mean(passed["excess_return_13w"])
        avg_excess_13w_failed = _safe_mean(failed["excess_return_13w"])

        # Distribution stats for passed 13w
        median_excess_13w_passed = _safe_median(passed["excess_return_13w"])
        std_excess_13w_passed = _safe_std(passed["excess_return_13w"])

        # Win/loss magnitudes for passed 13w
        passed_13w = passed["excess_return_13w"].dropna()
        wins = passed_13w[passed_13w > 0]
        losses = passed_13w[passed_13w < 0]
        avg_win = float(wins.mean()) if len(wins) > 0 else float("nan")
        avg_loss = float(losses.mean()) if len(losses) > 0 else float("nan")
        if not np.isnan(avg_loss) and abs(avg_loss) > 1e-12:
            win_loss_ratio = abs(avg_win / avg_loss) if not np.isnan(avg_win) else float("nan")
        else:
            win_loss_ratio = float("nan")

        # Win rates
        win_rate_4w_passed = _safe_win_rate(passed["excess_return_4w"])
        win_rate_4w_failed = _safe_win_rate(failed["excess_return_4w"])
        win_rate_13w_passed = _safe_win_rate(passed["excess_return_13w"])
        win_rate_13w_failed = _safe_win_rate(failed["excess_return_13w"])

        # Valid count
        n_valid_13w_passed = int(passed["excess_return_13w"].notna().sum())

        rows.append({
            "ticker": ticker,
            "n_observations": n_obs,
            "n_passed": n_passed,
            "n_failed": n_failed,
            "pass_rate": pass_rate,
            "avg_excess_4w_passed": avg_excess_4w_passed,
            "avg_excess_4w_failed": avg_excess_4w_failed,
            "avg_excess_13w_passed": avg_excess_13w_passed,
            "avg_excess_13w_failed": avg_excess_13w_failed,
            "median_excess_13w_passed": median_excess_13w_passed,
            "std_excess_13w_passed": std_excess_13w_passed,
            "avg_win_13w_passed": avg_win,
            "avg_loss_13w_passed": avg_loss,
            "win_loss_ratio_13w_passed": win_loss_ratio,
            "win_rate_4w_passed": win_rate_4w_passed,
            "win_rate_4w_failed": win_rate_4w_failed,
            "win_rate_13w_passed": win_rate_13w_passed,
            "win_rate_13w_failed": win_rate_13w_failed,
            "n_valid_13w_passed": n_valid_13w_passed,
        })

    return pd.DataFrame(rows)


def analyze_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year breakdown of excess returns, pass rates, and market regime.

    Groups observations by calendar year (extracted from date column) and
    computes the same metrics as analyze_by_ticker. Market regime is classified
    based on the average index return (not the filter's excess return) to avoid
    circular reasoning: "bull" year if avg index_return_13w > 0, "bear" otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level DataFrame with boolean passed_technical column.

    Returns
    -------
    pd.DataFrame
        One row per year with columns:
        year, n_observations, n_passed, n_failed, pass_rate,
        avg_excess_4w_passed, avg_excess_4w_failed,
        avg_excess_13w_passed, avg_excess_13w_failed,
        median_excess_13w_passed, std_excess_13w_passed,
        avg_win_13w_passed, avg_loss_13w_passed, win_loss_ratio_13w_passed,
        win_rate_4w_passed, win_rate_4w_failed,
        win_rate_13w_passed, win_rate_13w_failed,
        n_valid_13w_passed,
        avg_index_return_13w, market_regime.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    rows: list[dict[str, Any]] = []
    for year_val, group in df.groupby("year"):
        n_obs = len(group)
        passed = group[group["passed_technical"] == True]
        failed = group[group["passed_technical"] == False]
        n_passed = len(passed)
        n_failed = len(failed)
        pass_rate = n_passed / n_obs if n_obs > 0 else 0.0

        avg_excess_4w_passed = _safe_mean(passed["excess_return_4w"])
        avg_excess_4w_failed = _safe_mean(failed["excess_return_4w"])
        avg_excess_13w_passed = _safe_mean(passed["excess_return_13w"])
        avg_excess_13w_failed = _safe_mean(failed["excess_return_13w"])

        median_excess_13w_passed = _safe_median(passed["excess_return_13w"])
        std_excess_13w_passed = _safe_std(passed["excess_return_13w"])

        passed_13w = passed["excess_return_13w"].dropna()
        wins = passed_13w[passed_13w > 0]
        losses = passed_13w[passed_13w < 0]
        avg_win = float(wins.mean()) if len(wins) > 0 else float("nan")
        avg_loss = float(losses.mean()) if len(losses) > 0 else float("nan")
        if not np.isnan(avg_loss) and abs(avg_loss) > 1e-12:
            win_loss_ratio = abs(avg_win / avg_loss) if not np.isnan(avg_win) else float("nan")
        else:
            win_loss_ratio = float("nan")

        win_rate_4w_passed = _safe_win_rate(passed["excess_return_4w"])
        win_rate_4w_failed = _safe_win_rate(failed["excess_return_4w"])
        win_rate_13w_passed = _safe_win_rate(passed["excess_return_13w"])
        win_rate_13w_failed = _safe_win_rate(failed["excess_return_13w"])

        n_valid_13w_passed = int(passed["excess_return_13w"].notna().sum())

        # Market regime based on avg index return
        avg_index_return_13w = _safe_mean(group["index_return_13w"])
        market_regime = "bull" if avg_index_return_13w > 0 else "bear"

        rows.append({
            "year": int(year_val),
            "n_observations": n_obs,
            "n_passed": n_passed,
            "n_failed": n_failed,
            "pass_rate": pass_rate,
            "avg_excess_4w_passed": avg_excess_4w_passed,
            "avg_excess_4w_failed": avg_excess_4w_failed,
            "avg_excess_13w_passed": avg_excess_13w_passed,
            "avg_excess_13w_failed": avg_excess_13w_failed,
            "median_excess_13w_passed": median_excess_13w_passed,
            "std_excess_13w_passed": std_excess_13w_passed,
            "avg_win_13w_passed": avg_win,
            "avg_loss_13w_passed": avg_loss,
            "win_loss_ratio_13w_passed": win_loss_ratio,
            "win_rate_4w_passed": win_rate_4w_passed,
            "win_rate_4w_failed": win_rate_4w_failed,
            "win_rate_13w_passed": win_rate_13w_passed,
            "win_rate_13w_failed": win_rate_13w_failed,
            "n_valid_13w_passed": n_valid_13w_passed,
            "avg_index_return_13w": avg_index_return_13w,
            "market_regime": market_regime,
        })

    return pd.DataFrame(rows)


def analyze_by_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Per-signal-profile breakdown of excess returns using explicit bucket boundaries.

    Buckets distance_from_sma20 and relative_strength_13w using predefined
    thresholds (see _DISTANCE_BUCKETS, _RS_BUCKETS), splits by sma20_is_rising,
    and computes avg excess return per bucket. This reveals which signal
    combinations are predictive and which are not.

    Bucket boundaries:
        distance_from_sma20: [-inf, -5%), [-5%, 0%), [0%, 5%), [5%, 10%), [10%, 15%), [15%, inf)
        relative_strength_13w: [-inf, 0%), [0%, 5%), [5%, 10%), [10%, inf)

    NaN values in signal columns are excluded from bucketing (dropped before groupby).

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level DataFrame with boolean passed_technical column.

    Returns
    -------
    pd.DataFrame
        One row per signal profile with columns:
        distance_bucket, rs_bucket, sma20_is_rising, n_observations, n_passed,
        avg_excess_4w, avg_excess_13w, win_rate_4w, win_rate_13w.

    Returns empty DataFrame if optional columns (distance_from_sma20,
    relative_strength_13w, sma20_is_rising) are missing.
    """
    _EMPTY_COLUMNS = [
        "distance_bucket", "rs_bucket", "sma20_is_rising",
        "n_observations", "n_passed",
        "avg_excess_4w", "avg_excess_13w",
        "win_rate_4w", "win_rate_13w",
    ]

    if not {"distance_from_sma20", "relative_strength_13w", "sma20_is_rising"}.issubset(df.columns):
        logger.warning("Optional columns for signal analysis not found. Skipping.")
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    sub = df.dropna(subset=["distance_from_sma20", "relative_strength_13w", "sma20_is_rising"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    sub["distance_bucket"] = pd.cut(
        sub["distance_from_sma20"],
        bins=_DISTANCE_BUCKETS,
        labels=_DISTANCE_LABELS,
    )
    sub["rs_bucket"] = pd.cut(
        sub["relative_strength_13w"],
        bins=_RS_BUCKETS,
        labels=_RS_LABELS,
    )

    grouped = sub.groupby(["distance_bucket", "rs_bucket", "sma20_is_rising"], observed=True)

    result = grouped.agg(
        n_observations=("passed_technical", "count"),
        n_passed=("passed_technical", "sum"),
        avg_excess_4w=("excess_return_4w", _safe_mean),
        avg_excess_13w=("excess_return_13w", _safe_mean),
        win_rate_4w=("excess_return_4w", _safe_win_rate),
        win_rate_13w=("excess_return_13w", _safe_win_rate),
    ).reset_index()

    return result


def analyze_4w_vs_13w(df: pd.DataFrame) -> pd.DataFrame:
    """Horizon reversal analysis for passed stocks.

    Cross-tabulates 4-week excess (positive/negative) vs 13-week excess
    (positive/negative) to show how often 4-week winners become 13-week
    losers and vice versa. This directly tests whether the filter's
    underperformance at 13w is due to mean reversion after initial gains.

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level DataFrame with boolean passed_technical column.

    Returns
    -------
    pd.DataFrame
        2x2 contingency table with columns:
        excess_4w, n_4w_pos_13w_pos, n_4w_pos_13w_neg,
        n_4w_neg_13w_pos, n_4w_neg_13w_neg,
        pct_4w_pos_13w_neg, pct_4w_neg_13w_pos.

    Only includes passed_technical=True observations with non-NaN returns
    at both horizons.
    """
    passed = df[df["passed_technical"] == True].copy()
    valid = passed.dropna(subset=["excess_return_4w", "excess_return_13w"])

    if valid.empty:
        return pd.DataFrame(columns=[
            "excess_4w", "n_4w_pos_13w_pos", "n_4w_pos_13w_neg",
            "n_4w_neg_13w_pos", "n_4w_neg_13w_neg",
            "pct_4w_pos_13w_neg", "pct_4w_neg_13w_pos",
        ])

    mask_4w_pos = valid["excess_return_4w"] >= 0
    mask_4w_neg = valid["excess_return_4w"] < 0
    mask_13w_pos = valid["excess_return_13w"] >= 0
    mask_13w_neg = valid["excess_return_13w"] < 0

    n_4w_pos_13w_pos = int((mask_4w_pos & mask_13w_pos).sum())
    n_4w_pos_13w_neg = int((mask_4w_pos & mask_13w_neg).sum())
    n_4w_neg_13w_pos = int((mask_4w_neg & mask_13w_pos).sum())
    n_4w_neg_13w_neg = int((mask_4w_neg & mask_13w_neg).sum())

    total_4w_pos = n_4w_pos_13w_pos + n_4w_pos_13w_neg
    total_4w_neg = n_4w_neg_13w_pos + n_4w_neg_13w_neg

    pct_4w_pos_13w_neg = (
        (n_4w_pos_13w_neg / total_4w_pos * 100) if total_4w_pos > 0 else float("nan")
    )
    pct_4w_neg_13w_pos = (
        (n_4w_neg_13w_pos / total_4w_neg * 100) if total_4w_neg > 0 else float("nan")
    )

    result = pd.DataFrame([{
        "excess_4w": "all",
        "n_4w_pos_13w_pos": n_4w_pos_13w_pos,
        "n_4w_pos_13w_neg": n_4w_pos_13w_neg,
        "n_4w_neg_13w_pos": n_4w_neg_13w_pos,
        "n_4w_neg_13w_neg": n_4w_neg_13w_neg,
        "pct_4w_pos_13w_neg": pct_4w_pos_13w_neg,
        "pct_4w_neg_13w_pos": pct_4w_neg_13w_pos,
    }])

    return result


def analyze_extreme_observations(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Best and worst 13-week excess return observations for passed stocks.

    Returns both the top_n worst and top_n best observations stacked in a
    single DataFrame so the user can compare what the filter got wrong vs
    what it got right. This is essential for understanding whether the
    filter's logic is systematically wrong or just noisy.

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level DataFrame with boolean passed_technical column.
    top_n : int
        Number of best and worst observations to return.

    Returns
    -------
    pd.DataFrame
        2*top_n rows sorted by excess_return_13w ascending, with columns:
        date, ticker, company_name, sector, close, distance_from_sma20,
        relative_strength_13w, sma20_is_rising, forward_return_13w,
        index_return_13w, excess_return_13w,
        forward_return_4w, excess_return_4w,
        rank_label ("best" or "worst").

    Only includes passed_technical=True observations.
    """
    passed = df[df["passed_technical"] == True].copy()
    passed = passed.dropna(subset=["excess_return_13w"])

    if passed.empty:
        return pd.DataFrame(columns=[
            "date", "ticker", "company_name", "sector", "close",
            "distance_from_sma20", "relative_strength_13w", "sma20_is_rising",
            "forward_return_13w", "index_return_13w", "excess_return_13w",
            "forward_return_4w", "excess_return_4w", "rank_label",
        ])

    n_available = len(passed)

    # Edge case: not enough observations or top_n <= 0
    if n_available < 2 or top_n <= 0:
        return pd.DataFrame(columns=[
            "date", "ticker", "company_name", "sector", "close",
            "distance_from_sma20", "relative_strength_13w", "sma20_is_rising",
            "forward_return_13w", "index_return_13w", "excess_return_13w",
            "forward_return_4w", "excess_return_4w", "rank_label",
        ])

    # Select only columns that exist
    col_candidates = [
        "date", "ticker", "company_name", "sector", "close",
        "distance_from_sma20", "relative_strength_13w", "sma20_is_rising",
        "forward_return_13w", "index_return_13w", "excess_return_13w",
        "forward_return_4w", "excess_return_4w",
    ]
    available = [c for c in col_candidates if c in passed.columns]

    n = min(top_n, n_available)

    worst = passed.nsmallest(n, "excess_return_13w")[available].copy()
    worst["rank_label"] = "worst"
    best = passed.nlargest(n, "excess_return_13w")[available].copy()
    best["rank_label"] = "best"

    # Remove any overlap (same row appearing in both worst and best)
    best = best[~best.index.isin(worst.index)]

    result = pd.concat([worst, best], ignore_index=True)
    result = result.sort_values("excess_return_13w").reset_index(drop=True)
    return result


def analyze_failure_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency analysis of filter failure reasons and warnings.

    Parses the semicolon-delimited `reasons` column to extract individual
    PASS/FAIL conditions, categorizes them by stripping numeric values,
    then computes frequency counts and cross-tabulates with forward returns.
    This reveals which filter conditions are most commonly violated and
    whether certain failure reasons correlate with subsequent returns.

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level DataFrame with boolean passed_technical column.

    Returns
    -------
    pd.DataFrame
        One row per reason category with columns:
        reason_category, n_occurrences, pct_of_total,
        avg_excess_13w_when_passed, avg_excess_13w_when_failed.

    Returns empty DataFrame if `reasons` column is missing.
    """
    _EMPTY_COLUMNS = [
        "reason_category", "n_occurrences", "pct_of_total",
        "avg_excess_13w_when_passed", "avg_excess_13w_when_failed",
    ]

    if "reasons" not in df.columns:
        logger.warning("'reasons' column not found. Skipping failure reasons analysis.")
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    has_reasons = df["reasons"].notna() & (df["reasons"] != "")
    if not has_reasons.any():
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    # Split and explode reasons
    exploded = df[has_reasons].copy()
    exploded["reason"] = exploded["reasons"].str.split(";")
    exploded = exploded.explode("reason")
    exploded["reason"] = exploded["reason"].str.strip()

    # Categorize reasons by stripping numeric values
    exploded["reason_category"] = exploded["reason"].apply(_categorize_reason)

    total_occurrences = len(exploded)

    # Build result per reason category
    rows: list[dict[str, Any]] = []
    for cat_val, group in exploded.groupby("reason_category"):
        n_occ = len(group)
        pct = (n_occ / total_occurrences) * 100.0

        passed_group = group[group["passed_technical"] == True]
        failed_group = group[group["passed_technical"] == False]

        avg_passed = (
            float(passed_group["excess_return_13w"].dropna().mean())
            if len(passed_group) > 0
            else float("nan")
        )
        avg_failed = (
            float(failed_group["excess_return_13w"].dropna().mean())
            if len(failed_group) > 0
            else float("nan")
        )

        rows.append({
            "reason_category": cat_val,
            "n_occurrences": n_occ,
            "pct_of_total": pct,
            "avg_excess_13w_when_passed": avg_passed,
            "avg_excess_13w_when_failed": avg_failed,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("n_occurrences", ascending=False).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def _round_numerics(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round all numeric columns in a DataFrame to *decimals* places."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    out = df.copy()
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def write_diagnostics(
    diag_by_ticker: pd.DataFrame,
    diag_by_year: pd.DataFrame,
    diag_by_signal: pd.DataFrame,
    diag_4w_vs_13w: pd.DataFrame,
    diag_extreme: pd.DataFrame,
    diag_failure: pd.DataFrame,
    output_dir: str,
) -> dict[str, str]:
    """Write pre-computed diagnostic DataFrames to CSV files.

    Creates the output directory if it does not exist. Rounds all numeric
    columns to 4 decimal places centrally (before writing), and saves each
    result as a CSV file.

    Parameters
    ----------
    diag_by_ticker : pd.DataFrame
        Per-ticker analysis results.
    diag_by_year : pd.DataFrame
        Per-year analysis results.
    diag_by_signal : pd.DataFrame
        Per-signal-profile analysis results.
    diag_4w_vs_13w : pd.DataFrame
        Horizon reversal analysis results.
    diag_extreme : pd.DataFrame
        Extreme observations report.
    diag_failure : pd.DataFrame
        Failure reason frequency table.
    output_dir : str
        Directory to write diagnostic CSVs. Created if it does not exist.

    Returns
    -------
    dict[str, str]
        Mapping of analysis name -> output file path.
        Keys: "by_ticker", "by_year", "by_signal", "4w_vs_13w",
              "extreme_observations", "failure_reasons".
    """
    os.makedirs(output_dir, exist_ok=True)

    # Round all numeric columns centrally before writing
    diag_by_ticker = _round_numerics(diag_by_ticker)
    diag_by_year = _round_numerics(diag_by_year)
    diag_by_signal = _round_numerics(diag_by_signal)
    diag_4w_vs_13w = _round_numerics(diag_4w_vs_13w)
    diag_extreme = _round_numerics(diag_extreme)
    diag_failure = _round_numerics(diag_failure)

    # Write CSVs
    file_map = {
        "by_ticker": os.path.join(output_dir, "backtest_diag_by_ticker.csv"),
        "by_year": os.path.join(output_dir, "backtest_diag_by_year.csv"),
        "by_signal": os.path.join(output_dir, "backtest_diag_by_signal.csv"),
        "4w_vs_13w": os.path.join(output_dir, "backtest_diag_4w_vs_13w.csv"),
        "extreme_observations": os.path.join(output_dir, "backtest_diag_extreme_observations.csv"),
        "failure_reasons": os.path.join(output_dir, "backtest_diag_failure_reasons.csv"),
    }

    for key, df_out in [
        ("by_ticker", diag_by_ticker),
        ("by_year", diag_by_year),
        ("by_signal", diag_by_signal),
        ("4w_vs_13w", diag_4w_vs_13w),
        ("extreme_observations", diag_extreme),
        ("failure_reasons", diag_failure),
    ]:
        path = file_map[key]
        df_out.to_csv(path, index=False, encoding="utf-8")
        logger.info("Diagnostic CSV saved: %s", path)

    return file_map


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _make_diverging_bar(value: float, total_width: int = 40, scale: float = 200.0) -> str:
    """Create a diverging bar display with background fill.

    Returns a *total_width*-character string with the active portion
    (positive = \\u2588, negative = \\u2584) and background (\\u2591)
    centered around the midpoint. NaN or zero values produce full
    background fill.

    Parameters
    ----------
    value : float
        Numeric value to represent.
    total_width : int
        Total width of the bar display in characters.
    scale : float
        Multiplier to convert value to bar width.

    Returns
    -------
    str
        Diverging bar display string.
    """
    half = total_width // 2
    if pd.isna(value) or abs(value) < 1e-12:
        return _BAR_BG * total_width
    n = min(int(abs(value) * scale), half)
    if n <= 0:
        return _BAR_BG * total_width
    if value > 0:
        return _BAR_BG * half + _BAR_POS * n + _BAR_BG * (half - n)
    else:
        return _BAR_BG * (half - n) + _BAR_NEG * n + _BAR_BG * half


def print_summary(
    df: pd.DataFrame,
    diag_by_ticker: pd.DataFrame,
    diag_by_year: pd.DataFrame,
    diag_by_signal: pd.DataFrame,
    diag_4w_vs_13w: pd.DataFrame,
    diag_extreme: pd.DataFrame,
    diag_failure_reasons: pd.DataFrame,
) -> None:
    """Print text-based summary with diverging ASCII bar charts to console.

    Output sections (in order):
        1. Headline summary: overall pass rate, avg excess (4w/13w) for passed/failed,
           valid-return counts, NaN rate.
        2. Per-ticker breakdown with diverging bars for avg_excess_13w_passed.
        3. Per-year breakdown with market regime flag and diverging bars.
        4. Signal scorecard (top 10 signal profiles by avg_excess_13w).
        5. 4w vs 13w reversal contingency table.
        6. Extreme observations (best/worst passed stocks).
        7. Top 5 failure reasons by frequency.
        8. Small-sample warnings (tickers/years with n_valid_13w_passed < 10).

    If diag_by_signal is empty (optional columns missing), section 4 is
    skipped with a note. If diag_failure_reasons is empty, section 7 is
    skipped with a note. If diag_extreme is empty, section 6 is
    skipped with a note.

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level DataFrame with boolean passed_technical column.
    diag_by_ticker : pd.DataFrame
        Per-ticker analysis results.
    diag_by_year : pd.DataFrame
        Per-year analysis results.
    diag_by_signal : pd.DataFrame
        Per-signal-profile analysis results (may be empty).
    diag_4w_vs_13w : pd.DataFrame
        Horizon reversal analysis results.
    diag_extreme : pd.DataFrame
        Extreme observations report.
    diag_failure_reasons : pd.DataFrame
        Failure reason frequency table (may be empty).
    """
    separator = "=" * 70
    sub_heading = "-" * 50

    # ------------------------------------------------------------------
    # Section 1: Headline stats
    # ------------------------------------------------------------------
    n_total = len(df)
    n_passed = int(df["passed_technical"].sum())
    n_failed = n_total - n_passed
    pass_rate_pct = (n_passed / n_total * 100) if n_total > 0 else 0.0

    print(separator)
    print("  BACKTEST DIAGNOSTICS SUMMARY")
    print(separator)
    print()

    print("--- Headline Stats ---")
    print(f"  Total observations:  {n_total}")
    print(f"  Pass rate:           {pass_rate_pct:.1f}% ({n_passed} / {n_total})")

    for h in [4, 13]:
        col = f"excess_return_{h}w"
        passed_er = df.loc[df["passed_technical"] == True, col].dropna()
        failed_er = df.loc[df["passed_technical"] == False, col].dropna()

        avg_p = float(passed_er.mean()) if len(passed_er) > 0 else float("nan")
        avg_f = float(failed_er.mean()) if len(failed_er) > 0 else float("nan")

        n_valid_p = len(passed_er)
        n_valid_f = len(failed_er)

        sign_p = "+" if avg_p >= 0 else ""
        sign_f = "+" if avg_f >= 0 else ""
        print(f"  Avg excess return ({h}w):  passed={sign_p}{avg_p*100:.2f}%  failed={sign_f}{avg_f*100:.2f}%")
        print(f"  Valid returns ({h}w):      passed={n_valid_p}/{n_passed}  failed={n_valid_f}/{n_failed}")

        nan_rate_p = ((n_passed - n_valid_p) / n_passed * 100) if n_passed > 0 else 0.0
        nan_rate_f = ((n_failed - n_valid_f) / n_failed * 100) if n_failed > 0 else 0.0
        print(f"  NaN rate ({h}w):           passed={nan_rate_p:.1f}%  failed={nan_rate_f:.1f}%")

    # Average IHSG benchmark returns
    for h in [4, 13]:
        col = f"index_return_{h}w"
        idx_vals = df[col].dropna()
        avg_idx = float(idx_vals.mean()) if len(idx_vals) > 0 else float("nan")
        sign_idx = "+" if avg_idx >= 0 else ""
        print(f"  Avg IHSG return ({h}w):      {sign_idx}{avg_idx*100:.2f}%")
    print()

    # ------------------------------------------------------------------
    # Section 2: Per-ticker breakdown
    # ------------------------------------------------------------------
    print("--- Per-Ticker Breakdown ---")
    print(f"  {'Ticker':<8} {'Obs':>5} {'Pass%':>7} {'ER-4w':>8} {'ER-13w':>8} "
          f"{'Win%4w':>7} {'Win%13w':>7} {'N-Val':>6}  Bar (diverging, 13w excess)")
    print(f"  {'------':<8} {'---':>5} {'-----':>7} {'-----':>8} {'------':>8} "
          f"{'-----':>7} {'------':>7} {'----':>6}  {'-' * 40}")

    if not diag_by_ticker.empty:
        for _, row in diag_by_ticker.iterrows():
            er4w = row.get("avg_excess_4w_passed", float("nan"))
            er13w = row.get("avg_excess_13w_passed", float("nan"))
            bar = _make_diverging_bar(er13w)
            nv = row.get("n_valid_13w_passed", 0)
            wr4 = row.get("win_rate_4w_passed", float("nan"))
            wr13 = row.get("win_rate_13w_passed", float("nan"))
            print(
                f"  {str(row.get('ticker', '')):<8} "
                f"{int(row.get('n_observations', 0)):>5} "
                f"{row.get('pass_rate', 0)*100:>6.1f}% "
                f"{'' if pd.isna(er4w) else f'{er4w*100:>+7.2f}%':>8} "
                f"{'' if pd.isna(er13w) else f'{er13w*100:>+7.2f}%':>8} "
                f"{'' if pd.isna(wr4) else f'{wr4*100:>6.1f}%':>7} "
                f"{'' if pd.isna(wr13) else f'{wr13*100:>6.1f}%':>7} "
                f"{int(nv):>6}  {bar}"
            )
    else:
        print("  (No ticker data available)")
    print()

    # ------------------------------------------------------------------
    # Section 3: Per-year breakdown
    # ------------------------------------------------------------------
    print("--- Per-Year Breakdown ---")
    print(f"  {'Year':<6} {'Obs':>5} {'Pass%':>7} {'ER-4w':>8} {'ER-13w':>8} "
          f"{'Win%13w':>7} {'N-Val':>6} {'Regime':<7}  Bar (diverging, 13w excess)")
    print(f"  {'----':<6} {'---':>5} {'-----':>7} {'-----':>8} {'------':>8} "
          f"{'------':>7} {'----':>6} {'------':<7}  {'-' * 40}")

    if not diag_by_year.empty:
        for _, row in diag_by_year.iterrows():
            er4w = row.get("avg_excess_4w_passed", float("nan"))
            er13w = row.get("avg_excess_13w_passed", float("nan"))
            bar = _make_diverging_bar(er13w)
            nv = row.get("n_valid_13w_passed", 0)
            wr13 = row.get("win_rate_13w_passed", float("nan"))
            regime = row.get("market_regime", "")
            print(
                f"  {int(row.get('year', 0)):<6} "
                f"{int(row.get('n_observations', 0)):>5} "
                f"{row.get('pass_rate', 0)*100:>6.1f}% "
                f"{'' if pd.isna(er4w) else f'{er4w*100:>+7.2f}%':>8} "
                f"{'' if pd.isna(er13w) else f'{er13w*100:>+7.2f}%':>8} "
                f"{'' if pd.isna(wr13) else f'{wr13*100:>6.1f}%':>7} "
                f"{int(nv):>6} "
                f"{'BULL  ' if regime == 'bull' else 'BEAR  '} "
                f"{bar}"
            )
    else:
        print("  (No yearly data available)")
    print()

    # ------------------------------------------------------------------
    # Section 4: Signal scorecard
    # ------------------------------------------------------------------
    print("--- Signal Scorecard ---")
    if diag_by_signal.empty:
        print("  (Skipped: signal columns not available in trades CSV)")
    else:
        top_signals = diag_by_signal.sort_values("avg_excess_13w", ascending=False).head(10)
        print(f"  {'Distance Bucket':<20} {'RS Bucket':<15} {'Rising':>7} "
              f"{'N':>5} {'ER-4w':>8} {'ER-13w':>8} {'Win%4w':>7} {'Win%13w':>7}")
        print(f"  {'-'*20} {'-'*15} {'-'*7} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
        for _, row in top_signals.iterrows():
            er4 = row.get("avg_excess_4w", float("nan"))
            er13 = row.get("avg_excess_13w", float("nan"))
            wr4 = row.get("win_rate_4w", float("nan"))
            wr13 = row.get("win_rate_13w", float("nan"))
            rising = row.get("sma20_is_rising", False)
            print(
                f"  {str(row.get('distance_bucket', '')):<20} "
                f"{str(row.get('rs_bucket', '')):<15} "
                f"{'True  ' if rising else 'False '} "
                f"{int(row.get('n_observations', 0)):>5} "
                f"{'' if pd.isna(er4) else f'{er4*100:>+7.2f}%':>8} "
                f"{'' if pd.isna(er13) else f'{er13*100:>+7.2f}%':>8} "
                f"{'' if pd.isna(wr4) else f'{wr4*100:>6.1f}%':>7} "
                f"{'' if pd.isna(wr13) else f'{wr13*100:>6.1f}%':>7}"
            )
    print()

    # ------------------------------------------------------------------
    # Section 5: 4w vs 13w reversal
    # ------------------------------------------------------------------
    print("--- 4w vs 13w Reversal (passed stocks only) ---")
    if diag_4w_vs_13w.empty:
        print("  (No valid observations with both horizons)")
    else:
        row = diag_4w_vs_13w.iloc[0]
        n_pp = int(row.get("n_4w_pos_13w_pos", 0))
        n_pn = int(row.get("n_4w_pos_13w_neg", 0))
        n_np = int(row.get("n_4w_neg_13w_pos", 0))
        n_nn = int(row.get("n_4w_neg_13w_neg", 0))
        pct_pn = row.get("pct_4w_pos_13w_neg", float("nan"))
        pct_np = row.get("pct_4w_neg_13w_pos", float("nan"))

        total = n_pp + n_pn + n_np + n_nn
        print(f"  Total passed with both horizons: {total}")
        print()
        print(f"                       13w Positive   13w Negative   Total")
        print(f"  {'-' * 60}")
        print(f"  4w Positive          {n_pp:<14} {n_pn:<14} {n_pp + n_pn:<6}")
        print(f"  4w Negative          {n_np:<14} {n_nn:<14} {n_np + n_nn:<6}")
        print(f"  {'-' * 60}")
        print()
        print(f"  4w winners reversing at 13w: {pct_pn:.1f}%  ({n_pn} of {n_pp + n_pn})")
        print(f"  4w losers recovering at 13w: {pct_np:.1f}%  ({n_np} of {n_np + n_nn})")
    print()

    # ------------------------------------------------------------------
    # Section 6: Extreme observations
    # ------------------------------------------------------------------
    print("--- Extreme Observations (13-Week Excess Return) ---")
    if diag_extreme.empty:
        print("  (No extreme observation data available)")
    else:
        worst = diag_extreme[diag_extreme["rank_label"] == "worst"].head(5)
        best = diag_extreme[diag_extreme["rank_label"] == "best"].head(5)

        if len(worst) > 0:
            print(f"\n  Worst {len(worst)} Passed:")
            print(f"  {'Date':<12} {'Ticker':<8} {'DistSMA20':>9} {'RS13w':>8} {'ER-13w':>9}")
            print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*8} {'-'*9}")
            for _, row in worst.iterrows():
                date_v = str(row.get("date", ""))
                ticker_v = str(row.get("ticker", ""))
                dist = row.get("distance_from_sma20", float("nan"))
                rs = row.get("relative_strength_13w", float("nan"))
                er = row.get("excess_return_13w", float("nan"))
                dist_s = f"{dist*100:>+7.2f}%" if not pd.isna(dist) else "     N/A"
                rs_s = f"{rs*100:>+6.2f}%" if not pd.isna(rs) else "    N/A"
                er_s = f"{er*100:>+7.2f}%" if not pd.isna(er) else "     N/A"
                print(f"  {date_v:<12} {ticker_v:<8} {dist_s:>9} {rs_s:>8} {er_s:>9}")

        if len(best) > 0:
            print(f"\n  Best {len(best)} Passed:")
            print(f"  {'Date':<12} {'Ticker':<8} {'DistSMA20':>9} {'RS13w':>8} {'ER-13w':>9}")
            print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*8} {'-'*9}")
            for _, row in best.iterrows():
                date_v = str(row.get("date", ""))
                ticker_v = str(row.get("ticker", ""))
                dist = row.get("distance_from_sma20", float("nan"))
                rs = row.get("relative_strength_13w", float("nan"))
                er = row.get("excess_return_13w", float("nan"))
                dist_s = f"{dist*100:>+7.2f}%" if not pd.isna(dist) else "     N/A"
                rs_s = f"{rs*100:>+6.2f}%" if not pd.isna(rs) else "    N/A"
                er_s = f"{er*100:>+7.2f}%" if not pd.isna(er) else "     N/A"
                print(f"  {date_v:<12} {ticker_v:<8} {dist_s:>9} {rs_s:>8} {er_s:>9}")
    print()

    # ------------------------------------------------------------------
    # Section 7: Failure reasons
    # ------------------------------------------------------------------
    print("--- Top Failure Reasons ---")
    if diag_failure_reasons.empty:
        print("  (Skipped: reasons column not available in trades CSV)")
    else:
        top5 = diag_failure_reasons.head(5)
        print(f"  {'Reason Category':<60} {'N':>6} {'%':>7} {'ER-13w Passed':>14} {'ER-13w Failed':>14}")
        print(f"  {'-'*60} {'-'*6} {'-'*7} {'-'*14} {'-'*14}")
        for _, r in top5.iterrows():
            reason_str = str(r.get("reason_category", ""))[:58]
            er_p = r.get("avg_excess_13w_when_passed", float("nan"))
            er_f = r.get("avg_excess_13w_when_failed", float("nan"))
            print(
                f"  {reason_str:<60} "
                f"{int(r.get('n_occurrences', 0)):>6} "
                f"{r.get('pct_of_total', 0):>6.1f}% "
                f"{'' if pd.isna(er_p) else f'{er_p*100:>+7.2f}%':>14} "
                f"{'' if pd.isna(er_f) else f'{er_f*100:>+7.2f}%':>14}"
            )
    print()

    # ------------------------------------------------------------------
    # Section 8: Small-sample warnings
    # ------------------------------------------------------------------
    warnings_found = False

    if not diag_by_ticker.empty:
        small_tickers = diag_by_ticker[diag_by_ticker["n_valid_13w_passed"] < 10]
        if not small_tickers.empty:
            warnings_found = True
            print("--- Small-Sample Warnings ---")
            print("  Tickers with n_valid_13w_passed < 10:")
            for _, r in small_tickers.iterrows():
                print(f"    {r.get('ticker', '?')}: n_valid_13w_passed={int(r.get('n_valid_13w_passed', 0))}")

    if not diag_by_year.empty:
        small_years = diag_by_year[diag_by_year["n_valid_13w_passed"] < 10]
        if not small_years.empty:
            if not warnings_found:
                warnings_found = True
                print("--- Small-Sample Warnings ---")
            print("  Years with n_valid_13w_passed < 10:")
            for _, r in small_years.iterrows():
                print(f"    {int(r.get('year', 0))}: n_valid_13w_passed={int(r.get('n_valid_13w_passed', 0))}")

    print(separator)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_diagnostics() -> None:
    """Run the full backtest diagnostics pipeline.

    Orchestrates: CLI parsing, data loading, analysis, report writing,
    and console output. Exits with code 1 on fatal errors (missing file,
    empty CSV, missing required columns).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        args = parse_args()
    except Exception as exc:
        logger.error("Argument parsing failed: %s", exc)
        sys.exit(1)

    logger.info("Loading trades CSV: %s", args.trades_csv)
    try:
        df = load_trades_csv(args.trades_csv)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load trades CSV: %s", exc)
        sys.exit(1)

    logger.info("Loaded %d observations with %d columns", len(df), len(df.columns))

    # Run analyses (compute once, pass results to both CSV writer and console printer)
    logger.info("Running diagnostics...")
    diag_by_ticker = analyze_by_ticker(df)
    diag_by_year = analyze_by_year(df)
    diag_by_signal = analyze_by_signal(df)
    diag_4w_vs_13w = analyze_4w_vs_13w(df)
    diag_extreme = analyze_extreme_observations(df, top_n=args.top_n)
    diag_failure = analyze_failure_reasons(df)

    # Write reports
    logger.info("Writing diagnostic CSVs to: %s", args.output_dir)
    file_map = write_diagnostics(
        diag_by_ticker=diag_by_ticker,
        diag_by_year=diag_by_year,
        diag_by_signal=diag_by_signal,
        diag_4w_vs_13w=diag_4w_vs_13w,
        diag_extreme=diag_extreme,
        diag_failure=diag_failure,
        output_dir=args.output_dir,
    )
    for name, path in file_map.items():
        logger.info("  %s -> %s", name, os.path.basename(path))

    # Print summary
    print_summary(
        df=df,
        diag_by_ticker=diag_by_ticker,
        diag_by_year=diag_by_year,
        diag_by_signal=diag_by_signal,
        diag_4w_vs_13w=diag_4w_vs_13w,
        diag_extreme=diag_extreme,
        diag_failure_reasons=diag_failure,
    )

    logger.info("Diagnostics complete.")


if __name__ == "__main__":
    run_diagnostics()
