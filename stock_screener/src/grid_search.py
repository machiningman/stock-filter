"""
Grid search over technical filter parameter combinations for the LQ45
Stock Screener backtest.

Evaluates 37 parameter combinations (36 grid + 1 baseline) and outputs
a ranked CSV report to help identify the best-performing parameter set.

Usage:
    python -m stock_screener.src.grid_search
    python -m stock_screener.src.grid_search --top-n 5 --sort-by win_rate_13w
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from stock_screener.src.config import load_config
from stock_screener.src.backtest import (
    load_backtest_inputs,
    get_backtest_dates,
    run_evaluation,
)
from stock_screener.src.pipeline import resample_to_weekly

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARAM_GRID: dict[str, list[Any]] = {
    "max_distance_from_sma20": [0.05, 0.08, 0.10],
    "min_relative_strength_13w": [-0.05, 0.00],
    "max_relative_strength_13w": [0.08, 0.10, 0.15],
    "require_sma20_rising": [True, False],
}

_MIN_SIGNALS: int = 30
_MIN_SIGNALS_HARD_FLOOR: int = 10
_TEST_FROM_YEAR: int = 2024

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the grid search.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: ``config``, ``output_dir``,
        ``sort_by``, ``top_n``, ``min_signals``.
    """
    parser = argparse.ArgumentParser(
        description="Grid search over technical filter parameters for the "
        "LQ45 stock screener backtest."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default: config.yaml next to project root).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the grid search CSV report (default: reports/).",
    )
    parser.add_argument(
        "--sort-by",
        default="avg_excess_13w",
        help="Metric to sort results by (default: avg_excess_13w).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configurations to show in summary (default: 10).",
    )
    parser.add_argument(
        "--min-signals",
        type=int,
        default=None,
        help="Minimum number of signals for inclusion (default: _MIN_SIGNALS).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Parameter combination generation
# ---------------------------------------------------------------------------


def generate_param_combinations(
    param_grid: dict[str, list[Any]],
    baseline: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate all Cartesian product parameter combinations from *param_grid*.

    Optionally appends a *baseline* combination if it is not already present.

    Parameters
    ----------
    param_grid : dict[str, list[Any]]
        Mapping of parameter names to lists of values to try.
    baseline : dict[str, Any] or None
        A single parameter combination to include (e.g. current config values).

    Returns
    -------
    list[dict[str, Any]]
        List of parameter dicts, each suitable for shallow-merging into
        ``config["technical"]``.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    if not keys:
        combos: list[dict[str, Any]] = []
    else:
        combos = [dict(zip(keys, prod)) for prod in itertools.product(*values)]

    if baseline is not None and baseline not in combos:
        combos.append(baseline)

    return combos


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------


def load_grid_data(
    config: dict,
    data_dir: str,
    cache_dir: str,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    list[pd.Timestamp],
    list[int],
]:
    """Load universe, prices, index data and generate evaluation dates.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    data_dir : str
        Path to the data directory (contains universe CSV).
    cache_dir : str
        Path to the price cache directory.

    Returns
    -------
    tuple
        ``(universe_df, stock_weekly_data, index_weekly, eval_dates, horizons)``
    """
    universe_df, price_data, index_data = load_backtest_inputs(
        config, data_dir, cache_dir
    )
    tickers = universe_df["ticker"].tolist()

    logger.info("Resampling price data to weekly...")
    index_weekly = resample_to_weekly(index_data)
    if index_weekly.empty:
        raise RuntimeError("Index weekly data is empty after resampling.")

    stock_weekly_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = price_data.get(ticker, pd.DataFrame())
        stock_weekly_data[ticker] = resample_to_weekly(df)

    bt_cfg = config.get("backtest", {})
    eval_dates = get_backtest_dates(
        index_weekly,
        bt_cfg.get("min_warmup_weeks", 60),
        bt_cfg.get("horizons_weeks", [4, 13]),
    )
    if not eval_dates:
        raise RuntimeError("No valid backtest evaluation dates.")

    horizons = bt_cfg.get("horizons_weeks", [4, 13])

    return universe_df, stock_weekly_data, index_weekly, eval_dates, horizons


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(trades_df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary metrics from a backtest trades DataFrame.

    Filters to rows where ``passed_technical`` is ``True``, then computes
    performance statistics for the 4-week and 13-week excess returns.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade-level DataFrame as returned by ``run_evaluation()``.

    Returns
    -------
    dict
        Dictionary with keys:

        - **n_signals** — number of passed signals
        - **pass_rate** — fraction of total rows that passed
        - **n_valid_returns** — number of passed signals with valid 13w return
        - **avg_excess_13w** — mean 13-week excess return
        - **median_excess_13w** — median 13-week excess return
        - **std_excess_13w** — std of 13-week excess return
        - **snr_13w** — signal-to-noise ratio (avg / std, 0 if std == 0)
        - **se_excess_13w** — standard error of the mean (std / sqrt(n))
        - **win_rate_13w** — fraction of 13-week excess returns > 0
        - **avg_excess_4w** — mean 4-week excess return
        - **win_rate_4w** — fraction of 4-week excess returns > 0
        - **reversal_rate_4w_to_13w** — fraction of observations where
          4-week excess is positive but 13-week excess is negative
        - **worst_excess_13w** — minimum 13-week excess return
    """
    total = len(trades_df)

    if trades_df.empty or "passed_technical" not in trades_df.columns:
        return {
            "n_signals": 0,
            "pass_rate": 0.0,
            "n_valid_returns": 0,
            "avg_excess_13w": float("nan"),
            "median_excess_13w": float("nan"),
            "std_excess_13w": float("nan"),
            "snr_13w": float("nan"),
            "se_excess_13w": float("nan"),
            "win_rate_13w": float("nan"),
            "avg_excess_4w": float("nan"),
            "win_rate_4w": float("nan"),
            "reversal_rate_4w_to_13w": float("nan"),
            "worst_excess_13w": float("nan"),
        }

    passed_df = trades_df[trades_df["passed_technical"] == True].copy()
    n_signals = len(passed_df)
    pass_rate = n_signals / total if total > 0 else 0.0

    if n_signals == 0:
        return {
            "n_signals": 0,
            "pass_rate": pass_rate,
            "n_valid_returns": 0,
            "avg_excess_13w": float("nan"),
            "median_excess_13w": float("nan"),
            "std_excess_13w": float("nan"),
            "snr_13w": float("nan"),
            "se_excess_13w": float("nan"),
            "win_rate_13w": float("nan"),
            "avg_excess_4w": float("nan"),
            "win_rate_4w": float("nan"),
            "reversal_rate_4w_to_13w": float("nan"),
            "worst_excess_13w": float("nan"),
        }

    # 13-week excess return metrics
    excess_13 = passed_df["excess_return_13w"].dropna()
    n_valid = len(excess_13)

    if n_valid == 0:
        avg_13 = float("nan")
        median_13 = float("nan")
        std_13 = float("nan")
        snr_13 = float("nan")
        se_13 = float("nan")
        win_13 = float("nan")
        worst_13 = float("nan")
    else:
        avg_13 = float(excess_13.mean())
        median_13 = float(excess_13.median())
        std_13 = float(excess_13.std(ddof=1))
        snr_13 = avg_13 / std_13 if std_13 != 0.0 else 0.0
        se_13 = std_13 / np.sqrt(n_valid)
        win_13 = float((excess_13 > 0).mean())
        worst_13 = float(excess_13.min())

    # 4-week excess return metrics
    excess_4 = passed_df["excess_return_4w"].dropna()
    if len(excess_4) == 0:
        avg_4 = float("nan")
        win_4 = float("nan")
    else:
        avg_4 = float(excess_4.mean())
        win_4 = float((excess_4 > 0).mean())

    # Reversal rate: among rows with both returns valid, fraction where
    # 4w > 0 but 13w < 0 (short-term positive, longer-term negative)
    both_valid = passed_df[
        passed_df["excess_return_4w"].notna()
        & passed_df["excess_return_13w"].notna()
    ]
    n_both = len(both_valid)
    if n_both == 0:
        reversal_rate = float("nan")
    else:
        reversed_count = (
            (both_valid["excess_return_4w"] > 0)
            & (both_valid["excess_return_13w"] < 0)
        ).sum()
        reversal_rate = reversed_count / n_both

    return {
        "n_signals": n_signals,
        "pass_rate": pass_rate,
        "n_valid_returns": n_valid,
        "avg_excess_13w": avg_13,
        "median_excess_13w": median_13,
        "std_excess_13w": std_13,
        "snr_13w": snr_13,
        "se_excess_13w": se_13,
        "win_rate_13w": win_13,
        "avg_excess_4w": avg_4,
        "win_rate_4w": win_4,
        "reversal_rate_4w_to_13w": reversal_rate,
        "worst_excess_13w": worst_13,
    }


def compute_train_test_metrics(
    trades_df: pd.DataFrame,
    test_from_year: int = 2025,
) -> dict[str, Any]:
    """Compute train/test performance by splitting on year.

    Observations with ``date`` >= ``test_from_year``-01-01 are considered
    test set; earlier observations are the training set.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade-level DataFrame (must include ``date`` and
        ``excess_return_13w`` columns).
    test_from_year : int
        First year of the test period (default 2025).

    Returns
    -------
    dict
        Dictionary with keys: ``train_avg_excess_13w``, ``test_avg_excess_13w``,
        ``train_test_gap`` (train minus test).
    """
    if trades_df.empty or "date" not in trades_df.columns:
        return {
            "train_avg_excess_13w": float("nan"),
            "test_avg_excess_13w": float("nan"),
            "train_test_gap": float("nan"),
        }

    passed_df = trades_df[trades_df["passed_technical"] == True].copy()
    if passed_df.empty:
        return {
            "train_avg_excess_13w": float("nan"),
            "test_avg_excess_13w": float("nan"),
            "train_test_gap": float("nan"),
        }

    if "excess_return_13w" not in passed_df.columns:
        return {
            "train_avg_excess_13w": float("nan"),
            "test_avg_excess_13w": float("nan"),
            "train_test_gap": float("nan"),
        }

    dates = pd.to_datetime(passed_df["date"], errors="coerce")
    if dates.isna().all():
        logger.warning(
            "compute_train_test_metrics: all dates are NaT for combo with %d signals",
            len(passed_df),
        )
        return {
            "train_avg_excess_13w": float("nan"),
            "test_avg_excess_13w": float("nan"),
            "train_test_gap": float("nan"),
        }
    passed_df["_year"] = dates.dt.year

    train = passed_df[passed_df["_year"] < test_from_year]["excess_return_13w"].dropna()
    test = passed_df[passed_df["_year"] >= test_from_year]["excess_return_13w"].dropna()

    train_avg = float(train.mean()) if len(train) > 0 else float("nan")
    test_avg = float(test.mean()) if len(test) > 0 else float("nan")

    logger.info(
        "compute_train_test_metrics: train=%d obs, test=%d obs, total_passed=%d",
        len(train), len(test), len(passed_df),
    )

    if not np.isnan(train_avg) and not np.isnan(test_avg):
        gap = train_avg - test_avg
    else:
        gap = float("nan")

    return {
        "train_avg_excess_13w": train_avg,
        "test_avg_excess_13w": test_avg,
        "train_test_gap": gap,
    }


# ---------------------------------------------------------------------------
# Single combo evaluation
# ---------------------------------------------------------------------------

def evaluate_param_combo(
    combo: dict[str, Any],
    base_config: dict,
    universe_df: pd.DataFrame,
    stock_weekly_data: dict[str, pd.DataFrame],
    index_weekly: pd.DataFrame,
    eval_dates: list[pd.Timestamp],
    horizons: list[int],
    is_baseline: bool = False,
) -> dict[str, Any]:
    """Evaluate a single parameter combination.

    Deep copies *base_config*, shallow-merges *combo* into
    ``config["technical"]``, calls ``run_evaluation()``, and computes
    summary metrics.

    Parameters
    ----------
    combo : dict[str, Any]
        Parameter values to merge into ``config["technical"]``.
    base_config : dict
        Base configuration (will be deep-copied before modification).
    universe_df : pd.DataFrame
        LQ45 universe data.
    stock_weekly_data : dict[str, pd.DataFrame]
        Per-ticker weekly OHLCV DataFrames.
    index_weekly : pd.DataFrame
        Weekly index OHLCV DataFrame.
    eval_dates : list[pd.Timestamp]
        Sorted list of valid evaluation dates.
    horizons : list[int]
        Forward return horizons in weeks.
    is_baseline : bool
        Whether this is the baseline config (default False).

    Returns
    -------
    dict
        Merged combo keys + all metrics + ``is_baseline`` flag.
    """
    config_copy = copy.deepcopy(base_config)
    config_copy["technical"].update(combo)

    trades_df = run_evaluation(
        config_copy,
        universe_df,
        stock_weekly_data,
        index_weekly,
        eval_dates,
        horizons,
    )

    metrics = compute_metrics(trades_df)
    train_test = compute_train_test_metrics(trades_df, _TEST_FROM_YEAR)

    result: dict[str, Any] = {
        **combo,
        **metrics,
        **train_test,
        "is_baseline": is_baseline,
    }
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_grid_search_report(
    results: list[dict[str, Any]],
    output_dir: str,
    sort_by: str = "avg_excess_13w",
    min_signals: int = 30,
    hard_floor: int = 10,
    data_range: str = "",
    config_path: str = "",
) -> str:
    """Write a ranked grid search report CSV.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from ``evaluate_param_combo()``.
    output_dir : str
        Directory to write the CSV into (created if missing).
    sort_by : str
        Column name to sort by descending (default ``avg_excess_13w``).
    min_signals : int
        Threshold for the ``low_sample_warning`` flag.
    hard_floor : int
        Configs with ``n_signals < hard_floor`` are excluded.
    data_range : str
        Date range string for metadata (e.g. "2020-01 to 2025-12").
    config_path : str
        Path to the config file used, for metadata.

    Returns
    -------
    str
        Absolute path to the generated CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)

    # Handle empty results gracefully
    if df.empty:
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"grid_search_results_{date_str}.csv"
        filepath = os.path.join(output_dir, filename)
        # Write a header-only CSV with metadata columns
        empty_out = pd.DataFrame(
            {"_timestamp": [], "_data_range": [], "_config_path": []}
        )
        empty_out.to_csv(filepath, index=False, encoding="utf-8")
        logger.info("Grid search report saved to %s", filepath)
        return os.path.abspath(filepath)

    # Add low_sample_warning flag
    df["low_sample_warning"] = df["n_signals"].apply(
        lambda x: x < min_signals if pd.notna(x) else True
    )

    # Exclude configs below hard floor
    df_to_write = df[df["n_signals"] >= hard_floor].copy()
    if df_to_write.empty:
        logger.warning(
            "All configurations have n_signals < %d (hard floor). "
            "Writing empty report.",
            hard_floor,
        )
        df_to_write = df  # fall back to including all

    # Validate sort_by
    if sort_by not in df_to_write.columns:
        available = [c for c in df_to_write.columns if c not in ("is_baseline",)]
        raise ValueError(
            f"sort_by metric '{sort_by}' not found in results. "
            f"Available metrics: {available}"
        )

    # Sort descending
    df_sorted = df_to_write.sort_values(by=sort_by, ascending=False).reset_index(
        drop=True
    )

    # Round numeric columns to 4 decimal places
    numeric_cols = df_sorted.select_dtypes(include=[np.floating]).columns.tolist()
    for col in numeric_cols:
        df_sorted[col] = df_sorted[col].round(4)

    # Add metadata columns
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_sorted["_timestamp"] = now_str
    df_sorted["_data_range"] = data_range
    df_sorted["_config_path"] = config_path

    # Build filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"grid_search_results_{date_str}.csv"
    filepath = os.path.join(output_dir, filename)

    df_sorted.to_csv(filepath, index=False, encoding="utf-8")
    logger.info("Grid search report saved to %s", filepath)

    return os.path.abspath(filepath)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS = [
    "max_distance_from_sma20",
    "min_relative_strength_13w",
    "max_relative_strength_13w",
    "require_sma20_rising",
    "n_signals",
    "avg_excess_13w",
    "win_rate_13w",
    "avg_excess_4w",
    "win_rate_4w",
    "snr_13w",
    "reversal_rate_4w_to_13w",
]


def _fmt(val: Any, decimals: int = 4) -> str:
    """Format a value for console display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def print_summary(
    results: list[dict[str, Any]],
    baseline_result: dict[str, Any] | None = None,
    top_n: int = 10,
) -> None:
    """Print a formatted summary of grid search results to the console.

    Parameters
    ----------
    results : list[dict]
        Sorted list of result dicts (best first).
    baseline_result : dict or None
        The baseline result dict for comparison.
    top_n : int
        Number of top configurations to display.
    """
    separator = "=" * 90
    sub_separator = "-" * 90

    print()
    print(separator)
    print("  GRID SEARCH SUMMARY")
    print(separator)

    # --- Baseline comparison ---
    if baseline_result is not None:
        print()
        print("  Baseline configuration:")
        print(sub_separator)
        for key in _PARAM_GRID:
            print(f"    {key}: {_fmt(baseline_result.get(key))}")
        print()
        print(f"    n_signals         : {_fmt(baseline_result.get('n_signals'), 0)}")
        print(f"    avg_excess_13w    : {_fmt(baseline_result.get('avg_excess_13w'))}")
        print(f"    win_rate_13w      : {_fmt(baseline_result.get('win_rate_13w'))}")
        print(f"    avg_excess_4w     : {_fmt(baseline_result.get('avg_excess_4w'))}")
        print(f"    win_rate_4w       : {_fmt(baseline_result.get('win_rate_4w'))}")
        print(f"    snr_13w           : {_fmt(baseline_result.get('snr_13w'))}")
        print(f"    reversal_rate     : {_fmt(baseline_result.get('reversal_rate_4w_to_13w'))}")
        print(f"    train_avg_excess  : {_fmt(baseline_result.get('train_avg_excess_13w'))}")
        print(f"    test_avg_excess   : {_fmt(baseline_result.get('test_avg_excess_13w'))}")
        print(f"    train_test_gap    : {_fmt(baseline_result.get('train_test_gap'))}")

    # --- Top N configs ---
    print()
    print(f"  Top {min(top_n, len(results))} configurations "
          f"(sorted by avg_excess_13w descending):")
    print(sub_separator)

    # Header
    headers = ["Rank"] + [col.replace("_", " ").title() for col in _SUMMARY_COLUMNS]
    print("  " + "  ".join(f"{h:<22}" for h in headers[:6]))
    print("  " + "  ".join(f"{'':<22}" for _ in range(6)))
    print("  " + "  ".join(f"{h:<22}" for h in headers[6:]))

    # Rows
    for rank, res in enumerate(results[:top_n], start=1):
        vals = [_fmt(res.get(c)) for c in _SUMMARY_COLUMNS]
        line1_vals = [str(rank)] + vals[:5]
        line1 = "  " + "  ".join(f"{v:<22}" for v in line1_vals)
        line2_vals = vals[5:]
        line2 = "  " + "  ".join(f"{v:<22}" for v in line2_vals)
        print(line1)
        print(line2)
        print()

    # --- Warnings ---
    print(separator)
    print("  WARNINGS")
    print(sub_separator)

    low_signal_configs = [
        r for r in results if r.get("n_signals", 0) < _MIN_SIGNALS
    ]
    if low_signal_configs:
        print(
            f"  * {len(low_signal_configs)} config(s) have n_signals < "
            f"{_MIN_SIGNALS} (min_signals threshold)."
        )
        for r in low_signal_configs[:5]:
            params = ", ".join(
                f"{k}={_fmt(r.get(k))}" for k in _PARAM_GRID
            )
            print(f"    - [{params}] n_signals={r.get('n_signals', 0)}")
        if len(low_signal_configs) > 5:
            print(f"    ... and {len(low_signal_configs) - 5} more")
    else:
        print(f"  * All configs meet the min_signals={_MIN_SIGNALS} threshold.")

    # Selection bias warning
    if results and baseline_result is not None:
        best = results[0]
        diff_keys = []
        for key in _PARAM_GRID:
            bv = baseline_result.get(key)
            tv = best.get(key)
            if bv != tv:
                diff_keys.append(f"{key}: baseline={_fmt(bv)} vs best={_fmt(tv)}")
        if diff_keys:
            print()
            print(
                "  * SELECTION BIAS: The top-ranked configuration differs from "
                "baseline."
            )
            for d in diff_keys:
                print(f"    - {d}")
            print(
                "    Review whether the selected parameters are robust "
                "out-of-sample."
            )

    print(separator)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_grid_search() -> None:
    """Run the full grid search.

    Orchestrates: CLI parsing, config loading, logging setup, baseline
    derivation, data loading, parameter generation, sequential evaluation,
    report writing, and console summary.
    """
    args = parse_args()

    # --- Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    cache_dir = os.path.join(data_dir, "cache")

    config_path = args.config or os.path.join(project_root, "config.yaml")
    output_dir = args.output_dir or os.path.join(project_root, "reports")
    sort_by = args.sort_by
    top_n = args.top_n
    min_signals = args.min_signals if args.min_signals is not None else _MIN_SIGNALS

    # --- Load config ---
    try:
        config = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        logging.error("Failed to load config: %s", exc)
        sys.exit(1)

    # --- Setup logging ---
    log_cfg = config.get("logging", {})
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = log_cfg.get("format", "%(message)s")
    logging.basicConfig(level=level, format=fmt, force=True)

    logger.info("Starting grid search")
    logger.info("Config: %s", config_path)
    start_time = time.monotonic()

    # --- Derive baseline from config["technical"] ---
    tech_cfg = config["technical"]
    baseline: dict[str, Any] = {}
    for key in _PARAM_GRID:
        baseline[key] = tech_cfg.get(key)

    logger.info("Baseline params: %s", baseline)

    # --- Load data ---
    logger.info("Loading data...")
    try:
        universe_df, stock_weekly_data, index_weekly, eval_dates, horizons = (
            load_grid_data(config, data_dir, cache_dir)
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Data loading failed: %s", exc)
        sys.exit(1)

    data_range_str = (
        f"{eval_dates[0].strftime('%Y-%m-%d')} to "
        f"{eval_dates[-1].strftime('%Y-%m-%d')}"
    )
    logger.info(
        "Data loaded: %d tickers, %d eval dates, %d horizons",
        len(universe_df),
        len(eval_dates),
        len(horizons),
    )

    # --- Generate parameter combinations ---
    combos = generate_param_combinations(_PARAM_GRID, baseline)
    logger.info("Generated %d parameter combinations (%d grid + baseline)", len(combos), len(combos) - 1)

    # --- Sequential evaluation ---
    results: list[dict[str, Any]] = []
    baseline_result: dict[str, Any] | None = None

    for idx, combo in enumerate(combos):
        is_bl = combo == baseline
        logger.info(
            "Evaluating combo %d/%d%s: %s",
            idx + 1,
            len(combos),
            " (baseline)" if is_bl else "",
            combo,
        )

        try:
            result = evaluate_param_combo(
                combo,
                config,
                universe_df,
                stock_weekly_data,
                index_weekly,
                eval_dates,
                horizons,
                is_baseline=is_bl,
            )
        except Exception as exc:
            logger.error("Combo %d failed: %s", idx + 1, exc)
            # Add combo with error indicators
            result = {
                **combo,
                "n_signals": -1,
                "pass_rate": float("nan"),
                "n_valid_returns": 0,
                "avg_excess_13w": float("nan"),
                "median_excess_13w": float("nan"),
                "std_excess_13w": float("nan"),
                "snr_13w": float("nan"),
                "se_excess_13w": float("nan"),
                "win_rate_13w": float("nan"),
                "avg_excess_4w": float("nan"),
                "win_rate_4w": float("nan"),
                "reversal_rate_4w_to_13w": float("nan"),
                "worst_excess_13w": float("nan"),
                "train_avg_excess_13w": float("nan"),
                "test_avg_excess_13w": float("nan"),
                "train_test_gap": float("nan"),
                "is_baseline": is_bl,
            }

        results.append(result)

        if is_bl:
            baseline_result = result

        logger.info(
            "  -> n_signals=%d, avg_excess_13w=%.4f, win_rate_13w=%.4f",
            result.get("n_signals", -1),
            result.get("avg_excess_13w", float("nan")),
            result.get("win_rate_13w", float("nan")),
        )

    # --- Sort results by sort_by descending ---
    sort_values = [r.get(sort_by, float("-inf")) for r in results]
    sort_values = [
        v if (isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))) else float("-inf")
        for v in sort_values
    ]
    sorted_indices = sorted(
        range(len(results)), key=lambda i: sort_values[i], reverse=True
    )
    sorted_results = [results[i] for i in sorted_indices]

    # --- Write report ---
    report_path = write_grid_search_report(
        sorted_results,
        output_dir,
        sort_by=sort_by,
        min_signals=min_signals,
        hard_floor=_MIN_SIGNALS_HARD_FLOOR,
        data_range=data_range_str,
        config_path=os.path.abspath(config_path),
    )

    # --- Print summary ---
    print_summary(sorted_results, baseline_result, top_n=top_n)

    elapsed = time.monotonic() - start_time
    logger.info("Grid search complete in %.2f seconds", elapsed)
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    run_grid_search()
