"""
Unit tests for the grid search module (stock_screener.src.grid_search).
"""

from __future__ import annotations

import copy
import os
import sys
from unittest.mock import ANY, Mock, patch

import numpy as np
import pandas as pd
import pytest

from stock_screener.src.grid_search import (
    _MIN_SIGNALS,
    _MIN_SIGNALS_HARD_FLOOR,
    _PARAM_GRID,
    _TEST_FROM_YEAR,
    compute_metrics,
    compute_train_test_metrics,
    evaluate_param_combo,
    generate_param_combinations,
    parse_args,
    print_summary,
    run_grid_search,
    write_grid_search_report,
)
from stock_screener.src import grid_search as gs_module


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_minimal_trades_df() -> pd.DataFrame:
    """Create a minimal valid trades DataFrame for testing.

    Returns a DataFrame with columns matching ``run_evaluation()`` output.
    Includes a mix of True/False ``passed_technical`` values and at least
    4 rows.
    """
    return pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-05"),
                pd.Timestamp("2024-01-12"),
                pd.Timestamp("2024-01-19"),
                pd.Timestamp("2024-01-26"),
                pd.Timestamp("2024-02-02"),
                pd.Timestamp("2025-01-03"),
            ],
            "ticker": ["A"] * 6,
            "company_name": ["A Corp"] * 6,
            "sector": ["Bank"] * 6,
            "passed_technical": [True, True, True, False, False, True],
            "close": [100.0] * 6,
            "sma20": [99.0] * 6,
            "sma50": [98.0] * 6,
            "distance_from_sma20": [0.01] * 6,
            "relative_strength_13w": [0.02] * 6,
            "sma20_is_rising": [True] * 6,
            "forward_return_4w": [0.05, 0.03, -0.02, 0.01, 0.04, -0.01],
            "forward_return_13w": [0.10, -0.05, 0.03, 0.02, -0.01, 0.06],
            "index_return_4w": [0.02] * 6,
            "index_return_13w": [0.05] * 6,
            "excess_return_4w": [0.03, 0.01, -0.04, -0.01, 0.02, -0.03],
            "excess_return_13w": [0.05, -0.10, -0.02, -0.03, -0.06, 0.01],
            "reasons": ["PASS"] * 6,
            "warnings": [""] * 6,
        }
    )


def _make_minimal_config() -> dict:
    """Create a minimal config dict for testing grid search operations.

    Includes ``technical`` and ``backtest`` sections with all expected keys.
    """
    return {
        "technical": {
            "sma_short": 20,
            "sma_long": 50,
            "max_distance_from_sma20": 0.15,
            "relative_strength_weeks": 13,
            "sma_rising_lookback": 3,
            "min_relative_strength_13w": 0.0,
            "max_relative_strength_13w": None,
            "require_sma20_rising": True,
        },
        "backtest": {
            "history_months": 60,
            "horizons_weeks": [4, 13],
            "min_warmup_weeks": 60,
        },
    }


def _make_mock_data_for_evaluate() -> tuple:
    """Create mock data arguments for ``evaluate_param_combo``."""
    universe_df = pd.DataFrame(
        {"ticker": ["A"], "company_name": ["A Corp"], "sector": ["Bank"]}
    )
    dates = pd.date_range(start="2024-01-05", periods=10, freq="W-FRI")
    stock_weekly = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000,
        },
        index=dates,
    )
    index_weekly = stock_weekly.copy()
    stock_weekly_data = {"A": stock_weekly}
    eval_dates = dates[3:7].tolist()
    horizons = [4, 13]
    return universe_df, stock_weekly_data, index_weekly, eval_dates, horizons


# ---------------------------------------------------------------------------
# Tests: parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_default_values(self):
        """Default arguments are set correctly."""
        args = parse_args([])
        assert args.config is None
        assert args.output_dir is None
        assert args.sort_by == "avg_excess_13w"
        assert args.top_n == 10
        assert args.min_signals is None

    def test_custom_values(self):
        """Custom arguments are parsed correctly."""
        argv = [
            "--config",
            "/custom/path/config.yaml",
            "--output-dir",
            "/out",
            "--sort-by",
            "win_rate_13w",
            "--top-n",
            "5",
            "--min-signals",
            "50",
        ]
        args = parse_args(argv)
        assert args.config == "/custom/path/config.yaml"
        assert args.output_dir == "/out"
        assert args.sort_by == "win_rate_13w"
        assert args.top_n == 5
        assert args.min_signals == 50


# ---------------------------------------------------------------------------
# Tests: generate_param_combinations
# ---------------------------------------------------------------------------


class TestGenerateParamCombinations:
    def test_cartesian_product_is_correct(self):
        """Cartesian product of param grid has correct length and keys."""
        grid = {"a": [1, 2], "b": ["x", "y"]}
        combos = generate_param_combinations(grid)
        assert len(combos) == 4
        for combo in combos:
            assert "a" in combo
            assert "b" in combo

    def test_empty_grid_returns_empty_list(self):
        """Empty grid returns empty list of combinations."""
        combos = generate_param_combinations({})
        assert combos == []

    def test_single_value_grid_returns_one_combo(self):
        """Grid with single values for each param returns one combo."""
        grid = {"a": [1], "b": ["x"]}
        combos = generate_param_combinations(grid)
        assert len(combos) == 1
        assert combos[0] == {"a": 1, "b": "x"}

    def test_baseline_is_included_when_not_in_combos(self):
        """Baseline is appended when not already in grid combos."""
        grid = {"a": [1, 2]}
        baseline = {"a": 99}
        combos = generate_param_combinations(grid, baseline)
        assert len(combos) == 3
        assert baseline in combos

    def test_baseline_not_duplicated_when_already_present(self):
        """Baseline is not duplicated if already part of the grid."""
        grid = {"a": [1, 2]}
        baseline = {"a": 1}
        combos = generate_param_combinations(grid, baseline)
        assert len(combos) == 2
        assert combos.count(baseline) == 1


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        """Result dict contains all expected metric keys."""
        df = _make_minimal_trades_df()
        metrics = compute_metrics(df)
        expected_keys = {
            "n_signals",
            "pass_rate",
            "n_valid_returns",
            "avg_excess_13w",
            "median_excess_13w",
            "std_excess_13w",
            "snr_13w",
            "se_excess_13w",
            "win_rate_13w",
            "avg_excess_4w",
            "win_rate_4w",
            "reversal_rate_4w_to_13w",
            "worst_excess_13w",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_handles_all_passed_signals(self):
        """Works correctly when all rows passed_technical."""
        df = _make_minimal_trades_df()
        df["passed_technical"] = True
        metrics = compute_metrics(df)
        assert metrics["n_signals"] == 6
        assert metrics["pass_rate"] == 1.0

    def test_handles_all_failed_signals(self):
        """Returns zero signals with NaN metrics when no rows pass."""
        df = _make_minimal_trades_df()
        df["passed_technical"] = False
        metrics = compute_metrics(df)
        assert metrics["n_signals"] == 0
        assert metrics["pass_rate"] == 0.0
        assert np.isnan(metrics["avg_excess_13w"])

    def test_handles_empty_dataframe(self):
        """Returns zero signals with NaN metrics for empty DataFrame."""
        df = pd.DataFrame()
        metrics = compute_metrics(df)
        assert metrics["n_signals"] == 0
        assert metrics["pass_rate"] == 0.0
        assert np.isnan(metrics["avg_excess_13w"])

    def test_handles_dataframe_without_passed_technical_column(self):
        """Returns zero signals for DataFrame missing passed_technical."""
        df = pd.DataFrame({"ticker": ["A"], "excess_return_13w": [0.05]})
        metrics = compute_metrics(df)
        assert metrics["n_signals"] == 0
        assert metrics["pass_rate"] == 0.0

    def test_reversal_rate_calculation(self):
        """Reversal rate counts 4w>0 & 13w<0 over both-valid observations."""
        df = pd.DataFrame(
            {
                "passed_technical": [True, True, True, True],
                "excess_return_4w": [0.05, 0.02, -0.01, np.nan],
                "excess_return_13w": [-0.03, 0.01, 0.02, np.nan],
            }
        )
        metrics = compute_metrics(df)
        # Both valid: rows 0,1,2 (3 rows)
        # Reversed: row 0 only (4w>0, 13w<0)
        # Expected: 1/3 ≈ 0.3333
        assert abs(metrics["reversal_rate_4w_to_13w"] - 1.0 / 3.0) < 1e-10

    def test_reversal_rate_no_valid_pairs(self):
        """Reversal rate is NaN when no pair has both returns valid."""
        df = pd.DataFrame(
            {
                "passed_technical": [True, True],
                "excess_return_4w": [np.nan, np.nan],
                "excess_return_13w": [np.nan, np.nan],
            }
        )
        metrics = compute_metrics(df)
        assert np.isnan(metrics["reversal_rate_4w_to_13w"])

    def test_snr_with_zero_std(self):
        """SNR is 0 when std is 0 (all returns identical)."""
        df = pd.DataFrame(
            {
                "passed_technical": [True, True, True],
                "excess_return_4w": [0.05, 0.05, 0.05],
                "excess_return_13w": [0.02, 0.02, 0.02],
            }
        )
        metrics = compute_metrics(df)
        # n_valid_returns = 3, avg = 0.02, std = 0.0 -> snr = 0 (avoid div by 0)
        assert metrics["snr_13w"] == 0.0

    def test_single_observation_snr(self):
        """SNR, std, and SE are NaN when there is exactly 1 passed row
        (std with ddof=1 yields NaN)."""
        df = pd.DataFrame(
            {
                "passed_technical": [True, False],
                "excess_return_4w": [0.05, 0.01],
                "excess_return_13w": [0.02, 0.01],
            }
        )
        metrics = compute_metrics(df)
        assert metrics["n_signals"] == 1
        assert metrics["n_valid_returns"] == 1
        assert np.isnan(metrics["std_excess_13w"])
        assert np.isnan(metrics["snr_13w"])
        assert np.isnan(metrics["se_excess_13w"])
        # avg should still be the single value (not NaN)
        assert metrics["avg_excess_13w"] == 0.02


# ---------------------------------------------------------------------------
# Tests: compute_train_test_metrics
# ---------------------------------------------------------------------------


class TestComputeTrainTestMetrics:
    def test_returns_train_and_test_values(self):
        """Splits data by year and returns train/test averages."""
        df = _make_minimal_trades_df()
        tt = compute_train_test_metrics(df, test_from_year=2025)
        assert "train_avg_excess_13w" in tt
        assert "test_avg_excess_13w" in tt
        assert "train_test_gap" in tt
        # Train: 2024 rows (indices 0-4), passed True: rows 0,1,2
        # excess_13w: 0.05, -0.10, -0.02 -> avg = -0.02333...
        # Test: 2025 rows (index 5), passed True: row 5
        # excess_13w: 0.01
        assert not np.isnan(tt["train_avg_excess_13w"])
        assert not np.isnan(tt["test_avg_excess_13w"])

    def test_returns_nan_when_no_observations_in_split(self):
        """Returns NaN for a split with no valid observations."""
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-05")],
                "passed_technical": [False],
                "excess_return_13w": [0.05],
            }
        )
        tt = compute_train_test_metrics(df, test_from_year=2025)
        assert np.isnan(tt["train_avg_excess_13w"])
        assert np.isnan(tt["test_avg_excess_13w"])
        assert np.isnan(tt["train_test_gap"])

    def test_returns_nan_for_empty_dataframe(self):
        """Returns NaN for an empty DataFrame."""
        tt = compute_train_test_metrics(pd.DataFrame(), test_from_year=2025)
        assert np.isnan(tt["train_avg_excess_13w"])
        assert np.isnan(tt["test_avg_excess_13w"])
        assert np.isnan(tt["train_test_gap"])

    def test_correctly_splits_by_year(self):
        """Observations before test_from_year go to train, others to test."""
        df = _make_minimal_trades_df()  # 2024 dates + one 2025 date
        tt = compute_train_test_metrics(df, test_from_year=2025)
        # Train years are < 2025, test years >= 2025
        train = df[
            (df["passed_technical"] == True)
            & (pd.to_datetime(df["date"]).dt.year < 2025)
        ]["excess_return_13w"].dropna()
        test = df[
            (df["passed_technical"] == True)
            & (pd.to_datetime(df["date"]).dt.year >= 2025)
        ]["excess_return_13w"].dropna()
        expected_train_avg = float(train.mean()) if len(train) > 0 else float("nan")
        expected_test_avg = float(test.mean()) if len(test) > 0 else float("nan")
        if not np.isnan(expected_train_avg) and not np.isnan(expected_test_avg):
            assert abs(tt["train_avg_excess_13w"] - expected_train_avg) < 1e-10
            assert abs(tt["test_avg_excess_13w"] - expected_test_avg) < 1e-10

    def test_compute_train_test_metrics_empty_passed_df(self):
        """All rows have passed_technical=False => passed_df empty => all NaN."""
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-05")],
                "passed_technical": [False],
                "excess_return_13w": [0.05],
            }
        )
        tt = compute_train_test_metrics(df, test_from_year=2025)
        assert np.isnan(tt["train_avg_excess_13w"])
        assert np.isnan(tt["test_avg_excess_13w"])
        assert np.isnan(tt["train_test_gap"])

    def test_compute_train_test_metrics_invalid_dates(self):
        """All dates are unparseable strings => dates.isna().all() => all NaN."""
        df = pd.DataFrame(
            {
                "date": pd.Series(["not_a_date"], dtype=object),
                "passed_technical": [True],
                "excess_return_13w": [0.05],
            }
        )
        tt = compute_train_test_metrics(df, test_from_year=2025)
        assert np.isnan(tt["train_avg_excess_13w"])
        assert np.isnan(tt["test_avg_excess_13w"])
        assert np.isnan(tt["train_test_gap"])

    def test_compute_train_test_metrics_mixed_valid_invalid_dates(self):
        """Only valid dates contribute; NaT-producing rows are excluded."""
        df = pd.DataFrame(
            {
                "date": pd.Series(
                    ["2024-01-05", "not_a_date", "2025-01-03"], dtype=object
                ),
                "passed_technical": [True, True, True],
                "excess_return_13w": [0.05, 0.10, 0.01],
            }
        )
        tt = compute_train_test_metrics(df, test_from_year=2025)
        # Valid dates: 2024-01-05 (train, excess=0.05), 2025-01-03 (test, excess=0.01)
        # "not_a_date" row is excluded from both splits
        assert not np.isnan(tt["train_avg_excess_13w"])
        assert not np.isnan(tt["test_avg_excess_13w"])
        assert abs(tt["train_avg_excess_13w"] - 0.05) < 1e-10
        assert abs(tt["test_avg_excess_13w"] - 0.01) < 1e-10

    def test_compute_train_test_metrics_single_year_all_train(self):
        """All dates < 2025 => test_avg is NaN, train_avg is valid."""
        df = pd.DataFrame(
            {
                "date": [
                    pd.Timestamp("2024-01-05"),
                    pd.Timestamp("2024-06-14"),
                ],
                "passed_technical": [True, True],
                "excess_return_13w": [0.05, -0.02],
            }
        )
        tt = compute_train_test_metrics(df, test_from_year=2025)
        assert not np.isnan(tt["train_avg_excess_13w"])
        assert np.isnan(tt["test_avg_excess_13w"])
        assert np.isnan(tt["train_test_gap"])
        assert abs(tt["train_avg_excess_13w"] - 0.015) < 1e-10

    def test_compute_train_test_metrics_single_year_all_test(self):
        """All dates >= 2025 => train_avg is NaN, test_avg is valid."""
        df = pd.DataFrame(
            {
                "date": [
                    pd.Timestamp("2025-01-03"),
                    pd.Timestamp("2025-06-13"),
                ],
                "passed_technical": [True, True],
                "excess_return_13w": [0.01, 0.03],
            }
        )
        tt = compute_train_test_metrics(df, test_from_year=2025)
        assert np.isnan(tt["train_avg_excess_13w"])
        assert not np.isnan(tt["test_avg_excess_13w"])
        assert np.isnan(tt["train_test_gap"])
        assert abs(tt["test_avg_excess_13w"] - 0.02) < 1e-10

    def test_compute_train_test_metrics_year_boundary(self):
        """Dates at 2024-12-31 and 2025-01-03 split correctly."""
        df = pd.DataFrame(
            {
                "date": [
                    pd.Timestamp("2024-12-31"),
                    pd.Timestamp("2025-01-03"),
                ],
                "passed_technical": [True, True],
                "excess_return_13w": [0.04, 0.02],
            }
        )
        tt = compute_train_test_metrics(df, test_from_year=2025)
        # 2024-12-31 -> train (year < 2025)
        # 2025-01-03 -> test (year >= 2025)
        assert not np.isnan(tt["train_avg_excess_13w"])
        assert not np.isnan(tt["test_avg_excess_13w"])
        assert abs(tt["train_avg_excess_13w"] - 0.04) < 1e-10
        assert abs(tt["test_avg_excess_13w"] - 0.02) < 1e-10

    def test_compute_train_test_metrics_all_nat_logs_warning(self, caplog):
        """logger.warning fires when all dates are NaT."""
        import logging

        caplog.set_level(logging.WARNING)
        df = pd.DataFrame(
            {
                "date": pd.Series(["not_a_date", "also_bad"], dtype=object),
                "passed_technical": [True, True],
                "excess_return_13w": [0.05, -0.02],
            }
        )
        tt = compute_train_test_metrics(df, test_from_year=2025)
        assert np.isnan(tt["train_avg_excess_13w"])
        assert np.isnan(tt["test_avg_excess_13w"])
        assert np.isnan(tt["train_test_gap"])
        assert "all dates are NaT" in caplog.text


# ---------------------------------------------------------------------------
# Tests: evaluate_param_combo
# ---------------------------------------------------------------------------


class TestEvaluateParamCombo:
    def test_returns_expected_keys(self):
        """Result contains combo keys, metric keys, and is_baseline."""
        combo = {"max_distance_from_sma20": 0.10}
        config = _make_minimal_config()
        data = _make_mock_data_for_evaluate()

        with patch(
            "stock_screener.src.grid_search.run_evaluation"
        ) as mock_run:
            mock_run.return_value = _make_minimal_trades_df()
            result = evaluate_param_combo(combo, config, *data, is_baseline=False)

        assert "max_distance_from_sma20" in result
        assert "n_signals" in result
        assert "avg_excess_13w" in result
        assert "is_baseline" in result
        assert result["is_baseline"] is False
        assert result["max_distance_from_sma20"] == 0.10

    def test_config_override_does_not_modify_base(self):
        """The base config dict is not mutated by evaluate_param_combo."""
        combo = {"max_distance_from_sma20": 0.05, "min_relative_strength_13w": -0.05}
        config = _make_minimal_config()
        original_tech = copy.deepcopy(config["technical"])
        data = _make_mock_data_for_evaluate()

        with patch(
            "stock_screener.src.grid_search.run_evaluation"
        ) as mock_run:
            mock_run.return_value = _make_minimal_trades_df()
            evaluate_param_combo(combo, config, *data, is_baseline=False)

        # Original config should be unchanged
        assert config["technical"] == original_tech

    def test_filters_with_custom_params(self):
        """Custom params are passed to run_evaluation in config copy."""
        combo = {
            "max_distance_from_sma20": 0.05,
            "min_relative_strength_13w": -0.05,
            "max_relative_strength_13w": 0.08,
            "require_sma20_rising": False,
        }
        config = _make_minimal_config()
        data = _make_mock_data_for_evaluate()

        with patch(
            "stock_screener.src.grid_search.run_evaluation"
        ) as mock_run:
            mock_run.return_value = _make_minimal_trades_df()
            evaluate_param_combo(combo, config, *data, is_baseline=False)
            mock_run.assert_called_once()
            call_config = mock_run.call_args[0][0]
            for k, v in combo.items():
                assert call_config["technical"][k] == v

    def test_baseline_equivalence(self):
        """evaluate_param_combo passes correct config to run_evaluation
        when baseline values are supplied.

        Verifies that the config passed to ``run_evaluation`` includes the
        baseline parameter values merged into ``config["technical"]``,
        while preserving other technical keys.
        """
        config = _make_minimal_config()
        baseline = {}
        for k in _PARAM_GRID:
            baseline[k] = config["technical"].get(k)

        data = _make_mock_data_for_evaluate()
        expected_trades = _make_minimal_trades_df()

        with patch(
            "stock_screener.src.grid_search.run_evaluation",
            return_value=expected_trades,
        ) as mock_run:
            evaluate_param_combo(baseline, config, *data, is_baseline=True)

        # Verify run_evaluation was called exactly once
        mock_run.assert_called_once()

        # Get the config that was passed to run_evaluation
        call_config = mock_run.call_args[0][0]

        # Verify the technical section contains the baseline values
        for k in baseline:
            assert call_config["technical"][k] == baseline[k], (
                f"Expected {k}={baseline[k]}, got {call_config['technical'][k]}"
            )

        # Verify other technical keys are preserved
        for k in config["technical"]:
            if k not in _PARAM_GRID:
                assert call_config["technical"][k] == config["technical"][k]

    def test_handles_empty_trades_dataframe(self):
        """Returns metrics dict with n_signals=0 when run_evaluation returns empty."""
        combo = {"max_distance_from_sma20": 0.10}
        config = _make_minimal_config()
        data = _make_mock_data_for_evaluate()

        with patch(
            "stock_screener.src.grid_search.run_evaluation"
        ) as mock_run:
            mock_run.return_value = pd.DataFrame()
            result = evaluate_param_combo(combo, config, *data, is_baseline=False)

        assert result["n_signals"] == 0
        assert np.isnan(result["avg_excess_13w"])
        assert np.isnan(result["win_rate_13w"])


# ---------------------------------------------------------------------------
# Tests: write_grid_search_report
# ---------------------------------------------------------------------------


class TestWriteGridSearchReport:
    def _make_sample_results(self) -> list[dict]:
        """Create sample result dicts for report testing."""
        return [
            {
                "max_distance_from_sma20": 0.05,
                "min_relative_strength_13w": -0.05,
                "max_relative_strength_13w": 0.08,
                "require_sma20_rising": True,
                "n_signals": 50,
                "pass_rate": 0.5,
                "n_valid_returns": 45,
                "avg_excess_13w": 0.0345,
                "median_excess_13w": 0.0210,
                "std_excess_13w": 0.12,
                "snr_13w": 0.2875,
                "se_excess_13w": 0.0179,
                "win_rate_13w": 0.65,
                "avg_excess_4w": 0.0123,
                "win_rate_4w": 0.55,
                "reversal_rate_4w_to_13w": 0.15,
                "worst_excess_13w": -0.25,
                "train_avg_excess_13w": 0.04,
                "test_avg_excess_13w": 0.02,
                "train_test_gap": 0.02,
                "is_baseline": True,
            },
            {
                "max_distance_from_sma20": 0.08,
                "min_relative_strength_13w": 0.00,
                "max_relative_strength_13w": 0.10,
                "require_sma20_rising": False,
                "n_signals": 25,
                "pass_rate": 0.3,
                "n_valid_returns": 20,
                "avg_excess_13w": 0.0210,
                "median_excess_13w": 0.0150,
                "std_excess_13w": 0.10,
                "snr_13w": 0.21,
                "se_excess_13w": 0.0224,
                "win_rate_13w": 0.58,
                "avg_excess_4w": 0.0080,
                "win_rate_4w": 0.52,
                "reversal_rate_4w_to_13w": 0.20,
                "worst_excess_13w": -0.30,
                "train_avg_excess_13w": 0.03,
                "test_avg_excess_13w": 0.01,
                "train_test_gap": 0.02,
                "is_baseline": False,
            },
            {
                "max_distance_from_sma20": 0.10,
                "min_relative_strength_13w": -0.05,
                "max_relative_strength_13w": 0.15,
                "require_sma20_rising": True,
                "n_signals": 8,
                "pass_rate": 0.1,
                "n_valid_returns": 7,
                "avg_excess_13w": 0.0150,
                "median_excess_13w": 0.0100,
                "std_excess_13w": 0.08,
                "snr_13w": 0.1875,
                "se_excess_13w": 0.0302,
                "win_rate_13w": 0.52,
                "avg_excess_4w": 0.0050,
                "win_rate_4w": 0.50,
                "reversal_rate_4w_to_13w": 0.25,
                "worst_excess_13w": -0.15,
                "train_avg_excess_13w": 0.02,
                "test_avg_excess_13w": 0.00,
                "train_test_gap": 0.02,
                "is_baseline": False,
            },
        ]

    def test_creates_output_directory(self, tmp_path):
        """Output directory is created if it does not exist."""
        out_dir = tmp_path / "grid_reports" / "nested"
        results = self._make_sample_results()
        write_grid_search_report(
            results, str(out_dir), sort_by="avg_excess_13w"
        )
        assert os.path.isdir(out_dir)

    def test_creates_csv_file(self, tmp_path):
        """A CSV file is created in the output directory."""
        out_dir = tmp_path / "out"
        results = self._make_sample_results()
        path = write_grid_search_report(results, str(out_dir))
        assert os.path.isfile(path)
        assert path.endswith(".csv")

    def test_csv_is_sorted_by_specified_metric(self, tmp_path):
        """CSV rows are sorted descending by sort_by column."""
        out_dir = tmp_path / "sorted"
        results = self._make_sample_results()
        path = write_grid_search_report(
            results, str(out_dir), sort_by="avg_excess_13w"
        )
        df = pd.read_csv(path)
        avg_vals = df["avg_excess_13w"].dropna().values
        for i in range(len(avg_vals) - 1):
            assert avg_vals[i] >= avg_vals[i + 1] or pd.isna(avg_vals[i + 1])

    def test_flags_configs_below_min_signals(self, tmp_path):
        """Configs with n_signals < min_signals have low_sample_warning=True."""
        out_dir = tmp_path / "flags"
        results = self._make_sample_results()
        path = write_grid_search_report(
            results, str(out_dir), min_signals=30
        )
        df = pd.read_csv(path)
        # Second config has n_signals=25 < 30, should be flagged
        for _, row in df.iterrows():
            if row["n_signals"] < 30:
                assert row["low_sample_warning"] == True
            else:
                assert row["low_sample_warning"] == False

    def test_includes_metadata_columns(self, tmp_path):
        """CSV includes _timestamp, _data_range, _config_path columns."""
        out_dir = tmp_path / "meta"
        results = self._make_sample_results()
        path = write_grid_search_report(
            results,
            str(out_dir),
            data_range="2024-01 to 2025-12",
            config_path="/path/to/config.yaml",
        )
        df = pd.read_csv(path)
        assert "_timestamp" in df.columns
        assert "_data_range" in df.columns
        assert "_config_path" in df.columns
        assert (df["_data_range"] == "2024-01 to 2025-12").all()
        assert (df["_config_path"] == "/path/to/config.yaml").all()

    def test_excludes_configs_below_hard_floor(self, tmp_path):
        """Configs with n_signals < hard_floor are excluded from report."""
        out_dir = tmp_path / "floor"
        results = self._make_sample_results()  # n_signals: 50, 25, 8
        path = write_grid_search_report(
            results, str(out_dir), hard_floor=10
        )
        df = pd.read_csv(path)
        # Config with n_signals=8 (index 2) should be excluded
        assert len(df) >= 2  # at least 2 configs (excluding 8-signal one)
        assert (df["n_signals"] >= 10).all()

    def test_rejects_invalid_sort_by_metric(self, tmp_path):
        """Raises ValueError for non-existent sort_by column."""
        out_dir = tmp_path / "invalid"
        results = self._make_sample_results()
        with pytest.raises(ValueError, match="sort_by metric"):
            write_grid_search_report(
                results, str(out_dir), sort_by="nonexistent_metric"
            )

    def test_all_configs_below_hard_floor_fallback(self, tmp_path):
        """When all configs are below hard floor, all rows are included
        via fallback (report is not empty)."""
        out_dir = tmp_path / "fallback"
        results = self._make_sample_results()
        for r in results:
            r["n_signals"] = 5
        path = write_grid_search_report(
            results, str(out_dir), hard_floor=10
        )
        df = pd.read_csv(path)
        # All 3 configs should be present (fallback includes all)
        assert len(df) == 3

    def test_empty_results_list(self, tmp_path):
        """Empty results list produces a valid CSV with no data rows."""
        out_dir = tmp_path / "empty"
        path = write_grid_search_report([], str(out_dir))
        df = pd.read_csv(path)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests: print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def _make_sample_with_baseline(self) -> tuple:
        """Return (sorted_results, baseline_result) for print testing."""
        results = [
            {
                "max_distance_from_sma20": 0.05,
                "min_relative_strength_13w": -0.05,
                "max_relative_strength_13w": 0.08,
                "require_sma20_rising": True,
                "n_signals": 50,
                "avg_excess_13w": 0.0345,
                "win_rate_13w": 0.65,
                "avg_excess_4w": 0.0123,
                "win_rate_4w": 0.55,
                "snr_13w": 0.2875,
                "reversal_rate_4w_to_13w": 0.15,
                "train_avg_excess_13w": 0.04,
                "test_avg_excess_13w": 0.02,
                "train_test_gap": 0.02,
                "is_baseline": False,
            },
            {
                "max_distance_from_sma20": 0.15,
                "min_relative_strength_13w": 0.0,
                "max_relative_strength_13w": None,
                "require_sma20_rising": True,
                "n_signals": 5,
                "avg_excess_13w": 0.0210,
                "win_rate_13w": 0.58,
                "avg_excess_4w": 0.0080,
                "win_rate_4w": 0.52,
                "snr_13w": 0.21,
                "reversal_rate_4w_to_13w": 0.20,
                "train_avg_excess_13w": 0.03,
                "test_avg_excess_13w": 0.01,
                "train_test_gap": 0.02,
                "is_baseline": True,
            },
        ]
        baseline = results[1]
        return results, baseline

    def test_prints_without_crash(self, capsys):
        """print_summary runs without raising exceptions."""
        results, baseline = self._make_sample_with_baseline()
        print_summary(results, baseline, top_n=5)
        captured = capsys.readouterr()
        assert "GRID SEARCH SUMMARY" in captured.out

    def test_shows_top_n_configs(self, capsys):
        """Top configs and key columns are visible in output."""
        results, baseline = self._make_sample_with_baseline()
        print_summary(results, baseline, top_n=5)
        captured = capsys.readouterr()
        assert "n_signals" in captured.out or "Avg_Excess" in captured.out

    def test_shows_baseline_comparison(self, capsys):
        """Baseline parameters and metrics are displayed."""
        results, baseline = self._make_sample_with_baseline()
        print_summary(results, baseline, top_n=5)
        captured = capsys.readouterr()
        assert "Baseline" in captured.out
        assert "max_distance_from_sma20" in captured.out

    def test_prints_selection_bias_warning(self, capsys):
        """Selection bias warning is shown when best differs from baseline."""
        results, baseline = self._make_sample_with_baseline()
        print_summary(results, baseline, top_n=5)
        captured = capsys.readouterr()
        assert "SELECTION BIAS" in captured.out

    def test_no_baseline_no_crash(self, capsys):
        """print_summary works when baseline_result is None."""
        results, _ = self._make_sample_with_baseline()
        print_summary(results, baseline_result=None, top_n=5)
        captured = capsys.readouterr()
        assert "GRID SEARCH SUMMARY" in captured.out


# ---------------------------------------------------------------------------
# Tests: Config backward compatibility
# ---------------------------------------------------------------------------


class TestConfigBackwardCompat:
    def test_new_keys_have_correct_defaults(self):
        """Real config.yaml has expected default values for grid search keys."""
        from stock_screener.src.config import load_config

        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(here, ".."))
        config_path = os.path.join(project_root, "config.yaml")
        config = load_config(config_path)
        tech = config["technical"]

        assert tech.get("min_relative_strength_13w") == 0.0
        assert tech.get("max_relative_strength_13w") is None
        assert tech.get("require_sma20_rising") is True
        assert tech.get("max_distance_from_sma20") is not None


# ---------------------------------------------------------------------------
# Tests: Integration (end-to-end with minimal fixture)
# ---------------------------------------------------------------------------


class TestRunGridSearchIntegration:
    @patch("stock_screener.src.grid_search.load_grid_data")
    @patch("stock_screener.src.grid_search.load_config")
    @patch("stock_screener.src.grid_search.write_grid_search_report")
    @patch("stock_screener.src.grid_search.run_evaluation")
    def test_end_to_end_with_minimal_fixture(
        self,
        mock_run_evaluation,
        mock_write_report,
        mock_load_config,
        mock_load_grid_data,
        tmp_path,
    ):
        """End-to-end run_grid_search completes without error with mocked IO."""
        # Setup mocks
        cfg = _make_minimal_config()
        mock_load_config.return_value = cfg

        data = _make_mock_data_for_evaluate()
        mock_load_grid_data.return_value = data

        mock_run_evaluation.return_value = _make_minimal_trades_df()
        mock_write_report.return_value = str(tmp_path / "report.csv")

        # Run grid search with custom args pointing to tmp
        test_args = [
            "grid_search.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--output-dir",
            str(tmp_path / "reports"),
            "--top-n",
            "3",
            "--min-signals",
            "1",
            "--sort-by",
            "n_signals",
        ]
        with patch.object(sys, "argv", test_args):
            try:
                run_grid_search()
            except SystemExit as exc:
                # Some code paths call sys.exit on error - should not happen here
                assert exc.code != 1, (
                    f"run_grid_search exited with code {exc.code}"
                )

        mock_write_report.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: run_grid_search error paths
# ---------------------------------------------------------------------------


class TestRunGridSearchErrors:
    @patch("stock_screener.src.grid_search.load_config")
    def test_config_load_failure_exits(self, mock_load_config):
        """run_grid_search exits when config loading fails."""
        mock_load_config.side_effect = ValueError("Bad config")
        test_args = ["grid_search.py", "--config", "/nonexistent/config.yaml"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                run_grid_search()
        assert exc_info.value.code == 1

    @patch("stock_screener.src.grid_search.load_grid_data")
    @patch("stock_screener.src.grid_search.load_config")
    def test_data_loading_failure_exits(
        self, mock_load_config, mock_load_grid_data
    ):
        """run_grid_search exits when data loading fails."""
        cfg = _make_minimal_config()
        mock_load_config.return_value = cfg
        mock_load_grid_data.side_effect = RuntimeError("No data")

        test_args = ["grid_search.py", "--config", "/dummy/path.yaml"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                run_grid_search()
        assert exc_info.value.code == 1
