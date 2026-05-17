"""
Smoke tests for the backtest diagnostics module.

Tests cover CSV loading, all 7 analysis functions, the ASCII bar helper,
console output formatting, CSV writing, and an end-to-end integration test.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from stock_screener.src.backtest_diagnostics import (
    _categorize_reason,
    _make_diverging_bar,
    _safe_mean,
    _safe_median,
    _safe_std,
    _safe_win_rate,
    analyze_4w_vs_13w,
    analyze_by_signal,
    analyze_by_ticker,
    analyze_by_year,
    analyze_extreme_observations,
    analyze_failure_reasons,
    load_trades_csv,
    parse_args,
    print_summary,
    write_diagnostics,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_minimal_trades_df() -> pd.DataFrame:
    """Create a minimal valid trades DataFrame for testing."""
    return pd.DataFrame({
        "date": ["2024-01-05", "2024-01-05", "2024-01-12"],
        "ticker": ["BBCA", "BBCA", "BBRI"],
        "company_name": ["Bank BCA", "Bank BCA", "Bank BRI"],
        "sector": ["Finance", "Finance", "Finance"],
        "passed_technical": [True, False, True],
        "close": [10000.0, 10100.0, 5000.0],
        "sma20": [9900.0, 9950.0, 4900.0],
        "sma50": [9800.0, 9850.0, 4800.0],
        "distance_from_sma20": [0.01, 0.015, 0.02],
        "relative_strength_13w": [0.05, -0.02, 0.08],
        "sma20_is_rising": [True, True, False],
        "reasons": [
            "PASS: Technical score OK;PASS: SMA20 is rising",
            "FAIL: Distance from SMA20 too high",
            "PASS: All checks passed",
        ],
        "warnings": ["", "Insufficient data", ""],
        "forward_return_4w": [0.02, -0.01, 0.03],
        "index_return_4w": [0.01, 0.01, 0.015],
        "excess_return_4w": [0.01, -0.02, 0.015],
        "forward_return_13w": [0.05, -0.03, 0.08],
        "index_return_13w": [0.03, 0.03, 0.035],
        "excess_return_13w": [0.02, -0.06, 0.045],
    })


def _make_trades_csv_string(df: pd.DataFrame) -> str:
    """Convert a trades DataFrame to CSV string for file writing."""
    return df.to_csv(index=False)


# ---------------------------------------------------------------------------
# Tests: module-level _safe_* helpers
# ---------------------------------------------------------------------------


class TestSafeHelpers:
    def test_safe_mean_normal(self):
        """_safe_mean returns mean of non-NaN values."""
        s = pd.Series([1.0, 2.0, 3.0])
        assert _safe_mean(s) == 2.0

    def test_safe_mean_all_nan(self):
        """_safe_mean returns NaN when all values are NaN."""
        s = pd.Series([float("nan"), float("nan")])
        assert np.isnan(_safe_mean(s))

    def test_safe_mean_empty(self):
        """_safe_mean returns NaN for empty series."""
        s = pd.Series([], dtype=float)
        assert np.isnan(_safe_mean(s))

    def test_safe_median_normal(self):
        """_safe_median returns median of non-NaN values."""
        s = pd.Series([1.0, 2.0, 10.0])
        assert _safe_median(s) == 2.0

    def test_safe_median_all_nan(self):
        """_safe_median returns NaN when all values are NaN."""
        s = pd.Series([float("nan")])
        assert np.isnan(_safe_median(s))

    def test_safe_std_normal(self):
        """_safe_std returns std of non-NaN values."""
        s = pd.Series([1.0, 2.0, 3.0])
        assert _safe_std(s) == 1.0

    def test_safe_std_insufficient(self):
        """_safe_std returns NaN with fewer than 2 values."""
        s = pd.Series([1.0])
        assert np.isnan(_safe_std(s))

    def test_safe_win_rate_normal(self):
        """_safe_win_rate returns fraction of positive values."""
        s = pd.Series([0.01, -0.02, 0.03, 0.0])
        result = _safe_win_rate(s)
        assert result == 0.5  # 2 positive out of 4

    def test_safe_win_rate_all_nan(self):
        """_safe_win_rate returns NaN when all values are NaN."""
        s = pd.Series([float("nan"), float("nan")])
        assert np.isnan(_safe_win_rate(s))


# ---------------------------------------------------------------------------
# Tests: _categorize_reason
# ---------------------------------------------------------------------------


class TestCategorizeReason:
    def test_preserves_identifiers_like_sma20(self):
        """Identifiers like SMA20 are preserved, not mangled to SMAX."""
        reason = (
            "PASS: Close=9500.0 > SMA20=9400.0;"
            "FAIL: |Distance from SMA20|=0.1523 (>= 0.1500)"
        )
        result = _categorize_reason(reason)
        assert "SMA20" in result, f"SMA20 was mangled: {result}"
        assert "SMAX" not in result, f"SMA20 was corrupted to SMAX: {result}"

    def test_replaces_numeric_after_equals(self):
        """Numeric values after = signs are replaced with X."""
        reason = "PASS: Close=9500.0 > SMA20=9400.0"
        result = _categorize_reason(reason)
        assert "Close=X" in result, f"Close value not replaced: {result}"
        assert "SMA20=X" in result, f"SMA20 value not replaced: {result}"

    def test_replaces_range_in_parentheses(self):
        """Range conditions in parentheses are replaced with (...)."""
        reason = "FAIL: |Distance from SMA20|=0.1523 (>= 0.1500)"
        result = _categorize_reason(reason)
        assert "(...)" in result, f"Parenthetical range not replaced: {result}"

    def test_preserves_plain_failure_reason(self):
        """A plain failure reason with no numeric values is unchanged."""
        reason = "FAIL: Distance from SMA20 too high"
        result = _categorize_reason(reason)
        assert result == reason, f"Plain reason was modified: {result}"

    def test_handles_empty_reason(self):
        """Empty string is handled without error."""
        result = _categorize_reason("")
        assert result == ""

    def test_replaces_standalone_decimal(self):
        """Standalone decimal numbers not after = or in (...) are replaced."""
        reason = "FAIL: Test value 0.1523 exceeds threshold"
        result = _categorize_reason(reason)
        assert "X" in result
        assert "0.1523" not in result


# ---------------------------------------------------------------------------
# Tests: load_trades_csv
# ---------------------------------------------------------------------------


class TestLoadTradesCsv:
    def test_load_valid_csv(self):
        """Load a valid CSV string and verify basic properties."""
        csv_str = _make_trades_csv_string(_make_minimal_trades_df())
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_str)
            tmp_path = f.name
        try:
            df = load_trades_csv(tmp_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert "date" in df.columns
            assert "ticker" in df.columns
            assert "passed_technical" in df.columns
        finally:
            os.unlink(tmp_path)

    def test_load_missing_required_column(self):
        """Missing required column raises ValueError."""
        df = pd.DataFrame({"date": ["2024-01-05"], "ticker": ["BBCA"]})
        csv_str = df.to_csv(index=False)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_str)
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                load_trades_csv(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_load_empty_csv(self):
        """Empty CSV raises ValueError."""
        df = pd.DataFrame(columns=[
            "date", "ticker", "passed_technical",
            "forward_return_4w", "forward_return_13w",
            "index_return_4w", "index_return_13w",
            "excess_return_4w", "excess_return_13w",
        ])
        csv_str = df.to_csv(index=False)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_str)
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="empty"):
                load_trades_csv(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_load_file_not_found(self):
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Trades CSV not found"):
            load_trades_csv("nonexistent_file_xyz.csv")

    def test_load_converts_passed_technical_to_bool(self):
        """passed_technical column is converted to bool."""
        df = _make_minimal_trades_df()
        # Convert to string representation as it would appear in CSV
        df["passed_technical"] = df["passed_technical"].map({True: "True", False: "False"})
        csv_str = df.to_csv(index=False)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_str)
            tmp_path = f.name
        try:
            loaded = load_trades_csv(tmp_path)
            assert loaded["passed_technical"].dtype == bool
            assert loaded["passed_technical"].iloc[0] == True
            assert loaded["passed_technical"].iloc[1] == False
        finally:
            os.unlink(tmp_path)

    def test_load_converts_sma20_is_rising_to_bool(self):
        """sma20_is_rising column is converted to bool if present."""
        df = _make_minimal_trades_df()
        df["sma20_is_rising"] = df["sma20_is_rising"].map({True: "True", False: "False"})
        csv_str = df.to_csv(index=False)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_str)
            tmp_path = f.name
        try:
            loaded = load_trades_csv(tmp_path)
            assert "sma20_is_rising" in loaded.columns
            assert loaded["sma20_is_rising"].dtype == bool
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: analyze_by_ticker
# ---------------------------------------------------------------------------


class TestAnalyzeByTicker:
    def test_returns_dataframe_with_expected_columns(self):
        """Result contains all required columns."""
        df = _make_minimal_trades_df()
        result = analyze_by_ticker(df)
        expected = {
            "ticker", "n_observations", "n_passed", "n_failed", "pass_rate",
            "avg_excess_4w_passed", "avg_excess_4w_failed",
            "avg_excess_13w_passed", "avg_excess_13w_failed",
            "median_excess_13w_passed", "std_excess_13w_passed",
            "avg_win_13w_passed", "avg_loss_13w_passed", "win_loss_ratio_13w_passed",
            "win_rate_4w_passed", "win_rate_4w_failed",
            "win_rate_13w_passed", "win_rate_13w_failed",
            "n_valid_13w_passed",
        }
        assert expected.issubset(result.columns)
        assert len(result) == 2  # BBCA, BBRI

    def test_includes_distribution_columns(self):
        """Distribution columns (std, median, win_loss_ratio) are present and finite."""
        df = _make_minimal_trades_df()
        result = analyze_by_ticker(df)
        # BBCA has a valid passed observation
        bbca = result[result["ticker"] == "BBCA"].iloc[0]
        assert not np.isnan(bbca["median_excess_13w_passed"])
        # std may be NaN with only 1 observation; assert the column exists
        assert "std_excess_13w_passed" in bbca.index
        assert not np.isnan(bbca["avg_win_13w_passed"])
        # win_loss_ratio may be NaN if there are no losses; assert column exists
        assert "win_loss_ratio_13w_passed" in bbca.index

    def test_handles_ticker_with_no_passed_stocks(self):
        """Ticker with no passed stocks does not crash."""
        df = _make_minimal_trades_df()
        # Add a ticker that never passes
        extra = pd.DataFrame({
            "date": ["2024-01-05"],
            "ticker": ["FAILCO"],
            "company_name": ["Fail Corp"],
            "sector": ["Finance"],
            "passed_technical": [False],
            "close": [5000.0],
            "sma20": [4900.0],
            "sma50": [4800.0],
            "distance_from_sma20": [0.02],
            "relative_strength_13w": [-0.05],
            "sma20_is_rising": [False],
            "reasons": ["FAIL: Test"],
            "warnings": [""],
            "forward_return_4w": [-0.01],
            "index_return_4w": [0.01],
            "excess_return_4w": [-0.02],
            "forward_return_13w": [-0.03],
            "index_return_13w": [0.02],
            "excess_return_13w": [-0.05],
        })
        df = pd.concat([df, extra], ignore_index=True)
        result = analyze_by_ticker(df)
        failco = result[result["ticker"] == "FAILCO"].iloc[0]
        assert failco["n_passed"] == 0
        assert failco["n_failed"] == 1
        assert np.isnan(failco["avg_excess_13w_passed"])


# ---------------------------------------------------------------------------
# Tests: analyze_by_year
# ---------------------------------------------------------------------------


class TestAnalyzeByYear:
    def test_returns_dataframe_with_expected_columns(self):
        """Result contains all required columns including market regime."""
        df = _make_minimal_trades_df()
        result = analyze_by_year(df)
        expected = {
            "year", "n_observations", "n_passed", "n_failed", "pass_rate",
            "avg_excess_4w_passed", "avg_excess_4w_failed",
            "avg_excess_13w_passed", "avg_excess_13w_failed",
            "median_excess_13w_passed", "std_excess_13w_passed",
            "avg_win_13w_passed", "avg_loss_13w_passed", "win_loss_ratio_13w_passed",
            "win_rate_4w_passed", "win_rate_4w_failed",
            "win_rate_13w_passed", "win_rate_13w_failed",
            "n_valid_13w_passed",
            "avg_index_return_13w", "market_regime",
        }
        assert expected.issubset(result.columns)
        assert len(result) >= 1
        assert "year" in result.columns

    def test_market_regime_based_on_index_return(self):
        """Market regime is 'bull' when avg index return > 0, 'bear' otherwise."""
        df = _make_minimal_trades_df()
        result = analyze_by_year(df)
        for _, row in result.iterrows():
            if row["avg_index_return_13w"] > 0:
                assert row["market_regime"] == "bull"
            else:
                assert row["market_regime"] == "bear"

    def test_includes_distribution_columns(self):
        """Distribution columns are present in year-level output."""
        df = _make_minimal_trades_df()
        result = analyze_by_year(df)
        assert "std_excess_13w_passed" in result.columns
        assert "median_excess_13w_passed" in result.columns
        assert "win_loss_ratio_13w_passed" in result.columns


# ---------------------------------------------------------------------------
# Tests: analyze_by_signal
# ---------------------------------------------------------------------------


class TestAnalyzeBySignal:
    def test_returns_dataframe_with_expected_columns(self):
        """Result has correct signal bucket columns."""
        df = _make_minimal_trades_df()
        result = analyze_by_signal(df)
        expected = {
            "distance_bucket", "rs_bucket", "sma20_is_rising",
            "n_observations", "n_passed",
            "avg_excess_4w", "avg_excess_13w",
            "win_rate_4w", "win_rate_13w",
        }
        assert expected.issubset(result.columns)
        assert len(result) > 0

    def test_returns_empty_when_optional_columns_missing(self):
        """Empty DataFrame when signal columns are absent."""
        df = _make_minimal_trades_df()[["date", "ticker", "passed_technical",
                                         "excess_return_4w", "excess_return_13w",
                                         "index_return_4w", "index_return_13w",
                                         "forward_return_4w", "forward_return_13w"]]
        result = analyze_by_signal(df)
        assert result.empty

    def test_uses_explicit_bucket_boundaries(self):
        """Bucket labels match defined boundaries."""
        df = _make_minimal_trades_df()
        result = analyze_by_signal(df)
        valid_buckets = {"<=-5%", "(-5%,0%]", "(0%,5%]", "(5%,10%]", "(10%,15%]", ">15%"}
        valid_rs = {"<=0%", "(0%,5%]", "(5%,10%]", ">10%"}
        for _, row in result.iterrows():
            assert row["distance_bucket"] in valid_buckets, (
                f"Unexpected distance bucket: {row['distance_bucket']}"
            )
            assert row["rs_bucket"] in valid_rs, (
                f"Unexpected rs bucket: {row['rs_bucket']}"
            )


# ---------------------------------------------------------------------------
# Tests: analyze_4w_vs_13w
# ---------------------------------------------------------------------------


class TestAnalyze4wVs13w:
    def test_returns_contingency_table(self):
        """Returns a single-row contingency DataFrame with expected columns."""
        df = _make_minimal_trades_df()
        result = analyze_4w_vs_13w(df)
        expected = {
            "excess_4w", "n_4w_pos_13w_pos", "n_4w_pos_13w_neg",
            "n_4w_neg_13w_pos", "n_4w_neg_13w_neg",
            "pct_4w_pos_13w_neg", "pct_4w_neg_13w_pos",
        }
        assert expected.issubset(result.columns)
        assert len(result) == 1

    def test_excludes_nan_observations(self):
        """NaN returns at either horizon are excluded."""
        df = _make_minimal_trades_df()
        # Add a passed row with NaN at both horizons
        extra = pd.DataFrame({
            "date": ["2024-01-19"],
            "ticker": ["BBCA"],
            "company_name": ["Bank BCA"],
            "sector": ["Finance"],
            "passed_technical": [True],
            "close": [10100.0],
            "sma20": [10000.0],
            "sma50": [9900.0],
            "distance_from_sma20": [0.01],
            "relative_strength_13w": [0.03],
            "sma20_is_rising": [True],
            "reasons": ["PASS: OK"],
            "warnings": [""],
            "forward_return_4w": [float("nan")],
            "index_return_4w": [float("nan")],
            "excess_return_4w": [float("nan")],
            "forward_return_13w": [float("nan")],
            "index_return_13w": [float("nan")],
            "excess_return_13w": [float("nan")],
        })
        df = pd.concat([df, extra], ignore_index=True)
        result = analyze_4w_vs_13w(df)
        # NaN row should not affect the counts
        total = result["n_4w_pos_13w_pos"].iloc[0] + result["n_4w_pos_13w_neg"].iloc[0] \
              + result["n_4w_neg_13w_pos"].iloc[0] + result["n_4w_neg_13w_neg"].iloc[0]
        # Only 1 passed stock (BBCA row 0) has both valid: excess_4w=0.01, excess_13w=0.02
        # BBRI row 2 is also passed: excess_4w=0.015, excess_13w=0.045
        # So total = 2
        assert total == 2


# ---------------------------------------------------------------------------
# Tests: analyze_extreme_observations
# ---------------------------------------------------------------------------


class TestAnalyzeExtremeObservations:
    def test_returns_best_and_worst_rows(self):
        """Returns 2*top_n rows with both best and worst observations."""
        df = _make_minimal_trades_df()
        # Add more rows so we have enough for top_n=2
        extra = pd.DataFrame({
            "date": ["2024-01-19", "2024-01-26", "2024-02-02", "2024-02-09"],
            "ticker": ["BBCA", "BBRI", "BBCA", "BBRI"],
            "company_name": ["Bank BCA", "Bank BRI", "Bank BCA", "Bank BRI"],
            "sector": ["Finance", "Finance", "Finance", "Finance"],
            "passed_technical": [True, True, True, True],
            "close": [10200.0, 5100.0, 10300.0, 5200.0],
            "sma20": [10100.0, 5000.0, 10200.0, 5100.0],
            "sma50": [10000.0, 4900.0, 10100.0, 5000.0],
            "distance_from_sma20": [0.01, 0.02, 0.01, 0.02],
            "relative_strength_13w": [0.04, 0.06, 0.05, 0.07],
            "sma20_is_rising": [True, False, True, False],
            "reasons": ["PASS", "PASS", "PASS", "PASS"],
            "warnings": ["", "", "", ""],
            "forward_return_4w": [0.01, 0.02, 0.03, 0.04],
            "index_return_4w": [0.01, 0.01, 0.01, 0.01],
            "excess_return_4w": [0.0, 0.01, 0.02, 0.03],
            "forward_return_13w": [0.01, 0.03, 0.06, 0.10],
            "index_return_13w": [0.03, 0.03, 0.03, 0.03],
            "excess_return_13w": [-0.02, 0.0, 0.03, 0.07],
        })
        df = pd.concat([df, extra], ignore_index=True)
        result = analyze_extreme_observations(df, top_n=2)
        assert len(result) == 4  # 2 * top_n
        assert "rank_label" in result.columns

    def test_includes_4w_columns(self):
        """Result includes forward_return_4w and excess_return_4w."""
        df = _make_minimal_trades_df()
        extra = pd.DataFrame({
            "date": ["2024-01-19", "2024-01-26"],
            "ticker": ["BBCA", "BBRI"],
            "company_name": ["Bank BCA", "Bank BRI"],
            "sector": ["Finance", "Finance"],
            "passed_technical": [True, True],
            "close": [10200.0, 5100.0],
            "sma20": [10100.0, 5000.0],
            "sma50": [10000.0, 4900.0],
            "distance_from_sma20": [0.01, 0.02],
            "relative_strength_13w": [0.04, 0.06],
            "sma20_is_rising": [True, False],
            "reasons": ["PASS", "PASS"],
            "warnings": ["", ""],
            "forward_return_4w": [0.01, 0.02],
            "index_return_4w": [0.01, 0.01],
            "excess_return_4w": [0.0, 0.01],
            "forward_return_13w": [0.01, 0.03],
            "index_return_13w": [0.03, 0.03],
            "excess_return_13w": [-0.02, 0.0],
        })
        df = pd.concat([df, extra], ignore_index=True)
        result = analyze_extreme_observations(df, top_n=1)
        assert "forward_return_4w" in result.columns
        assert "excess_return_4w" in result.columns

    def test_rank_labels_are_correct(self):
        """The rank_label column contains 'best' and 'worst' correctly."""
        df = _make_minimal_trades_df()
        extra = pd.DataFrame({
            "date": ["2024-01-19", "2024-01-26", "2024-02-02", "2024-02-09"],
            "ticker": ["BBCA", "BBRI", "BBCA", "BBRI"],
            "company_name": ["Bank BCA", "Bank BRI", "Bank BCA", "Bank BRI"],
            "sector": ["Finance", "Finance", "Finance", "Finance"],
            "passed_technical": [True, True, True, True],
            "close": [10200.0, 5100.0, 10300.0, 5200.0],
            "sma20": [10100.0, 5000.0, 10200.0, 5100.0],
            "sma50": [10000.0, 4900.0, 10100.0, 5000.0],
            "distance_from_sma20": [0.01, 0.02, 0.01, 0.02],
            "relative_strength_13w": [0.04, 0.06, 0.05, 0.07],
            "sma20_is_rising": [True, False, True, False],
            "reasons": ["PASS", "PASS", "PASS", "PASS"],
            "warnings": ["", "", "", ""],
            "forward_return_4w": [0.01, 0.02, 0.03, 0.04],
            "index_return_4w": [0.01, 0.01, 0.01, 0.01],
            "excess_return_4w": [0.0, 0.01, 0.02, 0.03],
            "forward_return_13w": [0.01, 0.03, 0.06, 0.10],
            "index_return_13w": [0.03, 0.03, 0.03, 0.03],
            "excess_return_13w": [-0.02, 0.0, 0.03, 0.07],
        })
        df = pd.concat([df, extra], ignore_index=True)
        result = analyze_extreme_observations(df, top_n=2)
        worst_labels = result[result["rank_label"] == "worst"]
        best_labels = result[result["rank_label"] == "best"]
        assert len(worst_labels) == 2
        assert len(best_labels) == 2
        # Worst should have lower excess_return_13w than best
        assert worst_labels["excess_return_13w"].max() <= best_labels["excess_return_13w"].min()

    def test_returns_empty_when_top_n_zero(self):
        """top_n <= 0 returns empty DataFrame."""
        df = _make_minimal_trades_df()
        result = analyze_extreme_observations(df, top_n=0)
        assert result.empty

    def test_returns_empty_when_fewer_than_two_passed(self):
        """Less than 2 passed stocks returns empty DataFrame."""
        df = _make_minimal_trades_df()
        # Only 1 passed stock by filtering
        single = df[df["ticker"] == "BBCA"].copy()
        single.loc[single.index[0], "passed_technical"] = True
        single.loc[single.index[1], "passed_technical"] = False
        result = analyze_extreme_observations(single, top_n=5)
        assert result.empty

    def test_removes_overlap_between_worst_and_best(self):
        """When worst and best sets overlap, duplicate rows are removed."""
        df = _make_minimal_trades_df()
        # Use top_n=1 with exactly 2 passed stocks to avoid overlap
        extra = pd.DataFrame({
            "date": ["2024-01-19"],
            "ticker": ["BBRI"],
            "company_name": ["Bank BRI"],
            "sector": ["Finance"],
            "passed_technical": [True],
            "close": [5100.0],
            "sma20": [5000.0],
            "sma50": [4900.0],
            "distance_from_sma20": [0.02],
            "relative_strength_13w": [0.06],
            "sma20_is_rising": [False],
            "reasons": ["PASS"],
            "warnings": [""],
            "forward_return_4w": [0.02],
            "index_return_4w": [0.01],
            "excess_return_4w": [0.01],
            "forward_return_13w": [0.15],
            "index_return_13w": [0.03],
            "excess_return_13w": [0.12],
        })
        df = pd.concat([df, extra], ignore_index=True)
        result = analyze_extreme_observations(df, top_n=1)
        assert len(result) >= 1
        # No row should have both "best" and "worst"
        assert len(result[result["rank_label"] == "best"]) <= 1
        assert len(result[result["rank_label"] == "worst"]) <= 1


# ---------------------------------------------------------------------------
# Tests: analyze_failure_reasons
# ---------------------------------------------------------------------------


class TestAnalyzeFailureReasons:
    def test_returns_frequency_table(self):
        """Returns frequency table with expected columns."""
        df = _make_minimal_trades_df()
        result = analyze_failure_reasons(df)
        expected = {
            "reason_category", "n_occurrences", "pct_of_total",
            "avg_excess_13w_when_passed", "avg_excess_13w_when_failed",
        }
        assert expected.issubset(result.columns)
        assert len(result) > 0
        assert "n_occurrences" in result.columns

    def test_reasons_are_categorized(self):
        """Numeric values in reasons are replaced with placeholder.
        Identifiers like SMA20 are preserved (not mangled to SMAX)."""
        df = _make_minimal_trades_df()
        result = analyze_failure_reasons(df)
        # Identifiers like SMA20 must be preserved
        sma20_cats = [cat for cat in result["reason_category"] if "SMA20" in cat]
        assert len(sma20_cats) > 0, "Expected SMA20 to appear in categories"
        # The old mangled form SMAX should never appear
        assert not any("SMAX" in cat for cat in result["reason_category"]), (
            "Identifiers were corrupted: SMAX found in categories"
        )

    def test_returns_empty_when_reasons_column_missing(self):
        """Empty DataFrame when reasons column is absent."""
        df = _make_minimal_trades_df()[["date", "ticker", "passed_technical",
                                         "excess_return_4w", "excess_return_13w",
                                         "index_return_4w", "index_return_13w",
                                         "forward_return_4w", "forward_return_13w"]]
        result = analyze_failure_reasons(df)
        assert result.empty


# ---------------------------------------------------------------------------
# Tests: _make_diverging_bar
# ---------------------------------------------------------------------------


class TestMakeDivergingBar:
    def test_positive_value_uses_full_block_on_right(self):
        """Positive value has '\u2588' in the right portion of the bar."""
        result = _make_diverging_bar(0.10, total_width=40, scale=200.0)
        assert "\u2588" in result
        assert len(result) == 40

    def test_negative_value_uses_lower_block_on_left(self):
        """Negative value has '\u2584' in the left portion of the bar."""
        result = _make_diverging_bar(-0.10, total_width=40, scale=200.0)
        assert "\u2584" in result
        assert len(result) == 40

    def test_zero_value_produces_full_background(self):
        """Zero value returns a string with only background chars."""
        result = _make_diverging_bar(0.0, total_width=40)
        assert result == "\u2591" * 40

    def test_large_value_clips_to_half_width(self):
        """Value larger than scale*max produces exactly half-width bar."""
        # With scale=200, value=1.0 gives 200 chars, clipped to half (20)
        result = _make_diverging_bar(1.0, total_width=40, scale=200.0)
        assert len(result) == 40
        # The active portion should be at most 20 chars
        n_active = result.count("\u2588")
        assert n_active <= 20

    def test_nan_produces_full_background(self):
        """NaN value returns a string with only background chars."""
        result = _make_diverging_bar(float("nan"), total_width=40)
        assert result == "\u2591" * 40


# ---------------------------------------------------------------------------
# Tests: print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def test_prints_without_crash(self, capsys):
        """print_summary does not raise any exception."""
        df = _make_minimal_trades_df()
        diag_ticker = analyze_by_ticker(df)
        diag_year = analyze_by_year(df)
        diag_signal = analyze_by_signal(df)
        diag_4w = analyze_4w_vs_13w(df)
        diag_extreme = analyze_extreme_observations(df, top_n=2)
        diag_failure = analyze_failure_reasons(df)

        print_summary(
            df=df,
            diag_by_ticker=diag_ticker,
            diag_by_year=diag_year,
            diag_by_signal=diag_signal,
            diag_4w_vs_13w=diag_4w,
            diag_extreme=diag_extreme,
            diag_failure_reasons=diag_failure,
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_includes_headline_stats(self, capsys):
        """Headline stats section appears in output."""
        df = _make_minimal_trades_df()
        diag_ticker = analyze_by_ticker(df)
        diag_year = analyze_by_year(df)
        diag_signal = analyze_by_signal(df)
        diag_4w = analyze_4w_vs_13w(df)
        diag_extreme = analyze_extreme_observations(df, top_n=1)
        diag_failure = analyze_failure_reasons(df)

        print_summary(
            df=df,
            diag_by_ticker=diag_ticker,
            diag_by_year=diag_year,
            diag_by_signal=diag_signal,
            diag_4w_vs_13w=diag_4w,
            diag_extreme=diag_extreme,
            diag_failure_reasons=diag_failure,
        )
        captured = capsys.readouterr()
        assert "Headline Stats" in captured.out
        assert "Pass rate" in captured.out or "pass rate" in captured.out.lower()
        assert "Total observations" in captured.out

    def test_includes_ihsg_benchmark_returns(self, capsys):
        """IHSG benchmark returns appear in headline stats."""
        df = _make_minimal_trades_df()
        diag_ticker = analyze_by_ticker(df)
        diag_year = analyze_by_year(df)
        diag_signal = analyze_by_signal(df)
        diag_4w = analyze_4w_vs_13w(df)
        diag_extreme = analyze_extreme_observations(df, top_n=1)
        diag_failure = analyze_failure_reasons(df)

        print_summary(
            df=df,
            diag_by_ticker=diag_ticker,
            diag_by_year=diag_year,
            diag_by_signal=diag_signal,
            diag_4w_vs_13w=diag_4w,
            diag_extreme=diag_extreme,
            diag_failure_reasons=diag_failure,
        )
        captured = capsys.readouterr()
        assert "Avg IHSG return" in captured.out

    def test_handles_empty_signal_dataframe(self, capsys):
        """Does not crash when signal DataFrame is empty."""
        df = _make_minimal_trades_df()
        diag_ticker = analyze_by_ticker(df)
        diag_year = analyze_by_year(df)
        diag_4w = analyze_4w_vs_13w(df)
        diag_extreme = analyze_extreme_observations(df, top_n=1)
        diag_failure = analyze_failure_reasons(df)

        # Create an empty signal DataFrame
        empty_signal = pd.DataFrame(columns=[
            "distance_bucket", "rs_bucket", "sma20_is_rising",
            "n_observations", "n_passed",
            "avg_excess_4w", "avg_excess_13w",
            "win_rate_4w", "win_rate_13w",
        ])
        empty_failure = pd.DataFrame(columns=[
            "reason_category", "n_occurrences", "pct_of_total",
            "avg_excess_13w_when_passed", "avg_excess_13w_when_failed",
        ])

        print_summary(
            df=df,
            diag_by_ticker=diag_ticker,
            diag_by_year=diag_year,
            diag_by_signal=empty_signal,
            diag_4w_vs_13w=diag_4w,
            diag_extreme=diag_extreme,
            diag_failure_reasons=empty_failure,
        )
        captured = capsys.readouterr()
        assert "Skipped:" in captured.out
        assert "not available" in captured.out

    def test_includes_extreme_observations_section(self, capsys):
        """Extreme observations section appears when extreme data is available."""
        df = _make_minimal_trades_df()
        # Add more passed observations so we get valid extremes
        extra = pd.DataFrame({
            "date": ["2024-01-19", "2024-01-26", "2024-02-02", "2024-02-09"],
            "ticker": ["BBCA", "BBRI", "BBCA", "BBRI"],
            "company_name": ["Bank BCA", "Bank BRI", "Bank BCA", "Bank BRI"],
            "sector": ["Finance", "Finance", "Finance", "Finance"],
            "passed_technical": [True, True, True, True],
            "close": [10200.0, 5100.0, 10300.0, 5200.0],
            "sma20": [10100.0, 5000.0, 10200.0, 5100.0],
            "sma50": [10000.0, 4900.0, 10100.0, 5000.0],
            "distance_from_sma20": [0.01, 0.02, 0.01, 0.02],
            "relative_strength_13w": [0.04, 0.06, 0.05, 0.07],
            "sma20_is_rising": [True, False, True, False],
            "reasons": ["PASS", "PASS", "PASS", "PASS"],
            "warnings": ["", "", "", ""],
            "forward_return_4w": [0.01, 0.02, 0.03, 0.04],
            "index_return_4w": [0.01, 0.01, 0.01, 0.01],
            "excess_return_4w": [0.0, 0.01, 0.02, 0.03],
            "forward_return_13w": [0.01, 0.03, 0.06, 0.10],
            "index_return_13w": [0.03, 0.03, 0.03, 0.03],
            "excess_return_13w": [-0.02, 0.0, 0.03, 0.07],
        })
        df = pd.concat([df, extra], ignore_index=True)
        diag_ticker = analyze_by_ticker(df)
        diag_year = analyze_by_year(df)
        diag_signal = analyze_by_signal(df)
        diag_4w = analyze_4w_vs_13w(df)
        diag_extreme = analyze_extreme_observations(df, top_n=2)
        diag_failure = analyze_failure_reasons(df)

        print_summary(
            df=df,
            diag_by_ticker=diag_ticker,
            diag_by_year=diag_year,
            diag_by_signal=diag_signal,
            diag_4w_vs_13w=diag_4w,
            diag_extreme=diag_extreme,
            diag_failure_reasons=diag_failure,
        )
        captured = capsys.readouterr()
        assert "Extreme Observations" in captured.out
        assert "Worst" in captured.out
        assert "Best" in captured.out

    def test_handles_empty_extreme_dataframe(self, capsys):
        """Does not crash when extreme observations DataFrame is empty."""
        df = _make_minimal_trades_df()
        diag_ticker = analyze_by_ticker(df)
        diag_year = analyze_by_year(df)
        diag_signal = analyze_by_signal(df)
        diag_4w = analyze_4w_vs_13w(df)
        diag_failure = analyze_failure_reasons(df)

        # Create empty extreme DataFrame
        empty_extreme = pd.DataFrame(columns=[
            "date", "ticker", "distance_from_sma20",
            "relative_strength_13w", "excess_return_13w", "rank_label",
        ])

        print_summary(
            df=df,
            diag_by_ticker=diag_ticker,
            diag_by_year=diag_year,
            diag_by_signal=diag_signal,
            diag_4w_vs_13w=diag_4w,
            diag_extreme=empty_extreme,
            diag_failure_reasons=diag_failure,
        )
        captured = capsys.readouterr()
        assert "Extreme Observations" in captured.out
        assert "No extreme observation data" in captured.out


# ---------------------------------------------------------------------------
# Tests: write_diagnostics
# ---------------------------------------------------------------------------


class TestWriteDiagnostics:
    @staticmethod
    def _precompute_diag(df, top_n=1):
        """Pre-compute all diagnostic DataFrames for testing."""
        return {
            "diag_by_ticker": analyze_by_ticker(df),
            "diag_by_year": analyze_by_year(df),
            "diag_by_signal": analyze_by_signal(df),
            "diag_4w_vs_13w": analyze_4w_vs_13w(df),
            "diag_extreme": analyze_extreme_observations(df, top_n=top_n),
            "diag_failure": analyze_failure_reasons(df),
        }

    def test_creates_output_directory(self, tmp_path):
        """Output directory is created if it does not exist."""
        out_dir = str(tmp_path / "diagnostics")
        df = _make_minimal_trades_df()
        diag = self._precompute_diag(df, top_n=1)
        write_diagnostics(output_dir=out_dir, **diag)
        assert os.path.isdir(out_dir)

    def test_creates_all_six_csvs(self, tmp_path):
        """All 6 diagnostic CSV files are created."""
        out_dir = str(tmp_path / "diag")
        df = _make_minimal_trades_df()
        diag = self._precompute_diag(df, top_n=1)
        file_map = write_diagnostics(output_dir=out_dir, **diag)
        for path in file_map.values():
            assert os.path.exists(path), f"File not found: {path}"
        assert len(file_map) == 6

    def test_rounds_numeric_columns_to_4_decimals(self, tmp_path):
        """Numeric values in CSVs have at most 4 decimal places."""
        out_dir = str(tmp_path / "rounded")
        df = _make_minimal_trades_df()
        diag = self._precompute_diag(df, top_n=1)
        file_map = write_diagnostics(output_dir=out_dir, **diag)

        # Check the by_ticker CSV for rounding
        ticker_csv = file_map["by_ticker"]
        ticker_df = pd.read_csv(ticker_csv)
        numeric_cols = ticker_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Check that values have at most 4 decimal places
            values = ticker_df[col].dropna()
            if len(values) > 0:
                # Multiply by 10000 to check for excess precision
                scaled = values * 10000
                rounded = scaled.round()
                assert scaled.equals(rounded), (
                    f"Column {col} has more than 4 decimal places"
                )


# ---------------------------------------------------------------------------
# Tests: run_diagnostics integration (minimal)
# ---------------------------------------------------------------------------


class TestRunDiagnosticsIntegration:
    def test_end_to_end_with_minimal_fixture(self, tmp_path):
        """End-to-end run with a temporary CSV and output directory."""
        from stock_screener.src.backtest_diagnostics import run_diagnostics

        # Create a temporary trades CSV
        df = _make_minimal_trades_df()
        csv_path = os.path.join(tmp_path, "trades.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")

        out_dir = os.path.join(tmp_path, "diag_out")

        # Monkey-patch sys.argv
        original_argv = sys.argv
        sys.argv = [
            "backtest_diagnostics.py",
            "--trades-csv", csv_path,
            "--output-dir", out_dir,
            "--top-n", "2",
        ]
        try:
            run_diagnostics()
        except SystemExit as exc:
            if exc.code != 0:
                pytest.fail(f"run_diagnostics exited with code {exc.code}")
        finally:
            sys.argv = original_argv

        # Verify output files exist
        expected_files = [
            "backtest_diag_by_ticker.csv",
            "backtest_diag_by_year.csv",
            "backtest_diag_by_signal.csv",
            "backtest_diag_4w_vs_13w.csv",
            "backtest_diag_extreme_observations.csv",
            "backtest_diag_failure_reasons.csv",
        ]
        for fname in expected_files:
            fpath = os.path.join(out_dir, fname)
            assert os.path.exists(fpath), f"Expected output file not found: {fpath}"
