"""Unit tests for the technical-only historical backtest module."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from stock_screener.src.backtest import (
    evaluate_ticker_on_date,
    generate_backtest_reports,
    get_backtest_dates,
    get_forward_return,
    _is_within_week_tolerance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_weekly_index(start: str, periods: int) -> pd.DatetimeIndex:
    """Create a Friday-ending weekly DatetimeIndex."""
    return pd.date_range(start=start, periods=periods, freq="W-FRI")


def _make_weekly_df(dates: pd.DatetimeIndex, close_start: float = 100.0) -> pd.DataFrame:
    """Create a minimal weekly OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    n = len(dates)
    closes = close_start * (1 + rng.normal(0.001, 0.02, n).cumsum())
    return pd.DataFrame(
        {
            "Open": closes * 0.99,
            "High": closes * 1.02,
            "Low": closes * 0.98,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def config_minimal() -> dict:
    """Minimal config with technical and backtest sections."""
    return {
        "technical": {
            "sma_short": 20,
            "sma_long": 50,
            "max_distance_from_sma20": 0.15,
            "relative_strength_weeks": 13,
            "sma_rising_lookback": 3,
        },
        "backtest": {
            "history_months": 60,
            "horizons_weeks": [4, 13],
            "min_warmup_weeks": 60,
        },
    }


# ---------------------------------------------------------------------------
# Tests: _is_within_week_tolerance
# ---------------------------------------------------------------------------


def test_is_within_week_tolerance_same_day():
    d = pd.Timestamp("2025-01-10")
    assert _is_within_week_tolerance(d, d) is True


def test_is_within_week_tolerance_3_days():
    d1 = pd.Timestamp("2025-01-10")
    d2 = pd.Timestamp("2025-01-13")
    assert _is_within_week_tolerance(d1, d2) is True


def test_is_within_week_tolerance_6_days():
    d1 = pd.Timestamp("2025-01-10")
    d2 = pd.Timestamp("2025-01-16")
    assert _is_within_week_tolerance(d1, d2) is True


def test_is_within_week_tolerance_exactly_7_days():
    """Consecutive Fridays (7 days apart) should be rejected."""
    d1 = pd.Timestamp("2025-01-10")
    d2 = pd.Timestamp("2025-01-17")
    assert _is_within_week_tolerance(d1, d2) is False


def test_is_within_week_tolerance_8_days():
    d1 = pd.Timestamp("2025-01-10")
    d2 = pd.Timestamp("2025-01-18")
    assert _is_within_week_tolerance(d1, d2) is False


# ---------------------------------------------------------------------------
# Tests: get_backtest_dates
# ---------------------------------------------------------------------------


def test_get_backtest_dates_warmup():
    """First eval date is after warmup period."""
    dates = _make_weekly_index("2020-01-03", 100)
    index_df = _make_weekly_df(dates)
    result = get_backtest_dates(index_df, min_warmup_weeks=10, horizons_weeks=[4, 13])
    assert len(result) > 0
    assert result[0] == dates[10]


def test_get_backtest_dates_horizon():
    """Last eval date is before end minus max_horizon."""
    dates = _make_weekly_index("2020-01-03", 100)
    index_df = _make_weekly_df(dates)
    result = get_backtest_dates(index_df, min_warmup_weeks=10, horizons_weeks=[4, 13])
    # start_idx=10, end_idx=100-13=87, slice is [10:87] -> last index is 86
    assert result[-1] == dates[86]


def test_get_backtest_dates_insufficient_data():
    """Returns empty list when data is too short."""
    dates = _make_weekly_index("2020-01-03", 20)
    index_df = _make_weekly_df(dates)
    result = get_backtest_dates(index_df, min_warmup_weeks=10, horizons_weeks=[4, 13])
    assert result == []


def test_get_backtest_dates_empty_df():
    """Returns empty list for empty DataFrame."""
    index_df = pd.DataFrame()
    result = get_backtest_dates(index_df, min_warmup_weeks=10, horizons_weeks=[4, 13])
    assert result == []


def test_get_backtest_dates_empty_horizons():
    """Returns empty list when horizons_weeks is empty."""
    dates = _make_weekly_index("2020-01-03", 100)
    index_df = _make_weekly_df(dates)
    result = get_backtest_dates(index_df, min_warmup_weeks=10, horizons_weeks=[])
    assert result == []


# ---------------------------------------------------------------------------
# Tests: get_forward_return
# ---------------------------------------------------------------------------


def test_get_forward_return_normal():
    """Correct return calculation for a valid horizon."""
    dates = _make_weekly_index("2020-01-03", 50)
    df = _make_weekly_df(dates, close_start=100.0)
    df.loc[dates[10], "Close"] = 100.0
    df.loc[dates[14], "Close"] = 105.0
    result = get_forward_return(df, dates[10], horizon_weeks=4)
    assert abs(result - 0.05) < 1e-10


def test_get_forward_return_missing_future():
    """Returns NaN when future bar is missing."""
    dates = _make_weekly_index("2020-01-03", 20)
    df = _make_weekly_df(dates, close_start=100.0)
    result = get_forward_return(df, dates[-1], horizon_weeks=10)
    assert np.isnan(result)


def test_get_forward_return_zero_price():
    """Returns NaN when current close is zero."""
    dates = _make_weekly_index("2020-01-03", 50)
    df = _make_weekly_df(dates, close_start=100.0)
    df.loc[dates[10], "Close"] = 0.0
    result = get_forward_return(df, dates[10], horizon_weeks=4)
    assert np.isnan(result)


def test_get_forward_return_near_zero_price():
    """Returns NaN when current close is near-zero (tolerance check)."""
    dates = _make_weekly_index("2020-01-03", 50)
    df = _make_weekly_df(dates, close_start=100.0)
    df.loc[dates[10], "Close"] = 1e-9
    result = get_forward_return(df, dates[10], horizon_weeks=4)
    assert np.isnan(result)


def test_get_forward_return_nan_close():
    """Returns NaN when current close is NaN."""
    dates = _make_weekly_index("2020-01-03", 50)
    df = _make_weekly_df(dates, close_start=100.0)
    df.loc[dates[10], "Close"] = np.nan
    result = get_forward_return(df, dates[10], horizon_weeks=4)
    assert np.isnan(result)


def test_get_forward_return_date_not_in_index():
    """Returns NaN when date is not in the DataFrame index."""
    dates = _make_weekly_index("2020-01-03", 50)
    df = _make_weekly_df(dates, close_start=100.0)
    missing_date = pd.Timestamp("2020-03-15")
    result = get_forward_return(df, missing_date, horizon_weeks=4)
    assert np.isnan(result)


def test_get_forward_return_nan_future_close():
    """Returns NaN when future close is NaN."""
    dates = _make_weekly_index("2020-01-03", 50)
    df = _make_weekly_df(dates, close_start=100.0)
    df.loc[dates[10], "Close"] = 100.0
    df.loc[dates[14], "Close"] = np.nan
    result = get_forward_return(df, dates[10], horizon_weeks=4)
    assert np.isnan(result)


def test_get_forward_return_excessive_gap():
    """Returns NaN when the gap to the future bar exceeds horizon + 14 days."""
    dates = _make_weekly_index("2020-01-03", 50)
    df = _make_weekly_df(dates, close_start=100.0)
    df.loc[dates[10], "Close"] = 100.0
    # Remove intermediate bars so the next available bar is far away
    # horizon_weeks=4 -> expected 28 days, + 14 = 42 days max
    # dates[10] + 4 weeks = dates[14], but we remove dates[14] through dates[19]
    # so the next bar is dates[20] which is 10 weeks away (70 days > 42)
    df = df.drop(dates[14:20])
    result = get_forward_return(df, dates[10], horizon_weeks=4)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# Tests: evaluate_ticker_on_date
# ---------------------------------------------------------------------------


def test_evaluate_ticker_on_date_returns_dict(config_minimal):
    """Returns a dict with expected keys when data is sufficient."""
    dates = _make_weekly_index("2020-01-03", 120)
    stock_df = _make_weekly_df(dates, close_start=100.0)
    index_df = _make_weekly_df(dates, close_start=5000.0)
    stock_row = pd.Series({"ticker": "TEST", "company_name": "Test Corp", "sector": "Bank"})

    eval_date = dates[70]
    result = evaluate_ticker_on_date(
        "TEST", stock_row, stock_df, index_df, eval_date, [4, 13], config_minimal
    )
    assert result is not None
    assert isinstance(result, dict)
    assert result["ticker"] == "TEST"
    assert result["date"] == eval_date
    assert "passed_technical" in result
    assert "forward_return_4w" in result
    assert "forward_return_13w" in result
    assert "excess_return_4w" in result
    assert "excess_return_13w" in result


def test_evaluate_ticker_on_date_empty_stock_data(config_minimal):
    """Returns None when stock data is empty."""
    dates = _make_weekly_index("2020-01-03", 120)
    index_df = _make_weekly_df(dates, close_start=5000.0)
    stock_df = pd.DataFrame()
    stock_row = pd.Series({"ticker": "TEST", "company_name": "Test Corp", "sector": "Bank"})

    result = evaluate_ticker_on_date(
        "TEST", stock_row, stock_df, index_df, dates[70], [4, 13], config_minimal
    )
    assert result is None


def test_evaluate_ticker_on_date_stale_data(config_minimal):
    """Returns None when stock data is stale (>= 7 days before eval date)."""
    dates = _make_weekly_index("2020-01-03", 120)
    index_df = _make_weekly_df(dates, close_start=5000.0)
    stock_df = _make_weekly_df(dates[:67], close_start=100.0)
    stock_row = pd.Series({"ticker": "TEST", "company_name": "Test Corp", "sector": "Bank"})

    eval_date = dates[70]  # 3 weeks (21 days) after stock's last bar
    result = evaluate_ticker_on_date(
        "TEST", stock_row, stock_df, index_df, eval_date, [4, 13], config_minimal
    )
    assert result is None


def test_evaluate_ticker_on_date_index_empty_before_date(config_minimal):
    """Returns None when index has no data on or before eval date."""
    dates = _make_weekly_index("2020-01-03", 120)
    # Index data starts after eval date
    index_df = _make_weekly_df(dates[80:], close_start=5000.0)
    stock_df = _make_weekly_df(dates, close_start=100.0)
    stock_row = pd.Series({"ticker": "TEST", "company_name": "Test Corp", "sector": "Bank"})

    eval_date = dates[70]  # Before any index data
    result = evaluate_ticker_on_date(
        "TEST", stock_row, stock_df, index_df, eval_date, [4, 13], config_minimal
    )
    assert result is None


def test_evaluate_ticker_on_date_no_stock_data_before_date(config_minimal):
    """Returns None when stock data starts after eval date."""
    dates = _make_weekly_index("2020-01-03", 120)
    index_df = _make_weekly_df(dates, close_start=5000.0)
    # Stock data starts after eval date
    stock_df = _make_weekly_df(dates[80:], close_start=100.0)
    stock_row = pd.Series({"ticker": "TEST", "company_name": "Test Corp", "sector": "Bank"})

    eval_date = dates[70]
    result = evaluate_ticker_on_date(
        "TEST", stock_row, stock_df, index_df, eval_date, [4, 13], config_minimal
    )
    assert result is None


# ---------------------------------------------------------------------------
# Tests: generate_backtest_reports
# ---------------------------------------------------------------------------


def test_generate_backtest_reports_creates_files():
    """Creates both trade-level and summary CSV files."""
    dates = _make_weekly_index("2020-01-03", 10)
    trades_df = pd.DataFrame(
        {
            "date": dates[:5].tolist() + dates[:5].tolist(),
            "ticker": ["A"] * 5 + ["B"] * 5,
            "company_name": ["A Corp"] * 5 + ["B Corp"] * 5,
            "sector": ["Bank"] * 10,
            "passed_technical": [True] * 5 + [False] * 5,
            "close": [100.0] * 10,
            "sma20": [99.0] * 10,
            "sma50": [98.0] * 10,
            "distance_from_sma20": [0.01] * 10,
            "relative_strength_13w": [0.02] * 10,
            "sma20_is_rising": [True] * 10,
            "forward_return_4w": [0.05, 0.03, -0.02, 0.01, 0.04, -0.01, -0.03, 0.02, -0.05, 0.01],
            "forward_return_13w": [0.10] * 10,
            "index_return_4w": [0.02] * 10,
            "index_return_13w": [0.05] * 10,
            "excess_return_4w": [0.03, 0.01, -0.04, -0.01, 0.02, -0.03, -0.05, 0.0, -0.07, -0.01],
            "excess_return_13w": [0.05] * 10,
            "reasons": ["PASS"] * 10,
            "warnings": [""] * 10,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path, summary_path = generate_backtest_reports(
            trades_df, tmpdir, horizons=[4, 13]
        )
        assert os.path.exists(trades_path)
        assert os.path.exists(summary_path)

        summary = pd.read_csv(summary_path)
        assert len(summary) == 4  # 2 horizons x 2 pass/fail groups
        assert set(summary.columns) >= {
            "horizon_weeks",
            "passed_technical",
            "n_observations",
            "avg_forward_return",
            "win_rate",
            "avg_excess_return",
            "excess_win_rate",
        }
        # Verify pass/fail groups are separated
        assert (summary["passed_technical"] == True).sum() == 2
        assert (summary["passed_technical"] == False).sum() == 2


def test_generate_backtest_reports_empty_df():
    """Handles empty DataFrame without crashing."""
    trades_df = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path, summary_path = generate_backtest_reports(
            trades_df, tmpdir, horizons=[4, 13]
        )
        assert os.path.exists(trades_path)
        assert os.path.exists(summary_path)


def test_generate_backtest_reports_all_nan_returns():
    """Handles all-NaN forward returns without crashing."""
    dates = _make_weekly_index("2020-01-03", 10)
    trades_df = pd.DataFrame(
        {
            "date": dates[:3].tolist(),
            "ticker": ["A"] * 3,
            "company_name": ["A Corp"] * 3,
            "sector": ["Bank"] * 3,
            "passed_technical": [True] * 3,
            "close": [100.0] * 3,
            "sma20": [99.0] * 3,
            "sma50": [98.0] * 3,
            "distance_from_sma20": [0.01] * 3,
            "relative_strength_13w": [0.02] * 3,
            "sma20_is_rising": [True] * 3,
            "forward_return_4w": [float("nan")] * 3,
            "forward_return_13w": [float("nan")] * 3,
            "index_return_4w": [float("nan")] * 3,
            "index_return_13w": [float("nan")] * 3,
            "excess_return_4w": [float("nan")] * 3,
            "excess_return_13w": [float("nan")] * 3,
            "reasons": ["PASS"] * 3,
            "warnings": [""] * 3,
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path, summary_path = generate_backtest_reports(
            trades_df, tmpdir, horizons=[4, 13]
        )
        assert os.path.exists(trades_path)
        assert os.path.exists(summary_path)
        summary = pd.read_csv(summary_path)
        # n_valid_returns should be 0, metrics should be NaN
        assert (summary["n_valid_returns"] == 0).all()


def test_generate_backtest_reports_missing_passed_technical_column():
    """Handles DataFrame without passed_technical column."""
    trades_df = pd.DataFrame({"date": [], "ticker": []})
    with tempfile.TemporaryDirectory() as tmpdir:
        trades_path, summary_path = generate_backtest_reports(
            trades_df, tmpdir, horizons=[4, 13]
        )
        assert os.path.exists(trades_path)
        assert os.path.exists(summary_path)
        summary = pd.read_csv(summary_path)
        assert len(summary) == 0


def test_generate_backtest_reports_summary_values():
    """Verifies computed summary values against known inputs."""
    dates = _make_weekly_index("2020-01-03", 10)
    trades_df = pd.DataFrame(
        {
            "date": dates[:4].tolist(),
            "ticker": ["A"] * 4,
            "company_name": ["A Corp"] * 4,
            "sector": ["Bank"] * 4,
            "passed_technical": [True] * 4,
            "close": [100.0] * 4,
            "sma20": [99.0] * 4,
            "sma50": [98.0] * 4,
            "distance_from_sma20": [0.01] * 4,
            "relative_strength_13w": [0.02] * 4,
            "sma20_is_rising": [True] * 4,
            "forward_return_4w": [0.10, 0.20, -0.05, 0.05],
            "forward_return_13w": [0.10] * 4,
            "index_return_4w": [0.02] * 4,
            "index_return_13w": [0.05] * 4,
            "excess_return_4w": [0.08, 0.18, -0.07, 0.03],
            "excess_return_13w": [0.05] * 4,
            "reasons": ["PASS"] * 4,
            "warnings": [""] * 4,
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _, summary_path = generate_backtest_reports(
            trades_df, tmpdir, horizons=[4]
        )
        summary = pd.read_csv(summary_path)
        passed_row = summary[
            (summary["horizon_weeks"] == 4) & (summary["passed_technical"] == True)
        ].iloc[0]
        assert passed_row["n_observations"] == 4
        assert passed_row["n_valid_returns"] == 4
        assert abs(passed_row["avg_forward_return"] - 0.075) < 1e-10
        assert abs(passed_row["win_rate"] - 0.75) < 1e-10
        assert abs(passed_row["avg_excess_return"] - 0.055) < 1e-10
        assert abs(passed_row["excess_win_rate"] - 0.75) < 1e-10


# ---------------------------------------------------------------------------
# Tests: excess return calculation
# ---------------------------------------------------------------------------


def test_excess_return_calculation():
    """excess_return = forward_return - index_return."""
    dates = _make_weekly_index("2020-01-03", 50)
    stock_df = _make_weekly_df(dates, close_start=100.0)
    index_df = _make_weekly_df(dates, close_start=5000.0)

    eval_date = dates[20]
    stock_df.loc[eval_date, "Close"] = 100.0
    stock_df.loc[dates[24], "Close"] = 110.0
    index_df.loc[eval_date, "Close"] = 5000.0
    index_df.loc[dates[24], "Close"] = 5100.0

    fr = get_forward_return(stock_df, eval_date, 4)
    ir = get_forward_return(index_df, eval_date, 4)
    er = fr - ir

    assert abs(fr - 0.10) < 1e-10
    assert abs(ir - 0.02) < 1e-10
    assert abs(er - 0.08) < 1e-10
