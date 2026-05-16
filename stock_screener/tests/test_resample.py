"""
Tests for stock_screener.src.pipeline — weekly resampling and technical
indicator calculation (TASK-006).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_screener.src.pipeline import (
    calculate_distance_from_sma,
    calculate_relative_strength,
    calculate_sma,
    calculate_technical_features,
    resample_to_weekly,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _daily_df(dates: list[str], **overrides) -> pd.DataFrame:
    """Build a daily OHLCV DataFrame with controlled values.

    Default values are chosen so that Open, High, Low, Close differ,
    making OHLCV aggregation tests meaningful.

    Parameters
    ----------
    dates : list[str]
        Date strings in ``"YYYY-MM-DD"`` format.
    **overrides : dict
        Column values to override defaults.

    Returns
    -------
    pd.DataFrame
        Daily OHLCV DataFrame with a DatetimeIndex.
    """
    n = len(dates)
    defaults = {
        "Open": range(100, 100 + n),
        "High": range(105, 105 + n),
        "Low": range(95, 95 + n),
        "Close": range(102, 102 + n),
        "Volume": [1_000_000] * n,
    }
    data = {**defaults, **overrides}
    df = pd.DataFrame(data, index=pd.DatetimeIndex(pd.to_datetime(dates)))
    df.index.name = "Date"
    return df


def _weekly_df(
    n_weeks: int,
    start: str = "2025-01-06",
    close_start: float = 100.0,
    close_step: float = 1.0,
    **overrides,
) -> pd.DataFrame:
    """Build a weekly OHLCV DataFrame with Fridays as index.

    Parameters
    ----------
    n_weeks : int
        Number of weekly rows to generate.
    start : str
        Date of the first Monday (will be adjusted to Friday label).
    close_start : float
        Close value for the first week.
    close_step : float
        Step between consecutive weekly Close values.
    **overrides : dict
        Column values to override defaults.

    Returns
    -------
    pd.DataFrame
        Weekly OHLCV DataFrame with a DatetimeIndex of Fridays.
    """
    # Generate Mondays and shift to the Friday of that week
    mondays = pd.date_range(start, periods=n_weeks, freq="W-MON")
    fridays = mondays + pd.DateOffset(days=4)

    close_values = [close_start + i * close_step for i in range(n_weeks)]
    open_values = [c - 2.0 for c in close_values]
    high_values = [c + 3.0 for c in close_values]
    low_values = [c - 3.0 for c in close_values]
    volumes = [5_000_000] * n_weeks

    data = {
        "Open": open_values,
        "High": high_values,
        "Low": low_values,
        "Close": close_values,
        "Volume": volumes,
    }
    data.update(overrides)

    df = pd.DataFrame(data, index=pd.DatetimeIndex(fridays))
    df.index.name = "Date"
    return df


# ===================================================================
# TASK-006: Weekly Resampling
# ===================================================================


class TestResampleToWeekly:
    """Tests for ``resample_to_weekly()``."""

    def test_resample_to_weekly_basic(self):
        """5 daily rows in the same week → 1 weekly row with correct OHLCV."""
        dates = [
            "2025-01-06",  # Mon
            "2025-01-07",  # Tue
            "2025-01-08",  # Wed
            "2025-01-09",  # Thu
            "2025-01-10",  # Fri
        ]
        daily = _daily_df(
            dates,
            Open=[100, 102, 101, 103, 104],
            High=[105, 106, 104, 107, 108],
            Low=[99, 100, 99, 101, 102],
            Close=[102, 104, 103, 106, 108],
            Volume=[1_000_000, 1_500_000, 1_200_000, 1_800_000, 2_000_000],
        )

        weekly = resample_to_weekly(daily)

        assert len(weekly) == 1
        row = weekly.iloc[0]
        assert row["Open"] == 100.0
        assert row["High"] == 108.0
        assert row["Low"] == 99.0
        assert row["Close"] == 108.0
        assert row["Volume"] == 7_500_000.0
        assert weekly.index[0] == pd.Timestamp("2025-01-10")  # Friday label

    def test_resample_to_weekly_multi_week(self):
        """15 daily rows across 3 weeks → 3 weekly rows."""
        # Week 1: Mon 6 Jan – Fri 10 Jan
        # Week 2: Mon 13 Jan – Fri 17 Jan
        # Week 3: Mon 20 Jan – Fri 24 Jan
        dates = [
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
            "2025-01-13",
            "2025-01-14",
            "2025-01-15",
            "2025-01-16",
            "2025-01-17",
            "2025-01-20",
            "2025-01-21",
            "2025-01-22",
            "2025-01-23",
            "2025-01-24",
        ]
        n = len(dates)
        daily = _daily_df(
            dates,
            Open=list(range(100, 100 + n)),
            High=list(range(105, 105 + n)),
            Low=list(range(95, 95 + n)),
            Close=list(range(102, 102 + n)),
            Volume=[1_000_000] * n,
        )

        weekly = resample_to_weekly(daily)

        assert len(weekly) == 3

        # Week 1 (label 2025-01-10): rows 0-4
        row0 = weekly.loc[pd.Timestamp("2025-01-10")]
        assert row0["Open"] == 100.0
        assert row0["High"] == 109.0  # max of 105..109
        assert row0["Low"] == 95.0  # min of 95..99
        assert row0["Close"] == 106.0  # last of 102..106
        assert row0["Volume"] == 5_000_000.0

        # Week 2 (label 2025-01-17): rows 5-9
        row1 = weekly.loc[pd.Timestamp("2025-01-17")]
        assert row1["Open"] == 105.0
        assert row1["High"] == 114.0
        assert row1["Low"] == 100.0
        assert row1["Close"] == 111.0
        assert row1["Volume"] == 5_000_000.0

        # Week 3 (label 2025-01-24): rows 10-14
        row2 = weekly.loc[pd.Timestamp("2025-01-24")]
        assert row2["Open"] == 110.0
        assert row2["High"] == 119.0
        assert row2["Low"] == 105.0
        assert row2["Close"] == 116.0
        assert row2["Volume"] == 5_000_000.0

    def test_resample_to_weekly_friday_holiday(self):
        """Week where Friday is a holiday (Thu last trading day) still produces a bar."""
        dates = [
            "2025-01-06",  # Mon
            "2025-01-07",  # Tue
            "2025-01-08",  # Wed
            "2025-01-09",  # Thu  (no Fri data — holiday)
        ]
        daily = _daily_df(
            dates,
            Open=[100, 102, 101, 103],
            High=[105, 106, 104, 107],
            Low=[99, 100, 99, 101],
            Close=[102, 104, 103, 106],
            Volume=[1_000_000, 1_500_000, 1_200_000, 1_800_000],
        )

        weekly = resample_to_weekly(daily)

        assert len(weekly) == 1
        row = weekly.iloc[0]
        assert row["Open"] == 100.0
        assert row["High"] == 107.0
        assert row["Low"] == 99.0
        assert row["Close"] == 106.0  # Thursday's close
        assert row["Volume"] == 5_500_000.0
        # The label is the Friday even though Friday data is absent
        assert weekly.index[0] == pd.Timestamp("2025-01-10")

    def test_resample_to_weekly_empty(self):
        """Empty input returns an empty DataFrame."""
        empty = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        )
        result = resample_to_weekly(empty)
        assert result.empty

    def test_resample_to_weekly_single_day(self):
        """A single daily row produces one weekly bar."""
        daily = _daily_df(["2025-01-06"])
        weekly = resample_to_weekly(daily)
        assert len(weekly) == 1
        assert weekly.index[0] == pd.Timestamp("2025-01-10")

    def test_resample_to_weekly_uses_adj_close_when_both_present(self):
        """When both Close and Adj Close exist, weekly Close should reflect Adj Close values."""
        dates = pd.date_range("2025-01-06", periods=5, freq="D")  # Mon-Fri, same week
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],  # raw close
            "Adj Close": [98, 99, 100, 101, 102],  # adjusted close (different)
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }, index=dates)
        weekly = resample_to_weekly(df)
        assert len(weekly) == 1
        # Weekly Close should be the last Adj Close (102), not the last raw Close (104)
        assert weekly["Close"].iloc[0] == pytest.approx(102.0)
        # Adj Close column should also be preserved
        assert "Adj Close" in weekly.columns
        assert weekly["Adj Close"].iloc[0] == pytest.approx(102.0)


# ===================================================================
# TASK-006: SMA Calculation
# ===================================================================


class TestCalculateSMA:
    """Tests for ``calculate_sma()``."""

    def test_calculate_sma(self):
        """Known input produces known SMA output."""
        dates = pd.date_range("2025-01-10", periods=5, freq="W-FRI")
        close = [10.0, 20.0, 30.0, 40.0, 50.0]
        weekly = pd.DataFrame({"Close": close}, index=dates)

        sma2 = calculate_sma(weekly, period=2)
        expected_sma2 = [float("nan"), 15.0, 25.0, 35.0, 45.0]
        pd.testing.assert_series_equal(
            sma2,
            pd.Series(expected_sma2, index=dates, name="Close"),
            check_names=False,
        )

    def test_calculate_sma_period_longer_than_data(self):
        """All-NaN when period > length of data."""
        weekly = _weekly_df(n_weeks=3)
        sma20 = calculate_sma(weekly, period=20)
        assert sma20.isna().all()

    def test_calculate_sma_custom_column(self):
        """Can specify a column other than Close."""
        dates = pd.date_range("2025-01-10", periods=4, freq="W-FRI")
        weekly = pd.DataFrame(
            {"Close": [100, 200, 300, 400], "Volume": [1, 2, 3, 4]},
            index=dates,
        )
        sma_vol = calculate_sma(weekly, period=2, column="Volume")
        expected = [float("nan"), 1.5, 2.5, 3.5]
        pd.testing.assert_series_equal(
            sma_vol,
            pd.Series(expected, index=dates, name="Volume"),
            check_names=False,
        )


# ===================================================================
# TASK-006: Distance from SMA
# ===================================================================


class TestCalculateDistanceFromSMA:
    """Tests for ``calculate_distance_from_sma()``."""

    def test_distance_above_sma(self):
        """Close above SMA yields positive distance."""
        close = pd.Series([110.0, 120.0, 130.0])
        sma = pd.Series([100.0, 100.0, 100.0])
        dist = calculate_distance_from_sma(close, sma)
        expected = pd.Series([0.10, 0.20, 0.30])
        pd.testing.assert_series_equal(dist, expected)

    def test_distance_below_sma(self):
        """Close below SMA yields negative distance."""
        close = pd.Series([90.0, 80.0, 70.0])
        sma = pd.Series([100.0, 100.0, 100.0])
        dist = calculate_distance_from_sma(close, sma)
        expected = pd.Series([-0.10, -0.20, -0.30])
        pd.testing.assert_series_equal(dist, expected)

    def test_distance_at_sma(self):
        """Close equal to SMA yields zero distance."""
        close = pd.Series([100.0, 100.0, 100.0])
        sma = pd.Series([100.0, 100.0, 100.0])
        dist = calculate_distance_from_sma(close, sma)
        expected = pd.Series([0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(dist, expected)

    def test_distance_with_nan_sma(self):
        """NaN SMA values produce NaN distances."""
        close = pd.Series([100.0, 101.0])
        sma = pd.Series([float("nan"), 50.0])
        result = calculate_distance_from_sma(close, sma)
        assert pd.isna(result.iloc[0])  # NaN SMA → NaN distance
        assert result.iloc[1] == pytest.approx((101 - 50) / 50)


# ===================================================================
# TASK-006: Relative Strength
# ===================================================================


class TestCalculateRelativeStrength:
    """Tests for ``calculate_relative_strength()``."""

    def test_calculate_relative_strength(self):
        """Mock data verifies the RS difference formula."""
        # 15 weeks of data (14 needed for weeks=13)
        fridays = pd.date_range("2025-01-10", periods=15, freq="W-FRI")

        # Stock: first 14 values = 100, last = 200
        # → stock_return = 200 / 100 - 1 = 1.0
        stock_close = [100.0] * 14 + [200.0]
        stock_df = pd.DataFrame({"Close": stock_close}, index=fridays)

        # Index: all values = 100
        # → index_return = 100 / 100 - 1 = 0.0
        index_close = [100.0] * 15
        index_df = pd.DataFrame({"Close": index_close}, index=fridays)

        rs = calculate_relative_strength(stock_df, index_df, weeks=13)

        assert rs == pytest.approx(1.0)

    def test_calculate_relative_strength_insufficient_data(self):
        """Fewer than weeks+1 data points → NaN."""
        fridays = pd.date_range("2025-01-10", periods=13, freq="W-FRI")
        stock_df = pd.DataFrame({"Close": [100.0] * 13}, index=fridays)
        index_df = pd.DataFrame({"Close": [100.0] * 13}, index=fridays)

        rs = calculate_relative_strength(stock_df, index_df, weeks=13)

        assert math.isnan(rs)

    def test_relative_strength_with_negative_values(self):
        """Stock underperforming index yields negative RS."""
        fridays = pd.date_range("2025-01-10", periods=15, freq="W-FRI")

        # Stock: first 14 = 100, last = 50 → stock_return = 50/100 - 1 = -0.5
        stock_df = pd.DataFrame(
            {"Close": [100.0] * 14 + [50.0]}, index=fridays
        )
        # Index: first 14 = 100, last = 200 → index_return = 200/100 - 1 = 1.0
        index_df = pd.DataFrame(
            {"Close": [100.0] * 14 + [200.0]}, index=fridays
        )

        rs = calculate_relative_strength(stock_df, index_df, weeks=13)
        assert rs == pytest.approx(-1.5)

    def test_relative_strength_empty_dataframes(self):
        """Empty DataFrames return NaN."""
        empty = pd.DataFrame(columns=["Close"])
        rs = calculate_relative_strength(empty, empty, weeks=13)
        assert math.isnan(rs)

    def test_calculate_relative_strength_mismatched_lengths(self):
        """Stock and index with different lengths but enough overlapping data works."""
        fridays_stock = pd.date_range("2025-01-10", periods=20, freq="W-FRI")
        stock_df = pd.DataFrame(
            {"Close": [100.0] * 19 + [200.0]}, index=fridays_stock
        )
        fridays_index = pd.date_range("2025-01-10", periods=10, freq="W-FRI")
        index_df = pd.DataFrame({"Close": [100.0] * 10}, index=fridays_index)

        rs = calculate_relative_strength(stock_df, index_df, weeks=13)
        # Only 10 overlapping rows < 14 needed → NaN
        assert math.isnan(rs)

    def test_calculate_relative_strength_stock_sufficient_index_insufficient(self):
        """Stock has enough data but insufficient overlapping index data → NaN."""
        fridays_stock = pd.date_range("2025-01-10", periods=20, freq="W-FRI")
        stock_df = pd.DataFrame(
            {"Close": [100.0] * 19 + [200.0]}, index=fridays_stock
        )
        # Index has only 5 weeks (different date range)
        fridays_index = pd.date_range("2025-03-14", periods=5, freq="W-FRI")
        index_df = pd.DataFrame({"Close": [100.0] * 5}, index=fridays_index)

        rs = calculate_relative_strength(stock_df, index_df, weeks=13)
        # Overlap is only 0-5 rows → NaN
        assert math.isnan(rs)


# ===================================================================
# TASK-006: Technical Features (integrated)
# ===================================================================


class TestCalculateTechnicalFeatures:
    """Tests for ``calculate_technical_features()``."""

    @pytest.fixture
    def config(self):
        return {
            "technical": {
                "sma_short": 20,
                "sma_long": 50,
                "relative_strength_weeks": 13,
                "sma_rising_lookback": 3,
            }
        }

    @pytest.fixture
    def index_weekly(self):
        """15 weeks of index data with Close=100 for first 14, then 150."""
        fridays = pd.date_range("2025-01-10", periods=15, freq="W-FRI")
        close = [100.0] * 14 + [150.0]
        return pd.DataFrame({"Close": close}, index=fridays)

    def test_sma_rising_check(self, config, index_weekly):
        """SMA20 rising over 3 weeks is correctly detected."""
        # 25 weeks of monotonically increasing close: 100, 101, ..., 124
        stock_weekly = _weekly_df(n_weeks=25, close_start=100.0, close_step=1.0)

        result = calculate_technical_features(stock_weekly, index_weekly, config)

        assert result["sma_short_is_rising"] is True
        assert len(result["warnings"]) == 0
        assert not math.isnan(result["sma_short"])

    def test_sma_rising_insufficient_data(self, config, index_weekly):
        """Fewer than lookback+1 SMA20 values → False with warning."""
        # 21 weeks gives 2 valid SMA20 values (indices 19, 20)
        stock_weekly = _weekly_df(n_weeks=21, close_start=100.0, close_step=1.0)

        result = calculate_technical_features(stock_weekly, index_weekly, config)

        assert result["sma_short_is_rising"] is False
        assert any("Insufficient SMA20 history" in w for w in result["warnings"])

    def test_technical_features_full_output(self, config, index_weekly):
        """Full output dict has expected keys and types."""
        stock_weekly = _weekly_df(n_weeks=25, close_start=100.0, close_step=2.0)

        result = calculate_technical_features(stock_weekly, index_weekly, config)

        expected_keys = {
            "sma_short",
            "sma_long",
            "distance_from_sma20",
            "relative_strength_13w",
            "sma_short_is_rising",
            "close",
            "warnings",
        }
        assert set(result.keys()) == expected_keys

        # Type checks
        assert isinstance(result["sma_short"], float) or math.isnan(
            result["sma_short"]
        )
        assert isinstance(result["sma_long"], float) or math.isnan(
            result["sma_long"]
        )
        assert isinstance(result["distance_from_sma20"], float) or math.isnan(
            result["distance_from_sma20"]
        )
        assert isinstance(result["relative_strength_13w"], float) or math.isnan(
            result["relative_strength_13w"]
        )
        assert isinstance(result["sma_short_is_rising"], bool)
        assert isinstance(result["close"], float) or math.isnan(result["close"])
        assert isinstance(result["warnings"], list)

    def test_technical_features_empty_stock(self, config, index_weekly):
        """Empty stock weekly returns NaN values and a warning."""
        empty = pd.DataFrame(columns=["Close"])
        result = calculate_technical_features(empty, index_weekly, config)

        assert math.isnan(result["sma_short"])
        assert math.isnan(result["close"])
        assert result["sma_short_is_rising"] is False
        assert any("Empty stock weekly data" in w for w in result["warnings"])

    def test_technical_features_empty_index(self, config):
        """Empty index weekly produces NaN relative_strength."""
        stock_weekly = _weekly_df(n_weeks=25, close_start=100.0, close_step=1.0)
        empty_index = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = calculate_technical_features(stock_weekly, empty_index, config)
        assert math.isnan(result["relative_strength_13w"])
        assert result["close"] == pytest.approx(124.0)  # 100 + 24

    def test_technical_features_distance_from_sma20(self, config, index_weekly):
        """distance_from_sma20 is correctly calculated."""
        # Close consistently at 100, SMA20 will converge to ~100 → distance ~0
        stock_weekly = _weekly_df(n_weeks=25, close_start=100.0, close_step=0.0)

        result = calculate_technical_features(stock_weekly, index_weekly, config)

        # Distance should be very close to 0 (SMA20 ≈ 100 when all close=100)
        assert result["distance_from_sma20"] == pytest.approx(0.0, abs=0.01)

    def test_technical_features_relative_strength(self, config, index_weekly):
        """RS_13w is returned in the output dict."""
        stock_weekly = _weekly_df(n_weeks=25, close_start=100.0, close_step=1.0)

        result = calculate_technical_features(stock_weekly, index_weekly, config)

        # RS should be a numeric value (not NaN for 25 weeks of data)
        assert not math.isnan(result["relative_strength_13w"])
