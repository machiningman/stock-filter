"""
Integration tests for the LQ45 Weekly Stock Screener (TASK-013).

Tests the full pipeline end-to-end with mocked price data to avoid real
yfinance API calls.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stock_screener.src.config import is_bank_sector, load_config
from stock_screener.src.data_io import load_fundamentals, load_universe
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

# ---------------------------------------------------------------------------
# Sample CSV data
# ---------------------------------------------------------------------------

SAMPLE_UNIVERSE_CSV = """\
ticker,company_name,sector,effective_period
BBCA,Bank Central Asia Tbk,Bank,2025-01
BBRI,Bank Rakyat Indonesia Tbk,Bank,2025-01
TLKM,Telkom Indonesia Tbk,Telecommunication,2025-01
ASII,Astra International Tbk,Automotive,2025-01
"""

# Adjusted fundamentals where banks have PBV <= 2.5 so they pass the filter.
SAMPLE_FUNDAMENTALS_CSV = """\
ticker,sector,roe,der,revenue_growth_yoy,net_profit_growth_yoy,per,pbv,dividend_yield,operating_cashflow_positive,gross_npl,nim,loan_growth_yoy
BBCA,Bank,18.5,0.85,12.5,15.2,22.0,2.0,2.5,True,1.2,5.8,10.5
BBRI,Bank,20.0,0.90,10.0,12.0,18.0,2.0,3.0,True,2.1,6.5,8.0
TLKM,Telecommunication,15.0,1.20,5.0,3.0,14.0,2.5,4.0,True,,,
ASII,Automotive,18.0,0.80,8.0,10.0,10.0,2.0,3.5,True,,,
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BANK_REQUIRED_FIELDS = ["roe", "pbv", "net_profit_growth_yoy"]
_NON_BANK_REQUIRED_FIELDS = [
    "roe",
    "der",
    "revenue_growth_yoy",
    "net_profit_growth_yoy",
]


def _make_mock_price_df(n_days: int = 400) -> pd.DataFrame:
    """Generate mock OHLCV daily price data with a clear upward trend.

    Returns a DataFrame shaped like yfinance output with
    ``auto_adjust=True`` (columns: Open, High, Low, Close, Volume).
    """
    dates = pd.bdate_range("2024-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(42)

    # Monotonically increasing close with small noise
    close = np.linspace(100, 150, n_days) + rng.normal(0, 1, n_days)

    return pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 1, n_days),
            "High": close + rng.uniform(0, 2, n_days),
            "Low": close - rng.uniform(0, 2, n_days),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n_days),
        },
        index=dates,
    )


def _make_mock_index_df(n_days: int = 400) -> pd.DataFrame:
    """Generate mock index data with a slower upward trend (for RS > 0)."""
    dates = pd.bdate_range("2024-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(123)

    close = np.linspace(5000, 6000, n_days) + rng.normal(0, 10, n_days)

    return pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 20, n_days),
            "High": close + rng.uniform(0, 30, n_days),
            "Low": close - rng.uniform(0, 30, n_days),
            "Close": close,
            "Volume": rng.integers(100_000_000, 1_000_000_000, n_days),
        },
        index=dates,
    )


def _run_processing_loop(
    universe_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    index_weekly: pd.DataFrame,
    config: dict,
) -> list[dict]:
    """Execute the core stock processing loop.

    This replicates the logic inside ``main()`` so the test does not
    depend on file paths.
    """
    # Build fund lookup
    fund_lookup: dict[str, pd.Series] = {}
    for _, row in fundamentals_df.iterrows():
        fund_lookup[row["ticker"]] = row

    results: list[dict] = []

    for _, stock_row in universe_df.iterrows():
        ticker: str = stock_row["ticker"]
        sector: str = stock_row.get("sector", "")
        company_name: str = stock_row.get("company_name", "")

        # Get fundamentals row
        fund_row = fund_lookup.get(
            ticker, pd.Series({"ticker": ticker, "sector": sector})
        )

        # Get price data
        price_df = price_data.get(ticker, pd.DataFrame())

        # Resample
        stock_weekly = resample_to_weekly(price_df)

        # Technical features
        tech_features = calculate_technical_features(
            stock_weekly, index_weekly, config
        )

        # Fundamental filter
        fund_result = apply_fundamental_filter(fund_row, config)

        # Technical filter
        tech_result = apply_technical_filter(tech_features, config)

        # Data completeness
        has_price_data = not stock_weekly.empty
        required_fields = (
            _BANK_REQUIRED_FIELDS
            if is_bank_sector(sector, config)
            else _NON_BANK_REQUIRED_FIELDS
        )
        completeness = calculate_data_completeness(
            fund_row, required_fields, has_price_data
        )

        # Classify
        status = classify_stock(fund_result, tech_result, completeness, config)

        # Scores
        fundamental_score = calculate_fundamental_score(fund_row, config)
        earnings_momentum_score = calculate_earnings_momentum_score(fund_row, config)
        valuation_score = calculate_valuation_score(fund_row, sector, config)
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

        all_reasons = fund_result.get("reasons", []) + tech_result.get("reasons", [])
        all_warnings = fund_result.get("warnings", []) + tech_result.get(
            "warnings", []
        )

        results.append(
            {
                "ticker": ticker,
                "company_name": company_name,
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
                "distance_from_sma20": tech_features.get(
                    "distance_from_sma20", float("nan")
                ),
                "relative_strength_13w": tech_features.get(
                    "relative_strength_13w", float("nan")
                ),
                "reasons": all_reasons,
                "warnings": all_warnings,
                "missing_data_flags": fund_result.get("missing_fields", []),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_config():
    """Load the real project ``config.yaml``."""
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    return load_config(os.path.join(_root, "config.yaml"))


@pytest.fixture
def sample_universe_path(tmp_path):
    """Write a temporary universe CSV."""
    p = tmp_path / "universe.csv"
    p.write_text(SAMPLE_UNIVERSE_CSV, encoding="utf-8")
    return str(p)


@pytest.fixture
def sample_fundamentals_path(tmp_path):
    """Write a temporary fundamentals CSV."""
    p = tmp_path / "fundamentals.csv"
    p.write_text(SAMPLE_FUNDAMENTALS_CSV, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """Integration tests using mock yfinance responses."""

    # ------------------------------------------------------------------
    # TASK-013: End-to-End Integration Test
    # ------------------------------------------------------------------

    @patch("stock_screener.src.data_io.yf.download")
    def test_full_pipeline_with_mock_data(
        self,
        mock_download,
        sample_universe_path,
        sample_fundamentals_path,
        valid_config,
        tmp_path,
    ):
        """End-to-end test: loads CSVs, fetches mock prices, processes all
        stocks, generates a report, and verifies output integrity."""
        # Arrange: mock yfinance to return pre-recorded data.
        # Sequence: BBCA.JK, BBRI.JK, TLKM.JK, ASII.JK, ^JKSE
        mock_download.side_effect = [
            _make_mock_price_df(400),  # BBCA.JK
            _make_mock_price_df(400),  # BBRI.JK
            _make_mock_price_df(400),  # TLKM.JK
            _make_mock_price_df(400),  # ASII.JK
            _make_mock_index_df(400),  # ^JKSE
        ]

        # Load CSVs
        universe_df = load_universe(sample_universe_path)
        fundamentals_df = load_fundamentals(sample_fundamentals_path)

        # Fetch prices and index (uses mocked yfinance)
        from stock_screener.src.data_io import fetch_index, fetch_prices

        tickers = universe_df["ticker"].tolist()
        cache_dir = str(tmp_path / "cache")
        price_data = fetch_prices(tickers, valid_config, cache_dir)
        index_data = fetch_index(valid_config, cache_dir)

        # Index must have data
        assert not index_data.empty, "Mock index data should not be empty"

        index_weekly = resample_to_weekly(index_data)
        assert not index_weekly.empty, "Index weekly data should not be empty"

        # Act: run the core processing loop
        results = _run_processing_loop(
            universe_df, fundamentals_df, price_data, index_weekly, valid_config
        )

        # Generate report
        report_date = index_weekly.index[-1].strftime("%Y-%m-%d")
        report_dir = str(tmp_path / "reports")
        report_path = generate_report(results, report_dir, report_date)

        # ---------------------------------------------------------------
        # Assert
        # ---------------------------------------------------------------

        # --- Report file exists and is readable ---
        assert os.path.exists(report_path), f"Report not found at {report_path}"
        report_df = pd.read_csv(report_path)

        # --- All 4 stocks in report ---
        assert len(report_df) == 4, (
            f"Expected 4 rows in report, got {len(report_df)}"
        )

        # --- All required columns present ---
        required_columns = [
            "ticker",
            "company_name",
            "sector",
            "final_score",
            "status",
            "fundamental_score",
            "earnings_momentum_score",
            "technical_score",
            "valuation_score",
            "relative_strength_score",
            "close",
            "weekly_sma20",
            "weekly_sma50",
            "distance_from_sma20",
            "relative_strength_13w",
            "reasons",
            "warnings",
            "missing_data_flags",
            "suggested_review_note",
        ]
        for col in required_columns:
            assert col in report_df.columns, f"Missing required column: {col}"

        # --- At least one stock is not "Avoid" ---
        non_avoid = report_df[report_df["status"] != "Avoid"]
        assert len(non_avoid) > 0, (
            "Expected at least one stock classified as Candidate, Watch, "
            "or Speculative"
        )

        # --- All score columns are within [0, 100] ---
        score_cols = [
            "final_score",
            "fundamental_score",
            "earnings_momentum_score",
            "technical_score",
            "valuation_score",
            "relative_strength_score",
        ]
        for col in score_cols:
            scores = report_df[col]
            assert scores.between(0, 100, inclusive="both").all(), (
                f"Column '{col}' has values outside [0, 100]: "
                f"min={scores.min()}, max={scores.max()}"
            )

        # --- No NaN in final_score ---
        assert report_df["final_score"].notna().all(), (
            "Some rows have NaN final_score"
        )

        # --- Status column has valid values ---
        valid_statuses = {"Candidate", "Watch", "Speculative", "Avoid"}
        actual_statuses = set(report_df["status"].unique())
        assert actual_statuses.issubset(valid_statuses), (
            f"Unexpected status values: {actual_statuses - valid_statuses}"
        )

    @patch("stock_screener.src.data_io.yf.download")
    def test_pipeline_handles_missing_fundamentals(
        self,
        mock_download,
        sample_universe_path,
        valid_config,
        tmp_path,
    ):
        """Pipeline handles missing/empty fundamentals gracefully.

        All stocks should fail the fundamental filter and be classified
        as Speculative (technical-only pass) or Avoid. The report should
        still be generated.
        """
        # Arrange
        mock_download.side_effect = [
            _make_mock_price_df(400),  # BBCA.JK
            _make_mock_price_df(400),  # BBRI.JK
            _make_mock_price_df(400),  # TLKM.JK
            _make_mock_price_df(400),  # ASII.JK
            _make_mock_index_df(400),  # ^JKSE
        ]

        universe_df = load_universe(sample_universe_path)

        # Empty fundamentals (no data rows)
        empty_fundamentals = pd.DataFrame(
            columns=[
                "ticker",
                "sector",
                "roe",
                "der",
                "revenue_growth_yoy",
                "net_profit_growth_yoy",
                "per",
                "pbv",
                "dividend_yield",
                "operating_cashflow_positive",
                "gross_npl",
                "nim",
                "loan_growth_yoy",
            ]
        )

        from stock_screener.src.data_io import fetch_index, fetch_prices

        tickers = universe_df["ticker"].tolist()
        cache_dir = str(tmp_path / "cache")
        price_data = fetch_prices(tickers, valid_config, cache_dir)
        index_data = fetch_index(valid_config, cache_dir)

        index_weekly = resample_to_weekly(index_data)
        assert not index_weekly.empty

        # Act
        results = _run_processing_loop(
            universe_df,
            empty_fundamentals,
            price_data,
            index_weekly,
            valid_config,
        )

        # Generate report
        report_date = index_weekly.index[-1].strftime("%Y-%m-%d")
        report_dir = str(tmp_path / "reports")
        report_path = generate_report(results, report_dir, report_date)

        # ---------------------------------------------------------------
        # Assert
        # ---------------------------------------------------------------

        # --- Report exists ---
        assert os.path.exists(report_path), "Report should still be generated"

        # No stock should pass the fundamental filter
        # (no "Candidate" or "Watch" since fundamentals are missing)
        report_df = pd.read_csv(report_path)
        assert len(report_df) == 4, "All 4 stocks should be in the report"

        actual_statuses = set(report_df["status"].unique())
        assert "Candidate" not in actual_statuses, (
            "No stock should be Candidate without fundamentals"
        )
        assert "Watch" not in actual_statuses, (
            "No stock should be Watch without fundamentals"
        )
        assert actual_statuses.issubset({"Speculative", "Avoid"}), (
            f"Expected only Speculative or Avoid, got {actual_statuses}"
        )

        # --- All score columns are within [0, 100] ---
        score_cols = [
            "final_score",
            "fundamental_score",
            "earnings_momentum_score",
            "technical_score",
            "valuation_score",
            "relative_strength_score",
        ]
        for col in score_cols:
            scores = report_df[col]
            assert scores.between(0, 100, inclusive="both").all(), (
                f"Column '{col}' has values outside [0, 100]: "
                f"min={scores.min()}, max={scores.max()}"
            )

        # --- Fundamental-based scores should be 0 with missing data ---
        assert (report_df["fundamental_score"] == 0.0).all(), (
            "Fundamental score should be 0 when fundamentals are missing"
        )
