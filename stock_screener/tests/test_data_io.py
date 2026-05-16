"""
Tests for stock_screener.src.data_io — CSV loading, validation, and price
fetching with caching.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stock_screener.src.data_io import (
    fetch_index,
    fetch_prices,
    load_fundamentals,
    load_universe,
    validate_data,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_price_df(n_days: int = 100) -> pd.DataFrame:
    """Build a DataFrame shaped like what yfinance returns (auto_adjust=True)."""
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(42)
    data = {
        "Open": 100.0 + rng.standard_normal(n_days) * 5,
        "High": 102.0 + rng.standard_normal(n_days) * 5,
        "Low": 98.0 + rng.standard_normal(n_days) * 5,
        "Close": 101.0 + rng.standard_normal(n_days) * 5,
        "Volume": rng.integers(1_000_000, 10_000_000, n_days),
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# CSV string samples
# ---------------------------------------------------------------------------

SAMPLE_UNIVERSE_CSV = """\
ticker,company_name,sector,effective_period
BBCA,Bank Central Asia Tbk,Bank,2025-01
BBRI,Bank Rakyat Indonesia Tbk,Bank,2025-01
TLKM,Telkom Indonesia Tbk,Telecommunication,2025-01
ASII,Astra International Tbk,Automotive,2025-01
"""

SAMPLE_FUNDAMENTALS_CSV = """\
ticker,sector,roe,der,revenue_growth_yoy,net_profit_growth_yoy
BBCA,Bank,18.5,0.85,12.5,15.2
BBRI,Bank,20.0,0.90,10.0,12.0
TLKM,Telecommunication,15.0,1.20,5.0,3.0
ASII,Automotive,18.0,0.80,8.0,10.0
"""

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_universe_path(tmp_path):
    p = tmp_path / "universe.csv"
    p.write_text(SAMPLE_UNIVERSE_CSV, encoding="utf-8")
    return str(p)


@pytest.fixture
def sample_fundamentals_path(tmp_path):
    p = tmp_path / "fundamentals.csv"
    p.write_text(SAMPLE_FUNDAMENTALS_CSV, encoding="utf-8")
    return str(p)


@pytest.fixture
def valid_config():
    """Load the real project config.yaml."""
    from stock_screener.src.config import load_config

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    return load_config(os.path.join(_root, "config.yaml"))


# ===================================================================
# TASK-004: CSV Loading and Validation
# ===================================================================


class TestLoadUniverse:
    """Tests for ``load_universe()``."""

    def test_load_universe_valid(self, sample_universe_path):
        """Loads a valid CSV and returns a DataFrame with correct columns."""
        df = load_universe(sample_universe_path)
        assert list(df.columns) == [
            "ticker",
            "company_name",
            "sector",
            "effective_period",
        ]
        assert len(df) == 4
        assert df["ticker"].tolist() == ["BBCA", "BBRI", "TLKM", "ASII"]

    def test_load_universe_missing_columns(self, tmp_path):
        """Raises ValueError when required columns are missing."""
        p = tmp_path / "bad.csv"
        p.write_text("ticker,sector\nBBCA,Bank\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_universe(str(p))

    def test_load_universe_preserves_quotes_in_values(self, tmp_path):
        """Quotes inside CSV values are preserved (not whitespace stripping)."""
        p = tmp_path / "ws.csv"
        p.write_text(
            "ticker,company_name,sector,effective_period\n"
            ' "BBCA" ,"Bank Central Asia Tbk", "Bank" ,2025-01\n',
            encoding="utf-8",
        )
        df = load_universe(str(p))
        assert df["ticker"].iloc[0] == '"BBCA"'
        # With quoted CSV, pandas preserves quotes. Let's try without quotes.

    def test_load_universe_strips_whitespace_unquoted(self, tmp_path):
        """Whitespace around ticker / sector values (unquoted) is stripped."""
        p = tmp_path / "ws2.csv"
        p.write_text(
            "ticker,company_name,sector,effective_period\n"
            " BBCA ,Bank Central Asia Tbk, Bank ,2025-01\n",
            encoding="utf-8",
        )
        df = load_universe(str(p))
        assert df["ticker"].iloc[0] == "BBCA"
        assert df["sector"].iloc[0] == "Bank"

    def test_load_universe_title_case(self, tmp_path):
        """Sector values are normalised to title case."""
        p = tmp_path / "case.csv"
        p.write_text(
            "ticker,company_name,sector,effective_period\n"
            "BBCA,Bank Central Asia Tbk,bank,2025-01\n",
            encoding="utf-8",
        )
        df = load_universe(str(p))
        assert df["sector"].iloc[0] == "Bank"

    def test_load_universe_empty_csv(self, tmp_path):
        """An empty CSV (header only) returns an empty DataFrame."""
        p = tmp_path / "empty.csv"
        p.write_text(
            "ticker,company_name,sector,effective_period\n", encoding="utf-8"
        )
        df = load_universe(str(p))
        assert len(df) == 0
        assert list(df.columns) == [
            "ticker",
            "company_name",
            "sector",
            "effective_period",
        ]


class TestLoadFundamentals:
    """Tests for ``load_fundamentals()``."""

    def test_load_fundamentals_valid(self, sample_fundamentals_path):
        """Loads valid CSV; numeric columns are float64."""
        df = load_fundamentals(sample_fundamentals_path)
        assert "ticker" in df.columns
        assert "sector" in df.columns
        assert "roe" in df.columns
        assert df["roe"].dtype == np.float64
        assert df["der"].dtype == np.float64
        assert df["revenue_growth_yoy"].dtype == np.float64

    def test_load_fundamentals_non_numeric(self, tmp_path):
        """Non-numeric values in numeric columns become NaN."""
        p = tmp_path / "bad_num.csv"
        p.write_text(
            "ticker,sector,roe,der\n"
            "BBCA,Bank,18.5,0.85\n"
            "BBRI,Bank,abc,0.90\n",
            encoding="utf-8",
        )
        df = load_fundamentals(str(p))
        assert df.loc[1, "roe"] != df.loc[1, "roe"]  # NaN check (NaN != NaN)
        assert df.loc[0, "roe"] == 18.5

    def test_load_fundamentals_missing_columns(self, tmp_path):
        """Raises ValueError when required columns missing."""
        p = tmp_path / "bad.csv"
        p.write_text("ticker,sector\nBBCA,Bank\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_fundamentals(str(p))

    def test_load_fundamentals_logs_missing_fields(self, tmp_path, caplog):
        """Rows missing required fields trigger a warning."""
        p = tmp_path / "missing.csv"
        p.write_text(
            "ticker,sector,roe,der\nBBCA,Bank,18.5,0.85\n,Bank,,\n",
            encoding="utf-8",
        )
        caplog.set_level(logging.WARNING)
        load_fundamentals(str(p))
        assert any("missing required field" in r.message for r in caplog.records)


class TestValidateData:
    """Tests for ``validate_data()``."""

    def test_validate_data_valid(
        self, sample_universe_path, sample_fundamentals_path, valid_config
    ):
        """Clean data passes validation."""
        uni = load_universe(sample_universe_path)
        fund = load_fundamentals(sample_fundamentals_path)
        report = validate_data(uni, fund, valid_config)
        assert report["is_valid"] is True
        assert len(report["errors"]) == 0

    def test_validate_data_ticker_mismatch(self, valid_config):
        """Tickers in universe but not in fundamentals produce a warning."""
        uni = pd.DataFrame(
            {
                "ticker": ["BBCA", "BBRI", "EXTRA"],
                "company_name": ["a", "b", "c"],
                "sector": ["Bank", "Bank", "Tech"],
                "effective_period": ["2025-01", "2025-01", "2025-01"],
            }
        )
        fund = pd.DataFrame(
            {
                "ticker": ["BBCA", "BBRI"],
                "sector": ["Bank", "Bank"],
                "roe": [18.5, 20.0],
                "der": [0.85, 0.90],
            }
        )
        report = validate_data(uni, fund, valid_config)
        assert any("EXTRA" in w for w in report["warnings"])
        assert report["is_valid"] is True  # warnings are non-blocking

    def test_validate_data_out_of_range(self, valid_config):
        """ROE > 200 and DER < 0 are flagged as errors."""
        uni = pd.DataFrame(
            {
                "ticker": ["BBCA", "BBRI"],
                "company_name": ["a", "b"],
                "sector": ["Bank", "Bank"],
                "effective_period": ["2025-01", "2025-01"],
            }
        )
        fund = pd.DataFrame(
            {
                "ticker": ["BBCA", "BBRI"],
                "sector": ["Bank", "Bank"],
                "roe": [250.0, 20.0],
                "der": [0.85, -1.0],
            }
        )
        report = validate_data(uni, fund, valid_config)
        assert report["is_valid"] is False
        roe_errors = [e for e in report["errors"] if "ROE" in e]
        der_errors = [e for e in report["errors"] if "DER" in e]
        assert any("BBCA" in e for e in roe_errors)
        assert any("BBRI" in e for e in der_errors)

    def test_validate_data_duplicate_tickers(self, valid_config):
        """Duplicate tickers are detected, removed, and reported."""
        uni = pd.DataFrame(
            {
                "ticker": ["BBCA", "BBCA", "BBRI"],
                "company_name": ["a", "a", "b"],
                "sector": ["Bank", "Bank", "Bank"],
                "effective_period": ["2025-01", "2025-01", "2025-01"],
            }
        )
        fund = pd.DataFrame(
            {
                "ticker": ["BBCA", "BBRI", "BBRI"],
                "sector": ["Bank", "Bank", "Bank"],
                "roe": [18.5, 20.0, 20.0],
            }
        )
        report = validate_data(uni, fund, valid_config)
        assert report["is_valid"] is False
        uni_dup = [e for e in report["errors"] if "universe" in e.lower()]
        fund_dup = [e for e in report["errors"] if "fundamentals" in e.lower()]
        assert any("1 duplicate" in e for e in uni_dup)
        assert any("1 duplicate" in e for e in fund_dup)
        # Verify the cleaned DataFrames are returned without duplicates
        assert len(report["universe"]) == 2  # deduped from 3
        assert len(report["fundamentals"]) == 2  # deduped from 3


# ===================================================================
# TASK-005: Price Fetching and Caching
# ===================================================================


class TestFetchPrices:
    """Tests for ``fetch_prices()``."""

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _write_cache(
        cache_dir: str,
        ticker: str,
        df: pd.DataFrame | None = None,
        last_fetched: str | None = None,
    ):
        """Write cache files for *ticker* directly (bypass normal logic)."""
        os.makedirs(cache_dir, exist_ok=True)
        if df is not None:
            df.to_parquet(os.path.join(cache_dir, f"{ticker}.parquet"))
        meta = {
            "last_fetched": last_fetched
            or datetime.now().isoformat(timespec="seconds")
        }
        with open(
            os.path.join(cache_dir, f"{ticker}.meta.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(meta, fh)

    # -- tests ---------------------------------------------------------

    @patch("yfinance.download")
    def test_fetch_prices_uses_cache(
        self, mock_download, tmp_path, valid_config
    ):
        """First call fetches from yfinance; second call uses cache."""
        mock_df = _make_mock_price_df()
        mock_download.return_value = mock_df

        cache_dir = str(tmp_path / "cache")

        # First call — should fetch
        result1 = fetch_prices(["BBCA", "BBRI"], valid_config, cache_dir)
        assert mock_download.call_count == 2  # BBCA.JK + BBRI.JK
        assert "BBCA" in result1
        assert "BBRI" in result1
        assert not result1["BBCA"].empty

        # Reset mock for second call
        mock_download.reset_mock()
        result2 = fetch_prices(["BBCA", "BBRI"], valid_config, cache_dir)
        assert mock_download.call_count == 0  # No new fetches
        assert "BBCA" in result2
        assert not result2["BBCA"].empty

    @patch("stock_screener.src.data_io._fetch_single_ticker")
    def test_fetch_prices_handles_error(
        self, mock_fetch, tmp_path, valid_config
    ):
        """When yfinance fails, an empty DataFrame is returned for that ticker."""
        mock_fetch.side_effect = Exception("API Error")

        cache_dir = str(tmp_path / "cache")
        result = fetch_prices(["BBCA"], valid_config, cache_dir)
        assert "BBCA" in result
        assert result["BBCA"].empty

    @patch("yfinance.download")
    def test_fetch_prices_corrupted_cache(
        self, mock_download, tmp_path, valid_config
    ):
        """Corrupted cached data is cleaned up and re-fetched."""
        mock_df = _make_mock_price_df()
        mock_download.return_value = mock_df

        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)

        full_ticker = "BBCA.JK"

        # Write corrupted parquet (random bytes)
        bad_path = os.path.join(cache_dir, f"{full_ticker}.parquet")
        with open(bad_path, "wb") as fh:
            fh.write(b"not a valid parquet file\x00\xff")

        # Write valid meta (recent timestamp)
        meta_path = os.path.join(cache_dir, f"{full_ticker}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"last_fetched": datetime.now().isoformat(timespec="seconds")},
                fh,
            )

        result = fetch_prices(["BBCA"], valid_config, cache_dir)
        assert mock_download.called  # Re-fetched
        assert not result["BBCA"].empty
        # Corrupted cache should have been replaced with valid data
        assert os.path.exists(
            os.path.join(cache_dir, f"{full_ticker}.parquet")
        )

    @patch("yfinance.download")
    def test_fetch_prices_expired_cache(
        self, mock_download, tmp_path, valid_config
    ):
        """Expired cache triggers a re-fetch."""
        mock_df = _make_mock_price_df()
        mock_download.return_value = mock_df

        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)

        full_ticker = "BBCA.JK"
        old_date = (datetime.now() - timedelta(days=30)).isoformat(
            timespec="seconds"
        )

        # Write valid parquet + expired meta
        parquet_path = os.path.join(cache_dir, f"{full_ticker}.parquet")
        mock_df.to_parquet(parquet_path)
        meta_path = os.path.join(cache_dir, f"{full_ticker}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"last_fetched": old_date}, fh)

        mock_download.reset_mock()
        result = fetch_prices(["BBCA"], valid_config, cache_dir)
        assert mock_download.called  # Re-fetched due to expiry
        assert not result["BBCA"].empty

    @patch("yfinance.download")
    def test_fetch_prices_atomic_write(
        self, mock_download, tmp_path, valid_config
    ):
        """After fetch_prices, .tmp files should be gone and final files present."""
        mock_df = _make_mock_price_df()
        mock_download.return_value = mock_df

        cache_dir = str(tmp_path / "cache")
        fetch_prices(["BBCA"], valid_config, cache_dir)

        full_ticker = "BBCA.JK"

        # Temp files cleaned up
        assert not os.path.exists(
            os.path.join(cache_dir, f"{full_ticker}.tmp.parquet")
        )
        assert not os.path.exists(
            os.path.join(cache_dir, f"{full_ticker}.tmp.meta.json")
        )

        # Final files exist
        parquet_path = os.path.join(cache_dir, f"{full_ticker}.parquet")
        meta_path = os.path.join(cache_dir, f"{full_ticker}.meta.json")
        assert os.path.exists(parquet_path)
        assert os.path.exists(meta_path)

        # Verify meta content
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        assert "last_fetched" in meta

    @patch("yfinance.download")
    def test_fetch_prices_applies_suffix(
        self, mock_download, tmp_path, valid_config
    ):
        """Ticker suffix from config is applied (default: .JK)."""
        mock_df = _make_mock_price_df()
        mock_download.return_value = mock_df

        cache_dir = str(tmp_path / "cache")
        fetch_prices(["BBCA"], valid_config, cache_dir)

        # Should have called yfinance with 'BBCA.JK'
        call_args, _ = mock_download.call_args
        assert call_args[0] == "BBCA.JK"

    @patch("yfinance.download")
    def test_fetch_prices_empty_ticker_list(
        self, mock_download, tmp_path, valid_config
    ):
        """An empty ticker list returns an empty dict."""
        cache_dir = str(tmp_path / "cache")
        result = fetch_prices([], valid_config, cache_dir)
        assert result == {}
        mock_download.assert_not_called()

    @patch("yfinance.download")
    def test_fetch_prices_empty_cached_df(
        self, mock_download, tmp_path, valid_config
    ):
        """Empty cached DataFrame is detected, deleted, and re-fetched."""
        mock_df = _make_mock_price_df()
        mock_download.return_value = mock_df

        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)

        full_ticker = "BBCA.JK"

        # Write an empty parquet with correct columns + valid meta
        empty_df = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        )
        self._write_cache(
            cache_dir,
            full_ticker,
            empty_df,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        mock_download.reset_mock()
        result = fetch_prices(["BBCA"], valid_config, cache_dir)
        assert mock_download.called  # Re-fetched because cache was empty
        assert not result["BBCA"].empty

    @patch("yfinance.download")
    def test_fetch_prices_wrong_columns_cached(
        self, mock_download, tmp_path, valid_config
    ):
        """Cached data with wrong columns is detected, deleted, and re-fetched."""
        mock_df = _make_mock_price_df()
        mock_download.return_value = mock_df

        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)

        full_ticker = "BBCA.JK"

        # Write a parquet with only 2 columns (wrong) + valid meta
        wrong_df = _make_mock_price_df()[["Open", "Close"]]
        self._write_cache(
            cache_dir,
            full_ticker,
            wrong_df,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        mock_download.reset_mock()
        result = fetch_prices(["BBCA"], valid_config, cache_dir)
        assert mock_download.called  # Re-fetched because columns were wrong
        assert not result["BBCA"].empty


class TestFetchIndex:
    """Tests for ``fetch_index()``."""

    @patch("yfinance.download")
    def test_fetch_index_returns_dataframe(
        self, mock_download, tmp_path, valid_config
    ):
        """fetch_index returns a non-empty DataFrame."""
        mock_df = _make_mock_price_df()
        mock_download.return_value = mock_df

        cache_dir = str(tmp_path / "cache")
        df = fetch_index(valid_config, cache_dir)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @patch("yfinance.download")
    def test_fetch_index_uses_correct_ticker(
        self, mock_download, tmp_path, valid_config
    ):
        """fetch_index uses the index ticker from config (^JKSE)."""
        mock_df = _make_mock_price_df()
        mock_download.return_value = mock_df

        cache_dir = str(tmp_path / "cache")
        fetch_index(valid_config, cache_dir)

        call_args, _ = mock_download.call_args
        assert call_args[0] == "^JKSE"

    @patch("yfinance.download")
    def test_fetch_index_caches_and_reuses(
        self, mock_download, tmp_path, valid_config
    ):
        """Second call reuses cache."""
        mock_df = _make_mock_price_df()
        mock_download.return_value = mock_df

        cache_dir = str(tmp_path / "cache")
        fetch_index(valid_config, cache_dir)
        assert mock_download.call_count == 1

        mock_download.reset_mock()
        fetch_index(valid_config, cache_dir)
        assert mock_download.call_count == 0  # Cached
