"""
Data I/O module for the LQ45 Stock Screener.

Handles CSV loading/validation (TASK-004) and price fetching via yfinance
with caching (TASK-005).
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UNIVERSE_REQUIRED_COLUMNS = frozenset(
    {"ticker", "company_name", "sector", "effective_period"}
)
_FUNDAMENTALS_MINIMUM_COLUMNS = frozenset({"ticker", "sector", "roe"})
# NOTE: With ``auto_adjust=True``, yfinance returns ``Close`` as the
# adjusted price.  There is no separate ``Adj Close`` column in the
# returned data.  Downstream code should use the ``Close`` column,
# which IS the adjusted close.  This is an intentional deviation from
# the plan's literal column list, which had called for ``Adj Close``.
_PRICE_CACHE_EXPECTED_COLUMNS = frozenset(
    {"Open", "High", "Low", "Close", "Volume"}
)

# Hard sanity thresholds (not config-driven; these are just "this can't be
# right" guards, not filter cut-offs).
_MAX_ROE = 200.0
_MIN_DER = 0.0

# ---------------------------------------------------------------------------
# TASK-004: CSV Loading and Validation
# ---------------------------------------------------------------------------


def load_universe(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the LQ45 constituents CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the LQ45 universe.

    Returns
    -------
    pd.DataFrame
        DataFrame with validated universe data.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    missing = _UNIVERSE_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Universe CSV is missing required columns: {sorted(missing)}"
        )

    # Guard against NaN / empty ticker values (L2 requirement).
    # Must check before astype(str) since NaN becomes the string "nan".
    if df["ticker"].isna().any():
        raise ValueError("Universe CSV contains rows with empty ticker values")

    # Strip whitespace from ticker and sector values
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["sector"] = df["sector"].astype(str).str.strip()

    # Normalize sector to title case
    df["sector"] = df["sector"].str.title()

    return df


def load_fundamentals(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the fundamentals CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing fundamental data.

    Returns
    -------
    pd.DataFrame
        DataFrame with validated fundamental data.

    Raises
    ------
    ValueError
        If minimum required columns are missing.
    """
    df = pd.read_csv(csv_path)

    # Validate minimum required columns exist
    missing = _FUNDAMENTALS_MINIMUM_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Fundamentals CSV is missing required columns: {sorted(missing)}"
        )

    # Columns not expected to be numeric
    non_numeric_cols = frozenset({"ticker", "sector"})

    # Convert numeric columns to float (non-numeric -> NaN)
    for col in df.columns:
        if col not in non_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Log warning for rows with missing required fields
    for idx, row in df.iterrows():
        missing_fields = []
        for field in _FUNDAMENTALS_MINIMUM_COLUMNS:
            val = row.get(field)
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                missing_fields.append(field)
        if missing_fields:
            logger.warning(
                "Row %d (ticker='%s'): missing required field(s): %s",
                idx,
                row.get("ticker", "?"),
                missing_fields,
            )

    return df


def validate_data(
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    config: dict,  # noqa: ARG001
) -> dict[str, Any]:
    """
    Validate universe and fundamentals data.

    Checks performed:
    - Ticker overlap (universe tickers missing from fundamentals)
    - Duplicate tickers in both DataFrames
    - Out-of-range values (e.g., ROE > 200, DER < 0)

    Parameters
    ----------
    universe : pd.DataFrame
        Universe DataFrame from load_universe().
    fundamentals : pd.DataFrame
        Fundamentals DataFrame from load_fundamentals().
    config : dict
        Configuration dictionary (reserved for future extensibility).

    Returns
    -------
    dict
        Validation report with keys:
        - ``errors``: list of error messages (blocking)
        - ``warnings``: list of warning messages (non-blocking)
        - ``is_valid``: bool indicating if data passed all checks
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- Duplicate ticker checks & deduplication ---

    if "ticker" in universe.columns:
        dup_count = universe["ticker"].duplicated(keep="first").sum()
        if dup_count > 0:
            universe = universe.drop_duplicates(subset=["ticker"], keep="first")
            errors.append(
                f"Removed {dup_count} duplicate ticker(s) from universe (kept first)"
            )

    if "ticker" in fundamentals.columns:
        dup_count = fundamentals["ticker"].duplicated(keep="first").sum()
        if dup_count > 0:
            fundamentals = fundamentals.drop_duplicates(
                subset=["ticker"], keep="first"
            )
            errors.append(
                f"Removed {dup_count} duplicate ticker(s) from fundamentals "
                "(kept first)"
            )

    # --- Ticker overlap check ---

    if "ticker" in universe.columns and "ticker" in fundamentals.columns:
        uni_tickers: set[str] = set(universe["ticker"].unique())
        fund_tickers: set[str] = set(fundamentals["ticker"].unique())

        missing_in_fund = uni_tickers - fund_tickers
        if missing_in_fund:
            for t in sorted(missing_in_fund):
                warnings.append(
                    f"Ticker '{t}' is in universe but not in fundamentals"
                )

        missing_in_uni = fund_tickers - uni_tickers
        if missing_in_uni:
            for t in sorted(missing_in_uni):
                warnings.append(
                    f"Ticker '{t}' is in fundamentals but not in universe"
                )

    # --- Out-of-range value checks ---

    if "roe" in fundamentals.columns:
        bad_roe_mask = fundamentals["roe"] > _MAX_ROE
        bad_roe = fundamentals.loc[bad_roe_mask]
        for _, row in bad_roe.iterrows():
            errors.append(
                f"Ticker '{row['ticker']}' has ROE={row['roe']:.2f}, "
                f"exceeds max {_MAX_ROE}"
            )

    if "der" in fundamentals.columns:
        bad_der_mask = fundamentals["der"] < _MIN_DER
        bad_der = fundamentals.loc[bad_der_mask]
        for _, row in bad_der.iterrows():
            errors.append(
                f"Ticker '{row['ticker']}' has DER={row['der']:.2f}, "
                f"below min {_MIN_DER}"
            )

    is_valid = len(errors) == 0

    return {
        "errors": errors,
        "warnings": warnings,
        "is_valid": is_valid,
        "universe": universe,
        "fundamentals": fundamentals,
    }


# ---------------------------------------------------------------------------
# TASK-005: Price Fetching and Caching
# ---------------------------------------------------------------------------


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance columns to simple strings.

    yfinance 1.3.0+ returns DataFrames with ``MultiIndex`` columns of the
    form ``(col_name, ticker)`` even for single-ticker downloads.  This
    helper drops the ticker level so downstream code can access columns by
    simple names like ``"Close"``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may have MultiIndex columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with simple (flat) column names.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.droplevel(1)
    return df


@retry(
    wait=wait_exponential(min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
)
def _fetch_single_ticker(ticker: str, start, end) -> pd.DataFrame:
    """
    Fetch a single ticker from yfinance with automatic retries.

    Parameters
    ----------
    ticker : str
        Full ticker symbol (e.g., ``"BBCA.JK"``).
    start : datetime-like
        Start date for price history.
    end : datetime-like
        End date for price history.

    Returns
    -------
    pd.DataFrame
        Price data from yfinance (``auto_adjust=True`` so Close is adjusted).
        Columns are normalised to simple strings (MultiIndex flattened).
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
    )
    return _normalize_columns(df)


def _delete_cache_files(full_ticker: str, cache_dir: str) -> None:
    """Remove cache files (parquet + meta) for *full_ticker*."""
    for name in (
        f"{full_ticker}.parquet",
        f"{full_ticker}.meta.json",
    ):
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            os.remove(path)


def _load_cached_ticker(
    full_ticker: str,
    cache_dir: str,
    cache_ttl_days: int,
) -> pd.DataFrame | None:
    """
    Attempt to load cached price data for a ticker.

    Returns the cached DataFrame if valid and fresh, or ``None`` if the
    cache is missing, expired, or corrupted (in which case stale files are
    cleaned up).
    """
    parquet_path = os.path.join(cache_dir, f"{full_ticker}.parquet")
    meta_path = os.path.join(cache_dir, f"{full_ticker}.meta.json")

    if not (os.path.exists(parquet_path) and os.path.exists(meta_path)):
        return None

    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        last_fetched = datetime.fromisoformat(meta.get("last_fetched", ""))
        # Backward compat: older cache files stored naive timestamps
        if last_fetched.tzinfo is None:
            last_fetched = last_fetched.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(tz=timezone.utc) - last_fetched).days

        if age_days >= cache_ttl_days:
            logger.info("Cache expired for %s (age: %d days)", full_ticker, age_days)
            _delete_cache_files(full_ticker, cache_dir)
            return None

        df: pd.DataFrame = pd.read_parquet(parquet_path)

        # Validate cache content
        if df.empty:
            logger.warning("Cached data empty for %s, re-fetching", full_ticker)
            _delete_cache_files(full_ticker, cache_dir)
            return None

        if not _PRICE_CACHE_EXPECTED_COLUMNS.issubset(set(df.columns)):
            logger.warning(
                "Cached data for %s missing expected columns, re-fetching",
                full_ticker,
            )
            _delete_cache_files(full_ticker, cache_dir)
            return None

        # Normalise columns for backward compat with cached MultiIndex data
        df = _normalize_columns(df)

        logger.debug("Loaded cached data for %s", full_ticker)
        return df

    except Exception:
        logger.exception("Cache read error for %s, re-fetching", full_ticker)
        _delete_cache_files(full_ticker, cache_dir)
        return None


def _save_cached_ticker(
    df: pd.DataFrame,
    full_ticker: str,
    cache_dir: str,
) -> None:
    """Atomically save price data to cache (temp file + atomic rename)."""
    tmp_parquet = os.path.join(cache_dir, f"{full_ticker}.tmp.parquet")
    tmp_meta = os.path.join(cache_dir, f"{full_ticker}.tmp.meta.json")
    parquet_path = os.path.join(cache_dir, f"{full_ticker}.parquet")
    meta_path = os.path.join(cache_dir, f"{full_ticker}.meta.json")

    # Write to temp files
    df.to_parquet(tmp_parquet)

    meta = {
        "last_fetched": datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    }
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)

    # Atomic rename (os.replace is atomic on the same filesystem)
    os.replace(tmp_parquet, parquet_path)
    os.replace(tmp_meta, meta_path)


def _fetch_and_cache_ticker(
    full_ticker: str,
    start_date,
    end_date,
    cache_dir: str,
    cache_ttl_days: int,
) -> pd.DataFrame:
    """
    Internal: fetch a single ticker with caching.

    Checks cache first; if miss / expired / invalid, fetches from yfinance,
    saves atomically, and returns the data.
    """
    # Try cache first
    cached = _load_cached_ticker(full_ticker, cache_dir, cache_ttl_days)
    if cached is not None:
        return cached

    # Fetch from yfinance (with ``@retry``)
    try:
        df = _fetch_single_ticker(full_ticker, start_date, end_date)
    except Exception as exc:
        logger.error("Failed to fetch %s after retries: %s", full_ticker, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        logger.warning("No data returned for %s", full_ticker)
        return pd.DataFrame()

    # Save to cache atomically
    _save_cached_ticker(df, full_ticker, cache_dir)

    return df


def fetch_prices(
    tickers: list[str],
    config: dict,
    cache_dir: str,
) -> dict[str, pd.DataFrame]:
    """
    Fetch price data for a list of tickers with caching.

    Each ticker is fetched individually via yfinance with retry logic.
    Cached data is used if available and within the TTL window.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols (without suffix, e.g. ``"BBCA"``).
    config : dict
        Configuration dictionary with a ``data`` section.
    cache_dir : str
        Directory for caching price data. Created if it does not exist.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker -> DataFrame with price data.
        Empty DataFrames are returned for tickers that failed to fetch.
    """
    suffix = config.get("data", {}).get("ticker_suffix", ".JK")
    cache_ttl_days = config.get("data", {}).get("cache_ttl_days", 7)
    price_history_months = config.get("data", {}).get("price_history_months", 18)

    os.makedirs(cache_dir, exist_ok=True)

    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=price_history_months)

    result: dict[str, pd.DataFrame] = {}

    for i, raw_ticker in enumerate(tickers):
        full_ticker = raw_ticker + suffix

        df = _fetch_and_cache_ticker(
            full_ticker,
            start_date,
            end_date,
            cache_dir,
            cache_ttl_days,
        )

        result[raw_ticker] = df

        # Inter-request delay to avoid rate-limiting (skip after last ticker)
        if i < len(tickers) - 1:
            time.sleep(random.uniform(0.5, 1.5))

    return result


def fetch_index(
    config: dict,
    cache_dir: str,
) -> pd.DataFrame:
    """
    Fetch the IHSG index (``^JKSE``) price data with caching.

    Parameters
    ----------
    config : dict
        Configuration dictionary with a ``data`` section.
    cache_dir : str
        Directory for caching price data. Created if it does not exist.

    Returns
    -------
    pd.DataFrame
        DataFrame with index price data.
    """
    index_ticker = config.get("data", {}).get("index_ticker", "^JKSE")
    cache_ttl_days = config.get("data", {}).get("cache_ttl_days", 7)
    price_history_months = config.get("data", {}).get("price_history_months", 18)

    os.makedirs(cache_dir, exist_ok=True)

    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=price_history_months)

    return _fetch_and_cache_ticker(
        index_ticker,
        start_date,
        end_date,
        cache_dir,
        cache_ttl_days,
    )
