"""
Pipeline module for the LQ45 Stock Screener.

Handles weekly resampling of daily price data and calculation of
technical indicators (TASK-006).
"""

from __future__ import annotations

import logging
import os

from typing import Any

import numpy as np
import pandas as pd

from stock_screener.src.config import is_bank_sector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TASK-006: Weekly Resampling and Technical Indicators
# ---------------------------------------------------------------------------


def resample_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily price data to weekly (Friday-ending) OHLCV bars.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily price data with a DatetimeIndex and columns
        ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.
        The ``Close`` column is expected to be the adjusted close (when
        ``auto_adjust=True`` is used with ``yfinance``, the column is
        already named ``"Close"`` and is adjusted). If both ``"Adj Close"``
        and ``"Close"`` are present the adjusted value from ``"Adj Close"``
        is used for the weekly close.

    Returns
    -------
    pd.DataFrame
        Weekly OHLCV DataFrame with a DatetimeIndex of Friday labels.
        All-NaN rows are dropped. If the input contained an ``"Adj Close"``
        column it is carried through in the output.
    """
    if daily_df.empty:
        return pd.DataFrame()

    # If both "Close" and "Adj Close" exist, use "Adj Close" for the weekly Close
    if "Adj Close" in daily_df.columns and "Close" in daily_df.columns:
        daily_df = daily_df.copy()
        daily_df["Close"] = daily_df["Adj Close"]  # ensure adjusted close is used

    agg_map = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    if "Adj Close" in daily_df.columns:
        agg_map["Adj Close"] = "last"

    weekly = daily_df.resample("W-FRI").agg(agg_map)
    weekly = weekly.dropna(how="all")
    return weekly


def calculate_sma(
    weekly_df: pd.DataFrame,
    period: int,
    column: str = "Close",
) -> pd.Series:
    """Calculate simple moving average of the specified column.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly OHLCV DataFrame.
    period : int
        Rolling window period (weeks).
    column : str, optional
        Column to calculate SMA on (default ``"Close"``).

    Returns
    -------
    pd.Series
        SMA values, indexed like ``weekly_df``.
    """
    return weekly_df[column].rolling(window=period).mean()


def calculate_distance_from_sma(
    close: pd.Series,
    sma: pd.Series,
) -> pd.Series:
    """Calculate the distance of price from its SMA as a fraction.

    Formula: ``(close - sma) / sma``

    Parameters
    ----------
    close : pd.Series
        Close price series.
    sma : pd.Series
        SMA series (same length/index as ``close``).

    Returns
    -------
    pd.Series
        Distance as a decimal fraction (e.g., 0.05 = 5% above SMA).
    """
    return (close - sma) / sma


def calculate_relative_strength(
    stock_weekly: pd.DataFrame,
    index_weekly: pd.DataFrame,
    weeks: int,
) -> float:
    """Calculate the N-week relative strength vs the index.

    Formula::

        stock_return = (close[-1] / close[-weeks-1]) - 1
        index_return = (index_close[-1] / index_close[-weeks-1]) - 1
        relative_strength = stock_return - index_return

    Parameters
    ----------
    stock_weekly : pd.DataFrame
        Weekly OHLCV DataFrame for the stock.
    index_weekly : pd.DataFrame
        Weekly OHLCV DataFrame for the index (e.g., IHSG).
    weeks : int
        Lookback period in weeks (typically 13).

    Returns
    -------
    float
        Relative strength as a decimal (e.g., 0.05 = 5% outperformance).
        Returns ``NaN`` if insufficient data.
    """
    needed = weeks + 1
    # Align by date index so that stock and index dates match
    stock_close = stock_weekly["Close"]
    index_close = index_weekly["Close"]
    aligned = pd.DataFrame({"stock": stock_close, "index": index_close}).dropna()
    if len(aligned) < needed:
        return float("nan")
    stock_return = (aligned["stock"].iloc[-1] / aligned["stock"].iloc[-needed]) - 1.0
    index_return = (aligned["index"].iloc[-1] / aligned["index"].iloc[-needed]) - 1.0
    return float(stock_return - index_return)


def calculate_technical_features(
    stock_weekly: pd.DataFrame,
    index_weekly: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """Calculate technical indicator features for a stock.

    Computes SMA20, SMA50, distance from SMA20, relative strength,
    and SMA20 rising trend.

    Parameters
    ----------
    stock_weekly : pd.DataFrame
        Weekly OHLCV DataFrame for the stock.
    index_weekly : pd.DataFrame
        Weekly OHLCV DataFrame for the index.
    config : dict
        Full configuration dictionary with a ``technical`` section.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``sma_short`` — latest SMA20 value
        - ``sma_long`` — latest SMA50 value
        - ``distance_from_sma20`` — latest distance from SMA20
        - ``relative_strength_13w`` — 13-week relative strength
        - ``sma_short_is_rising`` — whether SMA20 is rising
        - ``close`` — latest close price
        - ``warnings`` — list of warning strings
    """
    tech_cfg = config.get("technical", {})
    sma_short_period = tech_cfg["sma_short"]
    sma_long_period = tech_cfg["sma_long"]
    rs_weeks = tech_cfg["relative_strength_weeks"]
    sma_rising_lookback = tech_cfg["sma_rising_lookback"]

    warnings: list[str] = []

    # --- Guard: empty input --------------------------------------------------
    if stock_weekly.empty:
        return {
            "sma_short": float("nan"),
            "sma_long": float("nan"),
            "distance_from_sma20": float("nan"),
            "relative_strength_13w": float("nan"),
            "sma_short_is_rising": False,
            "close": float("nan"),
            "warnings": ["Empty stock weekly data"],
        }

    # --- SMA20 ---------------------------------------------------------------
    sma_short_series = calculate_sma(stock_weekly, sma_short_period)
    sma_short_value = (
        float(sma_short_series.iloc[-1])
        if not sma_short_series.empty
        else float("nan")
    )

    # --- SMA50 ---------------------------------------------------------------
    sma_long_series = calculate_sma(stock_weekly, sma_long_period)
    sma_long_value = (
        float(sma_long_series.iloc[-1])
        if not sma_long_series.empty
        else float("nan")
    )

    # --- Distance from SMA20 -------------------------------------------------
    close_series = stock_weekly["Close"]
    distance_series = calculate_distance_from_sma(close_series, sma_short_series)
    distance_value = (
        float(distance_series.iloc[-1])
        if not distance_series.empty
        else float("nan")
    )

    # --- Relative strength ---------------------------------------------------
    rs_value = calculate_relative_strength(stock_weekly, index_weekly, rs_weeks)

    # --- SMA20 rising check --------------------------------------------------
    sma_20_valid = sma_short_series.dropna()
    if len(sma_20_valid) < sma_rising_lookback + 1:
        sma_short_is_rising = False
        warnings.append("Insufficient SMA20 history for rising check")
    else:
        sma_short_is_rising = bool(
            sma_20_valid.iloc[-1] > sma_20_valid.iloc[-1 - sma_rising_lookback]
        )

    # --- Latest close --------------------------------------------------------
    latest_close = (
        float(close_series.iloc[-1]) if not close_series.empty else float("nan")
    )

    return {
        "sma_short": sma_short_value,
        "sma_long": sma_long_value,
        "distance_from_sma20": distance_value,
        "relative_strength_13w": rs_value,
        "sma_short_is_rising": sma_short_is_rising,
        "close": latest_close,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# TASK-007: Fundamental Hard Filters
# ---------------------------------------------------------------------------


def apply_fundamental_filter(fundamentals_row: pd.Series, config: dict) -> dict:
    """Apply fundamental hard filters to a single stock's fundamentals row.

    Determines bank vs non-bank via :func:`~stock_screener.src.config.is_bank_sector`
    and applies the appropriate threshold rules from *config*.

    Parameters
    ----------
    fundamentals_row : pd.Series
        A row of fundamental data. Expected fields depend on sector type:

        **Bank sectors** — ``sector``, ``roe``, ``pbv``,
        ``net_profit_growth_yoy``, (optional) ``gross_npl``.

        **Non-bank sectors** — ``sector``, ``roe``, ``der``,
        ``revenue_growth_yoy``, ``net_profit_growth_yoy``,
        (optional) ``operating_cashflow_positive``.
    config : dict
        Full configuration dictionary (as returned by ``load_config``).

    Returns
    -------
    dict
        Dictionary with keys:

        - **passes** (*bool*) — ``True`` when no FAIL reasons exist.
        - **reasons** (*list[str]*) — per-condition PASS / FAIL messages.
        - **warnings** (*list[str]*) — non-blocking notes (e.g., missing
          optional fields).
        - **missing_fields** (*list[str]*) — names of required fields that
          were missing (``NaN``).
    """
    sector = fundamentals_row.get("sector", "")
    if pd.isna(sector):
        return {
            "passes": False,
            "reasons": ["FAIL: sector is missing"],
            "warnings": [],
            "missing_fields": ["sector"],
        }
    sector = str(sector).strip()
    bank_cfg = config.get("fundamental_bank", {})
    non_bank_cfg = config.get("fundamental_non_bank", {})

    reasons: list[str] = []
    warnings: list[str] = []
    missing_fields: list[str] = []

    if is_bank_sector(sector, config):
        # --- Bank filter rules -------------------------------------------

        # ROE
        _check_required_ge(
            fundamentals_row, "roe",
            bank_cfg["min_roe"],
            suffix="%",
            reasons=reasons,
            missing_fields=missing_fields,
        )

        # PBV
        _check_required_le(
            fundamentals_row, "pbv",
            bank_cfg["max_pbv"],
            reasons=reasons,
            missing_fields=missing_fields,
        )

        # Net Profit Growth YoY
        _check_required_ge(
            fundamentals_row, "net_profit_growth_yoy",
            bank_cfg["min_profit_growth_yoy"],
            suffix="%",
            label="Net Profit Growth YoY",
            reasons=reasons,
            missing_fields=missing_fields,
        )

        # Gross NPL (optional — skip if NaN, add warning)
        gross_npl = fundamentals_row.get("gross_npl")
        max_gross_npl = bank_cfg["max_gross_npl"]
        if pd.isna(gross_npl):
            warnings.append("Gross NPL data is missing – NPL check skipped")
        elif gross_npl > max_gross_npl:
            reasons.append(
                f"FAIL: Gross NPL={gross_npl}% (above max {max_gross_npl}%)"
            )
        else:
            reasons.append(
                f"PASS: Gross NPL={gross_npl}% (below max {max_gross_npl}%)"
            )

    else:
        # --- Non-bank filter rules ---------------------------------------

        # ROE
        _check_required_ge(
            fundamentals_row, "roe",
            non_bank_cfg["min_roe"],
            suffix="%",
            reasons=reasons,
            missing_fields=missing_fields,
        )

        # DER
        _check_required_le(
            fundamentals_row, "der",
            non_bank_cfg["max_der"],
            reasons=reasons,
            missing_fields=missing_fields,
        )

        # Revenue Growth YoY
        _check_required_ge(
            fundamentals_row, "revenue_growth_yoy",
            non_bank_cfg["min_revenue_growth_yoy"],
            suffix="%",
            label="Revenue Growth YoY",
            reasons=reasons,
            missing_fields=missing_fields,
        )

        # Net Profit Growth YoY
        _check_required_ge(
            fundamentals_row, "net_profit_growth_yoy",
            non_bank_cfg["min_profit_growth_yoy"],
            suffix="%",
            label="Net Profit Growth YoY",
            reasons=reasons,
            missing_fields=missing_fields,
        )

        # Operating Cashflow Positive (optional per config)
        require_ocf = non_bank_cfg["require_operating_cashflow_positive"]
        if require_ocf:
            ocf = fundamentals_row.get("operating_cashflow_positive")
            if pd.isna(ocf):
                warnings.append(
                    "Operating Cashflow Positive data is missing – "
                    "check skipped"
                )
            elif not ocf:
                reasons.append("FAIL: Operating Cashflow is not positive")
            else:
                reasons.append("PASS: Operating Cashflow is positive")

    passes = not any(r.startswith("FAIL:") for r in reasons)

    return {
        "passes": passes,
        "reasons": reasons,
        "warnings": warnings,
        "missing_fields": missing_fields,
    }


# ---------------------------------------------------------------------------
# TASK-008: Technical Hard Filters
# ---------------------------------------------------------------------------


def apply_technical_filter(tech_features: dict, config: dict) -> dict:
    """Apply technical hard filters based on computed technical features.

    Parameters
    ----------
    tech_features : dict
        Dictionary from :func:`calculate_technical_features` with keys:
        ``close``, ``sma_short``, ``sma_long``, ``sma_short_is_rising``,
        ``relative_strength_13w``, ``distance_from_sma20``, ``warnings``.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict
        Dictionary with keys:

        - **passes** (*bool*) — ``True`` when no FAIL reasons exist.
        - **reasons** (*list[str]*) — per-condition PASS / FAIL messages.
        - **warnings** (*list[str]*) — non-blocking notes.
    """
    tech_cfg = config.get("technical", {})
    max_distance = tech_cfg["max_distance_from_sma20"]

    reasons: list[str] = []
    warnings: list[str] = list(tech_features.get("warnings", []))

    close = tech_features.get("close")
    sma_short = tech_features.get("sma_short")
    sma_long = tech_features.get("sma_long")
    rs = tech_features.get("relative_strength_13w")
    distance = tech_features.get("distance_from_sma20")
    sma_rising = tech_features.get("sma_short_is_rising", False)

    # 1. Close > SMA20
    if pd.isna(sma_short) or pd.isna(close):
        reasons.append("FAIL: SMA20 data is insufficient")
        warnings.append("Insufficient SMA20 data for technical filter")
    elif close <= sma_short:
        reasons.append(f"FAIL: Close={close} <= SMA20={sma_short}")
    else:
        reasons.append(f"PASS: Close={close} > SMA20={sma_short}")

    # 2. Close > SMA50
    if pd.isna(sma_long) or pd.isna(close):
        reasons.append("FAIL: SMA50 data is insufficient")
        warnings.append("Insufficient SMA50 data for technical filter")
    elif close <= sma_long:
        reasons.append(f"FAIL: Close={close} <= SMA50={sma_long}")
    else:
        reasons.append(f"PASS: Close={close} > SMA50={sma_long}")

    # 3. SMA20 rising
    if not sma_rising:
        reasons.append("FAIL: SMA20 is not rising")
    else:
        reasons.append("PASS: SMA20 is rising")

    # 4. RS > 0
    if pd.isna(rs):
        reasons.append("FAIL: Relative Strength data is insufficient")
        warnings.append("Insufficient data for Relative Strength calculation")
    elif rs <= 0:
        reasons.append(f"FAIL: Relative Strength={rs} (below 0)")
    else:
        reasons.append(f"PASS: Relative Strength={rs} (above 0)")

    # 5. Distance from SMA20 < max_distance
    if pd.isna(distance):
        reasons.append("FAIL: Distance from SMA20 data is insufficient")
        warnings.append("Insufficient data for Distance from SMA20 calculation")
    elif abs(distance) >= max_distance:
        reasons.append(
            f"FAIL: |Distance from SMA20|={abs(distance):.4f} (>= {max_distance:.4f})"
        )
    else:
        reasons.append(
            f"PASS: |Distance from SMA20|={abs(distance):.4f} (< {max_distance:.4f})"
        )

    passes = not any(r.startswith("FAIL:") for r in reasons)

    return {
        "passes": passes,
        "reasons": reasons,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_required_ge(
    row: pd.Series,
    field: str,
    threshold: float,
    *,
    suffix: str = "",
    label: str | None = None,
    reasons: list[str],
    missing_fields: list[str],
) -> None:
    """Check that *field* >= *threshold*, treating NaN as a FAIL."""
    display = label or field.replace("_", " ").title()
    value = row.get(field)
    if pd.isna(value):
        reasons.append(f"FAIL: {display} is missing")
        missing_fields.append(field)
    elif value < threshold:
        reasons.append(
            f"FAIL: {display}={value}{suffix} (below min {threshold}{suffix})"
        )
    else:
        reasons.append(
            f"PASS: {display}={value}{suffix} (above min {threshold}{suffix})"
        )


def _check_required_le(
    row: pd.Series,
    field: str,
    threshold: float,
    *,
    suffix: str = "",
    label: str | None = None,
    reasons: list[str],
    missing_fields: list[str],
) -> None:
    """Check that *field* <= *threshold*, treating NaN as a FAIL."""
    display = label or field.replace("_", " ").title()
    value = row.get(field)
    if pd.isna(value):
        reasons.append(f"FAIL: {display} is missing")
        missing_fields.append(field)
    elif value > threshold:
        reasons.append(
            f"FAIL: {display}={value}{suffix} (above max {threshold}{suffix})"
        )
    else:
        reasons.append(
            f"PASS: {display}={value}{suffix} (below max {threshold}{suffix})"
        )


# ---------------------------------------------------------------------------
# TASK-009: Classification Logic
# ---------------------------------------------------------------------------


def calculate_data_completeness(
    fundamentals_row: pd.Series,
    required_fields: list[str],
    has_price_data: bool,
) -> float:
    """Calculate data completeness score (0.0–1.0) for a stock.

    The completeness is a weighted combination of fundamental field
    presence (50%) and price data availability (50%).

    Parameters
    ----------
    fundamentals_row : pd.Series
        A row of fundamental data.
    required_fields : list[str]
        Field names that are *required* for the completeness denominator.
        Optional fields (e.g. ``operating_cashflow_positive``,
        ``gross_npl``, ``nim``, ``loan_growth_yoy``) should be
        **excluded** from this list.
    has_price_data : bool
        Whether weekly price data is available for this stock.

    Returns
    -------
    float
        Completeness score between 0.0 and 1.0.
    """
    if not required_fields:
        fundamental_completeness = 1.0
    else:
        present = sum(
            1 for field in required_fields
            if not pd.isna(fundamentals_row.get(field))
        )
        fundamental_completeness = present / len(required_fields)

    price_completeness = 1.0 if has_price_data else 0.0
    return fundamental_completeness * 0.5 + price_completeness * 0.5


def classify_stock(
    fundamental_result: dict,
    technical_result: dict,
    data_completeness: float,
    config: dict,
) -> str:
    """Classify a stock based on filter results and data completeness.

    Decision tree::

        insufficient data          → "Avoid"
        fundamental + technical    → "Candidate"
        fundamental only           → "Watch"
        technical only             → "Speculative"
        neither                    → "Avoid"

    Parameters
    ----------
    fundamental_result : dict
        Output from :func:`apply_fundamental_filter` with a ``passes`` key.
    technical_result : dict
        Output from :func:`apply_technical_filter` with a ``passes`` key.
    data_completeness : float
        Completeness score from :func:`calculate_data_completeness`.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    str
        One of ``"Candidate"``, ``"Watch"``, ``"Speculative"``, ``"Avoid"``.
    """
    min_completeness = config["classification"]["min_data_completeness"]

    if data_completeness < min_completeness:
        logger.warning(
            "Data completeness %.2f < %.2f — classifying as Avoid",
            data_completeness,
            min_completeness,
        )
        return "Avoid"

    fundamental_pass = fundamental_result.get("passes", False)
    technical_pass = technical_result.get("passes", False)

    if fundamental_pass and technical_pass:
        return "Candidate"
    if fundamental_pass and not technical_pass:
        return "Watch"
    if not fundamental_pass and technical_pass:
        return "Speculative"
    return "Avoid"


# ---------------------------------------------------------------------------
# TASK-010: Scoring Module
# ---------------------------------------------------------------------------


def normalize_score(
    value: float,
    min_val: float,
    target_val: float,
    inverted: bool = False,
) -> float:
    """Normalise a raw metric value to a 0–100 score.

    Non-inverted formula the higher the better::

        score = clip((value - min) / (target - min), 0, 1) * 100

    Inverted formula the lower the better::

        score = clip((min - value) / (min - target), 0, 1) * 100

    Parameters
    ----------
    value : float
        Raw metric value.
    min_val : float
        Minimum expected value (maps to 0).
    target_val : float
        Target / ideal value (maps to 100).
    inverted : bool, optional
        If ``True``, lower raw values yield higher scores.

    Returns
    -------
    float
        Score in [0, 100].  Returns 0 for NaN or degenerate ranges.
    """
    if pd.isna(value):
        return 0.0

    # Guard against degenerate range
    if abs(target_val - min_val) < 1e-10:
        logger.warning(
            "Normalisation range degenerate: min=%.4f, target=%.4f — returning 0",
            min_val,
            target_val,
        )
        return 0.0

    if not inverted:
        raw = (value - min_val) / (target_val - min_val)
    else:
        raw = (min_val - value) / (min_val - target_val)

    return float(np.clip(raw, 0.0, 1.0) * 100.0)


def calculate_fundamental_score(
    fundamentals_row: pd.Series,
    config: dict,
) -> float:
    """Score fundamental quality using ROE.

    Parameters
    ----------
    fundamentals_row : pd.Series
        A row of fundamental data with a ``roe`` field.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    float
        Score 0–100.
    """
    norm = config.get("scoring", {}).get("normalization", {}).get("roe", {})
    roe = fundamentals_row.get("roe", float("nan"))
    return normalize_score(
        roe,
        min_val=norm["min"],
        target_val=norm["target"],
        inverted=norm.get("inverted", False),
    )


def calculate_earnings_momentum_score(
    fundamentals_row: pd.Series,
    config: dict,
) -> float:
    """Score earnings momentum using net profit growth YoY.

    Parameters
    ----------
    fundamentals_row : pd.Series
        A row of fundamental data with a ``net_profit_growth_yoy`` field.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    float
        Score 0–100.
    """
    norm = (
        config.get("scoring", {})
        .get("normalization", {})
        .get("profit_growth_yoy", {})
    )
    growth = fundamentals_row.get("net_profit_growth_yoy", float("nan"))
    return normalize_score(
        growth,
        min_val=norm["min"],
        target_val=norm["target"],
        inverted=norm.get("inverted", False),
    )


def calculate_valuation_score(
    fundamentals_row: pd.Series,
    sector: str,
    config: dict,
) -> float:
    """Score valuation using PBV (banks) or PER (non-banks).

    Parameters
    ----------
    fundamentals_row : pd.Series
        A row of fundamental data with ``pbv`` (banks) or ``per`` (non-banks).
    sector : str
        Sector name for bank / non-bank detection.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    float
        Score 0–100.
    """
    if pd.isna(sector) or not isinstance(sector, str):
        return 0.0

    if is_bank_sector(sector, config):
        norm = (
            config.get("scoring", {}).get("normalization", {}).get("pbv", {})
        )
        value = fundamentals_row.get("pbv", float("nan"))
    else:
        norm = (
            config.get("scoring", {}).get("normalization", {}).get("per", {})
        )
        value = fundamentals_row.get("per", float("nan"))

    return normalize_score(
        value,
        min_val=norm["min"],
        target_val=norm["target"],
        inverted=norm.get("inverted", True),
    )


def calculate_technical_score(
    tech_features: dict,
    config: dict,
) -> float:
    """Score technical trend using absolute deviation from target distance.

    Scoring uses two-sided absolute deviation from zero:
    - Score 100 when distance equals target (0.02 = 2% above SMA20)
    - Score 0 when |distance| reaches max boundary (0.15)
    - Linear interpolation between target and max boundary
    - Symmetric: +0.15 and -0.15 both score 0

    Formula::

        abs_distance = abs(distance)
        deviation = max(0, abs_distance - target)
        max_dev = max_distance - target
        score = clip(1 - deviation / max_dev, 0, 1) * 100

    Parameters
    ----------
    tech_features : dict
        Output from :func:`calculate_technical_features` containing
        a ``distance_from_sma20`` key.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    float
        Score 0–100.
    """
    norm = (
        config.get("scoring", {})
        .get("normalization", {})
        .get("distance_from_sma20", {})
    )
    distance = tech_features.get("distance_from_sma20", float("nan"))

    if pd.isna(distance):
        return 0.0

    target = norm["target"]    # 0.02
    min_val = norm["min"]      # 0.15
    max_dev = min_val - target  # 0.13

    if abs(max_dev) < 1e-10:
        logger.warning(
            "Technical score degenerate range: min=%.4f, target=%.4f — returning 0",
            min_val,
            target,
        )
        return 0.0

    abs_distance = abs(distance)
    deviation = max(0.0, abs_distance - target)
    raw = 1.0 - (deviation / max_dev)
    return float(np.clip(raw, 0.0, 1.0) * 100.0)


def calculate_relative_strength_score(
    rs: float,
    config: dict,
) -> float:
    """Score relative strength.

    Parameters
    ----------
    rs : float
        13-week relative strength value.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    float
        Score 0–100.
    """
    norm = (
        config.get("scoring", {})
        .get("normalization", {})
        .get("relative_strength_13w", {})
    )
    return normalize_score(
        rs,
        min_val=norm["min"],
        target_val=norm["target"],
        inverted=norm.get("inverted", False),
    )


def calculate_final_score(
    sub_scores: dict,
    weights: dict,
) -> float:
    """Calculate the weighted final score.

    Formula::

        final = sum(score * weight for each component)

    Clamped to [0, 100].

    Parameters
    ----------
    sub_scores : dict
        Dictionary of sub-score names to float values.
    weights : dict
        Dictionary of weight names to float values (must sum to ~1.0).

    Returns
    -------
    float
        Final score clamped to 0–100.
    """
    total = 0.0
    for key in weights:
        score = sub_scores.get(key, 0.0)
        weight = weights.get(key, 0.0)
        total += score * weight
    return float(np.clip(total, 0.0, 100.0))


# ---------------------------------------------------------------------------
# TASK-011: Report Export
# ---------------------------------------------------------------------------


def generate_report(results: list[dict], report_dir: str, report_date: str) -> str:
    """Generate a CSV screening report from a list of result dicts.

    Results are sorted first by status priority Candidate (0), Watch (1),
    Speculative (2), Avoid (3), then by ``final_score`` descending within
    each group.

    The CSV is saved to ``{report_dir}/weekly_screening_YYYY-MM-DD.csv``.

    Parameters
    ----------
    results : list[dict]
        Each dict has keys matching the output columns (see Returns).
    report_dir : str
        Directory to write the report file into (created if missing).
    report_date : str
        Report date string used for the filename (e.g. ``"2025-01-17"``).

    Returns
    -------
    str
        Absolute path to the generated CSV file.
    """
    _STATUS_PRIORITY = {
        "Candidate": 0,
        "Watch": 1,
        "Speculative": 2,
        "Avoid": 3,
    }

    # --- Build rows ---------------------------------------------------------
    rows = []
    for r in results:
        # Convert list fields to semicolon-separated strings
        reasons = r.get("reasons", [])
        warnings = r.get("warnings", [])
        missing_data_flags = r.get("missing_data_flags", [])

        if isinstance(reasons, list):
            reasons_str = ";".join(str(x) for x in reasons)
        else:
            reasons_str = str(reasons) if reasons else ""

        if isinstance(warnings, list):
            warnings_str = ";".join(str(x) for x in warnings)
        else:
            warnings_str = str(warnings) if warnings else ""

        if isinstance(missing_data_flags, list):
            missing_str = ";".join(str(x) for x in missing_data_flags)
        else:
            missing_str = str(missing_data_flags) if missing_data_flags else ""

        ticker = str(r.get("ticker", ""))
        sector = str(r.get("sector", ""))
        final_score = r.get("final_score", 0.0)
        status = str(r.get("status", "Avoid"))
        fundamental_score = r.get("fundamental_score", 0.0)
        earnings_momentum_score = r.get("earnings_momentum_score", 0.0)
        technical_score = r.get("technical_score", 0.0)
        valuation_score = r.get("valuation_score", 0.0)
        relative_strength_score = r.get("relative_strength_score", 0.0)
        company_name = str(r.get("company_name", ""))
        close = r.get("close", float("nan"))
        weekly_sma20 = r.get("weekly_sma20", float("nan"))
        weekly_sma50 = r.get("weekly_sma50", float("nan"))
        distance_from_sma20 = r.get("distance_from_sma20", float("nan"))
        relative_strength_13w = r.get("relative_strength_13w", float("nan"))

        warning_count = len(warnings_str.split(";")) if warnings_str else 0
        missing_count = len(missing_str.split(";")) if missing_str else 0

        if pd.isna(final_score):
            final_score = 0.0

        suggested_review_note = (
            f"Review {ticker}: {sector} sector, "
            f"score {final_score:.0f}/100 ({status}). "
            f"{warning_count} warning(s). "
            f"{missing_count} missing field(s)."
        )

        rows.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "final_score": final_score,
                "status": status,
                "fundamental_score": fundamental_score,
                "earnings_momentum_score": earnings_momentum_score,
                "technical_score": technical_score,
                "valuation_score": valuation_score,
                "relative_strength_score": relative_strength_score,
                "close": close,
                "weekly_sma20": weekly_sma20,
                "weekly_sma50": weekly_sma50,
                "distance_from_sma20": distance_from_sma20,
                "relative_strength_13w": relative_strength_13w,
                "reasons": reasons_str,
                "warnings": warnings_str,
                "missing_data_flags": missing_str,
                "suggested_review_note": suggested_review_note,
            }
        )

    # --- Sort: status priority ascending, final_score descending ------------
    def _sort_key(row: dict) -> tuple:
        priority = _STATUS_PRIORITY.get(row.get("status", "Avoid"), 99)
        score = row.get("final_score", 0.0)
        if pd.isna(score):
            score = 0.0
        return (priority, -score)

    rows.sort(key=_sort_key)

    # --- Write CSV ----------------------------------------------------------
    os.makedirs(report_dir, exist_ok=True)
    filename = f"weekly_screening_{report_date}.csv"
    filepath = os.path.join(report_dir, filename)

    # Define column ordering explicitly to ensure they exist even for empty results
    columns = [
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
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(filepath, index=False, encoding="utf-8")

    logger.info("Report saved to %s", filepath)
    return os.path.abspath(filepath)



