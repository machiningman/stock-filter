"""
Configuration loader and validator for the LQ45 Stock Screener.

All tunable thresholds come from config.yaml — no hardcoded values.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults for optional sub-keys
# ---------------------------------------------------------------------------

_TECHNICAL_DEFAULTS = {
    "sma_short": 20,
    "sma_long": 50,
    "max_distance_from_sma20": 0.15,
    "relative_strength_weeks": 13,
    "sma_rising_lookback": 3,
}

_FUNDAMENTAL_NON_BANK_DEFAULTS = {
    "min_roe": 10,
    "max_der": 1.5,
    "min_revenue_growth_yoy": 0,
    "min_profit_growth_yoy": -10,
    "require_operating_cashflow_positive": True,
}

_FUNDAMENTAL_BANK_DEFAULTS = {
    "min_roe": 10,
    "max_pbv": 2.5,
    "min_profit_growth_yoy": -10,
    "max_gross_npl": 3,
}

_SCORING_WEIGHTS_DEFAULTS = {
    "fundamental_quality": 0.35,
    "earnings_momentum": 0.20,
    "valuation": 0.15,
    "technical_trend": 0.20,
    "relative_strength": 0.10,
}

_SCORING_NORMALIZATION_DEFAULTS = {
    "roe": {"min": 0, "target": 25},
    "profit_growth_yoy": {"min": -20, "target": 30},
    "per": {"min": 30, "target": 10, "inverted": True},
    "pbv": {"min": 3.0, "target": 1.0, "inverted": True},
    "distance_from_sma20": {"min": 0.15, "target": 0.02, "inverted": True},
    "relative_strength_13w": {"min": -0.10, "target": 0.10},
}

_DATA_DEFAULTS = {
    "ticker_suffix": ".JK",
    "index_ticker": "^JKSE",
    "cache_ttl_days": 7,
    "price_history_months": 18,
}

_CLASSIFICATION_DEFAULTS = {
    "min_data_completeness": 0.6,
}

_SECTORS_DEFAULTS = {
    "bank": ["Bank", "Perbankan", "Financials - Bank"],
}

_LOGGING_DEFAULTS = {
    "level": "INFO",
    "format": "[%(asctime)s] %(levelname)s: %(message)s",
}

# ---------------------------------------------------------------------------
# Required top-level keys
# ---------------------------------------------------------------------------

_REQUIRED_TOP_KEYS = {
    "technical",
    "fundamental_non_bank",
    "fundamental_bank",
    "scoring",
    "data",
    "classification",
    "sectors",
    "logging",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEIGHT_TOLERANCE = 0.01


def _apply_section_defaults(
    section: dict, defaults: dict, section_name: str = "<unknown>"
) -> dict:
    """Fill missing sub-keys in *section* with values from *defaults*."""
    if section is not None and not isinstance(section, dict):
        raise ValueError(
            f"Expected a mapping for section '{section_name}', "
            f"got {type(section).__name__}."
        )
    merged = dict(defaults)
    merged.update(section or {})
    return merged


def _validate_scoring_weights(config: dict) -> None:
    """Log a warning if scoring weights do not sum to approximately 1.0."""
    weights = config.get("scoring", {}).get("weights", {})
    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_TOLERANCE:
        logger.warning(
            "Scoring weights sum to %.3f, expected ~1.0. "
            "Adjust weights in config.yaml under scoring.weights.",
            total,
        )


def _validate_normalization_ranges(config: dict) -> None:
    """Raise ValueError if any normalization range has min == target."""
    normalization = config.get("scoring", {}).get("normalization", {})
    for metric_name, params in normalization.items():
        min_val = params.get("min")
        target_val = params.get("target")
        if min_val is not None and target_val is not None and min_val == target_val:
            raise ValueError(
                f"Normalization range for '{metric_name}' has min == target "
                f"({min_val}). min and target must be different."
            )
        # Additional invariant checks (informative, not blocking):
        inverted = params.get("inverted", False)
        if inverted and min_val is not None and target_val is not None:
            if min_val <= target_val:
                logger.warning(
                    "Inverted metric '%s': expected min > target but got "
                    "min=%.4f, target=%.4f.",
                    metric_name,
                    min_val,
                    target_val,
                )
        elif not inverted and min_val is not None and target_val is not None:
            if min_val >= target_val:
                logger.warning(
                    "Non-inverted metric '%s': expected min < target but got "
                    "min=%.4f, target=%.4f.",
                    metric_name,
                    min_val,
                    target_val,
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """
    Load and validate a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Validated configuration dictionary with defaults applied.

    Raises
    ------
    ValueError
        If the file does not exist, YAML is invalid, or validation fails.
    """
    if not os.path.exists(config_path):
        raise ValueError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with open(config_path, encoding="utf-8") as fh:
            config: dict[str, Any] = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML in configuration file '{config_path}': {exc}"
        ) from exc

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file '{config_path}' must contain a YAML mapping "
            f"(key-value pairs), got {type(config).__name__}."
        )

    # Check required top-level keys
    missing = _REQUIRED_TOP_KEYS - set(config.keys())
    if missing:
        raise ValueError(
            f"Configuration file '{config_path}' is missing required "
            f"top-level key(s): {sorted(missing)}"
        )

    # Apply defaults for sub-keys
    config["technical"] = _apply_section_defaults(
        config.get("technical"), _TECHNICAL_DEFAULTS, "technical"
    )
    config["fundamental_non_bank"] = _apply_section_defaults(
        config.get("fundamental_non_bank"), _FUNDAMENTAL_NON_BANK_DEFAULTS, "fundamental_non_bank"
    )
    config["fundamental_bank"] = _apply_section_defaults(
        config.get("fundamental_bank"), _FUNDAMENTAL_BANK_DEFAULTS, "fundamental_bank"
    )

    # Scoring section
    scoring = config.get("scoring", {})
    weights = _apply_section_defaults(
        scoring.get("weights"), _SCORING_WEIGHTS_DEFAULTS, "scoring.weights"
    )
    normalization = _apply_section_defaults(
        scoring.get("normalization"), _SCORING_NORMALIZATION_DEFAULTS, "scoring.normalization"
    )
    config["scoring"] = {"weights": weights, "normalization": normalization}

    config["data"] = _apply_section_defaults(
        config.get("data"), _DATA_DEFAULTS, "data"
    )
    config["classification"] = _apply_section_defaults(
        config.get("classification"), _CLASSIFICATION_DEFAULTS, "classification"
    )
    config["sectors"] = _apply_section_defaults(
        config.get("sectors"), _SECTORS_DEFAULTS, "sectors"
    )
    config["logging"] = _apply_section_defaults(
        config.get("logging"), _LOGGING_DEFAULTS, "logging"
    )

    # Validation
    _validate_scoring_weights(config)
    _validate_normalization_ranges(config)

    return config


def get_bank_sectors(config: dict) -> list[str]:
    """
    Return lowercased bank sector names from the config.

    Parameters
    ----------
    config : dict
        Validated configuration dictionary.

    Returns
    -------
    list[str]
        Lowercased bank sector names.
    """
    return [s.lower() for s in config.get("sectors", {}).get("bank", [])]


def is_bank_sector(sector: str, config: dict) -> bool:
    """
    Check whether *sector* is a bank sector (case-insensitive).

    Parameters
    ----------
    sector : str
        Sector name to check.
    config : dict
        Validated configuration dictionary.

    Returns
    -------
    bool
    """
    return sector.lower() in get_bank_sectors(config)
