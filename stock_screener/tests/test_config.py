"""
Tests for stock_screener.src.config — config loading and validation.
"""

from __future__ import annotations

import logging
import os

import pytest
import yaml

from stock_screener.src.config import (
    get_bank_sectors,
    is_bank_sector,
    load_config,
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_config() -> dict:
    """Load the real config.yaml once per session (read-only test helper)."""
    return load_config(_CONFIG_PATH)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadConfigValid:
    def test_returns_dict_with_expected_keys(self, valid_config: dict):
        """load_config returns a dict with all required top-level keys."""
        expected_keys = {
            "technical",
            "fundamental_non_bank",
            "fundamental_bank",
            "scoring",
            "data",
            "classification",
            "sectors",
            "logging",
            "backtest",
        }
        assert expected_keys.issubset(valid_config.keys())

    def test_contains_technical_subs(self, valid_config: dict):
        tech = valid_config["technical"]
        for key in ("sma_short", "sma_long", "max_distance_from_sma20",
                     "relative_strength_weeks", "sma_rising_lookback"):
            assert key in tech

    def test_contains_scoring_weights(self, valid_config: dict):
        weights = valid_config["scoring"]["weights"]
        for key in ("fundamental_quality", "earnings_momentum",
                     "valuation", "technical_trend", "relative_strength"):
            assert key in weights

    def test_contains_normalization(self, valid_config: dict):
        norm = valid_config["scoring"]["normalization"]
        for key in ("roe", "profit_growth_yoy", "per", "pbv",
                     "distance_from_sma20", "relative_strength_13w"):
            assert key in norm

    def test_contains_sectors_bank(self, valid_config: dict):
        bank_sectors = valid_config["sectors"]["bank"]
        assert isinstance(bank_sectors, list)
        assert len(bank_sectors) > 0


class TestLoadConfigMissingFile:
    def test_raises_value_error(self):
        """Non-existent path raises ValueError with a clear message."""
        with pytest.raises(ValueError, match="not found"):
            load_config("nonexistent_file_xyz.yaml")

    def test_message_includes_filename(self, tmp_path):
        """Error message contains the path that was attempted."""
        bad_path = tmp_path / "missing.yaml"
        with pytest.raises(ValueError) as exc_info:
            load_config(str(bad_path))
        assert "missing.yaml" in str(exc_info.value)


class TestLoadConfigDefaults:
    def test_missing_technical_subs_get_defaults(self, tmp_path):
        """Missing technical sub-keys are filled with defaults."""
        cfg = {
            "technical": {},
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {},
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "minimal.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        result = load_config(str(p))
        assert result["technical"]["sma_short"] == 20
        assert result["technical"]["sma_long"] == 50

    def test_missing_scoring_weights_get_defaults(self, tmp_path):
        """Missing scoring.weights get filled with defaults."""
        cfg = {
            "technical": {},
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {},
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "no_weights.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        result = load_config(str(p))
        w = result["scoring"]["weights"]
        assert w["fundamental_quality"] == 0.35
        assert w["earnings_momentum"] == 0.20

    def test_missing_normalization_get_defaults(self, tmp_path):
        """Missing scoring.normalization get filled with defaults."""
        cfg = {
            "technical": {},
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {},
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "no_norm.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        result = load_config(str(p))
        n = result["scoring"]["normalization"]
        assert n["roe"]["min"] == 0
        assert n["roe"]["target"] == 25
        assert n["per"]["inverted"] is True

    def test_missing_data_subs_get_defaults(self, tmp_path):
        """Missing data sub-keys get filled with defaults."""
        cfg = {
            "technical": {},
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {},
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "no_data.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        result = load_config(str(p))
        assert result["data"]["ticker_suffix"] == ".JK"
        assert result["data"]["index_ticker"] == "^JKSE"
        assert result["data"]["cache_ttl_days"] == 7

    def test_missing_classification_subs_get_defaults(self, tmp_path):
        """Missing classification sub-keys get filled with defaults."""
        cfg = {
            "technical": {},
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {},
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "no_class.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        result = load_config(str(p))
        assert result["classification"]["min_data_completeness"] == 0.6

    def test_missing_sectors_subs_get_defaults(self, tmp_path):
        """Missing sectors sub-keys get filled with defaults."""
        cfg = {
            "technical": {},
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {},
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "no_sectors.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        result = load_config(str(p))
        assert result["sectors"]["bank"] == ["Bank", "Perbankan", "Financials - Bank"]

    def test_missing_logging_subs_get_defaults(self, tmp_path):
        """Missing logging sub-keys get filled with defaults."""
        cfg = {
            "technical": {},
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {},
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "no_logging.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        result = load_config(str(p))
        assert result["logging"]["level"] == "INFO"

    def test_missing_backtest_section_gets_defaults(self, tmp_path):
        """Missing backtest section gets filled with defaults."""
        cfg = {
            "technical": {},
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {},
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "no_backtest.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        result = load_config(str(p))
        assert result["backtest"]["history_months"] == 60
        assert result["backtest"]["horizons_weeks"] == [4, 13]
        assert result["backtest"]["min_warmup_weeks"] == 60


class TestWeightsValidation:
    def test_logs_warning_when_weights_not_summing_to_one(self, tmp_path, caplog):
        """Weights that sum != 1.0 (beyond tolerance) trigger a warning."""
        cfg = {
            "technical": {
                "sma_short": 20,
                "sma_long": 50,
                "max_distance_from_sma20": 0.15,
                "relative_strength_weeks": 13,
                "sma_rising_lookback": 3,
            },
            "fundamental_non_bank": {
                "min_roe": 10,
                "max_der": 1.5,
                "min_revenue_growth_yoy": 0,
                "min_profit_growth_yoy": -10,
                "require_operating_cashflow_positive": True,
            },
            "fundamental_bank": {
                "min_roe": 10,
                "max_pbv": 2.5,
                "min_profit_growth_yoy": -10,
                "max_gross_npl": 3,
            },
            "scoring": {
                "weights": {
                    "fundamental_quality": 0.50,
                    "earnings_momentum": 0.20,
                    "valuation": 0.15,
                    "technical_trend": 0.20,
                    "relative_strength": 0.10,
                },
                "normalization": {
                    "roe": {"min": 0, "target": 25},
                    "profit_growth_yoy": {"min": -20, "target": 30},
                    "per": {"min": 30, "target": 10, "inverted": True},
                    "pbv": {"min": 3.0, "target": 1.0, "inverted": True},
                    "distance_from_sma20": {
                        "min": 0.15, "target": 0.02, "inverted": True,
                    },
                    "relative_strength_13w": {"min": -0.10, "target": 0.10},
                },
            },
            "data": {
                "ticker_suffix": ".JK",
                "index_ticker": "^JKSE",
                "cache_ttl_days": 7,
                "price_history_months": 18,
            },
            "classification": {"min_data_completeness": 0.6},
            "sectors": {"bank": ["Bank"]},
            "logging": {
                "level": "INFO",
                "format": "[%(asctime)s] %(levelname)s: %(message)s",
            },
        }
        p = tmp_path / "bad_weights.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        caplog.set_level(logging.WARNING)
        load_config(str(p))
        assert any("Scoring weights sum to" in rec.message for rec in caplog.records)


class TestNormalizationValidation:
    def test_raises_value_error_when_min_equals_target(self, tmp_path):
        """Normalization range with min == target raises ValueError."""
        cfg = {
            "technical": {},
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {
                "normalization": {
                    "roe": {"min": 10, "target": 10},
                },
            },
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "bad_norm.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        with pytest.raises(ValueError, match="min == target"):
            load_config(str(p))


class TestIsBankSector:
    def test_identifies_bank_sectors(self, valid_config: dict):
        """Known bank sector returns True."""
        assert is_bank_sector("Bank", valid_config) is True

    def test_identifies_non_bank_sectors(self, valid_config: dict):
        """Known non-bank sector returns False."""
        assert is_bank_sector("Telecommunication", valid_config) is False
        assert is_bank_sector("Automotive", valid_config) is False
        assert is_bank_sector("Healthcare", valid_config) is False

    def test_case_insensitive(self, valid_config: dict):
        """Case-insensitive comparison for bank sector names."""
        assert is_bank_sector("bank", valid_config) is True
        assert is_bank_sector("BANK", valid_config) is True
        assert is_bank_sector("Bank", valid_config) is True

    def test_fully_case_insensitive_all_variants(self, valid_config: dict):
        """All casing variants of 'Bank' match."""
        for variant in ("bank", "BANK", "Bank", "bAnK"):
            assert is_bank_sector(variant, valid_config) is True, (
                f"Expected '{variant}' to be recognized as a bank sector"
            )


class TestGetBankSectors:
    def test_returns_lowercased_list(self, valid_config: dict):
        sectors = get_bank_sectors(valid_config)
        assert isinstance(sectors, list)
        assert all(s == s.lower() for s in sectors)
        assert "bank" in sectors


class TestLoadConfigInvalidYaml:
    def test_malformed_yaml_raises_value_error(self, tmp_path):
        """Invalid YAML syntax raises ValueError with filename in message."""
        p = tmp_path / "bad_syntax.yaml"
        with open(p, "w", encoding="utf-8") as f:
            f.write("unbalanced: [bracket")
        with pytest.raises(ValueError) as exc_info:
            load_config(str(p))
        assert "bad_syntax.yaml" in str(exc_info.value)


class TestLoadConfigEmptyYaml:
    def test_empty_yaml_raises_value_error(self, tmp_path):
        """Empty file raises ValueError mentioning missing required keys."""
        p = tmp_path / "empty.yaml"
        with open(p, "w", encoding="utf-8") as f:
            f.write("")
        with pytest.raises(ValueError) as exc_info:
            load_config(str(p))
        msg = str(exc_info.value)
        assert "missing required" in msg.lower()


class TestLoadConfigNonDictYaml:
    def test_list_root_raises_value_error(self, tmp_path):
        """YAML list root raises ValueError with clear mapping message."""
        p = tmp_path / "list_root.yaml"
        with open(p, "w", encoding="utf-8") as f:
            f.write("- item1\n- item2")
        with pytest.raises(ValueError, match="mapping"):
            load_config(str(p))

    def test_scalar_root_raises_value_error(self, tmp_path):
        """YAML scalar root raises ValueError with clear mapping message."""
        p = tmp_path / "scalar_root.yaml"
        with open(p, "w", encoding="utf-8") as f:
            f.write("42")
        with pytest.raises(ValueError, match="mapping"):
            load_config(str(p))


class TestLoadConfigPartialKeys:
    def test_only_some_keys_raises_value_error(self, tmp_path):
        """Providing only some required keys raises ValueError with sorted missing keys."""
        cfg = {
            "technical": {},
            "scoring": {},
            "data": {},
        }
        p = tmp_path / "partial.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        with pytest.raises(ValueError) as exc_info:
            load_config(str(p))
        msg = str(exc_info.value)
        missing_expected = sorted(
            {"fundamental_non_bank", "fundamental_bank", "classification", "sectors", "logging"}
        )
        for key in missing_expected:
            assert key in msg


class TestLoadConfigInvalidSectionType:
    def test_boolean_section_raises_value_error(self, tmp_path):
        """Non-dict section (bool) raises ValueError mentioning section name."""
        cfg = {
            "technical": True,
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {},
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "bool_section.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        with pytest.raises(ValueError) as exc_info:
            load_config(str(p))
        msg = str(exc_info.value)
        assert "technical" in msg
        assert "bool" in msg

    def test_list_section_raises_value_error(self, tmp_path):
        """Non-dict section (list) raises ValueError mentioning section name."""
        cfg = {
            "technical": [1, 2],
            "fundamental_non_bank": {},
            "fundamental_bank": {},
            "scoring": {},
            "data": {},
            "classification": {},
            "sectors": {},
            "logging": {},
        }
        p = tmp_path / "list_section.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        with pytest.raises(ValueError) as exc_info:
            load_config(str(p))
        msg = str(exc_info.value)
        assert "technical" in msg
        assert "list" in msg
