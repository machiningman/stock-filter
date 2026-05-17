
"""
Tests for stock_screener.src.pipeline — fundamental and technical hard
filters (TASK-007, TASK-008).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_screener.src.pipeline import (
    apply_fundamental_filter,
    apply_technical_filter,
)


# ===================================================================
# Helpers — Fundamental
# ===================================================================


def _bank_fundamentals(**overrides) -> pd.Series:
    """Build a bank fundamentals row with sensible defaults (all passing)."""
    defaults: dict = {
        "sector": "Bank",
        "roe": 15.0,
        "pbv": 1.5,
        "net_profit_growth_yoy": 5.0,
        "gross_npl": 2.0,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def _non_bank_fundamentals(**overrides) -> pd.Series:
    """Build a non-bank fundamentals row with sensible defaults (all passing)."""
    defaults: dict = {
        "sector": "Technology",
        "roe": 15.0,
        "der": 1.0,
        "revenue_growth_yoy": 5.0,
        "net_profit_growth_yoy": 5.0,
        "operating_cashflow_positive": True,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


# ===================================================================
# Helpers — Technical
# ===================================================================


def _passing_tech_features(**overrides) -> dict:
    """Build a tech-features dict with sensible defaults (all passing)."""
    defaults: dict = {
        "close": 110.0,
        "sma_short": 100.0,
        "sma_long": 90.0,
        "sma_short_is_rising": True,
        "relative_strength_13w": 0.05,
        "distance_from_sma20": 0.05,
        "warnings": [],
    }
    defaults.update(overrides)
    return defaults


# ===================================================================
# TASK-007: Fundamental Hard Filters — Bank
# ===================================================================


class TestFundamentalBankFilter:
    """Tests for ``apply_fundamental_filter()`` with bank sectors."""

    @pytest.fixture
    def config(self):
        return {
            "fundamental_bank": {
                "min_roe": 10,
                "max_pbv": 2.5,
                "min_profit_growth_yoy": -10,
                "max_gross_npl": 3,
            },
            "fundamental_non_bank": {},
            "sectors": {"bank": ["Bank", "Perbankan", "Financials - Bank"]},
        }

    def test_bank_filter_pass(self, config):
        """Bank stock with good fundamentals passes."""
        row = _bank_fundamentals()
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is True
        assert len(result["missing_fields"]) == 0
        # All individual checks should be PASS
        pass_reasons = [r for r in result["reasons"] if r.startswith("PASS:")]
        fail_reasons = [r for r in result["reasons"] if r.startswith("FAIL:")]
        assert len(pass_reasons) == 4  # ROE, PBV, NPG, Gross NPL
        assert len(fail_reasons) == 0

    def test_bank_filter_fail_roe(self, config):
        """Bank stock with ROE < 10 fails."""
        row = _bank_fundamentals(roe=4.0)
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("FAIL: Roe=4.0%" in r for r in result["reasons"])
        assert "roe" not in result["missing_fields"]

    def test_bank_filter_fail_pbv(self, config):
        """Bank stock with PBV > 2.5 fails."""
        row = _bank_fundamentals(pbv=3.0)
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("FAIL: Pbv=3.0" in r for r in result["reasons"])
        assert "pbv" not in result["missing_fields"]

    def test_bank_filter_missing_npl(self, config):
        """Bank stock with NaN NPL passes with warning."""
        row = _bank_fundamentals(gross_npl=float("nan"))
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is True
        assert any("Gross NPL" in w for w in result["warnings"])

    def test_bank_filter_missing_roe(self, config):
        """Bank stock with NaN ROE fails with 'FAIL: ROE is missing'."""
        row = _bank_fundamentals(roe=float("nan"))
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("FAIL: Roe is missing" in r for r in result["reasons"])
        assert "roe" in result["missing_fields"]

    def test_bank_filter_missing_pbv(self, config):
        """Bank stock with NaN PBV fails with missing reason."""
        row = _bank_fundamentals(pbv=float("nan"))
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("FAIL: Pbv is missing" in r for r in result["reasons"])
        assert "pbv" in result["missing_fields"]

    def test_bank_filter_missing_profit_growth(self, config):
        """Bank stock with NaN net_profit_growth_yoy fails."""
        row = _bank_fundamentals(net_profit_growth_yoy=float("nan"))
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("Net Profit Growth YoY is missing" in r for r in result["reasons"])
        assert "net_profit_growth_yoy" in result["missing_fields"]

    def test_bank_filter_fail_profit_growth(self, config):
        """Bank stock with profit growth below -10% fails."""
        row = _bank_fundamentals(net_profit_growth_yoy=-15.0)
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("Net Profit Growth YoY" in r and "below min" in r for r in result["reasons"])

    def test_bank_filter_fail_high_npl(self, config):
        """Bank stock with NPL above 3% fails."""
        row = _bank_fundamentals(gross_npl=5.0)
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("FAIL: Gross NPL" in r for r in result["reasons"])

    def test_non_bank_filter_per_bankan_sector(self, config):
        """Sector 'Perbankan' is treated as bank (not non-bank)."""
        row = _bank_fundamentals(sector="Perbankan")
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is True
        # Should have bank checks (PBV, etc.) not non-bank checks (DER, etc.)
        assert any("Pbv" in r for r in result["reasons"])

    def test_nan_sector_fails(self, config):
        """NaN sector should fail with missing-field reason."""
        row = _bank_fundamentals(sector=float("nan"))
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("sector is missing" in r for r in result["reasons"])
        assert "sector" in result["missing_fields"]


# ===================================================================
# TASK-007: Fundamental Hard Filters — Non-Bank
# ===================================================================


class TestFundamentalNonBankFilter:
    """Tests for ``apply_fundamental_filter()`` with non-bank sectors."""

    @pytest.fixture
    def config(self):
        return {
            "fundamental_non_bank": {
                "min_roe": 10,
                "max_der": 1.5,
                "min_revenue_growth_yoy": 0,
                "min_profit_growth_yoy": -10,
                "require_operating_cashflow_positive": True,
            },
            "fundamental_bank": {},
            "sectors": {"bank": ["Bank", "Perbankan", "Financials - Bank"]},
        }

    def test_non_bank_filter_pass(self, config):
        """Non-bank stock with good fundamentals passes."""
        row = _non_bank_fundamentals()
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is True
        assert len(result["missing_fields"]) == 0
        pass_reasons = [r for r in result["reasons"] if r.startswith("PASS:")]
        fail_reasons = [r for r in result["reasons"] if r.startswith("FAIL:")]
        assert len(pass_reasons) == 5  # ROE, DER, RevGrowth, ProfitGrowth, OCF
        assert len(fail_reasons) == 0

    def test_non_bank_filter_fail_roe(self, config):
        """Non-bank stock with ROE < 10 fails."""
        row = _non_bank_fundamentals(roe=4.0)
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("FAIL: Roe=4.0%" in r for r in result["reasons"])

    def test_non_bank_filter_fail_der(self, config):
        """Non-bank stock with DER > 1.5 fails."""
        row = _non_bank_fundamentals(der=2.0)
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("FAIL: Der=2.0" in r for r in result["reasons"])

    def test_non_bank_filter_fail_revenue_growth(self, config):
        """Non-bank stock with negative revenue growth fails."""
        row = _non_bank_fundamentals(revenue_growth_yoy=-5.0)
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("Revenue Growth YoY" in r and "below min" in r for r in result["reasons"])

    def test_non_bank_filter_fail_profit_growth(self, config):
        """Non-bank stock with profit growth below -10% fails."""
        row = _non_bank_fundamentals(net_profit_growth_yoy=-15.0)
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("Net Profit Growth YoY" in r and "below min" in r for r in result["reasons"])

    def test_non_bank_filter_missing_ocf(self, config):
        """Non-bank with NaN OCF passes with warning."""
        row = _non_bank_fundamentals(operating_cashflow_positive=float("nan"))
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is True
        assert any("Operating Cashflow Positive" in w for w in result["warnings"])

    def test_non_bank_filter_ocf_not_required(self, config):
        """Config require_operating_cashflow_positive: false skips OCF check."""
        config["fundamental_non_bank"]["require_operating_cashflow_positive"] = False
        row = _non_bank_fundamentals(operating_cashflow_positive=False)
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is True
        # OCF should not appear in reasons (neither PASS nor FAIL)
        ocf_reasons = [r for r in result["reasons"] if "Cashflow" in r or "OCF" in r]
        assert len(ocf_reasons) == 0

    def test_non_bank_filter_missing_roe(self, config):
        """Non-bank stock with NaN ROE fails with missing reason."""
        row = _non_bank_fundamentals(roe=float("nan"))
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("FAIL: Roe is missing" in r for r in result["reasons"])
        assert "roe" in result["missing_fields"]

    def test_non_bank_filter_missing_der(self, config):
        """Non-bank stock with NaN DER fails."""
        row = _non_bank_fundamentals(der=float("nan"))
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("Der is missing" in r for r in result["reasons"])
        assert "der" in result["missing_fields"]

    def test_non_bank_filter_ocf_false_fails(self, config):
        """Non-bank with OCF=False fails when OCF check is required."""
        row = _non_bank_fundamentals(operating_cashflow_positive=False)
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("Operating Cashflow is not positive" in r for r in result["reasons"])

    def test_non_bank_filter_missing_revenue_growth(self, config):
        """Non-bank with NaN revenue_growth_yoy fails."""
        row = _non_bank_fundamentals(revenue_growth_yoy=float("nan"))
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("FAIL: Revenue Growth YoY is missing" in r for r in result["reasons"])
        assert "revenue_growth_yoy" in result["missing_fields"]

    def test_non_bank_filter_missing_profit_growth(self, config):
        """Non-bank with NaN net_profit_growth_yoy fails."""
        row = _non_bank_fundamentals(net_profit_growth_yoy=float("nan"))
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is False
        assert any("FAIL: Net Profit Growth YoY is missing" in r for r in result["reasons"])
        assert "net_profit_growth_yoy" in result["missing_fields"]

    def test_unknown_sector_treated_as_non_bank(self, config):
        """Sector not in bank list is treated as non-bank."""
        row = _non_bank_fundamentals(sector="Mining")
        result = apply_fundamental_filter(row, config)

        assert result["passes"] is True
        # Should have DER check (non-bank) not PBV check (bank)
        assert any("Der" in r for r in result["reasons"])
        assert not any("Pbv" in r for r in result["reasons"])
# ===================================================================
# TASK-008: Technical Hard Filters
# ===================================================================


class TestTechnicalFilter:
    """Tests for ``apply_technical_filter()``."""

    @pytest.fixture
    def config(self):
        return {
            "technical": {
                "max_distance_from_sma20": 0.15,
            }
        }

    def test_technical_filter_pass(self, config):
        """All conditions met → passes."""
        features = _passing_tech_features()
        result = apply_technical_filter(features, config)

        assert result["passes"] is True
        pass_reasons = [r for r in result["reasons"] if r.startswith("PASS:")]
        fail_reasons = [r for r in result["reasons"] if r.startswith("FAIL:")]
        assert len(pass_reasons) == 5
        assert len(fail_reasons) == 0

    def test_technical_filter_fail_below_sma20(self, config):
        """Close < SMA20 → fails."""
        features = _passing_tech_features(close=90.0, sma_short=100.0)
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("Close=90.0 <= SMA20=100.0" in r for r in result["reasons"])

    def test_technical_filter_fail_below_sma50(self, config):
        """Close < SMA50 → fails."""
        features = _passing_tech_features(close=80.0, sma_long=100.0)
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("Close=80.0 <= SMA50=100.0" in r for r in result["reasons"])

    def test_technical_filter_fail_sma_not_rising(self, config):
        """SMA20 not rising → fails."""
        features = _passing_tech_features(sma_short_is_rising=False)
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("FAIL: SMA20 is not rising" in r for r in result["reasons"])

    def test_technical_filter_fail_negative_rs(self, config):
        """RS < 0 → fails."""
        features = _passing_tech_features(relative_strength_13w=-0.05)
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("Relative Strength=-0.05 (below 0)" in r for r in result["reasons"])

    def test_technical_filter_fail_too_far_from_sma(self, config):
        """Distance > 15% → fails."""
        features = _passing_tech_features(distance_from_sma20=0.20)
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("|Distance from SMA20|=0.2000" in r for r in result["reasons"])

    def test_technical_filter_boundary_distance_exactly_15pct(self, config):
        """Distance == 0.15 fails (strict less-than)."""
        features = _passing_tech_features(distance_from_sma20=0.15)
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any(">= 0.1500" in r for r in result["reasons"])

    def test_technical_filter_insufficient_data(self, config):
        """NaN SMA50 → fails with warning."""
        features = _passing_tech_features(sma_long=float("nan"))
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("FAIL: SMA50 data is insufficient" in r for r in result["reasons"])
        assert any("Insufficient SMA50 data" in w for w in result["warnings"])

    def test_technical_filter_insufficient_sma20(self, config):
        """NaN SMA20 → fails with warning."""
        features = _passing_tech_features(sma_short=float("nan"))
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("FAIL: SMA20 data is insufficient" in r for r in result["reasons"])
        assert any("Insufficient SMA20 data" in w for w in result["warnings"])

    def test_technical_filter_insufficient_rs(self, config):
        """NaN RS → fails with warning."""
        features = _passing_tech_features(relative_strength_13w=float("nan"))
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("Relative Strength data is insufficient" in r for r in result["reasons"])
        assert any("Insufficient data for Relative Strength" in w for w in result["warnings"])

    def test_technical_filter_rs_exactly_zero(self, config):
        """RS == 0 should fail (needs > 0)."""
        features = _passing_tech_features(relative_strength_13w=0.0)
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("Relative Strength=0.0 (below 0)" in r for r in result["reasons"])

    def test_technical_filter_custom_max_distance(self, config):
        """Custom max_distance_from_sma20 from config is respected."""
        config["technical"]["max_distance_from_sma20"] = 0.10
        # distance = 0.12 > max 0.10
        features = _passing_tech_features(distance_from_sma20=0.12)
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any(">= 0.1000" in r for r in result["reasons"])

        # distance = 0.08 < max 0.10 → should pass
        features2 = _passing_tech_features(distance_from_sma20=0.08)
        result2 = apply_technical_filter(features2, config)
        assert result2["passes"] is True

    def test_technical_filter_returns_warnings(self, config):
        """Warnings from tech_features propagate through the filter."""
        features = _passing_tech_features(
            warnings=["Some pre-existing warning"]
        )
        result = apply_technical_filter(features, config)
        assert "Some pre-existing warning" in result["warnings"]

    def test_technical_filter_preserves_all_reasons(self, config):
        """All 5 checks produce a reason entry."""
        features = _passing_tech_features()
        result = apply_technical_filter(features, config)
        assert len(result["reasons"]) == 5

    def test_technical_filter_boundary_close_equals_sma20(self, config):
        """Close == SMA20 should fail (needs strict greater-than)."""
        features = _passing_tech_features(close=100.0, sma_short=100.0)
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("Close=100.0 <= SMA20=100.0" in r for r in result["reasons"])

    def test_technical_filter_boundary_close_equals_sma50(self, config):
        """Close == SMA50 should fail (needs strict greater-than)."""
        features = _passing_tech_features(close=90.0, sma_long=90.0)
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("Close=90.0 <= SMA50=90.0" in r for r in result["reasons"])

    def test_technical_filter_nan_close(self, config):
        """NaN close → both SMA checks fail with warnings."""
        features = _passing_tech_features(close=float("nan"))
        result = apply_technical_filter(features, config)

        assert result["passes"] is False
        assert any("FAIL: SMA20 data is insufficient" in r for r in result["reasons"])
        assert any("FAIL: SMA50 data is insufficient" in r for r in result["reasons"])
        assert any("Insufficient SMA20 data" in w for w in result["warnings"])
        assert any("Insufficient SMA50 data" in w for w in result["warnings"])

    def test_technical_filter_empty_features(self, config):
        """Empty tech_features dict → all 5 checks should FAIL."""
        result = apply_technical_filter({}, config)

        assert result["passes"] is False
        fail_reasons = [r for r in result["reasons"] if r.startswith("FAIL:")]
        assert len(fail_reasons) == 5

    def test_technical_filter_missing_max_distance_raises_keyerror(self, config):
        """Missing max_distance_from_sma20 in config raises KeyError."""
        tech_features = _passing_tech_features()
        incomplete_config = dict(config)
        incomplete_config["technical"] = {"sma_short": 20, "sma_long": 50}
        with pytest.raises(KeyError):
            apply_technical_filter(tech_features, incomplete_config)
