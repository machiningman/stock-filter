"""
Tests for stock_screener.src.pipeline — scoring module
(TASK-010).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from stock_screener.src.pipeline import (
    calculate_earnings_momentum_score,
    calculate_final_score,
    calculate_fundamental_score,
    calculate_relative_strength_score,
    calculate_technical_score,
    calculate_valuation_score,
    normalize_score,
)


# ===================================================================
# Tests — normalize_score
# ===================================================================


class TestNormalizeScore:
    """Tests for ``normalize_score()``."""

    def test_normalize_score_basic(self):
        """Value at target → 100, value at min → 0."""
        assert normalize_score(25.0, min_val=0, target_val=25) == pytest.approx(100.0)
        assert normalize_score(0.0, min_val=0, target_val=25) == pytest.approx(0.0)

    def test_normalize_score_clipped(self):
        """Value beyond target → 100, value below min → 0."""
        assert normalize_score(50.0, min_val=0, target_val=25) == pytest.approx(100.0)
        assert normalize_score(-10.0, min_val=0, target_val=25) == pytest.approx(0.0)

    def test_normalize_score_inverted(self):
        """Lower value → higher score (e.g., PER=10 → 100, PER=30 → 0)."""
        assert normalize_score(10.0, min_val=30, target_val=10, inverted=True) == pytest.approx(100.0)
        assert normalize_score(30.0, min_val=30, target_val=10, inverted=True) == pytest.approx(0.0)

    def test_normalize_score_inverted_midpoint(self):
        """PER=20 with min=30, target=10 → 50."""
        score = normalize_score(20.0, min_val=30, target_val=10, inverted=True)
        assert score == pytest.approx(50.0)

    def test_normalize_score_nan(self):
        """NaN → 0."""
        score = normalize_score(float("nan"), min_val=0, target_val=25)
        assert score == pytest.approx(0.0)

    def test_normalize_score_min_equals_target(self, caplog):
        """min==target → 0 with warning."""
        caplog.set_level(logging.WARNING)
        score = normalize_score(10.0, min_val=25, target_val=25)
        assert score == pytest.approx(0.0)
        assert any("degenerate" in msg.lower() for msg in caplog.messages)

    def test_normalize_score_negative_distance(self):
        """distance_from_sma20=-0.05 with inverted min=0.15, target=0.02 → clips to 100."""
        score = normalize_score(-0.05, min_val=0.15, target_val=0.02, inverted=True)
        assert score == pytest.approx(100.0)


# ===================================================================
# Tests — calculate_fundamental_score
# ===================================================================


class TestFundamentalScore:
    """Tests for ``calculate_fundamental_score()``."""

    @pytest.fixture
    def config(self):
        return {
            "scoring": {
                "normalization": {
                    "roe": {"min": 0, "target": 25},
                }
            }
        }

    def test_fundamental_score_roe(self, config):
        """ROE=25 → 100, ROE=0 → 0, ROE=12.5 → 50."""
        row_25 = pd.Series({"roe": 25.0})
        row_0 = pd.Series({"roe": 0.0})
        row_12_5 = pd.Series({"roe": 12.5})

        assert calculate_fundamental_score(row_25, config) == pytest.approx(100.0)
        assert calculate_fundamental_score(row_0, config) == pytest.approx(0.0)
        assert calculate_fundamental_score(row_12_5, config) == pytest.approx(50.0)

    def test_fundamental_score_nan(self, config):
        """NaN ROE → 0."""
        row = pd.Series({"roe": float("nan")})
        assert calculate_fundamental_score(row, config) == pytest.approx(0.0)


# ===================================================================
# Tests — calculate_earnings_momentum_score
# ===================================================================


class TestEarningsMomentumScore:
    """Tests for ``calculate_earnings_momentum_score()``."""

    @pytest.fixture
    def config(self):
        return {
            "scoring": {
                "normalization": {
                    "profit_growth_yoy": {"min": -20, "target": 30},
                }
            }
        }

    def test_earnings_momentum_score(self, config):
        """Growth=30 → 100, growth=-20 → 0."""
        row_30 = pd.Series({"net_profit_growth_yoy": 30.0})
        row_neg20 = pd.Series({"net_profit_growth_yoy": -20.0})

        assert calculate_earnings_momentum_score(row_30, config) == pytest.approx(100.0)
        assert calculate_earnings_momentum_score(row_neg20, config) == pytest.approx(0.0)

    def test_earnings_momentum_midpoint(self, config):
        """Growth=5 → (5 - (-20)) / (30 - (-20)) * 100 = 25/50*100 = 50."""
        row = pd.Series({"net_profit_growth_yoy": 5.0})
        assert calculate_earnings_momentum_score(row, config) == pytest.approx(50.0)


# ===================================================================
# Tests — calculate_valuation_score
# ===================================================================


class TestValuationScore:
    """Tests for ``calculate_valuation_score()``."""

    @pytest.fixture
    def config(self):
        return {
            "scoring": {
                "normalization": {
                    "pbv": {"min": 3.0, "target": 1.0, "inverted": True},
                    "per": {"min": 30, "target": 10, "inverted": True},
                }
            },
            "sectors": {"bank": ["Bank", "Perbankan"]},
        }

    def test_valuation_score_bank(self, config):
        """Bank uses PBV: PBV=1.0 → 100, PBV=3.0 → 0."""
        row_pbv_1 = pd.Series({"pbv": 1.0, "per": float("nan"), "sector": "Bank"})
        row_pbv_3 = pd.Series({"pbv": 3.0, "per": float("nan"), "sector": "Bank"})

        assert calculate_valuation_score(row_pbv_1, "Bank", config) == pytest.approx(100.0)
        assert calculate_valuation_score(row_pbv_3, "Bank", config) == pytest.approx(0.0)

    def test_valuation_score_non_bank(self, config):
        """Non-bank uses PER: PER=10 → 100, PER=30 → 0."""
        row_per_10 = pd.Series({"per": 10.0, "pbv": float("nan"), "sector": "Technology"})
        row_per_30 = pd.Series({"per": 30.0, "pbv": float("nan"), "sector": "Technology"})

        assert calculate_valuation_score(row_per_10, "Technology", config) == pytest.approx(100.0)
        assert calculate_valuation_score(row_per_30, "Technology", config) == pytest.approx(0.0)

    def test_valuation_score_bank_pbv_midpoint(self, config):
        """PBV=2.0 → (3.0-2.0)/(3.0-1.0)*100 = 50."""
        row = pd.Series({"pbv": 2.0, "sector": "Bank"})
        assert calculate_valuation_score(row, "Bank", config) == pytest.approx(50.0)

    def test_valuation_score_non_bank_per_midpoint(self, config):
        """PER=20 → (30-20)/(30-10)*100 = 50."""
        row = pd.Series({"per": 20.0, "sector": "Technology"})
        assert calculate_valuation_score(row, "Technology", config) == pytest.approx(50.0)

    def test_valuation_score_bank_nan_pbv(self, config):
        """NaN PBV for bank → 0.0."""
        row = pd.Series({"pbv": float("nan"), "sector": "Bank"})
        assert calculate_valuation_score(row, "Bank", config) == pytest.approx(0.0)

    def test_valuation_score_non_bank_nan_per(self, config):
        """NaN PER for non-bank → 0.0."""
        row = pd.Series({"per": float("nan"), "sector": "Technology"})
        assert calculate_valuation_score(row, "Technology", config) == pytest.approx(0.0)

    def test_valuation_score_nan_sector(self, config):
        """NaN sector → 0.0 (guard against crash in is_bank_sector)."""
        row = pd.Series({"pbv": 2.0, "per": 15.0, "sector": float("nan")})
        assert calculate_valuation_score(row, float("nan"), config) == pytest.approx(0.0)


# ===================================================================
# Tests — calculate_technical_score
# ===================================================================


class TestTechnicalScore:
    """Tests for ``calculate_technical_score()``."""

    @pytest.fixture
    def config(self):
        return {
            "scoring": {
                "normalization": {
                    "distance_from_sma20": {"min": 0.15, "target": 0.02, "inverted": True},
                }
            }
        }

    def test_technical_score(self, config):
        """Distance=0.02 → 100, distance=0.15 → 0."""
        features_02 = {"distance_from_sma20": 0.02}
        features_15 = {"distance_from_sma20": 0.15}

        assert calculate_technical_score(features_02, config) == pytest.approx(100.0)
        assert calculate_technical_score(features_15, config) == pytest.approx(0.0)

    def test_technical_score_midpoint(self, config):
        """Distance=0.085 → (0.15-0.085)/(0.15-0.02)*100 = 50."""
        features = {"distance_from_sma20": 0.085}
        assert calculate_technical_score(features, config) == pytest.approx(50.0)

    def test_technical_score_nan(self, config):
        """NaN distance → 0."""
        features = {"distance_from_sma20": float("nan")}
        assert calculate_technical_score(features, config) == pytest.approx(0.0)


# ===================================================================
# Tests — calculate_relative_strength_score
# ===================================================================


class TestRelativeStrengthScore:
    """Tests for ``calculate_relative_strength_score()``."""

    @pytest.fixture
    def config(self):
        return {
            "scoring": {
                "normalization": {
                    "relative_strength_13w": {"min": -0.10, "target": 0.10},
                }
            }
        }

    def test_relative_strength_score(self, config):
        """RS=0.10 → 100, RS=-0.10 → 0."""
        assert calculate_relative_strength_score(0.10, config) == pytest.approx(100.0)
        assert calculate_relative_strength_score(-0.10, config) == pytest.approx(0.0)

    def test_relative_strength_midpoint(self, config):
        """RS=0.0 → (0 - (-0.10)) / (0.10 - (-0.10)) * 100 = 50."""
        assert calculate_relative_strength_score(0.0, config) == pytest.approx(50.0)

    def test_relative_strength_nan(self, config):
        """NaN RS → 0."""
        assert calculate_relative_strength_score(float("nan"), config) == pytest.approx(0.0)


# ===================================================================
# Tests — calculate_final_score
# ===================================================================


class TestFinalScore:
    """Tests for ``calculate_final_score()``."""

    def test_final_score_weighted(self):
        """Verify weighted sum calculation."""
        sub_scores = {
            "fundamental_quality": 80.0,
            "earnings_momentum": 60.0,
            "valuation": 40.0,
            "technical_trend": 70.0,
            "relative_strength": 50.0,
        }
        weights = {
            "fundamental_quality": 0.35,
            "earnings_momentum": 0.20,
            "valuation": 0.15,
            "technical_trend": 0.20,
            "relative_strength": 0.10,
        }
        # 80*0.35 + 60*0.20 + 40*0.15 + 70*0.20 + 50*0.10
        expected = 28 + 12 + 6 + 14 + 5  # = 65
        assert calculate_final_score(sub_scores, weights) == pytest.approx(expected)

    def test_final_score_bounds(self):
        """All zeros → 0, all 100s → 100."""
        weights = {
            "a": 0.5,
            "b": 0.5,
        }
        assert calculate_final_score({"a": 0.0, "b": 0.0}, weights) == pytest.approx(0.0)
        assert calculate_final_score({"a": 100.0, "b": 100.0}, weights) == pytest.approx(100.0)

    def test_final_score_partial_weights(self):
        """Sub-score missing a weight key → treated as 0."""
        sub_scores = {"a": 100.0}
        weights = {"a": 0.5, "b": 0.5}
        # 100 * 0.5 + 0 * 0.5 = 50
        assert calculate_final_score(sub_scores, weights) == pytest.approx(50.0)

    def test_final_score_clamping(self):
        """Values outside [0, 100] get clamped."""
        sub_scores = {"a": 200.0, "b": -50.0}
        weights = {"a": 0.5, "b": 0.5}
        # 200*0.5 + (-50)*0.5 = 100 - 25 = 75 (within [0, 100])
        assert calculate_final_score(sub_scores, weights) == pytest.approx(75.0)

        # Clamp upper bound
        sub_scores2 = {"a": 200.0, "b": 200.0}
        assert calculate_final_score(sub_scores2, weights) == pytest.approx(100.0)

        # Clamp lower bound
        sub_scores3 = {"a": -50.0, "b": -50.0}
        assert calculate_final_score(sub_scores3, weights) == pytest.approx(0.0)
