"""
Tests for stock_screener.src.pipeline — classification logic
(TASK-009).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_screener.src.pipeline import (
    calculate_data_completeness,
    classify_stock,
)


# ===================================================================
# Helpers
# ===================================================================


def _passing_fundamental_result(**overrides) -> dict:
    defaults = {
        "passes": True,
        "reasons": ["PASS: Roe=15.0%"],
        "warnings": [],
        "missing_fields": [],
    }
    defaults.update(overrides)
    return defaults


def _failing_fundamental_result(**overrides) -> dict:
    defaults = {
        "passes": False,
        "reasons": ["FAIL: Roe=4.0%"],
        "warnings": [],
        "missing_fields": [],
    }
    defaults.update(overrides)
    return defaults


def _passing_technical_result(**overrides) -> dict:
    defaults = {
        "passes": True,
        "reasons": ["PASS: Close > SMA20"],
        "warnings": [],
    }
    defaults.update(overrides)
    return defaults


def _failing_technical_result(**overrides) -> dict:
    defaults = {
        "passes": False,
        "reasons": ["FAIL: Close <= SMA20"],
        "warnings": [],
    }
    defaults.update(overrides)
    return defaults


# ===================================================================
# Tests — classify_stock
# ===================================================================


class TestClassifyStock:
    """Tests for ``classify_stock()``."""

    @pytest.fixture
    def config(self):
        return {
            "classification": {"min_data_completeness": 0.6},
            "sectors": {"bank": ["Bank"]},
        }

    def test_classify_candidate(self, config):
        """Both pass → Candidate."""
        result = classify_stock(
            _passing_fundamental_result(),
            _passing_technical_result(),
            data_completeness=1.0,
            config=config,
        )
        assert result == "Candidate"

    def test_classify_watch(self, config):
        """Fundamental pass, technical fail → Watch."""
        result = classify_stock(
            _passing_fundamental_result(),
            _failing_technical_result(),
            data_completeness=1.0,
            config=config,
        )
        assert result == "Watch"

    def test_classify_speculative(self, config):
        """Fundamental fail, technical pass → Speculative."""
        result = classify_stock(
            _failing_fundamental_result(),
            _passing_technical_result(),
            data_completeness=1.0,
            config=config,
        )
        assert result == "Speculative"

    def test_classify_avoid_both_fail(self, config):
        """Both fail → Avoid."""
        result = classify_stock(
            _failing_fundamental_result(),
            _failing_technical_result(),
            data_completeness=1.0,
            config=config,
        )
        assert result == "Avoid"

    def test_classify_avoid_insufficient_data(self, config):
        """Completeness < 0.6 → Avoid regardless of filters."""
        result = classify_stock(
            _passing_fundamental_result(),
            _passing_technical_result(),
            data_completeness=0.5,
            config=config,
        )
        assert result == "Avoid"

    def test_classify_with_warnings_still_passes(self, config):
        """Filter returns passes=True with warnings → still counts as pass."""
        fund_result = _passing_fundamental_result(
            warnings=["Some optional data missing"]
        )
        tech_result = _passing_technical_result(
            warnings=["Some technical data missing"]
        )
        result = classify_stock(
            fund_result,
            tech_result,
            data_completeness=1.0,
            config=config,
        )
        assert result == "Candidate"


# ===================================================================
# Tests — calculate_data_completeness
# ===================================================================


class TestCalculateDataCompleteness:
    """Tests for ``calculate_data_completeness()``."""

    def test_calculate_data_completeness_all_present(self):
        """All fields present, has price data → 1.0."""
        row = pd.Series({
            "roe": 15.0,
            "der": 1.0,
            "revenue_growth_yoy": 5.0,
            "net_profit_growth_yoy": 5.0,
        })
        score = calculate_data_completeness(
            row,
            required_fields=["roe", "der", "revenue_growth_yoy", "net_profit_growth_yoy"],
            has_price_data=True,
        )
        assert score == pytest.approx(1.0)

    def test_calculate_data_completeness_partial(self):
        """Half fundamental fields missing → 0.75."""
        row = pd.Series({
            "roe": 15.0,
            "der": float("nan"),
            "revenue_growth_yoy": 5.0,
            "net_profit_growth_yoy": float("nan"),
        })
        score = calculate_data_completeness(
            row,
            required_fields=["roe", "der", "revenue_growth_yoy", "net_profit_growth_yoy"],
            has_price_data=True,
        )
        # fundamental_completeness = 2/4 = 0.5
        # price_completeness = 1.0
        # total = 0.5 * 0.5 + 1.0 * 0.5 = 0.75
        assert score == pytest.approx(0.75)

    def test_calculate_data_completeness_no_price_data(self):
        """All fundamentals present but no price → 0.5."""
        row = pd.Series({
            "roe": 15.0,
            "der": 1.0,
            "revenue_growth_yoy": 5.0,
            "net_profit_growth_yoy": 5.0,
        })
        score = calculate_data_completeness(
            row,
            required_fields=["roe", "der", "revenue_growth_yoy", "net_profit_growth_yoy"],
            has_price_data=False,
        )
        # fundamental_completeness = 1.0
        # price_completeness = 0.0
        # total = 1.0 * 0.5 + 0.0 * 0.5 = 0.5
        assert score == pytest.approx(0.5)

    def test_calculate_data_completeness_excludes_optional(self):
        """Only optional fields missing → 1.0."""
        row = pd.Series({
            "roe": 15.0,
            "der": 1.0,
            "revenue_growth_yoy": 5.0,
            "net_profit_growth_yoy": 5.0,
            "operating_cashflow_positive": float("nan"),  # optional
        })
        score = calculate_data_completeness(
            row,
            required_fields=["roe", "der", "revenue_growth_yoy", "net_profit_growth_yoy"],
            has_price_data=True,
        )
        # All required fields present → fundamental_completeness = 1.0
        assert score == pytest.approx(1.0)

    def test_completeness_empty_required_fields(self):
        """Empty required_fields list → fundamental_completeness = 1.0."""
        row = pd.Series({"roe": float("nan")})
        score = calculate_data_completeness(
            row, required_fields=[], has_price_data=True
        )
        # fundamental_completeness = 1.0 (no required fields to miss)
        # price_completeness = 1.0
        # total = 1.0 * 0.5 + 1.0 * 0.5 = 1.0
        assert score == pytest.approx(1.0)
