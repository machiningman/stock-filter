"""
Tests for stock_screener.src.pipeline — report export
(TASK-011).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from stock_screener.src.pipeline import generate_report


# ===================================================================
# Helpers
# ===================================================================


def _result_dict(**overrides) -> dict:
    """Build a result dict with sensible defaults."""
    defaults = {
        "ticker": "BBCA.JK",
        "company_name": "PT Bank Central Asia Tbk",
        "sector": "Bank",
        "final_score": 75.0,
        "status": "Candidate",
        "fundamental_score": 80.0,
        "earnings_momentum_score": 70.0,
        "technical_score": 75.0,
        "valuation_score": 65.0,
        "relative_strength_score": 60.0,
        "close": 10250.0,
        "weekly_sma20": 10000.0,
        "weekly_sma50": 9500.0,
        "distance_from_sma20": 0.025,
        "relative_strength_13w": 0.05,
        "reasons": ["PASS: Roe=15.0%", "PASS: Pbv=1.5"],
        "warnings": [],
        "missing_data_flags": [],
    }
    defaults.update(overrides)
    return defaults


# ===================================================================
# Tests — generate_report
# ===================================================================


class TestGenerateReport:
    """Tests for ``generate_report()``."""

    def test_generate_report_basic(self, tmp_path):
        """Generates CSV with all required columns including earnings_momentum_score."""
        results = [
            _result_dict(ticker="BBCA.JK"),
            _result_dict(ticker="BBRI.JK"),
        ]

        filepath = generate_report(results, str(tmp_path), "2025-01-17")

        assert os.path.exists(filepath)
        df = pd.read_csv(filepath)

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
            assert col in df.columns, f"Missing column: {col}"

        assert len(df) == 2

    def test_generate_report_sorting(self, tmp_path):
        """Candidate rows before Watch, higher scores first within group."""
        results = [
            _result_dict(ticker="AVOID1", status="Avoid", final_score=10.0),
            _result_dict(ticker="WATCH1", status="Watch", final_score=50.0),
            _result_dict(ticker="CAND1", status="Candidate", final_score=90.0),
            _result_dict(ticker="CAND2", status="Candidate", final_score=80.0),
            _result_dict(ticker="SPEC1", status="Speculative", final_score=30.0),
            _result_dict(ticker="WATCH2", status="Watch", final_score=60.0),
        ]

        filepath = generate_report(results, str(tmp_path), "2025-01-17")
        df = pd.read_csv(filepath)

        # Order: CAND2(80), CAND1(90) → CAND1 first (higher score first)
        # Actually: Candidate(0): CAND1(90), CAND2(80)
        #           Watch(1): WATCH2(60), WATCH1(50)
        #           Speculative(2): SPEC1(30)
        #           Avoid(3): AVOID1(10)
        expected_order = ["CAND1", "CAND2", "WATCH2", "WATCH1", "SPEC1", "AVOID1"]
        assert list(df["ticker"]) == expected_order, (
            f"Expected {expected_order}, got {list(df['ticker'])}"
        )

    def test_generate_report_creates_directory(self, tmp_path):
        """Creates reports/ if not exists."""
        report_dir = os.path.join(str(tmp_path), "reports")
        assert not os.path.exists(report_dir)

        results = [_result_dict()]
        filepath = generate_report(results, report_dir, "2025-01-17")

        assert os.path.isdir(report_dir)
        assert os.path.exists(filepath)

    def test_generate_report_empty_results(self, tmp_path):
        """Empty input list produces valid CSV with headers only."""
        filepath = generate_report([], str(tmp_path), "2025-01-17")
        df = pd.read_csv(filepath)

        assert len(df) == 0
        # Should still have all columns
        assert "ticker" in df.columns
        assert "final_score" in df.columns
        assert "status" in df.columns
        assert "suggested_review_note" in df.columns

    def test_generate_report_suggested_review_note(self, tmp_path):
        """Note uses only existing columns, no undefined references."""
        results = [
            _result_dict(
                ticker="BBCA.JK",
                sector="Bank",
                final_score=75.0,
                status="Candidate",
                warnings=["Missing gross NPL"],
                missing_data_flags=["roe_not_found"],
            ),
        ]

        filepath = generate_report(results, str(tmp_path), "2025-01-17")
        df = pd.read_csv(filepath)

        note = df.iloc[0]["suggested_review_note"]
        expected = (
            "Review BBCA.JK: Bank sector, score 75/100 (Candidate). "
            "1 warning(s). 1 missing field(s)."
        )
        assert note == expected, f"Got: {note}"

    def test_generate_report_no_warnings(self, tmp_path):
        """Empty warnings/missing fields produce 0 counts in note."""
        results = [
            _result_dict(
                ticker="BBRI.JK",
                sector="Bank",
                final_score=80.0,
                status="Candidate",
                warnings=[],
                missing_data_flags=[],
            ),
        ]

        filepath = generate_report(results, str(tmp_path), "2025-01-17")
        df = pd.read_csv(filepath)
        note = df.iloc[0]["suggested_review_note"]

        assert "0 warning(s)" in note
        assert "0 missing field(s)" in note

    def test_generate_report_with_fundamental_score_column(self, tmp_path):
        """earnings_momentum_score column is populated correctly."""
        results = [
            _result_dict(
                ticker="TEST",
                fundamental_score=85.0,
                earnings_momentum_score=90.0,
            ),
        ]
        filepath = generate_report(results, str(tmp_path), "2025-01-17")
        df = pd.read_csv(filepath)

        assert df.iloc[0]["fundamental_score"] == pytest.approx(85.0)
        assert df.iloc[0]["earnings_momentum_score"] == pytest.approx(90.0)

    def test_generate_report_nan_final_score_review_note(self, tmp_path):
        """NaN final_score should produce 'score 0/100' not 'score nan/100'."""
        results = [_result_dict(
            ticker="TEST", company_name="Test Corp", sector="Technology",
            final_score=float("nan"), status="Candidate",
            fundamental_score=50.0, earnings_momentum_score=60.0,
            technical_score=70.0, valuation_score=40.0, relative_strength_score=30.0,
            close=100.0, weekly_sma20=95.0, weekly_sma50=90.0,
            distance_from_sma20=0.05, relative_strength_13w=0.03,
            reasons=["Good momentum"], warnings=[], missing_data_flags=[]
        )]
        path = generate_report(results, str(tmp_path), "2025-01-17")
        df = pd.read_csv(path)
        assert "score 0/100" in df["suggested_review_note"].iloc[0]
        assert "nan" not in df["suggested_review_note"].iloc[0].lower()
