"""Unit tests for Stage 3: Profiling, Pandera Validation, and Quality Scoring.

Verifies:
- Complete profiling with nulls, empty strings, infinities, and duplicates
- Pandera schema validation against JSON configuration rules
- Handling of required, range, categorical, and uniqueness constraints
- Transparent quality scoring with mathematical weight normalization
- Graceful omission of validity dimension when rules are unconfigured
- Immutability of raw input DataFrames across all operations
- Verification against deterministic demo benchmark datasets
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.profiling import DatasetProfile, profile_dataset
from src.quality_score import calculate_data_quality_score
from src.validation import (
    ValidationIssue,
    ValidationResult,
    build_pandera_schema,
    load_validation_rules,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_sample_df() -> pd.DataFrame:
    """Fixture providing a predictable 10-row, 4-column DataFrame."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age": [25, 30, 45, 22, 60, 35, 28, 42, 50, 31],
        "income": [50000.0, 60000.0, 75000.0, 48000.0, 92000.0, 58000.0, 62000.0, 71000.0, 83000.0, 54000.0],
        "category": ["Retail", "Travel", "Retail", "Groceries", "Travel", "Retail", "Groceries", "Travel", "Retail", "Groceries"],
    })


@pytest.fixture
def sample_rules() -> dict:
    """Fixture providing explicit validation rules."""
    return {
        "id": {"type": "int", "required": True, "unique": True, "nullable": False},
        "age": {"type": "int", "required": True, "nullable": False, "min_value": 18, "max_value": 100},
        "income": {"type": "float", "required": True, "nullable": False, "min_value": 0.0, "max_value": 500000.0},
        "category": {"type": "string", "required": True, "nullable": False, "allowed_values": ["Retail", "Travel", "Groceries"]},
    }


# ---------------------------------------------------------------------------
# Profiling Tests
# ---------------------------------------------------------------------------

class TestProfiling:
    """Tests for src/profiling.py."""

    def test_profile_empty_dataframe(self):
        empty_df = pd.DataFrame()
        prof = profile_dataset(empty_df)
        assert prof.row_count == 0
        assert prof.column_count == 0
        assert prof.total_cells == 0
        assert prof.duplicate_rows == 0

    def test_profile_dimensions_and_counts(self, synthetic_sample_df):
        prof = profile_dataset(synthetic_sample_df)
        assert prof.row_count == 10
        assert prof.column_count == 4
        assert prof.total_cells == 40
        assert prof.missing_cells == 0
        assert prof.missing_cells_pct == 0.0
        assert prof.duplicate_rows == 0
        assert set(prof.numeric_columns) == {"id", "age", "income"}
        assert prof.categorical_columns == ["category"]

    def test_profile_detects_nulls_and_empty_strings(self):
        df = pd.DataFrame({
            "num": [1.0, np.nan, 3.0, 4.0],
            "text": ["hello", "", "   ", "world"],
        })
        prof = profile_dataset(df)
        assert prof.columns["num"].missing_count == 1
        assert prof.columns["num"].missing_pct == 25.0
        # Empty string count should detect "" and "   "
        assert prof.columns["text"].empty_string_count == 2
        assert prof.empty_strings_total == 2

    def test_profile_detects_infinities(self):
        df = pd.DataFrame({
            "val": [10.0, np.inf, 20.0, -np.inf, 30.0],
        })
        prof = profile_dataset(df)
        assert prof.columns["val"].infinite_count == 2
        assert prof.infinite_values_total == 2
        # Finite distribution stats should compute over the 3 finite numbers
        stats = prof.columns["val"].stats
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert stats["mean"] == 20.0

    def test_profile_detects_duplicate_rows(self):
        df = pd.DataFrame({
            "a": [1, 2, 1, 2, 3],
            "b": ["x", "y", "x", "y", "z"],
        })
        prof = profile_dataset(df)
        assert prof.duplicate_rows == 2
        assert prof.duplicate_rows_pct == 40.0

    def test_profile_immutability(self, synthetic_sample_df):
        original_shape = synthetic_sample_df.shape
        original_first_val = synthetic_sample_df.iloc[0, 0]
        profile_dataset(synthetic_sample_df)
        assert synthetic_sample_df.shape == original_shape
        assert synthetic_sample_df.iloc[0, 0] == original_first_val


# ---------------------------------------------------------------------------
# Validation Tests (Pandera)
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for src/validation.py."""

    def test_validation_passes_on_clean_data(self, synthetic_sample_df, sample_rules):
        result = validate_dataset(synthetic_sample_df, rules=sample_rules)
        assert result.is_valid is True
        assert result.total_issues == 0
        assert result.rules_configured is True
        assert result.failing_rows_count == 0

    def test_validation_missing_required_column(self, sample_rules):
        df = pd.DataFrame({"id": [1, 2], "age": [25, 30]})  # Missing 'income' and 'category'
        result = validate_dataset(df, rules=sample_rules)
        assert result.is_valid is False
        assert "income" in result.missing_columns
        assert "category" in result.missing_columns
        critical_issues = [i for i in result.issues if i.severity == "critical"]
        assert len(critical_issues) == 2

    def test_validation_min_max_limits(self, sample_rules):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "age": [15, 45, 120],  # 15 < 18 (min violation), 120 > 100 (max violation)
            "income": [50000.0, -100.0, 60000.0],  # -100 < 0 (min violation)
            "category": ["Retail", "Retail", "Retail"],
        })
        result = validate_dataset(df, rules=sample_rules)
        assert result.is_valid is False
        age_issues = result.issues_by_column.get("age", [])
        assert len(age_issues) == 2
        issue_types = {i.issue_type for i in age_issues}
        assert "min_value_violation" in issue_types
        assert "max_value_violation" in issue_types

        income_issues = result.issues_by_column.get("income", [])
        assert len(income_issues) == 1
        assert income_issues[0].issue_type == "min_value_violation"

    def test_validation_invalid_categorical_values(self, sample_rules):
        df = pd.DataFrame({
            "id": [1, 2],
            "age": [30, 40],
            "income": [50000.0, 60000.0],
            "category": ["Retail", "InvalidCasino"],
        })
        result = validate_dataset(df, rules=sample_rules)
        assert result.is_valid is False
        cat_issues = result.issues_by_column.get("category", [])
        assert len(cat_issues) == 1
        assert cat_issues[0].issue_type == "invalid_category"

    def test_validation_non_nullable_violation(self, sample_rules):
        df = pd.DataFrame({
            "id": [1, 2],
            "age": [25, np.nan],  # NaN in non-nullable age
            "income": [50000.0, 60000.0],
            "category": ["Retail", "Travel"],
        })
        result = validate_dataset(df, rules=sample_rules)
        assert result.is_valid is False
        age_issues = result.issues_by_column.get("age", [])
        assert any(i.issue_type == "null_value" for i in age_issues)

    def test_validation_unconfigured_rules_returns_safe_result(self, synthetic_sample_df):
        # Empty rules dict
        result = validate_dataset(synthetic_sample_df, rules={})
        assert result.is_valid is True
        assert result.rules_configured is False
        assert result.total_issues == 0

    def test_validation_immutability(self, synthetic_sample_df, sample_rules):
        original_vals = synthetic_sample_df["income"].copy()
        validate_dataset(synthetic_sample_df, rules=sample_rules)
        pd.testing.assert_series_equal(synthetic_sample_df["income"], original_vals)


# ---------------------------------------------------------------------------
# Quality Scoring Tests
# ---------------------------------------------------------------------------

class TestQualityScoring:
    """Tests for src/quality_score.py."""

    def test_quality_score_perfect_dataset(self, synthetic_sample_df, sample_rules):
        val_res = validate_dataset(synthetic_sample_df, rules=sample_rules)
        scorecard = calculate_data_quality_score(synthetic_sample_df, validation_result=val_res)
        assert scorecard.composite_score == 100.0
        assert scorecard.grade == "Excellent"
        assert scorecard.dimensions["completeness"].score == 100.0
        assert scorecard.dimensions["uniqueness"].score == 100.0
        assert scorecard.dimensions["validity"].score == 100.0

    def test_known_mathematical_weights_and_normalization(self):
        # Create dataset with exactly:
        # 100 rows, 4 columns = 400 total cells.
        # 40 missing values -> Completeness = (1 - 40/400) * 100 = 90.0%
        # 10 duplicate rows out of 100 -> Uniqueness = (1 - 10/100) * 100 = 90.0%
        # Rules: 4 columns evaluated. 400 evaluated cells. 20 rule violations -> Validity = (1 - 20/400) * 100 = 95.0%
        # Weights: 40, 25, 35 (sum=100)
        # Expected DQS = (90 * 0.40) + (90 * 0.25) + (95 * 0.35) = 36.0 + 22.5 + 33.25 = 91.75
        rows = 100
        cols = ["a", "b", "c", "d"]
        data = {col: [float(i) for i in range(rows)] for col in cols}
        df = pd.DataFrame(data)

        # Inject 40 NaNs distributed across columns so rows do not become identical
        df.loc[0:9, "a"] = np.nan
        df.loc[10:19, "b"] = np.nan
        df.loc[20:29, "c"] = np.nan
        df.loc[30:39, "d"] = np.nan

        # Inject 10 duplicate rows by repeating row 50 into rows 90..99
        for i in range(10):
            df.iloc[90 + i] = df.iloc[50]

        # Fake validation result with 20 issues across 4 evaluated columns
        fake_issues = [
            ValidationIssue(
                issue_type="min_value_violation",
                column="a",
                row_reference=i,
                severity="error",
                description="Test violation",
            )
            for i in range(20)
        ]
        val_res = ValidationResult(
            is_valid=False,
            rules_configured=True,
            total_issues=20,
            critical_count=0,
            error_count=20,
            warning_count=0,
            evaluated_columns=cols,
            missing_columns=[],
            issues=fake_issues,
            failing_rows_count=20,
            total_rows=rows,
        )

        scorecard = calculate_data_quality_score(df, validation_result=val_res)
        assert scorecard.dimensions["completeness"].score == 90.0
        assert scorecard.dimensions["uniqueness"].score == 90.0
        assert scorecard.dimensions["validity"].score == 95.0
        assert math.isclose(scorecard.composite_score, 91.75, abs_tol=0.01)

    def test_validity_omitted_when_no_rules_configured(self, synthetic_sample_df):
        # When validation_result is None or has rules_configured=False
        scorecard = calculate_data_quality_score(synthetic_sample_df, validation_result=None)

        val_dim = scorecard.dimensions["validity"]
        assert val_dim.is_applicable is False
        assert val_dim.score is None
        assert val_dim.normalized_weight == 0.0

        # Weights completeness=40, uniqueness=25 -> Total applicable=65
        # Normalized completeness: 40/65 = 61.54%
        # Normalized uniqueness: 25/65 = 38.46%
        assert math.isclose(scorecard.normalized_weights["completeness"], 61.54, abs_tol=0.1)
        assert math.isclose(scorecard.normalized_weights["uniqueness"], 38.46, abs_tol=0.1)
        # On perfect data, DQS is still 100.0
        assert scorecard.composite_score == 100.0

    def test_disclaimer_and_limitations_present(self, synthetic_sample_df):
        scorecard = calculate_data_quality_score(synthetic_sample_df)
        assert "Project-Defined Indicator Notice" in scorecard.disclaimer
        assert "NOT an official ISO standard" in scorecard.disclaimer
        assert len(scorecard.limitations) >= 3
        assert "Dimensional Independence Assumption" in scorecard.limitations[0]

    def test_demo_baseline_vs_current_quality_scores(self):
        # Load the generated demo datasets
        baseline_path = Path("data/demo/baseline.csv")
        current_path = Path("data/demo/current.csv")
        rules_path = Path("config/validation_rules.json")

        assert baseline_path.exists()
        assert current_path.exists()

        baseline_df = load_dataset(baseline_path.read_bytes(), "baseline.csv")
        current_df = load_dataset(current_path.read_bytes(), "current.csv")

        val_rules = load_validation_rules(rules_path)

        # Baseline scoring
        base_val = validate_dataset(baseline_df, rules=val_rules)
        base_scorecard = calculate_data_quality_score(baseline_df, validation_result=base_val)
        assert base_val.is_valid is True
        assert base_scorecard.composite_score == 100.0
        assert base_scorecard.grade == "Excellent"

        # Current scoring (with injected defects)
        curr_val = validate_dataset(current_df, rules=val_rules)
        curr_scorecard = calculate_data_quality_score(current_df, validation_result=curr_val)
        assert curr_val.is_valid is False
        assert curr_val.total_issues > 0
        # Quality score should reflect the missing values, duplicates, and rule violations
        assert curr_scorecard.composite_score < 100.0
        assert curr_scorecard.dimensions["completeness"].score < 100.0
        assert curr_scorecard.dimensions["uniqueness"].score < 100.0
        assert curr_scorecard.dimensions["validity"].score < 100.0
