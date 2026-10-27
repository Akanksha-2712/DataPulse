"""Unit tests for Stage 4: Distribution & Schema Drift Monitoring.

Verifies:
- Detection of missing, newly added, and type-modified columns
- Two-sample Kolmogorov-Smirnov test and Wasserstein distance for continuous features
- Pearson Chi-Square test and Jensen-Shannon divergence for categorical features
- Unchanged identical data yields p=1.0 and zero drift
- Graceful handling of edge cases: insufficient samples, constants, high-cardinality IDs
- Verification against generated baseline.csv and current.csv demo datasets
- Requiring reference dataset (error when None)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.drift_detection import (
    ColumnDriftResult,
    DriftAnalysisResult,
    SchemaDiffResult,
    compare_schemas,
    detect_drift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stationary_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixture providing identical reference and current datasets."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "num1": rng.normal(10.0, 2.0, size=200),
        "num2": rng.exponential(5.0, size=200),
        "cat1": rng.choice(["A", "B", "C"], size=200, p=[0.5, 0.3, 0.2]),
    })
    return df, df.copy(deep=True)


@pytest.fixture
def drifted_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixture providing known drifted continuous and categorical distributions."""
    rng_ref = np.random.default_rng(10)
    rng_curr = np.random.default_rng(20)

    # Reference
    ref_df = pd.DataFrame({
        "num_drift": rng_ref.normal(0.0, 1.0, size=300),
        "num_stationary": rng_ref.normal(50.0, 5.0, size=300),
        "cat_drift": rng_ref.choice(["X", "Y"], size=300, p=[0.8, 0.2]),
        "cat_stationary": ["Gold"] * 180 + ["Silver"] * 120,
    })

    # Current: num_drift shifted to mean=3.0, cat_drift inverted to [0.2, 0.8]
    curr_df = pd.DataFrame({
        "num_drift": rng_curr.normal(3.0, 1.0, size=300),
        "num_stationary": rng_ref.normal(50.0, 5.0, size=300),  # Same generator state/distribution
        "cat_drift": rng_curr.choice(["X", "Y"], size=300, p=[0.2, 0.8]),
        "cat_stationary": ["Gold"] * 180 + ["Silver"] * 120,
    })

    return ref_df, curr_df


# ---------------------------------------------------------------------------
# Schema Diff Tests
# ---------------------------------------------------------------------------

class TestSchemaComparison:
    """Tests for compare_schemas and schema drift detection."""

    def test_schema_identical(self, stationary_datasets):
        ref, curr = stationary_datasets
        diff = compare_schemas(ref, curr)
        assert diff.has_schema_changes is False
        assert diff.missing_columns == []
        assert diff.added_columns == []
        assert diff.type_changes == {}
        assert set(diff.common_columns) == {"num1", "num2", "cat1"}

    def test_schema_missing_columns(self):
        ref = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        curr = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        diff = compare_schemas(ref, curr)
        assert diff.has_schema_changes is True
        assert diff.missing_columns == ["c"]
        assert diff.added_columns == []

    def test_schema_added_columns(self):
        ref = pd.DataFrame({"a": [1, 2]})
        curr = pd.DataFrame({"a": [1, 2], "new_col1": [10, 20], "new_col2": [30, 40]})
        diff = compare_schemas(ref, curr)
        assert diff.has_schema_changes is True
        assert diff.missing_columns == []
        assert diff.added_columns == ["new_col1", "new_col2"]

    def test_schema_type_changes(self):
        ref = pd.DataFrame({"id": [1, 2, 3], "val": [1.1, 2.2, 3.3]})
        curr = pd.DataFrame({"id": ["1", "2", "3"], "val": [1.1, 2.2, 3.3]})
        diff = compare_schemas(ref, curr)
        assert diff.has_schema_changes is True
        assert "id" in diff.type_changes
        assert "val" not in diff.type_changes


# ---------------------------------------------------------------------------
# Distribution Drift Tests
# ---------------------------------------------------------------------------

class TestDistributionDrift:
    """Tests for detect_drift function."""

    def test_missing_reference_raises_value_error(self):
        curr = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError) as exc_info:
            detect_drift(None, curr)
        assert "Reference dataset is required" in str(exc_info.value)

    def test_unchanged_identical_data_has_no_drift(self, stationary_datasets):
        ref, curr = stationary_datasets
        result = detect_drift(ref, curr, alpha=0.05)

        assert result.has_reference is True
        assert result.dataset_drift_detected is False
        assert result.drifted_columns_count == 0
        assert result.share_of_drifted_columns == 0.0

        for col, res in result.column_drift.items():
            assert res.drift_detected is False
            assert res.p_value is not None
            assert res.p_value >= 0.05

    def test_numerical_drift_detection(self, drifted_datasets):
        ref, curr = drifted_datasets
        result = detect_drift(ref, curr, alpha=0.05)

        # num_drift should be flagged (p-value < 0.001)
        drift_res = result.column_drift["num_drift"]
        assert drift_res.drift_detected is True
        assert drift_res.p_value < 0.001
        assert drift_res.statistic is not None
        assert drift_res.statistic > 0.5
        assert drift_res.additional_metrics.get("wasserstein_distance") is not None

        # num_stationary should NOT be flagged
        stat_res = result.column_drift["num_stationary"]
        assert stat_res.drift_detected is False
        assert stat_res.p_value >= 0.05

    def test_categorical_drift_detection(self, drifted_datasets):
        ref, curr = drifted_datasets
        result = detect_drift(ref, curr, alpha=0.05)

        # cat_drift should be flagged (Chi-Square p-value < 0.001)
        cat_drift_res = result.column_drift["cat_drift"]
        assert cat_drift_res.drift_detected is True
        assert cat_drift_res.p_value < 0.001
        assert cat_drift_res.statistic is not None
        assert cat_drift_res.additional_metrics.get("jensen_shannon_divergence") is not None

        # cat_stationary should NOT be flagged
        cat_stat_res = result.column_drift["cat_stationary"]
        assert cat_stat_res.drift_detected is False
        assert cat_stat_res.p_value >= 0.05

    def test_skip_insufficient_samples(self):
        ref = pd.DataFrame({"val": [1.0, 2.0, 3.0]})  # N=3 (< 5)
        curr = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = detect_drift(ref, curr)
        res = result.column_drift["val"]
        assert res.skipped is True
        assert "Insufficient sample size" in res.skip_reason
        assert any("Insufficient sample size" in w for w in result.warnings)

    def test_skip_high_cardinality_identifiers(self):
        n = 150
        ref = pd.DataFrame({"user_id": [f"ID_{i}" for i in range(n)]})
        curr = pd.DataFrame({"user_id": [f"ID_{i+1000}" for i in range(n)]})
        result = detect_drift(ref, curr)
        res = result.column_drift["user_id"]
        assert res.skipped is True
        assert "High-cardinality" in res.skip_reason

    def test_constant_value_columns(self):
        ref = pd.DataFrame({"constant_num": [5.0] * 50})
        curr = pd.DataFrame({"constant_num": [5.0] * 50})
        result = detect_drift(ref, curr)
        res = result.column_drift["constant_num"]
        assert res.skipped is False
        assert res.drift_detected is False
        assert res.statistic == 0.0
        assert res.p_value == 1.0

    def test_drift_on_demo_fixtures(self):
        # Load baseline.csv and current.csv
        demo_dir = Path("data/demo")
        baseline_path = demo_dir / "baseline.csv"
        current_path = demo_dir / "current.csv"

        assert baseline_path.exists()
        assert current_path.exists()

        baseline_df = load_dataset(baseline_path.read_bytes(), "baseline.csv")
        current_df = load_dataset(current_path.read_bytes(), "current.csv")

        result = detect_drift(baseline_df, current_df, alpha=0.05)

        # transaction_amount was deliberately shifted mean ~125 -> ~310
        tx_res = result.column_drift["transaction_amount"]
        assert tx_res.drift_detected is True
        assert tx_res.p_value < 0.01

        # age was shifted mean ~38 -> ~49
        age_res = result.column_drift["age"]
        assert age_res.drift_detected is True
        assert age_res.p_value < 0.01

        # category surged in Travel
        cat_res = result.column_drift["category"]
        assert cat_res.drift_detected is True
        assert cat_res.p_value < 0.01

        # customer_id should be skipped as high-cardinality identifier
        assert result.column_drift["customer_id"].skipped is True

        # Check that disclaimer is populated
        assert "Academic & Engineering Notice" in result.disclaimer
        assert "NOT synonymous with data corruption" in result.disclaimer
