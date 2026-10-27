"""Unit tests for Stage 5: Unsupervised Multivariate Anomaly Detection.

Verifies:
- Scikit-learn Isolation Forest execution in reference-trained and self-trained modes
- Detection of known extreme multivariate outliers
- Safe handling of missing values (NaNs) and non-finite floats (+/- inf) via median imputation
- Automatic feature alignment across schema column mismatches
- Configurable contamination parameter effects
- Defensive non-crashing behavior on empty inputs, tiny samples, and datasets with no numeric features
- DataFrame immutability guarantees
- Verification against benchmark fixtures (baseline.csv vs current.csv)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    AnomalyDetectionResult,
    AnomalyRecord,
    detect_anomalies,
)
from src.data_loader import load_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_synthetic_data() -> pd.DataFrame:
    """Fixture providing normal Gaussian baseline data."""
    rng = np.random.default_rng(42)
    n = 250
    return pd.DataFrame({
        "feat_a": rng.normal(50.0, 5.0, size=n),
        "feat_b": rng.normal(100.0, 10.0, size=n),
        "feat_c": rng.normal(20.0, 2.0, size=n),
    })


@pytest.fixture
def data_with_known_outliers(clean_synthetic_data) -> pd.DataFrame:
    """Fixture providing dataset with 3 known extreme multivariate outliers."""
    curr = clean_synthetic_data.copy(deep=True)
    # Row 10: extreme in feat_a
    curr.loc[10, "feat_a"] = 150.0
    # Row 50: extreme in feat_b and feat_c
    curr.loc[50, "feat_b"] = 500.0
    curr.loc[50, "feat_c"] = 80.0
    # Row 100: negative extreme in all features
    curr.loc[100, "feat_a"] = -80.0
    curr.loc[100, "feat_b"] = -200.0
    return curr


# ---------------------------------------------------------------------------
# Anomaly Detection Tests
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    """Tests for detect_anomalies."""

    def test_reference_trained_mode_detects_known_outliers(
        self, clean_synthetic_data, data_with_known_outliers
    ):
        result = detect_anomalies(
            current_df=data_with_known_outliers,
            reference_df=clean_synthetic_data,
            contamination=0.03,
            random_state=42,
        )

        assert result.success is True
        assert result.training_mode == "reference_trained"
        assert set(result.features_used) == {"feat_a", "feat_b", "feat_c"}
        assert result.total_records_analyzed == 250
        assert result.anomalies_detected_count > 0

        # Known outlier rows (10, 50, 100) must appear in top suspicious records
        top_indices = [rec.row_index for rec in result.anomaly_records[:10]]
        assert 10 in top_indices
        assert 50 in top_indices
        assert 100 in top_indices

        # Check records have normalized anomaly score and deviations
        top_rec = result.anomaly_records[0]
        assert 0.0 <= top_rec.anomaly_score <= 1.0
        assert top_rec.raw_decision_score < 0.0
        assert top_rec.is_anomaly is True
        assert "feat_a" in top_rec.feature_deviations

    def test_self_trained_mode_without_reference(self, data_with_known_outliers):
        result = detect_anomalies(
            current_df=data_with_known_outliers,
            reference_df=None,
            contamination=0.05,
            random_state=42,
        )

        assert result.success is True
        assert result.training_mode == "self_trained"
        assert any("self-trained mode" in w for w in result.warnings)
        # Outliers should still be captured
        top_indices = [rec.row_index for rec in result.anomaly_records[:10]]
        assert 50 in top_indices or 100 in top_indices

    def test_handles_missing_values_and_infinities(self, clean_synthetic_data):
        curr = clean_synthetic_data.copy(deep=True)
        # Inject NaNs and +/- inf
        curr.loc[5, "feat_a"] = np.nan
        curr.loc[15, "feat_b"] = np.inf
        curr.loc[25, "feat_c"] = -np.inf

        result = detect_anomalies(
            current_df=curr,
            reference_df=clean_synthetic_data,
            contamination=0.05,
        )

        assert result.success is True
        assert result.total_records_analyzed == 250
        assert result.error_message is None

    def test_column_mismatch_alignment(self):
        ref = pd.DataFrame({
            "common_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "common_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "ref_only": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        })
        curr = pd.DataFrame({
            "common_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "common_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "curr_only": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        })

        result = detect_anomalies(curr, ref, contamination=0.10)
        assert result.success is True
        # Only common columns should be used
        assert result.features_used == ["common_1", "common_2"]
        assert any("curr_only" in w for w in result.warnings)

    def test_empty_dataset_returns_clean_failure(self):
        empty_df = pd.DataFrame()
        result = detect_anomalies(empty_df)
        assert result.success is False
        assert result.training_mode == "failed"
        assert "empty" in result.error_message.lower()

    def test_insufficient_rows_returns_clean_failure(self):
        tiny_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})  # 3 rows < 5
        result = detect_anomalies(tiny_df)
        assert result.success is False
        assert "minimum 5 rows" in result.error_message

    def test_no_numerical_features_returns_clean_failure(self):
        text_df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "city": ["NY", "LA", "SF", "Chicago", "Boston"],
        })
        result = detect_anomalies(text_df)
        assert result.success is False
        assert "No suitable numerical features" in result.error_message

    def test_contamination_parameter_effects(self, clean_synthetic_data):
        res_low = detect_anomalies(clean_synthetic_data, contamination=0.02, random_state=42)
        res_high = detect_anomalies(clean_synthetic_data, contamination=0.10, random_state=42)

        assert res_low.success is True
        assert res_high.success is True
        assert res_high.anomalies_detected_count > res_low.anomalies_detected_count

    def test_invalid_contamination_returns_clean_failure(self, clean_synthetic_data):
        result = detect_anomalies(clean_synthetic_data, contamination=0.85)
        assert result.success is False
        assert "Contamination must be in (0.0, 0.5]" in result.error_message

    def test_immutability_of_input_dataframes(self, clean_synthetic_data, data_with_known_outliers):
        curr_copy = data_with_known_outliers.copy(deep=True)
        ref_copy = clean_synthetic_data.copy(deep=True)

        detect_anomalies(data_with_known_outliers, clean_synthetic_data)

        pd.testing.assert_frame_equal(data_with_known_outliers, curr_copy)
        pd.testing.assert_frame_equal(clean_synthetic_data, ref_copy)

    def test_demo_benchmark_anomalies_detection(self):
        demo_dir = Path("data/demo")
        baseline_path = demo_dir / "baseline.csv"
        current_path = demo_dir / "current.csv"

        assert baseline_path.exists()
        assert current_path.exists()

        baseline_df = load_dataset(baseline_path.read_bytes(), "baseline.csv")
        current_df = load_dataset(current_path.read_bytes(), "current.csv")

        result = detect_anomalies(
            current_df=current_df,
            reference_df=baseline_df,
            contamination=0.03,
            random_state=42,
            top_k=20,
        )

        assert result.success is True
        assert result.training_mode == "reference_trained"
        assert len(result.features_used) >= 4

        # The injected multivariate outlier IDs (e.g. ANOM-00001..ANOM-00008)
        # should be identified at the tail of the current dataset
        anom_row_indices = current_df[current_df["customer_id"].str.startswith("ANOM-")].index.tolist()
        top_flagged_indices = [rec.row_index for rec in result.anomaly_records]

        # Verify that majority of injected ANOM records are ranked in the top suspicious records
        detected_anoms = set(anom_row_indices).intersection(set(top_flagged_indices))
        assert len(detected_anoms) >= 4

        # Disclaimer verification
        assert "ML Outlier Interpretation Notice" in result.disclaimer
        assert "NOT definitive proof of corrupted" in result.disclaimer
