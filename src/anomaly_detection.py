"""Unsupervised Multivariate Anomaly Detection Engine using Scikit-Learn Isolation Forest.

Detects statistically isolated observations in multivariate continuous feature space:
1. Reference-trained mode: Fits Isolation Forest on reference baseline and scores current production data.
2. Self-trained mode: If no reference dataset is provided, detects internal multivariate anomalies.
3. Feature alignment: Selects valid numerical features common to both reference and current datasets.
4. Robust preprocessing: Imputes missing values with reference median and handles non-finite values safely.
5. Explainability heuristic: Calculates feature deviation contributions for flagged records.

Academic & Engineering Principle:
An ML anomaly is an uncharacteristically isolated data point in feature space. It indicates a
statistically suspicious record warranting human investigation, NOT proof of data corruption.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


@dataclass
class AnomalyRecord:
    """Individual flagged anomaly record with row index, scores, and feature contributions."""
    row_index: int
    is_anomaly: bool                            # True if predicted outlier by Isolation Forest
    anomaly_score: float                        # Calibrated suspicion score in [0.0, 1.0] (higher = more suspicious)
    raw_decision_score: float                   # Raw decision function output (negative = outlier)
    record_data: Dict[str, Any]                 # Raw values of the analyzed features
    feature_deviations: Dict[str, float]        # Relative distance from median in units of IQR

    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly record to dictionary."""
        return asdict(self)


@dataclass
class AnomalyDetectionResult:
    """Complete multivariate anomaly detection analysis report."""
    success: bool
    training_mode: str                          # 'reference_trained', 'self_trained', or 'failed'
    features_used: List[str]
    total_records_analyzed: int
    anomalies_detected_count: int
    anomaly_share: float
    contamination_parameter: Union[float, str]
    anomaly_records: List[AnomalyRecord]        # Sorted by anomaly_score descending
    warnings: List[str]
    disclaimer: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly detection result to dictionary."""
        return {
            "success": self.success,
            "training_mode": self.training_mode,
            "features_used": self.features_used,
            "total_records_analyzed": self.total_records_analyzed,
            "anomalies_detected_count": self.anomalies_detected_count,
            "anomaly_share": self.anomaly_share,
            "contamination_parameter": self.contamination_parameter,
            "anomalies_detected": [rec.to_dict() for rec in self.anomaly_records if rec.is_anomaly],
            "top_records": [rec.to_dict() for rec in self.anomaly_records],
            "warnings": self.warnings,
            "disclaimer": self.disclaimer,
            "error_message": self.error_message,
        }


def detect_anomalies(
    current_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    features: Optional[List[str]] = None,
    contamination: Union[float, str] = 0.05,
    random_state: int = 42,
    n_estimators: int = 100,
    top_k: int = 50,
) -> AnomalyDetectionResult:
    """Execute unsupervised multivariate anomaly detection using Isolation Forest.

    Guarantees:
    - Input DataFrames are never mutated.
    - Operates in reference-trained mode (when reference_df is provided) or self-trained mode.
    - Gracefully handles missing values, infinities, empty datasets, and column mismatches.
    - Labels flagged records as 'Suspicious Records', never as 'Confirmed Errors'.

    Args:
        current_df: Target dataset to score for anomalies.
        reference_df: Optional baseline dataset for training the Isolation Forest.
        features: Optional explicit list of numerical features to evaluate.
        contamination: Proportion of outliers in the dataset, in (0.0, 0.5] or 'auto'.
        random_state: Seed for reproducible tree partitioning.
        n_estimators: Number of trees in the isolation ensemble (default: 100).
        top_k: Maximum number of top suspicious records to return.

    Returns:
        AnomalyDetectionResult: Summary metrics, training mode, and ranked anomaly records.
    """
    disclaimer = (
        "ML Outlier Interpretation Notice: An anomaly identified by Isolation Forest represents "
        "an observation that is structurally isolated in multivariate feature space. It represents "
        "a statistically suspicious record requiring domain review, NOT definitive proof of "
        "corrupted, incorrect, or defective data."
    )

    # 1. Validate Input DataFrames
    if current_df is None or not isinstance(current_df, pd.DataFrame):
        return _build_failed_result("Current dataset must be a valid pandas DataFrame.", disclaimer, contamination)

    if current_df.empty or len(current_df) == 0:
        return _build_failed_result("Current dataset is empty (0 rows).", disclaimer, contamination)

    if len(current_df) < 5:
        return _build_failed_result(
            f"Current dataset has only {len(current_df)} rows; minimum 5 rows required for Isolation Forest.",
            disclaimer,
            contamination,
        )

    # Validate contamination parameter
    if isinstance(contamination, (int, float)):
        if contamination <= 0.0 or contamination > 0.5:
            return _build_failed_result(
                f"Contamination must be in (0.0, 0.5] or 'auto', got {contamination}.",
                disclaimer,
                contamination,
            )
    elif contamination != "auto":
        return _build_failed_result(
            f"Contamination must be a float or 'auto', got '{contamination}'.",
            disclaimer,
            contamination,
        )

    warnings_list: List[str] = []

    # 2. Identify and Align Suitable Numerical Features
    if reference_df is not None:
        if not isinstance(reference_df, pd.DataFrame):
            return _build_failed_result("Reference dataset must be a pandas DataFrame.", disclaimer, contamination)
        if len(reference_df) < 5:
            warnings_list.append(
                f"Reference dataset has only {len(reference_df)} rows. Falling back to self-trained mode."
            )
            training_mode = "self_trained"
            ref_data_to_use = None
        else:
            training_mode = "reference_trained"
            ref_data_to_use = reference_df
    else:
        training_mode = "self_trained"
        ref_data_to_use = None

    selected_features, feature_warnings = _select_and_align_features(
        current_df=current_df,
        reference_df=ref_data_to_use,
        explicit_features=features,
    )
    warnings_list.extend(feature_warnings)

    if not selected_features:
        return _build_failed_result(
            "No suitable numerical features available for anomaly detection. "
            "Features must be continuous numbers with non-zero variance.",
            disclaimer,
            contamination,
        )

    # 3. Extract Feature Matrices
    curr_numeric = current_df[selected_features].replace([np.inf, -np.inf], np.nan)

    if ref_data_to_use is not None:
        ref_numeric = ref_data_to_use[selected_features].replace([np.inf, -np.inf], np.nan)
    else:
        ref_numeric = curr_numeric

    # 4. Safe Median Imputation (Fitted on reference when available to prevent leakage)
    imputer = SimpleImputer(strategy="median")
    try:
        ref_imputed = imputer.fit_transform(ref_numeric)
        curr_imputed = imputer.transform(curr_numeric)
    except Exception as exc:
        return _build_failed_result(f"Feature preprocessing and imputation failed: {exc}", disclaimer, contamination)

    # Compute baseline median and IQR for explainability heuristic
    medians = np.median(ref_imputed, axis=0)
    q75 = np.percentile(ref_imputed, 75, axis=0)
    q25 = np.percentile(ref_imputed, 25, axis=0)
    iqrs = q75 - q25
    # Avoid zero division on constant or tight features
    iqrs = np.where(iqrs == 0.0, 1.0, iqrs)

    # 5. Fit Isolation Forest & Predict
    try:
        iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,
        )

        # Fit on reference (or current if self-trained)
        iso_forest.fit(ref_imputed)

        # Score current production data
        # raw decision_function: negative = outlier, positive = inlier
        raw_decision_scores = iso_forest.decision_function(curr_imputed)
        # raw predictions: -1 = outlier, +1 = inlier
        raw_predictions = iso_forest.predict(curr_imputed)
    except Exception as exc:
        return _build_failed_result(f"Isolation Forest execution failed: {exc}", disclaimer, contamination)

    # 6. Normalize Anomaly Scores to [0.0, 1.0] (Higher = More Suspicious)
    # IsolationForest decision_function values typically lie in [-0.5, 0.5]
    # Points with decision_function < 0 are flagged outliers
    min_dec = float(np.min(raw_decision_scores))
    max_dec = float(np.max(raw_decision_scores))
    score_range = max_dec - min_dec if max_dec > min_dec else 1.0

    # Invert so lower decision score maps to higher suspicion score (1.0 = most anomalous)
    normalized_scores = 1.0 - ((raw_decision_scores - min_dec) / score_range)
    normalized_scores = np.round(np.clip(normalized_scores, 0.0, 1.0), 4)

    # 7. Compile Anomaly Records
    total_records = len(current_df)
    anomaly_records: List[AnomalyRecord] = []
    anomalies_count = 0

    curr_indices = current_df.index.tolist()

    for i in range(total_records):
        is_anom = bool(raw_predictions[i] == -1)
        if is_anom:
            anomalies_count += 1

        # Heuristic feature deviation: |x - median| / IQR
        row_vec = curr_imputed[i]
        deviations = {
            feat: float(round(abs(row_vec[f_idx] - medians[f_idx]) / iqrs[f_idx], 2))
            for f_idx, feat in enumerate(selected_features)
        }

        # Original raw row values for selected features
        record_values = {
            feat: (
                None if pd.isna(current_df.iloc[i][feat])
                else float(current_df.iloc[i][feat])
                if isinstance(current_df.iloc[i][feat], (int, float, np.number))
                else str(current_df.iloc[i][feat])
            )
            for feat in selected_features
        }

        anomaly_records.append(
            AnomalyRecord(
                row_index=curr_indices[i],
                is_anomaly=is_anom,
                anomaly_score=float(normalized_scores[i]),
                raw_decision_score=float(round(raw_decision_scores[i], 4)),
                record_data=record_values,
                feature_deviations=deviations,
            )
        )

    # Sort records by anomaly score descending (most suspicious first)
    anomaly_records.sort(key=lambda r: r.anomaly_score, reverse=True)
    top_records = anomaly_records[:top_k]

    anomaly_share = round((anomalies_count / total_records), 4) if total_records > 0 else 0.0

    if training_mode == "self_trained":
        warnings_list.append(
            "Executed in self-trained mode: No baseline reference dataset was provided. "
            "Isolation Forest was fitted on current data features to detect internal distribution outliers."
        )

    return AnomalyDetectionResult(
        success=True,
        training_mode=training_mode,
        features_used=selected_features,
        total_records_analyzed=total_records,
        anomalies_detected_count=anomalies_count,
        anomaly_share=anomaly_share,
        contamination_parameter=contamination,
        anomaly_records=top_records,
        warnings=warnings_list,
        disclaimer=disclaimer,
        error_message=None,
    )


# ---------------------------------------------------------------------------
# Internal Helper Functions
# ---------------------------------------------------------------------------

def _select_and_align_features(
    current_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    explicit_features: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Determine eligible continuous numerical features present in both datasets."""
    warnings: List[str] = []

    # Get numerical columns in current_df
    curr_numeric_cols = [
        col for col in current_df.columns
        if pd.api.types.is_numeric_dtype(current_df[col]) and not pd.api.types.is_bool_dtype(current_df[col])
    ]

    # Filter by reference if provided
    if reference_df is not None:
        ref_numeric_cols = [
            col for col in reference_df.columns
            if pd.api.types.is_numeric_dtype(reference_df[col]) and not pd.api.types.is_bool_dtype(reference_df[col])
        ]
        common_candidates = [c for c in curr_numeric_cols if c in ref_numeric_cols]
        mismatched_curr = set(curr_numeric_cols) - set(common_candidates)
        if mismatched_curr:
            warnings.append(
                f"Features {sorted(list(mismatched_curr))} present in current dataset but absent in reference; omitted from anomaly model."
            )
    else:
        common_candidates = curr_numeric_cols

    # Filter by user-requested explicit features if specified
    if explicit_features:
        candidates = [c for c in explicit_features if c in common_candidates]
        missing_explicit = set(explicit_features) - set(candidates)
        if missing_explicit:
            warnings.append(
                f"Requested features {sorted(list(missing_explicit))} not available as valid common numerical columns."
            )
    else:
        candidates = common_candidates

    # Filter out identifier columns (e.g., columns where every value is distinct and monotonic)
    final_features: List[str] = []
    for col in candidates:
        s = current_df[col].dropna()
        if len(s) == 0:
            warnings.append(f"Feature '{col}' contains 100% null values; omitted from anomaly model.")
            continue
        # Skip constant columns
        if s.nunique() <= 1:
            warnings.append(f"Feature '{col}' is constant (zero variance); omitted from anomaly model.")
            continue
        # Skip pure sequential integer IDs
        if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and (col.lower().endswith("id") or col.lower() == "id"):
            warnings.append(f"Feature '{col}' resembles an auto-increment unique ID; omitted from anomaly model.")
            continue

        final_features.append(col)

    return final_features, warnings


def _build_failed_result(
    error_msg: str,
    disclaimer: str,
    contamination: Union[float, str],
) -> AnomalyDetectionResult:
    """Construct a clean, non-crashing failed result dataclass."""
    return AnomalyDetectionResult(
        success=False,
        training_mode="failed",
        features_used=[],
        total_records_analyzed=0,
        anomalies_detected_count=0,
        anomaly_share=0.0,
        contamination_parameter=contamination,
        anomaly_records=[],
        warnings=[],
        disclaimer=disclaimer,
        error_message=error_msg,
    )
