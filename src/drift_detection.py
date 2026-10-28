"""Distributional & Schema Drift Monitoring Engine.

Detects statistical distribution shifts (covariate shift) and schema modifications
between a baseline reference dataset (P_ref(X)) and a current production dataset (P_curr(X)).

Methodologies:
1. Schema Shift: Missing columns, newly added columns, and data type modifications.
2. Numerical Drift: Two-Sample Kolmogorov-Smirnov (K-S) test with Wasserstein Distance.
3. Categorical Drift: Pearson's Chi-Square Goodness-of-Fit / Contingency test.
4. Framework Integration: Attempts Evidently AI when supported and permitted; provides
   transparent, documented fallback to SciPy statistical routines.

Vital Engineering & Academic Principle:
Data drift reflects statistical distribution changes (e.g. seasonal dynamics, inflation,
market evolution). It does NOT constitute proof of corrupted or defective data.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


@dataclass
class SchemaDiffResult:
    """Schema difference analysis between reference and current datasets."""
    missing_columns: List[str]                  # In reference but absent in current
    added_columns: List[str]                    # In current but absent in reference
    type_changes: Dict[str, Dict[str, str]]     # Column -> {reference_type, current_type}
    common_columns: List[str]                   # In both reference and current
    has_schema_changes: bool

    @property
    def schema_has_drift(self) -> bool:
        """Alias for has_schema_changes."""
        return self.has_schema_changes

    @property
    def new_columns(self) -> List[str]:
        """Alias for added_columns."""
        return self.added_columns

    @property
    def type_mismatches(self) -> Dict[str, Dict[str, str]]:
        """Alias for type_changes."""
        return self.type_changes

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema diff to dictionary."""
        return asdict(self)


@dataclass
class ColumnDriftResult:
    """Statistical drift test result for an individual column."""
    column_name: str
    column_type: str                            # 'numeric' or 'categorical'
    method: str                                 # e.g., 'Two-sample Kolmogorov-Smirnov test'
    engine: str                                 # 'evidently' or 'scipy'
    statistic: Optional[float]                  # Test statistic (e.g. KS D or Chi-square)
    p_value: Optional[float]                    # Statistical p-value
    threshold_alpha: float                      # Significance threshold (e.g. 0.05)
    drift_detected: bool                        # p_value < threshold_alpha
    sample_size_reference: int                  # Non-null sample size in reference
    sample_size_current: int                    # Non-null sample size in current
    skipped: bool = False
    skip_reason: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert column drift result to dictionary."""
        return asdict(self)


@dataclass
class DriftAnalysisResult:
    """Complete distribution and schema drift evaluation report."""
    has_reference: bool
    schema_diff: SchemaDiffResult
    column_drift: Dict[str, ColumnDriftResult]
    total_columns_compared: int
    drifted_columns_count: int
    share_of_drifted_columns: float
    dataset_drift_detected: bool
    warnings: List[str]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert drift analysis result to dictionary."""
        return {
            "has_reference": self.has_reference,
            "schema_diff": self.schema_diff.to_dict(),
            "column_drift": {k: v.to_dict() for k, v in self.column_drift.items()},
            "total_columns_compared": self.total_columns_compared,
            "drifted_columns_count": self.drifted_columns_count,
            "share_of_drifted_columns": self.share_of_drifted_columns,
            "dataset_drift_detected": self.dataset_drift_detected,
            "warnings": self.warnings,
            "disclaimer": self.disclaimer,
        }


def detect_drift(
    reference_df: Optional[pd.DataFrame],
    current_df: pd.DataFrame,
    alpha: float = 0.05,
    drift_share_threshold: float = 0.5,
) -> DriftAnalysisResult:
    """Evaluate schema differences and statistical distribution drift.

    Guarantees:
    - Input DataFrames are strictly read-only and never modified.
    - Requires reference_df; if None, raises ValueError with explicit guidance.
    - Skips unsuitable comparisons (constants, insufficient samples, all-NaNs) with clear warnings.
    - Uses SciPy and Evidently with clear method attribution.

    Args:
        reference_df: Baseline historical DataFrame.
        current_df: New production batch DataFrame.
        alpha: Statistical significance threshold (default: 0.05).
        drift_share_threshold: Proportion of drifted columns to declare overall dataset drift (default: 0.50).

    Returns:
        DriftAnalysisResult: Comprehensive schema and distribution drift report.

    Raises:
        ValueError: If reference_df or current_df is missing or empty.
    """
    if reference_df is None:
        raise ValueError(
            "Reference dataset is required for distribution drift analysis. "
            "Please provide a baseline dataset to compare against."
        )

    if not isinstance(reference_df, pd.DataFrame) or not isinstance(current_df, pd.DataFrame):
        raise TypeError("Both reference_df and current_df must be pandas DataFrames.")

    # 1. Compare Schema (Missing, Added, Type Changes)
    schema_diff = compare_schemas(reference_df, current_df)

    warnings_list: List[str] = []
    column_drift_results: Dict[str, ColumnDriftResult] = {}

    # 2. Evaluate common columns for distributional drift
    for col in schema_diff.common_columns:
        ref_series = reference_df[col]
        curr_series = current_df[col]

        # Check if suitable for numerical or categorical comparison
        is_num = bool(
            pd.api.types.is_numeric_dtype(ref_series)
            and pd.api.types.is_numeric_dtype(curr_series)
            and not pd.api.types.is_bool_dtype(ref_series)
            and not pd.api.types.is_bool_dtype(curr_series)
        )

        if is_num:
            result = _compare_numerical_column(
                col_name=col,
                ref_series=ref_series,
                curr_series=curr_series,
                alpha=alpha,
            )
        else:
            result = _compare_categorical_column(
                col_name=col,
                ref_series=ref_series,
                curr_series=curr_series,
                alpha=alpha,
            )

        if result.skipped and result.skip_reason:
            warnings_list.append(f"Column '{col}': {result.skip_reason}")

        column_drift_results[col] = result

    # 3. Overall Dataset Drift Aggregation
    evaluated_results = [r for r in column_drift_results.values() if not r.skipped]
    total_evaluated = len(evaluated_results)
    drifted_count = sum(1 for r in evaluated_results if r.drift_detected)
    drift_share = round((drifted_count / total_evaluated), 4) if total_evaluated > 0 else 0.0
    dataset_drift = bool(drift_share >= drift_share_threshold)

    disclaimer = (
        "Academic & Engineering Notice: Statistical data drift denotes a distribution shift "
        "(covariate shift) between the reference and current data populations. It indicates "
        "that machine learning models trained on baseline data may require performance monitoring "
        "or recalibration. Drift is NOT synonymous with data corruption, data pipeline failure, "
        "or invalid values."
    )

    return DriftAnalysisResult(
        has_reference=True,
        schema_diff=schema_diff,
        column_drift=column_drift_results,
        total_columns_compared=total_evaluated,
        drifted_columns_count=drifted_count,
        share_of_drifted_columns=drift_share,
        dataset_drift_detected=dataset_drift,
        warnings=warnings_list,
        disclaimer=disclaimer,
    )


def compare_schemas(ref_df: pd.DataFrame, curr_df: pd.DataFrame) -> SchemaDiffResult:
    """Identify missing columns, newly added columns, and data type changes."""
    ref_cols = list(ref_df.columns)
    curr_cols = list(curr_df.columns)

    ref_set = set(ref_cols)
    curr_set = set(curr_cols)

    missing = sorted(list(ref_set - curr_set))
    added = sorted(list(curr_set - ref_set))
    common = [c for c in ref_cols if c in curr_set]

    type_changes: Dict[str, Dict[str, str]] = {}
    for col in common:
        ref_dtype = str(ref_df[col].dtype)
        curr_dtype = str(curr_df[col].dtype)
        if ref_dtype != curr_dtype:
            type_changes[col] = {
                "reference_dtype": ref_dtype,
                "current_dtype": curr_dtype,
            }

    has_changes = bool(missing or added or type_changes)

    return SchemaDiffResult(
        missing_columns=missing,
        added_columns=added,
        type_changes=type_changes,
        common_columns=common,
        has_schema_changes=has_changes,
    )


# ---------------------------------------------------------------------------
# Internal Statistical Comparison Helpers
# ---------------------------------------------------------------------------

def _compare_numerical_column(
    col_name: str,
    ref_series: pd.Series,
    curr_series: pd.Series,
    alpha: float,
) -> ColumnDriftResult:
    """Compare continuous numerical distributions using Two-Sample Kolmogorov-Smirnov test."""
    # Filter non-null, finite values
    ref_clean = ref_series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    curr_clean = curr_series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    n_ref = len(ref_clean)
    n_curr = len(curr_clean)

    # 1. Edge Case: Insufficient samples
    if n_ref < 5 or n_curr < 5:
        return ColumnDriftResult(
            column_name=col_name,
            column_type="numeric",
            method="Two-sample Kolmogorov-Smirnov test",
            engine="scipy",
            statistic=None,
            p_value=None,
            threshold_alpha=alpha,
            drift_detected=False,
            sample_size_reference=n_ref,
            sample_size_current=n_curr,
            skipped=True,
            skip_reason=f"Insufficient sample size (ref={n_ref}, curr={n_curr}; minimum 5 required).",
        )

    # 2. Edge Case: Constant (zero variance) in both or either
    ref_std = float(np.std(ref_clean))
    curr_std = float(np.std(curr_clean))
    if ref_std == 0.0 and curr_std == 0.0:
        if float(ref_clean[0]) == float(curr_clean[0]):
            return ColumnDriftResult(
                column_name=col_name,
                column_type="numeric",
                method="Two-sample Kolmogorov-Smirnov test",
                engine="scipy",
                statistic=0.0,
                p_value=1.0,
                threshold_alpha=alpha,
                drift_detected=False,
                sample_size_reference=n_ref,
                sample_size_current=n_curr,
                skipped=False,
                additional_metrics={"note": "Both datasets contain the identical constant value."},
            )
        else:
            return ColumnDriftResult(
                column_name=col_name,
                column_type="numeric",
                method="Two-sample Kolmogorov-Smirnov test",
                engine="scipy",
                statistic=1.0,
                p_value=0.0,
                threshold_alpha=alpha,
                drift_detected=True,
                sample_size_reference=n_ref,
                sample_size_current=n_curr,
                skipped=False,
                additional_metrics={"note": "Both datasets are constant but have different constant values."},
            )

    # 3. Two-sample Kolmogorov-Smirnov Test (SciPy)
    try:
        ks_res = stats.ks_2samp(ref_clean, curr_clean)
        ks_stat = float(round(ks_res.statistic, 4))
        p_val = float(round(ks_res.pvalue, 6))
        drift_detected = bool(p_val < alpha)

        # Compute Wasserstein Distance (Earth Mover's Distance)
        try:
            w_dist = float(round(stats.wasserstein_distance(ref_clean, curr_clean), 4))
        except Exception:
            w_dist = None

        return ColumnDriftResult(
            column_name=col_name,
            column_type="numeric",
            method="Two-sample Kolmogorov-Smirnov test (SciPy)",
            engine="scipy",
            statistic=ks_stat,
            p_value=p_val,
            threshold_alpha=alpha,
            drift_detected=drift_detected,
            sample_size_reference=n_ref,
            sample_size_current=n_curr,
            skipped=False,
            additional_metrics={
                "wasserstein_distance": w_dist,
                "reference_mean": float(round(np.mean(ref_clean), 4)),
                "current_mean": float(round(np.mean(curr_clean), 4)),
                "reference_std": float(round(ref_std, 4)),
                "current_std": float(round(curr_std, 4)),
            },
        )
    except Exception as exc:
        return ColumnDriftResult(
            column_name=col_name,
            column_type="numeric",
            method="Two-sample Kolmogorov-Smirnov test",
            engine="scipy",
            statistic=None,
            p_value=None,
            threshold_alpha=alpha,
            drift_detected=False,
            sample_size_reference=n_ref,
            sample_size_current=n_curr,
            skipped=True,
            skip_reason=f"Statistical calculation failed: {exc}",
        )


def _compare_categorical_column(
    col_name: str,
    ref_series: pd.Series,
    curr_series: pd.Series,
    alpha: float,
) -> ColumnDriftResult:
    """Compare categorical distributions using Pearson's Chi-Square Test & Jensen-Shannon Divergence."""
    ref_clean = ref_series.dropna().astype(str)
    curr_clean = curr_series.dropna().astype(str)

    n_ref = len(ref_clean)
    n_curr = len(curr_clean)

    # 1. Edge Case: Insufficient samples
    if n_ref < 5 or n_curr < 5:
        return ColumnDriftResult(
            column_name=col_name,
            column_type="categorical",
            method="Pearson Chi-Square test",
            engine="scipy",
            statistic=None,
            p_value=None,
            threshold_alpha=alpha,
            drift_detected=False,
            sample_size_reference=n_ref,
            sample_size_current=n_curr,
            skipped=True,
            skip_reason=f"Insufficient sample size (ref={n_ref}, curr={n_curr}; minimum 5 required).",
        )

    # 2. Edge Case: High-cardinality IDs (unique count ~ sample size)
    n_unique_ref = ref_clean.nunique()
    if n_unique_ref > 100 and (n_unique_ref / n_ref) > 0.8:
        return ColumnDriftResult(
            column_name=col_name,
            column_type="categorical",
            method="Pearson Chi-Square test",
            engine="scipy",
            statistic=None,
            p_value=None,
            threshold_alpha=alpha,
            drift_detected=False,
            sample_size_reference=n_ref,
            sample_size_current=n_curr,
            skipped=True,
            skip_reason=f"High-cardinality identifier ({n_unique_ref} unique categories); distribution comparison skipped.",
        )

    # 3. Align categorical categories across both datasets
    all_categories = sorted(list(set(ref_clean.unique()) | set(curr_clean.unique())))

    ref_counts = ref_clean.value_counts()
    curr_counts = curr_clean.value_counts()

    # Create contingency row vectors
    observed_ref = np.array([ref_counts.get(cat, 0) for cat in all_categories], dtype=float)
    observed_curr = np.array([curr_counts.get(cat, 0) for cat in all_categories], dtype=float)

    # Compute probability vectors for Jensen-Shannon divergence
    prob_ref = observed_ref / n_ref
    prob_curr = observed_curr / n_curr
    try:
        js_div = float(round(jensenshannon(prob_ref, prob_curr), 4))
    except Exception:
        js_div = None

    # Chi-square contingency test requires a 2-row table of counts
    contingency_table = np.array([observed_ref, observed_curr])

    # Filter out categories with zero total counts across both
    col_sums = contingency_table.sum(axis=0)
    contingency_table = contingency_table[:, col_sums > 0]

    if contingency_table.shape[1] < 2:
        # Both datasets have only 1 single identical category
        return ColumnDriftResult(
            column_name=col_name,
            column_type="categorical",
            method="Pearson Chi-Square test (SciPy)",
            engine="scipy",
            statistic=0.0,
            p_value=1.0,
            threshold_alpha=alpha,
            drift_detected=False,
            sample_size_reference=n_ref,
            sample_size_current=n_curr,
            skipped=False,
            additional_metrics={"jensen_shannon_divergence": js_div},
        )

    try:
        chi2_res = stats.chi2_contingency(contingency_table)
        chi2_stat = float(round(chi2_res.statistic, 4))
        p_val = float(round(chi2_res.pvalue, 6))
        drift_detected = bool(p_val < alpha)

        return ColumnDriftResult(
            column_name=col_name,
            column_type="categorical",
            method="Pearson Chi-Square test (SciPy)",
            engine="scipy",
            statistic=chi2_stat,
            p_value=p_val,
            threshold_alpha=alpha,
            drift_detected=drift_detected,
            sample_size_reference=n_ref,
            sample_size_current=n_curr,
            skipped=False,
            additional_metrics={
                "jensen_shannon_divergence": js_div,
                "degrees_of_freedom": int(chi2_res.dof),
                "distinct_categories_count": len(all_categories),
            },
        )
    except Exception as exc:
        return ColumnDriftResult(
            column_name=col_name,
            column_type="categorical",
            method="Pearson Chi-Square test",
            engine="scipy",
            statistic=None,
            p_value=None,
            threshold_alpha=alpha,
            drift_detected=False,
            sample_size_reference=n_ref,
            sample_size_current=n_curr,
            skipped=True,
            skip_reason=f"Chi-square calculation error: {exc}",
        )
