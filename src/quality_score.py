"""Data Quality Scoring Engine.

Calculates a transparent, deterministic composite Data Quality Score (DQS) in [0, 100]
based on project-defined dimensions:
- Completeness: Proportion of non-null and non-empty cells
- Row Uniqueness: Proportion of distinct rows (1 - duplicate rate)
- Rule-based Validity: Proportion of evaluated cells satisfying explicit Pandera rules

Project Starting Weights:
- completeness = 40
- uniqueness = 25
- validity = 35

Weights are dynamically normalized across dimensions that are actually applicable.
If no validation rules are configured, validity is marked not applicable and its weight
is omitted rather than simulated.

Academic Disclaimer:
This composite score is a project-defined indicator for academic monitoring and is
NOT an official industry or ISO standard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.profiling import DatasetProfile, profile_dataset
from src.validation import ValidationResult, validate_dataset


DEFAULT_WEIGHTS = {
    "completeness": 40.0,
    "uniqueness": 25.0,
    "validity": 35.0,
}


@dataclass
class DimensionScore:
    """Detailed score and weighting for a single data quality dimension."""
    dimension_name: str
    score: Optional[float]           # Score in [0, 100], or None if not applicable
    is_applicable: bool              # Whether this dimension was measured
    raw_weight: float                # Configured starting weight
    normalized_weight: float         # Proportion of total applicable weight [0.0, 1.0]
    description: str                 # Method of calculation
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dimension score to dictionary."""
        return asdict(self)


@dataclass
class QualityScorecard:
    """Comprehensive quality evaluation report containing composite score and breakdown."""
    composite_score: float           # Overall DQS in [0, 100]
    grade: str                       # 'Excellent', 'Good', 'Fair', 'Poor'
    dimensions: Dict[str, DimensionScore]
    weights_used: Dict[str, float]
    normalized_weights: Dict[str, float]
    calculation_explanation: str
    disclaimer: str
    limitations: list[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert scorecard to serializable dictionary."""
        return {
            "composite_score": self.composite_score,
            "grade": self.grade,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "weights_used": self.weights_used,
            "normalized_weights": self.normalized_weights,
            "calculation_explanation": self.calculation_explanation,
            "disclaimer": self.disclaimer,
            "limitations": self.limitations,
        }


def calculate_data_quality_score(
    df: pd.DataFrame,
    profile: Optional[DatasetProfile] = None,
    validation_result: Optional[ValidationResult] = None,
    custom_weights: Optional[Dict[str, float]] = None,
) -> QualityScorecard:
    """Compute transparent Data Quality Score (DQS) with dynamic weight normalization.

    Guarantees:
    - Input DataFrame is never mutated.
    - Dynamically normalizes weights if validity is not applicable.
    - Accurately computes completeness, uniqueness, and rule validity.
    - Documents formal mathematical rationale and academic limitations.

    Args:
        df: Input DataFrame.
        profile: Optional pre-computed DatasetProfile (computed if None).
        validation_result: Optional pre-computed ValidationResult (computed if None).
        custom_weights: Optional weight dictionary overriding default 40/25/35 weights.

    Returns:
        QualityScorecard: Structured scorecard with composite score and dimension breakdowns.
    """
    weights = dict(DEFAULT_WEIGHTS)
    if custom_weights:
        weights.update(custom_weights)

    # 1. Profile dataset if not provided
    prof = profile if profile is not None else profile_dataset(df)

    # 2. Compute Completeness Score (Q_comp)
    total_cells = prof.total_cells
    missing_cells = prof.missing_cells + prof.empty_strings_total

    if total_cells == 0:
        comp_score = 100.0
    else:
        comp_ratio = max(0.0, min(1.0, missing_cells / total_cells))
        comp_score = round((1.0 - comp_ratio) * 100.0, 2)

    comp_dimension = DimensionScore(
        dimension_name="Completeness",
        score=comp_score,
        is_applicable=True,
        raw_weight=weights["completeness"],
        normalized_weight=0.0,  # Will be normalized below
        description="Proportion of populated cells against missing (null or empty string) cells.",
        metrics={
            "total_cells": total_cells,
            "missing_cells": prof.missing_cells,
            "empty_strings": prof.empty_strings_total,
            "missing_percentage": prof.missing_cells_pct,
        },
    )

    # 3. Compute Uniqueness Score (Q_uniq)
    total_rows = prof.row_count
    dup_rows = prof.duplicate_rows

    if total_rows == 0:
        uniq_score = 100.0
    else:
        dup_ratio = max(0.0, min(1.0, dup_rows / total_rows))
        uniq_score = round((1.0 - dup_ratio) * 100.0, 2)

    uniq_dimension = DimensionScore(
        dimension_name="Uniqueness",
        score=uniq_score,
        is_applicable=True,
        raw_weight=weights["uniqueness"],
        normalized_weight=0.0,  # Will be normalized below
        description="Proportion of distinct rows (1 - duplicate row rate).",
        metrics={
            "total_rows": total_rows,
            "duplicate_rows": dup_rows,
            "duplicate_percentage": prof.duplicate_rows_pct,
        },
    )

    # 4. Compute Validity Score (Q_val)
    val_res = validation_result
    validity_applicable = bool(val_res is not None and val_res.rules_configured)

    val_dimension: DimensionScore
    if validity_applicable and val_res is not None:
        # Total evaluated cell checks = total_rows * number of configured columns present
        valid_cols_evaluated = len([c for c in val_res.evaluated_columns if c in df.columns])
        total_eval_cells = max(1, total_rows * max(1, valid_cols_evaluated))

        # Count violations
        total_violations = val_res.total_issues
        violation_ratio = max(0.0, min(1.0, total_violations / total_eval_cells))
        val_score = round((1.0 - violation_ratio) * 100.0, 2)

        val_dimension = DimensionScore(
            dimension_name="Validity",
            score=val_score,
            is_applicable=True,
            raw_weight=weights["validity"],
            normalized_weight=0.0,
            description="Proportion of evaluated values satisfying declared Pandera schema constraints.",
            metrics={
                "total_issues": total_violations,
                "critical_issues": val_res.critical_count,
                "error_issues": val_res.error_count,
                "warning_issues": val_res.warning_count,
                "evaluated_columns": valid_cols_evaluated,
                "failing_rows": val_res.failing_rows_count,
            },
        )
    else:
        val_dimension = DimensionScore(
            dimension_name="Validity",
            score=None,
            is_applicable=False,
            raw_weight=weights["validity"],
            normalized_weight=0.0,
            description="No validation rules configured. Validity was not measured.",
            metrics={
                "reason": "Explicit validation rules were not supplied or configured.",
            },
        )

    # 5. Dynamic Weight Normalization
    dimensions_dict = {
        "completeness": comp_dimension,
        "uniqueness": uniq_dimension,
        "validity": val_dimension,
    }

    applicable_weight_sum = sum(
        dim.raw_weight for dim in dimensions_dict.values() if dim.is_applicable
    )

    if applicable_weight_sum == 0.0:
        applicable_weight_sum = 1.0

    normalized_weights_dict: Dict[str, float] = {}
    composite_score = 0.0

    for key, dim in dimensions_dict.items():
        if dim.is_applicable and dim.score is not None:
            norm_w = dim.raw_weight / applicable_weight_sum
            dim.normalized_weight = round(norm_w, 4)
            normalized_weights_dict[key] = round(norm_w * 100.0, 2)
            composite_score += dim.score * norm_w
        else:
            dim.normalized_weight = 0.0
            normalized_weights_dict[key] = 0.0

    composite_score = round(max(0.0, min(100.0, composite_score)), 2)

    # 6. Qualitative Grade Assignment
    if composite_score >= 90.0:
        grade = "Excellent"
    elif composite_score >= 75.0:
        grade = "Good"
    elif composite_score >= 60.0:
        grade = "Fair"
    else:
        grade = "Poor"

    # 7. Step-by-Step Explanation
    explanation_parts = [
        f"Composite Data Quality Score (DQS): {composite_score:.2f} / 100 ({grade})",
        "",
        "Weight Normalization & Formula Breakdown:",
    ]

    for key, dim in dimensions_dict.items():
        if dim.is_applicable and dim.score is not None:
            explanation_parts.append(
                f"- {dim.dimension_name}: Score = {dim.score:.2f}%, "
                f"Configured Weight = {dim.raw_weight:.1f}, "
                f"Effective Normalized Weight = {dim.normalized_weight * 100:.2f}%"
            )
        else:
            explanation_parts.append(
                f"- {dim.dimension_name}: NOT APPLICABLE (Configured Weight = {dim.raw_weight:.1f} omitted from denominator)"
            )

    formula_str = " + ".join(
        [
            f"({dim.score:.2f} * {dim.normalized_weight:.4f})"
            for dim in dimensions_dict.values()
            if dim.is_applicable and dim.score is not None
        ]
    )
    explanation_parts.append("")
    explanation_parts.append(f"Calculation: DQS = {formula_str} = {composite_score:.2f}")

    explanation = "\n".join(explanation_parts)

    disclaimer = (
        "Project-Defined Indicator Notice: This Data Quality Score is a heuristic composite "
        "metric designed specifically for this system's architecture. It is NOT an official "
        "ISO standard (such as ISO 8000 or ISO 25012) or universal industry benchmark."
    )

    limitations = [
        "Dimensional Independence Assumption: The linear weighted aggregation assumes dimensions "
        "are independent. In practice, a corrupted primary key can invalidate row utility regardless "
        "of high completeness.",
        "Domain Context Sensitivity: Default weights (40% Completeness, 25% Uniqueness, 35% Validity) "
        "are heuristic starting points. Domain-critical columns (e.g. medical dosages, transaction amounts) "
        "may require custom weight assignments.",
        "Syntactic vs. Semantic Correctness: The validity score verifies that values conform to configured "
        "ranges and schemas; it cannot detect if a syntactically valid value is factually untrue.",
    ]

    return QualityScorecard(
        composite_score=composite_score,
        grade=grade,
        dimensions=dimensions_dict,
        weights_used=weights,
        normalized_weights=normalized_weights_dict,
        calculation_explanation=explanation,
        disclaimer=disclaimer,
        limitations=limitations,
    )
