"""Schema & Data Rule Validation Engine using Pandera.

Validates DataFrames against explicit declarative rules defined in JSON configuration:
- Required and non-null constraints
- Numeric range boundaries (min_value, max_value)
- Allowed categorical sets
- Uniqueness constraints
- Type conformity

All failures are parsed into structured issue records with row references and severity.
The raw input DataFrame is never modified.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors


DEFAULT_CONFIG_PATH = Path("config/validation_rules.json")


@dataclass
class ValidationIssue:
    """Individual data validation failure record."""
    issue_type: str        # e.g., 'null_value', 'min_value_violation', 'max_value_violation', 'invalid_category', 'missing_column'
    column: str            # Column name or '<dataset>'
    row_reference: Optional[int]  # Integer row index where violation occurred (or None if column-level)
    severity: str          # 'critical', 'error', 'warning'
    description: str       # Human-readable explanation
    observed_value: Any = None  # Actual value observed in the failing cell

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return asdict(self)


@dataclass
class ValidationResult:
    """Aggregated validation output containing issue logs and summary statistics."""
    is_valid: bool
    rules_configured: bool
    total_issues: int
    critical_count: int
    error_count: int
    warning_count: int
    evaluated_columns: List[str]
    missing_columns: List[str]
    issues: List[ValidationIssue]
    issues_by_column: Dict[str, List[ValidationIssue]] = field(default_factory=dict)
    failing_rows_count: int = 0
    total_rows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "rules_configured": self.rules_configured,
            "total_issues": self.total_issues,
            "critical_count": self.critical_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "evaluated_columns": self.evaluated_columns,
            "missing_columns": self.missing_columns,
            "failing_rows_count": self.failing_rows_count,
            "total_rows": self.total_rows,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_validation_rules(
    config_source: Union[str, Path, Dict[str, Any], None] = None
) -> Dict[str, Any]:
    """Load validation rules from a dict, file path, or default JSON configuration.

    Args:
        config_source: Dict of rules, path to JSON file, or None (loads config/validation_rules.json).

    Returns:
        dict: Parsed rule configuration mapping column names to rule specifications.
    """
    if isinstance(config_source, dict):
        return config_source.get("rules", config_source)

    path_to_load = Path(config_source) if config_source else DEFAULT_CONFIG_PATH

    if not path_to_load.exists():
        return {}

    try:
        with open(path_to_load, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("rules", data)
    except Exception:
        return {}


def build_pandera_schema(rules: Dict[str, Any]) -> Optional[pa.DataFrameSchema]:
    """Build a declarative Pandera DataFrameSchema from configured rules.

    Args:
        rules: Dictionary mapping column names to rule definitions.

    Returns:
        pa.DataFrameSchema: Compiled Pandera schema, or None if rules dictionary is empty.
    """
    if not rules:
        return None

    columns_schema: Dict[str, pa.Column] = {}

    for col_name, rule in rules.items():
        if not isinstance(rule, dict):
            continue

        checks: List[pa.Check] = []

        # 1. Numerical boundaries
        min_val = rule.get("min_value")
        if min_val is not None:
            checks.append(
                pa.Check.ge(
                    min_val,
                    name=f"min_value_{min_val}",
                    error=f"Value must be >= {min_val}",
                )
            )

        max_val = rule.get("max_value")
        if max_val is not None:
            checks.append(
                pa.Check.le(
                    max_val,
                    name=f"max_value_{max_val}",
                    error=f"Value must be <= {max_val}",
                )
            )

        # 2. Allowed categorical values
        allowed = rule.get("allowed_values")
        if allowed is not None and isinstance(allowed, (list, set, tuple)):
            allowed_set = list(allowed)
            checks.append(
                pa.Check.isin(
                    allowed_set,
                    name="allowed_categorical_values",
                    error=f"Value must be one of {allowed_set}",
                )
            )

        is_nullable = bool(rule.get("nullable", True))
        is_required = bool(rule.get("required", False))
        is_unique = bool(rule.get("unique", False))

        columns_schema[col_name] = pa.Column(
            checks=checks,
            nullable=is_nullable,
            required=is_required,
            unique=is_unique,
        )

    return pa.DataFrameSchema(
        columns=columns_schema,
        strict=False,  # Allow extra columns not declared in rules
        coerce=False,  # Do not mutate input types
    )


def validate_dataset(
    df: pd.DataFrame,
    rules: Union[Dict[str, Any], str, Path, None] = None,
) -> ValidationResult:
    """Validate a DataFrame against configured rules using Pandera.

    Guarantees:
    - Input DataFrame is never mutated.
    - Handles schema violations gracefully without throwing unhandled exceptions.
    - Generates fine-grained structured issue records with row indices and severity.

    Args:
        df: Input DataFrame to validate.
        rules: Rule dictionary, path to rule JSON, or None for default configuration.

    Returns:
        ValidationResult: Complete summary of validation status and individual issue records.
    """
    parsed_rules = load_validation_rules(rules)
    total_rows = len(df)

    if not parsed_rules:
        return ValidationResult(
            is_valid=True,
            rules_configured=False,
            total_issues=0,
            critical_count=0,
            error_count=0,
            warning_count=0,
            evaluated_columns=list(df.columns),
            missing_columns=[],
            issues=[],
            issues_by_column={},
            failing_rows_count=0,
            total_rows=total_rows,
        )

    schema = build_pandera_schema(parsed_rules)
    if schema is None:
        return ValidationResult(
            is_valid=True,
            rules_configured=False,
            total_issues=0,
            critical_count=0,
            error_count=0,
            warning_count=0,
            evaluated_columns=list(df.columns),
            missing_columns=[],
            issues=[],
            issues_by_column={},
            failing_rows_count=0,
            total_rows=total_rows,
        )

    issues: List[ValidationIssue] = []
    missing_columns: List[str] = []
    evaluated_columns: List[str] = list(parsed_rules.keys())
    failing_row_indices: Set[int] = set()

    # Pre-check for missing required columns to populate missing_columns list cleanly
    df_cols_set = set(df.columns)
    for col_name, rule in parsed_rules.items():
        if rule.get("required", False) and col_name not in df_cols_set:
            missing_columns.append(col_name)

    # Run Pandera schema validation
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as err:
        failure_df = err.failure_cases
        if failure_df is not None and not failure_df.empty:
            for _, row in failure_df.iterrows():
                raw_col = row.get("column")
                check_name = str(row.get("check", ""))
                val = row.get("failure_case")
                idx_val = row.get("index")

                # Handle missing column: Pandera sets column=None and failure_case=<col_name>
                if check_name == "column_in_dataframe":
                    col = str(val) if pd.notna(val) else "<dataset>"
                else:
                    col = str(raw_col) if pd.notna(raw_col) else "<dataset>"

                # Parse row index reference safely
                row_idx: Optional[int] = None
                try:
                    if pd.notna(idx_val):
                        row_idx = int(idx_val)
                        failing_row_indices.add(row_idx)
                except (ValueError, TypeError):
                    row_idx = None

                # Determine issue type, severity, and human-readable description
                issue_type, severity, description = _interpret_pandera_failure(
                    col=col,
                    check_name=check_name,
                    observed_val=val,
                    row_idx=row_idx,
                    rules=parsed_rules,
                )

                issues.append(
                    ValidationIssue(
                        issue_type=issue_type,
                        column=col,
                        row_reference=row_idx,
                        severity=severity,
                        description=description,
                        observed_value=val,
                    )
                )

    # Group issues by column
    issues_by_column: Dict[str, List[ValidationIssue]] = {}
    critical_count = 0
    error_count = 0
    warning_count = 0

    for issue in issues:
        issues_by_column.setdefault(issue.column, []).append(issue)
        if issue.severity == "critical":
            critical_count += 1
        elif issue.severity == "warning":
            warning_count += 1
        else:
            error_count += 1

    total_issues = len(issues)
    is_valid = (total_issues == 0)

    return ValidationResult(
        is_valid=is_valid,
        rules_configured=True,
        total_issues=total_issues,
        critical_count=critical_count,
        error_count=error_count,
        warning_count=warning_count,
        evaluated_columns=evaluated_columns,
        missing_columns=missing_columns,
        issues=issues,
        issues_by_column=issues_by_column,
        failing_rows_count=len(failing_row_indices),
        total_rows=total_rows,
    )


def _interpret_pandera_failure(
    col: str,
    check_name: str,
    observed_val: Any,
    row_idx: Optional[int],
    rules: Dict[str, Any],
) -> tuple[str, str, str]:
    """Translate raw Pandera check identifier into human-readable description and severity."""
    col_rule = rules.get(col, {})
    row_str = f"at row {row_idx}" if row_idx is not None else ""

    if check_name in ("not_nullable", "non_nullable") or "not_nullable" in check_name:
        return (
            "null_value",
            "error",
            f"Non-nullable column '{col}' contains a missing/null value {row_str}.".strip(),
        )

    if check_name == "is_unique":
        return (
            "duplicate_value",
            "warning",
            f"Unique column '{col}' contains a duplicate value '{observed_val}' {row_str}.".strip(),
        )

    if ">=" in check_name or "greater_than" in check_name or "min_value" in check_name:
        min_v = col_rule.get("min_value")
        return (
            "min_value_violation",
            "error",
            f"Value '{observed_val}' in column '{col}' is below minimum limit of {min_v} {row_str}.".strip(),
        )

    if "<=" in check_name or "less_than" in check_name or "max_value" in check_name:
        max_v = col_rule.get("max_value")
        return (
            "max_value_violation",
            "error",
            f"Value '{observed_val}' in column '{col}' exceeds maximum limit of {max_v} {row_str}.".strip(),
        )

    if any(k in check_name for k in ("isin", "allowed_categorical", "one of", "allowed")):
        allowed = col_rule.get("allowed_values", [])
        return (
            "invalid_category",
            "error",
            f"Value '{observed_val}' in column '{col}' is not among allowed categories: {allowed} {row_str}.".strip(),
        )

    if check_name == "column_in_dataframe":
        return (
            "missing_column",
            "critical",
            f"Required column '{col}' was not found in the dataset.",
        )

    # General fallback
    return (
        "rule_violation",
        "error",
        f"Validation check '{check_name}' failed for column '{col}' with value '{observed_val}' {row_str}.".strip(),
    )
