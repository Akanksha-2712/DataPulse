"""Data Profiling Module.

Provides comprehensive descriptive profiling of tabular DataFrames:
- Dataset dimensions (rows, columns, total cells)
- Missing, empty, and infinite value metrics
- Duplicate row analysis
- Column-level data types and statistical distribution summaries
- Non-mutating and memory-efficient
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Detailed profile metrics for an individual DataFrame column."""
    name: str
    inferred_dtype: str
    total_count: int
    missing_count: int
    missing_pct: float
    empty_string_count: int
    infinite_count: int
    unique_count: int
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert column profile to dictionary."""
        return asdict(self)


@dataclass
class DatasetProfile:
    """Comprehensive dataset-level summary and column profiles."""
    row_count: int
    column_count: int
    total_cells: int
    missing_cells: int
    missing_cells_pct: float
    empty_strings_total: int
    infinite_values_total: int
    duplicate_rows: int
    duplicate_rows_pct: float
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    memory_usage_bytes: int
    columns: Dict[str, ColumnProfile]

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset profile to serializable dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "total_cells": self.total_cells,
            "missing_cells": self.missing_cells,
            "missing_cells_pct": self.missing_cells_pct,
            "empty_strings_total": self.empty_strings_total,
            "infinite_values_total": self.infinite_values_total,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_rows_pct": self.duplicate_rows_pct,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "memory_usage_bytes": self.memory_usage_bytes,
            "columns": {col: prof.to_dict() for col, prof in self.columns.items()},
        }


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Compute complete descriptive profile for a pandas DataFrame.

    Guarantees:
    - Input DataFrame is strictly read-only and never mutated.
    - Handles empty or single-column DataFrames safely.
    - Accurately tracks nulls, empty strings, and IEEE floating-point infinities (+/- inf).

    Args:
        df: Input DataFrame to profile.

    Returns:
        DatasetProfile: Structured profile with dataset and column-level metrics.
    """
    row_count, col_count = df.shape
    total_cells = row_count * col_count

    if row_count == 0 or col_count == 0:
        return DatasetProfile(
            row_count=row_count,
            column_count=col_count,
            total_cells=total_cells,
            missing_cells=0,
            missing_cells_pct=0.0,
            empty_strings_total=0,
            infinite_values_total=0,
            duplicate_rows=0,
            duplicate_rows_pct=0.0,
            numeric_columns=[],
            categorical_columns=[],
            datetime_columns=[],
            memory_usage_bytes=int(df.memory_usage(deep=True).sum()) if col_count > 0 else 0,
            columns={},
        )

    # Duplicate row calculation
    dup_mask = df.duplicated()
    duplicate_rows = int(dup_mask.sum())
    duplicate_rows_pct = round((duplicate_rows / row_count) * 100.0, 3)

    column_profiles: Dict[str, ColumnProfile] = {}
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []

    missing_cells_total = 0
    empty_strings_total = 0
    infinite_values_total = 0

    for col in df.columns:
        series = df[col]
        total_len = len(series)

        # 1. Missingness & Empty strings
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_len) * 100.0, 3)
        missing_cells_total += null_count

        empty_str_count = 0
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # Check for empty strings or whitespace-only strings among non-null values
            non_null_str = series.dropna().astype(str)
            empty_str_count = int((non_null_str.str.strip() == "").sum())
            empty_strings_total += empty_str_count

        # 2. Infinite values (for numeric)
        inf_count = 0
        is_num = bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))
        is_dt = bool(pd.api.types.is_datetime64_any_dtype(series))
        is_cat = bool(isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))

        stats: Dict[str, Any] = {}

        if is_num:
            numeric_cols.append(col)
            # Count infinities
            inf_count = int(np.isinf(series.dropna()).sum())
            infinite_values_total += inf_count

            # Filter finite values for distribution stats
            valid_vals = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_vals) > 0:
                stats = {
                    "min": float(valid_vals.min()),
                    "q25": float(valid_vals.quantile(0.25)),
                    "median": float(valid_vals.median()),
                    "mean": float(round(valid_vals.mean(), 4)),
                    "q75": float(valid_vals.quantile(0.75)),
                    "max": float(valid_vals.max()),
                    "std": float(round(valid_vals.std(ddof=1), 4)) if len(valid_vals) > 1 else 0.0,
                    "zeros_count": int((valid_vals == 0).sum()),
                    "finite_count": int(len(valid_vals)),
                }
            else:
                stats = {
                    "min": None,
                    "q25": None,
                    "median": None,
                    "mean": None,
                    "q75": None,
                    "max": None,
                    "std": None,
                    "zeros_count": 0,
                    "finite_count": 0,
                }
        elif is_dt:
            datetime_cols.append(col)
            valid_dt = series.dropna()
            if len(valid_dt) > 0:
                stats = {
                    "min": str(valid_dt.min()),
                    "max": str(valid_dt.max()),
                }
        else:
            categorical_cols.append(col)
            valid_cats = series.dropna().astype(str)
            if len(valid_cats) > 0:
                val_counts = valid_cats.value_counts()
                top_mode = str(val_counts.index[0])
                mode_freq = int(val_counts.iloc[0])
                stats = {
                    "mode": top_mode,
                    "mode_frequency": mode_freq,
                    "mode_pct": round((mode_freq / len(valid_cats)) * 100.0, 2),
                    "top_5_categories": {str(k): int(v) for k, v in val_counts.head(5).items()},
                }
            else:
                stats = {
                    "mode": None,
                    "mode_frequency": 0,
                    "mode_pct": 0.0,
                    "top_5_categories": {},
                }

        unique_count = int(series.nunique(dropna=True))

        column_profiles[col] = ColumnProfile(
            name=col,
            inferred_dtype=str(series.dtype),
            total_count=total_len,
            missing_count=null_count,
            missing_pct=null_pct,
            empty_string_count=empty_str_count,
            infinite_count=inf_count,
            unique_count=unique_count,
            is_numeric=is_num,
            is_categorical=is_cat,
            is_datetime=is_dt,
            stats=stats,
        )

    missing_cells_pct = round((missing_cells_total / total_cells) * 100.0, 3) if total_cells > 0 else 0.0
    mem_usage = int(df.memory_usage(deep=True).sum())

    return DatasetProfile(
        row_count=row_count,
        column_count=col_count,
        total_cells=total_cells,
        missing_cells=missing_cells_total,
        missing_cells_pct=missing_cells_pct,
        empty_strings_total=empty_strings_total,
        infinite_values_total=infinite_values_total,
        duplicate_rows=duplicate_rows,
        duplicate_rows_pct=duplicate_rows_pct,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        memory_usage_bytes=mem_usage,
        columns=column_profiles,
    )
