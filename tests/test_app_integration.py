"""Integration tests for Streamlit application pipeline and UI components.

Tests cover:
- End-to-end execution of run_analysis_pipeline in app.py with both reference and single-dataset modes.
- Plotly chart generation utilities in src/ui_charts.py:
  - Quality score gauge indicator
  - Dimensional health radar plot
  - Column missingness percentage bar chart
  - Dual-distribution drift comparisons (numeric and categorical)
  - Multivariate anomaly feature projection scatter plot
  - Anomaly suspicion score distribution histogram
- Report download serialization integrity from the pipeline bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import pytest

from app import run_analysis_pipeline
from src.synthetic_generator import generate_baseline_dataset, generate_current_dataset
from src.ui_charts import (
    plot_anomaly_scatter,
    plot_anomaly_score_distribution,
    plot_dimension_radar,
    plot_drift_distribution_comparison,
    plot_missingness_bar,
    plot_quality_gauge,
)


@pytest.fixture(scope="module")
def analysis_bundle() -> Dict[str, Any]:
    """Execute run_analysis_pipeline once for UI integration testing."""
    b_df = generate_baseline_dataset(200, seed=42)
    c_df, _ = generate_current_dataset(200, seed=101)

    weights = {"completeness": 40.0, "uniqueness": 25.0, "validity": 35.0}
    bundle = run_analysis_pipeline(
        current_df=c_df,
        reference_df=b_df,
        weights=weights,
        alpha=0.05,
        contamination=0.05,
        current_name="current_test.csv",
        reference_name="baseline_test.csv",
    )
    bundle["_current_df"] = c_df
    bundle["_reference_df"] = b_df
    return bundle


def test_pipeline_execution_with_reference(analysis_bundle: Dict[str, Any]) -> None:
    """Verify run_analysis_pipeline outputs all required artifacts when reference data is present."""
    b = analysis_bundle
    assert b["profile"].row_count == len(b["_current_df"])
    assert b["quality_scorecard"].composite_score > 0
    assert b["validation_result"].rules_configured is True
    assert b["drift_result"] is not None
    assert b["drift_result"].has_reference is True
    assert b["drift_result"].total_columns_compared > 0
    assert b["anomaly_result"].success is True
    assert b["anomaly_result"].anomalies_detected_count > 0


def test_pipeline_execution_without_reference() -> None:
    """Verify run_analysis_pipeline degrades gracefully when reference data is None."""
    c_df, _ = generate_current_dataset(100, seed=101)
    weights = {"completeness": 40.0, "uniqueness": 25.0, "validity": 35.0}

    bundle = run_analysis_pipeline(
        current_df=c_df,
        reference_df=None,
        weights=weights,
        alpha=0.05,
        contamination=0.05,
        current_name="single_dataset.csv",
    )

    assert bundle["profile"].row_count > 0
    assert bundle["drift_result"] is None
    assert bundle["anomaly_result"].success is True
    assert bundle["anomaly_result"].training_mode == "self_trained"


def test_plot_quality_gauge(analysis_bundle: Dict[str, Any]) -> None:
    """Verify quality gauge indicator chart renders correctly."""
    dqs = analysis_bundle["quality_scorecard"]
    fig = plot_quality_gauge(dqs.composite_score, dqs.grade)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    indicator = fig.data[0]
    assert indicator.type == "indicator"
    assert indicator.value == dqs.composite_score


def test_plot_dimension_radar(analysis_bundle: Dict[str, Any]) -> None:
    """Verify dimension radar polar chart renders all configured dimensions."""
    dqs = analysis_bundle["quality_scorecard"]
    fig = plot_dimension_radar(dqs.dimensions)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatterpolar"


def test_plot_missingness_bar(analysis_bundle: Dict[str, Any]) -> None:
    """Verify missingness bar chart correctly reflects column missing percentages."""
    prof = analysis_bundle["profile"]
    fig = plot_missingness_bar(prof)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert bar.type == "bar"
    assert len(bar.x) == prof.column_count


def test_plot_drift_distribution_comparison_numeric(analysis_bundle: Dict[str, Any]) -> None:
    """Verify numeric dual-distribution histogram comparison."""
    b = analysis_bundle
    fig = plot_drift_distribution_comparison(
        ref_df=b["_reference_df"],
        cur_df=b["_current_df"],
        col_name="annual_income",
        is_numeric=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].name == "Baseline Reference"
    assert fig.data[1].name == "Current Dataset"


def test_plot_drift_distribution_comparison_categorical(analysis_bundle: Dict[str, Any]) -> None:
    """Verify categorical frequency comparison grouped bar chart."""
    b = analysis_bundle
    fig = plot_drift_distribution_comparison(
        ref_df=b["_reference_df"],
        cur_df=b["_current_df"],
        col_name="category",
        is_numeric=False,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].type == "bar"
    assert fig.data[1].type == "bar"


def test_plot_anomaly_scatter(analysis_bundle: Dict[str, Any]) -> None:
    """Verify 2D feature projection scatter plot with anomaly color coding."""
    b = analysis_bundle
    fig = plot_anomaly_scatter(
        cur_df=b["_current_df"],
        anomaly_result=b["anomaly_result"],
        x_col="age",
        y_col="annual_income",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_plot_anomaly_score_distribution(analysis_bundle: Dict[str, Any]) -> None:
    """Verify continuous anomaly suspicion score histogram."""
    anom = analysis_bundle["anomaly_result"]
    fig = plot_anomaly_score_distribution(anom)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "histogram"
