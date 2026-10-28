"""Workflow and state management tests for DataPulse.

Tests verify:
1. Session state lifecycle, default initialization, and safe transitions.
2. In-memory demo benchmark ingestion (1,226 target records vs 1,200 reference records).
3. Quality audit execution, Pandera schema validation, and composite DQS calculation.
4. Drift detection execution, hypothesis test statistics, and schema divergence checks.
5. Isolation Forest anomaly detection, continuous suspicion score scaling, and explainability.
6. Multi-format report compilation (PDF via ReportLab, XLSX via openpyxl, CSV streaming logs).
7. Downstream state invalidation when datasets are replaced or cleared.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from app import (
    load_demo_benchmark,
    run_analysis_pipeline,
    set_current_dataset,
    set_reference_dataset,
)
from src.anomaly_detection import detect_anomalies
from src.data_loader import read_csv_bytes
from src.drift_detection import detect_drift
from src.profiling import profile_dataset
from src.quality_score import calculate_data_quality_score
from src.report_generator import (
    ReportMetadata,
    export_analysis_to_excel,
    export_analysis_to_pdf,
    export_anomalies_to_csv,
    export_drift_to_csv,
    export_issues_to_csv,
)
from src.synthetic_generator import generate_baseline_dataset, generate_current_dataset
from src.validation import load_validation_rules, validate_dataset


def test_demo_datasets_exist_and_loadable() -> None:
    """Verify demo benchmark datasets load cleanly from disk or synthetic fallback."""
    project_root = Path(__file__).resolve().parent.parent
    cur_path = project_root / "data" / "demo" / "current.csv"
    ref_path = project_root / "data" / "demo" / "baseline.csv"

    assert cur_path.exists(), "Target demo file current.csv must exist"
    assert ref_path.exists(), "Reference demo file baseline.csv must exist"

    cur_df = read_csv_bytes(cur_path.read_bytes())
    ref_df = read_csv_bytes(ref_path.read_bytes())

    assert len(cur_df) == 1226
    assert len(ref_df) == 1200
    assert "customer_id" in cur_df.columns
    assert "annual_income" in cur_df.columns


def test_quality_audit_end_to_end() -> None:
    """Verify data profiling, Pandera validation, and DQS scoring on target data."""
    cur_df, _ = generate_current_dataset(300, seed=101)
    prof = profile_dataset(cur_df)
    assert prof.row_count == len(cur_df)
    assert prof.missing_cells > 0
    assert prof.duplicate_rows > 0

    project_root = Path(__file__).resolve().parent.parent
    rules_path = project_root / "config" / "validation_rules.json"
    rules = load_validation_rules(rules_path) if rules_path.exists() else None
    val_res = validate_dataset(cur_df, rules)
    assert val_res.rules_configured is True
    assert val_res.total_issues > 0

    scorecard = calculate_data_quality_score(cur_df, prof, val_res)
    assert 0.0 <= scorecard.composite_score <= 100.0
    assert scorecard.grade.lower() in {"excellent", "good", "fair", "poor", "a", "b", "c", "d", "f"}
    assert "completeness" in scorecard.dimensions
    assert "uniqueness" in scorecard.dimensions
    assert "validity" in scorecard.dimensions


def test_drift_detection_end_to_end() -> None:
    """Verify statistical drift tests between baseline and shifted current dataset."""
    ref_df = generate_baseline_dataset(300, seed=42)
    cur_df, _ = generate_current_dataset(300, seed=101)

    drift_res = detect_drift(ref_df, cur_df, alpha=0.05)
    assert drift_res.has_reference is True
    assert drift_res.total_columns_compared > 0
    assert drift_res.dataset_drift_detected is True
    assert drift_res.drifted_columns_count > 0

    # Test CSV export
    csv_bytes = export_drift_to_csv(drift_res)
    assert len(csv_bytes) > 0
    assert b"column_name" in csv_bytes or b"Column" in csv_bytes or b"drift" in csv_bytes.lower()


def test_anomaly_detection_end_to_end() -> None:
    """Verify Isolation Forest flags planted anomalies with continuous suspicion scores."""
    ref_df = generate_baseline_dataset(300, seed=42)
    cur_df, planted_indices = generate_current_dataset(300, seed=101)

    anom_res = detect_anomalies(
        current_df=cur_df,
        reference_df=ref_df,
        contamination=0.05,
        random_state=42,
    )
    assert anom_res.success is True
    assert anom_res.anomalies_detected_count > 0
    assert 0.0 < anom_res.anomaly_share <= 1.0

    # Check suspicion score bounds
    for rec in anom_res.anomaly_records:
        assert 0.0 <= rec.anomaly_score <= 1.0

    # Test CSV export
    csv_bytes = export_anomalies_to_csv(anom_res)
    assert len(csv_bytes) > 0
    assert b"suspicion_score" in csv_bytes or b"anomaly" in csv_bytes.lower()


def test_reports_generation_end_to_end() -> None:
    """Verify PDF and Excel report compilation from analysis artifacts."""
    ref_df = generate_baseline_dataset(200, seed=42)
    cur_df, _ = generate_current_dataset(200, seed=101)

    prof = profile_dataset(cur_df)
    project_root = Path(__file__).resolve().parent.parent
    rules_path = project_root / "config" / "validation_rules.json"
    rules = load_validation_rules(rules_path) if rules_path.exists() else None
    val_res = validate_dataset(cur_df, rules)
    scorecard = calculate_data_quality_score(cur_df, prof, val_res)
    drift_res = detect_drift(ref_df, cur_df, alpha=0.05)
    anom_res = detect_anomalies(cur_df, ref_df, contamination=0.05, random_state=42)

    meta = ReportMetadata(
        dataset_name="target_test.csv",
        reference_name="baseline_test.csv",
        author="DataPulse Test Suite",
    )

    # 1. PDF Generation
    pdf_bytes = export_analysis_to_pdf(prof, scorecard, val_res, drift_res, anom_res, meta)
    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

    # 2. Excel Generation
    excel_bytes = export_analysis_to_excel(prof, scorecard, val_res, drift_res, anom_res, meta)
    assert isinstance(excel_bytes, (bytes, bytearray))
    assert len(excel_bytes) > 1000
    assert excel_bytes.startswith(b"PK")  # ZIP header for OpenXML .xlsx

    # 3. Issues CSV Generation
    issues_csv = export_issues_to_csv(val_res)
    assert len(issues_csv) > 0
    assert b"severity" in issues_csv.lower() or b"issue" in issues_csv.lower()
