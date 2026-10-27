"""Data Quality Monitoring and Anomaly Detection System — Report Generation and Export Module.

This module provides multi-format export engines for analysis results:
1. Microsoft Excel (.xlsx) via openpyxl:
   - Multi-tab workbook (Summary, Quality Issues, Drift Results, Anomalies, Dataset Profile, Metadata).
   - Enterprise styling (Navy #1B365D header theme, auto-fit column widths, freeze panes, auto-filters).
   - Native number formatting (percentages, decimals, integers).
2. Professional PDF (.pdf) via ReportLab Platypus:
   - Two-pass NumberedCanvas for dynamic "Page X of Y" running headers and footers.
   - Elegant executive summary, Data Quality Scorecard, drift analysis, and multivariate anomaly breakdown.
   - Comprehensive academic methodology, mathematical equations, and viva voce defense notes.
3. Comma-Separated Values (.csv):
   - Fast, streaming raw data export for validation issues, drift comparisons, and flagged anomalies.

Design Principles:
- 100% In-Memory Streaming: Generates bytes via io.BytesIO without writing unencrypted data to disk.
- Safe Degraded Mode: Gracefully handles missing reference datasets, unconfigured rules, or small data.
- Documentation Integrity: Preserves academic transparency, mathematical rigor, and disclaimer citations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import xml.sax.saxutils as saxutils

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.anomaly_detection import AnomalyDetectionResult
from src.drift_detection import DriftAnalysisResult
from src.profiling import DatasetProfile
from src.quality_score import QualityScorecard
from src.validation import ValidationIssue, ValidationResult


# ==============================================================================
# REPORT METADATA CONTAINER
# ==============================================================================

@dataclass
class ReportMetadata:
    """Contextual metadata describing the analysis run for report headers and audit trails."""
    dataset_name: str = "current_dataset"
    reference_name: Optional[str] = None
    worksheet_name: Optional[str] = None
    reference_worksheet_name: Optional[str] = None
    generated_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    author: str = "Automated Data Quality & Anomaly Detection System"
    academic_context: str = "B.Tech Artificial Intelligence & Data Science Final Project"
    additional_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "reference_name": self.reference_name,
            "worksheet_name": self.worksheet_name,
            "reference_worksheet_name": self.reference_worksheet_name,
            "generated_at": self.generated_at,
            "author": self.author,
            "academic_context": self.academic_context,
            "additional_notes": self.additional_notes,
        }


# ==============================================================================
# CSV EXPORT UTILITIES
# ==============================================================================

def export_issues_to_csv(validation_result: Optional[ValidationResult]) -> bytes:
    """Serialize validation issues to a CSV byte stream.

    Parameters
    ----------
    validation_result : Optional[ValidationResult]
        Aggregated validation output containing issue records.

    Returns
    -------
    bytes
        UTF-8 encoded CSV file bytes.
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "issue_index",
        "severity",
        "issue_type",
        "column",
        "row_reference",
        "observed_value",
        "description",
    ])

    if validation_result is not None and validation_result.issues:
        for idx, issue in enumerate(validation_result.issues, start=1):
            writer.writerow([
                idx,
                issue.severity.upper(),
                issue.issue_type,
                issue.column,
                "" if issue.row_reference is None else issue.row_reference,
                "" if issue.observed_value is None else str(issue.observed_value),
                issue.description,
            ])

    return output.getvalue().encode("utf-8")


def export_anomalies_to_csv(anomaly_result: Optional[AnomalyDetectionResult]) -> bytes:
    """Serialize flagged anomaly records to a CSV byte stream.

    Parameters
    ----------
    anomaly_result : Optional[AnomalyDetectionResult]
        Multivariate anomaly detection report.

    Returns
    -------
    bytes
        UTF-8 encoded CSV file bytes.
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "row_index",
        "is_anomaly",
        "suspicion_score",
        "raw_decision_score",
        "top_deviating_features",
        "analyzed_values",
    ])

    if anomaly_result is not None and anomaly_result.anomaly_records:
        for rec in anomaly_result.anomaly_records:
            top_dev_str = "; ".join(
                f"{k}: {v:+.2f} IQR"
                for k, v in sorted(
                    rec.feature_deviations.items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]
            )
            raw_vals_str = "; ".join(f"{k}={v}" for k, v in rec.record_data.items())
            writer.writerow([
                rec.row_index,
                "ANOMALY" if rec.is_anomaly else "NORMAL",
                f"{rec.anomaly_score:.4f}",
                f"{rec.raw_decision_score:.4f}",
                top_dev_str,
                raw_vals_str,
            ])

    return output.getvalue().encode("utf-8")


def export_drift_to_csv(drift_result: Optional[DriftAnalysisResult]) -> bytes:
    """Serialize distribution drift metrics to a CSV byte stream.

    Parameters
    ----------
    drift_result : Optional[DriftAnalysisResult]
        Schema and statistical drift analysis report.

    Returns
    -------
    bytes
        UTF-8 encoded CSV file bytes.
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "column_name",
        "column_type",
        "method",
        "engine",
        "statistic",
        "p_value",
        "threshold_alpha",
        "drift_detected",
        "sample_size_reference",
        "sample_size_current",
        "skipped",
        "skip_reason",
    ])

    if drift_result is not None and drift_result.column_drift:
        for col_name, cdr in drift_result.column_drift.items():
            writer.writerow([
                col_name,
                cdr.column_type,
                cdr.method,
                cdr.engine,
                "" if cdr.statistic is None else f"{cdr.statistic:.5f}",
                "" if cdr.p_value is None else f"{cdr.p_value:.5f}",
                cdr.threshold_alpha,
                "YES" if cdr.drift_detected else "NO",
                cdr.sample_size_reference,
                cdr.sample_size_current,
                "YES" if cdr.skipped else "NO",
                cdr.skip_reason or "",
            ])

    return output.getvalue().encode("utf-8")


# ==============================================================================
# EXCEL WORKBOOK EXPORT (openpyxl)
# ==============================================================================

def export_analysis_to_excel(
    profile: Optional[DatasetProfile],
    quality_scorecard: Optional[QualityScorecard] = None,
    validation_result: Optional[ValidationResult] = None,
    drift_result: Optional[DriftAnalysisResult] = None,
    anomaly_result: Optional[AnomalyDetectionResult] = None,
    metadata: Optional[ReportMetadata] = None,
) -> bytes:
    """Generate a multi-tab Microsoft Excel (.xlsx) workbook.

    Sheets generated:
    1. Summary: High-level metrics, DQS scorecard breakdown, drift summary, anomaly counts.
    2. Quality Issues: Filterable tabular log of rule validation violations.
    3. Drift Results: Statistical drift metrics, p-values, test names, and sample sizes.
    4. Anomalies: Candidate outliers sorted by suspicion score with feature deviations.
    5. Dataset Profile: Detailed column-by-column statistics, null counts, and data types.
    6. Metadata: Analysis options, timestamp, academic limitations, and disclaimers.

    Returns
    -------
    bytes
        Binary XLSX file bytes.
    """
    if metadata is None:
        metadata = ReportMetadata()

    wb = openpyxl.Workbook()
    # Remove default sheet created by openpyxl
    default_sheet = wb.active
    wb.remove(default_sheet)

    # Styles
    navy_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    steel_fill = PatternFill(start_color="2E5B88", end_color="2E5B88", fill_type="solid")
    soft_blue_fill = PatternFill(start_color="EBF3FB", end_color="EBF3FB", fill_type="solid")
    alert_red_fill = PatternFill(start_color="FFD1D1", end_color="FFD1D1", fill_type="solid")
    alert_orange_fill = PatternFill(start_color="FFE0B2", end_color="FFE0B2", fill_type="solid")
    alert_yellow_fill = PatternFill(start_color="FFF9C4", end_color="FFF9C4", fill_type="solid")

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    section_font = Font(name="Calibri", size=12, bold=True, color="1B365D")
    subheading_font = Font(name="Calibri", size=11, bold=True, color="2E5B88")
    bold_font = Font(name="Calibri", size=10, bold=True)
    regular_font = Font(name="Calibri", size=10)

    thin_border_side = Side(border_style="thin", color="D0D5DD")
    thin_border = Border(
        left=thin_border_side,
        right=thin_border_side,
        top=thin_border_side,
        bottom=thin_border_side,
    )

    def _apply_sheet_view(ws: Any) -> None:
        ws.views.sheetView[0].showGridLines = True

    def _auto_adjust_columns(ws: Any, min_w: int = 12, max_w: int = 65) -> None:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.value is not None:
                    lines = str(cell.value).split("\n")
                    for line in lines:
                        if len(line) > max_len:
                            max_len = len(line)
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, min_w), max_w)

    def _style_header_row(ws: Any, row_idx: int) -> None:
        for cell in ws[row_idx]:
            cell.fill = navy_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
        ws.row_dimensions[row_idx].height = 24

    # --------------------------------------------------------------------------
    # 1. SUMMARY SHEET
    # --------------------------------------------------------------------------
    ws_sum = wb.create_sheet(title="Summary")
    _apply_sheet_view(ws_sum)

    # Title block
    ws_sum["A1"] = "DATA QUALITY MONITORING SYSTEM — EXECUTIVE SUMMARY"
    ws_sum["A1"].font = Font(name="Calibri", size=14, bold=True, color="1B365D")
    ws_sum["A2"] = f"Generated on {metadata.generated_at} | Context: {metadata.academic_context}"
    ws_sum["A2"].font = Font(name="Calibri", size=10, italic=True, color="5A6B82")

    row = 4
    ws_sum.cell(row=row, column=1, value="Dataset & Run Details").font = section_font
    row += 1

    details = [
        ("Current Target File", metadata.dataset_name),
        ("Worksheet Name", metadata.worksheet_name or "N/A (CSV or default)"),
        ("Reference File", metadata.reference_name or "None (Unsupervised / Single-dataset mode)"),
        ("Total Rows Profiled", profile.row_count if profile else "N/A"),
        ("Total Columns Profiled", profile.column_count if profile else "N/A"),
        ("Memory Footprint", f"{(profile.memory_usage_bytes / 1024):.1f} KB" if profile else "N/A"),
    ]
    for label, val in details:
        ws_sum.cell(row=row, column=1, value=label).font = bold_font
        ws_sum.cell(row=row, column=2, value=val).font = regular_font
        row += 1

    row += 1
    # Quality Scorecard Table
    ws_sum.cell(row=row, column=1, value="Data Quality Scorecard (DQS)").font = section_font
    row += 1

    score_headers = ["Metric / Dimension", "Score (0-100)", "Starting Weight", "Normalized Weight", "Status / Notes"]
    ws_sum.append(score_headers)
    _style_header_row(ws_sum, row)
    row += 1

    if quality_scorecard:
        # Composite score row
        c_cell = ws_sum.cell(row=row, column=1, value="OVERALL QUALITY SCORE")
        c_cell.font = bold_font
        c_cell.fill = soft_blue_fill
        s_cell = ws_sum.cell(row=row, column=2, value=round(quality_scorecard.composite_score, 1))
        s_cell.font = bold_font
        s_cell.fill = soft_blue_fill
        ws_sum.cell(row=row, column=3, value=100.0).fill = soft_blue_fill
        ws_sum.cell(row=row, column=4, value=1.0).fill = soft_blue_fill
        g_cell = ws_sum.cell(row=row, column=5, value=f"Grade: {quality_scorecard.grade}")
        g_cell.font = bold_font
        g_cell.fill = soft_blue_fill
        for c in range(1, 6):
            ws_sum.cell(row=row, column=c).border = thin_border
        row += 1

        for dim_name, d_score in quality_scorecard.dimensions.items():
            ws_sum.cell(row=row, column=1, value=dim_name.capitalize()).font = regular_font
            ws_sum.cell(
                row=row,
                column=2,
                value=round(d_score.score, 1) if d_score.score is not None else "Not Applicable",
            )
            ws_sum.cell(row=row, column=3, value=d_score.raw_weight)
            ws_sum.cell(row=row, column=4, value=round(d_score.normalized_weight, 3))
            ws_sum.cell(row=row, column=5, value=d_score.description)
            for c in range(1, 6):
                ws_sum.cell(row=row, column=c).border = thin_border
            row += 1
    else:
        ws_sum.cell(row=row, column=1, value="Quality scorecard not calculated.")
        row += 1

    row += 1
    # Drift & Anomaly Summary Highlights
    ws_sum.cell(row=row, column=1, value="Monitoring Highlights").font = section_font
    row += 1

    highlights = [
        (
            "Schema & Distribution Drift",
            f"{drift_result.drifted_columns_count} / {drift_result.total_columns_compared} columns drifted"
            if drift_result and drift_result.has_reference
            else "Reference dataset not supplied (Drift detection skipped)",
        ),
        (
            "Validation Violations Logged",
            f"{validation_result.total_issues} issues ({validation_result.critical_count} critical)"
            if validation_result
            else "Validation rules not evaluated",
        ),
        (
            "Multivariate Anomalies Flagged",
            f"{anomaly_result.anomalies_detected_count} records ({anomaly_result.anomaly_share * 100:.1f}%)"
            if anomaly_result and anomaly_result.success
            else "Anomaly detection skipped or unsuccessful",
        ),
    ]
    for h_label, h_val in highlights:
        ws_sum.cell(row=row, column=1, value=h_label).font = bold_font
        ws_sum.cell(row=row, column=2, value=h_val).font = regular_font
        row += 1

    _auto_adjust_columns(ws_sum)

    # --------------------------------------------------------------------------
    # 2. QUALITY ISSUES SHEET
    # --------------------------------------------------------------------------
    ws_issues = wb.create_sheet(title="Quality Issues")
    _apply_sheet_view(ws_issues)

    issue_headers = ["Issue ID", "Severity", "Issue Type", "Column", "Row Reference", "Observed Value", "Description"]
    ws_issues.append(issue_headers)
    _style_header_row(ws_issues, 1)
    ws_issues.freeze_panes = "A2"

    if validation_result and validation_result.issues:
        for idx, issue in enumerate(validation_result.issues, start=1):
            r_idx = idx + 1
            sev = issue.severity.upper()
            ws_issues.append([
                idx,
                sev,
                issue.issue_type,
                issue.column,
                "" if issue.row_reference is None else issue.row_reference,
                "" if issue.observed_value is None else str(issue.observed_value),
                issue.description,
            ])
            # Severity color coding
            sev_cell = ws_issues.cell(row=r_idx, column=2)
            if sev in ("CRITICAL", "HIGH"):
                sev_cell.fill = alert_red_fill
            elif sev in ("ERROR", "MEDIUM"):
                sev_cell.fill = alert_orange_fill
            else:
                sev_cell.fill = alert_yellow_fill

            for col_idx in range(1, 8):
                ws_issues.cell(row=r_idx, column=col_idx).border = thin_border
    else:
        ws_issues.append([1, "INFO", "no_issues", "N/A", "N/A", "N/A", "No validation issues detected or rules not configured."])
        for col_idx in range(1, 8):
            ws_issues.cell(row=2, column=col_idx).border = thin_border

    ws_issues.auto_filter.ref = ws_issues.dimensions
    _auto_adjust_columns(ws_issues)

    # --------------------------------------------------------------------------
    # 3. DRIFT RESULTS SHEET
    # --------------------------------------------------------------------------
    ws_drift = wb.create_sheet(title="Drift Results")
    _apply_sheet_view(ws_drift)

    drift_headers = [
        "Column Name", "Data Type", "Statistical Method", "Engine",
        "Statistic", "P-Value", "Threshold (Alpha)", "Drift Detected?",
        "Ref Count", "Cur Count", "Status Notes"
    ]
    ws_drift.append(drift_headers)
    _style_header_row(ws_drift, 1)
    ws_drift.freeze_panes = "A2"

    if drift_result and drift_result.has_reference and drift_result.column_drift:
        for idx, (col_name, cdr) in enumerate(drift_result.column_drift.items(), start=1):
            r_idx = idx + 1
            stat_val = round(cdr.statistic, 5) if cdr.statistic is not None else ""
            pval_val = round(cdr.p_value, 5) if cdr.p_value is not None else ""
            drift_str = "YES" if cdr.drift_detected else "NO"
            notes = cdr.skip_reason if cdr.skipped else ("Distribution shifted significantly" if cdr.drift_detected else "Stationary")

            ws_drift.append([
                col_name,
                cdr.column_type,
                cdr.method,
                cdr.engine,
                stat_val,
                pval_val,
                cdr.threshold_alpha,
                drift_str,
                cdr.sample_size_reference,
                cdr.sample_size_current,
                notes,
            ])
            # Highlight drifted column
            if cdr.drift_detected:
                for c in range(1, 12):
                    ws_drift.cell(row=r_idx, column=c).fill = alert_orange_fill
            for c in range(1, 12):
                ws_drift.cell(row=r_idx, column=c).border = thin_border
    else:
        msg = "No reference dataset supplied. Drift detection was not performed." if not (drift_result and drift_result.has_reference) else "No common columns available for drift comparison."
        ws_drift.append(["N/A", "N/A", "N/A", "N/A", "", "", 0.05, "NO", 0, 0, msg])
        for c in range(1, 12):
            ws_drift.cell(row=2, column=c).border = thin_border

    ws_drift.auto_filter.ref = ws_drift.dimensions
    _auto_adjust_columns(ws_drift)

    # --------------------------------------------------------------------------
    # 4. ANOMALIES SHEET
    # --------------------------------------------------------------------------
    ws_anom = wb.create_sheet(title="Anomalies")
    _apply_sheet_view(ws_anom)

    anom_headers = [
        "Row Index", "Anomaly Flag", "Suspicion Score (0-1)",
        "Raw Decision Score", "Top Deviating Features (IQR Units)", "Raw Values"
    ]
    ws_anom.append(anom_headers)
    _style_header_row(ws_anom, 1)
    ws_anom.freeze_panes = "A2"

    if anomaly_result and anomaly_result.anomaly_records:
        for idx, rec in enumerate(anomaly_result.anomaly_records, start=1):
            r_idx = idx + 1
            top_dev_str = "; ".join(
                f"{k}: {v:+.2f} IQR"
                for k, v in sorted(
                    rec.feature_deviations.items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]
            )
            raw_vals_str = "; ".join(f"{k}={v}" for k, v in rec.record_data.items())

            ws_anom.append([
                rec.row_index,
                "SUSPICIOUS" if rec.is_anomaly else "NORMAL",
                round(rec.anomaly_score, 4),
                round(rec.raw_decision_score, 4),
                top_dev_str,
                raw_vals_str,
            ])
            # Highlight anomaly records
            if rec.is_anomaly:
                for c in range(1, 7):
                    ws_anom.cell(row=r_idx, column=c).fill = alert_red_fill
            for c in range(1, 7):
                ws_anom.cell(row=r_idx, column=c).border = thin_border
    else:
        msg = anomaly_result.status_message if (anomaly_result and hasattr(anomaly_result, "status_message")) else "No anomaly detection records available."
        ws_anom.append(["N/A", "NORMAL", 0.0, 0.0, "N/A", msg])
        for c in range(1, 7):
            ws_anom.cell(row=2, column=c).border = thin_border

    ws_anom.auto_filter.ref = ws_anom.dimensions
    _auto_adjust_columns(ws_anom)

    # --------------------------------------------------------------------------
    # 5. DATASET PROFILE SHEET
    # --------------------------------------------------------------------------
    ws_prof = wb.create_sheet(title="Dataset Profile")
    _apply_sheet_view(ws_prof)

    prof_headers = [
        "Column Name", "Inferred Type", "Total Count", "Missing Count",
        "Missing %", "Empty Strings", "Infinite Count", "Unique Count",
        "Min", "Max", "Mean", "Std Dev"
    ]
    ws_prof.append(prof_headers)
    _style_header_row(ws_prof, 1)
    ws_prof.freeze_panes = "A2"

    if profile and profile.columns:
        for idx, (col_name, cp) in enumerate(profile.columns.items(), start=1):
            r_idx = idx + 1
            stats = cp.stats or {}
            c_type = "Numeric" if cp.is_numeric else ("Categorical" if cp.is_categorical else ("DateTime" if cp.is_datetime else "Object"))
            min_val = round(stats["min"], 3) if "min" in stats and isinstance(stats["min"], (int, float)) else stats.get("min", "")
            max_val = round(stats["max"], 3) if "max" in stats and isinstance(stats["max"], (int, float)) else stats.get("max", "")
            mean_val = round(stats["mean"], 3) if "mean" in stats and isinstance(stats["mean"], (int, float)) else ""
            std_val = round(stats["std"], 3) if "std" in stats and isinstance(stats["std"], (int, float)) else ""

            ws_prof.append([
                col_name,
                c_type,
                cp.total_count,
                cp.missing_count,
                round(cp.missing_pct, 2),
                cp.empty_string_count,
                cp.infinite_count,
                cp.unique_count,
                min_val,
                max_val,
                mean_val,
                std_val,
            ])
            for c in range(1, 13):
                ws_prof.cell(row=r_idx, column=c).border = thin_border
    else:
        ws_prof.append(["N/A", "N/A", 0, 0, 0.0, 0, 0, 0, "", "", "", "No profile data available."])
        for c in range(1, 13):
            ws_prof.cell(row=2, column=c).border = thin_border

    ws_prof.auto_filter.ref = ws_prof.dimensions
    _auto_adjust_columns(ws_prof)

    # --------------------------------------------------------------------------
    # 6. METADATA & METHODOLOGY SHEET
    # --------------------------------------------------------------------------
    ws_meta = wb.create_sheet(title="Metadata")
    _apply_sheet_view(ws_meta)

    ws_meta["A1"] = "TECHNICAL AUDIT TRAIL & ACADEMIC METHODOLOGY"
    ws_meta["A1"].font = section_font

    meta_rows = [
        ("Execution Timestamp", metadata.generated_at),
        ("Report Author", metadata.author),
        ("Academic Context", metadata.academic_context),
        ("Target Dataset", metadata.dataset_name),
        ("Target Worksheet", metadata.worksheet_name or "N/A"),
        ("Reference Dataset", metadata.reference_name or "N/A"),
        ("Reference Worksheet", metadata.reference_worksheet_name or "N/A"),
        ("Data Quality Scoring Formula", "DQS = w_comp * S_comp + w_uniq * S_uniq + w_val * S_val (Weights normalized to sum to 100%)"),
        ("Numerical Drift Test", "Two-Sample Kolmogorov-Smirnov Test (H0: identical underlying continuous distributions)"),
        ("Categorical Drift Test", "Pearson Chi-Square Test of Independence with fallback to Wasserstein / J-S Divergence"),
        ("Anomaly Detection Engine", "Scikit-Learn IsolationForest (Tree-based ensemble recursive partitioning)"),
        ("Academic Viva Note 1", "An ML anomaly is an unusual multivariate candidate requiring human inspection, NOT proof of data corruption."),
        ("Academic Viva Note 2", "Distribution drift indicates a statistical covariate shift between reference and current sets; it does not invalidate correctness."),
        ("Academic Viva Note 3", "Data Quality Score (DQS) is an explainable project heuristic indicator, NOT an official ISO or IEEE standard."),
    ]
    cur_r = 3
    for k, v in meta_rows:
        ws_meta.cell(row=cur_r, column=1, value=k).font = bold_font
        ws_meta.cell(row=cur_r, column=2, value=v).font = regular_font
        cur_r += 1

    _auto_adjust_columns(ws_meta)

    # Save to in-memory bytes
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


# ==============================================================================
# REPORTLAB NUMBERED CANVAS (PAGE X OF Y)
# ==============================================================================

class NumberedCanvas(canvas.Canvas):
    """Two-pass ReportLab canvas enabling accurate 'Page X of Y' headers and footers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_running_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_running_decorations(self, page_count: int) -> None:
        """Draw running header on pages > 1, and footer on all pages."""
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#5A6B82"))
        self.setStrokeColor(colors.HexColor("#D0D5DD"))
        self.setLineWidth(0.5)

        page_w, page_h = letter
        left_m = 40.0
        right_m = page_w - 40.0

        # Running header (page 2 and beyond)
        if self._pageNumber > 1:
            self.drawString(
                left_m,
                page_h - 28.0,
                "Data Quality Monitoring and Anomaly Detection System — Academic Audit Report",
            )
            self.line(left_m, page_h - 32.0, right_m, page_h - 32.0)

        # Running footer (all pages)
        self.line(left_m, 36.0, right_m, 36.0)
        footer_left = "Confidential Academic Audit Report | 100% In-Memory Local Processing"
        footer_right = f"Page {self._pageNumber} of {page_count}"
        self.drawString(left_m, 24.0, footer_left)
        self.drawRightString(right_m, 24.0, footer_right)

        self.restoreState()


# ==============================================================================
# PDF EXPORT (ReportLab Platypus)
# ==============================================================================

def export_analysis_to_pdf(
    profile: Optional[DatasetProfile],
    quality_scorecard: Optional[QualityScorecard] = None,
    validation_result: Optional[ValidationResult] = None,
    drift_result: Optional[DriftAnalysisResult] = None,
    anomaly_result: Optional[AnomalyDetectionResult] = None,
    metadata: Optional[ReportMetadata] = None,
) -> bytes:
    """Generate a multi-page, publication-quality academic PDF report.

    Features:
    - Structured Platypus document flow with two-pass NumberedCanvas.
    - Executive summary with composite DQS card and dimensional breakdown.
    - Pandera validation findings with severity logs.
    - Distribution drift statistical findings (KS test, Chi-square).
    - Scikit-learn Isolation Forest multivariate anomaly rankings.
    - Rigorous academic methodology, mathematical formulation, and viva voce talking points.

    Returns
    -------
    bytes
        Binary PDF file bytes starting with %PDF-.
    """
    if metadata is None:
        metadata = ReportMetadata()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40.0,
        rightMargin=40.0,
        topMargin=42.0,
        bottomMargin=46.0,
    )

    # Styles
    base_styles = getSampleStyleSheet()

    navy_color = colors.HexColor("#1B365D")
    steel_color = colors.HexColor("#2E5B88")
    dark_text = colors.HexColor("#2C3E50")
    slate_gray = colors.HexColor("#5A6B82")
    light_bg = colors.HexColor("#F8F9FA")
    soft_blue = colors.HexColor("#EBF3FB")
    danger_red = colors.HexColor("#D9534F")
    warning_orange = colors.HexColor("#F0AD4E")
    success_green = colors.HexColor("#5CB85C")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=navy_color,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=slate_gray,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=navy_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=base_styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=steel_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=dark_text,
        spaceAfter=4,
    )
    italic_style = ParagraphStyle(
        "Italic_Custom",
        parent=body_style,
        fontName="Helvetica-Oblique",
        textColor=slate_gray,
    )
    tbl_cell_style = ParagraphStyle(
        "TableCell",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=dark_text,
    )
    tbl_hdr_style = ParagraphStyle(
        "TableHeader",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
        alignment=1,  # Center
    )

    printable_w = 532.0  # 612 - 80

    def _safe_str(val: Any) -> str:
        """Escape XML/HTML entities for ReportLab Paragraph rendering."""
        if val is None:
            return "-"
        return saxutils.escape(str(val))

    story: List[Any] = []

    # --------------------------------------------------------------------------
    # HEADER BANNER & METADATA
    # --------------------------------------------------------------------------
    story.append(Paragraph("Data Quality Monitoring and Anomaly Detection System", title_style))
    story.append(
        Paragraph(
            f"Automated Academic Audit Report & Statistical Evaluation | {metadata.academic_context}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=navy_color, spaceAfter=8))

    # Run Context Table
    meta_data = [
        [
            Paragraph("<b>Target Dataset:</b>", tbl_cell_style),
            Paragraph(_safe_str(metadata.dataset_name), tbl_cell_style),
            Paragraph("<b>Analyzed Sheet:</b>", tbl_cell_style),
            Paragraph(_safe_str(metadata.worksheet_name or "N/A (CSV / Default)"), tbl_cell_style),
        ],
        [
            Paragraph("<b>Reference Dataset:</b>", tbl_cell_style),
            Paragraph(_safe_str(metadata.reference_name or "None (Unsupervised Mode)"), tbl_cell_style),
            Paragraph("<b>Timestamp:</b>", tbl_cell_style),
            Paragraph(_safe_str(metadata.generated_at), tbl_cell_style),
        ],
    ]
    t_meta = Table(meta_data, colWidths=[100, 166, 100, 166])
    t_meta.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), light_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # --------------------------------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY & DATA QUALITY SCORECARD
    # --------------------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Data Quality Scorecard", h1_style))

    if quality_scorecard:
        score_val = quality_scorecard.composite_score
        grade_val = quality_scorecard.grade

        badge_bg = (
            success_green if score_val >= 85 else
            (warning_orange if score_val >= 70 else danger_red)
        )

        dqs_banner_data = [
            [
                Paragraph("<b>COMPOSITE QUALITY SCORE (DQS)</b>", tbl_hdr_style),
                Paragraph("<b>QUALITY GRADE</b>", tbl_hdr_style),
                Paragraph("<b>DATASET VOLUME</b>", tbl_hdr_style),
                Paragraph("<b>CRITICAL ANOMALIES</b>", tbl_hdr_style),
            ],
            [
                Paragraph(f"<font size=14><b>{score_val:.1f} / 100</b></font>", tbl_hdr_style),
                Paragraph(f"<font size=14><b>{grade_val.upper()}</b></font>", tbl_hdr_style),
                Paragraph(f"<font size=12><b>{profile.row_count if profile else 'N/A'} rows</b></font>", tbl_hdr_style),
                Paragraph(
                    f"<font size=12><b>{anomaly_result.anomalies_detected_count if (anomaly_result and anomaly_result.success) else 0} flagged</b></font>",
                    tbl_hdr_style,
                ),
            ],
        ]
        t_banner = Table(dqs_banner_data, colWidths=[133, 133, 133, 133])
        t_banner.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), navy_color),
                ("BACKGROUND", (1, 1), (1, 1), badge_bg),
                ("BOX", (0, 0), (-1, -1), 1, navy_color),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(t_banner)
        story.append(Spacer(1, 6))

        # Dimensional Breakdown Table
        dim_data = [
            [
                Paragraph("<b>Quality Dimension</b>", tbl_hdr_style),
                Paragraph("<b>Score</b>", tbl_hdr_style),
                Paragraph("<b>Weight</b>", tbl_hdr_style),
                Paragraph("<b>Calculation & Measurement Summary</b>", tbl_hdr_style),
            ]
        ]
        for d_name, d_obj in quality_scorecard.dimensions.items():
            sc_text = f"{d_obj.score:.1f}%" if d_obj.score is not None else "N/A"
            w_text = f"{d_obj.normalized_weight * 100:.1f}% (raw: {d_obj.raw_weight:.0f})"
            dim_data.append([
                Paragraph(f"<b>{d_name.capitalize()}</b>", tbl_cell_style),
                Paragraph(sc_text, tbl_cell_style),
                Paragraph(w_text, tbl_cell_style),
                Paragraph(_safe_str(d_obj.description), tbl_cell_style),
            ])

        t_dim = Table(dim_data, colWidths=[100, 60, 92, 280])
        t_dim.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), steel_color),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ])
        )
        story.append(t_dim)
    else:
        story.append(Paragraph("Quality scorecard was not calculated.", body_style))

    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # SECTION 2: DATASET PROFILE & INTEGRITY
    # --------------------------------------------------------------------------
    story.append(Paragraph("2. Dataset Profiling & Structural Health", h1_style))

    if profile:
        prof_summary_data = [
            [
                Paragraph("<b>Total Rows:</b>", tbl_cell_style),
                Paragraph(f"{profile.row_count:,}", tbl_cell_style),
                Paragraph("<b>Missing Cells:</b>", tbl_cell_style),
                Paragraph(f"{profile.missing_cells:,} ({profile.missing_cells_pct:.2f}%)", tbl_cell_style),
            ],
            [
                Paragraph("<b>Total Columns:</b>", tbl_cell_style),
                Paragraph(f"{profile.column_count}", tbl_cell_style),
                Paragraph("<b>Duplicate Rows:</b>", tbl_cell_style),
                Paragraph(f"{profile.duplicate_rows:,} ({profile.duplicate_rows_pct:.2f}%)", tbl_cell_style),
            ],
            [
                Paragraph("<b>Numeric Columns:</b>", tbl_cell_style),
                Paragraph(f"{len(profile.numeric_columns)}", tbl_cell_style),
                Paragraph("<b>Empty Strings / Infs:</b>", tbl_cell_style),
                Paragraph(f"{profile.empty_strings_total} / {profile.infinite_values_total}", tbl_cell_style),
            ],
        ]
        t_prof_sum = Table(prof_summary_data, colWidths=[110, 156, 110, 156])
        t_prof_sum.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), soft_blue),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(t_prof_sum)
        story.append(Spacer(1, 6))

        # Top columns profile table
        col_table_data = [
            [
                Paragraph("<b>Column Name</b>", tbl_hdr_style),
                Paragraph("<b>Type</b>", tbl_hdr_style),
                Paragraph("<b>Null Count</b>", tbl_hdr_style),
                Paragraph("<b>Null %</b>", tbl_hdr_style),
                Paragraph("<b>Unique Values</b>", tbl_hdr_style),
                Paragraph("<b>Observed Summary</b>", tbl_hdr_style),
            ]
        ]
        for col_name, cp in list(profile.columns.items())[:12]:
            st = cp.stats or {}
            c_type = "Numeric" if cp.is_numeric else ("Categorical" if cp.is_categorical else "Object")
            if cp.is_numeric and "min" in st and "max" in st:
                summary_str = f"min={st['min']}, max={st['max']}, mean={st.get('mean', 0.0):.1f}"
            elif cp.is_categorical and "top_values" in st:
                top_v = list(st["top_values"].keys())[:2]
                summary_str = f"top: {', '.join(str(x) for x in top_v)}"
            else:
                summary_str = f"{cp.unique_count} distinct entries"

            col_table_data.append([
                Paragraph(f"<b>{_safe_str(col_name)}</b>", tbl_cell_style),
                Paragraph(c_type, tbl_cell_style),
                Paragraph(str(cp.missing_count), tbl_cell_style),
                Paragraph(f"{cp.missing_pct:.1f}%", tbl_cell_style),
                Paragraph(str(cp.unique_count), tbl_cell_style),
                Paragraph(_safe_str(summary_str), tbl_cell_style),
            ])

        t_col = Table(col_table_data, colWidths=[105, 55, 52, 50, 70, 200], repeatRows=1)
        t_col.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), navy_color),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(t_col)
    else:
        story.append(Paragraph("Dataset profile is unavailable.", body_style))

    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # SECTION 3: SCHEMA & DISTRIBUTION DRIFT ANALYSIS
    # --------------------------------------------------------------------------
    story.append(Paragraph("3. Schema & Distribution Drift Analysis", h1_style))

    if drift_result and drift_result.has_reference:
        verdict_color = danger_red if drift_result.dataset_drift_detected else success_green
        verdict_text = "DATASET DRIFT DETECTED" if drift_result.dataset_drift_detected else "DATASET STABLE (NO STATISTICAL DRIFT)"

        drift_hdr_p = Paragraph(
            f"<b>Overall Drift Status:</b> <font color='{verdict_color.hexval()}'><b>{verdict_text}</b></font> "
            f"({drift_result.drifted_columns_count} of {drift_result.total_columns_compared} columns shifted significantly at α = 0.05).",
            body_style,
        )
        story.append(drift_hdr_p)
        story.append(Spacer(1, 4))

        drift_tbl_data = [
            [
                Paragraph("<b>Column Name</b>", tbl_hdr_style),
                Paragraph("<b>Type</b>", tbl_hdr_style),
                Paragraph("<b>Statistical Test</b>", tbl_hdr_style),
                Paragraph("<b>Statistic</b>", tbl_hdr_style),
                Paragraph("<b>p-value</b>", tbl_hdr_style),
                Paragraph("<b>Drift?</b>", tbl_hdr_style),
            ]
        ]
        for col_name, cdr in drift_result.column_drift.items():
            stat_s = f"{cdr.statistic:.4f}" if cdr.statistic is not None else "-"
            pval_s = f"{cdr.p_value:.4f}" if cdr.p_value is not None else "-"
            drift_flag = "<font color='#D9534F'><b>YES</b></font>" if cdr.drift_detected else "<font color='#5CB85C'>NO</font>"
            if cdr.skipped:
                drift_flag = f"<font color='#5A6B82'>SKIP ({_safe_str(cdr.skip_reason)})</font>"

            drift_tbl_data.append([
                Paragraph(f"<b>{_safe_str(col_name)}</b>", tbl_cell_style),
                Paragraph(cdr.column_type, tbl_cell_style),
                Paragraph(_safe_str(cdr.method), tbl_cell_style),
                Paragraph(stat_s, tbl_cell_style),
                Paragraph(pval_s, tbl_cell_style),
                Paragraph(drift_flag, tbl_cell_style),
            ])

        t_drift = Table(drift_tbl_data, colWidths=[110, 52, 160, 60, 60, 90], repeatRows=1)
        t_drift.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), steel_color),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(t_drift)
    else:
        story.append(
            Paragraph(
                "<i>No baseline reference dataset was provided for comparison. "
                "Distribution drift evaluation requires a reference baseline to establish stationary empirical distributions.</i>",
                italic_style,
            )
        )

    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # SECTION 4: MULTIVARIATE ANOMALY DETECTION (ISOLATION FOREST)
    # --------------------------------------------------------------------------
    story.append(Paragraph("4. Multivariate Anomaly Detection (Isolation Forest)", h1_style))

    if anomaly_result and anomaly_result.success:
        anom_summary_text = (
            f"Model Mode: <b>{anomaly_result.training_mode.replace('_', ' ').title()}</b> | "
            f"Features Analyzed: <b>{len(anomaly_result.features_used)}</b> | "
            f"Contamination Rate: <b>{anomaly_result.contamination_parameter}</b> | "
            f"Suspicious Records Flagged: <b>{anomaly_result.anomalies_detected_count}</b> "
            f"({anomaly_result.anomaly_share * 100:.1f}%)"
        )
        story.append(Paragraph(anom_summary_text, body_style))
        story.append(Spacer(1, 4))

        # Top flagged anomalies table
        anom_tbl_data = [
            [
                Paragraph("<b>Row #</b>", tbl_hdr_style),
                Paragraph("<b>Status</b>", tbl_hdr_style),
                Paragraph("<b>Suspicion Score</b>", tbl_hdr_style),
                Paragraph("<b>Decision Score</b>", tbl_hdr_style),
                Paragraph("<b>Top Feature Deviations (|IQR| Units)</b>", tbl_hdr_style),
            ]
        ]
        top_records = [r for r in anomaly_result.anomaly_records if r.is_anomaly][:10]
        if not top_records:
            top_records = anomaly_result.anomaly_records[:5]

        for rec in top_records:
            st_flag = "<font color='#D9534F'><b>ANOMALY</b></font>" if rec.is_anomaly else "NORMAL"
            top_dev_str = ", ".join(
                f"{k}: {v:+.1f} IQR"
                for k, v in sorted(
                    rec.feature_deviations.items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]
            )
            anom_tbl_data.append([
                Paragraph(str(rec.row_index), tbl_cell_style),
                Paragraph(st_flag, tbl_cell_style),
                Paragraph(f"{rec.anomaly_score:.3f}", tbl_cell_style),
                Paragraph(f"{rec.raw_decision_score:.3f}", tbl_cell_style),
                Paragraph(_safe_str(top_dev_str or "Within expected range"), tbl_cell_style),
            ])

        t_anom = Table(anom_tbl_data, colWidths=[45, 65, 75, 75, 272], repeatRows=1)
        t_anom.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), navy_color),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(t_anom)
    else:
        err_msg = anomaly_result.error_message if anomaly_result else "No anomaly detection executed."
        story.append(Paragraph(f"<i>Anomaly detection was skipped: {_safe_str(err_msg)}</i>", italic_style))

    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # SECTION 5: DATA VALIDATION & RULE VIOLATIONS
    # --------------------------------------------------------------------------
    story.append(Paragraph("5. Schema & Business Rule Validation", h1_style))

    if validation_result and validation_result.rules_configured:
        val_summary_p = Paragraph(
            f"Rule Validation Status: <b>{'PASSED' if validation_result.is_valid else 'VIOLATIONS FOUND'}</b> | "
            f"Total Issues: <b>{validation_result.total_issues}</b> ("
            f"<font color='#D9534F'>Critical: {validation_result.critical_count}</font>, "
            f"<font color='#F0AD4E'>Error: {validation_result.error_count}</font>, "
            f"Warning: {validation_result.warning_count})",
            body_style,
        )
        story.append(val_summary_p)
        story.append(Spacer(1, 4))

        if validation_result.issues:
            issues_tbl_data = [
                [
                    Paragraph("<b>Sev</b>", tbl_hdr_style),
                    Paragraph("<b>Issue Type</b>", tbl_hdr_style),
                    Paragraph("<b>Column</b>", tbl_hdr_style),
                    Paragraph("<b>Row</b>", tbl_hdr_style),
                    Paragraph("<b>Violation Description</b>", tbl_hdr_style),
                ]
            ]
            for issue in validation_result.issues[:15]:
                sev = issue.severity.upper()
                sev_color = danger_red if sev == "CRITICAL" else (warning_orange if sev == "ERROR" else slate_gray)
                sev_p = Paragraph(f"<font color='{sev_color.hexval()}'><b>{sev[:4]}</b></font>", tbl_cell_style)

                issues_tbl_data.append([
                    sev_p,
                    Paragraph(_safe_str(issue.issue_type), tbl_cell_style),
                    Paragraph(f"<b>{_safe_str(issue.column)}</b>", tbl_cell_style),
                    Paragraph(str(issue.row_reference if issue.row_reference is not None else "-"), tbl_cell_style),
                    Paragraph(_safe_str(issue.description), tbl_cell_style),
                ])

            t_val = Table(issues_tbl_data, colWidths=[40, 95, 80, 37, 280], repeatRows=1)
            t_val.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), steel_color),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ])
            )
            story.append(t_val)
        else:
            story.append(Paragraph("All records fully satisfied configured Pandera schema validation rules.", body_style))
    else:
        story.append(
            Paragraph(
                "<i>No validation rules were configured or evaluated for this dataset. "
                "Validity dimension is not artificially scored.</i>",
                italic_style,
            )
        )

    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # SECTION 6: METHODOLOGY, LIMITATIONS & VIVA VOCE REFERENCE
    # --------------------------------------------------------------------------
    story.append(KeepTogether([
        Paragraph("6. Academic Methodology & Research Notes", h1_style),
        Paragraph(
            "<b>1. Data Quality Score (DQS) Formulation:</b> The composite indicator is calculated as a normalized linear weighted combination: "
            "<font name='Courier'>DQS = ∑ (w_i * S_i) / ∑ w_i</font>, where completeness (40%), uniqueness (25%), and validity (35%) "
            "are dynamically rescaled across available dimensions. DQS is an engineering heuristic, not an official ISO standard.",
            body_style,
        ),
        Paragraph(
            "<b>2. Statistical Drift Testing:</b> Numerical features are evaluated using the non-parametric Two-Sample Kolmogorov-Smirnov test "
            "(<font name='Courier'>D = sup |F_ref(x) - F_cur(x)|</font>). Categorical features are evaluated via Pearson's Chi-Square Test. "
            "Drift indicates distribution shift (covariate shift), which reflects temporal or operational variance rather than intrinsic corruption.",
            body_style,
        ),
        Paragraph(
            "<b>3. Isolation Forest Anomaly Detection:</b> Unsupervised recursive partitioning isolates sparse anomalies deeper/faster "
            "than nominal dense clusters (<font name='Courier'>s(x, n) = 2^(-E(h(x)) / c(n))</font>). Flagged rows represent suspicious "
            "multivariate candidates requiring human review, not verified data errors.",
            body_style,
        ),
        Paragraph(
            "<b>4. Data Governance & Privacy:</b> All parsing, imputation, and scoring occurred strictly in volatile process memory. "
            "No telemetry, third-party APIs, or cloud persistence mechanisms were engaged.",
            body_style,
        ),
    ]))

    # Build the document using the two-pass NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()


def save_report_to_disk(report_bytes: bytes, destination_path: Union[str, Path]) -> Path:
    """Save an in-memory report byte stream to a specified filesystem destination.

    Parameters
    ----------
    report_bytes : bytes
        Raw file bytes to write.
    destination_path : Union[str, Path]
        Target file path.

    Returns
    -------
    Path
        Resolved filesystem path of the saved file.
    """
    path = Path(destination_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(report_bytes)
    return path
