"""DataPulse — Keep Your Data Clean. Reliable. Ready.

An intelligent data quality monitoring tool that helps you understand, validate,
and improve your data. Upload your dataset to detect missing values, duplicates,
data inconsistencies, drift, and unusual patterns—all through a simple, interactive dashboard.

Your data. Our insights. Better decisions.

100% In-Memory Local Execution with Zero Disk Persistence.
"""

from __future__ import annotations

from datetime import datetime
import io
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import AnomalyDetectionResult, detect_anomalies
from src.data_loader import (
    DataLoaderError,
    EmptyDataError,
    MalformedDataError,
    UnsupportedFileFormatError,
    WorksheetNotFoundError,
    list_excel_sheets,
    load_dataset,
    read_csv_bytes,
    read_excel_bytes,
)
from src.drift_detection import DriftAnalysisResult, detect_drift
from src.profiling import DatasetProfile, profile_dataset
from src.quality_score import QualityScorecard, calculate_data_quality_score
from src.report_generator import (
    ReportMetadata,
    export_analysis_to_excel,
    export_analysis_to_pdf,
    export_anomalies_to_csv,
    export_drift_to_csv,
    export_issues_to_csv,
)
from src.ui_charts import (
    plot_anomaly_scatter,
    plot_anomaly_score_distribution,
    plot_dimension_radar,
    plot_drift_distribution_comparison,
    plot_missingness_bar,
    plot_quality_gauge,
)
from src.validation import ValidationResult, load_validation_rules, validate_dataset


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION (Must be first Streamlit call)
# ==============================================================================

st.set_page_config(
    page_title="DataPulse — Keep Your Data Clean. Reliable. Ready.",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# PROFESSIONAL DESIGN & CUSTOM STYLESHEET (Navy & Teal Palette)
# ==============================================================================

def get_application_css() -> str:
    """Return polished, professional CSS for the DataPulse interface.

    Palette:
    - Background: Crisp off-white (#F8FAFC)
    - Deep Navy: #0F172A (headers), #1E293B (cards, text)
    - Restrained Teal: #0D9488 (primary brand & action), #0F766E (hover), #F0FDFA (light tint)
    - Borders: #E2E8F0
    - Status: Green (#15803D), Amber (#B45309), Red (#B91C1C)
    """
    return """
    <style>
        /* Base typography & page resets */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1E293B;
        }

        /* Top Header Banner */
        .sentry-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            color: #FFFFFF;
            padding: 22px 28px;
            border-radius: 10px;
            margin-bottom: 24px;
            border-left: 6px solid #0D9488;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        }
        .sentry-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 14px;
        }
        .sentry-brand {
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .sentry-subtitle {
            font-size: 0.95rem;
            color: #94A3B8;
            margin-top: 5px;
            font-weight: 400;
        }
        .sentry-status-badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.84rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .status-empty {
            background-color: #334155;
            color: #E2E8F0;
        }
        .status-pending {
            background-color: #0369A1;
            color: #E0F2FE;
        }
        .status-audited {
            background-color: #0D9488;
            color: #FFFFFF;
        }

        /* Metric Cards */
        .sentry-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: border-color 0.15s ease-in-out;
        }
        .sentry-card:hover {
            border-color: #CBD5E1;
        }
        .sentry-card-primary {
            border-left: 4px solid #0D9488;
        }
        .sentry-metric-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 4px;
        }
        .sentry-metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
        }
        .sentry-metric-sub {
            font-size: 0.82rem;
            color: #64748B;
            margin-top: 4px;
        }

        /* Informative Callout Boxes */
        .sentry-callout {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #0D9488;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 14px 0;
            font-size: 0.90rem;
            color: #1E293B;
            line-height: 1.55;
        }
        .sentry-callout-info {
            background-color: #F0FDFA;
            border-left-color: #0D9488;
            color: #115E59;
        }
        .sentry-callout-warning {
            background-color: #FFFBEB;
            border-left-color: #F59E0B;
            color: #92400E;
        }
        .sentry-callout-danger {
            background-color: #FEF2F2;
            border-left-color: #EF4444;
            color: #991B1B;
        }

        /* Inline Status Tags */
        .badge-green {
            background-color: #DCFCE7;
            color: #15803D;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.80rem;
            display: inline-block;
        }
        .badge-amber {
            background-color: #FEF3C7;
            color: #B45309;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.80rem;
            display: inline-block;
        }
        .badge-red {
            background-color: #FEE2E2;
            color: #B91C1C;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.80rem;
            display: inline-block;
        }
        .badge-blue {
            background-color: #E0F2FE;
            color: #0369A1;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.80rem;
            display: inline-block;
        }

        /* Streamlit Button Tweaks (Teal Primary Accent) */
        div.stButton > button[kind="primary"] {
            background-color: #0D9488 !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            padding: 0.5rem 1.25rem !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #0F766E !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
        }
    </style>
    """


# ==============================================================================
# SESSION STATE MANAGEMENT & SAFE STATE TRANSITIONS
# ==============================================================================

def init_session_state() -> None:
    """Initialize all persistent state variables with defensive defaults."""
    defaults: Dict[str, Any] = {
        "nav_section": "📊 Overview",
        "current_df": None,
        "current_meta": {
            "name": "No dataset loaded",
            "sheet": None,
            "source": None,
            "rows": 0,
            "cols": 0,
            "memory_kb": 0.0,
        },
        "reference_df": None,
        "reference_meta": {
            "name": None,
            "sheet": None,
            "source": None,
            "rows": 0,
            "cols": 0,
        },
        "profile": None,
        "validation_result": None,
        "quality_scorecard": None,
        "drift_result": None,
        "anomaly_result": None,
        "dqs_weights": {"completeness": 40.0, "uniqueness": 25.0, "validity": 35.0},
        "drift_alpha": 0.05,
        "anomaly_contamination": 0.05,
        "last_audit_timestamp": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def set_current_dataset(
    df: pd.DataFrame,
    name: str,
    sheet: Optional[str] = None,
    source: str = "upload",
) -> None:
    """Safely store the current production dataset and invalidate downstream stale results."""
    st.session_state["current_df"] = df
    st.session_state["current_meta"] = {
        "name": name,
        "sheet": sheet,
        "source": source,
        "rows": len(df),
        "cols": df.shape[1],
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }
    # Invalidate stale downstream results
    st.session_state["profile"] = None
    st.session_state["validation_result"] = None
    st.session_state["quality_scorecard"] = None
    st.session_state["drift_result"] = None
    st.session_state["anomaly_result"] = None
    st.session_state["last_audit_timestamp"] = None


def set_reference_dataset(
    df: Optional[pd.DataFrame],
    name: Optional[str] = None,
    sheet: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    """Store the reference baseline dataset and invalidate drift and reference-trained anomaly results."""
    st.session_state["reference_df"] = df
    if df is not None:
        st.session_state["reference_meta"] = {
            "name": name or "reference_baseline.csv",
            "sheet": sheet,
            "source": source or "upload",
            "rows": len(df),
            "cols": df.shape[1],
        }
    else:
        st.session_state["reference_meta"] = {
            "name": None,
            "sheet": None,
            "source": None,
            "rows": 0,
            "cols": 0,
        }
    # Invalidate drift result
    st.session_state["drift_result"] = None
    # If anomaly model was reference-trained, invalidate it
    anom = st.session_state.get("anomaly_result")
    if anom is not None and getattr(anom, "training_mode", None) == "reference_trained":
        st.session_state["anomaly_result"] = None


def reset_all_state() -> None:
    """Clear all loaded datasets, models, and analytical artifacts from memory."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


def load_demo_benchmark() -> bool:
    """Load the deterministic synthetic benchmark files into session state."""
    demo_dir = PROJECT_ROOT / "data" / "demo"
    cur_file = demo_dir / "current.csv"
    ref_file = demo_dir / "baseline.csv"

    try:
        if cur_file.exists() and ref_file.exists():
            c_df = read_csv_bytes(cur_file.read_bytes())
            r_df = read_csv_bytes(ref_file.read_bytes())
        else:
            from src.synthetic_generator import generate_baseline_dataset, generate_current_dataset
            r_df = generate_baseline_dataset(500, seed=42)
            c_df, _ = generate_current_dataset(500, seed=101)

        set_current_dataset(c_df, name="demo_current.csv", source="demo")
        set_reference_dataset(r_df, name="demo_baseline.csv", source="demo")
        return True
    except Exception as e:
        st.error(f"Failed to load demo data: {e}")
        return False


# ==============================================================================
# PIPELINE ORCHESTRATOR (Preserved for tests & automated execution)
# ==============================================================================

def run_analysis_pipeline(
    current_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame],
    weights: Dict[str, float],
    alpha: float,
    contamination: float,
    current_name: str = "current_dataset",
    reference_name: Optional[str] = None,
    current_sheet: Optional[str] = None,
    reference_sheet: Optional[str] = None,
    app_title: str = "DataPulse",
    app_subtitle: str = "Keep Your Data Clean. Reliable. Ready.",
) -> Dict[str, Any]:
    """Execute the end-to-end analytical workflow across all modules.

    Maintained with backward-compatible signature for test suites.
    """
    # 1. Profile current dataset
    profile = profile_dataset(current_df)

    # 2. Validate against declarative rules
    rules_path = PROJECT_ROOT / "config" / "validation_rules.json"
    rules = load_validation_rules(rules_path) if rules_path.exists() else None
    validation_result = validate_dataset(current_df, rules)

    # 3. Compute Data Quality Score
    quality_scorecard = calculate_data_quality_score(
        current_df, profile, validation_result, custom_weights=weights
    )

    # 4. Statistical Distribution & Schema Drift (if reference provided)
    drift_result = None
    if reference_df is not None:
        drift_result = detect_drift(reference_df, current_df, alpha=alpha)

    # 5. Unsupervised Multivariate Anomaly Detection (Isolation Forest)
    anomaly_result = detect_anomalies(
        current_df=current_df,
        reference_df=reference_df,
        contamination=contamination,
        random_state=42,
    )

    # 6. Context Metadata
    metadata = ReportMetadata(
        dataset_name=current_name,
        reference_name=reference_name,
        worksheet_name=current_sheet,
        reference_worksheet_name=reference_sheet,
        author=app_title,
        academic_context=app_subtitle,
    )

    return {
        "profile": profile,
        "validation_result": validation_result,
        "quality_scorecard": quality_scorecard,
        "drift_result": drift_result,
        "anomaly_result": anomaly_result,
        "metadata": metadata,
    }


# ==============================================================================
# HEADER COMPONENT (Recognizable branding, purpose & state badge)
# ==============================================================================

def render_application_header() -> None:
    """Render the application header with DataPulse branding and real-time state badge."""
    current_df = st.session_state["current_df"]
    dqs: Optional[QualityScorecard] = st.session_state["quality_scorecard"]
    c_meta = st.session_state["current_meta"]

    if current_df is None:
        badge_html = "<span class='sentry-status-badge status-empty'>● No Dataset Loaded</span>"
    elif dqs is None:
        badge_html = (
            f"<span class='sentry-status-badge status-pending'>"
            f"● Loaded: {c_meta['name']} ({c_meta['rows']:,} rows) — Pending Audit</span>"
        )
    else:
        badge_html = (
            f"<span class='sentry-status-badge status-audited'>"
            f"● Audited: DQS {dqs.composite_score:.1f}/100 ({dqs.grade.upper()}) | {c_meta['rows']:,} rows</span>"
        )

    st.markdown(
        f"""
        <div class="sentry-header">
            <div class="sentry-header-row">
                <div>
                    <div class="sentry-brand">⚡ DataPulse</div>
                    <div style="font-weight: 700; color: #5EEAD4; margin-top: 3px; font-size: 1.05rem; letter-spacing: -0.2px;">
                        Keep Your Data Clean. Reliable. Ready.
                    </div>
                    <div class="sentry-subtitle" style="max-width: 950px; margin-top: 6px; line-height: 1.5; color: #CBD5E1;">
                        DataPulse is an intelligent data quality monitoring tool that helps you understand, validate, 
                        and improve your data. Upload your dataset to detect missing values, duplicates, data inconsistencies, 
                        drift, and unusual patterns—all through a simple, interactive dashboard.
                    </div>
                    <div style="font-weight: 700; color: #F8FAFC; margin-top: 6px; font-size: 0.92rem;">
                        Your data. Our insights. Better decisions.
                    </div>
                </div>
                <div style="align-self: flex-start;">{badge_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SIDEBAR NAVIGATION COMPONENT
# ==============================================================================

def render_sidebar_navigation() -> str:
    """Render the sidebar navigation, ingestion telemetry, and reset control."""
    with st.sidebar:
        st.markdown("### ⚡ **DataPulse**")
        st.caption("Keep Your Data Clean. Reliable. Ready.")

        st.markdown("---")

        sections = [
            "📊 Overview",
            "📥 Data Input",
            "🔍 Quality Audit",
            "📈 Drift Detection",
            "🧠 Anomaly Detection",
            "📑 Reports & Downloads",
        ]

        # Sync with session state if redirected
        current_idx = 0
        if st.session_state["nav_section"] in sections:
            current_idx = sections.index(st.session_state["nav_section"])

        selected_section = st.radio(
            "Navigation",
            sections,
            index=current_idx,
            key="sidebar_nav_radio",
            label_visibility="collapsed",
        )
        st.session_state["nav_section"] = selected_section

        st.markdown("---")

        # Telemetry Summary
        st.markdown("##### 📁 Pipeline Telemetry")
        c_df = st.session_state["current_df"]
        r_df = st.session_state["reference_df"]
        dqs = st.session_state["quality_scorecard"]
        drift = st.session_state["drift_result"]
        anom = st.session_state["anomaly_result"]

        if c_df is not None:
            c_meta = st.session_state["current_meta"]
            st.markdown(f"**Target:** `{c_meta['name']}`")
            st.caption(f"{c_meta['rows']:,} rows × {c_meta['cols']} cols ({c_meta['memory_kb']} KB)")
        else:
            st.caption("Target: *No dataset loaded*")

        if r_df is not None:
            r_meta = st.session_state["reference_meta"]
            st.markdown(f"**Baseline:** `{r_meta['name']}`")
            st.caption(f"{r_meta['rows']:,} rows × {r_meta['cols']} cols")
        else:
            st.caption("Baseline: *None (Single dataset mode)*")

        st.markdown("##### ⚙️ Status Checks")
        q_status = "✅ Audited" if dqs else "⏳ Pending"
        d_status = "✅ Evaluated" if drift else ("⚠️ Needs Baseline" if r_df is None else "⏳ Ready")
        a_status = "✅ Flagged" if anom else "⏳ Ready"
        st.caption(f"• Quality Audit: **{q_status}**\n• Drift Monitoring: **{d_status}**\n• Anomaly Detection: **{a_status}**")

        st.markdown("---")

        if st.button("🔄 Reset Analysis", use_container_width=True, help="Clear loaded datasets and cached audit results"):
            reset_all_state()
            st.rerun()

        st.markdown(
            """
            <div style="font-size:0.75rem; color:#94A3B8; text-align:center; margin-top:20px;">
                🔒 100% In-Memory Processing<br>Zero Cloud Disk Persistence
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_section


# ==============================================================================
# SECTION 1: OVERVIEW PAGE
# ==============================================================================

def render_overview_page() -> None:
    """Render the executive Overview dashboard with KPI cards, charts, and findings."""
    current_df = st.session_state["current_df"]
    reference_df = st.session_state["reference_df"]
    profile: Optional[DatasetProfile] = st.session_state["profile"]
    val: Optional[ValidationResult] = st.session_state["validation_result"]
    dqs: Optional[QualityScorecard] = st.session_state["quality_scorecard"]
    drift: Optional[DriftAnalysisResult] = st.session_state["drift_result"]
    anom: Optional[AnomalyDetectionResult] = st.session_state["anomaly_result"]

    # EMPTY STATE
    if current_df is None:
        st.markdown(
            """
            <div class="sentry-callout sentry-callout-info">
                <h3 style="margin-top:0; color:#0F766E;">⚡ Welcome to DataPulse</h3>
                <p style="font-size: 1.05rem; font-weight: 700; color: #0D9488; margin-bottom: 6px;">
                    Keep Your Data Clean. Reliable. Ready.
                </p>
                <p style="font-size: 0.96rem; line-height: 1.6; color: #1E293B;">
                    DataPulse is an intelligent data quality monitoring tool that helps you understand, validate, 
                    and improve your data. Upload your dataset to detect missing values, duplicates, data inconsistencies, 
                    drift, and unusual patterns—all through a simple, interactive dashboard.
                </p>
                <p style="font-weight: 700; color: #0F766E; margin-bottom: 14px; font-size: 0.95rem;">
                    Your data. Our insights. Better decisions.
                </p>
                <hr style="border: 0; border-top: 1px solid #CCFBF1; margin: 12px 0;">
                <p style="font-weight: 600; color: #334155; margin-bottom: 6px;">Key Observability Layers:</p>
                <ol style="margin-bottom:8px;">
                    <li><b>Deterministic Syntactic & Schema Checks:</b> Catches missingness, duplicates, and rule violations using <b>Pandera</b>.</li>
                    <li><b>Statistical Distribution Drift:</b> Measures covariate shift against historical baselines using <b>Kolmogorov-Smirnov</b> and <b>Chi-Square</b> tests.</li>
                    <li><b>Unsupervised Multivariate Anomalies:</b> Flags high-dimensional outliers using <b>Isolation Forest</b>.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            if st.button("🚀 Quick Load Benchmark Demo Data", type="primary", use_container_width=True):
                if load_demo_benchmark():
                    st.success("✅ Synthetic demo benchmark data loaded successfully!")
                    st.rerun()
        with col_b2:
            st.caption("Loads 508 Target rows (`current.csv`) and 500 Reference rows (`baseline.csv`) with planted defects.")

        st.info("💡 To upload your own CSV or multi-sheet Excel file (.xlsx), navigate to **📥 Data Input** in the sidebar.")
        return

    # DATASET STATUS SECTION
    c_meta = st.session_state["current_meta"]
    r_meta = st.session_state["reference_meta"]
    st.subheader("System Health & Observability Overview")

    # If dataset loaded but not audited yet, offer 1-click audit
    if dqs is None or profile is None or val is None:
        st.markdown(
            f"""
            <div class="sentry-callout sentry-callout-warning">
                <b>Dataset Loaded:</b> <code>{c_meta['name']}</code> ({c_meta['rows']:,} rows, {c_meta['cols']} columns).<br>
                A comprehensive data quality audit has not been executed yet. Click below to initiate profiling and Pandera schema validation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("▶ Run Full Quality Audit Now", type="primary"):
            with st.spinner("Executing data profiling, declarative schema validation, and DQS scoring..."):
                prof = profile_dataset(current_df)
                rules_path = PROJECT_ROOT / "config" / "validation_rules.json"
                rules = load_validation_rules(rules_path) if rules_path.exists() else None
                v_res = validate_dataset(current_df, rules)
                scorecard = calculate_data_quality_score(
                    current_df, prof, v_res, custom_weights=st.session_state["dqs_weights"]
                )
                st.session_state["profile"] = prof
                st.session_state["validation_result"] = v_res
                st.session_state["quality_scorecard"] = scorecard
                st.session_state["last_audit_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.rerun()
        return

    # 4 MEANINGFUL METRIC CARDS
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        sc_badge = "badge-green" if dqs.composite_score >= 85 else ("badge-amber" if dqs.composite_score >= 70 else "badge-red")
        st.markdown(
            f"""
            <div class="sentry-card sentry-card-primary">
                <div class="sentry-metric-label">Data Quality Score (DQS)</div>
                <div class="sentry-metric-value">{dqs.composite_score:.1f}<span style="font-size:1rem; color:#64748B;">/100</span></div>
                <div class="sentry-metric-sub">Grade: <span class="{sc_badge}">{dqs.grade.upper()}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        null_badge = "badge-green" if profile.missing_cells == 0 else "badge-amber"
        st.markdown(
            f"""
            <div class="sentry-card sentry-card-primary">
                <div class="sentry-metric-label">Missing Values</div>
                <div class="sentry-metric-value">{profile.missing_cells:,}</div>
                <div class="sentry-metric-sub"><span class="{null_badge}">{profile.missing_cells_pct:.1f}%</span> of total cells</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        dup_badge = "badge-green" if profile.duplicate_rows == 0 else "badge-amber"
        st.markdown(
            f"""
            <div class="sentry-card sentry-card-primary">
                <div class="sentry-metric-label">Duplicate Rows</div>
                <div class="sentry-metric-value">{profile.duplicate_rows:,}</div>
                <div class="sentry-metric-sub"><span class="{dup_badge}">{profile.duplicate_rows_pct:.1f}%</span> redundant records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        if anom and anom.success:
            anom_text = f"{anom.anomalies_detected_count} ({anom.anomaly_share*100:.1f}%)"
            anom_badge = "badge-red" if anom.anomalies_detected_count > 0 else "badge-green"
            anom_sub = f"<span class='{anom_badge}'>{anom.anomalies_detected_count} flagged</span> (Isolation Forest)"
        else:
            anom_text = "Pending"
            anom_sub = "Run Anomaly Detection"

        st.markdown(
            f"""
            <div class="sentry-card sentry-card-primary">
                <div class="sentry-metric-label">Detected Anomalies</div>
                <div class="sentry-metric-value">{anom_text}</div>
                <div class="sentry-metric-sub">{anom_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # CHARTS ROW (Gauge & Radar)
    ch_col1, ch_col2 = st.columns([1, 1])
    with ch_col1:
        st.plotly_chart(plot_quality_gauge(dqs.composite_score, dqs.grade), use_container_width=True)
    with ch_col2:
        st.plotly_chart(plot_dimension_radar(dqs.dimensions), use_container_width=True)

    # DISTRIBUTION / DATA DRIFT SUMMARY (If reference comparison exists)
    if drift is not None and drift.has_reference:
        st.markdown("---")
        st.markdown("#### 📈 Distribution Drift Status")
        d_badge = "badge-red" if drift.dataset_drift_detected else "badge-green"
        d_text = "DATASET DRIFT DETECTED" if drift.dataset_drift_detected else "STABLE (NO DRIFT)"
        st.markdown(
            f"Overall Verdict: <span class='{d_badge}'>{d_text}</span> &nbsp;|&nbsp; "
            f"<b>{drift.drifted_columns_count} of {drift.total_columns_compared}</b> columns shifted significantly.",
            unsafe_allow_html=True,
        )

    # RECENT FINDINGS TABLE
    st.markdown("---")
    st.markdown("#### 🚨 Key Quality Findings & Rule Violations")
    if val.issues:
        recent_issues = val.issues[:8]
        issue_rows = []
        for idx, iss in enumerate(recent_issues, 1):
            issue_rows.append({
                "#": idx,
                "Severity": iss.severity.upper(),
                "Column": iss.column,
                "Row Ref": iss.row_reference if iss.row_reference is not None else "Dataset",
                "Description": iss.description,
            })
        st.dataframe(pd.DataFrame(issue_rows), use_container_width=True)
        if len(val.issues) > 8:
            st.caption(f"Showing top 8 of {len(val.issues)} issues. Navigate to **Quality Audit** for full filtered list.")
    else:
        st.success("✅ Zero schema violations detected across all checked validation constraints.")

    # WORKFLOW GUIDANCE ACTIONS
    st.markdown("---")
    st.markdown("#### 🧭 Recommended Next Actions")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        if st.button("🔍 Deep-Dive Quality", use_container_width=True):
            st.session_state["nav_section"] = "🔍 Quality Audit"
            st.rerun()
    with g2:
        if st.button("📈 Run Drift Analysis", use_container_width=True):
            st.session_state["nav_section"] = "📈 Drift Detection"
            st.rerun()
    with g3:
        if st.button("🧠 Detect Outliers", use_container_width=True):
            st.session_state["nav_section"] = "🧠 Anomaly Detection"
            st.rerun()
    with g4:
        if st.button("📑 Download Reports", use_container_width=True):
            st.session_state["nav_section"] = "📑 Reports & Downloads"
            st.rerun()


# ==============================================================================
# SECTION 2: DATA INPUT PAGE
# ==============================================================================

def render_data_input_page() -> None:
    """Render the data ingestion, worksheet selector, and inspection view."""
    st.subheader("Data Ingestion & Invariant Inspection")
    st.markdown(
        "Upload your target tabular dataset (and optional historical baseline reference) "
        "or load the synthetic benchmark dataset."
    )

    # Top Action Buttons
    act_col1, act_col2 = st.columns([1, 1])
    with act_col1:
        if st.button("🚀 Load Demo Benchmark Data", type="primary", use_container_width=True):
            if load_demo_benchmark():
                st.success("✅ Demo datasets loaded: Target (508 rows) & Baseline (500 rows).")
                st.rerun()
    with act_col2:
        if st.button("🔄 Clear All Datasets", use_container_width=True):
            reset_all_state()
            st.info("Analysis state and datasets cleared.")
            st.rerun()

    st.markdown("---")

    # Ingestion Columns
    u_col1, u_col2 = st.columns(2)

    with u_col1:
        st.markdown("#### 1. Target Dataset (Required for Audit)")
        target_file = st.file_uploader(
            "Upload Target Dataset (.csv, .xlsx)",
            type=["csv", "xlsx"],
            key="uploader_target",
            help="Current production or testing dataset to profile, validate, and score.",
        )
        if target_file is not None:
            t_name = target_file.name
            t_bytes = target_file.getvalue()
            t_sheet = None
            try:
                if t_name.lower().endswith(".xlsx"):
                    sheets = list_excel_sheets(t_bytes)
                    if len(sheets) > 1:
                        t_sheet = st.selectbox("Select Target Worksheet", sheets, key="target_sheet_picker")
                    else:
                        t_sheet = sheets[0]
                    parsed_df = read_excel_bytes(t_bytes, sheet_name=t_sheet, filename=t_name)
                else:
                    parsed_df = read_csv_bytes(t_bytes, filename=t_name)

                # Store if new or modified
                if st.session_state["current_df"] is None or st.session_state["current_meta"]["name"] != t_name or st.session_state["current_meta"]["sheet"] != t_sheet:
                    set_current_dataset(parsed_df, name=t_name, sheet=t_sheet, source="upload")
                    st.success(f"Loaded: `{t_name}` ({len(parsed_df)} rows, {parsed_df.shape[1]} columns)")
                    st.rerun()
            except DataLoaderError as e:
                st.error(f"Data loading error: {e}")
            except Exception as e:
                st.error(f"Unexpected error parsing file: {e}")

    with u_col2:
        st.markdown("#### 2. Baseline Reference (Optional for Drift)")
        ref_file = st.file_uploader(
            "Upload Reference Baseline (.csv, .xlsx)",
            type=["csv", "xlsx"],
            key="uploader_reference",
            help="Clean training or historical dataset used to compare distributions and detect drift.",
        )
        if ref_file is not None:
            r_name = ref_file.name
            r_bytes = ref_file.getvalue()
            r_sheet = None
            try:
                if r_name.lower().endswith(".xlsx"):
                    r_sheets = list_excel_sheets(r_bytes)
                    if len(r_sheets) > 1:
                        r_sheet = st.selectbox("Select Baseline Worksheet", r_sheets, key="ref_sheet_picker")
                    else:
                        r_sheet = r_sheets[0]
                    ref_parsed_df = read_excel_bytes(r_bytes, sheet_name=r_sheet, filename=r_name)
                else:
                    ref_parsed_df = read_csv_bytes(r_bytes, filename=r_name)

                if st.session_state["reference_df"] is None or st.session_state["reference_meta"]["name"] != r_name or st.session_state["reference_meta"]["sheet"] != r_sheet:
                    set_reference_dataset(ref_parsed_df, name=r_name, sheet=r_sheet, source="upload")
                    st.info(f"Loaded Baseline: `{r_name}` ({len(ref_parsed_df)} rows, {ref_parsed_df.shape[1]} columns)")
                    st.rerun()
            except DataLoaderError as e:
                st.error(f"Baseline loading error: {e}")
            except Exception as e:
                st.error(f"Unexpected error parsing baseline: {e}")
        elif st.session_state["reference_df"] is not None and st.session_state["reference_meta"]["source"] == "demo":
            st.caption("Active Baseline: `demo_baseline.csv` (Synthetic Benchmark)")

    # DATASET PREVIEW & SCHEMA INSPECTION
    current_df = st.session_state["current_df"]
    if current_df is not None:
        c_meta = st.session_state["current_meta"]
        st.markdown("---")
        st.markdown(f"### 📋 Active Dataset: `{c_meta['name']}`")

        info_c1, info_c2, info_c3, info_c4 = st.columns(4)
        info_c1.metric("Row Count", f"{c_meta['rows']:,}")
        info_c2.metric("Column Count", f"{c_meta['cols']}")
        info_c3.metric("Memory Footprint", f"{c_meta['memory_kb']} KB")
        info_c4.metric("Worksheet", c_meta['sheet'] if c_meta['sheet'] else "N/A (CSV)")

        tab_preview, tab_schema = st.tabs(["📄 Scrollable Data Preview", "🏷️ Detected Column Types & Health"])
        with tab_preview:
            st.dataframe(current_df.head(50), use_container_width=True)
            st.caption(f"Showing first {min(50, len(current_df))} rows of {len(current_df)} total records.")

        with tab_schema:
            schema_data = []
            for col in current_df.columns:
                null_c = int(current_df[col].isna().sum())
                null_p = (null_c / len(current_df)) * 100
                schema_data.append({
                    "Column Name": col,
                    "Pandas Dtype": str(current_df[col].dtype),
                    "Inferred Kind": "Numeric" if pd.api.types.is_numeric_dtype(current_df[col]) else "Categorical/Text",
                    "Non-Null Count": len(current_df) - null_c,
                    "Null Count": null_c,
                    "Null Percentage": f"{null_p:.2f}%",
                })
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True)
    else:
        st.markdown("---")
        st.markdown(
            """
            <div class="sentry-callout sentry-callout-info">
                <b>No Dataset Ingested Yet:</b><br>
                Upload a CSV or XLSX file above, or click <b>"🚀 Load Demo Benchmark Data"</b> to explore 
                the monitoring capabilities immediately.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# SECTION 3: QUALITY AUDIT PAGE
# ==============================================================================

def render_quality_audit_page() -> None:
    """Render the declarative schema validation, profiling, and DQS scoring view."""
    st.subheader("Declarative Schema Validation & Data Quality Score (DQS)")

    current_df = st.session_state["current_df"]
    if current_df is None:
        st.warning("⚠️ No dataset loaded. Please go to **📥 Data Input** to upload a dataset or load demo data.")
        return

    # Configuration & Action Row
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        run_audit_btn = st.button("▶ Run Quality Audit", type="primary", use_container_width=True)

    with ctrl_col2:
        with st.expander("⚙️ Customize Quality Dimension Weights & Formula", expanded=False):
            st.markdown("Adjust relative importance of data quality dimensions in the composite DQS formula:")
            w_comp = st.slider("Completeness Weight", 0, 100, int(st.session_state["dqs_weights"]["completeness"]), 5)
            w_uniq = st.slider("Uniqueness Weight", 0, 100, int(st.session_state["dqs_weights"]["uniqueness"]), 5)
            w_val = st.slider("Validity Weight", 0, 100, int(st.session_state["dqs_weights"]["validity"]), 5)
            st.session_state["dqs_weights"] = {
                "completeness": float(w_comp),
                "uniqueness": float(w_uniq),
                "validity": float(w_val),
            }
            st.caption("Note: If schema validation rules are not configured, weights dynamically re-scale across available dimensions.")

    if run_audit_btn or st.session_state["quality_scorecard"] is None:
        with st.spinner("Profiling dataset structure and verifying Pandera constraints..."):
            prof = profile_dataset(current_df)
            rules_path = PROJECT_ROOT / "config" / "validation_rules.json"
            rules = load_validation_rules(rules_path) if rules_path.exists() else None
            v_res = validate_dataset(current_df, rules)
            scorecard = calculate_data_quality_score(
                current_df, prof, v_res, custom_weights=st.session_state["dqs_weights"]
            )
            st.session_state["profile"] = prof
            st.session_state["validation_result"] = v_res
            st.session_state["quality_scorecard"] = scorecard
            st.session_state["last_audit_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    profile: DatasetProfile = st.session_state["profile"]
    val: ValidationResult = st.session_state["validation_result"]
    dqs: QualityScorecard = st.session_state["quality_scorecard"]

    st.markdown("---")

    # Executive Scoreboard
    # dqs.dimensions values are DimensionScore objects — extract .score (float or None)
    def _dim_score(dqs, key: str) -> float:
        """Safely extract numeric score from a DimensionScore object or fallback."""
        dim = dqs.dimensions.get(key)
        if dim is None:
            return 0.0
        if hasattr(dim, "score"):
            return float(dim.score) if dim.score is not None else 0.0
        return float(dim) if dim is not None else 0.0

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Composite DQS", f"{dqs.composite_score:.1f} / 100", f"Grade: {dqs.grade.upper()}")
    with sc2:
        st.metric("Completeness Score", f"{_dim_score(dqs, 'completeness'):.1f}%", f"{profile.missing_cells} null cells")
    with sc3:
        st.metric("Uniqueness Score", f"{_dim_score(dqs, 'uniqueness'):.1f}%", f"{profile.duplicate_rows} duplicate rows")
    with sc4:
        v_dim = dqs.dimensions.get("validity")
        v_val = (float(v_dim.score) if hasattr(v_dim, "score") and v_dim.score is not None else None) if v_dim is not None else None
        v_str = f"{v_val:.1f}%" if v_val is not None else "N/A"
        st.metric("Validity Score", v_str, f"{val.total_issues} rule issues")

    # DQS Formula Callout
    w = st.session_state["dqs_weights"]
    w_sum = sum(w.values()) or 1.0
    st.markdown(
        f"""
        <div class="sentry-callout sentry-callout-info">
            <b>Mathematical Data Quality Score Formulation:</b><br>
            <code>DQS = w_comp × S_comp + w_uniq × S_uniq + w_val × S_val</code><br>
            Active Normalized Weights: Completeness = {w['completeness']/w_sum:.2f}, 
            Uniqueness = {w['uniqueness']/w_sum:.2f}, Validity = {w['validity']/w_sum:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Missingness & Duplicates Visualization
    st.markdown("#### Missingness & Duplicate Records Breakdown")
    m_col1, m_col2 = st.columns([3, 2])
    with m_col1:
        st.plotly_chart(plot_missingness_bar(profile), use_container_width=True)
    with m_col2:
        st.markdown("**Duplicate Records Inspection**")
        if profile.duplicate_rows > 0:
            st.warning(f"Detected **{profile.duplicate_rows}** identical rows ({profile.duplicate_rows_pct:.2f}%).")
            with st.expander("View Duplicate Row Preview"):
                dups = current_df[current_df.duplicated(keep=False)]
                st.dataframe(dups.head(20), use_container_width=True)
        else:
            st.success("✅ Zero duplicate records detected (100% row uniqueness).")

    # Declarative Schema Violations
    st.markdown("---")
    st.markdown("#### Declarative Schema Rule Violations (Pandera)")

    if val.rules_configured:
        st.markdown(
            f"Overall Status: **{'PASSED' if val.is_valid else 'VIOLATIONS DETECTED'}** | "
            f"Total Issues: **{val.total_issues}** ("
            f"<span class='badge-red'>{val.critical_count} Critical</span> "
            f"<span class='badge-amber'>{val.error_count} Error</span> "
            f"<span class='badge-green'>{val.warning_count} Warning</span>)",
            unsafe_allow_html=True,
        )

        if val.issues:
            # Interactive Filter Controls
            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                sev_filter = st.selectbox("Filter by Severity", ["ALL", "CRITICAL", "ERROR", "WARNING"])
            with f_col2:
                col_search = st.text_input("Filter by Column Name", "")

            filtered = val.issues
            if sev_filter != "ALL":
                filtered = [i for i in filtered if i.severity.upper() == sev_filter]
            if col_search.strip():
                filtered = [i for i in filtered if col_search.lower() in i.column.lower()]

            table_rows = []
            for idx, iss in enumerate(filtered, 1):
                table_rows.append({
                    "#": idx,
                    "Severity": iss.severity.upper(),
                    "Type": iss.issue_type,
                    "Column": iss.column,
                    "Row Ref": iss.row_reference if iss.row_reference is not None else "Column-Level",
                    "Observed Value": str(iss.observed_value) if iss.observed_value is not None else "-",
                    "Description": iss.description,
                })

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
            st.caption(f"Showing {len(table_rows)} issues matching selected filters.")

            # CSV Download Button
            csv_issues = export_issues_to_csv(val)
            st.download_button(
                label="⬇️ Download Quality Issues CSV",
                data=csv_issues,
                file_name=f"quality_issues_{st.session_state['current_meta']['name']}.csv",
                mime="text/csv",
            )
        else:
            st.success("✅ All records satisfied configured Pandera schema checks.")
    else:
        st.info("ℹ️ No validation rules were configured. Validity dimension was omitted from scoring.")


# ==============================================================================
# SECTION 4: DRIFT DETECTION PAGE
# ==============================================================================

def render_drift_detection_page() -> None:
    """Render the schema and statistical distribution drift monitoring view."""
    st.subheader("Statistical Distribution Drift & Schema Consistency")

    current_df = st.session_state["current_df"]
    reference_df = st.session_state["reference_df"]

    if current_df is None:
        st.warning("⚠️ No target dataset loaded. Please go to **📥 Data Input** to load data.")
        return

    # Check for reference dataset
    if reference_df is None:
        st.markdown(
            """
            <div class="sentry-callout sentry-callout-warning">
                <h4>⚠️ Baseline Reference Required for Drift Detection</h4>
                <p>
                    Data drift detection compares incoming production records against a clean historical baseline 
                    or training batch to discover <b>covariate shift</b>.
                </p>
                <p>
                    To enable drift analysis, please go to <b>📥 Data Input</b> and upload a reference baseline 
                    dataset or click <b>"🚀 Load Demo Benchmark Data"</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("▶ Compare Datasets (Disabled)", disabled=True)
        return

    # Reference is present
    c_meta = st.session_state["current_meta"]
    r_meta = st.session_state["reference_meta"]

    st.markdown(
        f"Comparing Target Dataset **`{c_meta['name']}`** ({len(current_df):,} rows) "
        f"against Baseline **`{r_meta['name']}`** ({len(reference_df):,} rows)."
    )

    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        run_drift_btn = st.button("▶ Compare Datasets", type="primary", use_container_width=True)
    with ctrl_col2:
        alpha = st.slider(
            "Significance Threshold (α)",
            0.01, 0.10, float(st.session_state["drift_alpha"]), 0.01,
            help="Threshold p-value: if p < α, the null hypothesis of identical distributions is rejected.",
        )
        st.session_state["drift_alpha"] = alpha

    if run_drift_btn or st.session_state["drift_result"] is None:
        with st.spinner("Computing Kolmogorov-Smirnov continuous & Chi-Square categorical tests..."):
            drift_res = detect_drift(reference_df, current_df, alpha=alpha)
            st.session_state["drift_result"] = drift_res

    drift: DriftAnalysisResult = st.session_state["drift_result"]

    st.markdown("---")

    # Overall Status Banner
    v_badge = "badge-red" if drift.dataset_drift_detected else "badge-green"
    v_text = "DATASET DRIFT DETECTED" if drift.dataset_drift_detected else "DATASET STABLE (NO STATISTICAL DRIFT)"
    st.markdown(
        f"### Overall Status: <span class='{v_badge}'>{v_text}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Evaluated with threshold α = {alpha}. "
        f"Drift triggered: {drift.drifted_columns_count} of {drift.total_columns_compared} columns shifted significantly."
    )

    st.markdown(
        """
        <div class="sentry-callout sentry-callout-info">
            <b>Educational Note on Data Drift:</b><br>
            A detected drift represents a change in real-world feature distribution (covariate shift) over time. 
            <b>Drift is not necessarily a data error or corrupted file</b>; it informs engineers that downstream 
            machine learning models may suffer degraded performance due to changing input environments.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Schema Consistency
    sd = drift.schema_diff
    has_changes = getattr(sd, "has_schema_changes", getattr(sd, "schema_has_drift", False))
    missing_cols = getattr(sd, "missing_columns", [])
    added_cols = getattr(sd, "added_columns", getattr(sd, "new_columns", []))
    type_chgs = getattr(sd, "type_changes", getattr(sd, "type_mismatches", {}))

    if has_changes:
        st.error("⚠️ Schema alterations detected between baseline and target data!")
        if missing_cols:
            st.write(f"- Dropped Columns: `{', '.join(missing_cols)}`")
        if added_cols:
            st.write(f"- New Columns: `{', '.join(added_cols)}`")
        if type_chgs:
            st.write(f"- Type Mismatches: `{type_chgs}`")
    else:
        st.success("✅ Schema structures match: Identical column names and compatible data types.")

    # Column Drift Table
    st.markdown("---")
    st.markdown("#### Column-Level Statistical Hypothesis Tests")

    d_filter = st.radio("Filter Columns", ["All Columns", "Drifted Only (🚨)", "Stable Only (✅)"], horizontal=True)

    rows = []
    for col_name, cdr in drift.column_drift.items():
        if d_filter == "Drifted Only (🚨)" and not cdr.drift_detected:
            continue
        if d_filter == "Stable Only (✅)" and cdr.drift_detected:
            continue

        rows.append({
            "Column": col_name,
            "Type": cdr.column_type.capitalize(),
            "Method": cdr.method,
            "Statistic": f"{cdr.statistic:.4f}" if cdr.statistic is not None else "-",
            "p-Value": f"{cdr.p_value:.4f}" if cdr.p_value is not None else "-",
            "Threshold (α)": cdr.threshold_alpha,
            "Drift Detected": "🚨 YES" if cdr.drift_detected else "✅ NO",
            "Status": cdr.skip_reason if cdr.skipped else "Evaluated",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Interactive Distribution Explorer
    st.markdown("---")
    st.markdown("#### Interactive Distribution Explorer")
    available_cols = list(drift.column_drift.keys())
    if available_cols:
        inspect_col = st.selectbox("Select Feature to Inspect Distribution", available_cols)
        col_type = drift.column_drift[inspect_col].column_type
        fig = plot_drift_distribution_comparison(reference_df, current_df, inspect_col, is_numeric=(col_type == "numeric"))
        st.plotly_chart(fig, use_container_width=True)

    # CSV Download Button
    csv_drift = export_drift_to_csv(drift)
    st.download_button(
        label="⬇️ Download Drift Results CSV",
        data=csv_drift,
        file_name=f"drift_results_{c_meta['name']}.csv",
        mime="text/csv",
    )


# ==============================================================================
# SECTION 5: ANOMALY DETECTION PAGE
# ==============================================================================

def render_anomaly_detection_page() -> None:
    """Render the unsupervised multivariate Isolation Forest outlier detection view."""
    st.subheader("Unsupervised Multivariate Anomaly Detection (Isolation Forest)")

    current_df = st.session_state["current_df"]
    reference_df = st.session_state["reference_df"]

    if current_df is None:
        st.warning("⚠️ No target dataset loaded. Please go to **📥 Data Input** to load data.")
        return

    # Check for numeric columns
    num_cols = current_df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) == 0:
        st.warning("⚠️ The current dataset contains no numeric features. Isolation Forest requires continuous numerical variables.")
        return

    # Controls & Parameters
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        run_anom_btn = st.button("▶ Run Anomaly Detection", type="primary", use_container_width=True)
    with ctrl_col2:
        contamination = st.slider(
            "Expected Contamination Rate",
            0.01, 0.20, float(st.session_state["anomaly_contamination"]), 0.01,
            help="Estimated prior proportion of multivariate outliers in the dataset (default: 0.05).",
        )
        st.session_state["anomaly_contamination"] = contamination

    st.markdown(
        """
        <div class="sentry-callout sentry-callout-info">
            <b>Understanding the Contamination Parameter:</b><br>
            Contamination represents the expected fraction of outliers in the data. Setting it to <code>0.05</code> 
            configures the algorithm's decision threshold to flag approximately the top 5% most isolated records. 
            <b>Flagged observations are statistical multivariate anomalies requiring human review, not confirmed data corruption.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if run_anom_btn or st.session_state["anomaly_result"] is None:
        with st.spinner("Partitioning feature space with Isolation Forest tree ensemble..."):
            anom_res = detect_anomalies(
                current_df=current_df,
                reference_df=reference_df,
                contamination=contamination,
                random_state=42,
            )
            st.session_state["anomaly_result"] = anom_res

    anom: AnomalyDetectionResult = st.session_state["anomaly_result"]

    if not anom.success:
        st.error(f"Anomaly detection failed: {anom.error_message}")
        return

    st.markdown("---")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training Mode", anom.training_mode.replace("_", " ").title())
    m2.metric("Features Used", len(anom.features_used))
    m3.metric("Contamination Rate", f"{anom.contamination_parameter}")
    m4.metric("Flagged Outliers", f"{anom.anomalies_detected_count} ({anom.anomaly_share*100:.1f}%)")

    # Suspicion Score Histogram
    st.plotly_chart(plot_anomaly_score_distribution(anom), use_container_width=True)

    # 2D Feature Projection Scatter
    st.markdown("#### 2D Multivariate Feature Projection")
    if len(anom.features_used) >= 2:
        sc1, sc2 = st.columns(2)
        with sc1:
            x_col = st.selectbox("X-Axis Feature", anom.features_used, index=0)
        with sc2:
            y_col = st.selectbox("Y-Axis Feature", anom.features_used, index=min(1, len(anom.features_used) - 1))
        fig_scatter = plot_anomaly_scatter(current_df, anom, x_col, y_col)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("At least two numerical features are required for 2D scatter visualization.")

    # Filtered Anomaly Table
    st.markdown("---")
    st.markdown("#### Suspicious Observations Ranked by Suspicion Score")
    min_score = st.slider("Filter by Minimum Suspicion Score s(x, n)", 0.0, 1.0, 0.50, 0.02)

    table_rows = []
    for rec in anom.anomaly_records:
        if rec.anomaly_score < min_score:
            continue
        top_dev = "; ".join(
            f"{k}: {v:+.2f} IQR"
            for k, v in sorted(rec.feature_deviations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        )
        table_rows.append({
            "Row #": rec.row_index,
            "Verdict": "🚨 ANOMALY" if rec.is_anomaly else "NORMAL",
            "Suspicion Score (0-1)": round(rec.anomaly_score, 4),
            "Decision Score": round(rec.raw_decision_score, 4),
            "Top Feature Deviations": top_dev,
            "Feature Values": str(rec.record_data),
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    st.caption(f"Showing {len(table_rows)} observations with suspicion score ≥ {min_score:.2f}.")

    # CSV Download Button
    csv_anom = export_anomalies_to_csv(anom)
    st.download_button(
        label="⬇️ Download Flagged Anomalies CSV",
        data=csv_anom,
        file_name=f"anomalies_{st.session_state['current_meta']['name']}.csv",
        mime="text/csv",
    )


# ==============================================================================
# SECTION 6: REPORTS & DOWNLOADS PAGE
# ==============================================================================

def render_reports_page() -> None:
    """Render the multi-format audit export center with real download buttons."""
    st.subheader("Multi-Format Audit Export Center")
    st.markdown(
        "Generate and download publication-ready audit reports compiled 100% in-memory. "
        "No temporary files are written to disk."
    )

    current_df = st.session_state["current_df"]
    profile: Optional[DatasetProfile] = st.session_state["profile"]
    val: Optional[ValidationResult] = st.session_state["validation_result"]
    dqs: Optional[QualityScorecard] = st.session_state["quality_scorecard"]
    drift: Optional[DriftAnalysisResult] = st.session_state["drift_result"]
    anom: Optional[AnomalyDetectionResult] = st.session_state["anomaly_result"]

    # PREREQUISITE CHECK
    if current_df is None or dqs is None or profile is None or val is None:
        st.markdown(
            """
            <div class="sentry-callout sentry-callout-warning">
                <h4>⚠️ Quality Audit Required Before Generating Reports</h4>
                <p>
                    Audit reports compile findings from data profiling and schema validation. 
                    Please navigate to <b>🔍 Quality Audit</b> and click <b>"▶ Run Quality Audit"</b> 
                    to generate your scorecard before exporting.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button("⬇️ Download PDF Report (Disabled)", disabled=True, use_container_width=True)
        with c2:
            st.button("⬇️ Download Excel Workbook (Disabled)", disabled=True, use_container_width=True)
        with c3:
            st.button("⬇️ Download Issues CSV (Disabled)", disabled=True, use_container_width=True)
        return

    c_meta = st.session_state["current_meta"]
    r_meta = st.session_state["reference_meta"]

    meta = ReportMetadata(
        dataset_name=c_meta["name"],
        reference_name=r_meta["name"],
        worksheet_name=c_meta["sheet"],
        reference_worksheet_name=r_meta["sheet"],
        author="DataPulse",
        academic_context="Keep Your Data Clean. Reliable. Ready. | Data Quality & Anomaly Detection System",
    )

    st.markdown("---")

    r_col1, r_col2, r_col3 = st.columns(3)

    # 1. ACADEMIC PDF REPORT
    with r_col1:
        st.markdown("### 📄 Academic PDF Report")
        st.markdown(
            "Multi-page formal report generated with ReportLab Platypus. "
            "Features executive scorecard, Pandera violation logs, drift statistics, and anomaly rankings."
        )
        try:
            pdf_bytes = export_analysis_to_pdf(profile, dqs, val, drift, anom, meta)
            st.download_button(
                label="⬇️ Download PDF Audit Report",
                data=pdf_bytes,
                file_name=f"DataPulse_Audit_Report_{c_meta['name']}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Failed to compile PDF: {e}")

    # 2. FORMATTED EXCEL WORKBOOK
    with r_col2:
        st.markdown("### 📊 Formatted Excel Workbook")
        st.markdown(
            "Multi-tab spreadsheet (.xlsx) compiled with openpyxl. "
            "Includes styled Navy headers, freeze panes, auto-filters, and formulaic number formatting."
        )
        try:
            excel_bytes = export_analysis_to_excel(profile, dqs, val, drift, anom, meta)
            st.download_button(
                label="⬇️ Download Excel Workbook (.xlsx)",
                data=excel_bytes,
                file_name=f"DataPulse_Audit_Workbook_{c_meta['name']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Failed to compile Excel workbook: {e}")

    # 3. STREAMING CSV LOGS
    with r_col3:
        st.markdown("### 📝 Raw Data CSV Logs")
        st.markdown("Streaming CSV exports ready for automated ingestion into ETL pipelines.")

        csv_issues = export_issues_to_csv(val)
        st.download_button(
            label="⬇️ Validation Issues CSV",
            data=csv_issues,
            file_name=f"validation_issues_{c_meta['name']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if drift and drift.has_reference:
            csv_drift = export_drift_to_csv(drift)
            st.download_button(
                label="⬇️ Drift Test Results CSV",
                data=csv_drift,
                file_name=f"drift_results_{c_meta['name']}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if anom and anom.success:
            csv_anom = export_anomalies_to_csv(anom)
            st.download_button(
                label="⬇️ Flagged Anomalies CSV",
                data=csv_anom,
                file_name=f"anomalies_{c_meta['name']}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("#### 🔍 Audit Provenance & Metadata")
    st.json(meta.to_dict())


# ==============================================================================
# MAIN APPLICATION ROUTING
# ==============================================================================

def main() -> None:
    """Main application lifecycle and view routing."""
    # 1. Initialize persistent session state
    init_session_state()

    # 2. Inject professional design styling
    st.markdown(get_application_css(), unsafe_allow_html=True)

    # 3. Render application header
    render_application_header()

    # 4. Render sidebar navigation
    selected_section = render_sidebar_navigation()

    # 5. Route to selected view
    if selected_section == "📊 Overview":
        render_overview_page()
    elif selected_section == "📥 Data Input":
        render_data_input_page()
    elif selected_section == "🔍 Quality Audit":
        render_quality_audit_page()
    elif selected_section == "📈 Drift Detection":
        render_drift_detection_page()
    elif selected_section == "🧠 Anomaly Detection":
        render_anomaly_detection_page()
    elif selected_section == "📑 Reports & Downloads":
        render_reports_page()


if __name__ == "__main__":
    main()
