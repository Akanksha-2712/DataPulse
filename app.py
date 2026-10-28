"""Data Quality Monitoring and Anomaly Detection System — Streamlit Web Application.

A highly customizable, user-friendly, and production-ready academic dashboard featuring:
1. Dynamic Personalization: Custom app name, custom logo icon, and multi-theme color palettes.
2. Multi-Format Ingestion: CSV and XLSX upload with worksheet discovery and one-click demo benchmarking.
3. Data Profiling & Structural Health: Missing cells, duplicate records, data types, memory footprint.
4. Declarative Schema Validation: Pandera-driven constraints (nulls, ranges, allowed categories).
5. Transparent Quality Scoring: Explainable DQS indicator with normalized dimensional weighting.
6. Distribution Drift Monitoring: Two-Sample Kolmogorov-Smirnov test and Pearson Chi-Square test.
7. Unsupervised Anomaly Detection: Scikit-learn Isolation Forest with continuous suspicion scores.
8. Multi-Format Report Center: In-memory download of styled openpyxl Excel workbooks, ReportLab PDFs, and CSV logs.

Academic Persona:
B.Tech Artificial Intelligence and Data Science Student Final Project & Viva Voce Defense.
100% In-Memory Local Execution with Zero Outbound Network Persistence.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import AnomalyDetectionResult, detect_anomalies
from src.data_loader import (
    DataLoaderError,
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
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="DataSentinel AI — Data Quality & Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# COLOR THEME PALETTES & DYNAMIC CSS GENERATOR
# ==============================================================================

THEME_PALETTES = {
    "Midnight Navy & Steel Blue": {
        "primary": "#1B365D",
        "secondary": "#2E5B88",
        "accent": "#008080",
        "bg_soft": "#EBF3FB",
        "card_border": "#D0D5DD",
    },
    "Cyber Teal & Deep Emerald": {
        "primary": "#004D40",
        "secondary": "#00796B",
        "accent": "#00B4D8",
        "bg_soft": "#E0F2F1",
        "card_border": "#B2DFDB",
    },
    "Royal Indigo & Amethyst": {
        "primary": "#311B92",
        "secondary": "#512DA8",
        "accent": "#7C4DFF",
        "bg_soft": "#EDE7F6",
        "card_border": "#D1C4E9",
    },
    "Crimson Sentinel & Dark Slate": {
        "primary": "#880E4F",
        "secondary": "#AD1457",
        "accent": "#C2185B",
        "bg_soft": "#FCE4EC",
        "card_border": "#F8BBD0",
    },
    "Carbon Black & Gold Minimalist": {
        "primary": "#212121",
        "secondary": "#424242",
        "accent": "#D4AF37",
        "bg_soft": "#F5F5F5",
        "card_border": "#E0E0E0",
    },
}

APP_NAME_PRESETS = [
    "🛡️ DataSentinel AI",
    "⚡ QualiPulse Monitor",
    "🌐 AegisData Observability",
    "🔍 VeriData Studio",
    "🦅 SentinelWatch MLOps",
    "✏️ Custom Application Name",
]

LOGO_PRESETS = [
    ("🛡️", "Shield — Security & Data Governance"),
    ("⚡", "Lightning — High-Velocity Pulse & Observability"),
    ("🦅", "Sentinel Eagle — Continuous Vigilance"),
    ("💎", "Diamond — Precision Data Excellence"),
    ("🔬", "Microscope — Deep Diagnostic Audit"),
    ("🎯", "Target — Accurate Boundary Enforcement"),
    ("🌐", "Globe — Enterprise Distributed Quality"),
]


def generate_custom_css(theme: Dict[str, str]) -> str:
    """Generate dynamic CSS styled according to the user's selected theme."""
    return f"""
    <style>
        /* Card Aesthetics */
        .metric-card {{
            background-color: #FFFFFF;
            border: 1px solid {theme['card_border']};
            border-left: 4px solid {theme['primary']};
            border-radius: 8px;
            padding: 14px 18px;
            text-align: left;
            box-shadow: 0 2px 5px rgba(0,0,0,0.04);
            margin-bottom: 12px;
            transition: transform 0.15s ease-in-out;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        }}
        .metric-label {{
            font-size: 0.80rem;
            font-weight: 600;
            color: #5A6B82;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: {theme['primary']};
            margin-top: 3px;
        }}
        .metric-delta {{
            font-size: 0.82rem;
            color: #555555;
            margin-top: 2px;
        }}

        /* Status Badges */
        .status-badge-green {{
            color: #1B5E20;
            background-color: #E8F5E9;
            padding: 3px 9px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.82rem;
            display: inline-block;
        }}
        .status-badge-red {{
            color: #B71C1C;
            background-color: #FFEBEE;
            padding: 3px 9px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.82rem;
            display: inline-block;
        }}
        .status-badge-orange {{
            color: #E65100;
            background-color: #FFF3E0;
            padding: 3px 9px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.82rem;
            display: inline-block;
        }}

        /* Hero Banner */
        .hero-banner {{
            background: linear-gradient(135deg, {theme['primary']} 0%, {theme['secondary']} 100%);
            color: white;
            padding: 24px 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .hero-title {{
            font-size: 1.9rem;
            font-weight: 800;
            margin-bottom: 6px;
            letter-spacing: -0.3px;
        }}
        .hero-subtitle {{
            font-size: 0.95rem;
            opacity: 0.92;
            line-height: 1.45;
        }}

        /* Viva Voce Callout Box */
        .viva-box {{
            background-color: {theme['bg_soft']};
            border-left: 4px solid {theme['primary']};
            padding: 14px 18px;
            border-radius: 6px;
            margin-top: 14px;
            margin-bottom: 14px;
            font-size: 0.9rem;
            line-height: 1.5;
            color: #1E293B;
        }}

        /* Stepper Guide */
        .step-pill {{
            background-color: #F1F5F9;
            color: #334155;
            border: 1px solid #CBD5E1;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 6px;
            display: inline-block;
        }}
        .step-pill-active {{
            background-color: {theme['primary']};
            color: white;
            border: 1px solid {theme['primary']};
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 6px;
            display: inline-block;
        }}
    </style>
    """


# ==============================================================================
# HELPER: ANALYSIS PIPELINE ORCHESTRATOR
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
    app_title: str = "DataSentinel AI",
    app_subtitle: str = "B.Tech Final Project",
) -> Dict[str, Any]:
    """Execute the end-to-end analytical workflow across all modules.

    Parameters
    ----------
    current_df : pd.DataFrame
        Current dataset to profile and score.
    reference_df : Optional[pd.DataFrame]
        Baseline reference dataset for drift and anomaly modeling.
    weights : Dict[str, float]
        DQS dimensional weights (completeness, uniqueness, validity).
    alpha : float
        Statistical drift significance threshold.
    contamination : float
        Expected proportion of outliers for Isolation Forest.
    app_title : str
        Branding name for report headers.
    app_subtitle : str
        Context subtitle for report headers.

    Returns
    -------
    Dict[str, Any]
        Dictionary bundle with all structured analysis artifacts.
    """
    # 1. Profile current dataset
    profile = profile_dataset(current_df)

    # 2. Validate against declarative rules
    rules_path = PROJECT_ROOT / "config" / "validation_rules.json"
    rules = load_validation_rules(rules_path) if rules_path.exists() else None
    validation_result = validate_dataset(current_df, rules)

    # 3. Compute transparent Data Quality Score
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
        author=f"{app_title} (B.Tech AI & Data Science Capstone)",
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
# SIDEBAR: BRANDING, DATA INGESTION & PARAMETERS
# ==============================================================================

def render_sidebar() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    """Render sidebar controls for branding, data ingestion, and hyperparameters."""
    with st.sidebar:
        # ----------------------------------------------------------------------
        # 1. APPLICATION BRANDING & PERSONALIZATION (User-Requested Feature)
        # ----------------------------------------------------------------------
        st.markdown("## 🎨 App Branding & Theme")

        preset_choice = st.selectbox(
            "Application Name Preset",
            APP_NAME_PRESETS,
            index=0,
            help="Choose a pre-configured brand identity or customize your own title for your viva voce.",
        )

        if preset_choice == "✏️ Custom Application Name":
            app_title = st.text_input(
                "Enter Custom Application Name",
                value="Data Quality & Anomaly Detection System",
                help="Type your preferred title, student name, or institution.",
            )
        else:
            # Strip emoji prefix for clean text
            app_title = preset_choice.split(" ", 1)[-1] if " " in preset_choice else preset_choice

        # Logo / Badge Picker
        logo_options = [opt[0] for opt in LOGO_PRESETS]
        logo_labels = {opt[0]: f"{opt[0]} {opt[1]}" for opt in LOGO_PRESETS}
        selected_logo = st.selectbox(
            "Select Logo / Icon",
            logo_options,
            format_func=lambda x: logo_labels[x],
            index=0,
            help="Choose an icon representing your system's operational focus.",
        )

        # Color Theme Palette Picker
        theme_names = list(THEME_PALETTES.keys())
        selected_theme_name = st.selectbox(
            "Color Palette Theme",
            theme_names,
            index=0,
            help="Select a color scheme for the dashboard banner, metric cards, and chart accents.",
        )
        selected_theme = THEME_PALETTES[selected_theme_name]

        app_subtitle = st.text_input(
            "Subtitle / Academic Context",
            value="Automated Profiling, Pandera Schema Validation, Drift Monitoring & Isolation Forest | B.Tech AI & DS",
            help="Appears directly under the main header and in exported audit reports.",
        )

        st.markdown(
            """
            <div style="font-size: 0.8rem; background-color: #E8F5E9; color: #2E7D32; padding: 6px 10px; border-radius: 4px; margin-top: 6px; margin-bottom: 14px; font-weight: 600;">
                🔒 100% In-Memory Local Processing & Zero Cloud Persistence
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ----------------------------------------------------------------------
        # 2. DATASET INGESTION CONTROLS
        # ----------------------------------------------------------------------
        st.markdown("## 📂 Data Ingestion")

        data_mode = st.radio(
            "Select Data Source",
            ["🚀 Load Synthetic Demo Benchmark", "📁 Upload My Datasets"],
            help="Benchmark mode loads clean baseline and corrupted current datasets instantly.",
        )

        current_df = None
        reference_df = None
        cur_name = "current.csv"
        ref_name = "baseline.csv"
        cur_sheet = None
        ref_sheet = None

        if data_mode == "🚀 Load Synthetic Demo Benchmark":
            demo_dir = PROJECT_ROOT / "data" / "demo"
            cur_file = demo_dir / "current.csv"
            ref_file = demo_dir / "baseline.csv"

            if cur_file.exists() and ref_file.exists():
                try:
                    current_df = read_csv_bytes(cur_file.read_bytes())
                    reference_df = read_csv_bytes(ref_file.read_bytes())
                    st.success("✅ Demo Benchmark loaded (508 current rows, 500 baseline rows).")
                except Exception as e:
                    st.error(f"Failed to read demo benchmark files: {e}")
            else:
                st.warning("Demo files not found on disk. Generating dynamically...")
                from src.synthetic_generator import generate_baseline_dataset, generate_current_dataset
                reference_df = generate_baseline_dataset(500, seed=42)
                current_df, _ = generate_current_dataset(500, seed=101)
                st.success("✅ Synthetic benchmark generated successfully.")

        else:
            # Custom File Uploads
            st.markdown("#### 1. Target Dataset (Required)")
            cur_uploader = st.file_uploader(
                "Upload Current Production Data (.csv, .xlsx)",
                type=["csv", "xlsx"],
                key="cur_upload",
                help="Accepts CSV or modern Excel workbooks (.xlsx).",
            )
            if cur_uploader is not None:
                cur_name = cur_uploader.name
                cur_bytes = cur_uploader.getvalue()
                if cur_name.lower().endswith(".xlsx"):
                    sheets = list_excel_sheets(cur_bytes)
                    cur_sheet = st.selectbox("Select Target Worksheet", sheets, key="cur_sheet_sel")
                    current_df = read_excel_bytes(cur_bytes, sheet_name=cur_sheet)
                else:
                    current_df = read_csv_bytes(cur_bytes)

            st.markdown("#### 2. Baseline Dataset (Optional)")
            st.caption("Required for distribution and schema drift testing.")
            ref_uploader = st.file_uploader(
                "Upload Reference Baseline Data (.csv, .xlsx)",
                type=["csv", "xlsx"],
                key="ref_upload",
                help="Historical or clean training batch to compare against.",
            )
            if ref_uploader is not None:
                ref_name = ref_uploader.name
                ref_bytes = ref_uploader.getvalue()
                if ref_name.lower().endswith(".xlsx"):
                    r_sheets = list_excel_sheets(ref_bytes)
                    ref_sheet = st.selectbox("Select Baseline Worksheet", r_sheets, key="ref_sheet_sel")
                    reference_df = read_excel_bytes(ref_bytes, sheet_name=ref_sheet)
                else:
                    reference_df = read_csv_bytes(ref_bytes)
            else:
                ref_name = None

        # ----------------------------------------------------------------------
        # 3. ADVANCED PARAMETERS & HYPERPARAMETERS
        # ----------------------------------------------------------------------
        with st.expander("⚙️ Analysis Parameters & DQS Weights", expanded=False):
            st.markdown("**Data Quality Score (DQS) Weights:**")
            w_comp = st.slider("Completeness Weight", 0, 100, 40, 5, help="Weight assigned to cell non-nullness.")
            w_uniq = st.slider("Uniqueness Weight", 0, 100, 25, 5, help="Weight assigned to row uniqueness.")
            w_val = st.slider("Validity Weight", 0, 100, 35, 5, help="Weight assigned to Pandera rule compliance.")

            st.markdown("---")
            st.markdown("**Statistical Drift & ML Thresholds:**")
            alpha = st.slider(
                "Drift Significance Alpha (α)", 0.01, 0.10, 0.05, 0.01,
                help="Significance threshold for Kolmogorov-Smirnov and Chi-Square hypothesis tests.",
            )
            contamination = st.slider(
                "Anomaly Contamination Rate", 0.01, 0.20, 0.05, 0.01,
                help="Estimated proportion of outliers in the dataset for Isolation Forest.",
            )

        # Cache Reset Button
        if st.button("🔄 Clear Cache & Re-run Analysis", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        params = {
            "app_title": app_title,
            "selected_logo": selected_logo,
            "selected_theme": selected_theme,
            "app_subtitle": app_subtitle,
            "weights": {"completeness": float(w_comp), "uniqueness": float(w_uniq), "validity": float(w_val)},
            "alpha": float(alpha),
            "contamination": float(contamination),
            "current_name": cur_name,
            "reference_name": ref_name,
            "current_sheet": cur_sheet,
            "reference_sheet": ref_sheet,
        }

    return current_df, reference_df, params


# ==============================================================================
# MAIN PAGE RENDERING
# ==============================================================================

def main() -> None:
    """Main application lifecycle and view routing."""
    # Render Sidebar and extract branding/parameters
    current_df, reference_df, params = render_sidebar()

    # Apply user-selected dynamic theme CSS
    st.markdown(generate_custom_css(params["selected_theme"]), unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TOP HERO BANNER (Drowsily customized by user)
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">{params['selected_logo']} {params['app_title']}</div>
            <div class="hero-subtitle">{params['app_subtitle']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Stepper Guide (User-Friendly Overview)
    with st.expander("💡 Quick Navigation & Workflow Stepper (Click to expand)", expanded=False):
        st.markdown(
            """
            <div style="margin-bottom: 8px;">
                <span class="step-pill">1️⃣ Ingestion</span>
                <span class="step-pill">2️⃣ Executive Health</span>
                <span class="step-pill">3️⃣ Pandera Validation</span>
                <span class="step-pill">4️⃣ Drift Testing</span>
                <span class="step-pill">5️⃣ Anomaly Isolation</span>
                <span class="step-pill">6️⃣ Export Audit</span>
            </div>
            This application audits tabular datasets through three complementary layers:
            1. <b>Deterministic Syntactic & Schema Checks:</b> Catches missing cells, duplicates, and rule violations using <b>Pandera</b>.
            2. <b>Statistical Distribution Drift:</b> Compares reference baseline distributions with current batches using <b>Kolmogorov-Smirnov</b> and <b>Chi-Square</b> tests.
            3. <b>Unsupervised Multivariate Anomalies:</b> Discovers complex hidden outliers using <b>Isolation Forest</b>.
            """,
            unsafe_allow_html=True,
        )

    if current_df is None:
        st.info("👋 Welcome! Please upload a dataset in the sidebar or select **'Load Synthetic Demo Benchmark'** to begin.")
        return

    # Trigger or retrieve cached analysis bundle
    bundle_key = (
        f"{params['app_title']}_{params['current_name']}_{params['current_sheet']}_"
        f"{params['reference_name']}_{params['reference_sheet']}_"
        f"{params['alpha']}_{params['contamination']}_"
        f"{params['weights']['completeness']}_{params['weights']['uniqueness']}_{params['weights']['validity']}"
    )

    if "last_bundle_key" not in st.session_state or st.session_state["last_bundle_key"] != bundle_key:
        with st.spinner("Analyzing dataset structure, evaluating drift, and isolating anomalies..."):
            bundle = run_analysis_pipeline(
                current_df=current_df,
                reference_df=reference_df,
                weights=params["weights"],
                alpha=params["alpha"],
                contamination=params["contamination"],
                current_name=params["current_name"],
                reference_name=params["reference_name"],
                current_sheet=params["current_sheet"],
                reference_sheet=params["reference_sheet"],
                app_title=params["app_title"],
                app_subtitle=params["app_subtitle"],
            )
            st.session_state["bundle"] = bundle
            st.session_state["last_bundle_key"] = bundle_key
    else:
        bundle = st.session_state["bundle"]

    # Unpack analysis artifacts
    profile: DatasetProfile = bundle["profile"]
    val: ValidationResult = bundle["validation_result"]
    dqs: QualityScorecard = bundle["quality_scorecard"]
    drift: Optional[DriftAnalysisResult] = bundle["drift_result"]
    anom: Optional[AnomalyDetectionResult] = bundle["anomaly_result"]
    meta: ReportMetadata = bundle["metadata"]

    # --------------------------------------------------------------------------
    # MAIN WORKSPACE TABS
    # --------------------------------------------------------------------------
    tab_overview, tab_quality, tab_drift, tab_anom, tab_reports = st.tabs([
        "📊 Overview",
        "🔍 Quality Checks",
        "📈 Data Drift",
        "🧠 Anomaly Detection",
        "📑 Reports & Export",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # --------------------------------------------------------------------------
    with tab_overview:
        st.subheader("Executive System Health")

        # Top KPI Metric Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Dataset Volume</div>
                    <div class="metric-value">{profile.row_count:,}</div>
                    <div class="metric-delta">{profile.column_count} columns ({(profile.memory_usage_bytes/1024):.1f} KB)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            sc_color = "#2E7D32" if dqs.composite_score >= 85 else ("#EF6C00" if dqs.composite_score >= 70 else "#C62828")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Quality Score (DQS)</div>
                    <div class="metric-value" style="color: {sc_color};">{dqs.composite_score:.1f}<span style="font-size:1rem;">/100</span></div>
                    <div class="metric-delta">Grade: <b>{dqs.grade.upper()}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            null_badge = f"<span class='status-badge-{'green' if profile.missing_cells == 0 else 'orange'}'>{profile.missing_cells_pct:.1f}%</span>"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Missing Cells</div>
                    <div class="metric-value">{profile.missing_cells:,}</div>
                    <div class="metric-delta">{null_badge} of total cells</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            dup_badge = f"<span class='status-badge-{'green' if profile.duplicate_rows == 0 else 'orange'}'>{profile.duplicate_rows_pct:.1f}%</span>"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Duplicate Rows</div>
                    <div class="metric-value">{profile.duplicate_rows:,}</div>
                    <div class="metric-delta">{dup_badge} redundant rows</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c5:
            crit_badge = f"<span class='status-badge-{'green' if val.critical_count == 0 else 'red'}'>{val.critical_count} critical</span>"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Validation Issues</div>
                    <div class="metric-value">{val.total_issues}</div>
                    <div class="metric-delta">{crit_badge} issues logged</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Status Summary Bar
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            if drift and drift.has_reference:
                d_color = "red" if drift.dataset_drift_detected else "green"
                d_txt = "DRIFT DETECTED" if drift.dataset_drift_detected else "STABLE (NO DRIFT)"
                st.markdown(
                    f"**Distribution Drift:** <span class='status-badge-{d_color}'>{d_txt}</span> "
                    f"({drift.drifted_columns_count} of {drift.total_columns_compared} columns shifted significantly)",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "**Distribution Drift:** <span class='status-badge-orange'>NO BASELINE SUPPLIED</span> "
                    "(Upload reference dataset to compare distributions)",
                    unsafe_allow_html=True,
                )
        with s_c2:
            if anom and anom.success:
                st.markdown(
                    f"**ML Anomalies:** <span class='status-badge-red'>{anom.anomalies_detected_count} Flagged</span> "
                    f"({anom.anomaly_share * 100:.1f}% flagged as suspicious by Isolation Forest)",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "**ML Anomalies:** <span class='status-badge-orange'>SKIPPED</span> (No suitable numeric features)",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Visual Scorecard Row
        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            st.plotly_chart(plot_quality_gauge(dqs.composite_score, dqs.grade), use_container_width=True)
        with v_col2:
            st.plotly_chart(plot_dimension_radar(dqs.dimensions), use_container_width=True)

        # Dataset Preview Table
        st.subheader("Dataset Preview & Schema Invariants")
        st.dataframe(current_df.head(10), use_container_width=True)

        st.markdown(
            f"""
            <div class="viva-box">
                <b>🎓 Viva Voce Key Takeaway (Architecture Overview):</b><br>
                The Data Quality Score (DQS) combines syntactic completeness, row uniqueness, and schema validity 
                into an explainable normalized indicator in [0, 100]. If validity rules are omitted, 
                weights dynamically re-scale across available dimensions rather than artificially penalizing the data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # TAB 2: QUALITY CHECKS
    # --------------------------------------------------------------------------
    with tab_quality:
        st.subheader("Data Profiling & Pandera Schema Validation")

        # Missingness Visualization
        st.plotly_chart(plot_missingness_bar(profile), use_container_width=True)

        col_q1, col_q2 = st.columns([1, 1])
        with col_q1:
            st.markdown("#### Duplicate Records Inspection")
            if profile.duplicate_rows > 0:
                st.warning(f"Detected **{profile.duplicate_rows}** duplicate rows ({profile.duplicate_rows_pct:.2f}%).")
                with st.expander("Inspect Duplicate Rows"):
                    dups = current_df[current_df.duplicated(keep=False)]
                    st.dataframe(dups.head(20), use_container_width=True)
            else:
                st.success("✅ Zero duplicate rows detected. Row uniqueness is 100%.")

        with col_q2:
            st.markdown("#### Column Datatype & Health Breakdown")
            st.markdown(
                f"- **Numeric Features:** {len(profile.numeric_columns)} ({', '.join(profile.numeric_columns)})\n"
                f"- **Categorical Features:** {len(profile.categorical_columns)} ({', '.join(profile.categorical_columns)})\n"
                f"- **Empty Strings Logged:** {profile.empty_strings_total}\n"
                f"- **Infinite Values Logged:** {profile.infinite_values_total}"
            )

        st.markdown("---")
        st.markdown("#### Declarative Rule Violations (Pandera)")

        if val.rules_configured:
            st.markdown(
                f"Status: **{'PASSED' if val.is_valid else 'VIOLATIONS DETECTED'}** | "
                f"Total Issues: **{val.total_issues}** ("
                f"<span class='status-badge-red'>{val.critical_count} Critical</span> "
                f"<span class='status-badge-orange'>{val.error_count} Error</span> "
                f"<span class='status-badge-green'>{val.warning_count} Warning</span>)",
                unsafe_allow_html=True,
            )

            if val.issues:
                # Severity and Column Search Filters (User-Friendly Control)
                f_c1, f_c2 = st.columns([1, 2])
                with f_c1:
                    sev_filter = st.selectbox("Filter by Severity", ["ALL", "CRITICAL", "ERROR", "WARNING"])
                with f_c2:
                    col_search = st.text_input("Search issues by column name", "")

                filtered_issues = val.issues
                if sev_filter != "ALL":
                    filtered_issues = [i for i in filtered_issues if i.severity.upper() == sev_filter]
                if col_search.strip():
                    filtered_issues = [i for i in filtered_issues if col_search.lower() in i.column.lower()]

                issue_rows = []
                for idx, iss in enumerate(filtered_issues, 1):
                    issue_rows.append({
                        "#": idx,
                        "Severity": iss.severity.upper(),
                        "Type": iss.issue_type,
                        "Column": iss.column,
                        "Row Index": iss.row_reference if iss.row_reference is not None else "Column-Level",
                        "Observed Value": str(iss.observed_value) if iss.observed_value is not None else "-",
                        "Description": iss.description,
                    })

                st.dataframe(pd.DataFrame(issue_rows), use_container_width=True)
            else:
                st.success("✅ All records satisfied configured Pandera validation schema constraints.")
        else:
            st.info("ℹ️ No validation rules were configured. Validity dimension was not measured.")

        st.markdown(
            """
            <div class="viva-box">
                <b>🎓 Viva Voce Key Takeaway (Syntactic Validation):</b><br>
                Pandera provides deterministic, rule-based schema validation (ranges, allowed categories, required fields). 
                Unlike machine learning, rule validation provides exact, non-negotiable assertions regarding operational schema conformity.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # TAB 3: DATA DRIFT
    # --------------------------------------------------------------------------
    with tab_drift:
        st.subheader("Schema & Statistical Distribution Drift")

        if drift is None or not drift.has_reference:
            st.warning("⚠️ **Drift Detection Skipped:** No reference baseline dataset was provided.")
            st.info(
                "To evaluate distribution and schema drift, please upload a baseline reference dataset "
                "in the sidebar or select **'Load Synthetic Demo Benchmark'**."
            )
        else:
            v_color = "red" if drift.dataset_drift_detected else "green"
            v_txt = "DATASET DRIFT DETECTED" if drift.dataset_drift_detected else "DATASET STABLE (NO STATISTICAL DRIFT)"
            st.markdown(
                f"### Overall Status: <span class='status-badge-{v_color}'>{v_txt}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Dataset drift triggered because {drift.drifted_columns_count} of {drift.total_columns_compared} "
                f"columns shifted significantly (exceeding threshold α = {drift.column_drift[list(drift.column_drift.keys())[0]].threshold_alpha if drift.column_drift else 0.05})."
            )

            # Schema Changes (Robust Access via getattr)
            sd = drift.schema_diff
            has_changes = getattr(sd, "has_schema_changes", getattr(sd, "schema_has_drift", False))
            missing_cols = getattr(sd, "missing_columns", [])
            added_cols = getattr(sd, "added_columns", getattr(sd, "new_columns", []))
            type_chgs = getattr(sd, "type_changes", getattr(sd, "type_mismatches", {}))

            if has_changes:
                st.error("⚠️ Schema alterations detected between baseline and current data!")
                if missing_cols:
                    st.write(f"- **Dropped Columns:** `{', '.join(missing_cols)}`")
                if added_cols:
                    st.write(f"- **New Columns:** `{', '.join(added_cols)}`")
                if type_chgs:
                    st.write(f"- **Type Mismatches:** `{type_chgs}`")
            else:
                st.success("✅ Schema structures match: Identical column names and compatible data types.")

            st.markdown("---")
            st.markdown("#### Column-Level Statistical Tests")

            # Filter options for Drift Table
            d_filter = st.radio("Show Columns", ["All Columns", "Drifted Only (🚨)", "Stable Only (✅)"], horizontal=True)

            d_rows = []
            for col_name, cdr in drift.column_drift.items():
                if d_filter == "Drifted Only (🚨)" and not cdr.drift_detected:
                    continue
                if d_filter == "Stable Only (✅)" and cdr.drift_detected:
                    continue

                d_rows.append({
                    "Column": col_name,
                    "Type": cdr.column_type,
                    "Statistical Method": cdr.method,
                    "Statistic": f"{cdr.statistic:.4f}" if cdr.statistic is not None else "-",
                    "p-Value": f"{cdr.p_value:.4f}" if cdr.p_value is not None else "-",
                    "Threshold (α)": cdr.threshold_alpha,
                    "Drift Detected": "🚨 YES" if cdr.drift_detected else "✅ NO",
                    "Status": cdr.skip_reason if cdr.skipped else "Evaluated",
                })
            st.dataframe(pd.DataFrame(d_rows), use_container_width=True)

            # Interactive Distribution Visualizer
            st.markdown("---")
            st.markdown("#### Interactive Distribution Explorer")
            available_cols = list(drift.column_drift.keys())
            if available_cols:
                sel_col = st.selectbox("Select Column to Inspect Distribution", available_cols)
                col_type = drift.column_drift[sel_col].column_type
                is_num = (col_type == "numeric")
                fig_drift = plot_drift_distribution_comparison(reference_df, current_df, sel_col, is_num)
                st.plotly_chart(fig_drift, use_container_width=True)

            st.markdown(
                """
                <div class="viva-box">
                    <b>🎓 Viva Voce Key Takeaway (Data Drift vs. Errors):</b><br>
                    Data drift represents a statistical distribution shift (covariate shift) over time, calculated using 
                    the non-parametric Two-Sample Kolmogorov-Smirnov test (continuous) or Pearson's Chi-Square test (categorical). 
                    <b>A drifted column is NOT necessarily corrupt or wrong</b>—it signals that the real-world population has changed, 
                    warning data scientists that downstream ML models may experience performance degradation.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------------------------
    # TAB 4: ANOMALY DETECTION
    # --------------------------------------------------------------------------
    with tab_anom:
        st.subheader("Unsupervised Multivariate Anomaly Detection (Isolation Forest)")

        if anom is None or not anom.success:
            st.warning("⚠️ Anomaly detection could not be executed.")
            st.info(anom.error_message if anom else "No numeric features available.")
        else:
            # Key Model Metrics
            a_c1, a_c2, a_c3, a_c4 = st.columns(4)
            with a_c1:
                st.metric("Training Mode", anom.training_mode.replace("_", " ").title())
            with a_c2:
                st.metric("Features Analyzed", len(anom.features_used))
            with a_c3:
                st.metric("Contamination Rate", f"{anom.contamination_parameter}")
            with a_c4:
                st.metric("Flagged Outliers", f"{anom.anomalies_detected_count} ({anom.anomaly_share*100:.1f}%)")

            # Suspicion Score Distribution
            st.plotly_chart(plot_anomaly_score_distribution(anom), use_container_width=True)

            # 2D Feature Projection Scatter Plot
            st.markdown("#### Multivariate Feature Projection (2D)")
            if len(anom.features_used) >= 2:
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    x_feat = st.selectbox("X-Axis Feature", anom.features_used, index=0)
                with s_c2:
                    y_feat = st.selectbox("Y-Axis Feature", anom.features_used, index=min(1, len(anom.features_used) - 1))

                fig_scatter = plot_anomaly_scatter(current_df, anom, x_feat, y_feat)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("At least two numerical features are required for 2D scatter visualization.")

            # Filter Anomalies by Score Threshold (User-Friendly Feature)
            st.markdown("#### Candidate Suspicious Records Ranked by Suspicion Score")
            min_score = st.slider(
                "Filter records by Minimum Suspicion Score",
                0.0, 1.0, 0.50, 0.02,
                help="Only show records whose isolation suspicion score s(x, n) meets or exceeds this threshold.",
            )

            anom_table_rows = []
            for rec in anom.anomaly_records:
                if rec.anomaly_score < min_score:
                    continue
                top_dev = "; ".join(
                    f"{k}: {v:+.2f} IQR"
                    for k, v in sorted(rec.feature_deviations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                )
                anom_table_rows.append({
                    "Row #": rec.row_index,
                    "Status": "🚨 ANOMALY" if rec.is_anomaly else "NORMAL",
                    "Suspicion Score (0-1)": round(rec.anomaly_score, 4),
                    "Raw Decision Score": round(rec.raw_decision_score, 4),
                    "Top Feature Deviations": top_dev,
                    "Feature Values": str(rec.record_data),
                })

            st.caption(f"Showing {len(anom_table_rows)} records with suspicion score ≥ {min_score:.2f}.")
            st.dataframe(pd.DataFrame(anom_table_rows), use_container_width=True)

            st.markdown(
                """
                <div class="viva-box">
                    <b>🎓 Viva Voce Key Takeaway (Isolation Forest Nuance):</b><br>
                    Isolation Forest isolates anomalies by randomly partitioning feature space: abnormal records require 
                    fewer random recursive splits (shorter path length h(x)) to be isolated. 
                    <b>Vital viva distinction:</b> An ML anomaly is an uncharacteristic multivariate outlier requiring human review; 
                    it is <b>NOT</b> automatic proof of corrupted or fraudulent data.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------------------------
    # TAB 5: REPORTS & AUDIT EXPORT CENTER
    # --------------------------------------------------------------------------
    with tab_reports:
        st.subheader("Multi-Format Audit Export Center")
        st.markdown(
            "Download publication-grade academic and industrial audit reports compiled in-memory. "
            "Your chosen app branding and subtitle will be embedded directly in the downloaded reports."
        )

        st.markdown("---")

        r_c1, r_c2, r_c3 = st.columns(3)

        with r_c1:
            st.markdown("### 📄 Academic PDF Report")
            st.markdown(
                "Multi-page formal report containing executive scorecard, Pandera violation logs, "
                "drift test statistics, anomaly rankings, and mathematical methodology."
            )
            try:
                pdf_bytes = export_analysis_to_pdf(profile, dqs, val, drift, anom, meta)
                st.download_button(
                    label=f"⬇️ Download {params['app_title']} PDF Report",
                    data=pdf_bytes,
                    file_name=f"audit_report_{meta.dataset_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")

        with r_c2:
            st.markdown("### 📊 Formatted Excel Workbook")
            st.markdown(
                "Multi-tab spreadsheet (.xlsx) styled with Navy theme, auto-fit columns, freeze panes, "
                "and auto-filters across Summary, Issues, Drift, Anomalies, and Profile."
            )
            try:
                excel_bytes = export_analysis_to_excel(profile, dqs, val, drift, anom, meta)
                st.download_button(
                    label="⬇️ Download Excel Workbook (.xlsx)",
                    data=excel_bytes,
                    file_name=f"audit_workbook_{meta.dataset_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Failed to generate Excel workbook: {e}")

        with r_c3:
            st.markdown("### 📝 Raw Data CSV Logs")
            st.markdown("Download raw, streaming CSV tables for downstream ETL and reporting tools.")

            csv_issues = export_issues_to_csv(val)
            st.download_button(
                label="⬇️ Validation Issues (.csv)",
                data=csv_issues,
                file_name="validation_issues.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if drift and drift.has_reference:
                csv_drift = export_drift_to_csv(drift)
                st.download_button(
                    label="⬇️ Drift Test Results (.csv)",
                    data=csv_drift,
                    file_name="drift_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            if anom and anom.success:
                csv_anom = export_anomalies_to_csv(anom)
                st.download_button(
                    label="⬇️ Flagged Anomalies (.csv)",
                    data=csv_anom,
                    file_name="anomaly_records.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.markdown("---")
        st.markdown("#### Audit Metadata & Provenance")
        st.json(meta.to_dict())


if __name__ == "__main__":
    main()
