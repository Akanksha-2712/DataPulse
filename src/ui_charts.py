"""Interactive Plotly Chart Generators for Streamlit UI.

Provides production-ready visualization utilities:
- Metric Gauge: Composite Data Quality Score (DQS) with graded color bands.
- Dimension Radar: Polar polygon comparing completeness, uniqueness, and validity.
- Missingness Breakdown: Bar chart highlighting columns with data gaps.
- Dual-Distribution Drift Comparison: Overlaid KDE/histograms and grouped category proportions.
- Multivariate Anomaly Scatter: 2D feature projection showing isolated points vs dense normal clusters.
- Anomaly Suspicion Distribution: Histogram of continuous suspicion scores in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.anomaly_detection import AnomalyDetectionResult
from src.profiling import DatasetProfile


NAVY = "#1B365D"
STEEL = "#2E5B88"
TEAL = "#008080"
DANGER = "#D9534F"
WARNING = "#F0AD4E"
SUCCESS = "#5CB85C"
LIGHT_GRAY = "#F8F9FA"


def plot_quality_gauge(composite_score: float, grade: str) -> go.Figure:
    """Create a gauge chart representing the composite Data Quality Score.

    Parameters
    ----------
    composite_score : float
        Overall DQS in [0, 100].
    grade : str
        Qualitative evaluation (e.g. 'Excellent', 'Good', 'Fair', 'Poor').

    Returns
    -------
    go.Figure
        Plotly Indicator figure.
    """
    grade_color = SUCCESS if composite_score >= 85 else (WARNING if composite_score >= 70 else DANGER)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=composite_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"DQS: {grade.upper()}", "font": {"size": 18, "color": NAVY}},
            number={"suffix": "/100", "font": {"size": 28, "color": NAVY}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": NAVY},
                "bar": {"color": grade_color},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#D0D5DD",
                "steps": [
                    {"range": [0, 60], "color": "#FFEBEE"},
                    {"range": [60, 80], "color": "#FFF8E1"},
                    {"range": [80, 100], "color": "#E8F5E9"},
                ],
                "threshold": {
                    "line": {"color": NAVY, "width": 3},
                    "thickness": 0.8,
                    "value": composite_score,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif"),
    )
    return fig


def plot_dimension_radar(dimensions: Dict[str, Any]) -> go.Figure:
    """Create a polar radar chart comparing individual data quality dimensions.

    Parameters
    ----------
    dimensions : Dict[str, Any]
        Dictionary of dimension objects with .score attributes.

    Returns
    -------
    go.Figure
        Plotly Scatterpolar figure.
    """
    categories = []
    scores = []

    for name, dim_obj in dimensions.items():
        categories.append(name.capitalize())
        sc = getattr(dim_obj, "score", None)
        scores.append(sc if sc is not None else 0.0)

    # Close the radar loop
    if categories:
        categories.append(categories[0])
        scores.append(scores[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scores,
            theta=categories,
            fill="toself",
            fillcolor="rgba(46, 91, 136, 0.25)",
            line=dict(color=STEEL, width=2),
            marker=dict(size=6, color=NAVY),
            name="Dimension Health",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9),
            )
        ),
        showlegend=False,
        height=230,
        margin=dict(l=30, r=30, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_missingness_bar(profile: DatasetProfile) -> go.Figure:
    """Create a bar chart showing null/missing cell percentages by column.

    Parameters
    ----------
    profile : DatasetProfile
        Dataset summary profile.

    Returns
    -------
    go.Figure
        Plotly Bar figure.
    """
    col_names = []
    missing_pcts = []
    missing_counts = []

    for col, cp in profile.columns.items():
        col_names.append(col)
        missing_pcts.append(cp.missing_pct)
        missing_counts.append(cp.missing_count)

    # Color code bars: Red (>5%), Orange (>0%), Blue (0%)
    colors_list = [
        DANGER if p > 5.0 else (WARNING if p > 0.0 else STEEL)
        for p in missing_pcts
    ]

    fig = go.Figure(
        go.Bar(
            x=col_names,
            y=missing_pcts,
            marker_color=colors_list,
            text=[f"{p:.1f}% ({c})" for p, c in zip(missing_pcts, missing_counts)],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Missing: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Missingness Percentage per Column (%)",
        xaxis_title="Column",
        yaxis_title="Missing Rate (%)",
        yaxis=dict(range=[0, max(max(missing_pcts, default=0) * 1.15, 10)]),
        height=320,
        margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_drift_distribution_comparison(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    col_name: str,
    is_numeric: bool,
) -> go.Figure:
    """Create comparison chart showing reference vs current distributions.

    Parameters
    ----------
    ref_df : pd.DataFrame
        Baseline reference dataset.
    cur_df : pd.DataFrame
        Current production dataset.
    col_name : str
        Target column to compare.
    is_numeric : bool
        True if column is numerical continuous, False if categorical.

    Returns
    -------
    go.Figure
        Plotly comparison figure.
    """
    fig = go.Figure()

    if is_numeric:
        ref_s = pd.to_numeric(ref_df[col_name], errors="coerce").dropna()
        cur_s = pd.to_numeric(cur_df[col_name], errors="coerce").dropna()

        fig.add_trace(
            go.Histogram(
                x=ref_s,
                name="Baseline Reference",
                marker_color=STEEL,
                opacity=0.6,
                histnorm="probability density",
                nbinsx=35,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=cur_s,
                name="Current Dataset",
                marker_color=DANGER,
                opacity=0.6,
                histnorm="probability density",
                nbinsx=35,
            )
        )
        fig.update_layout(
            barmode="overlay",
            title=f"Empirical Continuous Distribution: '{col_name}'",
            xaxis_title=col_name,
            yaxis_title="Probability Density",
            height=360,
            margin=dict(l=20, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    else:
        ref_vc = ref_df[col_name].dropna().astype(str).value_counts(normalize=True) * 100
        cur_vc = cur_df[col_name].dropna().astype(str).value_counts(normalize=True) * 100

        all_cats = sorted(list(set(ref_vc.index) | set(cur_vc.index)))
        ref_props = [ref_vc.get(c, 0.0) for c in all_cats]
        cur_props = [cur_vc.get(c, 0.0) for c in all_cats]

        fig.add_trace(
            go.Bar(
                name="Baseline Reference",
                x=all_cats,
                y=ref_props,
                marker_color=STEEL,
            )
        )
        fig.add_trace(
            go.Bar(
                name="Current Dataset",
                x=all_cats,
                y=cur_props,
                marker_color=DANGER,
            )
        )
        fig.update_layout(
            barmode="group",
            title=f"Categorical Frequency Distribution: '{col_name}'",
            xaxis_title="Category",
            yaxis_title="Proportion (%)",
            height=360,
            margin=dict(l=20, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_anomaly_scatter(
    cur_df: pd.DataFrame,
    anomaly_result: AnomalyDetectionResult,
    x_col: str,
    y_col: str,
) -> go.Figure:
    """Create 2D interactive scatter plot of features with highlighted anomalies.

    Parameters
    ----------
    cur_df : pd.DataFrame
        Current dataset.
    anomaly_result : AnomalyDetectionResult
        Isolation Forest output.
    x_col : str
        Feature on horizontal axis.
    y_col : str
        Feature on vertical axis.

    Returns
    -------
    go.Figure
        Plotly scatter figure.
    """
    plot_df = pd.DataFrame()
    plot_df[x_col] = pd.to_numeric(cur_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(cur_df[y_col], errors="coerce")

    # Map anomaly records by row_index
    rec_map = {r.row_index: r for r in anomaly_result.anomaly_records}

    statuses = []
    scores = []
    top_devs = []

    for idx in cur_df.index:
        rec = rec_map.get(idx)
        if rec:
            statuses.append("Suspicious Anomaly" if rec.is_anomaly else "Normal Observation")
            scores.append(round(rec.anomaly_score, 3))
            dev_str = ", ".join(f"{k}: {v:+.1f} IQR" for k, v in list(rec.feature_deviations.items())[:2])
            top_devs.append(dev_str)
        else:
            statuses.append("Normal Observation")
            scores.append(0.0)
            top_devs.append("N/A")

    plot_df["Status"] = statuses
    plot_df["Suspicion_Score"] = scores
    plot_df["Key_Deviations"] = top_devs
    plot_df["Row_ID"] = cur_df.index

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="Status",
        color_discrete_map={
            "Normal Observation": STEEL,
            "Suspicious Anomaly": DANGER,
        },
        hover_data=["Row_ID", "Suspicion_Score", "Key_Deviations"],
        title=f"Multivariate Feature Space: '{x_col}' vs '{y_col}'",
        opacity=0.8,
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_anomaly_score_distribution(anomaly_result: AnomalyDetectionResult) -> go.Figure:
    """Create histogram showing continuous suspicion scores in [0.0, 1.0].

    Parameters
    ----------
    anomaly_result : AnomalyDetectionResult
        Isolation Forest output.

    Returns
    -------
    go.Figure
        Plotly Histogram figure.
    """
    scores = [r.anomaly_score for r in anomaly_result.anomaly_records]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=30,
            marker_color=NAVY,
            name="Records",
            hovertemplate="Score range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    # Highlight threshold
    if anomaly_result.anomalies_detected_count > 0:
        anom_scores = [r.anomaly_score for r in anomaly_result.anomaly_records if r.is_anomaly]
        cutoff = min(anom_scores) if anom_scores else 0.5
        fig.add_vline(
            x=cutoff,
            line_width=2,
            line_dash="dash",
            line_color=DANGER,
            annotation_text=f"Outlier Cutoff ({cutoff:.3f})",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Distribution of Isolation Forest Suspicion Scores (0.0 = Nominal, 1.0 = Highly Isolated)",
        xaxis_title="Suspicion Score (s)",
        yaxis_title="Record Count",
        height=280,
        margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
