"""
dashboard/components/charts.py
================================
Interactive Plotly chart builders for the DeepFake dashboard.
"""

from typing import List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Shared theme ──────────────────────────────────────────────────────────────
_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(15,17,23,0)",
    plot_bgcolor="rgba(15,17,23,0)",
    font=dict(family="Inter, sans-serif", color="#ecf0f1"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
)


def fake_probability_timeline(
    timestamps_ms: List[float],
    fake_probs: List[float],
    threshold: float = 0.5,
    title: str = "Fake Probability Timeline",
) -> go.Figure:
    """
    Line chart of fake probability over video frames.

    Args:
        timestamps_ms: Frame timestamps in milliseconds.
        fake_probs:    Fake probability per frame (0–1).
        threshold:     Decision threshold line.
        title:         Chart title.

    Returns:
        Plotly :class:`go.Figure`.
    """
    seconds = [t / 1000 for t in timestamps_ms]

    fig = go.Figure()

    # Shaded fake region
    fig.add_hrect(
        y0=threshold, y1=1.0,
        fillcolor="rgba(231,76,60,0.08)",
        line_width=0,
    )

    # Fill area under curve
    fig.add_trace(
        go.Scatter(
            x=seconds,
            y=fake_probs,
            fill="tozeroy",
            fillcolor="rgba(41,128,185,0.12)",
            line=dict(color="#5dade2", width=2),
            name="Fake Probability",
            hovertemplate="Time: %{x:.2f}s<br>Fake Prob: %{y:.3f}<extra></extra>",
        )
    )

    # Threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color="#e74c3c", width=1.5, dash="dash"),
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top right",
        annotation_font_color="#e74c3c",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#ecf0f1")),
        xaxis_title="Time (seconds)",
        yaxis_title="Fake Probability",
        yaxis=dict(range=[0, 1], **_LAYOUT_DEFAULTS["yaxis"]),
        showlegend=False,
        **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def frame_distribution_pie(
    fake_count: int,
    real_count: int,
) -> go.Figure:
    """
    Donut chart showing real vs fake frame distribution.

    Args:
        fake_count: Number of fake frames.
        real_count: Number of real frames.

    Returns:
        Plotly :class:`go.Figure`.
    """
    fig = go.Figure(
        go.Pie(
            labels=["Fake", "Real"],
            values=[fake_count, real_count],
            hole=0.55,
            marker=dict(
                colors=["#e74c3c", "#2ecc71"],
                line=dict(color="rgba(0,0,0,0.3)", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=13, color="white"),
            hovertemplate="%{label}: %{value} frames (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Frame Distribution", font=dict(size=14, color="#ecf0f1")),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color="#ecf0f1"),
        ),
        **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def roc_curve_chart(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    model_name: str = "Model",
) -> go.Figure:
    """
    ROC curve with AUC annotation.

    Args:
        fpr:        False positive rates.
        tpr:        True positive rates.
        auc_score:  Area under curve.
        model_name: Label for the legend.

    Returns:
        Plotly :class:`go.Figure`.
    """
    fig = go.Figure()

    # Random baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="#7f8c8d", dash="dash", width=1),
            name="Random Baseline",
        )
    )

    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            line=dict(color="#5dade2", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(41,128,185,0.1)",
            name=f"{model_name} (AUC={auc_score:.3f})",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="ROC Curve", font=dict(size=14, color="#ecf0f1")),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(range=[0, 1.02], **_LAYOUT_DEFAULTS["yaxis"]),
        xaxis=dict(range=[0, 1], **_LAYOUT_DEFAULTS["xaxis"]),
        showlegend=True,
        legend=dict(font=dict(color="#ecf0f1")),
        **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def confidence_histogram(
    confidences: List[float],
    prediction: str = "FAKE",
) -> go.Figure:
    """
    Histogram of per-frame confidence scores.

    Args:
        confidences: List of confidence values (0–1).
        prediction:  Overall verdict for colour theming.

    Returns:
        Plotly :class:`go.Figure`.
    """
    colour = "#e74c3c" if prediction == "FAKE" else "#2ecc71"

    fig = go.Figure(
        go.Histogram(
            x=confidences,
            nbinsx=20,
            marker=dict(
                color=colour,
                opacity=0.75,
                line=dict(color="rgba(0,0,0,0.3)", width=1),
            ),
            hovertemplate="Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Confidence Distribution", font=dict(size=14, color="#ecf0f1")),
        xaxis_title="Confidence",
        yaxis_title="Frame Count",
        bargap=0.05,
        **_LAYOUT_DEFAULTS,
    )
    return fig


def model_comparison_bar(comparison_df: "pd.DataFrame") -> go.Figure:  # type: ignore[name-defined]
    """
    Grouped bar chart comparing model metrics.

    Args:
        comparison_df: DataFrame with columns: model, accuracy, precision,
                       recall, f1_score, auc.

    Returns:
        Plotly :class:`go.Figure`.
    """
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]
    colours = ["#5dade2", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c"]

    fig = go.Figure()
    for metric, colour in zip(metrics, colours):
        if metric in comparison_df.columns:
            fig.add_trace(
                go.Bar(
                    x=comparison_df["model"],
                    y=comparison_df[metric],
                    name=metric.replace("_", " ").title(),
                    marker_color=colour,
                    hovertemplate=f"{metric}: %{{y:.4f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(text="Model Comparison", font=dict(size=14, color="#ecf0f1")),
        barmode="group",
        yaxis=dict(range=[0, 1.05], **_LAYOUT_DEFAULTS["yaxis"]),
        legend=dict(font=dict(color="#ecf0f1")),
        **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")},
    )
    return fig
