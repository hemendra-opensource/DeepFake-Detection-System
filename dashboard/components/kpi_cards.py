"""
dashboard/components/kpi_cards.py
===================================
KPI metric card components for the Streamlit dashboard.

Probability display contract:
    fake_probability + real_probability == 1.0  (always)
    confidence == probability of the predicted class
    label    ∈ {"FAKE", "REAL"}

This module only renders — it never computes probabilities.
All values come from ImagePrediction / VideoPrediction.
"""

from typing import Optional
import streamlit as st


def render_kpi_cards(
    total: int,
    fake_count: int,
    real_count: int,
    avg_confidence: float,
    avg_processing_time: float,
    today_count: int,
) -> None:
    """
    Render 6 KPI metric cards in a responsive grid.

    Args:
        total:               Total detections ever made.
        fake_count:          Total FAKE predictions.
        real_count:          Total REAL predictions.
        avg_confidence:      Average confidence score [0, 1].
        avg_processing_time: Average processing time (ms).
        today_count:         Detections made today.
    """
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        st.metric(
            label="🔍 Total Detections",
            value=f"{total:,}",
            delta=f"+{today_count} today" if today_count else None,
        )
    with col2:
        fake_pct = (fake_count / total * 100) if total else 0
        st.metric(
            label="🚨 Fake Detected",
            value=f"{fake_count:,}",
            delta=f"{fake_pct:.1f}% of total",
            delta_color="inverse",
        )
    with col3:
        real_pct = (real_count / total * 100) if total else 0
        st.metric(
            label="✅ Real Verified",
            value=f"{real_count:,}",
            delta=f"{real_pct:.1f}% of total",
        )
    with col4:
        st.metric(
            label="🎯 Avg Confidence",
            value=f"{avg_confidence:.1%}",
        )
    with col5:
        st.metric(
            label="⚡ Avg Speed",
            value=f"{avg_processing_time:.0f} ms",
        )
    with col6:
        st.metric(
            label="📅 Today",
            value=f"{today_count:,}",
        )


def render_prediction_badge(label: str, confidence: float) -> None:
    """
    Render a large, styled prediction verdict badge.

    Args:
        label:      "FAKE" or "REAL".
        confidence: Probability of the predicted class [0, 1].
                    This is ALWAYS the confidence in the displayed label —
                    never the probability of the opposite class.
    """
    css_class = "fake-badge" if label == "FAKE" else "real-badge"
    icon = "⚠️" if label == "FAKE" else "✅"
    conf_color = "#e74c3c" if label == "FAKE" else "#2ecc71"

    st.markdown(
        f"""
        <div style="text-align:center; margin: 1.5rem 0;">
            <div class="{css_class}">{icon} {label}</div>
            <p style="color:#95a5a6; font-size:1.1rem; margin-top:0.8rem;">
                Confidence: <strong style="color:{conf_color}">
                {confidence:.1%}</strong>
            </p>
            <p style="color:#7f8c8d; font-size:0.8rem; margin-top:0.3rem;">
                (confidence = probability that this image is {label})
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_bar(
    fake_probability: float,
    label: str,
) -> None:
    """
    Render an animated horizontal confidence bar showing REAL vs FAKE probability.

    Args:
        fake_probability: P(FAKE) from the model, ∈ [0, 1].
        label:            "FAKE" or "REAL" — used for colour selection only.

    The bar always shows:
        Real%  = (1 − fake_probability) × 100
        Fake%  = fake_probability × 100
        Real% + Fake% = 100%  (mathematically guaranteed)
    """
    # Clamp to [0, 1] for display safety
    fake_prob = max(0.0, min(1.0, fake_probability))
    real_prob = 1.0 - fake_prob

    fake_pct = round(fake_prob * 100, 1)
    real_pct = round(real_prob * 100, 1)

    bar_class = "confidence-bar-fill-fake" if label == "FAKE" else "confidence-bar-fill-real"
    # The bar width represents fake_probability (red = fake)
    bar_width_pct = int(fake_prob * 100)

    st.markdown(
        f"""
        <div style="margin: 1rem 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem;
                        color:#95a5a6; margin-bottom:4px;">
                <span>✅ Real: {real_pct}%</span>
                <span>⚠️ Fake: {fake_pct}%</span>
            </div>
            <div class="confidence-bar-container">
                <div class="{bar_class}" style="width:{bar_width_pct}%"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.75rem;
                        color:#555; margin-top:4px;">
                <span>← More Real</span>
                <span>More Fake →</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
