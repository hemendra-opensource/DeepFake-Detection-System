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

    fake_pct = (fake_count / total * 100) if total else 0.0
    real_pct = (real_count / total * 100) if total else 0.0

    with col1:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid var(--primary);">
                <div style="color: var(--text-secondary); font-size: 0.82rem; font-weight: 600;">🔍 TOTAL DETECTIONS</div>
                <div style="font-family: var(--font-heading); font-size: 1.8rem; font-weight: 800; color: var(--primary-light); margin: 0.3rem 0;">
                    {total:,}
                </div>
                <div style="color: var(--text-secondary); font-size: 0.72rem;">+{today_count} added today</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid var(--danger);">
                <div style="color: var(--text-secondary); font-size: 0.82rem; font-weight: 600;">🚨 FAKE DETECTED</div>
                <div style="font-family: var(--font-heading); font-size: 1.8rem; font-weight: 800; color: var(--danger); margin: 0.3rem 0;">
                    {fake_count:,}
                </div>
                <div style="color: var(--text-secondary); font-size: 0.72rem;">{fake_pct:.1f}% of total</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid var(--success);">
                <div style="color: var(--text-secondary); font-size: 0.82rem; font-weight: 600;">✅ REAL VERIFIED</div>
                <div style="font-family: var(--font-heading); font-size: 1.8rem; font-weight: 800; color: var(--success); margin: 0.3rem 0;">
                    {real_count:,}
                </div>
                <div style="color: var(--text-secondary); font-size: 0.72rem;">{real_pct:.1f}% of total</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid var(--warning);">
                <div style="color: var(--text-secondary); font-size: 0.82rem; font-weight: 600;">🎯 AVG CONFIDENCE</div>
                <div style="font-family: var(--font-heading); font-size: 1.8rem; font-weight: 800; color: var(--warning); margin: 0.3rem 0;">
                    {avg_confidence:.1%}
                </div>
                <div style="color: var(--text-secondary); font-size: 0.72rem;">Model prediction confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid var(--primary-light);">
                <div style="color: var(--text-secondary); font-size: 0.82rem; font-weight: 600;">⚡ AVG SPEED</div>
                <div style="font-family: var(--font-heading); font-size: 1.8rem; font-weight: 800; color: var(--primary-light); margin: 0.3rem 0;">
                    {avg_processing_time:.0f} ms
                </div>
                <div style="color: var(--text-secondary); font-size: 0.72rem;">Average inference latency</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid var(--text-primary);">
                <div style="color: var(--text-secondary); font-size: 0.82rem; font-weight: 600;">📅 TODAY'S RUNS</div>
                <div style="font-family: var(--font-heading); font-size: 1.8rem; font-weight: 800; color: var(--text-primary); margin: 0.3rem 0;">
                    {today_count:,}
                </div>
                <div style="color: var(--text-secondary); font-size: 0.72rem;">Logged in last 24h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_badge(label: str, confidence: float) -> None:
    """
    Render a large, styled prediction verdict badge.

    Args:
        label:      "FAKE" or "REAL".
        confidence: Probability of the predicted class [0, 1].
    """
    css_class = "fake-badge" if label == "FAKE" else "real-badge"
    icon = "⚠️" if label == "FAKE" else "✅"
    conf_color = "var(--danger)" if label == "FAKE" else "var(--success)"

    st.markdown(
        f"""
        <div style="text-align:center; margin: 1.5rem 0; animation: fadeIn 0.4s ease-out;">
            <div class="{css_class}">{icon} {label}</div>
            <p style="color: var(--text-secondary); font-size:1.15rem; margin-top:1.2rem; font-weight: 500;">
                Confidence: <strong style="color:{conf_color}; font-size:1.3rem;">
                {confidence:.1%}</strong>
            </p>
            <p style="color: var(--text-secondary); font-size:0.8rem; opacity: 0.7; margin-top:0.3rem;">
                (probability that this input is {label})
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
    """
    fake_prob = max(0.0, min(1.0, fake_probability))
    real_prob = 1.0 - fake_prob

    fake_pct = round(fake_prob * 100, 1)
    real_pct = round(real_prob * 100, 1)

    bar_class = "confidence-bar-fill-fake" if label == "FAKE" else "confidence-bar-fill-real"
    bar_width_pct = int(fake_prob * 100)

    st.markdown(
        f"""
        <div style="margin: 1.5rem 0; padding: 1rem; background: rgba(0,0,0,0.1); border-radius: 12px; border: 1px solid var(--border);">
            <div style="display:flex; justify-content:space-between; font-size:0.9rem;
                        color: var(--text-primary); font-weight: 600; margin-bottom:6px;">
                <span style="color: var(--success);">✅ Real: {real_pct}%</span>
                <span style="color: var(--danger);">⚠️ Fake: {fake_pct}%</span>
            </div>
            <div class="confidence-bar-container">
                <div class="{bar_class}" style="width:{bar_width_pct}%"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.75rem;
                        color: var(--text-secondary); opacity: 0.8; margin-top:6px;">
                <span>← Real Verdict Range</span>
                <span>Fake Verdict Range →</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
