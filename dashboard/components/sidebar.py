"""
dashboard/components/sidebar.py
================================
Sidebar navigation component for the Streamlit dashboard.

Provides helper functions for rendering the sidebar branding,
navigation menu, and model status indicator.
"""

import streamlit as st
from pathlib import Path


def render_sidebar_branding() -> None:
    """Render the logo and project title in the sidebar."""
    st.markdown(
        """
        <div style="text-align:center; padding:1.2rem 0 0.5rem;">
            <div style="font-size:3rem; line-height:1;">🛡️</div>
            <div style="font-family:'Outfit',sans-serif; font-weight:800;
                        font-size:1.15rem; color:#5dade2; margin-top:0.4rem;
                        letter-spacing:0.5px;">
                DeepFake Detector
            </div>
            <div style="font-size:0.7rem; color:#7f8c8d; margin-top:0.2rem;">
                v1.0 · XceptionNet + Grad-CAM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_model_status(model_status: str, model_name: str) -> None:
    """
    Render the model status indicator in the sidebar.

    Args:
        model_status: Status string (e.g. ``"✅ Loaded: xceptionnet_final.keras"``).
        model_name:   Display name of the active model.
    """
    is_loaded = model_status.startswith("✅")
    colour = "#2ecc71" if is_loaded else "#f39c12"
    icon = "✅" if is_loaded else "⚠️"

    st.markdown(
        f"""
        <div style="background:rgba(30,39,56,0.7); border-radius:10px;
                    padding:0.8rem; font-size:0.78rem;
                    border-left: 3px solid {colour};">
            <div style="color:#95a5a6; margin-bottom:0.3rem; font-weight:600;">
                {icon} Model Status
            </div>
            <div style="color:#ecf0f1;">{model_name}</div>
            <div style="color:{colour}; font-size:0.72rem; margin-top:0.2rem;">
                {"Weights loaded" if is_loaded else "Demo mode — no weights found"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_stats(total: int, fake_count: int, today: int) -> None:
    """
    Render a compact statistics widget in the sidebar.

    Args:
        total:       Total detections ever.
        fake_count:  Total FAKE predictions.
        today:       Detections made today.
    """
    fake_pct = (fake_count / total * 100) if total else 0.0
    st.markdown(
        f"""
        <div style="background:rgba(30,39,56,0.5); border-radius:10px;
                    padding:0.8rem; font-size:0.78rem; margin-top:0.5rem;">
            <div style="color:#95a5a6; font-weight:600; margin-bottom:0.4rem;">
                📊 Quick Stats
            </div>
            <div style="display:flex; justify-content:space-between;
                        color:#ecf0f1; margin-bottom:0.2rem;">
                <span>Total</span><span style="font-weight:700;">{total:,}</span>
            </div>
            <div style="display:flex; justify-content:space-between;
                        color:#e74c3c; margin-bottom:0.2rem;">
                <span>Fake</span>
                <span style="font-weight:700;">{fake_count:,} ({fake_pct:.0f}%)</span>
            </div>
            <div style="display:flex; justify-content:space-between; color:#95a5a6;">
                <span>Today</span><span style="font-weight:700;">{today:,}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
