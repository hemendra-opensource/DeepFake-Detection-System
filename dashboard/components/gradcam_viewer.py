"""
dashboard/components/gradcam_viewer.py
=======================================
Grad-CAM image viewer component for Streamlit.

Renders:
- Original image
- Colourised heatmap
- Overlay (blended)
Side by side with captions and explanation text.
"""

from typing import Optional

import numpy as np
import streamlit as st


def render_gradcam_viewer(
    original: np.ndarray,
    heatmap_colored: np.ndarray,
    overlay: np.ndarray,
    gradient_strength: float = 0.0,
    model_name: str = "XceptionNet",
    prediction: str = "FAKE",
) -> None:
    """
    Display three side-by-side Grad-CAM images with explanation.

    Args:
        original:          Original RGB image array.
        heatmap_colored:   Colourised heatmap RGB array.
        overlay:           Blended overlay RGB array.
        gradient_strength: Mean gradient strength (proxy for explanation quality).
        model_name:        Name of the model producing explanations.
        prediction:        Overall prediction (``"FAKE"`` or ``"REAL"``).
    """
    st.markdown("### 🔬 Grad-CAM Explainability")

    colour = "#e74c3c" if prediction == "FAKE" else "#2ecc71"

    st.markdown(
        f"""
        <div style="background:rgba(30,39,56,0.7); border:1px solid rgba(93,173,226,0.2);
             border-radius:12px; padding:1rem; margin-bottom:1rem;">
            <p style="color:#95a5a6; font-size:0.9rem; margin:0;">
                🧠 <strong style="color:{colour};">{model_name}</strong> attention map —
                <em>Red/yellow regions most influenced the {prediction} classification.</em>
                Gradient strength: <strong>{gradient_strength:.4f}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            original,
            caption="📷 Original Image",
            use_container_width=True,
        )
    with col2:
        st.image(
            heatmap_colored,
            caption="🌡️ Grad-CAM Heatmap",
            use_container_width=True,
        )
    with col3:
        st.image(
            overlay,
            caption="🔀 Overlay",
            use_container_width=True,
        )

    # Colour scale legend
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem;
             margin-top:0.5rem; font-size:0.8rem; color:#95a5a6;">
            <span>Low importance</span>
            <div style="flex:1; height:8px; border-radius:4px;
                 background:linear-gradient(90deg,
                     #00008b, #0000ff, #00ffff, #00ff00,
                     #ffff00, #ff8c00, #ff0000);"></div>
            <span>High importance</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gradcam_unavailable(reason: str = "No model loaded") -> None:
    """Show a placeholder when Grad-CAM cannot be generated."""
    st.markdown(
        f"""
        <div style="background:rgba(243,156,18,0.1); border:1px solid rgba(243,156,18,0.3);
             border-radius:12px; padding:1.5rem; text-align:center; color:#f39c12;">
            <div style="font-size:2rem;">🔬</div>
            <div style="font-weight:600; margin:0.5rem 0;">Grad-CAM Unavailable</div>
            <div style="font-size:0.85rem; color:#95a5a6;">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
