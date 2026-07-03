"""
dashboard/pages/home.py
========================
Home / landing page of the DeepFake Detection dashboard.

Displays:
- Hero section with project branding
- KPI cards from detection history
- Quick navigation cards
- Recent detections table
"""

import streamlit as st


def render(repo: "DetectionRepository") -> None:  # type: ignore[name-defined]
    """
    Render the Home page.

    Args:
        repo: :class:`DetectionRepository` for KPI data.
    """
    from dashboard.components.kpi_cards import render_kpi_cards

    # ── Hero section ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:2.5rem 0 1.5rem;">
            <div style="font-size:3.5rem; margin-bottom:0.3rem;">🛡️</div>
            <h1 style="font-family:'Outfit',sans-serif; font-size:2.8rem;
                       font-weight:900; margin:0;
                       background:linear-gradient(135deg,#5dade2,#2ecc71);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                DeepFake Detection System
            </h1>
            <p style="color:#95a5a6; font-size:1.1rem; margin-top:0.5rem;">
                Powered by XceptionNet · Grad-CAM · Frame-wise Analysis
            </p>
            <div style="display:flex; justify-content:center; gap:0.5rem;
                        flex-wrap:wrap; margin-top:1rem;">
                <span style="background:rgba(41,128,185,0.2); color:#5dade2;
                             padding:0.3rem 0.9rem; border-radius:50px;
                             font-size:0.8rem; font-weight:600;">
                    🤖 AI-Powered
                </span>
                <span style="background:rgba(46,204,113,0.2); color:#2ecc71;
                             padding:0.3rem 0.9rem; border-radius:50px;
                             font-size:0.8rem; font-weight:600;">
                    🔬 Explainable
                </span>
                <span style="background:rgba(231,76,60,0.2); color:#e74c3c;
                             padding:0.3rem 0.9rem; border-radius:50px;
                             font-size:0.8rem; font-weight:600;">
                    ⚡ Real-time
                </span>
                <span style="background:rgba(155,89,182,0.2); color:#9b59b6;
                             padding:0.3rem 0.9rem; border-radius:50px;
                             font-size:0.8rem; font-weight:600;">
                    📊 Analytics
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.markdown("### 📊 Detection Statistics")
    try:
        stats = repo.get_statistics()
        render_kpi_cards(
            total=stats.get("total", 0),
            fake_count=stats.get("fake_count", 0),
            real_count=stats.get("real_count", 0),
            avg_confidence=stats.get("avg_confidence", 0.0),
            avg_processing_time=stats.get("avg_processing_time", 0.0),
            today_count=stats.get("today_count", 0),
        )
    except Exception as exc:
        st.warning(f"⚠️ Could not load statistics: {exc}")

    st.divider()

    # ── Feature cards ─────────────────────────────────────────────────────────
    st.markdown("### 🚀 Features")
    cols = st.columns(4)
    features = [
        ("🖼️", "Image Detection", "Detect DeepFakes in photos with Grad-CAM explanation"),
        ("🎬", "Video Analysis",  "Frame-wise analysis with temporal smoothing"),
        ("📡", "Live Webcam",     "Real-time detection from your webcam feed"),
        ("📦", "Batch Detection", "Process multiple files at once"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center; min-height:130px;">
                    <div style="font-size:2rem;">{icon}</div>
                    <div style="font-weight:700; margin:0.4rem 0; color:#ecf0f1;">
                        {title}
                    </div>
                    <div style="font-size:0.8rem; color:#95a5a6;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Recent detections ──────────────────────────────────────────────────────
    st.markdown("### 🕒 Recent Detections")
    try:
        df = repo.get_all(limit=10)
        if df.empty:
            st.info("No detections yet. Upload an image or video to get started.")
        else:
            display_cols = ["id", "file_name", "prediction", "confidence", "model_name", "timestamp"]
            display_df = df[[c for c in display_cols if c in df.columns]]
            st.dataframe(
                display_df.style.applymap(
                    lambda v: "color: #e74c3c; font-weight:bold"
                    if v == "FAKE"
                    else "color: #2ecc71; font-weight:bold"
                    if v == "REAL"
                    else "",
                    subset=["prediction"],
                ),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as exc:
        st.warning(f"⚠️ Could not load recent detections: {exc}")

    # ── Architecture overview ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏗️ System Architecture")
    st.markdown(
        """
        ```
        Input (Image / Video / Webcam)
               ↓
        Face Detection  [MediaPipe | OpenCV Haar]
               ↓
        Preprocessing   [Resize 299×299 · Normalize]
               ↓
        XceptionNet     [Pre-trained on ImageNet]
               ↓
        Grad-CAM        [Explainability Heatmap]
               ↓
        Prediction      [FAKE / REAL + Confidence]
               ↓
        Dashboard       [Analytics · PDF Report · History]
        ```
        """
    )
