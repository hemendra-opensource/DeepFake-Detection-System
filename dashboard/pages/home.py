"""
dashboard/pages/home.py
========================
Home / landing page of the DeepFake Detection dashboard.
Redesigned as a modern AI SaaS landing page.
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
        <div style="text-align:center; padding:3rem 0 2rem; animation: fadeIn 0.5s ease-out;">
            <div style="font-size:4rem; margin-bottom:0.5rem; line-height: 1;">🛡️</div>
            <h1 style="font-family:'Outfit',sans-serif; font-size:3.5rem;
                       font-weight:900; margin:0;
                       background:linear-gradient(135deg, #3B82F6, #60A5FA, #22C55E);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                Explainable DeepFake Detection
            </h1>
            <p style="color: var(--text-secondary); font-size:1.25rem; margin-top:0.8rem; font-weight: 500; max-width: 800px; margin-left: auto; margin-right: auto;">
                Detect manipulated images and videos using Explainable AI powered by XceptionNet and Grad-CAM.
            </p>
            <div style="display:flex; justify-content:center; gap:0.6rem;
                        flex-wrap:wrap; margin-top:1.5rem;">
                <span style="background:rgba(59,130,246,0.15); color:#60A5FA;
                             padding:0.4rem 1.1rem; border-radius:50px;
                             font-size:0.82rem; font-weight:600; border: 1px solid rgba(59,130,246,0.25);">
                    🤖 XceptionNet Engine
                </span>
                <span style="background:rgba(34,197,94,0.15); color:#22C55E;
                             padding:0.4rem 1.1rem; border-radius:50px;
                             font-size:0.82rem; font-weight:600; border: 1px solid rgba(34,197,94,0.25);">
                    🔬 Grad-CAM Explanations
                </span>
                <span style="background:rgba(239,68,68,0.15); color:#EF4444;
                             padding:0.4rem 1.1rem; border-radius:50px;
                             font-size:0.82rem; font-weight:600; border: 1px solid rgba(239,68,68,0.25);">
                    ⚡ Real-Time Inference
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Quick Navigation buttons ──────────────────────────────────────────────
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button("🖼️ Upload Image", use_container_width=True, key="home_btn_image"):
            st.session_state["home_redirect"] = "image"
            st.rerun()
    with col_b2:
        if st.button("🎥 Upload Video", use_container_width=True, key="home_btn_video"):
            st.session_state["home_redirect"] = "video"
            st.rerun()
    with col_b3:
        if st.button("📡 Start Webcam", use_container_width=True, key="home_btn_webcam"):
            st.session_state["home_redirect"] = "webcam"
            st.rerun()

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
    st.markdown("### 🚀 Core Capabilities")
    cols = st.columns(4)
    features = [
        ("🖼️", "Image Detection", "Detect DeepFakes in photos with detailed Grad-CAM visual activation heatmaps."),
        ("🎥", "Video Analysis",  "Frame-by-frame deep neural network analysis with temporal probability smoothing."),
        ("📷", "Live Webcam",     "Capture live frames from connected cameras and run real-time inference."),
        ("📦", "Batch Processing", "Upload multiple media assets simultaneously and download aggregated CSV/JSON reports."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center; min-height:160px; display:flex; flex-direction:column; justify-content:center; align-items:center; padding:1.2rem;">
                    <div style="font-size:2.2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-weight:700; margin:0.4rem 0; color: var(--text-primary); font-size:1.05rem;">
                        {title}
                    </div>
                    <div style="font-size:0.82rem; color: var(--text-secondary); line-height:1.4;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Recent detections ──────────────────────────────────────────────────────
    st.markdown("### 🕒 Recent Analysis Log")
    try:
        df = repo.get_all(limit=10)
        if df.empty:
            st.info("No detections logged yet. Run a detection to populate history.")
        else:
            display_cols = ["id", "file_name", "prediction", "confidence", "model_name", "timestamp"]
            display_df = df[[c for c in display_cols if c in df.columns]]
            
            # Reformat df columns for prettier headers
            display_df = display_df.rename(columns={
                "id": "ID",
                "file_name": "File Name",
                "prediction": "Prediction",
                "confidence": "Confidence",
                "model_name": "Model",
                "timestamp": "Timestamp"
            })
            
            def highlight_row(val):
                if val == "FAKE":
                    return "color: var(--danger); font-weight:bold"
                if val == "REAL":
                    return "color: var(--success); font-weight:bold"
                return ""

            st.dataframe(
                display_df.style.applymap(highlight_row, subset=["Prediction"]),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as exc:
        st.warning(f"⚠️ Could not load recent detections: {exc}")

    # ── Premium Footer ────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; color: var(--text-secondary); font-size:0.82rem; padding:1rem 0;">
            🛡️ <b>DeepFake Detection AI</b> &nbsp;|&nbsp; Built with <b>TensorFlow</b> · <b>OpenCV</b> · <b>Streamlit</b> · <b>Grad-CAM</b><br>
            <span style="opacity:0.7;">Explainable Neural Networks for Media Integrity Verification · Final Year Internship Project</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
