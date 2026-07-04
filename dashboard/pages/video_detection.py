"""
dashboard/pages/video_detection.py
====================================
Video DeepFake detection page.

User flow:
1. Upload video file
2. Select parameters (sample rate, max frames)
3. Run extraction & classification
4. Display premium summary card, charts, timeline, and report options
"""

import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from dashboard.components.charts import fake_probability_timeline, frame_distribution_pie
from dashboard.components.kpi_cards import render_confidence_bar, render_prediction_badge


def render(
    video_detector: "VideoDetector",   # type: ignore[name-defined]
    pdf_gen: "PDFReportGenerator",      # type: ignore[name-defined]
    repo: "DetectionRepository",        # type: ignore[name-defined]
    model_name: str = "XceptionNet",
) -> None:
    """
    Render the Video Detection page.

    Args:
        video_detector: Configured VideoDetector.
        pdf_gen:        PDF report generator.
        repo:           Detection history repository.
        model_name:     Model display name.
    """
    st.markdown("## 🎥 Video DeepFake Detection")
    st.markdown(
        "<p style='color: var(--text-secondary);'>Upload a video file to run frame-wise neural network analysis "
        "and temporal classification.</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_upload",
        label_visibility="collapsed",
    )

    if uploaded is None:
        _render_upload_placeholder()
        return

    # ── Settings panel ────────────────────────────────────────────────────────
    with st.expander("⚙️ Analysis Configuration Parameters", expanded=False):
        sample_rate = st.slider("Frame Sample Rate (extract every Nth frame)", 1, 30, 5, key="vid_sample_slider")
        max_frames = st.slider("Max Frames to Analyse", 10, 300, 100, key="vid_max_slider")

    # Layout preview + button
    col_preview, col_action = st.columns([1, 1], gap="medium")
    with col_preview:
        st.markdown("**🎥 Video Preview**")
        st.video(uploaded)

    with col_action:
        st.markdown("**⚙️ SaaS Video Analysis**")
        st.info("💡 Frame extraction is optimized for CPU/GPU memory safety.")
        
        run_clicked = st.button("▶️ Run Video Analysis", type="primary", use_container_width=True, key="run_vid_analysis_btn")

    if run_clicked:
        status_box = st.empty()
        from dashboard.components.loader import render_ai_loader
        
        with status_box.container():
            render_ai_loader("Caching video and initializing frame pipeline...")

        try:
            # Save uploaded video to a temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded.name).suffix
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with status_box.container():
                render_ai_loader("Analyzing facial frames with XceptionNet model...")

            # Override detector settings
            video_detector.sample_rate = sample_rate
            video_detector.max_frames = max_frames

            result = video_detector.predict(tmp_path)
            
            with status_box.container():
                render_ai_loader("Generating timeline probabilities...")
                
            status_box.empty()

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        except Exception as exc:
            st.error(f"❌ Video analysis failed: {exc}")
            status_box.empty()
            return

        # Store result in session state to persist through UI reruns
        st.session_state["last_video_result"] = result
        st.session_state["last_video_name"] = uploaded.name

    # ── Render active prediction result ───────────────────────────────────────
    if (
        st.session_state.get("last_video_name") == uploaded.name
        and "last_video_result" in st.session_state
    ):
        result = st.session_state["last_video_result"]
        
        st.divider()
        st.markdown("### 📊 Classification Summary")

        col_verdict, col_meta = st.columns([1, 1], gap="large")
        with col_verdict:
            render_prediction_badge(result.label, result.confidence)
            render_confidence_bar(result.fake_probability, result.label)

            # Video level probability breakdown
            real_pct = round(result.real_probability * 100, 1)
            fake_pct = round(result.fake_probability * 100, 1)

            st.markdown(
                f"""
                <div style="display:grid; grid-template-columns:1fr 1fr;
                            gap:0.75rem; margin-top:1.2rem; font-size:0.88rem; text-align:center;">
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius:12px; padding:0.8rem;">
                        <div style="color: var(--text-secondary); font-weight:600; font-size:0.78rem;">✅ VIDEO REAL PROB</div>
                        <div style="font-weight:800; color: var(--success); font-size:1.2rem; margin-top:0.2rem;">{real_pct}%</div>
                    </div>
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius:12px; padding:0.8rem;">
                        <div style="color: var(--text-secondary); font-weight:600; font-size:0.78rem;">⚠️ VIDEO FAKE PROB</div>
                        <div style="font-weight:800; color: var(--danger); font-size:1.2rem; margin-top:0.2rem;">{fake_pct}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_meta:
            st.markdown(
                f"""
                <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius:18px;
                            padding:1.2rem; font-size:0.9rem; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; line-height: 1.6;">
                        <div><span style="color: var(--text-secondary); font-weight:600;">📹 File:</span>
                             <span style="color: var(--text-primary); font-weight:700;"> {uploaded.name}</span></div>
                        <div><span style="color: var(--text-secondary); font-weight:600;">⏱ Duration:</span>
                             <span style="color: var(--text-primary); font-weight:700;"> {result.duration_seconds:.1f}s</span></div>
                        <div><span style="color: var(--text-secondary); font-weight:600;">🎞 FPS:</span>
                             <span style="color: var(--text-primary); font-weight:700;"> {result.fps:.1f}</span></div>
                        <div><span style="color: var(--text-secondary); font-weight:600;">🔍 Frames:</span>
                             <span style="color: var(--text-primary); font-weight:700;"> {result.total_frames_analysed}</span></div>
                        <div><span style="color: var(--danger); font-weight:600;">⚠️ Fake Frames:</span>
                             <span style="color: var(--danger); font-weight:700;"> {result.fake_frame_count}</span></div>
                        <div><span style="color: var(--success); font-weight:600;">✅ Real Frames:</span>
                             <span style="color: var(--success); font-weight:700;"> {result.real_frame_count}</span></div>
                        <div><span style="color: var(--text-secondary); font-weight:600;">⚡ Speed:</span>
                             <span style="color: var(--text-primary); font-weight:700;"> {result.processing_time_ms:.0f} ms</span></div>
                        <div><span style="color: var(--text-secondary); font-weight:600;">🤖 Model:</span>
                             <span style="color: var(--text-primary); font-weight:700;"> {model_name}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Charts ────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📈 Visual Metrics Timeline")
        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            if result.frame_predictions:
                fig_timeline = fake_probability_timeline(
                    timestamps_ms=result.timestamps,
                    fake_probs=result.fake_percentages,
                    title="Fake Probability Timeline",
                )
                fig_timeline.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=st.session_state.get("text_color", "#F8FAFC"),
                    margin=dict(t=30, b=10, l=10, r=10),
                    height=280,
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

        with col_chart2:
            if result.total_frames_analysed > 0:
                fig_pie = frame_distribution_pie(
                    fake_count=result.fake_frame_count,
                    real_count=result.real_frame_count,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=st.session_state.get("text_color", "#F8FAFC"),
                    margin=dict(t=30, b=10, l=10, r=10),
                    height=280,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # ── Per-frame table ───────────────────────────────────────────────────
        if result.frame_predictions:
            with st.expander("📋 Detailed Frame Probability Logs", expanded=False):
                import pandas as pd
                frame_df = pd.DataFrame(
                    [
                        {
                            "Frame": fp.frame_index,
                            "Time (s)": f"{fp.timestamp_ms/1000:.2f}",
                            "Prediction": fp.label,
                            "Confidence": f"{fp.confidence:.3f}",
                            "Fake Prob": f"{fp.fake_probability:.4f}",
                            "Face Detected": "✅" if fp.face_detected else "❌",
                        }
                        for fp in result.frame_predictions
                    ]
                )
                st.dataframe(frame_df, use_container_width=True, hide_index=True)

                csv = frame_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇_ Download CSV Log",
                    data=csv,
                    file_name=f"frames_{uploaded.name}.csv",
                    mime="text/csv",
                    key="frame_csv_dl",
                )

        # ── Save + PDF ────────────────────────────────────────────────────────
        st.divider()
        col_save, col_pdf = st.columns(2)

        if st.session_state.get("last_saved_vid_filename") != uploaded.name:
            st.session_state["last_saved_vid_filename"] = uploaded.name
            st.session_state["last_saved_vid_id"] = None

        with col_save:
            if st.session_state.get("last_saved_vid_id") is not None:
                st.success("✅ Saved to detection history!")
            else:
                if st.button("💾 Save to History", use_container_width=True, key="vid_save"):
                    try:
                        record_id = repo.insert_detection(
                            file_name=uploaded.name,
                            prediction=result.label,
                            confidence=result.confidence,
                            fake_probability=result.fake_frame_ratio,
                            file_type="video",
                            model_name=model_name,
                            processing_time=result.processing_time_ms,
                            num_frames=result.total_frames_analysed,
                            fake_frames=result.fake_frame_count,
                            real_frames=result.real_frame_count,
                        )
                        st.session_state["last_saved_vid_id"] = record_id
                        st.success("✅ Saved to detection history!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to save: {exc}")

        with col_pdf:
            if st.button("📄 Generate PDF Report", use_container_width=True, key="vid_pdf"):
                try:
                    record_id = st.session_state.get("last_saved_vid_id")
                    if record_id is None:
                        record_id = repo.insert_detection(
                            file_name=uploaded.name,
                            prediction=result.label,
                            confidence=result.confidence,
                            fake_probability=result.fake_frame_ratio,
                            file_type="video",
                            model_name=model_name,
                            processing_time=result.processing_time_ms,
                            num_frames=result.total_frames_analysed,
                            fake_frames=result.fake_frame_count,
                            real_frames=result.real_frame_count,
                        )
                        st.session_state["last_saved_vid_id"] = record_id

                    pdf_path = pdf_gen.generate(
                        file_name=uploaded.name,
                        prediction=result.label,
                        confidence=result.confidence,
                        processing_time_ms=result.processing_time_ms,
                        timestamp=datetime.now(),
                        num_frames=result.total_frames_analysed,
                        fake_frames=result.fake_frame_count,
                        real_frames=result.real_frame_count,
                        frame_confidences=result.fake_percentages,
                        model_name=model_name,
                    )
                    repo.update_report_path(record_id, str(pdf_path))
                    
                    with open(str(pdf_path), "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF Report",
                            data=f.read(),
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            key="vid_dl_btn",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")


def _render_upload_placeholder() -> None:
    """Show a placeholder UI before any file is uploaded."""
    st.markdown(
        """
        <div style="text-align:center; padding:5rem 3rem; border:2px dashed var(--border);
             border-radius:var(--radius); color: var(--text-secondary); margin-top:1rem; background: var(--bg-card);">
            <div style="font-size:3.5rem; margin-bottom: 1rem;">🎥</div>
            <div style="font-size:1.25rem; font-weight:700; margin:0.5rem 0; color: var(--text-primary);">
                Drop a video here or click to browse
            </div>
            <div style="font-size:0.88rem; opacity: 0.85;">Supported file formats: MP4 · AVI · MOV · MKV · WEBM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
