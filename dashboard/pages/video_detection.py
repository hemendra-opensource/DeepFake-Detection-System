"""
dashboard/pages/video_detection.py
====================================
Video DeepFake detection page.

Features:
- Upload video file
- Frame-wise analysis with progress bar
- Fake probability timeline chart
- Frame distribution pie chart
- Per-frame results table
- PDF report download
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st


def render(
    video_detector: "VideoDetector",   # type: ignore[name-defined]
    pdf_gen: "PDFReportGenerator",      # type: ignore[name-defined]
    repo: "DetectionRepository",        # type: ignore[name-defined]
    model_name: str = "XceptionNet",
) -> None:
    """
    Render the Video Detection page.

    Args:
        video_detector: Loaded video detector.
        pdf_gen:        PDF report generator.
        repo:           Detection history repository.
        model_name:     Display name of the model.
    """
    from dashboard.components.kpi_cards import render_prediction_badge, render_confidence_bar
    from dashboard.components.charts import (
        fake_probability_timeline,
        frame_distribution_pie,
        confidence_histogram,
    )

    st.markdown("## 🎬 Video DeepFake Detection")
    st.markdown(
        "<p style='color:#95a5a6;'>Upload a video to analyse frame-by-frame. "
        "Supported: MP4, AVI, MOV, MKV</p>",
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
    with st.expander("⚙️ Detection Settings", expanded=False):
        sample_rate = st.slider("Frame Sample Rate (every N frames)", 1, 30, 5)
        max_frames = st.slider("Max Frames to Analyse", 10, 300, 100)

    if st.button("▶️ Run Video Analysis", type="primary", use_container_width=True):
        progress = st.progress(0, text="Saving video to temp file…")

        try:
            # Save uploaded video to a temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded.name).suffix
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            progress.progress(10, text="Extracting frames…")

            # Override detector settings
            video_detector.sample_rate = sample_rate
            video_detector.max_frames = max_frames

            result = video_detector.predict(tmp_path)
            progress.progress(100, text="Analysis complete!")

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        except Exception as exc:
            st.error(f"❌ Video analysis failed: {exc}")
            progress.empty()
            return

        # ── Results ───────────────────────────────────────────────────────────
        st.divider()

        col_verdict, col_meta = st.columns([1, 1])
        with col_verdict:
            render_prediction_badge(result.label, result.confidence)
            render_confidence_bar(result.fake_frame_ratio, result.label)

        with col_meta:
            st.markdown(
                f"""
                <div style="background:rgba(30,39,56,0.7); border-radius:12px;
                            padding:1rem; font-size:0.9rem;">
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;">
                        <div><span style="color:#95a5a6;">📹 File:</span>
                             <span style="color:#ecf0f1;"> {uploaded.name}</span></div>
                        <div><span style="color:#95a5a6;">⏱ Duration:</span>
                             <span style="color:#ecf0f1;">
                             {result.duration_seconds:.1f}s</span></div>
                        <div><span style="color:#95a5a6;">🎞 FPS:</span>
                             <span style="color:#ecf0f1;"> {result.fps:.1f}</span></div>
                        <div><span style="color:#95a5a6;">🔍 Frames:</span>
                             <span style="color:#ecf0f1;">
                             {result.total_frames_analysed}</span></div>
                        <div><span style="color:#e74c3c;">⚠ Fake:</span>
                             <span style="color:#e74c3c;">
                             {result.fake_frame_count}</span></div>
                        <div><span style="color:#2ecc71;">✅ Real:</span>
                             <span style="color:#2ecc71;">
                             {result.real_frame_count}</span></div>
                        <div><span style="color:#95a5a6;">⚡ Time:</span>
                             <span style="color:#ecf0f1;">
                             {result.processing_time_ms:.0f} ms</span></div>
                        <div><span style="color:#95a5a6;">🤖 Model:</span>
                             <span style="color:#ecf0f1;"> {model_name}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Charts ────────────────────────────────────────────────────────────
        st.divider()
        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            if result.frame_predictions:
                fig_timeline = fake_probability_timeline(
                    timestamps_ms=result.timestamps,
                    fake_probs=result.fake_percentages,
                    title="Fake Probability over Time",
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

        with col_chart2:
            if result.total_frames_analysed > 0:
                fig_pie = frame_distribution_pie(
                    fake_count=result.fake_frame_count,
                    real_count=result.real_frame_count,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # ── Per-frame table ───────────────────────────────────────────────────
        if result.frame_predictions:
            with st.expander("📋 Per-Frame Results", expanded=False):
                import pandas as pd
                frame_df = pd.DataFrame(
                    [
                        {
                            "Frame": fp.frame_index,
                            "Time (s)": f"{fp.timestamp_ms/1000:.2f}",
                            "Prediction": fp.label,
                            "Confidence": f"{fp.confidence:.3f}",
                            "Fake Prob": f"{fp.fake_probability:.4f}",
                            "Smoothed": f"{fp.smoothed_fake_prob:.4f}",
                            "Face": "✅" if fp.face_detected else "❌",
                        }
                        for fp in result.frame_predictions
                    ]
                )
                st.dataframe(frame_df, use_container_width=True, hide_index=True)

                csv = frame_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv,
                    file_name=f"frames_{uploaded.name}.csv",
                    mime="text/csv",
                )

        # ── Save + PDF ────────────────────────────────────────────────────────
        st.divider()
        col_save, col_pdf = st.columns(2)

        with col_save:
            if st.button("💾 Save to History", use_container_width=True, key="vid_save"):
                try:
                    repo.insert_detection(
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
                    st.success("✅ Saved to detection history!")
                except Exception as exc:
                    st.error(f"Failed to save: {exc}")

        with col_pdf:
            if st.button("📄 Generate PDF Report", use_container_width=True, key="vid_pdf"):
                try:
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
                    with open(str(pdf_path), "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF",
                            data=f.read(),
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")


def _render_upload_placeholder() -> None:
    st.markdown(
        """
        <div style="text-align:center; padding:3rem; border:2px dashed rgba(93,173,226,0.3);
             border-radius:16px; color:#95a5a6; margin-top:1rem;">
            <div style="font-size:3rem;">🎬</div>
            <div style="font-size:1.1rem; font-weight:600; margin:0.5rem 0;">
                Drop a video here or click to upload
            </div>
            <div style="font-size:0.85rem;">Supported: MP4, AVI, MOV, MKV, WEBM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
