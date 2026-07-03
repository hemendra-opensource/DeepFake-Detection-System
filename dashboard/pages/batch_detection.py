"""
dashboard/pages/batch_detection.py
====================================
Batch image/video detection page.
Upload multiple files, process all, download results as CSV.
"""

import time
from datetime import datetime
from typing import List

import pandas as pd
import streamlit as st


def render(
    image_detector: "ImageDetector",  # type: ignore[name-defined]
    video_detector: "VideoDetector",  # type: ignore[name-defined]
    repo: "DetectionRepository",       # type: ignore[name-defined]
    model_name: str = "XceptionNet",
) -> None:
    """
    Render the Batch Detection page.

    Args:
        image_detector: Loaded image detector.
        video_detector: Loaded video detector.
        repo:           Detection history repository.
        model_name:     Display name of the model.
    """
    import tempfile
    from pathlib import Path

    st.markdown("## 📦 Batch Detection")
    st.markdown(
        "<p style='color:#95a5a6;'>Upload multiple images or videos to process all at once.</p>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload Files (Images & Videos)",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        key="batch_upload",
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown(
            """
            <div style="text-align:center; padding:3rem; border:2px dashed rgba(93,173,226,0.3);
                 border-radius:16px; color:#95a5a6; margin-top:1rem;">
                <div style="font-size:3rem;">📦</div>
                <div style="font-size:1.1rem; font-weight:600; margin:0.5rem 0;">
                    Upload Multiple Files
                </div>
                <div style="font-size:0.85rem;">
                    Select several images and/or videos to analyse in batch
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"**{len(uploaded_files)} file(s) selected.**")

    if st.button("▶️ Run Batch Detection", type="primary", use_container_width=True):
        results: List[dict] = []
        overall_progress = st.progress(0, text="Starting batch…")
        status_area = st.empty()

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        for i, uploaded_file in enumerate(uploaded_files):
            pct = int((i / len(uploaded_files)) * 100)
            overall_progress.progress(pct, text=f"Processing {uploaded_file.name}…")
            status_area.info(f"🔍 Analysing: **{uploaded_file.name}**")

            ext = Path(uploaded_file.name).suffix.lower()
            row = {
                "file_name": uploaded_file.name,
                "file_type": "unknown",
                "prediction": "ERROR",
                "confidence": 0.0,
                "fake_probability": 0.0,
                "processing_time_ms": 0.0,
                "error": "",
            }

            try:
                if ext in image_exts:
                    row["file_type"] = "image"
                    pred = image_detector.predict_from_bytes(uploaded_file.read())
                    row.update(
                        prediction=pred.label,
                        confidence=round(pred.confidence, 4),
                        fake_probability=round(pred.fake_probability, 4),
                        processing_time_ms=round(pred.processing_time_ms, 1),
                    )

                elif ext in video_exts:
                    row["file_type"] = "video"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    vid_pred = video_detector.predict(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)
                    row.update(
                        prediction=vid_pred.label,
                        confidence=round(vid_pred.confidence, 4),
                        fake_probability=round(vid_pred.fake_frame_ratio, 4),
                        processing_time_ms=round(vid_pred.processing_time_ms, 1),
                    )

                else:
                    row["error"] = "Unsupported file type"

                # Save to history
                if row["prediction"] in ("FAKE", "REAL"):
                    repo.insert_detection(
                        file_name=row["file_name"],
                        prediction=row["prediction"],
                        confidence=row["confidence"],
                        fake_probability=row["fake_probability"],
                        file_type=row["file_type"],
                        model_name=model_name,
                        processing_time=row["processing_time_ms"],
                    )

            except Exception as exc:
                row["error"] = str(exc)

            results.append(row)

        overall_progress.progress(100, text="Batch complete!")
        status_area.empty()

        # ── Results table ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 Batch Results")

        df = pd.DataFrame(results)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        total = len(df)
        fake_n = (df["prediction"] == "FAKE").sum()
        real_n = (df["prediction"] == "REAL").sum()
        errors = (df["prediction"] == "ERROR").sum()
        with col1:
            st.metric("Total Files", total)
        with col2:
            st.metric("🚨 Fake", fake_n)
        with col3:
            st.metric("✅ Real", real_n)
        with col4:
            st.metric("❌ Errors", errors)

        st.dataframe(df, use_container_width=True, hide_index=True)

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
