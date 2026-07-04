"""
dashboard/pages/batch_detection.py
====================================
Batch image/video detection page.

Features:
- Mixed file uploads (images + videos in one batch)
- Per-file independent processing with premium status loaders
- KPI summary cards (Total / Fake / Real / Errors / Avg Confidence)
- Sortable + filterable results table with glassmorphic styles
- Download results as CSV or JSON
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st


def render(
    image_detector: "ImageDetector",   # type: ignore[name-defined]
    video_detector: "VideoDetector",   # type: ignore[name-defined]
    repo: "DetectionRepository",        # type: ignore[name-defined]
    model_name: str = "XceptionNet",
) -> None:
    """
    Render the Batch Detection page.

    Args:
        image_detector: Loaded :class:`ImageDetector`.
        video_detector: Loaded :class:`VideoDetector`.
        repo:           Detection history repository.
        model_name:     Display name of the model.
    """
    import tempfile
    from dashboard.components.loader import render_ai_loader

    st.markdown("## 📦 Batch Detection")
    st.markdown(
        "<p style='color: var(--text-secondary);'>Upload multiple images or videos to process "
        "all at once. Results are saved to history automatically.</p>",
        unsafe_allow_html=True,
    )

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload Files (Images & Videos)",
        type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        key="batch_upload",
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown(
            """
            <div style="text-align:center; padding:5rem 3rem;
                 border:2px dashed var(--border);
                 border-radius:var(--radius); color: var(--text-secondary); margin-top:1rem; background: var(--bg-card);">
                <div style="font-size:3.5rem; margin-bottom: 1rem;">📦</div>
                <div style="font-size:1.25rem; font-weight:700; margin:0.5rem 0; color: var(--text-primary);">
                    Upload Multiple Files
                </div>
                <div style="font-size:0.88rem; opacity: 0.85;">
                    Supported formats: JPG · PNG · BMP · WEBP · MP4 · AVI · MOV · MKV
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    n = len(uploaded_files)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    n_images = sum(
        1 for f in uploaded_files
        if Path(f.name).suffix.lower() in image_exts
    )
    n_videos = n - n_images

    # File summary stats
    st.markdown("### 📂 Selected Workload Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left:4px solid var(--primary); text-align:center;">
                <div style="color:var(--text-secondary); font-size:0.8rem; font-weight:600;">TOTAL FILES</div>
                <div style="font-size:1.8rem; font-weight:800; color:var(--text-primary); margin-top:0.2rem;">{n}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left:4px solid var(--primary-light); text-align:center;">
                <div style="color:var(--text-secondary); font-size:0.8rem; font-weight:600;">IMAGES</div>
                <div style="font-size:1.8rem; font-weight:800; color:var(--primary-light); margin-top:0.2rem;">{n_images}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left:4px solid var(--accent); text-align:center;">
                <div style="color:var(--text-secondary); font-size:0.8rem; font-weight:600;">VIDEOS</div>
                <div style="font-size:1.8rem; font-weight:800; color:var(--accent); margin-top:0.2rem;">{n_videos}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("▶️ Run Batch Detection", type="primary", use_container_width=True, key="run_batch_btn"):
        results: List[dict] = []
        overall_bar = st.progress(0, text="Starting batch…")
        status_box = st.empty()
        t_batch_start = time.perf_counter()

        for i, uf in enumerate(uploaded_files):
            pct = int((i / n) * 100)
            overall_bar.progress(pct, text=f"[{i+1}/{n}] Processing {uf.name}…")
            
            with status_box.container():
                render_ai_loader(f"Analysing batch element {i+1} of {n}: {uf.name}...")

            ext = Path(uf.name).suffix.lower()
            row: dict = {
                "file_name": uf.name,
                "file_type": "unknown",
                "prediction": "ERROR",
                "confidence_%": 0.0,
                "fake_prob_%": 0.0,
                "real_prob_%": 0.0,
                "processing_ms": 0.0,
                "face_detected": False,
                "error": "",
            }

            try:
                if ext in image_exts:
                    row["file_type"] = "image"
                    pred = image_detector.predict_from_bytes(uf.read())
                    row.update(
                        prediction=pred.label,
                        **_prob_fields(pred),
                        processing_ms=round(pred.processing_time_ms, 1),
                        face_detected=pred.face_detected,
                    )

                elif ext in video_exts:
                    row["file_type"] = "video"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    vid_pred = video_detector.predict(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)
                    row.update(
                        prediction=vid_pred.label,
                        **_vid_prob_fields(vid_pred),
                        processing_ms=round(vid_pred.processing_time_ms, 1),
                        face_detected=True,
                    )

                else:
                    row["error"] = "Unsupported file type"

                # Save to history
                if row["prediction"] in ("FAKE", "REAL"):
                    repo.insert_detection(
                        file_name=row["file_name"],
                        prediction=row["prediction"],
                        confidence=row["confidence_%"] / 100.0,
                        fake_probability=row["fake_prob_%"] / 100.0,
                        file_type=row["file_type"],
                        model_name=model_name,
                        processing_time=row["processing_ms"],
                    )

            except Exception as exc:
                row["error"] = str(exc)

            results.append(row)

        overall_bar.progress(100, text="✅ Batch complete!")
        status_box.empty()

        t_batch_ms = (time.perf_counter() - t_batch_start) * 1000

        # ── Build DataFrame ────────────────────────────────────────────────────
        df = pd.DataFrame(results)

        total = len(df)
        fake_n = (df["prediction"] == "FAKE").sum()
        real_n = (df["prediction"] == "REAL").sum()
        errors = (df["prediction"] == "ERROR").sum()
        avg_conf = (
            df.loc[df["prediction"].isin(["FAKE", "REAL"]), "confidence_%"].mean()
            if total > errors else 0.0
        )

        # ── KPI Cards ─────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 Batch Summary")

        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        with kc1:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left:4px solid var(--text-primary); text-align:center; padding:1rem 0.5rem;">
                    <div style="color:var(--text-secondary); font-size:0.75rem; font-weight:600;">TOTAL RUN</div>
                    <div style="font-size:1.6rem; font-weight:800; color:var(--text-primary); margin-top:0.2rem;">{total}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kc2:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left:4px solid var(--danger); text-align:center; padding:1rem 0.5rem;">
                    <div style="color:var(--text-secondary); font-size:0.75rem; font-weight:600;">🚨 FAKE</div>
                    <div style="font-size:1.6rem; font-weight:800; color:var(--danger); margin-top:0.2rem;">{fake_n}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kc3:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left:4px solid var(--success); text-align:center; padding:1rem 0.5rem;">
                    <div style="color:var(--text-secondary); font-size:0.75rem; font-weight:600;">✅ REAL</div>
                    <div style="font-size:1.6rem; font-weight:800; color:var(--success); margin-top:0.2rem;">{real_n}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kc4:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left:4px solid var(--warning); text-align:center; padding:1rem 0.5rem;">
                    <div style="color:var(--text-secondary); font-size:0.75rem; font-weight:600;">❌ ERRORS</div>
                    <div style="font-size:1.6rem; font-weight:800; color:var(--warning); margin-top:0.2rem;">{errors}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kc5:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left:4px solid var(--primary-light); text-align:center; padding:1rem 0.5rem;">
                    <div style="color:var(--text-secondary); font-size:0.75rem; font-weight:600;">📈 AVG CONF</div>
                    <div style="font-size:1.6rem; font-weight:800; color:var(--primary-light); margin-top:0.2rem;">{avg_conf:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            f"⏱️ Total batch time: **{t_batch_ms/1000:.1f}s** | "
            f"Avg per file: **{t_batch_ms/total:.0f}ms**"
        )

        # Donut Distribution Chart
        if total > errors:
            import plotly.express as px

            fig = px.pie(
                values=[fake_n, real_n],
                names=["FAKE", "REAL"],
                color=["FAKE", "REAL"],
                color_discrete_map={"FAKE": "#EF4444", "REAL": "#22C55E"},
                hole=0.55,
                title="Fake vs Real Distribution",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=st.session_state.get("text_color", "#F8FAFC"),
                margin=dict(t=40, b=0, l=0, r=0),
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Results table ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📋 Detailed Results Grid")

        col_flt1, col_flt2, col_flt3 = st.columns(3)
        with col_flt1:
            pred_filter = st.selectbox(
                "Filter Prediction", ["All", "FAKE", "REAL", "ERROR"],
                key="batch_pred_filter",
            )
        with col_flt2:
            type_filter = st.selectbox(
                "Filter File Type", ["All", "image", "video"],
                key="batch_type_filter",
            )
        with col_flt3:
            sort_col = st.selectbox(
                "Sort By Column",
                ["file_name", "prediction", "confidence_%", "fake_prob_%", "processing_ms"],
                key="batch_sort_col",
            )

        filtered_df = df.copy()
        if pred_filter != "All":
            filtered_df = filtered_df[filtered_df["prediction"] == pred_filter]
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df["file_type"] == type_filter]

        filtered_df = filtered_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

        def _colour_pred(val: str) -> str:
            if val == "FAKE":
                return "color: var(--danger); font-weight:bold"
            if val == "REAL":
                return "color: var(--success); font-weight:bold"
            return "color: var(--text-secondary)"

        st.markdown(f"Showing **{len(filtered_df)}** of **{total}** files")
        st.dataframe(
            filtered_df.style.applymap(_colour_pred, subset=["prediction"]),
            use_container_width=True,
            hide_index=True,
        )

        # Exports
        st.divider()
        st.markdown("### ⬇️ Export Batch Run")
        dl1, dl2 = st.columns(2)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        with dl1:
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download CSV Log",
                data=csv,
                file_name=f"batch_results_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
                key="batch_csv_dl_btn",
            )

        with dl2:
            json_data = filtered_df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                "{ } Download JSON Log",
                data=json_data,
                file_name=f"batch_results_{ts}.json",
                mime="application/json",
                use_container_width=True,
                key="batch_json_dl_btn",
            )


# ── Private helpers ───────────────────────────────────────────────────────────

def _prob_fields(pred: "ImagePrediction") -> dict:  # type: ignore[name-defined]
    return {
        "confidence_%": round(pred.confidence * 100, 2),
        "fake_prob_%":  round(pred.fake_probability * 100, 2),
        "real_prob_%":  round(pred.real_probability * 100, 2),
    }


def _vid_prob_fields(pred: "VideoPrediction") -> dict:  # type: ignore[name-defined]
    return {
        "confidence_%": round(pred.confidence * 100, 2),
        "fake_prob_%":  round(pred.fake_probability * 100, 2),
        "real_prob_%":  round(pred.real_probability * 100, 2),
    }
