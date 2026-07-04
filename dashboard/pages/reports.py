"""
dashboard/pages/reports.py
============================
Reports page — list and generate PDF reports from detection history.

Features:
- Live stats container (Total records, PDFs generated)
- Interactive table of all database records
- Generate PDF report on-demand for any past detection (webcam, batch, image, video)
- Direct PDF downloads
- Bulk database export (CSV and JSON format)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from database.repository import DetectionRepository
from reports.pdf_generator import PDFReportGenerator


def render(
    reports_dir: str = "outputs/reports",
    repo: Optional[DetectionRepository] = None,
    pdf_gen: Optional[PDFReportGenerator] = None,
) -> None:
    """
    Render the Reports page.

    Args:
        reports_dir: Directory where PDF reports are saved.
        repo:        Loaded DetectionRepository.
        pdf_gen:     Loaded PDFReportGenerator.
    """
    st.markdown("## 📄 Detection Reports")
    st.markdown(
        "<p style='color:#95a5a6;'>Generate, view, and download professional "
        "PDF analysis reports from your detection history.</p>",
        unsafe_allow_html=True,
    )

    if repo is None or pdf_gen is None:
        st.error("Error: Database repository or PDF generator is not loaded.")
        return

    # Ensure reports directory exists
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # ── Load history data ──────────────────────────────────────────────────────
    try:
        df = repo.get_all(limit=500)
    except Exception as exc:
        st.error(f"Failed to load detection records: {exc}")
        return

    if df.empty:
        st.info("📭 No detection history found. Run some detections first to generate reports!")
        return

    # ── Calculate report statistics ────────────────────────────────────────────
    total_records = len(df)
    
    # Check physical existence of report files referenced in DB
    valid_pdf_count = 0
    for idx, row in df.iterrows():
        path_str = row.get("report_path")
        if path_str and Path(path_str).is_file():
            valid_pdf_count += 1

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🕒 Total Detections", total_records)
    with col2:
        st.metric("📄 Generated Reports", valid_pdf_count)
    with col3:
        st.metric("📭 Pending Reports", total_records - valid_pdf_count)

    # ── Bulk Data Export ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📥 Bulk Metadata Export")
    st.markdown(
        "<p style='color:#95a5a6;'>Export the complete database log of all detections.</p>",
        unsafe_allow_html=True,
    )
    col_csv, col_json = st.columns(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with col_csv:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Metadata (CSV)",
            data=csv_data,
            file_name=f"deepfake_detection_log_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_json:
        json_data = df.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Export Metadata (JSON)",
            data=json_data,
            file_name=f"deepfake_detection_log_{ts}.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── Report Management Table ──────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 PDF Report Management")
    st.markdown(
        "<p style='color:#95a5a6;'>Generate and download PDF reports for individual detection files below.</p>",
        unsafe_allow_html=True,
    )

    # Search & filters
    c_f1, c_f2 = st.columns([2, 1])
    with c_f1:
        search_query = st.text_input("🔍 Search by File Name", "", key="report_search")
    with c_f2:
        filter_type = st.selectbox("Filter File Type", ["All", "image", "video"], key="report_filter_type")

    # Filter dataframe
    filtered_df = df.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df["file_name"].str.contains(search_query, case=False, na=False)
        ]
    if filter_type != "All":
        filtered_df = filtered_df[filtered_df["file_type"] == filter_type]

    st.markdown(f"Found **{len(filtered_df)}** records matching criteria.")

    # Render interactive list
    for idx, row in filtered_df.iterrows():
        record_id = int(row["id"])
        file_name = row["file_name"]
        file_type = row["file_type"]
        prediction = row["prediction"]
        confidence = float(row["confidence"])
        timestamp_str = row["timestamp"]
        saved_path = row.get("report_path")

        # Parse timestamp for display
        try:
            dt = datetime.fromisoformat(timestamp_str)
            time_display = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            time_display = timestamp_str

        # Frame stats for display
        frame_info = ""
        if file_type == "video" and row.get("num_frames"):
            frame_info = f" ({row['num_frames']} frames)"

        # Style prediction label
        pred_color = "#e74c3c" if prediction == "FAKE" else "#2ecc71"
        badge = f"<span style='color:{pred_color}; font-weight:bold;'>{prediction}</span>"

        # Check if PDF physically exists
        pdf_exists = False
        if saved_path:
            pdf_path_obj = Path(saved_path)
            if pdf_path_obj.is_file():
                pdf_exists = True

        # Render row container
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background: rgba(30,39,56,0.4);
                    border-left: 4px solid {pred_color};
                    border-radius: 8px;
                    padding: 0.8rem 1.2rem;
                    margin-bottom: 0.8rem;
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-weight:600; font-size:1.05rem;">{file_name}</span>
                            <span style="color:#7f8c8d; font-size:0.85rem; margin-left:8px;">{file_type}{frame_info}</span>
                        </div>
                        <div style="font-size:0.9rem; color:#95a5a6;">{time_display}</div>
                    </div>
                    <div style="margin-top:0.4rem; font-size:0.9rem; color:#ecf0f1;">
                        Verdict: {badge} &nbsp;|&nbsp; Confidence: <b>{confidence:.1%}</b> &nbsp;|&nbsp; 
                        Status: <b>{"Report Available" if pdf_exists else "No Report Generated"}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Row Action Buttons
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
            
            if pdf_exists and saved_path:
                with btn_col1:
                    with open(saved_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF",
                            data=f.read(),
                            file_name=Path(saved_path).name,
                            mime="application/pdf",
                            key=f"dl_rep_{record_id}",
                            use_container_width=True,
                        )
                with btn_col2:
                    if st.button("🔄 Regenerate", key=f"regen_{record_id}", use_container_width=True):
                        _generate_pdf_for_record(repo, pdf_gen, record_id)
                        st.rerun()
            else:
                with btn_col1:
                    if st.button("📄 Generate PDF", key=f"gen_{record_id}", use_container_width=True, type="primary"):
                        _generate_pdf_for_record(repo, pdf_gen, record_id)
                        st.rerun()


def _generate_pdf_for_record(
    repo: DetectionRepository,
    pdf_gen: PDFReportGenerator,
    record_id: int,
) -> None:
    """Helper to query the db record and generate its PDF report."""
    record = repo.get_for_report(record_id)
    if not record:
        st.error("Error: Record not found in database.")
        return

    try:
        pdf_path = pdf_gen.generate(
            file_name=record["file_name"],
            prediction=record["prediction"],
            confidence=record["confidence"],
            processing_time_ms=record["processing_time"],
            timestamp=record["timestamp"],
            num_frames=record.get("num_frames") or 0,
            fake_frames=record.get("fake_frames") or 0,
            real_frames=record.get("real_frames") or 0,
            model_name=record.get("model_name") or "XceptionNet",
            notes=record.get("notes") or "",
        )
        # Update path in database
        repo.update_report_path(record_id, str(pdf_path))
        st.success(f"Report generated successfully: {pdf_path.name}")
    except Exception as exc:
        st.error(f"Failed to generate report for record #{record_id}: {exc}")
