"""
dashboard/pages/history.py
============================
Detection history page — displays, filters, and deletes past records.
Redesigned with premium filters, date range pickers, and record management.
"""

import pandas as pd
import streamlit as st
from datetime import datetime


def render(repo: "DetectionRepository") -> None:  # type: ignore[name-defined]
    """
    Render the Detection History page.

    Args:
        repo: :class:`DetectionRepository` instance.
    """
    st.markdown("## 🕒 Detection History Log")
    st.markdown(
        "<p style='color: var(--text-secondary);'>Search, filter, and manage past detection runs "
        "stored in the local SQLite database.</p>",
        unsafe_allow_html=True,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    try:
        df = repo.get_all(limit=500)
    except Exception as exc:
        st.error(f"Failed to load history: {exc}")
        return

    if df.empty:
        st.info("📭 No detection history found yet. Run some detections first!")
        return

    # ── Modern Filters Section ──────────────────────────────────────────────────
    st.markdown("### 🔍 Search & Filters")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_pred = st.selectbox("Filter Prediction", ["All", "FAKE", "REAL"], key="hist_filter_pred")
    with col_f2:
        filter_type = st.selectbox("Filter File Type", ["All", "image", "video"], key="hist_filter_type")
    with col_f3:
        search_name = st.text_input("Search File Name", "", key="hist_search_name")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", value=None, key="hist_start_date")
    with col_d2:
        end_date = st.date_input("End Date", value=None, key="hist_end_date")

    # Apply filters
    filtered = df.copy()
    if filter_pred != "All":
        filtered = filtered[filtered["prediction"] == filter_pred]
    if filter_type != "All":
        filtered = filtered[filtered["file_type"] == filter_type]
    if search_name:
        filtered = filtered[
            filtered["file_name"].str.contains(search_name, case=False, na=False)
        ]
        
    # Date range filters
    if start_date is not None:
        filtered = filtered[pd.to_datetime(filtered["timestamp"]).dt.date >= start_date]
    if end_date is not None:
        filtered = filtered[pd.to_datetime(filtered["timestamp"]).dt.date <= end_date]

    st.markdown(
        f"Showing **{len(filtered)}** of **{len(df)}** records",
    )

    # ── Results Grid / Table ──────────────────────────────────────────────────
    display_cols = [
        "id", "file_name", "file_type", "prediction",
        "confidence", "model_name", "processing_time", "timestamp",
    ]
    
    # Rename columns for prettier headers
    display_df = filtered[[c for c in display_cols if c in filtered.columns]].copy()
    display_df = display_df.rename(columns={
        "id": "ID",
        "file_name": "File Name",
        "file_type": "Medium",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "model_name": "Model",
        "processing_time": "Speed (ms)",
        "timestamp": "Timestamp"
    })

    def highlight_prediction(val: str) -> str:
        if val == "FAKE":
            return "color: var(--danger); font-weight: bold"
        elif val == "REAL":
            return "color: var(--success); font-weight: bold"
        return ""

    st.dataframe(
        display_df.style.applymap(highlight_prediction, subset=["Prediction"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Record Management Actions ──────────────────────────────────────────────
    st.divider()
    st.markdown("### 🛠️ Record Management")
    
    col_del_single, col_del_bulk = st.columns(2)
    
    with col_del_single:
        # Delete specific record ID
        st.markdown("<h5 style='font-size:0.95rem;'>🗑️ Delete Single Record</h5>", unsafe_allow_html=True)
        delete_id = st.number_input("Record ID to Delete", min_value=1, step=1, key="hist_delete_id")
        if st.button("Delete Record", type="secondary", key="hist_delete_btn"):
            if repo.delete_by_id(delete_id):
                st.success(f"Record #{delete_id} deleted successfully.")
                st.rerun()
            else:
                st.error(f"Record #{delete_id} not found in database.")

    with col_del_bulk:
        st.markdown("<h5 style='font-size:0.95rem;'>⚠️ Bulk Actions</h5>", unsafe_allow_html=True)
        # Clear all
        if st.button("🗑️ Clear ALL History Logs", use_container_width=True, type="secondary", key="hist_clear_all"):
            if st.session_state.get("confirm_delete_all"):
                deleted = repo.delete_all()
                st.success(f"Deleted {deleted} records.")
                st.session_state["confirm_delete_all"] = False
                st.rerun()
            else:
                st.session_state["confirm_delete_all"] = True
                st.warning("⚠️ Click again to confirm deletion of ALL records from SQLite database.")

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⬇️ Export Log Metadata")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Filtered CSV Log",
        data=csv,
        file_name="detection_history_log.csv",
        mime="text/csv",
        use_container_width=True,
        key="hist_csv_export",
    )
