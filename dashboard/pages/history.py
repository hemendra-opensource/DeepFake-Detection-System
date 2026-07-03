"""
dashboard/pages/history.py
============================
Detection history page — displays, filters, and deletes past records.
"""

import pandas as pd
import streamlit as st


def render(repo: "DetectionRepository") -> None:  # type: ignore[name-defined]
    """
    Render the Detection History page.

    Args:
        repo: :class:`DetectionRepository` instance.
    """
    st.markdown("## 🕒 Detection History")
    st.markdown(
        "<p style='color:#95a5a6;'>All past detection records stored in the local SQLite database.</p>",
        unsafe_allow_html=True,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    try:
        df = repo.get_all(limit=500)
    except Exception as exc:
        st.error(f"Failed to load history: {exc}")
        return

    if df.empty:
        st.info("📭 No detection history yet. Run some detections first!")
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_pred = st.selectbox("Filter by Prediction", ["All", "FAKE", "REAL"])
    with col_f2:
        filter_type = st.selectbox("Filter by File Type", ["All", "image", "video"])
    with col_f3:
        search_name = st.text_input("Search by File Name", "")

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

    st.markdown(
        f"Showing **{len(filtered)}** of **{len(df)}** records",
    )

    # ── Table ─────────────────────────────────────────────────────────────────
    display_cols = [
        "id", "file_name", "file_type", "prediction",
        "confidence", "model_name", "processing_time", "timestamp",
    ]
    display_df = filtered[[c for c in display_cols if c in filtered.columns]]

    def highlight_prediction(val: str) -> str:
        if val == "FAKE":
            return "color: #e74c3c; font-weight: bold"
        elif val == "REAL":
            return "color: #2ecc71; font-weight: bold"
        return ""

    st.dataframe(
        display_df.style.applymap(highlight_prediction, subset=["prediction"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    col_csv, col_del = st.columns(2)
    with col_csv:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Filtered CSV",
            data=csv,
            file_name="detection_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_del:
        if st.button("🗑️ Clear ALL History", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_delete"):
                deleted = repo.delete_all()
                st.success(f"Deleted {deleted} records.")
                st.session_state["confirm_delete"] = False
                st.rerun()
            else:
                st.session_state["confirm_delete"] = True
                st.warning("⚠️ Click again to confirm deletion of ALL records.")

    # ── Charts ────────────────────────────────────────────────────────────────
    if len(filtered) > 0:
        st.divider()
        st.markdown("### 📊 History Analytics")
        import plotly.express as px

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            pie = px.pie(
                values=filtered["prediction"].value_counts().values,
                names=filtered["prediction"].value_counts().index,
                color=filtered["prediction"].value_counts().index,
                color_discrete_map={"FAKE": "#e74c3c", "REAL": "#2ecc71"},
                hole=0.5,
                title="Prediction Distribution",
            )
            pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ecf0f1",
            )
            st.plotly_chart(pie, use_container_width=True)

        with col_c2:
            hist = px.histogram(
                filtered,
                x="confidence",
                nbins=20,
                title="Confidence Distribution",
                color_discrete_sequence=["#5dade2"],
            )
            hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ecf0f1",
                xaxis_gridcolor="rgba(255,255,255,0.05)",
                yaxis_gridcolor="rgba(255,255,255,0.05)",
            )
            st.plotly_chart(hist, use_container_width=True)
