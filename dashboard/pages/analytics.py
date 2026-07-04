"""
dashboard/pages/analytics.py
==============================
Premium Analytics Page for the DeepFake Detection System.
Renders advanced charts from SQLite history.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime


def render(repo: "DetectionRepository") -> None:  # type: ignore[name-defined]
    """
    Render the Analytics page.

    Args:
        repo: Detection history repository.
    """
    st.markdown("## 📊 System Analytics")
    st.markdown(
        "<p style='color: var(--text-secondary);'>Comprehensive insights, metrics, and patterns "
        "mined from past DeepFake detection records.</p>",
        unsafe_allow_html=True,
    )

    try:
        df = repo.get_all(limit=500)
    except Exception as exc:
        st.error(f"Failed to load analytics: {exc}")
        return

    if df.empty:
        st.info("📭 No data available. Analyze some images or videos first to populate analytics!")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    total_records = len(df)
    fake_df = df[df["prediction"] == "FAKE"]
    real_df = df[df["prediction"] == "REAL"]
    fake_count = len(fake_df)
    real_count = len(real_df)
    
    fake_ratio = (fake_count / total_records * 100) if total_records else 0.0
    avg_conf = df["confidence"].mean() if total_records else 0.0
    avg_speed = df["processing_time"].mean() if total_records else 0.0

    st.markdown("### 📈 Performance Overview")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📂 Total Analyzed", f"{total_records:,}")
    with c2:
        st.metric("🚨 Fake Detected", f"{fake_count:,}", f"{fake_ratio:.1f}% ratio")
    with c3:
        st.metric("🎯 Avg Confidence", f"{avg_conf:.1%}")
    with c4:
        st.metric("⚡ Avg Speed", f"{avg_speed:.0f} ms")

    st.divider()

    # ── Advanced Charts Grid ──────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        # 1. Donut Chart - Real vs Fake
        st.markdown("#### 🚨 Prediction Distribution")
        fig_donut = px.pie(
            values=[fake_count, real_count],
            names=["FAKE", "REAL"],
            color=["FAKE", "REAL"],
            color_discrete_map={"FAKE": "#EF4444", "REAL": "#22C55E"},
            hole=0.55,
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=st.session_state.get("text_color", "#F8FAFC"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_r:
        # 2. Histogram - Confidence Distribution
        st.markdown("#### 🎯 Confidence Spread")
        fig_hist = px.histogram(
            df,
            x="confidence",
            nbins=20,
            color="prediction",
            color_discrete_map={"FAKE": "#EF4444", "REAL": "#22C55E"},
            labels={"confidence": "Confidence Score", "count": "Frequency"},
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=st.session_state.get("text_color", "#F8FAFC"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            xaxis_gridcolor="rgba(255,255,255,0.05)",
            yaxis_gridcolor="rgba(255,255,255,0.05)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        # 3. Bar Chart - Avg Processing Time by File Type
        st.markdown("#### ⚡ Speed by Medium")
        speed_df = df.groupby("file_type")["processing_time"].mean().reset_index()
        fig_bar = px.bar(
            speed_df,
            x="file_type",
            y="processing_time",
            color="file_type",
            color_discrete_map={"image": "#3B82F6", "video": "#60A5FA"},
            labels={"processing_time": "Time (ms)", "file_type": "Medium"},
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=st.session_state.get("text_color", "#F8FAFC"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            xaxis_gridcolor="rgba(255,255,255,0.05)",
            yaxis_gridcolor="rgba(255,255,255,0.05)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b2:
        # 4. Area Chart - Timeline of detections
        st.markdown("#### 🕒 Analysis History Timeline")
        
        # Sort by timestamp
        timeline_df = df.copy()
        timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"])
        timeline_df = timeline_df.sort_values("timestamp")
        
        # Resample or group by day to show count
        timeline_df["date"] = timeline_df["timestamp"].dt.date
        date_df = timeline_df.groupby(["date", "prediction"]).size().unstack(fill_value=0).reset_index()
        
        if not date_df.empty:
            melted_date_df = date_df.melt(id_vars=["date"], value_vars=["FAKE", "REAL"], var_name="Prediction", value_name="Count")
            fig_area = px.area(
                melted_date_df,
                x="date",
                y="Count",
                color="Prediction",
                color_discrete_map={"FAKE": "rgba(239, 68, 68, 0.4)", "REAL": "rgba(34, 197, 94, 0.4)"},
            )
            fig_area.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=st.session_state.get("text_color", "#F8FAFC"),
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
                xaxis_gridcolor="rgba(255,255,255,0.05)",
                yaxis_gridcolor="rgba(255,255,255,0.05)",
            )
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("Insufficient timeline records to show history trend.")
