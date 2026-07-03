"""
dashboard/pages/reports.py
============================
Reports page — list all generated PDF reports and allow download.
"""

from pathlib import Path

import streamlit as st

# About page has moved to dashboard/pages/about.py


def render(reports_dir: str = "outputs/reports") -> None:
    """
    Render the Reports page.

    Args:
        reports_dir: Directory where PDF reports are saved.
    """
    st.markdown("## 📄 Detection Reports")
    st.markdown(
        "<p style='color:#95a5a6;'>All PDF reports generated from past detections.</p>",
        unsafe_allow_html=True,
    )

    reports_path = Path(reports_dir)
    if not reports_path.is_dir():
        st.info("No reports directory found. Run a detection to generate your first report.")
        return

    pdf_files = sorted(
        reports_path.glob("*.pdf"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not pdf_files:
        st.info(
            "📭 No reports yet. Generate a report from the Image or Video Detection pages."
        )
        return

    st.markdown(f"**{len(pdf_files)} report(s) found.**")

    for pdf in pdf_files:
        col_name, col_size, col_btn = st.columns([4, 1, 1])
        size_kb = pdf.stat().st_size / 1024
        with col_name:
            st.markdown(f"📄 `{pdf.name}`")
        with col_size:
            st.markdown(
                f"<span style='color:#95a5a6;'>{size_kb:.1f} KB</span>",
                unsafe_allow_html=True,
            )
        with col_btn:
            with open(str(pdf), "rb") as f:
                st.download_button(
                    "⬇️",
                    data=f.read(),
                    file_name=pdf.name,
                    mime="application/pdf",
                    key=f"dl_{pdf.name}",
                    use_container_width=True,
                )
