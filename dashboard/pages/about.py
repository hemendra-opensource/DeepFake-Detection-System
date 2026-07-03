"""
dashboard/pages/about.py
==========================
About page — tech stack, datasets, model performance table.
"""

from pathlib import Path
import streamlit as st


def render() -> None:
    """Render the About page."""
    st.markdown("## ℹ️ About This Project")

    st.markdown(
        """
        <div style="background:rgba(30,39,56,0.7); border:1px solid rgba(93,173,226,0.2);
             border-radius:16px; padding:2rem; margin-bottom:1.5rem;">
            <h2 style="font-family:'Outfit',sans-serif; color:#5dade2; margin:0 0 0.5rem;">
                🛡️ Explainable DeepFake Detection
            </h2>
            <p style="color:#95a5a6; margin:0;">
                A production-grade AI application for detecting DeepFake images and videos
                using XceptionNet with Grad-CAM explainability, frame-wise video analysis,
                and a modern Streamlit dashboard. Built as a Final Year Internship Project.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔧 Technology Stack")
        tech = {
            "Language":      "Python 3.11.x",
            "Deep Learning": "TensorFlow 2.15 + Keras",
            "Primary Model": "XceptionNet (ImageNet pre-trained)",
            "Alt Models":    "EfficientNet-B0 · ResNet50",
            "Face Detection":"MediaPipe + OpenCV Haar",
            "Explainability":"Grad-CAM (Gradient-weighted CAM)",
            "Dashboard":     "Streamlit 1.35",
            "Visualization": "Plotly · Matplotlib · Seaborn",
            "Database":      "SQLite (via stdlib sqlite3)",
            "PDF Reports":   "ReportLab 4.2",
            "Augmentation":  "Albumentations 1.4",
            "Testing":       "pytest + unittest",
        }
        for key, val in tech.items():
            st.markdown(f"- **{key}**: `{val}`")

    with col2:
        st.markdown("### 📊 Datasets Used")
        datasets = [
            ("FaceForensics++",
             "Face manipulation dataset with 4 manipulation types.",
             "https://github.com/ondyari/FaceForensics"),
            ("Celeb-DF v2",
             "High-quality celebrity deepfake video dataset.",
             "https://github.com/yuezunli/celeb-deepfakeforensics"),
            ("DFDC",
             "DeepFake Detection Challenge by Meta AI (470 GB).",
             "https://ai.meta.com/datasets/dfdc/"),
        ]
        for name, desc, url in datasets:
            st.markdown(
                f"""
                <div style="background:rgba(30,39,56,0.5); border-radius:10px;
                            padding:0.8rem; margin-bottom:0.6rem;
                            border-left:3px solid #5dade2;">
                    <strong style="color:#5dade2;">{name}</strong><br>
                    <span style="color:#95a5a6; font-size:0.85rem;">{desc}</span><br>
                    <a href="{url}" target="_blank"
                       style="color:#2ecc71; font-size:0.8rem;">🔗 Dataset Link</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Training Strategy ─────────────────────────────────────────────────────
    st.markdown("### 🏋️ Progressive Training Strategy")
    col_p1, col_p2, col_p3 = st.columns(3)
    phases = [
        ("Phase 1", "FaceForensics++", "Freeze base, train head only", "#5dade2"),
        ("Phase 2", "FF++ + Celeb-DF", "Unfreeze last 20 layers", "#2ecc71"),
        ("Phase 3", "FF++ + Celeb-DF + DFDC", "Full fine-tune", "#9b59b6"),
    ]
    for col, (phase, data, strategy, colour) in zip([col_p1, col_p2, col_p3], phases):
        with col:
            st.markdown(
                f"""
                <div style="background:rgba(30,39,56,0.7); border-radius:10px;
                            padding:1rem; text-align:center;
                            border-top:3px solid {colour};">
                    <div style="font-weight:700; color:{colour};">{phase}</div>
                    <div style="font-size:0.8rem; color:#ecf0f1; margin:0.3rem 0;">
                        {data}
                    </div>
                    <div style="font-size:0.75rem; color:#95a5a6;">{strategy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Model Comparison ──────────────────────────────────────────────────────
    st.markdown("### 📈 Model Performance")

    comp_path = Path("outputs/model_comparison.csv")
    if comp_path.is_file():
        import pandas as pd
        from dashboard.components.charts import model_comparison_bar
        df = pd.read_csv(comp_path)
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = model_comparison_bar(df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "📊 Model comparison results will appear here after training completes.\n\n"
            "Run: `py -3.11 train.py --evaluate`"
        )

    st.divider()

    # ── References ────────────────────────────────────────────────────────────
    st.markdown("### 📚 Key References")
    refs = [
        ("Xception: Deep Learning with Depthwise Separable Convolutions",
         "Chollet, 2017", "https://arxiv.org/abs/1610.02357"),
        ("Grad-CAM: Visual Explanations from Deep Networks",
         "Selvaraju et al., 2020", "https://arxiv.org/abs/1610.02391"),
        ("FaceForensics++: Learning to Detect Manipulated Facial Images",
         "Rössler et al., 2019", "https://arxiv.org/abs/1901.08971"),
        ("Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics",
         "Li et al., 2020", "https://arxiv.org/abs/1909.12962"),
        ("The DeepFake Detection Challenge (DFDC)",
         "Dolhansky et al., 2020", "https://arxiv.org/abs/2006.07397"),
    ]
    for title, authors, url in refs:
        st.markdown(f"- [{title}]({url}) — *{authors}*")

    st.divider()
    st.markdown(
        """
        <div style="text-align:center; color:#7f8c8d; font-size:0.85rem; padding:1rem;">
            Built with ❤️ as a Final Year Internship Project · Python 3.11 · TensorFlow 2.15
        </div>
        """,
        unsafe_allow_html=True,
    )
