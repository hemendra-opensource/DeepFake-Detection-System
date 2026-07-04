"""
dashboard/pages/about.py
==========================
About page — tech stack, datasets, model performance table.
Redesigned with glassmorphic cards and typography.
"""

from pathlib import Path
import streamlit as st


def render() -> None:
    """Render the About page."""
    st.markdown("## ℹ️ About Explainable DeepFake Detection AI")

    st.markdown(
        """
        <div class="glass-card" style="border-left: 5px solid var(--primary); padding: 2rem; margin-bottom: 2rem; animation: fadeIn 0.4s ease-out;">
            <h2 style="font-family:'Outfit',sans-serif; color: var(--primary-light); margin:0 0 0.5rem; font-size: 1.6rem; font-weight: 800;">
                🛡️ AI Integrity Verification Platform
            </h2>
            <p style="color: var(--text-secondary); margin:0; font-size: 1rem; line-height: 1.6;">
                A production-grade AI application for detecting face manipulations in images and videos. 
                Utilizing XceptionNet with Grad-CAM visual explainability, face tracking, Platt scaling probability calibration,
                and A4 PDF report compilation. Built as a Final Year Internship Project.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🔧 Platform Tech Stack")
        tech = {
            "Programming Language": "Python 3.11.x",
            "Deep Learning Framework": "TensorFlow 2.15 + Keras",
            "Backbone Neural Network": "XceptionNet (ImageNet pre-trained)",
            "Alternative Topologies": "EfficientNet-B0 · ResNet50",
            "Face Segmentation API": "MediaPipe Face Landmarker + Haar Cascades",
            "Gradient Explanations": "Grad-CAM (activation maps)",
            "Web App Dashboard": "Streamlit 1.35+",
            "Data & Logs Store": "SQLite DB",
            "Report Generation": "ReportLab 4.2",
            "Data Augmentations": "Albumentations 1.4",
            "Pipeline Testing": "pytest + unittest",
        }
        
        for key, val in tech.items():
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding: 0.5rem 0; font-size: 0.9rem;">
                    <span style="color: var(--text-secondary); font-weight: 500;">{key}</span>
                    <code style="color: var(--primary-light); font-weight: 700;">{val}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### 📊 Benchmark Datasets")
        datasets = [
            ("FaceForensics++",
             "A forensic dataset with 1,000 video sequences manipulated using DeepFakes, Face2Face, FaceSwap, and NeuralTextures.",
             "https://github.com/ondyari/FaceForensics"),
            ("Celeb-DF v2",
             "High-quality celebrity deepfake dataset with 5,639 videos, addressing visible stitching artifacts of earlier generations.",
             "https://github.com/yuezunli/celeb-deepfakeforensics"),
            ("DFDC (Challenge)",
             "Large-scale public challenge dataset released by Meta AI containing over 100,000 videos from paid actors.",
             "https://ai.meta.com/datasets/dfdc/"),
        ]
        for name, desc, url in datasets:
            st.markdown(
                f"""
                <div class="glass-card" style="padding: 1rem; border-left: 3px solid var(--success); margin-bottom: 0.8rem;">
                    <strong style="color: var(--text-primary); font-size: 1rem;">{name}</strong><br>
                    <span style="color: var(--text-secondary); font-size: 0.82rem; line-height: 1.4; display: block; margin: 0.3rem 0;">{desc}</span>
                    <a href="{url}" target="_blank" style="color: var(--success); font-size: 0.8rem; font-weight: 600; text-decoration: none;">🔗 Source Dataset Link</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Training Strategy ─────────────────────────────────────────────────────
    st.markdown("### 🏋️ Progressive Model Fine-Tuning Strategy")
    col_p1, col_p2, col_p3 = st.columns(3)
    phases = [
        ("Phase 1: Warmup", "FaceForensics++", "Freeze convolutional base, train dense classification heads only.", "var(--primary-light)"),
        ("Phase 2: Fine-tune", "FF++ + Celeb-DF", "Unfreeze the top 20 blocks of XceptionNet to learn domain features.", "var(--success)"),
        ("Phase 3: Generalization", "FF++ + Celeb-DF + DFDC", "Full end-to-end training with low learning rate & scheduling.", "var(--accent)"),
    ]
    for col, (phase, data, strategy, colour) in zip([col_p1, col_p2, col_p3], phases):
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center; padding: 1.2rem; min-height: 150px; border-top: 4px solid {colour};">
                    <div style="font-weight:800; color:{colour}; font-size: 1.05rem;">{phase}</div>
                    <div style="font-size:0.85rem; color: var(--text-primary); font-weight: 700; margin: 0.4rem 0;">
                        {data}
                    </div>
                    <div style="font-size:0.78rem; color: var(--text-secondary); line-height: 1.4;">{strategy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Model Comparison ──────────────────────────────────────────────────────
    st.markdown("### 📈 Evaluated Architecture Performance")

    comp_path = Path("outputs/model_comparison.csv")
    if comp_path.is_file():
        import pandas as pd
        from dashboard.components.charts import model_comparison_bar
        df = pd.read_csv(comp_path)
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = model_comparison_bar(df)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=st.session_state.get("text_color", "#F8FAFC"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "📊 Model comparison results will appear here after training completes.\n\n"
            "Run: `py -3.11 train.py --evaluate` to build comparisons."
        )

    st.divider()

    # ── References ────────────────────────────────────────────────────────────
    st.markdown("### 📚 Academic & Technical References")
    refs = [
        ("Xception: Deep Learning with Depthwise Separable Convolutions",
         "Francois Chollet, CVPR 2017", "https://arxiv.org/abs/1610.02357"),
        ("Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
         "Ramprasaath R. Selvaraju et al., ICCV 2017", "https://arxiv.org/abs/1610.02391"),
        ("FaceForensics++: Learning to Detect Manipulated Facial Images",
         "Andreas Rössler et al., ICCV 2019", "https://arxiv.org/abs/1901.08971"),
        ("Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics",
         "Yuezun Li et al., CVPR 2020", "https://arxiv.org/abs/1909.12962"),
        ("The DeepFake Detection Challenge (DFDC) Dataset",
         "Brian Dolhansky et al., arXiv 2020", "https://arxiv.org/abs/2006.07397"),
    ]
    for title, authors, url in refs:
        st.markdown(
            f"""
            <div style="font-size:0.9rem; padding:0.3rem 0;">
                🛡️ <a href="{url}" target="_blank" style="color: var(--primary-light); font-weight:600; text-decoration:none;">{title}</a> 
                — <span style="color: var(--text-secondary);">({authors})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
