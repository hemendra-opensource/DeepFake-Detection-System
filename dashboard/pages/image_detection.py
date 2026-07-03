"""
dashboard/pages/image_detection.py
====================================
Image DeepFake detection page.

User flow:
1. Upload image
2. Run detection (face crop → model → Grad-CAM)
3. View prediction badge, confidence bar, Grad-CAM viewer
4. Download PDF report
5. Record saved to history
"""

from datetime import datetime
from typing import Optional

import numpy as np
import streamlit as st


def render(
    image_detector: "ImageDetector",   # type: ignore[name-defined]
    gradcam: Optional["GradCAM"],       # type: ignore[name-defined]
    pdf_gen: "PDFReportGenerator",      # type: ignore[name-defined]
    repo: "DetectionRepository",        # type: ignore[name-defined]
    model_name: str = "XceptionNet",
) -> None:
    """
    Render the Image Detection page.

    Args:
        image_detector: Loaded image detector.
        gradcam:        Grad-CAM instance (may be None).
        pdf_gen:        PDF report generator.
        repo:           Detection history repository.
        model_name:     Model display name.
    """
    from dashboard.components.kpi_cards import render_prediction_badge, render_confidence_bar
    from dashboard.components.gradcam_viewer import render_gradcam_viewer, render_gradcam_unavailable
    from dashboard.components.charts import confidence_histogram

    st.markdown("## 🖼️ Image DeepFake Detection")
    st.markdown(
        "<p style='color:#95a5a6;'>Upload an image to analyse. "
        "Supported formats: JPG, JPEG, PNG</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
        label_visibility="collapsed",
    )

    if uploaded is None:
        _render_upload_placeholder()
        return

    # Show preview + run detection
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**📷 Uploaded Image**")
        st.image(uploaded, width=500)

    with col_result:
        st.markdown("**⚙️ Processing…**")
        progress = st.progress(0, text="Loading image…")

        try:
            image_bytes = uploaded.read()
            progress.progress(25, text="Detecting face…")

            prediction = image_detector.predict_from_bytes(image_bytes)
            progress.progress(70, text="Running model…")

            # Grad-CAM
            gradcam_result = None
            if gradcam is not None and prediction.preprocessed_face is not None:
                try:
                    gradcam_result = gradcam.explain_prediction(
                        prediction.preprocessed_face,
                        fake_probability=prediction.fake_probability,
                        threshold=prediction.threshold_used,
                    )
                except Exception as exc:
                    st.warning(f"Grad-CAM failed: {exc}")
            progress.progress(100, text="Done!")

            # ── Results ───────────────────────────────────────────────────────
            render_prediction_badge(prediction.label, prediction.confidence)
            render_confidence_bar(prediction.fake_probability, prediction.label)

            # Probability breakdown — mathematically guaranteed to sum to 100%
            real_pct = round(prediction.real_probability * 100, 1)
            fake_pct = round(prediction.fake_probability * 100, 1)

            st.markdown(
                f"""
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr;
                            gap:0.5rem; margin-top:1rem; font-size:0.85rem;">
                    <div style="background:rgba(30,39,56,0.7); border-radius:8px;
                                padding:0.6rem; text-align:center;">
                        <div style="color:#95a5a6;">✅ Real Prob</div>
                        <div style="font-weight:700; color:#2ecc71;">{real_pct}%</div>
                    </div>
                    <div style="background:rgba(30,39,56,0.7); border-radius:8px;
                                padding:0.6rem; text-align:center;">
                        <div style="color:#95a5a6;">⚠️ Fake Prob</div>
                        <div style="font-weight:700; color:#e74c3c;">{fake_pct}%</div>
                    </div>
                    <div style="background:rgba(30,39,56,0.7); border-radius:8px;
                                padding:0.6rem; text-align:center;">
                        <div style="color:#95a5a6;">⚡ Speed</div>
                        <div style="font-weight:700; color:#5dade2;">
                            {prediction.processing_time_ms:.0f} ms
                        </div>
                    </div>
                    <div style="background:rgba(30,39,56,0.7); border-radius:8px;
                                padding:0.6rem; text-align:center;">
                        <div style="color:#95a5a6;">👤 Face</div>
                        <div style="font-weight:700;
                                    color:{'#2ecc71' if prediction.face_detected else '#e74c3c'}">
                            {"Detected" if prediction.face_detected else "Not Found"}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        except Exception as exc:
            st.error(f"❌ Detection failed: {exc}")
            progress.empty()
            return

    # ── Grad-CAM section ──────────────────────────────────────────────────────
    st.divider()
    if gradcam_result is not None:
        render_gradcam_viewer(
            original=gradcam_result.original_image,
            heatmap_colored=gradcam_result.heatmap_colored,
            overlay=gradcam_result.overlay,
            gradient_strength=gradcam_result.gradient_strength,
            model_name=model_name,
            prediction=prediction.label,
        )
    else:
        render_gradcam_unavailable("Model not loaded or Grad-CAM computation failed.")

    # ── Save to history + PDF ─────────────────────────────────────────────────
    st.divider()
    col_save, col_pdf = st.columns(2)

    with col_save:
        if st.button("💾 Save to History", use_container_width=True):
            try:
                repo.insert_detection(
                    file_name=uploaded.name,
                    prediction=prediction.label,
                    confidence=prediction.confidence,
                    fake_probability=prediction.fake_probability,
                    file_type="image",
                    model_name=model_name,
                    processing_time=prediction.processing_time_ms,
                )
                st.success("✅ Saved to detection history!")
            except Exception as exc:
                st.error(f"Failed to save: {exc}")

    with col_pdf:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            try:
                overlay_arr = gradcam_result.overlay if gradcam_result else None
                pdf_path = pdf_gen.generate(
                    file_name=uploaded.name,
                    prediction=prediction.label,
                    confidence=prediction.confidence,
                    processing_time_ms=prediction.processing_time_ms,
                    timestamp=datetime.now(),
                    gradcam_image=overlay_arr,
                    model_name=model_name,
                )
                with open(str(pdf_path), "rb") as f:
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=f.read(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")


def _render_upload_placeholder() -> None:
    """Show a placeholder UI before any file is uploaded."""
    st.markdown(
        """
        <div style="text-align:center; padding:3rem; border:2px dashed rgba(93,173,226,0.3);
             border-radius:16px; color:#95a5a6; margin-top:1rem;">
            <div style="font-size:3rem;">🖼️</div>
            <div style="font-size:1.1rem; font-weight:600; margin:0.5rem 0;">
                Drop an image here or click to upload
            </div>
            <div style="font-size:0.85rem;">Supported: JPG, JPEG, PNG, BMP, WEBP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
