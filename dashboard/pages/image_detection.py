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

from inference.image_detector import ImageDetector, ImagePrediction


def render(
    image_detector: ImageDetector,
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
    from dashboard.components.loader import render_ai_loader

    st.markdown("## 🖼️ Image DeepFake Detection")
    st.markdown(
        "<p style='color: var(--text-secondary);'>Upload an image to analyze for face manipulation. "
        "Supported formats: JPG, JPEG, PNG, BMP, WEBP</p>",
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
        st.image(uploaded, use_container_width=True)

    with col_result:
        st.markdown("**⚙️ SaaS Inference Analysis**")
        
        status_box = st.empty()
        
        with status_box.container():
            render_ai_loader("Initializing face detection cascades...")

        try:
            image_bytes = uploaded.read()
            
            with status_box.container():
                render_ai_loader("Analyzing facial crops with XceptionNet...")
            prediction = image_detector.predict_from_bytes(image_bytes)

            # Grad-CAM
            gradcam_result = None
            if gradcam is not None and prediction.preprocessed_face is not None:
                with status_box.container():
                    render_ai_loader("Extracting Grad-CAM gradient heatmaps...")
                try:
                    gradcam_result = gradcam.explain_prediction(
                        prediction.preprocessed_face,
                        fake_probability=prediction.fake_probability,
                        threshold=prediction.threshold_used,
                    )
                except Exception as exc:
                    st.warning(f"Grad-CAM failed: {exc}")
            
            with status_box.container():
                render_ai_loader("Packaging final verification metrics...")
                
            status_box.empty()  # Clear loader

            # ── Results ───────────────────────────────────────────────────────
            render_prediction_badge(prediction.label, prediction.confidence)
            render_confidence_bar(prediction.fake_probability, prediction.label)

            # Probability breakdown — mathematically guaranteed to sum to 100%
            real_pct = round(prediction.real_probability * 100, 1)
            fake_pct = round(prediction.fake_probability * 100, 1)

            st.markdown(
                f"""
                <div style="display:grid; grid-template-columns:1fr 1fr;
                            gap:0.75rem; margin-top:1.2rem; font-size:0.88rem;">
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius:12px;
                                padding:0.8rem; text-align:center;">
                        <div style="color: var(--text-secondary); font-weight:600; font-size:0.78rem;">✅ REAL PROB</div>
                        <div style="font-weight:800; color: var(--success); font-size:1.2rem; margin-top:0.2rem;">{real_pct}%</div>
                    </div>
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius:12px;
                                padding:0.8rem; text-align:center;">
                        <div style="color: var(--text-secondary); font-weight:600; font-size:0.78rem;">⚠️ FAKE PROB</div>
                        <div style="font-weight:800; color: var(--danger); font-size:1.2rem; margin-top:0.2rem;">{fake_pct}%</div>
                    </div>
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius:12px;
                                padding:0.8rem; text-align:center;">
                        <div style="color: var(--text-secondary); font-weight:600; font-size:0.78rem;">⚡ SPEED</div>
                        <div style="font-weight:800; color: var(--primary-light); font-size:1.2rem; margin-top:0.2rem;">
                            {prediction.processing_time_ms:.0f} ms
                        </div>
                    </div>
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius:12px;
                                padding:0.8rem; text-align:center;">
                        <div style="color: var(--text-secondary); font-weight:600; font-size:0.78rem;">👤 FACE CROP</div>
                        <div style="font-weight:800;
                                    color:{'var(--success)' if prediction.face_detected else 'var(--danger)'}; font-size:1.2rem; margin-top:0.2rem;">
                            {"Detected" if prediction.face_detected else "Not Found"}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error(f"❌ Detection failed: {exc}")
            status_box.empty()
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

    if st.session_state.get("last_saved_filename") != uploaded.name:
        st.session_state["last_saved_filename"] = uploaded.name
        st.session_state["last_saved_id"] = None

    with col_save:
        if st.session_state.get("last_saved_id") is not None:
            st.success("✅ Saved to detection history!")
        else:
            if st.button("💾 Save to History", use_container_width=True, key="img_save_btn"):
                try:
                    record_id = repo.insert_detection(
                        file_name=uploaded.name,
                        prediction=prediction.label,
                        confidence=prediction.confidence,
                        fake_probability=prediction.fake_probability,
                        file_type="image",
                        model_name=model_name,
                        processing_time=prediction.processing_time_ms,
                    )
                    st.session_state["last_saved_id"] = record_id
                    st.success("✅ Saved to detection history!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to save: {exc}")

    with col_pdf:
        if st.button("📄 Generate PDF Report", use_container_width=True, key="img_pdf_btn"):
            try:
                record_id = st.session_state.get("last_saved_id")
                if record_id is None:
                    record_id = repo.insert_detection(
                        file_name=uploaded.name,
                        prediction=prediction.label,
                        confidence=prediction.confidence,
                        fake_probability=prediction.fake_probability,
                        file_type="image",
                        model_name=model_name,
                        processing_time=prediction.processing_time_ms,
                    )
                    st.session_state["last_saved_id"] = record_id

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
                repo.update_report_path(record_id, str(pdf_path))
                
                with open(str(pdf_path), "rb") as f:
                    st.download_button(
                        label="⬇️ Download PDF Report",
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
        <div style="text-align:center; padding:5rem 3rem; border:2px dashed var(--border);
             border-radius:var(--radius); color: var(--text-secondary); margin-top:1rem; background: var(--bg-card);">
            <div style="font-size:3.5rem; margin-bottom: 1rem;">🖼️</div>
            <div style="font-size:1.25rem; font-weight:700; margin:0.5rem 0; color: var(--text-primary);">
                Drop an image here or click to browse
            </div>
            <div style="font-size:0.88rem; opacity: 0.85;">Supported file formats: JPG · JPEG · PNG · BMP · WEBP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
