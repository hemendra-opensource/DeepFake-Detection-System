"""
dashboard/pages/settings.py
=============================
Settings page for the DeepFake Detection dashboard.
Enables fine-tuning confidence thresholds and model settings.
"""

import streamlit as st


def render(image_detector: "ImageDetector", video_detector: "VideoDetector") -> None:  # type: ignore[name-defined]
    """
    Render the Settings page.

    Args:
        image_detector: Active ImageDetector instance.
        video_detector: Active VideoDetector instance.
    """
    st.markdown("## ⚙️ Settings & Configuration")
    st.markdown(
        "<p style='color: var(--text-secondary);'>Configure decision parameters, thresholds, and pipeline settings.</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Inference settings ────────────────────────────────────────────────────
    st.markdown("### 🎯 Detection Thresholds")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        new_thresh = st.slider(
            "Image Decision Threshold (P(FAKE) >= threshold)",
            min_value=0.05,
            max_value=0.95,
            value=float(image_detector.threshold),
            step=0.01,
            key="settings_thresh",
            help="Higher threshold reduces False Positives (more REAL predictions). Lower threshold increases sensitivity.",
        )
        if new_thresh != image_detector.threshold:
            image_detector.threshold = new_thresh
            st.success(f"Threshold updated to **{new_thresh:.2f}** for Image and Video detection.")

    with col_t2:
        st.info(
            "💡 **ROC Calibration Info**: Changing this overrides any automatic "
            "weights directory threshold (threshold.json). If you reset settings, "
            "the system will fall back to model-fitted values."
        )

    st.divider()

    # ── Video processing settings ─────────────────────────────────────────────
    st.markdown("### 🎥 Video Extraction Parameters")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        new_sample_rate = st.slider(
            "Frame Sample Rate (extract every Nth frame)",
            min_value=1,
            max_value=20,
            value=int(video_detector.sample_rate),
            step=1,
            key="settings_sample_rate",
        )
        if new_sample_rate != video_detector.sample_rate:
            video_detector.sample_rate = new_sample_rate
            st.success(f"Video frame sample rate updated to **{new_sample_rate}**.")

    with col_v2:
        new_max_frames = st.slider(
            "Maximum Video Frames to Analyze",
            min_value=10,
            max_value=500,
            value=int(video_detector.max_frames),
            step=10,
            key="settings_max_frames",
        )
        if new_max_frames != video_detector.max_frames:
            video_detector.max_frames = new_max_frames
            st.success(f"Max video frames updated to **{new_max_frames}**.")

    st.divider()

    # ── Pipeline components settings ──────────────────────────────────────────
    st.markdown("### 🧩 Detector Modules")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        face_conf = st.slider(
            "Minimum Face Detection Confidence",
            min_value=0.5,
            max_value=0.95,
            value=float(image_detector.face_extractor.min_confidence),
            step=0.05,
            key="settings_face_conf",
        )
        if face_conf != image_detector.face_extractor.min_confidence:
            image_detector.face_extractor.min_confidence = face_conf
            st.success(f"Face extraction confidence updated to **{face_conf:.2f}**.")

    with col_m2:
        # Calibration toggles
        cal_enabled = st.toggle(
            "Enable Platt Scaling Probability Calibration",
            value=image_detector.calibration_params.get("slope", 1.0) != 1.0,
            key="settings_cal_enabled",
            help="Apply fitted sigmoid calibration parameters. If disabled, raw logits are output.",
        )
        st.caption(
            f"Active parameters: Slope = **{image_detector.calibration_params.get('slope', 1.0):.4f}** | "
            f"Intercept = **{image_detector.calibration_params.get('intercept', 0.0):.4f}**"
        )
