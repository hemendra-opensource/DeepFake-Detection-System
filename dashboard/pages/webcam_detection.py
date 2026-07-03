"""
dashboard/pages/webcam_detection.py
=====================================
Real-time webcam DeepFake detection page.
"""

import streamlit as st


def render(
    image_detector: "ImageDetector",  # type: ignore[name-defined]
    repo: "DetectionRepository",       # type: ignore[name-defined]
    model_name: str = "XceptionNet",
) -> None:
    """
    Render the Webcam Detection page.

    Args:
        image_detector: Loaded image detector.
        repo:           Detection history repository.
        model_name:     Model display name.
    """
    from webcam.webcam_detector import WebcamDetector
    from dashboard.components.kpi_cards import render_prediction_badge

    st.markdown("## 📡 Live Webcam Detection")
    st.markdown(
        "<p style='color:#95a5a6;'>Capture a frame from your webcam and run "
        "DeepFake detection in real time.</p>",
        unsafe_allow_html=True,
    )

    # ── Settings ──────────────────────────────────────────────────────────────
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        camera_index = st.number_input(
            "Camera Index", min_value=0, max_value=5, value=0, step=1
        )
    with col_set2:
        st.info("💡 Camera index 0 = built-in webcam")

    detector = WebcamDetector(
        image_detector=image_detector,
        camera_index=int(camera_index),
        fps_cap=5,
    )

    # ── Single frame capture mode ─────────────────────────────────────────────
    st.divider()
    st.markdown("### 📸 Capture & Analyse")
    st.markdown(
        "<p style='color:#95a5a6;'>Click the button to capture one frame "
        "and run detection.</p>",
        unsafe_allow_html=True,
    )

    if st.button("📷 Capture Frame & Detect", type="primary", use_container_width=True):
        with st.spinner("Capturing from webcam…"):
            try:
                prediction = detector.capture_single_frame()
                if prediction is None:
                    st.error(
                        "❌ Could not access webcam. "
                        "Check that your webcam is connected and not in use by another app."
                    )
                else:
                    render_prediction_badge(prediction.label, prediction.confidence)

                    st.markdown(
                        f"""
                        <div style="text-align:center; font-size:0.9rem; color:#95a5a6;">
                            ⚡ Processing time: {prediction.processing_time_ms:.0f} ms &nbsp;|&nbsp;
                            👤 Face: {"✅ Detected" if prediction.face_detected else "❌ Not Found"}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Save to history
                    try:
                        repo.insert_detection(
                            file_name="webcam_capture",
                            prediction=prediction.label,
                            confidence=prediction.confidence,
                            fake_probability=prediction.fake_probability,
                            file_type="image",
                            model_name=model_name,
                            processing_time=prediction.processing_time_ms,
                        )
                    except Exception:
                        pass

            except Exception as exc:
                st.error(f"❌ Webcam error: {exc}")

    # ── Live stream mode (Streamlit-compatible) ───────────────────────────────
    st.divider()
    st.markdown("### 🎥 Live Stream Mode")

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.info(
            "⚠️ **Streamlit Note**: True live video streaming is limited in "
            "Streamlit's web model. Use the 'Capture Frame' button above for "
            "on-demand detection. For full real-time streaming, run the "
            "OpenCV window directly via `webcam_detector.py`."
        )

    with col_btn:
        if st.button("📖 View Instructions", use_container_width=True):
            st.markdown(
                """
                **To run live webcam detection in a terminal:**
                ```bash
                # From the project root:
                py -3.11 -c "
                from webcam.webcam_detector import WebcamDetector
                # (requires a loaded model — see app.py)
                "
                ```
                """
            )
