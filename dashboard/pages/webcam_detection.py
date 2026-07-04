"""
dashboard/pages/webcam_detection.py
=====================================
Live Webcam DeepFake Detection page.

Workflow:
  1. Auto-detect available cameras (no manual index needed).
  2. User clicks "Capture Frame & Detect".
  3. A single frame is captured, face detected, model run, Grad-CAM rendered.
  4. Results are shown: annotated frame with class-colored border, badge, metrics.
"""

import time
from typing import Optional

import numpy as np
import streamlit as st

from inference.image_detector import ImageDetector, ImagePrediction


def render(
    image_detector: ImageDetector,
    repo: "DetectionRepository",   # type: ignore[name-defined]
    model_name: str = "XceptionNet",
    gradcam: "Optional[GradCAM]" = None,  # type: ignore[name-defined]
) -> None:
    """
    Render the Webcam Detection page.

    Args:
        image_detector: Loaded :class:`ImageDetector`.
        repo:           Detection history repository.
        model_name:     Model display name shown in history.
        gradcam:        Optional Grad-CAM instance for explanations.
    """
    from webcam.webcam_detector import WebcamDetector, find_available_cameras

    st.markdown("## 📡 Live Webcam Detection")
    st.markdown(
        "<p style='color: var(--text-secondary);'>Analyze live video frames from your webcam. "
        "Captures faces, runs models, and visualizes attention regions.</p>",
        unsafe_allow_html=True,
    )

    # ── Camera discovery ──────────────────────────────────────────────────────
    with st.spinner("🔍 Probing cameras (indices 0–3)…"):
        cameras = find_available_cameras()

    if not cameras:
        st.error(
            "❌ **No webcam detected.**\n\n"
            "Please connect a webcam, close other camera apps, and reload this page."
        )
        _render_standalone_instructions()
        return

    # Camera selector
    if len(cameras) == 1:
        selected_cam = cameras[0]
        st.success(
            f"✅ Camera detected at index **{selected_cam.index}** "
            f"({selected_cam.width}×{selected_cam.height})"
        )
    else:
        cam_labels = {
            f"Camera {c.index} — {c.width}×{c.height}": c
            for c in cameras
        }
        choice = st.selectbox("📷 Select Camera", list(cam_labels.keys()), key="webcam_select")
        selected_cam = cam_labels[choice]

    # ── Single-frame capture & detect ─────────────────────────────────────────
    st.divider()
    st.markdown("### 📸 Capture & Analyse")

    col_btn, col_tip = st.columns([2, 3])
    with col_btn:
        capture_clicked = st.button(
            "📷 Capture Frame & Detect",
            type="primary",
            use_container_width=True,
            key="webcam_capture_btn",
        )
    with col_tip:
        st.info(
            "💡 Captures a frame, runs face detection, evaluates fake probability, "
            "and generates explanations."
        )

    if capture_clicked:
        _run_capture_and_display(
            camera_index=selected_cam.index,
            image_detector=image_detector,
            repo=repo,
            model_name=model_name,
            gradcam=gradcam,
        )

    # ── Continuous mode (repeated captures) ───────────────────────────────────
    st.divider()
    st.markdown("### 🔄 Continuous Capture Mode")

    col_auto, col_int = st.columns([3, 1])
    with col_int:
        interval = st.slider(
            "Capture Interval (seconds)", min_value=1, max_value=10, value=3, step=1,
            key="webcam_interval",
        )
    with col_auto:
        auto_on = st.toggle(
            "▶️ Start Continuous Capture",
            value=st.session_state.get("webcam_auto_on", False),
            key="webcam_auto_on",
        )

    if auto_on:
        frame_slot = st.empty()
        result_slot = st.empty()
        st.info(f"⏱️ Capturing every **{interval}s** — toggle off above to stop.")

        last_capture = st.session_state.get("webcam_last_capture_ts", 0.0)
        now = time.time()

        if now - last_capture >= interval:
            st.session_state["webcam_last_capture_ts"] = now
            detector = WebcamDetector(
                image_detector=image_detector,
                camera_index=selected_cam.index,
            )
            prediction, annotated_rgb = detector.capture_single_frame()

            if prediction is not None and annotated_rgb is not None:
                border_color = "var(--danger)" if prediction.is_fake() else "var(--success)"
                
                with frame_slot.container():
                    st.markdown(
                        f"""
                        <div style="border: 4px solid {border_color}; border-radius: var(--radius);
                                    box-shadow: 0 0 15px {border_color}; padding: 4px; overflow: hidden;">
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(
                        annotated_rgb,
                        caption=f"{prediction.label} | {prediction.confidence:.1%}",
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                _render_compact_result(result_slot, prediction)
            else:
                frame_slot.warning("⚠️ Could not capture frame — check camera status.")

        # Rerun loop
        time.sleep(0.2)
        st.rerun()

    # ── Standalone instructions ────────────────────────────────────────────────
    st.divider()
    _render_standalone_instructions()


# ── Private helpers ───────────────────────────────────────────────────────────

def _run_capture_and_display(
    camera_index: int,
    image_detector: ImageDetector,
    repo: "DetectionRepository",  # type: ignore[name-defined]
    model_name: str,
    gradcam: "Optional[GradCAM]" = None,  # type: ignore[name-defined]
) -> None:
    """Capture one frame, run detection, and render results."""
    from webcam.webcam_detector import WebcamDetector
    from dashboard.components.kpi_cards import render_prediction_badge

    detector = WebcamDetector(
        image_detector=image_detector,
        camera_index=camera_index,
    )

    with st.spinner(f"📷 Accessing Camera {camera_index}…"):
        prediction, annotated_rgb = detector.capture_single_frame()

    if prediction is None or annotated_rgb is None:
        st.error(f"❌ Could not capture a frame from Camera {camera_index}.")
        return

    # ── Display annotated frame with glowing border ────────────────────────────
    col_frame, col_result = st.columns([3, 2], gap="large")

    with col_frame:
        border_color = "var(--danger)" if prediction.is_fake() else "var(--success)"
        st.markdown(
            f"""
            <div style="border: 4px solid {border_color}; border-radius: var(--radius);
                        box-shadow: 0 0 20px {border_color}; padding: 4px; overflow: hidden;
                        margin-bottom: 0.5rem; animation: fadeIn 0.4s ease-out;">
            """,
            unsafe_allow_html=True,
        )
        st.image(
            annotated_rgb,
            caption=f"Camera {camera_index} — Live Frame",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        render_prediction_badge(prediction.label, prediction.confidence)
        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bars
        _render_probability_bars(prediction)

        # Stats
        st.markdown(
            f"""
            <div style="
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius:12px;
                padding:1rem;
                font-size:0.88rem;
                margin-top:1.2rem;
                animation: fadeIn 0.4s ease-out;
            ">
                <div style="color: var(--text-secondary); margin-bottom:0.5rem; font-weight:600;">⚙️ DETECTION STATS</div>
                <div style="color: var(--text-primary); line-height: 1.6;">
                    ⚡ Latency: <b>{prediction.processing_time_ms:.0f} ms</b><br>
                    👤 Face Detection: {"<b style='color: var(--success)'>Detected</b>" if prediction.face_detected
                                       else "<b style='color: var(--danger)'>Not Found</b>"}<br>
                    🎯 Face Confidence: <b>{prediction.face_confidence:.1%}</b><br>
                    📊 Threshold Used: <b>{prediction.threshold_used:.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Grad-CAM explanation ───────────────────────────────────────────────────
    if gradcam is not None and prediction.preprocessed_face is not None:
        st.markdown("#### 🔥 Grad-CAM Activation Map")
        try:
            from dashboard.components.gradcam_viewer import render_gradcam
            render_gradcam(
                gradcam=gradcam,
                image_rgb=prediction.preprocessed_face,
                target_class=1 if prediction.is_fake() else 0,
            )
        except Exception as exc:
            st.warning(f"Grad-CAM could not be rendered: {exc}")

    # ── Save to history ────────────────────────────────────────────────────────
    try:
        repo.insert_detection(
            file_name=f"webcam_camera{camera_index}",
            prediction=prediction.label,
            confidence=prediction.confidence,
            fake_probability=prediction.fake_probability,
            file_type="image",
            model_name=model_name,
            processing_time=prediction.processing_time_ms,
        )
        st.caption("✅ Auto-saved to history.")
    except Exception:
        pass


def _render_probability_bars(prediction: ImagePrediction) -> None:
    """Render Real / Fake probability bars."""
    real_pct = prediction.real_probability * 100
    fake_pct = prediction.fake_probability * 100

    st.markdown(
        f"""
        <div style="font-size:0.88rem; margin-bottom:0.5rem; animation: fadeIn 0.4s ease-out;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color: var(--success); font-weight:600;">✅ REAL</span>
                <span>{real_pct:.1f}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.06); border-radius:6px; height:12px; overflow:hidden;">
                <div style="width:{real_pct:.1f}%; background:linear-gradient(90deg, #10B981, var(--success));
                            height:100%; border-radius:6px;"></div>
            </div>

            <div style="display:flex; justify-content:space-between; margin-top:10px; margin-bottom:4px;">
                <span style="color: var(--danger); font-weight:600;">🚨 FAKE</span>
                <span>{fake_pct:.1f}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.06); border-radius:6px; height:12px; overflow:hidden;">
                <div style="width:{fake_pct:.1f}%; background:linear-gradient(90deg, var(--warning), var(--danger));
                            height:100%; border-radius:6px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_compact_result(slot: "st.delta_generator.DeltaGenerator", prediction: ImagePrediction) -> None:  # type: ignore[name-defined]
    """Write a compact one-liner result into *slot*."""
    colour = "var(--danger)" if prediction.is_fake() else "var(--success)"
    slot.markdown(
        f"<div style='text-align:center; font-size:1.2rem; color:{colour}; font-weight:700; "
        f"margin-top:0.5rem; text-shadow: 0 0 10px rgba(0,0,0,0.3);'>"
        f"{prediction.label} — {prediction.confidence:.1%} confidence</div>",
        unsafe_allow_html=True,
    )


def _render_standalone_instructions() -> None:
    """Show instructions for running the OpenCV live-stream window."""
    with st.expander("🖥️ How to run low-latency OpenCV stream window"):
        st.markdown(
            """
            Streamlit's browser interface has processing delay for live camera.
            For zero-latency local window, run from the root directory:

            ```python
            from app import resources
            from webcam.webcam_detector import WebcamDetector
            import cv2

            detector = WebcamDetector(
                image_detector=resources["image_detector"],
                camera_index=0,
                fps_cap=10,
            )
            if detector.start():
                for wf in detector.stream():
                    bgr = cv2.cvtColor(wf.frame_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Webcam Live DeepFake Detector", bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                detector.stop()
            cv2.destroyAllWindows()
            ```
            """
        )
