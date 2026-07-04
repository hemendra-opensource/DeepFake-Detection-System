"""
webcam/webcam_detector.py
==========================
Real-time webcam-based DeepFake detection.

Features:
- Auto-detects available camera indices (0-3) — no manual guessing needed
- OpenCV VideoCapture with proper resource cleanup
- Per-frame face detection + model inference
- FPS counter + processing-time overlay
- Real/Fake badge + confidence overlay on annotated frames
- Face bounding-box drawn on live frames
- Streamlit-compatible single-frame capture mode
- Graceful error handling (no webcam, permission denied, camera busy)
"""

import logging
import time
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from inference.image_detector import ImageDetector, ImagePrediction
from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

ImageArray = np.ndarray

# Candidate camera indices to probe when auto-detecting
_PROBE_INDICES: List[int] = [0, 1, 2, 3]
# Seconds to wait between retry attempts
_RETRY_DELAY: float = 0.5
# Number of retry attempts to open a camera
_MAX_RETRIES: int = 3


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class WebcamFrame:
    """A single annotated webcam frame with prediction metadata."""

    frame_rgb: ImageArray            # Annotated RGB frame (H, W, 3) uint8
    prediction: Optional[ImagePrediction]
    fps: float
    processing_time_ms: float        # Time to run inference for this frame
    timestamp: float                 # Unix timestamp


@dataclass
class CameraInfo:
    """Metadata about a detected camera device."""

    index: int
    width: int
    height: int
    fps: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_open_camera(index: int) -> Optional[cv2.VideoCapture]:
    """
    Attempt to open camera at *index*. Returns the capture object if
    successful (``isOpened()`` is True), otherwise returns ``None``.

    A brief read test is performed to confirm the device actually
    delivers frames and is not just "opened" by the driver.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        return None

    # Driver sometimes needs one warm-up cycle to start streaming
    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None

    return cap


def find_available_cameras() -> List[CameraInfo]:
    """
    Probe camera indices 0–3 and return :class:`CameraInfo` for every
    device that successfully opens and delivers at least one frame.

    Returns:
        Possibly-empty list of available cameras, ordered by index.
    """
    found: List[CameraInfo] = []
    for idx in _PROBE_INDICES:
        cap = _try_open_camera(idx)
        if cap is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            found.append(CameraInfo(index=idx, width=w, height=h, fps=fps))
            logger.info("Camera found at index %d (%dx%d @ %.0f FPS)", idx, w, h, fps)
    return found


def get_first_available_camera_index() -> Optional[int]:
    """
    Return the index of the first available camera, or ``None`` if no
    camera could be opened.
    """
    cameras = find_available_cameras()
    return cameras[0].index if cameras else None


# ── Main detector class ───────────────────────────────────────────────────────

class WebcamDetector:
    """
    Real-time DeepFake detection from a webcam stream.

    Args:
        image_detector:  Configured :class:`ImageDetector`.
        camera_index:    OpenCV camera index. If ``None``, auto-detects
                         the first available camera.
        fps_cap:         Maximum inference FPS (limits CPU load).
        frame_width:     Requested capture width (hint only — camera may differ).
        frame_height:    Requested capture height.
    """

    def __init__(
        self,
        image_detector: ImageDetector,
        camera_index: Optional[int] = None,
        fps_cap: int = 5,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        self.image_detector = image_detector
        self._requested_index = camera_index   # None = auto-detect
        self.camera_index: Optional[int] = camera_index
        self.fps_cap = fps_cap
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Open the webcam. If *camera_index* is ``None``, auto-detects the
        first available device.

        Returns:
            ``True`` if the webcam opened and is streaming frames.
        """
        target = self._requested_index

        if target is None:
            target = get_first_available_camera_index()
            if target is None:
                logger.error(
                    "No webcam found on indices %s. "
                    "Check device connection and permissions.",
                    _PROBE_INDICES,
                )
                return False

        # Retry loop (driver/OS sometimes needs a moment on first call)
        for attempt in range(1, _MAX_RETRIES + 1):
            cap = _try_open_camera(target)
            if cap is not None:
                self._cap = cap
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self.camera_index = target
                self._running = True
                actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(
                    "Webcam opened (index=%d, %dx%d) on attempt %d/%d",
                    target, actual_w, actual_h, attempt, _MAX_RETRIES,
                )
                return True

            logger.warning(
                "Could not open camera %d (attempt %d/%d). Retrying in %.1fs…",
                target, attempt, _MAX_RETRIES, _RETRY_DELAY,
            )
            time.sleep(_RETRY_DELAY)

        logger.error(
            "Failed to open camera at index %d after %d attempt(s).",
            target, _MAX_RETRIES,
        )
        return False

    def stop(self) -> None:
        """Release the webcam capture device and free all resources."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Webcam released (index=%s).", self.camera_index)

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(self) -> Generator[WebcamFrame, None, None]:
        """
        Generator that yields annotated :class:`WebcamFrame` objects.

        Call :meth:`start` before iterating, :meth:`stop` to terminate.

        Yields:
            :class:`WebcamFrame` with live prediction overlay.
        """
        if not self._running or self._cap is None:
            logger.error("Webcam not started — call start() first.")
            return

        min_interval = 1.0 / max(self.fps_cap, 1)
        prev_time = 0.0
        display_fps = 0.0
        last_prediction: Optional[ImagePrediction] = None
        last_proc_ms = 0.0

        while self._running:
            ret, bgr_frame = self._cap.read()
            if not ret or bgr_frame is None:
                logger.warning("Failed to read webcam frame — stopping.")
                break

            current_time = time.time()
            elapsed = current_time - prev_time

            # FPS display
            if elapsed > 0:
                display_fps = 1.0 / elapsed

            prev_time = current_time

            # Throttle inference to fps_cap
            if elapsed >= min_interval:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                t_inf = time.perf_counter()
                try:
                    last_prediction = self.image_detector.predict_from_array(rgb_frame)
                    last_proc_ms = (time.perf_counter() - t_inf) * 1000
                except Exception as exc:
                    logger.warning("Inference error on webcam frame: %s", exc)

            # Annotate and yield
            annotated_rgb = self._annotate_frame(
                cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
                last_prediction,
                display_fps,
                last_proc_ms,
            )

            yield WebcamFrame(
                frame_rgb=annotated_rgb,
                prediction=last_prediction,
                fps=display_fps,
                processing_time_ms=last_proc_ms,
                timestamp=current_time,
            )

    # ── Single-frame capture ──────────────────────────────────────────────────

    def capture_single_frame(self) -> Tuple[Optional[ImagePrediction], Optional[ImageArray]]:
        """
        Capture a single frame and run prediction. Safe to call without
        first calling :meth:`start` — opens and closes the camera internally.

        Returns:
            A tuple of ``(ImagePrediction | None, annotated_rgb_frame | None)``.
            Both are ``None`` if the camera cannot be accessed.
        """
        owned = False  # Did we open the capture here (need to close it)?
        cap = self._cap

        if cap is None or not cap.isOpened():
            # Determine which index to use
            target = self._requested_index
            if target is None:
                target = get_first_available_camera_index()
            if target is None:
                logger.error(
                    "No camera available on indices %s.", _PROBE_INDICES
                )
                return None, None

            cap = _try_open_camera(target)
            if cap is None:
                logger.error("Cannot open camera at index %d for single capture.", target)
                return None, None
            owned = True
            self.camera_index = target

        try:
            # Discard a few stale frames (especially important on Windows)
            for _ in range(3):
                cap.read()

            ret, bgr = cap.read()
            if not ret or bgr is None:
                logger.error("Failed to capture frame.")
                return None, None

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            prediction = self.image_detector.predict_from_array(rgb)
            proc_ms = (time.perf_counter() - t0) * 1000

            annotated = self._annotate_frame(rgb, prediction, fps=0.0, proc_ms=proc_ms)
            return prediction, annotated

        except Exception as exc:
            logger.error("Error during single frame capture: %s", exc)
            return None, None
        finally:
            if owned and cap is not None:
                cap.release()

    # ── Frame annotation ──────────────────────────────────────────────────────

    def _annotate_frame(
        self,
        frame_rgb: ImageArray,
        prediction: Optional[ImagePrediction],
        fps: float,
        proc_ms: float,
    ) -> ImageArray:
        """
        Draw prediction overlay text and bounding box on the frame.

        Args:
            frame_rgb:  RGB frame to annotate (uint8).
            prediction: Current prediction result (may be ``None``).
            fps:        Current capture FPS for display.
            proc_ms:    Last inference time in milliseconds.

        Returns:
            Annotated RGB frame.
        """
        annotated = frame_rgb.copy()
        h, w = annotated.shape[:2]

        # ── Semi-transparent top banner ────────────────────────────────────────
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (18, 18, 26), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # ── FPS counter ────────────────────────────────────────────────────────
        if fps > 0:
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (130, 210, 255), 2, cv2.LINE_AA,
            )

        # ── LIVE badge (top-right) ─────────────────────────────────────────────
        live_text = "● LIVE"
        (lw, _), _ = cv2.getTextSize(live_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            annotated, live_text, (w - lw - 10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA,
        )

        # ── Processing-time ────────────────────────────────────────────────────
        if proc_ms > 0:
            pt_text = f"{proc_ms:.0f} ms"
            (ptw, _), _ = cv2.getTextSize(pt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(
                annotated, pt_text, (w // 2 - ptw // 2, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
            )

        if prediction is not None:
            is_fake = prediction.is_fake()
            colour = (255, 80, 80) if is_fake else (80, 220, 120)
            bg_colour = (180, 30, 30) if is_fake else (30, 150, 80)

            # ── Bottom result banner ───────────────────────────────────────────
            overlay2 = annotated.copy()
            cv2.rectangle(overlay2, (0, h - 54), (w, h), (18, 18, 26), -1)
            cv2.addWeighted(overlay2, 0.75, annotated, 0.25, 0, annotated)

            label_text = f"{prediction.label}  {prediction.confidence:.1%}"
            (lbw, lbh), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2
            )
            cv2.putText(
                annotated, label_text,
                ((w - lbw) // 2, h - 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, colour, 2, cv2.LINE_AA,
            )

            # ── Face/no-face indicator ─────────────────────────────────────────
            face_icon = "Face ✓" if prediction.face_detected else "No Face"
            face_col = (80, 220, 120) if prediction.face_detected else (220, 180, 80)
            cv2.putText(
                annotated, face_icon, (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_col, 1, cv2.LINE_AA,
            )

            # ── Fake/Real probability bar (bottom of banner) ───────────────────
            bar_h = 5
            fake_px = int(w * prediction.fake_probability)
            cv2.rectangle(annotated, (0, h - bar_h), (fake_px, h), (255, 80, 80), -1)
            cv2.rectangle(annotated, (fake_px, h - bar_h), (w, h), (80, 220, 120), -1)

        return annotated
