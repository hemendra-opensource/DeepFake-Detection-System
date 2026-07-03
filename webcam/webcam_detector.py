"""
webcam/webcam_detector.py
==========================
Real-time webcam-based DeepFake detection.

Features:
- OpenCV VideoCapture loop
- Per-frame face detection + model inference
- FPS counter
- Graceful error handling (no webcam, permission denied)
- Streamlit-compatible frame streaming
"""

import logging
import time
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from inference.image_detector import ImageDetector, ImagePrediction
from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

ImageArray = np.ndarray


@dataclass
class WebcamFrame:
    """A single annotated webcam frame with prediction metadata."""

    frame_rgb: ImageArray           # Annotated RGB frame for display
    prediction: Optional[ImagePrediction]
    fps: float
    timestamp: float                # Unix timestamp


class WebcamDetector:
    """
    Real-time DeepFake detection from a webcam stream.

    Args:
        image_detector:  Configured :class:`ImageDetector`.
        camera_index:    OpenCV camera index (usually 0 for built-in webcam).
        fps_cap:         Maximum processing FPS (to limit CPU load).
        frame_width:     Capture width.
        frame_height:    Capture height.
    """

    def __init__(
        self,
        image_detector: ImageDetector,
        camera_index: int = 0,
        fps_cap: int = 10,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        self.image_detector = image_detector
        self.camera_index = camera_index
        self.fps_cap = fps_cap
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False

    def start(self) -> bool:
        """
        Open the webcam capture device.

        Returns:
            ``True`` if the webcam was opened successfully.
        """
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error(
                "Cannot open webcam at index %d. "
                "Check device connection and permissions.",
                self.camera_index,
            )
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._running = True
        logger.info(
            "Webcam opened (index=%d, %dx%d)",
            self.camera_index, self.frame_width, self.frame_height,
        )
        return True

    def stop(self) -> None:
        """Release the webcam capture device."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Webcam released.")

    def stream(self) -> Generator[WebcamFrame, None, None]:
        """
        Generator that yields annotated :class:`WebcamFrame` objects.

        Call :meth:`start` before iterating. Call :meth:`stop` to end.

        Yields:
            :class:`WebcamFrame` with prediction overlay.
        """
        if not self._running or self._cap is None:
            logger.error("Webcam not started. Call start() first.")
            return

        min_interval = 1.0 / self.fps_cap
        prev_time = 0.0
        fps = 0.0
        last_prediction: Optional[ImagePrediction] = None

        while self._running:
            ret, bgr_frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read webcam frame — stopping.")
                break

            current_time = time.time()
            elapsed = current_time - prev_time

            # FPS calculation
            if elapsed > 0:
                fps = 1.0 / elapsed if elapsed < 10 else fps
            prev_time = current_time

            # Throttle inference
            if elapsed >= min_interval:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                try:
                    last_prediction = self.image_detector.predict_from_array(rgb_frame)
                except Exception as exc:
                    logger.warning("Inference error on webcam frame: %s", exc)

            # Annotate frame
            display_frame = self._annotate_frame(
                cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
                last_prediction,
                fps,
            )

            yield WebcamFrame(
                frame_rgb=display_frame,
                prediction=last_prediction,
                fps=fps,
                timestamp=current_time,
            )

    def capture_single_frame(self) -> Optional[ImagePrediction]:
        """
        Capture a single frame and run prediction.

        Useful for Streamlit's manual "Capture & Analyse" button.

        Returns:
            :class:`ImagePrediction` or ``None`` if capture fails.
        """
        if self._cap is None or not self._cap.isOpened():
            # One-shot capture
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                logger.error("Cannot open webcam for single capture.")
                return None
            ret, frame = cap.read()
            cap.release()
        else:
            ret, frame = self._cap.read()

        if not ret or frame is None:
            logger.error("Failed to capture frame.")
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.image_detector.predict_from_array(rgb)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _annotate_frame(
        self,
        frame_rgb: ImageArray,
        prediction: Optional[ImagePrediction],
        fps: float,
    ) -> ImageArray:
        """
        Draw prediction overlay text on the frame.

        Args:
            frame_rgb:  RGB frame to annotate.
            prediction: Current prediction (may be None before first inference).
            fps:        Current frames per second.

        Returns:
            Annotated RGB frame.
        """
        annotated = frame_rgb.copy()
        h, w = annotated.shape[:2]

        # FPS counter (top-left)
        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

        # LIVE badge (top-right)
        live_text = "● LIVE"
        (lw, lh), _ = cv2.getTextSize(live_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            annotated, live_text, (w - lw - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

        if prediction is not None:
            colour = (255, 60, 60) if prediction.is_fake() else (60, 200, 60)
            label_text = (
                f"{prediction.label}  {prediction.confidence:.1%}"
            )
            cv2.putText(
                annotated, label_text, (10, h - 15),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, colour, 2, cv2.LINE_AA,
            )

        return annotated
