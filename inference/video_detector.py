"""
inference/video_detector.py
============================
Frame-wise video DeepFake detection module.

Pipeline:
  1. Extract frames at configurable sample rate
  2. Detect and crop face per frame
  3. Run model inference per frame
  4. Apply temporal smoothing (rolling average)
  5. Majority voting for overall verdict
  6. Return per-frame results + aggregated prediction
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from inference.image_detector import ImageDetector, ImagePrediction
from utils.file_utils import load_yaml_config
from utils.logger import get_logger
from utils.video_utils import extract_frames, frame_iterator, get_video_metadata

logger: logging.Logger = get_logger(__name__)


@dataclass
class FramePrediction:
    """Prediction result for a single video frame."""

    frame_index: int
    sample_index: int
    timestamp_ms: float
    label: str
    confidence: float
    fake_probability: float
    real_probability: float       # = 1 - fake_probability
    face_detected: bool
    smoothed_fake_prob: float = 0.0   # After temporal smoothing


@dataclass
class VideoPrediction:
    """Aggregated DeepFake prediction result for an entire video."""

    label: str                          # Final verdict: "FAKE" or "REAL"
    confidence: float                   # Confidence in the final verdict
    fake_frame_count: int
    real_frame_count: int
    total_frames_analysed: int
    fake_frame_ratio: float
    processing_time_ms: float
    video_path: Optional[str] = None
    fps: float = 0.0
    duration_seconds: float = 0.0
    frame_predictions: List[FramePrediction] = field(default_factory=list)

    @property
    def fake_percentages(self) -> List[float]:
        """Smoothed fake probabilities over time (for charts)."""
        return [fp.smoothed_fake_prob for fp in self.frame_predictions]

    @property
    def timestamps(self) -> List[float]:
        """Timestamps in milliseconds for each analysed frame."""
        return [fp.timestamp_ms for fp in self.frame_predictions]

    def __str__(self) -> str:
        return (
            f"[{self.label}] confidence={self.confidence:.1%} "
            f"fake={self.fake_frame_ratio:.1%} real={1-self.fake_frame_ratio:.1%} "
            f"fake_frames={self.fake_frame_count}/{self.total_frames_analysed} "
            f"time={self.processing_time_ms:.0f}ms"
        )


class VideoDetector:
    """
    Performs frame-wise DeepFake detection on video files.

    Args:
        image_detector:      A configured :class:`ImageDetector` instance.
        sample_rate:         Analyse every Nth frame.
        max_frames:          Maximum frames to analyse per video.
        temporal_window:     Rolling average window for smoothing.
        threshold:           Decision threshold for FAKE.
    """

    def __init__(
        self,
        image_detector: ImageDetector,
        sample_rate: int = 5,
        max_frames: int = 200,
        temporal_window: int = 5,
        threshold: Optional[float] = None,
    ) -> None:
        self.image_detector = image_detector
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.temporal_window = temporal_window
        # Prefer the tuned threshold from ImageDetector so video-level verdict
        # is consistent with per-frame calibration.
        if threshold is None:
            self.threshold = getattr(image_detector, "threshold", 0.5)
        else:
            self.threshold = threshold
        logger.info(
            "VideoDetector ready | sample_rate=%d | max_frames=%d | window=%d | threshold=%.3f",
            sample_rate, max_frames, temporal_window, self.threshold,
        )

    def predict(self, video_path: str | Path) -> VideoPrediction:
        """
        Run full DeepFake detection on a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            :class:`VideoPrediction` with per-frame and aggregated results.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        t_start = time.perf_counter()

        # Get metadata
        meta = get_video_metadata(path)
        if not meta.is_valid:
            raise ValueError(f"Invalid or unreadable video: {path}")

        # Extract and analyse frames
        frame_preds: List[FramePrediction] = []
        frames = extract_frames(
            path,
            sample_rate=self.sample_rate,
            max_frames=self.max_frames,
        )

        if not frames:
            logger.warning("No frames extracted from video: %s", path)
            return self._empty_prediction(str(path), meta, t_start)

        logger.info("Analysing %d frames from: %s", len(frames), path.name)

        for frame_result in frames:
            img_pred: ImagePrediction = self.image_detector.predict_from_array(
                frame_result.image
            )
            frame_preds.append(
                FramePrediction(
                    frame_index=frame_result.frame_index,
                    sample_index=frame_result.sample_index,
                    timestamp_ms=frame_result.timestamp_ms,
                    label=img_pred.label,
                    confidence=img_pred.confidence,
                    fake_probability=img_pred.fake_probability,
                    real_probability=img_pred.real_probability,
                    face_detected=img_pred.face_detected,
                )
            )

        # Temporal smoothing
        frame_preds = self._apply_temporal_smoothing(frame_preds)

        # Majority voting
        verdict = self._majority_vote(frame_preds)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        fake_count = sum(1 for fp in frame_preds if fp.label == "FAKE")
        real_count = len(frame_preds) - fake_count
        fake_ratio = fake_count / len(frame_preds)

        result = VideoPrediction(
            label=verdict["label"],
            confidence=verdict["confidence"],
            fake_frame_count=fake_count,
            real_frame_count=real_count,
            total_frames_analysed=len(frame_preds),
            fake_frame_ratio=fake_ratio,
            processing_time_ms=elapsed_ms,
            video_path=str(path),
            fps=meta.fps,
            duration_seconds=meta.duration_seconds,
            frame_predictions=frame_preds,
        )

        logger.info("Video prediction: %s", result)
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_temporal_smoothing(
        self, predictions: List[FramePrediction]
    ) -> List[FramePrediction]:
        """
        Apply a rolling-average temporal smoothing to fake probabilities.

        Reduces noise from single erroneous frames.

        Args:
            predictions: Raw per-frame predictions.

        Returns:
            Same list with ``smoothed_fake_prob`` populated.
        """
        raw_probs = np.array([fp.fake_probability for fp in predictions])
        kernel = np.ones(self.temporal_window) / self.temporal_window
        if len(raw_probs) >= self.temporal_window:
            smoothed = np.convolve(raw_probs, kernel, mode="same")
        else:
            smoothed = raw_probs.copy()

        for fp, sp in zip(predictions, smoothed):
            fp.smoothed_fake_prob = float(np.clip(sp, 0.0, 1.0))

        return predictions

    def _majority_vote(
        self, predictions: List[FramePrediction]
    ) -> dict:
        """
        Compute final verdict by majority vote over smoothed probabilities.

        Uses the same threshold as the per-frame ImageDetector so that the
        video-level verdict is consistent with per-frame decisions.

            fake_probability = mean(smoothed_fake_prob across frames)
            real_probability = 1 − fake_probability
            label            = "FAKE" if fake_probability >= threshold else "REAL"
            confidence       = probability of the predicted class
        """
        avg_fake_prob = float(
            np.mean([fp.smoothed_fake_prob for fp in predictions])
        )
        avg_fake_prob = float(np.clip(avg_fake_prob, 0.0, 1.0))
        avg_real_prob = 1.0 - avg_fake_prob

        label = "FAKE" if avg_fake_prob >= self.threshold else "REAL"
        confidence = avg_fake_prob if label == "FAKE" else avg_real_prob
        return {"label": label, "confidence": confidence}

    def _empty_prediction(
        self,
        video_path: str,
        meta: "VideoMetadata",  # type: ignore[name-defined]
        t_start: float,
    ) -> VideoPrediction:
        """Return an empty prediction when no frames could be extracted."""
        return VideoPrediction(
            label="UNKNOWN",
            confidence=0.0,
            fake_frame_count=0,
            real_frame_count=0,
            total_frames_analysed=0,
            fake_frame_ratio=0.0,
            processing_time_ms=(time.perf_counter() - t_start) * 1000,
            video_path=video_path,
            fps=meta.fps,
            duration_seconds=meta.duration_seconds,
        )
