"""
inference/image_detector.py
============================
Single-image DeepFake detection module.

Label convention (must match training):
    0 = REAL   (negative class)
    1 = FAKE   (positive class)

The model's single sigmoid output = P(FAKE).
Therefore:
    fake_probability = model_output           ∈ [0, 1]
    real_probability = 1 − fake_probability   ∈ [0, 1]

Decision rule:
    label      = "FAKE" if fake_probability >= threshold else "REAL"
    confidence = fake_probability  if label == "FAKE"
               = real_probability  if label == "REAL"

This guarantees that confidence always equals the probability of the
predicted class — never the probability of the opposite class.

Threshold default: 0.50 (configurable via config.yaml → inference.confidence_threshold
or overridden by outputs/weights/threshold.json after proper ROC calibration on
REAL validation data, NOT toy data).
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from preprocessing.face_extractor import FaceExtractor, FaceDetection
from utils.image_utils import load_image_rgb, load_image_from_bytes, preprocess_for_model
from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

ImageArray = np.ndarray


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ImagePrediction:
    """
    Result of a single image DeepFake prediction.

    All four probability/confidence fields are mathematically consistent:
        real_probability + fake_probability == 1.0
        confidence == fake_probability  if label == "FAKE"
        confidence == real_probability  if label == "REAL"
        confidence == probability of the PREDICTED class, always in [0.5, 1.0]
    """

    label: str                    # "FAKE" or "REAL"
    fake_probability: float       # P(FAKE) — direct model output ∈ [0, 1]
    real_probability: float       # P(REAL) = 1 − fake_probability
    confidence: float             # P(predicted class) — always >= 0.5 when threshold == 0.5
    processing_time_ms: float
    face_detected: bool
    face_confidence: float
    image_path: Optional[str] = None
    preprocessed_face: Optional[ImageArray] = None  # (H, W, 3) for Grad-CAM
    raw_model_output: float = 0.0  # Uncalibrated sigmoid output, for debugging
    threshold_used: float = 0.5    # Decision threshold that produced this result

    def is_fake(self) -> bool:
        return self.label == "FAKE"

    def __post_init__(self) -> None:
        """Enforce mathematical consistency — raises if values are incoherent."""
        # Round to avoid floating-point noise
        rp = round(self.real_probability, 6)
        fp = round(self.fake_probability, 6)
        if abs(rp + fp - 1.0) > 1e-4:
            raise ValueError(
                f"Incoherent probabilities: real={rp} + fake={fp} = {rp + fp:.6f} ≠ 1.0"
            )
        # Confidence must match the predicted class
        expected_conf = fp if self.label == "FAKE" else rp
        if abs(round(self.confidence, 6) - round(expected_conf, 6)) > 1e-4:
            raise ValueError(
                f"Confidence {self.confidence:.6f} does not match predicted class "
                f"({self.label}) probability {expected_conf:.6f}"
            )

    def __str__(self) -> str:
        return (
            f"[{self.label}] confidence={self.confidence:.1%}  "
            f"fake={self.fake_probability:.4f}  real={self.real_probability:.4f}  "
            f"raw={self.raw_model_output:.4f}  thresh={self.threshold_used:.3f}  "
            f"time={self.processing_time_ms:.1f}ms"
        )


# ── Detector ──────────────────────────────────────────────────────────────────

class ImageDetector:
    """
    Detects whether an image is a DeepFake.

    The model outputs a single sigmoid value = P(FAKE).
    Label mapping: 0 = REAL, 1 = FAKE (matches training label_int convention).

    Args:
        model:               Loaded Keras model (binary sigmoid classifier).
        model_name:          Model identifier string (for logging/display).
        input_size:          (height, width) expected by the model.
        threshold:           Decision threshold. Default 0.50.
                             IMPORTANT: Only override with a threshold derived from
                             a proper ROC curve on REAL validation data — not toy data.
        face_margin:         Fractional margin around detected face crop.
        min_face_confidence: Minimum MediaPipe face detection confidence.
        weights_dir:         Directory to search for calibration.json and threshold.json.
                             If None, searches outputs/weights then models/weights.
    """

    def __init__(
        self,
        model: "tf.keras.Model",          # type: ignore[name-defined]
        model_name: str = "xceptionnet",
        input_size: Tuple[int, int] = (299, 299),
        threshold: float = 0.5,
        face_margin: float = 0.3,
        min_face_confidence: float = 0.7,
        weights_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.input_size = input_size
        self.threshold = threshold  # Will be overridden by threshold.json if valid

        # ── Load calibration parameters ────────────────────────────────────────
        # slope=1.0, intercept=0.0 = identity (no-op calibration)
        self.calibration_params: dict = {"slope": 1.0, "intercept": 0.0}

        search_dirs = []
        if weights_dir is not None:
            search_dirs.append(Path(weights_dir))
        search_dirs += [Path("outputs/weights"), Path("models/weights")]

        for path in search_dirs:
            cal_file = path / "calibration.json"
            thresh_file = path / "threshold.json"

            # Load calibration
            if cal_file.is_file():
                try:
                    with open(cal_file, "r") as f:
                        loaded_cal = json.load(f)
                    slope = float(loaded_cal.get("slope", 1.0))
                    intercept = float(loaded_cal.get("intercept", 0.0))
                    # SAFETY: reject toy-dataset calibration artefacts.
                    # A slope > 3.0 means the calibration was fitted on trivially
                    # separable data and will push real-image probabilities to
                    # extremes, making predictions unreliable.
                    if slope > 3.0:
                        logger.warning(
                            "Calibration slope=%.4f from %s is suspiciously large "
                            "(likely fitted on toy/synthetic data). "
                            "Ignoring — using identity calibration instead.",
                            slope, cal_file,
                        )
                    else:
                        self.calibration_params = {"slope": slope, "intercept": intercept}
                        logger.info(
                            "Calibration loaded: slope=%.4f intercept=%.4f from %s",
                            slope, intercept, cal_file,
                        )
                    break
                except Exception as exc:
                    logger.warning("Failed to load calibration from %s: %s", cal_file, exc)

        for path in search_dirs:
            thresh_file = path / "threshold.json"
            if thresh_file.is_file():
                try:
                    with open(thresh_file, "r") as f:
                        loaded = json.load(f)
                    loaded_thresh = float(loaded.get("threshold", 0.5))
                    # SAFETY: reject extreme thresholds produced by toy-data ROC.
                    # A threshold > 0.95 means the G-mean maximisation found a
                    # degenerate operating point on trivially-separable data.
                    if loaded_thresh > 0.95 or loaded_thresh < 0.05:
                        logger.warning(
                            "Threshold %.4f from %s is extreme (likely from toy data). "
                            "Using config default: %.4f",
                            loaded_thresh, thresh_file, threshold,
                        )
                    else:
                        self.threshold = loaded_thresh
                        logger.info(
                            "Threshold loaded: %.4f from %s", self.threshold, thresh_file
                        )
                    break
                except Exception as exc:
                    logger.warning("Failed to load threshold from %s: %s", thresh_file, exc)

        # ── Face extractor ────────────────────────────────────────────────────
        self.face_extractor = FaceExtractor(
            target_size=input_size,
            margin=face_margin,
            min_confidence=min_face_confidence,
            max_faces=1,
        )

        logger.info(
            "ImageDetector ready | model=%s | input=%s | threshold=%.4f | "
            "calibration slope=%.4f intercept=%.4f",
            model_name, input_size, self.threshold,
            self.calibration_params["slope"],
            self.calibration_params["intercept"],
        )

    # ── Public predict methods ────────────────────────────────────────────────

    def predict_from_path(self, image_path: "str | Path") -> ImagePrediction:
        """
        Run DeepFake detection on an image file.

        Args:
            image_path: Path to an image file.

        Returns:
            :class:`ImagePrediction` result.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        image_rgb = load_image_rgb(path)
        if image_rgb is None:
            raise ValueError(f"Cannot load image: {path}")
        result = self._predict(image_rgb)
        result.image_path = str(path)
        return result

    def predict_from_bytes(self, data: bytes) -> ImagePrediction:
        """
        Run DeepFake detection on raw image bytes (Streamlit upload).

        Args:
            data: Raw image bytes.

        Returns:
            :class:`ImagePrediction` result.
        """
        image_rgb = load_image_from_bytes(data)
        if image_rgb is None:
            raise ValueError("Cannot decode image from bytes.")
        return self._predict(image_rgb)

    def predict_from_array(self, image_rgb: ImageArray) -> ImagePrediction:
        """
        Run DeepFake detection on an RGB NumPy array.

        Args:
            image_rgb: RGB uint8 image array (H, W, 3).

        Returns:
            :class:`ImagePrediction` result.
        """
        return self._predict(image_rgb)

    # ── Core prediction ───────────────────────────────────────────────────────

    def _apply_calibration(self, raw_prob: float) -> float:
        """
        Apply Platt scaling calibration to the raw sigmoid output.

        Identity transform when slope=1.0, intercept=0.0.

        Args:
            raw_prob: Raw sigmoid model output ∈ [0, 1].

        Returns:
            Calibrated probability ∈ [0, 1].
        """
        slope = self.calibration_params["slope"]
        intercept = self.calibration_params["intercept"]

        # Identity shortcut — avoids log/exp precision issues
        if slope == 1.0 and intercept == 0.0:
            return raw_prob

        epsilon = 1e-7
        clipped = max(epsilon, min(1.0 - epsilon, raw_prob))
        logit = math.log(clipped / (1.0 - clipped))
        calibrated_logit = slope * logit + intercept
        return 1.0 / (1.0 + math.exp(-calibrated_logit))

    def _predict(self, image_rgb: ImageArray) -> ImagePrediction:
        """
        Core prediction pipeline:
            face detection → preprocessing → model → calibration → decision

        Label convention: model output = P(FAKE)
            fake_probability = calibrated model output
            real_probability = 1 − fake_probability
            label            = "FAKE" if fake_probability >= threshold else "REAL"
            confidence       = probability of the PREDICTED class
        """
        import cv2
        t_start = time.perf_counter()

        # ── 1. Face detection ──────────────────────────────────────────────────
        detection: Optional[FaceDetection] = self.face_extractor.extract_largest(image_rgb)
        face_detected = detection is not None
        face_conf = detection.confidence if detection else 0.0

        if detection is not None and detection.cropped_face is not None:
            face_img = detection.cropped_face
        else:
            logger.warning("No face detected — using full image for prediction.")
            # INTER_LINEAR matches tf.image.resize default (bilinear)
            face_img = cv2.resize(image_rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        # ── 2. Preprocess — must be IDENTICAL to training ─────────────────────
        # • resize: cv2.INTER_LINEAR  (matches tf.image.resize bilinear)
        # • normalize: / 255.0        (matches tf.cast(img, float32) / 255.0)
        # • channel order: RGB        (MediaPipe already returns RGB)
        # • shape: (1, H, W, 3)       (batch dimension added by preprocess_for_model)
        input_tensor = preprocess_for_model(face_img, target_size=self.input_size)

        # ── 3. Model inference ────────────────────────────────────────────────
        raw_model_output = float(self.model.predict(input_tensor, verbose=0)[0][0])

        # ── 4. Platt calibration (identity by default) ────────────────────────
        fake_probability = self._apply_calibration(raw_model_output)
        fake_probability = float(np.clip(fake_probability, 0.0, 1.0))

        # ── 5. Complementary probability ──────────────────────────────────────
        real_probability = 1.0 - fake_probability

        # ── 6. Decision — threshold on P(FAKE) ───────────────────────────────
        # Label 0 = REAL, Label 1 = FAKE (matches training label_int)
        label = "FAKE" if fake_probability >= self.threshold else "REAL"

        # ── 7. Confidence = P(predicted class) ───────────────────────────────
        # ALWAYS the probability of whichever class was predicted.
        # This ensures: if label=REAL then confidence=real_probability,
        #               if label=FAKE then confidence=fake_probability.
        # Consequence: confidence >= 0.5 when threshold == 0.5 (cannot be
        # simultaneously REAL with 9% confidence and FAKE at 91%).
        confidence = fake_probability if label == "FAKE" else real_probability

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # ── 8. Build result (dataclass enforces consistency in __post_init__) ──
        result = ImagePrediction(
            label=label,
            fake_probability=fake_probability,
            real_probability=real_probability,
            confidence=confidence,
            processing_time_ms=elapsed_ms,
            face_detected=face_detected,
            face_confidence=face_conf,
            raw_model_output=raw_model_output,
            threshold_used=self.threshold,
        )

        # ── 9. Structured debug log ───────────────────────────────────────────
        logger.info(
            "PREDICTION | label=%-4s | confidence=%5.1f%% | "
            "fake_prob=%6.4f | real_prob=%6.4f | "
            "raw_output=%6.4f | threshold=%.4f | "
            "face=%s | time=%.0fms",
            result.label,
            result.confidence * 100,
            result.fake_probability,
            result.real_probability,
            result.raw_model_output,
            result.threshold_used,
            "yes" if result.face_detected else "no",
            result.processing_time_ms,
        )

        return result
