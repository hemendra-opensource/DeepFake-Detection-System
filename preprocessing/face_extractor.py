"""
preprocessing/face_extractor.py
================================
Face detection and extraction from images and video frames.

Primary detector:  MediaPipe Face Detection
Fallback detector: OpenCV Haar Cascade (works offline without extra models)

Outputs:
- Cropped face images (with configurable margin)
- Face bounding boxes
- Detection confidence
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

ImageArray = np.ndarray


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    """Result of a single face detection."""

    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixel coords
    confidence: float                  # Detection confidence in [0, 1]
    cropped_face: Optional[ImageArray] = None  # Cropped & resized face image


# ── MediaPipe detector ────────────────────────────────────────────────────────

class MediaPipeFaceDetector:
    """
    Face detector backed by MediaPipe's Short-Range Face Detection model.

    Falls back gracefully if MediaPipe is not installed.

    Args:
        min_confidence: Minimum detection confidence threshold.
        model_selection: MediaPipe model (0 = short-range ≤2m, 1 = full-range).
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        model_selection: int = 0,
    ) -> None:
        self.min_confidence = min_confidence
        self._detector = None
        self._available = False
        self._init_detector(model_selection)

    def _init_detector(self, model_selection: int) -> None:
        """Attempt to initialise the MediaPipe face detector."""
        try:
            import mediapipe as mp  # type: ignore
            face_detection = mp.solutions.face_detection
            self._detector = face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=self.min_confidence,
            )
            self._available = True
            logger.info("MediaPipe face detector initialised.")
        except ImportError:
            logger.warning(
                "MediaPipe not available. Install with: pip install mediapipe"
            )
        except Exception as exc:
            logger.warning("MediaPipe initialisation failed: %s", exc)

    @property
    def is_available(self) -> bool:
        """``True`` if MediaPipe loaded successfully."""
        return self._available

    def detect(self, image_rgb: ImageArray) -> List[FaceDetection]:
        """
        Detect faces in an RGB image.

        Args:
            image_rgb: Input image in RGB format (H, W, 3) uint8.

        Returns:
            List of :class:`FaceDetection` objects sorted by confidence (desc).
        """
        if not self._available or self._detector is None:
            return []

        h, w = image_rgb.shape[:2]
        results = self._detector.process(image_rgb)

        if not results.detections:
            return []

        detections: List[FaceDetection] = []
        for det in results.detections:
            score = det.score[0] if det.score else 0.0
            bbox = det.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            detections.append(FaceDetection(bbox=(x1, y1, x2, y2), confidence=score))

        return sorted(detections, key=lambda d: d.confidence, reverse=True)


# ── OpenCV Haar Cascade fallback ──────────────────────────────────────────────

class HaarCascadeFaceDetector:
    """
    Face detector using OpenCV's Haar Cascade classifier.

    Always available since it uses built-in OpenCV data files.

    Args:
        scale_factor:   Image scale factor per detection pass.
        min_neighbours: Minimum neighbour rectangles needed for a positive.
        min_size:       Minimum face bounding box size in pixels.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbours: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbours = min_neighbours
        self.min_size = min_size

        if self._cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from: {cascade_path}"
            )
        logger.info("Haar Cascade face detector initialised.")

    def detect(self, image_rgb: ImageArray) -> List[FaceDetection]:
        """
        Detect faces in an RGB image.

        Args:
            image_rgb: Input image in RGB format.

        Returns:
            List of :class:`FaceDetection` objects (confidence is approximate).
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbours,
            minSize=self.min_size,
        )

        detections: List[FaceDetection] = []
        if len(faces) == 0:
            return detections

        for x, y, fw, fh in faces:
            detections.append(
                FaceDetection(
                    bbox=(x, y, x + fw, y + fh),
                    confidence=0.8,  # Haar doesn't return probabilities
                )
            )
        return detections


# ── High-level FaceExtractor ──────────────────────────────────────────────────

class FaceExtractor:
    """
    High-level face extraction utility that combines MediaPipe (primary)
    with Haar Cascade (fallback).

    Args:
        target_size:    ``(width, height)`` of the output cropped face.
        margin:         Fractional margin to add around detected face
                        (0.3 = 30% padding on each side).
        min_confidence: Minimum face detection confidence to accept.
        max_faces:      Maximum number of faces to return per image.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (299, 299),
        margin: float = 0.3,
        min_confidence: float = 0.7,
        max_faces: int = 1,
    ) -> None:
        self.target_size = target_size
        self.margin = margin
        self.min_confidence = min_confidence
        self.max_faces = max_faces

        # Initialise detectors
        self._mediapipe = MediaPipeFaceDetector(min_confidence=min_confidence)
        self._haar = HaarCascadeFaceDetector()

        if self._mediapipe.is_available:
            logger.info("Primary detector: MediaPipe")
        else:
            logger.info("Primary detector: Haar Cascade (MediaPipe unavailable)")

    def extract(self, image_rgb: ImageArray) -> List[FaceDetection]:
        """
        Detect and crop faces from an RGB image.

        Args:
            image_rgb: Input RGB image array.

        Returns:
            List of :class:`FaceDetection` with ``cropped_face`` populated.
        """
        # Try MediaPipe first
        if self._mediapipe.is_available:
            detections = self._mediapipe.detect(image_rgb)
        else:
            detections = []

        # Fall back to Haar Cascade
        if not detections:
            detections = self._haar.detect(image_rgb)

        # Filter by confidence and cap count
        detections = [d for d in detections if d.confidence >= self.min_confidence]
        detections = detections[: self.max_faces]

        # Crop faces
        h, w = image_rgb.shape[:2]
        for det in detections:
            det.cropped_face = self._crop_face(image_rgb, det.bbox, h, w)

        return detections

    def extract_largest(self, image_rgb: ImageArray) -> Optional[FaceDetection]:
        """
        Extract the single largest detected face.

        Args:
            image_rgb: Input RGB image array.

        Returns:
            The highest-confidence :class:`FaceDetection`, or ``None`` if no
            face is found.
        """
        detections = self.extract(image_rgb)
        if not detections:
            return None

        # Return largest by bounding-box area
        return max(
            detections,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _crop_face(
        self,
        image: ImageArray,
        bbox: Tuple[int, int, int, int],
        img_h: int,
        img_w: int,
    ) -> ImageArray:
        """
        Crop a face with margin padding and resize to ``target_size``.

        Args:
            image: Source RGB image.
            bbox:  ``(x1, y1, x2, y2)`` bounding box.
            img_h: Image height.
            img_w: Image width.

        Returns:
            Cropped and resized face as RGB uint8 array.
        """
        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1

        # Add margin
        dx = int(face_w * self.margin)
        dy = int(face_h * self.margin)

        x1_m = max(0, x1 - dx)
        y1_m = max(0, y1 - dy)
        x2_m = min(img_w, x2 + dx)
        y2_m = min(img_h, y2 + dy)

        face_crop = image[y1_m:y2_m, x1_m:x2_m]

        if face_crop.size == 0:
            # Fallback: return the centre of the image
            logger.warning("Empty face crop — using image centre instead.")
            face_crop = image

        return cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_AREA)
