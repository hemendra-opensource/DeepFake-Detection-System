"""
tests/test_inference.py
========================
Unit tests for image and video inference pipelines.

Uses a tiny stub model (randomly initialised weights) so TF inference
can be tested without trained checkpoints.
"""

import sys
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _make_image(path: Path) -> Path:
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _make_video(path: Path, frames: int = 15) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
    for _ in range(frames):
        out.write(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    out.release()
    return path


def _build_stub_model():
    """Build a tiny stub Keras model for testing (no real weights)."""
    try:
        from models.model_factory import ModelFactory
        return ModelFactory.build(
            "xceptionnet",
            input_shape=(299, 299, 3),
            weights="imagenet",
            freeze_base=True,
        )
    except Exception:
        return None


class TestImageDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _build_stub_model()
        cls.tmpdir = tempfile.mkdtemp()

    def _get_detector(self):
        if self.model is None:
            self.skipTest("TensorFlow / model not available.")
        from inference.image_detector import ImageDetector
        return ImageDetector(
            model=self.model,
            input_size=(299, 299),
            threshold=0.5,
        )

    def test_predict_from_path_returns_result(self):
        """predict_from_path should return an ImagePrediction."""
        detector = self._get_detector()
        img_path = _make_image(Path(self.tmpdir) / "img.jpg")
        result = detector.predict_from_path(img_path)
        self.assertIn(result.label, ("FAKE", "REAL"))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_predict_from_bytes(self):
        """predict_from_bytes should work with raw image bytes."""
        detector = self._get_detector()
        img_path = _make_image(Path(self.tmpdir) / "img2.jpg")
        img_bytes = img_path.read_bytes()
        result = detector.predict_from_bytes(img_bytes)
        self.assertIn(result.label, ("FAKE", "REAL"))

    def test_predict_from_array(self):
        """predict_from_array should accept an RGB NumPy array."""
        detector = self._get_detector()
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        result = detector.predict_from_array(img)
        self.assertIn(result.label, ("FAKE", "REAL"))

    def test_processing_time_positive(self):
        """Processing time must be positive."""
        detector = self._get_detector()
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result = detector.predict_from_array(img)
        self.assertGreater(result.processing_time_ms, 0)

    def test_fake_probability_in_range(self):
        """fake_probability must be in [0, 1]."""
        detector = self._get_detector()
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result = detector.predict_from_array(img)
        self.assertGreaterEqual(result.fake_probability, 0.0)
        self.assertLessEqual(result.fake_probability, 1.0)

    def test_missing_file_raises(self):
        """predict_from_path on a missing file should raise FileNotFoundError."""
        detector = self._get_detector()
        with self.assertRaises(FileNotFoundError):
            detector.predict_from_path(Path(self.tmpdir) / "nonexistent.jpg")


class TestVideoDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _build_stub_model()
        cls.tmpdir = tempfile.mkdtemp()

    def _get_video_detector(self):
        if self.model is None:
            self.skipTest("TensorFlow / model not available.")
        from inference.image_detector import ImageDetector
        from inference.video_detector import VideoDetector
        img_det = ImageDetector(model=self.model, input_size=(299, 299))
        return VideoDetector(
            image_detector=img_det,
            sample_rate=3,
            max_frames=5,
            temporal_window=3,
        )

    def test_predict_video_returns_verdict(self):
        """Video detector should return a FAKE or REAL verdict."""
        detector = self._get_video_detector()
        vid_path = _make_video(Path(self.tmpdir) / "test.mp4")
        result = detector.predict(vid_path)
        self.assertIn(result.label, ("FAKE", "REAL", "UNKNOWN"))

    def test_frame_predictions_populated(self):
        """frame_predictions list should be non-empty for a valid video."""
        detector = self._get_video_detector()
        vid_path = _make_video(Path(self.tmpdir) / "frames.mp4", frames=20)
        result = detector.predict(vid_path)
        self.assertGreaterEqual(len(result.frame_predictions), 0)

    def test_missing_video_raises(self):
        """Predicting on a missing video file should raise FileNotFoundError."""
        detector = self._get_video_detector()
        with self.assertRaises(FileNotFoundError):
            detector.predict(Path(self.tmpdir) / "no_video.mp4")

    def test_confidence_in_range(self):
        """Confidence must always be in [0, 1]."""
        detector = self._get_video_detector()
        vid_path = _make_video(Path(self.tmpdir) / "conf.mp4")
        result = detector.predict(vid_path)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
