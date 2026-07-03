"""
tests/test_preprocessing.py
=============================
Unit tests for the preprocessing pipeline modules.

Tests:
- DatasetValidator (file validation, duplicate detection)
- FaceExtractor (face detection and cropping)
- Augmentor (augmentation output shape/dtype)
- Image & video utility helpers
"""

import sys
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import cv2

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dummy_image(path: Path, w: int = 200, h: int = 200) -> Path:
    """Create a valid JPEG image at *path* and return the path."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _make_dummy_video(path: Path, frames: int = 10) -> Path:
    """Create a minimal MP4 video at *path* and return the path."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 5, (64, 64))
    for _ in range(frames):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


# ── DatasetValidator tests ────────────────────────────────────────────────────

class TestDatasetValidator(unittest.TestCase):

    def setUp(self):
        from preprocessing.dataset_validator import DatasetValidator
        self.validator = DatasetValidator(dataset_name="test", compute_hashes=True)
        self.tmpdir = tempfile.mkdtemp()

    def test_valid_image_passes(self):
        """A properly written JPEG should be reported as valid."""
        img_path = Path(self.tmpdir) / "test.jpg"
        _make_dummy_image(img_path)
        record = self.validator.validate_file(img_path, label="real")
        self.assertTrue(record.is_valid, "Expected image to be valid.")
        self.assertGreater(record.size_mb, 0)

    def test_corrupted_file_fails(self):
        """A file with garbage bytes should fail validation."""
        bad_path = Path(self.tmpdir) / "bad.jpg"
        bad_path.write_bytes(b"\xff\xd8garbage")
        record = self.validator.validate_file(bad_path, label="fake")
        self.assertFalse(record.is_valid, "Expected corrupted file to fail.")

    def test_missing_file_fails(self):
        """A non-existent file should fail validation gracefully."""
        record = self.validator.validate_file(
            Path(self.tmpdir) / "nonexistent.jpg", label="real"
        )
        self.assertFalse(record.is_valid)

    def test_duplicate_detection(self):
        """Two identical files should be flagged as duplicates."""
        img_path = Path(self.tmpdir) / "orig.jpg"
        dup_path = Path(self.tmpdir) / "copy.jpg"
        _make_dummy_image(img_path)
        import shutil
        shutil.copy2(img_path, dup_path)

        import shutil as sh
        from pathlib import Path as P
        files = [P(self.tmpdir) / "orig.jpg", P(self.tmpdir) / "copy.jpg"]
        from utils.file_utils import find_duplicates
        dupes = find_duplicates(files)
        self.assertTrue(len(dupes) > 0, "Expected duplicate to be detected.")


# ── FaceExtractor tests ───────────────────────────────────────────────────────

class TestFaceExtractor(unittest.TestCase):

    def setUp(self):
        from preprocessing.face_extractor import FaceExtractor
        self.extractor = FaceExtractor(target_size=(299, 299), margin=0.2)

    def test_extract_from_random_image(self):
        """
        On a random-noise image (no real face), extract_largest should
        return None OR return a correctly-sized cropped image.
        """
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = self.extractor.extract_largest(img)
        # Either None (no face found) or a FaceDetection with correct crop size
        if result is not None and result.cropped_face is not None:
            self.assertEqual(result.cropped_face.shape, (299, 299, 3))

    def test_crop_preserves_dtype(self):
        """The cropped face should be a uint8 array."""
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        result = self.extractor.extract_largest(img)
        if result is not None and result.cropped_face is not None:
            self.assertEqual(result.cropped_face.dtype, np.uint8)


# ── Augmentor tests ───────────────────────────────────────────────────────────

class TestAugmentor(unittest.TestCase):

    def setUp(self):
        from preprocessing.augmentor import Augmentor
        self.aug = Augmentor(seed=42)

    def test_output_shape_unchanged(self):
        """Augmented image must have the same shape as the input."""
        img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
        out = self.aug.augment(img)
        self.assertEqual(out.shape, img.shape)

    def test_output_dtype_is_uint8(self):
        """Augmented image must be uint8."""
        img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
        out = self.aug.augment(img)
        self.assertEqual(out.dtype, np.uint8)

    def test_batch_augment_length(self):
        """Batch augmentation must return the same number of images."""
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        outputs = self.aug.augment_batch(images)
        self.assertEqual(len(outputs), 5)

    def test_pixel_values_in_range(self):
        """All pixel values must remain in [0, 255]."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = self.aug.augment(img)
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLessEqual(int(out.max()), 255)


# ── Image utilities tests ─────────────────────────────────────────────────────

class TestImageUtils(unittest.TestCase):

    def test_load_rgb_from_disk(self):
        """load_image_rgb should return an (H, W, 3) uint8 array."""
        from utils.image_utils import load_image_rgb
        tmpdir = tempfile.mkdtemp()
        img_path = _make_dummy_image(Path(tmpdir) / "test.jpg")
        result = load_image_rgb(img_path)
        self.assertIsNotNone(result)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[2], 3)
        self.assertEqual(result.dtype, np.uint8)

    def test_normalize_range(self):
        """normalize_image should produce values in [0, 1]."""
        from utils.image_utils import normalize_image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        norm = normalize_image(img)
        self.assertAlmostEqual(float(norm.min()), 0.0, places=3)
        self.assertLessEqual(float(norm.max()), 1.0)

    def test_preprocess_for_model_shape(self):
        """preprocess_for_model should return shape (1, H, W, 3)."""
        from utils.image_utils import preprocess_for_model
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = preprocess_for_model(img, target_size=(299, 299))
        self.assertEqual(out.shape, (1, 299, 299, 3))

    def test_is_valid_image_valid(self):
        """A correctly written image should pass validation."""
        from utils.image_utils import is_valid_image
        tmpdir = tempfile.mkdtemp()
        img_path = _make_dummy_image(Path(tmpdir) / "valid.jpg")
        self.assertTrue(is_valid_image(img_path))

    def test_is_valid_image_invalid(self):
        """A garbage file should fail image validation."""
        from utils.image_utils import is_valid_image
        tmpdir = tempfile.mkdtemp()
        bad = Path(tmpdir) / "bad.jpg"
        bad.write_bytes(b"notanimage")
        self.assertFalse(is_valid_image(bad))


# ── Video utilities tests ─────────────────────────────────────────────────────

class TestVideoUtils(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_get_video_metadata(self):
        """Valid video should return non-zero fps and frame count."""
        from utils.video_utils import get_video_metadata
        vid_path = _make_dummy_video(Path(self.tmpdir) / "test.mp4")
        meta = get_video_metadata(vid_path)
        self.assertTrue(meta.is_valid)
        self.assertGreater(meta.total_frames, 0)
        self.assertGreater(meta.fps, 0)

    def test_extract_frames_returns_correct_type(self):
        """extract_frames should yield FrameResult objects with RGB images."""
        from utils.video_utils import extract_frames
        vid_path = _make_dummy_video(Path(self.tmpdir) / "frames.mp4", frames=30)
        frames = extract_frames(vid_path, sample_rate=5, max_frames=10)
        self.assertIsInstance(frames, list)
        if frames:
            self.assertEqual(frames[0].image.ndim, 3)
            self.assertEqual(frames[0].image.shape[2], 3)

    def test_is_valid_video_true(self):
        """A properly written video should pass validation."""
        from utils.video_utils import is_valid_video
        vid_path = _make_dummy_video(Path(self.tmpdir) / "valid.mp4")
        self.assertTrue(is_valid_video(vid_path))

    def test_is_valid_video_false(self):
        """A garbage file should fail video validation."""
        from utils.video_utils import is_valid_video
        bad = Path(self.tmpdir) / "bad.mp4"
        bad.write_bytes(b"notavideo")
        self.assertFalse(is_valid_video(bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
