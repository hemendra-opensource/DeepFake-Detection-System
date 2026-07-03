"""
tests/test_gradcam.py
======================
Unit tests for the Grad-CAM explainability module.
"""

import sys
import os
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _build_stub_model():
    try:
        from models.model_factory import ModelFactory
        return ModelFactory.build(
            "xceptionnet", input_shape=(299, 299, 3),
            weights="imagenet", freeze_base=True,
        )
    except Exception:
        return None


class TestGradCAM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _build_stub_model()
        if cls.model is not None:
            from gradcam.grad_cam import GradCAM
            from models.model_factory import ModelFactory
            target_layer = ModelFactory.get_gradcam_layer("xceptionnet")
            try:
                cls.gradcam = GradCAM(model=cls.model, target_layer=target_layer)
            except Exception:
                cls.gradcam = None
        else:
            cls.gradcam = None

    def _skip_if_unavailable(self):
        if self.gradcam is None:
            self.skipTest("Grad-CAM or model not available.")

    def test_generate_returns_result(self):
        """generate() should return a GradCAMResult with all fields."""
        self._skip_if_unavailable()
        img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
        result = self.gradcam.generate(img, class_index=1)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.heatmap)
        self.assertIsNotNone(result.overlay)
        self.assertIsNotNone(result.heatmap_colored)

    def test_heatmap_shape_matches_input(self):
        """Heatmap should be resized to match the original image dimensions."""
        self._skip_if_unavailable()
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = self.gradcam.generate(img, class_index=1)
        self.assertEqual(result.heatmap.shape, (200, 300))

    def test_heatmap_values_in_range(self):
        """Heatmap values should be in [0, 1]."""
        self._skip_if_unavailable()
        img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
        result = self.gradcam.generate(img, class_index=1)
        self.assertGreaterEqual(float(result.heatmap.min()), 0.0)
        self.assertLessEqual(float(result.heatmap.max()), 1.0 + 1e-6)

    def test_overlay_dtype_uint8(self):
        """Overlay image must be uint8."""
        self._skip_if_unavailable()
        img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        result = self.gradcam.generate(img, class_index=0)
        self.assertEqual(result.overlay.dtype, np.uint8)

    def test_overlay_shape_matches_original(self):
        """Overlay must be the same size as the original image."""
        self._skip_if_unavailable()
        img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        result = self.gradcam.generate(img, class_index=1)
        self.assertEqual(result.overlay.shape, img.shape)

    def test_explain_prediction_selects_class(self):
        """explain_prediction should select class 1 for fake_prob >= 0.5."""
        self._skip_if_unavailable()
        img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
        result = self.gradcam.explain_prediction(img, fake_probability=0.8)
        self.assertEqual(result.class_index, 1)
        result2 = self.gradcam.explain_prediction(img, fake_probability=0.2)
        self.assertEqual(result2.class_index, 0)

    def test_gradient_strength_non_negative(self):
        """gradient_strength should always be >= 0."""
        self._skip_if_unavailable()
        img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
        result = self.gradcam.generate(img)
        self.assertGreaterEqual(result.gradient_strength, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
