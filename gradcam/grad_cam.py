"""
gradcam/grad_cam.py
====================
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.

Produces visual explanations highlighting which regions of the input
image influenced the DeepFake classification decision.

Reference:
    Selvaraju, R. R., et al. (2020). Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization. IJCV.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.image_utils import apply_heatmap_overlay, normalize_image
from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

ImageArray = np.ndarray


@dataclass
class GradCAMResult:
    """Output of a Grad-CAM computation."""

    original_image: ImageArray      # Original RGB image (H, W, 3)
    heatmap: np.ndarray             # Normalised heatmap float (H, W) in [0, 1]
    heatmap_colored: ImageArray     # Colourised heatmap (H, W, 3) uint8
    overlay: ImageArray             # Blended overlay (H, W, 3) uint8
    target_layer: str               # Keras layer name used
    class_index: int                # Output class that was visualised
    gradient_strength: float        # Mean gradient magnitude (proxy for confidence)


class GradCAM:
    """
    Grad-CAM explainability for Keras binary classification models.

    Args:
        model:        Compiled Keras model.
        target_layer: Name of the last convolutional layer for gradient extraction.
        alpha:        Opacity of the heatmap overlay (0–1).
    """

    def __init__(
        self,
        model: "tf.keras.Model",  # type: ignore[name-defined]
        target_layer: str,
        alpha: float = 0.4,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.alpha = alpha
        self._grad_model = None
        self._build_grad_model()

    def _build_grad_model(self) -> None:
        """Build the gradient model that outputs both the conv layer and predictions."""
        import tensorflow as tf

        try:
            layer = self.model.get_layer(self.target_layer)
            self._grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[layer.output, self.model.output],
            )
            logger.info("Grad-CAM model built with target layer: %s", self.target_layer)
        except ValueError as exc:
            logger.error(
                "Target layer '%s' not found in model. "
                "Available layers: %s\nError: %s",
                self.target_layer,
                [l.name for l in self.model.layers],
                exc,
            )
            raise

    def generate(
        self,
        image_rgb: ImageArray,
        class_index: int = 1,
        input_size: Tuple[int, int] = (299, 299),
    ) -> GradCAMResult:
        """
        Generate a Grad-CAM visualisation for a single image.

        Args:
            image_rgb:   RGB image array (H, W, 3) — can be any size.
            class_index: Output class to explain (1 = FAKE for binary classifier).
            input_size:  ``(width, height)`` for model input preprocessing.

        Returns:
            :class:`GradCAMResult` with all visualisation components.
        """
        import tensorflow as tf

        h_orig, w_orig = image_rgb.shape[:2]

        # Preprocess
        resized = cv2.resize(image_rgb, input_size, interpolation=cv2.INTER_AREA)
        input_tensor = tf.cast(
            np.expand_dims(resized.astype(np.float32) / 255.0, axis=0),
            dtype=tf.float32,
        )

        # Forward pass with gradient tape
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self._grad_model(input_tensor)
            tape.watch(conv_outputs)

            # For binary classifier (sigmoid output), class 0 or 1
            if predictions.shape[-1] == 1:
                # Binary: use the sigmoid output directly for class 1
                if class_index == 1:
                    loss = predictions[:, 0]
                else:
                    loss = 1.0 - predictions[:, 0]
            else:
                loss = predictions[:, class_index]

        # Compute gradients of the class score w.r.t. the conv output
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            logger.error("Gradient computation returned None — check target layer.")
            raise RuntimeError("Grad-CAM gradient is None.")

        # Global average pool gradients → importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps
        conv_out = conv_outputs[0]  # (H', W', C)
        weighted = conv_out * pooled_grads  # (H', W', C)
        heatmap_raw = tf.reduce_sum(weighted, axis=-1).numpy()  # (H', W')

        # ReLU + normalise
        heatmap_raw = np.maximum(heatmap_raw, 0)
        max_val = heatmap_raw.max()
        if max_val > 0:
            heatmap_norm = heatmap_raw / max_val
        else:
            heatmap_norm = heatmap_raw

        grad_strength = float(np.mean(np.abs(pooled_grads.numpy())))

        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap_norm, (w_orig, h_orig))

        # Colourised heatmap (BGR → RGB)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Overlay on original
        overlay = apply_heatmap_overlay(
            image_rgb, heatmap_resized, alpha=self.alpha
        )

        logger.debug(
            "Grad-CAM generated | layer=%s | class=%d | grad_strength=%.4f",
            self.target_layer, class_index, grad_strength,
        )

        return GradCAMResult(
            original_image=image_rgb,
            heatmap=heatmap_resized,
            heatmap_colored=heatmap_colored,
            overlay=overlay,
            target_layer=self.target_layer,
            class_index=class_index,
            gradient_strength=grad_strength,
        )

    def explain_prediction(
        self,
        image_rgb: ImageArray,
        fake_probability: float,
        threshold: float = 0.5,
        input_size: Tuple[int, int] = (299, 299),
    ) -> GradCAMResult:
        """
        Generate Grad-CAM for the predicted class.

        Automatically selects class 1 (FAKE) if fake_probability ≥ threshold,
        else explains class 0 (REAL).

        Args:
            image_rgb:        RGB image array.
            fake_probability: Model's sigmoid output for the FAKE class.
            threshold:        Decision boundary threshold. Default 0.5.
            input_size:       Model input dimensions.

        Returns:
            :class:`GradCAMResult`.
        """
        class_idx = 1 if fake_probability >= threshold else 0
        return self.generate(image_rgb, class_index=class_idx, input_size=input_size)
