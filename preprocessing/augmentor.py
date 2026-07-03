"""
preprocessing/augmentor.py
===========================
Data augmentation for DeepFake detection training.

Uses Albumentations for fast, label-preserving augmentation.
Falls back to OpenCV-based transforms if Albumentations is unavailable.

Augmentations applied:
- Horizontal flip
- Brightness & contrast adjustment
- Random rotation
- Gaussian blur
- Gaussian noise
- JPEG compression artifacts (simulates compressed video)
"""

import logging
import random
from typing import Optional

import cv2
import numpy as np

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

ImageArray = np.ndarray


# ── Albumentations-backed augmentor ───────────────────────────────────────────

class Augmentor:
    """
    Image augmentation pipeline for training data.

    Attempts to use Albumentations; falls back to a simple
    OpenCV-based implementation if unavailable.

    Args:
        horizontal_flip:    Enable random horizontal flip.
        brightness_limit:   Max brightness change (±fraction).
        contrast_limit:     Max contrast change (±fraction).
        rotation_limit:     Max rotation angle in degrees.
        noise_probability:  Probability of adding Gaussian noise.
        blur_probability:   Probability of applying Gaussian blur.
        jpeg_probability:   Probability of JPEG compression artefacts.
        seed:               Random seed for reproducibility.
    """

    def __init__(
        self,
        horizontal_flip: bool = True,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        rotation_limit: int = 10,
        noise_probability: float = 0.3,
        blur_probability: float = 0.2,
        jpeg_probability: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.horizontal_flip = horizontal_flip
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.rotation_limit = rotation_limit
        self.noise_probability = noise_probability
        self.blur_probability = blur_probability
        self.jpeg_probability = jpeg_probability
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self._transform = None
        self._use_albumentations = False
        self._init_albumentations()

    def _init_albumentations(self) -> None:
        """Try to build an Albumentations pipeline."""
        try:
            import albumentations as A  # type: ignore

            transforms = []

            if self.horizontal_flip:
                transforms.append(A.HorizontalFlip(p=0.5))

            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=self.brightness_limit,
                    contrast_limit=self.contrast_limit,
                    p=0.5,
                )
            )

            if self.rotation_limit > 0:
                transforms.append(A.Rotate(limit=self.rotation_limit, p=0.4))

            if self.blur_probability > 0:
                transforms.append(
                    A.GaussianBlur(blur_limit=(3, 7), p=self.blur_probability)
                )

            if self.noise_probability > 0:
                transforms.append(
                    A.GaussNoise(var_limit=(10.0, 50.0), p=self.noise_probability)
                )

            if self.jpeg_probability > 0:
                transforms.append(
                    A.ImageCompression(
                        quality_lower=60, quality_upper=95, p=self.jpeg_probability
                    )
                )

            self._transform = A.Compose(transforms)
            self._use_albumentations = True
            logger.info("Albumentations augmentation pipeline ready.")

        except ImportError:
            logger.warning(
                "Albumentations not installed; using lightweight OpenCV fallback."
            )
        except Exception as exc:
            logger.warning("Albumentations init failed (%s); using fallback.", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def augment(self, image: ImageArray) -> ImageArray:
        """
        Apply augmentation to a single image.

        Args:
            image: RGB uint8 image array (H, W, 3).

        Returns:
            Augmented RGB uint8 image array of the same spatial size.
        """
        if self._use_albumentations and self._transform is not None:
            return self._albumentations_augment(image)
        return self._opencv_augment(image)

    def augment_batch(self, images: list[ImageArray]) -> list[ImageArray]:
        """
        Augment a list of images independently.

        Args:
            images: List of RGB uint8 image arrays.

        Returns:
            List of augmented images.
        """
        return [self.augment(img) for img in images]

    # ── Albumentations path ────────────────────────────────────────────────────

    def _albumentations_augment(self, image: ImageArray) -> ImageArray:
        """Apply the Albumentations pipeline."""
        try:
            result = self._transform(image=image)
            return result["image"]
        except Exception as exc:
            logger.warning("Albumentations augment error: %s — returning original.", exc)
            return image

    # ── OpenCV fallback ───────────────────────────────────────────────────────

    def _opencv_augment(self, image: ImageArray) -> ImageArray:
        """Lightweight augmentation using only NumPy and OpenCV."""
        aug = image.copy()

        # Horizontal flip
        if self.horizontal_flip and self._rng.random() < 0.5:
            aug = cv2.flip(aug, 1)

        # Brightness / contrast
        if self._rng.random() < 0.5:
            alpha = 1.0 + self._rng.uniform(-self.contrast_limit, self.contrast_limit)
            beta = self._np_rng.integers(
                int(-self.brightness_limit * 255),
                int(self.brightness_limit * 255),
            )
            aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(
                np.uint8
            )

        # Rotation
        if self.rotation_limit > 0 and self._rng.random() < 0.4:
            angle = self._rng.uniform(-self.rotation_limit, self.rotation_limit)
            h, w = aug.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # Gaussian blur
        if self._rng.random() < self.blur_probability:
            ksize = self._rng.choice([3, 5, 7])
            aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

        # Gaussian noise
        if self._rng.random() < self.noise_probability:
            noise = self._np_rng.normal(0, 15, aug.shape).astype(np.float32)
            aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # JPEG compression artefact
        if self._rng.random() < self.jpeg_probability:
            quality = self._rng.randint(60, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(aug, cv2.COLOR_RGB2BGR), encode_param)
            aug = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        return aug
