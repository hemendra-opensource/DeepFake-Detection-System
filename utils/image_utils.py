"""
utils/image_utils.py
====================
Image preprocessing and manipulation helpers.

Provides:
- Image loading (OpenCV + Pillow fallback)
- Resizing with aspect-ratio options
- Normalisation for model input
- BGR ↔ RGB conversion
- Heatmap overlay generation
- Image saving utilities
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# Type alias for clarity
ImageArray = np.ndarray


# ── Loading ───────────────────────────────────────────────────────────────────

def load_image_cv2(path: str | Path, color: bool = True) -> Optional[ImageArray]:
    """
    Load an image using OpenCV.

    Args:
        path:  Path to the image file.
        color: If ``True``, load as BGR (default). If ``False``, load as grayscale.

    Returns:
        NumPy array (H, W, C) in BGR format, or ``None`` on failure.
    """
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        logger.warning("cv2.imread failed for: %s", path)
    return img


def load_image_rgb(path: str | Path) -> Optional[ImageArray]:
    """
    Load an image and convert to RGB (H, W, 3) uint8 array.

    Tries OpenCV first; falls back to Pillow on failure.

    Args:
        path: Image file path.

    Returns:
        RGB NumPy array or ``None`` if both loaders fail.
    """
    img_bgr = load_image_cv2(path)
    if img_bgr is not None:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Pillow fallback
    try:
        pil_img = Image.open(str(path)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        logger.error("Cannot load image %s: %s", path, exc)
        return None


def load_image_from_bytes(data: bytes) -> Optional[ImageArray]:
    """
    Decode an image from raw bytes (e.g. from a Streamlit file upload).

    Args:
        data: Raw image bytes.

    Returns:
        RGB NumPy array or ``None`` on failure.
    """
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.error("Failed to decode image from bytes: %s", exc)
        return None


# ── Conversion ────────────────────────────────────────────────────────────────

def bgr_to_rgb(img: ImageArray) -> ImageArray:
    """Convert a BGR image array to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: ImageArray) -> ImageArray:
    """Convert an RGB image array to BGR."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def to_pil(img: ImageArray) -> Image.Image:
    """
    Convert an RGB NumPy array to a PIL Image.

    Args:
        img: RGB image as uint8 array.

    Returns:
        PIL Image object.
    """
    return Image.fromarray(img.astype(np.uint8))


# ── Resizing ──────────────────────────────────────────────────────────────────

def resize_image(
    img: ImageArray,
    target_size: Tuple[int, int],
    keep_aspect: bool = False,
) -> ImageArray:
    """
    Resize an image to the target size.

    Args:
        img:         Input image (H, W, C).
        target_size: ``(width, height)`` tuple.
        keep_aspect: If ``True``, pad with black to preserve aspect ratio.

    Returns:
        Resized image array.
    """
    target_w, target_h = target_size

    if keep_aspect:
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to exact target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        return canvas

    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_image(img: ImageArray) -> np.ndarray:
    """
    Normalize pixel values to ``[0.0, 1.0]`` (float32).

    Args:
        img: Input image (uint8 or float).

    Returns:
        Float32 array with values in ``[0, 1]``.
    """
    return img.astype(np.float32) / 255.0


def preprocess_for_model(
    img: ImageArray,
    target_size: Tuple[int, int] = (299, 299),
) -> np.ndarray:
    """
    Full preprocessing pipeline for model inference.

    Steps: resize → normalize → add batch dimension.

    Args:
        img:         RGB image array.
        target_size: ``(width, height)`` for the model input.

    Returns:
        Float32 array of shape ``(1, H, W, 3)`` ready for model prediction.
    """
    resized = resize_image(img, target_size)
    normalized = normalize_image(resized)
    return np.expand_dims(normalized, axis=0)


# ── Heatmap overlay ───────────────────────────────────────────────────────────

def apply_heatmap_overlay(
    original: ImageArray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> ImageArray:
    """
    Blend a Grad-CAM heatmap onto the original image.

    Args:
        original: Original RGB image (H, W, 3) uint8.
        heatmap:  Normalised heatmap float array (H, W) in ``[0, 1]``.
        alpha:    Heatmap opacity (0 = invisible, 1 = full heatmap).
        colormap: OpenCV colormap constant (default ``COLORMAP_JET``).

    Returns:
        Blended RGB image as uint8 array.
    """
    # Resize heatmap to match original
    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap (produces BGR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Weighted blend
    original_float = original.astype(np.float32)
    heatmap_float = heatmap_rgb.astype(np.float32)
    blended = (1 - alpha) * original_float + alpha * heatmap_float
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── Saving ────────────────────────────────────────────────────────────────────

def save_image(img: ImageArray, path: str | Path, as_rgb: bool = True) -> bool:
    """
    Save an image array to disk.

    Args:
        img:    Image array (H, W, 3).
        path:   Destination file path.
        as_rgb: If ``True``, assumes the array is in RGB order and converts
                to BGR before saving (OpenCV default). Set ``False`` if the
                array is already in BGR.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if as_rgb else img
    success = cv2.imwrite(str(path), save_img)
    if not success:
        logger.error("Failed to save image to: %s", path)
    return success


def is_valid_image(path: str | Path) -> bool:
    """
    Check whether a file is a valid, readable image.

    Attempts to decode the first bytes of the file using both
    OpenCV and Pillow for robustness.

    Args:
        path: Path to the image file.

    Returns:
        ``True`` if the file is a valid image.
    """
    try:
        img = load_image_cv2(str(path))
        if img is not None:
            return True
        # Try Pillow fallback
        with Image.open(str(path)) as pil_img:
            pil_img.verify()
        return True
    except Exception:
        return False
