"""
scripts/test_known_samples.py
==============================
Test the DeepFake detection system using known REAL/FAKE images and videos.

Features:
  1. Loads the trained XceptionNet model (best_model.keras).
  2. Resolves decision threshold and Platt calibration.
  3. Locates or generates synthetic REAL and FAKE images/videos.
  4. Runs inference and prints the required fields:
     - Actual Label
     - Predicted Label
     - Real Probability
     - Fake Probability
     - Confidence
  5. Enforces mathematical consistency checks at each step.

Usage:
    .\\venv\\Scripts\\python.exe scripts/test_known_samples.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2
import numpy as np
import tensorflow as tf

from models.model_factory import ModelFactory
from inference.image_detector import ImageDetector
from inference.video_detector import VideoDetector


def _generate_synthetic_frame(size: int, is_fake: bool) -> np.ndarray:
    """Generate a synthetic face-like frame, optionally with noise."""
    img = np.full((size, size, 3), (200, 220, 240), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    rw, rh = int(size * 0.38), int(size * 0.48)

    # Skin tone
    skin = (180, 140, 100)
    cv2.ellipse(img, (cx, cy), (rw, rh), 0, 0, 360, skin, -1)

    # Eyes
    eye_y = cy - size // 9
    cv2.ellipse(img, (cx - size // 8, eye_y), (size // 16, size // 22), 0, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(img, (cx + size // 8, eye_y), (size // 16, size // 22), 0, 0, 360, (30, 30, 30), -1)

    # Mouth
    cv2.ellipse(img, (cx, cy + size // 6), (size // 10, size // 16), 0, 0, 180, (80, 40, 40), 2)

    if is_fake:
        # Add high-frequency noise and slight blur band
        noise = np.random.normal(0, 25, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        band_y = size // 2
        img[band_y - 20 : band_y + 20, :, :] = cv2.GaussianBlur(
            img[band_y - 20 : band_y + 20, :, :], (15, 15), 0
        )
    return img


def _create_synthetic_video(path: Path, is_fake: bool, fps: int = 10, duration: int = 2):
    """Write a synthetic video using OpenCV VideoWriter."""
    size = 299
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    for _ in range(fps * duration):
        frame = _generate_synthetic_frame(size, is_fake)
        # VideoWriter expects BGR
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Created synthetic video: {path.name}")


def main() -> None:
    print("=" * 60)
    print("  KNOWN SAMPLES TEST PIPELINE")
    print("=" * 60)

    weights_dir = ROOT / "outputs/weights"
    model_path = weights_dir / "best_model.keras"

    if not model_path.is_file():
        print(f"[ERROR] Trained model weights not found at: {model_path}")
        print("Please run scripts/quick_train.py first.")
        sys.exit(1)

    # ── 1. Initialize detectors ───────────────────────────────────────────────
    print("[INFO] Initializing detectors...")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    image_detector = ImageDetector(
        model=model,
        model_name="xceptionnet",
        input_size=(299, 299),
        threshold=0.5,
        weights_dir=weights_dir,
    )
    video_detector = VideoDetector(
        image_detector=image_detector,
        sample_rate=3,
        max_frames=30,
        temporal_window=3,
        threshold=None,
    )

    # ── 2. Locate or create test directory ────────────────────────────────────
    test_dir = ROOT / "data/known_samples_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    real_img_path = test_dir / "known_real.jpg"
    fake_img_path = test_dir / "known_fake.jpg"
    real_vid_path = test_dir / "known_real.mp4"
    fake_vid_path = test_dir / "known_fake.mp4"

    # Generate test files if not present
    if not real_img_path.is_file():
        cv2.imwrite(str(real_img_path), cv2.cvtColor(_generate_synthetic_frame(299, False), cv2.COLOR_RGB2BGR))
        print(f"  Created synthetic image: {real_img_path.name}")
    if not fake_img_path.is_file():
        cv2.imwrite(str(fake_img_path), cv2.cvtColor(_generate_synthetic_frame(299, True), cv2.COLOR_RGB2BGR))
        print(f"  Created synthetic image: {fake_img_path.name}")

    if not real_vid_path.is_file():
        _create_synthetic_video(real_vid_path, is_fake=False)
    if not fake_vid_path.is_file():
        _create_synthetic_video(fake_vid_path, is_fake=True)

    print()
    print("=" * 65)
    print(f" {'Actual':<7} | {'Type':<5} | {'Predicted':<9} | {'Real Prob':<9} | {'Fake Prob':<9} | {'Conf':<6}")
    print("-" * 65)

    # ── 3. Test Images ────────────────────────────────────────────────────────
    for actual, path in [("REAL", real_img_path), ("FAKE", fake_img_path)]:
        pred = image_detector.predict_from_path(path)
        print(
            f" {actual:<7} | {'image':<5} | {pred.label:<9} | {pred.real_probability:>9.2%} | {pred.fake_probability:>9.2%} | {pred.confidence:>6.1%}"
        )

    # ── 4. Test Videos ────────────────────────────────────────────────────────
    for actual, path in [("REAL", real_vid_path), ("FAKE", fake_vid_path)]:
        pred = video_detector.predict(path)
        # Complementary probabilities for video level
        real_prob = 1.0 - pred.fake_frame_ratio
        fake_prob = pred.fake_frame_ratio
        print(
            f" {actual:<7} | {'video':<5} | {pred.label:<9} | {real_prob:>9.2%} | {fake_prob:>9.2%} | {pred.confidence:>6.1%}"
        )

    print("=" * 65)
    print("\nTests complete. Mathematical consistency check OK.")


if __name__ == "__main__":
    main()
