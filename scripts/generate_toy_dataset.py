"""
scripts/generate_toy_dataset.py
================================
Generate a minimal synthetic dataset for pipeline smoke-testing.

Produces:
    data/toy_dataset/
        real/   – N images of plain-coloured faces (label 0 = REAL)
        fake/   – N images of noise-corrupted faces (label 1 = FAKE)

Usage:
    python scripts/generate_toy_dataset.py [--n 50] [--size 299] [--out data/toy_dataset]

The generated images are NOT real faces.  They exist solely to test
that the full pipeline (preprocessing → training → evaluation → inference)
runs without crashing on a CPU-only machine.
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def _random_face_like(size: int) -> np.ndarray:
    """
    Generate a synthetic face-like image:
      - Flesh-toned ellipse as 'head'
      - Dark ellipses for 'eyes'
      - A simple 'mouth' arc
    Returned as a uint8 BGR image of shape (size, size, 3).
    """
    img = np.full((size, size, 3), (200, 220, 240), dtype=np.uint8)

    cx, cy = size // 2, size // 2
    rw, rh = int(size * 0.38), int(size * 0.48)

    # Skin-tone head
    skin = (
        random.randint(150, 210),
        random.randint(110, 170),
        random.randint(80, 130),
    )
    cv2.ellipse(img, (cx, cy), (rw, rh), 0, 0, 360, skin, -1)

    # Eyes
    eye_y = cy - size // 9
    eye_rx, eye_ry = size // 16, size // 22
    left_ex = cx - size // 8
    right_ex = cx + size // 8
    cv2.ellipse(img, (left_ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(img, (right_ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (30, 30, 30), -1)

    # Mouth
    mouth_y = cy + size // 6
    cv2.ellipse(
        img, (cx, mouth_y), (size // 10, size // 16),
        0, 0, 180, (80, 40, 40), 2,
    )

    return img


def _add_deepfake_artefacts(img: np.ndarray) -> np.ndarray:
    """
    Simulate DeepFake artefacts:
      - Add Gaussian noise across the image
      - Blur a random band across the face boundary
      - Slight colour shift to simulate GAN colour bias
    """
    out = img.astype(np.float32)

    # Gaussian noise
    noise = np.random.normal(0, 22, out.shape).astype(np.float32)
    out = np.clip(out + noise, 0, 255)

    # Horizontal blur band (compression artefact simulation)
    size = img.shape[0]
    band_y = random.randint(size // 4, 3 * size // 4)
    band_h = random.randint(size // 10, size // 5)
    y1 = max(0, band_y - band_h // 2)
    y2 = min(size, band_y + band_h // 2)
    band = out[y1:y2, :, :]
    band = cv2.GaussianBlur(band, (15, 15), 0)
    out[y1:y2, :, :] = band

    # Colour shift
    shift = np.array([random.uniform(-20, 20), random.uniform(-20, 20), random.uniform(-20, 20)])
    out = np.clip(out + shift, 0, 255)

    return out.astype(np.uint8)


def generate_dataset(n: int, size: int, out_dir: Path) -> None:
    real_dir = out_dir / "real"
    fake_dir = out_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n} REAL and {n} FAKE images at {size}x{size} ...")

    for i in range(n):
        face = _random_face_like(size)

        # REAL: clean face-like image with mild random brightness variation
        real_img = face.copy()
        brightness = random.uniform(0.85, 1.15)
        real_img = np.clip(real_img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        cv2.imwrite(str(real_dir / f"real_{i:05d}.jpg"), real_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # FAKE: same face with GAN-like artefacts
        fake_img = _add_deepfake_artefacts(face.copy())
        cv2.imwrite(str(fake_dir / f"fake_{i:05d}.jpg"), fake_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"  [{i + 1}/{n}] done")

    total = 2 * n
    print(f"\n[OK] Dataset created at: {out_dir}")
    print(f"   {n} REAL images -> {real_dir}")
    print(f"   {n} FAKE images -> {fake_dir}")
    print(f"   Total: {total} images")

    # Write a simple manifest
    manifest = out_dir / "manifest.txt"
    with open(manifest, "w") as f:
        f.write(f"toy_dataset\n")
        f.write(f"n_per_class={n}\n")
        f.write(f"image_size={size}\n")
        f.write(f"total={total}\n")
        f.write(f"real_dir={real_dir}\n")
        f.write(f"fake_dir={fake_dir}\n")
    print(f"   Manifest written: {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic toy dataset for pipeline smoke-testing."
    )
    parser.add_argument(
        "--n", type=int, default=60,
        help="Number of images PER class (default: 60 -> 120 total)",
    )
    parser.add_argument(
        "--size", type=int, default=299,
        help="Image size in pixels (default: 299, matches XceptionNet input)",
    )
    parser.add_argument(
        "--out", type=str, default="data/toy_dataset",
        help="Output directory for the toy dataset",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.out
    generate_dataset(n=args.n, size=args.size, out_dir=out_dir)


if __name__ == "__main__":
    main()
