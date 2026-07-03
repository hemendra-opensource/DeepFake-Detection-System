"""
scripts/quick_train.py
=======================
Train a DeepFake detector from scratch on a flat folder structure:

    <data_dir>/
        real/  (*.jpg / *.png)
        fake/  (*.jpg / *.png)

This script bypasses the full preprocessing pipeline and is intended for:
  1. Smoke-testing the full training → evaluation → inference pipeline
  2. Training on the synthetic toy dataset

Usage:
    # Train on the toy dataset (generate it first):
    python scripts/generate_toy_dataset.py --n 80
    python scripts/quick_train.py --data data/toy_dataset --epochs 10

    # Train on a real dataset folder:
    python scripts/quick_train.py --data data/ff_plus_plus_processed --epochs 30

Output:
    outputs/weights/best_model.keras        ← best val-loss checkpoint
    outputs/weights/calibration.json        ← Platt scaling parameters
    outputs/weights/threshold.json          ← G-mean optimised threshold

"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_paths(data_dir: Path):
    """
    Walk data_dir/real and data_dir/fake; returns (paths, labels).
    labels: 0 = REAL, 1 = FAKE
    """
    paths, labels = [], []
    for label_name, label_int in [("real", 0), ("fake", 1)]:
        class_dir = data_dir / label_name
        if not class_dir.is_dir():
            print(f"  [WARN] Directory not found: {class_dir}")
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in VALID_EXTS:
                paths.append(str(p))
                labels.append(label_int)
    return paths, labels


def _make_tf_dataset(paths, labels, image_size, batch_size, augment: bool = False):
    """Build a tf.data.Dataset from file paths + integer labels."""
    import tensorflow as tf

    def _load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, image_size)  # bilinear by default

        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.85, 1.15)
            img = tf.image.random_saturation(img, 0.85, 1.15)
            img = tf.image.random_hue(img, 0.05)
            img = tf.clip_by_value(img, 0.0, 1.0)

        return img, tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_calibration(model, val_ds, weights_dir: Path):
    """
    Fit Platt scaling on validation logits and find G-mean optimal threshold.
    Saves calibration.json and threshold.json to weights_dir.
    """
    import math
    import numpy as np

    print("\n[INFO] Computing Platt scaling calibration ...")

    raw_probs, true_labels = [], []
    for batch_imgs, batch_labels in val_ds:
        preds = model.predict(batch_imgs, verbose=0).flatten()
        raw_probs.extend(preds.tolist())
        true_labels.extend(batch_labels.numpy().tolist())

    raw_probs = np.array(raw_probs, dtype=np.float64)
    true_labels = np.array(true_labels, dtype=np.float64)

    # Convert to logits for Platt scaling
    eps = 1e-7
    logits = np.log(
        np.clip(raw_probs, eps, 1 - eps) / np.clip(1 - raw_probs, eps, 1 - eps)
    )

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_curve

        lr = LogisticRegression(C=1e5, solver="lbfgs", max_iter=200)
        lr.fit(logits.reshape(-1, 1), true_labels)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])

        calib = {"slope": slope, "intercept": intercept}
        calib_path = weights_dir / "calibration.json"
        with open(calib_path, "w") as f:
            json.dump(calib, f, indent=2)
        print(f"  Calibration saved: slope={slope:.4f}  intercept={intercept:.4f}")

        # Apply calibration and compute calibrated probs
        cal_logits = slope * logits + intercept
        cal_probs = 1.0 / (1.0 + np.exp(-cal_logits))

        # G-mean threshold selection from ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, cal_probs)
        gmean = np.sqrt(tpr * (1.0 - fpr))
        best_idx = np.argmax(gmean)
        best_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
        if best_thresh < 0.05 or best_thresh > 0.95 or np.isnan(best_thresh):
            best_thresh = 0.5

        thresh_path = weights_dir / "threshold.json"
        with open(thresh_path, "w") as f:
            json.dump({"threshold": best_thresh}, f, indent=2)
        print(f"  G-mean threshold saved: {best_thresh:.4f}")

    except ImportError:
        print("  [WARN] scikit-learn not installed. Skipping calibration.")
        print("         Install with: pip install scikit-learn")
        calib = {"slope": 1.0, "intercept": 0.0}
        calib_path = weights_dir / "calibration.json"
        with open(calib_path, "w") as f:
            json.dump(calib, f, indent=2)
        thresh_path = weights_dir / "threshold.json"
        with open(thresh_path, "w") as f:
            json.dump({"threshold": 0.5}, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick training script on a real/fake folder structure."
    )
    parser.add_argument(
        "--data", type=str, default="data/toy_dataset",
        help="Directory containing 'real/' and 'fake/' sub-dirs.",
    )
    parser.add_argument(
        "--model", type=str, default="xceptionnet",
        choices=["xceptionnet", "efficientnet_b0", "resnet50"],
        help="Model architecture (default: xceptionnet).",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (default: 8 for CPU).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Fraction of data for validation (default: 0.2).",
    )
    parser.add_argument(
        "--image-size", type=int, default=299,
        help="Input image size (default: 299 for XceptionNet).",
    )
    parser.add_argument(
        "--no-imagenet", action="store_true",
        help="Initialize without ImageNet weights (random init).",
    )
    args = parser.parse_args()

    import random
    import tensorflow as tf

    # Deterministic seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    data_dir = ROOT / args.data
    if not data_dir.is_dir():
        print(f"[ERROR] Data directory not found: {data_dir}")
        print("  Run: python scripts/generate_toy_dataset.py --n 80")
        sys.exit(1)

    weights_dir = ROOT / "outputs" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    image_size = (args.image_size, args.image_size)

    # ── Collect paths ─────────────────────────────────────────────────────────
    all_paths, all_labels = _collect_paths(data_dir)
    if not all_paths:
        print("[ERROR] No images found under real/ or fake/ sub-directories.")
        sys.exit(1)

    n_total = len(all_paths)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    # Stratified shuffle split
    indices = list(range(n_total))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_paths = [all_paths[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    n_real_train = sum(1 for l in train_labels if l == 0)
    n_fake_train = sum(1 for l in train_labels if l == 1)
    n_real_val = sum(1 for l in val_labels if l == 0)
    n_fake_val = sum(1 for l in val_labels if l == 1)

    print(f"\n[INFO] Dataset summary:")
    print(f"  Total images  : {n_total}")
    print(f"  Training set  : {n_train}  (REAL={n_real_train}, FAKE={n_fake_train})")
    print(f"  Validation set: {n_val}   (REAL={n_real_val}, FAKE={n_fake_val})")

    train_ds = _make_tf_dataset(train_paths, train_labels, image_size, args.batch_size, augment=True)
    val_ds = _make_tf_dataset(val_paths, val_labels, image_size, args.batch_size, augment=False)

    # ── Build model ───────────────────────────────────────────────────────────
    from models.model_factory import ModelFactory

    init_weights = None if args.no_imagenet else "imagenet"
    print(f"\n[INFO] Building {args.model} (weights={init_weights}) ...")

    input_shape = (args.image_size, args.image_size, 3)
    model = ModelFactory.build(
        args.model,
        input_shape=input_shape,
        num_classes=2,
        dropout_rate=0.5,
        weights=init_weights,
        freeze_base=True,
        learning_rate=args.lr,
    )
    print(f"  Parameters: {model.count_params():,}")

    # ── Class weights (handles imbalance) ────────────────────────────────────
    n_total_train = n_train
    class_weights = None
    if n_total_train > 0 and n_real_train > 0 and n_fake_train > 0:
        w_real = n_total_train / (2.0 * n_real_train)
        w_fake = n_total_train / (2.0 * n_fake_train)
        class_weights = {0: w_real, 1: w_fake}
        print(f"  Class weights: REAL={w_real:.3f}, FAKE={w_fake:.3f}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_model_path = weights_dir / "best_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Training for up to {args.epochs} epochs ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    print(f"\n[OK] Training complete. Best model saved to: {best_model_path}")

    # ── Final validation metrics ──────────────────────────────────────────────
    val_results = model.evaluate(val_ds, verbose=0)
    metric_names = model.metrics_names
    print("\n[INFO] Validation metrics (best weights):")
    for name, val in zip(metric_names, val_results):
        print(f"  {name}: {val:.4f}")

    # ── Calibration ───────────────────────────────────────────────────────────
    _compute_calibration(model, val_ds, weights_dir)

    print("\n" + "=" * 58)
    print("  QUICK TRAINING COMPLETE")
    print("=" * 58)
    print(f"  Model    : {best_model_path}")
    print(f"  Calibration: {weights_dir / 'calibration.json'}")
    print(f"  Threshold  : {weights_dir / 'threshold.json'}")
    print("\n  Next steps:")
    print("    python scripts/verify_performance.py --data", args.data)
    print("    streamlit run app.py")
    print("=" * 58)


if __name__ == "__main__":
    main()
