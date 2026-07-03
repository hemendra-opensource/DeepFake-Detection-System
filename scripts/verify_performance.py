"""
scripts/verify_performance.py
==============================
End-to-end performance verification script.

Runs the trained model on a held-out set (or the toy dataset) and reports:
  - Accuracy, Precision, Recall, F1, AUC-ROC
  - Confusion Matrix
  - False Positive Rate (FPR) and False Negative Rate (FNR)
  - Mean inference time per sample
  - Calibration quality (Expected Calibration Error)

Usage:
    # Verify on toy dataset (default):
    python scripts/verify_performance.py

    # Verify on a custom directory (expects real/ and fake/ sub-dirs):
    python scripts/verify_performance.py --data data/my_val_set

    # Specify a model checkpoint explicitly:
    python scripts/verify_performance.py --weights outputs/weights/best_model.keras

    # Override the decision threshold:
    python scripts/verify_performance.py --threshold 0.45
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    import math
    eps = 1e-7
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def _apply_platt(raw_prob: float, calib: dict) -> float:
    slope = calib.get("slope", 1.0)
    intercept = calib.get("intercept", 0.0)
    return _sigmoid(slope * _logit(raw_prob) + intercept)


def _collect_samples(data_dir: Path, limit: int = 500):
    """
    Walk data_dir expecting sub-directories named 'real' and 'fake'.
    Returns a list of (image_path, label_int) tuples.
      label_int: 0 = REAL, 1 = FAKE
    """
    samples = []
    for label_name, label_int in [("real", 0), ("fake", 1)]:
        class_dir = data_dir / label_name
        if not class_dir.is_dir():
            print(f"  [WARN] Missing directory: {class_dir}")
            continue
        imgs = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        # Balance: take at most limit//2 per class
        imgs = imgs[: limit // 2]
        for p in imgs:
            samples.append((p, label_int))
    return samples


def _preprocess(img_bgr: np.ndarray, size: tuple) -> np.ndarray:
    """Resize and normalize to float32 [0, 1] matching training pipeline."""
    resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, H, W, 3)


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (lower is better)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / len(probs))


def _print_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float,
    times_ms: list,
) -> None:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_prob)
    ece = _ece(y_pred_prob, y_true.astype(float))
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    mean_ms = float(np.mean(times_ms)) if times_ms else 0.0

    print("\n" + "=" * 58)
    print("  PERFORMANCE VERIFICATION REPORT")
    print("=" * 58)
    print(f"  Threshold used     : {threshold:.4f}")
    print(f"  Samples evaluated  : {len(y_true)}")
    print("-" * 58)
    print(f"  Accuracy           : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision (FAKE)   : {prec:.4f}")
    print(f"  Recall    (FAKE)   : {rec:.4f}")
    print(f"  F1-Score  (FAKE)   : {f1:.4f}")
    print(f"  AUC-ROC            : {auc:.4f}")
    print(f"  ECE (calibration)  : {ece:.4f}  (lower = better)")
    print("-" * 58)
    print(f"  False Positive Rate: {fpr:.4f}  ({fpr * 100:.2f}%)")
    print(f"  False Negative Rate: {fnr:.4f}  ({fnr * 100:.2f}%)")
    print(f"  False Positives    : {fp}")
    print(f"  False Negatives    : {fn}")
    print("-" * 58)
    print(f"  Confusion Matrix   :")
    print(f"      Predicted REAL  Predicted FAKE")
    print(f"  True REAL   {tn:6d}          {fp:6d}")
    print(f"  True FAKE   {fn:6d}          {tp:6d}")
    print("-" * 58)
    print(f"  Mean inference time: {mean_ms:.1f} ms / image")
    print("=" * 58)

    # Interpretation aid
    if auc < 0.55:
        print(
            "\n  [WARN] AUC-ROC < 0.55: model is near-random.\n"
            "  This is EXPECTED for a toy dataset. Train on real data for\n"
            "  meaningful AUC."
        )
    elif auc >= 0.80:
        print(f"\n  [OK] AUC-ROC = {auc:.4f} — good discriminative power.")
    else:
        print(f"\n  [INFO] AUC-ROC = {auc:.4f} — moderate; more training data may help.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify model performance on a labeled image directory."
    )
    parser.add_argument(
        "--data", type=str, default="data/toy_dataset",
        help="Directory with 'real/' and 'fake/' sub-directories.",
    )
    parser.add_argument(
        "--weights", type=str, default="",
        help="Path to model weights (.keras). Searches outputs/weights/ if empty.",
    )
    parser.add_argument(
        "--threshold", type=float, default=-1.0,
        help="Decision threshold. Uses saved threshold.json or 0.5 if -1.",
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Max total images to evaluate (balanced across classes).",
    )
    args = parser.parse_args()

    data_dir = ROOT / args.data
    if not data_dir.is_dir():
        print(f"[ERROR] Data directory not found: {data_dir}")
        print("  Run: python scripts/generate_toy_dataset.py --n 80")
        sys.exit(1)

    print(f"\n[INFO] Loading model ...")
    import tensorflow as tf  # noqa: E402

    weights_dir = ROOT / "outputs" / "weights"

    # ── Find model weights ────────────────────────────────────────────────────
    if args.weights:
        model_path = Path(args.weights)
    else:
        candidates = [
            weights_dir / "best_model.keras",
            weights_dir / "xceptionnet_final.keras",
            weights_dir / "xceptionnet_phase_3_best.keras",
            weights_dir / "xceptionnet_phase_2_best.keras",
            weights_dir / "xceptionnet_phase_1_best.keras",
        ]
        model_path = next((p for p in candidates if p.is_file()), None)

    if model_path is None or not Path(model_path).is_file():
        print(
            "\n[WARN] No trained weights found.  Running with randomly-initialised weights.\n"
            "  Metrics will be meaningless (near-chance).  Train first:\n"
            "    python train.py --train --data data/toy_dataset\n"
        )
        from models.model_factory import ModelFactory
        model = ModelFactory.build("xceptionnet", input_shape=(299, 299, 3), weights=None)
    else:
        print(f"  Loading: {model_path}")
        model = tf.keras.models.load_model(str(model_path), compile=False)
        print("  Model loaded OK.")

    # ── Load calibration / threshold ──────────────────────────────────────────
    calib_path = weights_dir / "calibration.json"
    thresh_path = weights_dir / "threshold.json"

    calibration = None
    if calib_path.is_file():
        with open(calib_path) as f:
            calibration = json.load(f)
        print(f"  Calibration loaded: slope={calibration.get('slope'):.4f}  intercept={calibration.get('intercept'):.4f}")

    threshold = args.threshold
    if threshold < 0:
        if thresh_path.is_file():
            with open(thresh_path) as f:
                threshold = float(json.load(f).get("threshold", 0.5))
            print(f"  Threshold loaded from file: {threshold:.4f}")
        else:
            threshold = 0.5
            print(f"  Using default threshold: {threshold}")

    # ── Input shape ───────────────────────────────────────────────────────────
    input_shape = model.input_shape  # (None, H, W, C)
    h, w = input_shape[1], input_shape[2]
    input_size = (w, h)   # cv2.resize takes (W, H)
    print(f"  Model input size: {h}x{w}")

    # ── Collect samples ───────────────────────────────────────────────────────
    samples = _collect_samples(data_dir, limit=args.limit)
    if not samples:
        print("[ERROR] No images found. Check --data path and real/fake subdirs.")
        sys.exit(1)

    print(f"\n[INFO] Evaluating {len(samples)} samples ...")
    y_true = []
    y_pred_prob = []
    times_ms = []

    for idx, (img_path, label) in enumerate(samples):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Skipping unreadable: {img_path}")
            continue

        t0 = time.perf_counter()
        tensor = _preprocess(img, input_size)
        raw = float(model.predict(tensor, verbose=0)[0][0])
        t1 = time.perf_counter()

        # Apply Platt calibration if available
        if calibration is not None:
            prob = _apply_platt(raw, calibration)
        else:
            prob = raw

        y_true.append(label)
        y_pred_prob.append(prob)
        times_ms.append((t1 - t0) * 1000)

        if (idx + 1) % 50 == 0:
            print(f"  ... {idx + 1}/{len(samples)}")

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    try:
        _print_metrics(y_true, y_pred_prob, threshold, times_ms)
    except ImportError:
        print("\n[WARN] scikit-learn not installed. Install with: pip install scikit-learn")
        # Fallback: simple accuracy
        y_pred = (y_pred_prob >= threshold).astype(int)
        acc = (y_pred == y_true).mean()
        print(f"\n  Simple Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
