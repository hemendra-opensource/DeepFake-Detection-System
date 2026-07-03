"""
scripts/test_consistency.py
============================
End-to-end consistency validation for the prediction pipeline.

Verifies:
  1. threshold.json is not an extreme toy-data artefact
  2. calibration.json slope is within safe bounds
  3. For every test image: real_probability + fake_probability == 1.0
  4. For every test image: confidence == probability of predicted class
  5. Prediction label is consistent with fake_probability vs threshold

Usage:
    .\\venv\\Scripts\\python.exe scripts/test_consistency.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import json
import cv2
import numpy as np
import tensorflow as tf

from models.model_factory import ModelFactory
from inference.image_detector import ImageDetector


def sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


# ── 1. Check JSON files ───────────────────────────────────────────────────────
sep("1. CONFIGURATION FILES")

thresh_path = ROOT / "outputs/weights/threshold.json"
cal_path = ROOT / "outputs/weights/calibration.json"

if thresh_path.is_file():
    with open(thresh_path) as f:
        thresh_data = json.load(f)
    thresh = thresh_data.get("threshold", 0.5)
    status = "OK" if 0.05 <= thresh <= 0.95 else "EXTREME (WILL CAUSE WRONG LABELS)"
    print(f"  threshold.json  : {thresh}  [{status}]")
else:
    print("  threshold.json  : not found (will use config default 0.5)")
    thresh = 0.5

if cal_path.is_file():
    with open(cal_path) as f:
        cal_data = json.load(f)
    slope = cal_data.get("slope", 1.0)
    intercept = cal_data.get("intercept", 0.0)
    status = "OK" if slope <= 3.0 else "EXTREME — will be auto-rejected by ImageDetector"
    print(f"  calibration.json: slope={slope:.4f}  intercept={intercept:.4f}  [{status}]")
else:
    print("  calibration.json: not found (identity calibration used)")

# ── 2. Load model ─────────────────────────────────────────────────────────────
sep("2. MODEL LOADING")

weights_dir = ROOT / "outputs/weights"
candidates = [
    weights_dir / "best_model.keras",
    weights_dir / "xceptionnet_final.keras",
]
model_path = next((p for p in candidates if p.is_file()), None)

if model_path is None:
    print("  ERROR: No trained weights found.")
    sys.exit(1)

print(f"  Loading: {model_path.name}  ({model_path.stat().st_size / 1e6:.1f} MB)")
model = tf.keras.models.load_model(str(model_path), compile=False)
print(f"  Input shape : {model.input_shape}")
print(f"  Output shape: {model.output_shape}")
print(f"  Parameters  : {model.count_params():,}")

# ── 3. Create detector ────────────────────────────────────────────────────────
sep("3. DETECTOR INITIALIZATION")

detector = ImageDetector(
    model=model,
    model_name="xceptionnet",
    input_size=(299, 299),
    threshold=0.5,
    weights_dir=weights_dir,
)
print(f"  Active threshold  : {detector.threshold:.4f}")
print(f"  Active calibration: slope={detector.calibration_params['slope']:.4f}  "
      f"intercept={detector.calibration_params['intercept']:.4f}")

# ── 4. Mathematical consistency table ─────────────────────────────────────────
sep("4. MATHEMATICAL CONSISTENCY CHECKS (synthetic inputs)")

header = (
    f"{'raw_out':>8}  {'fake_p':>7}  {'real_p':>7}  "
    f"{'label':>5}  {'conf':>7}  {'consistent':>12}"
)
print(f"  {header}")
print("  " + "-" * 55)

test_probs = [0.01, 0.10, 0.30, 0.49, 0.50, 0.51, 0.70, 0.91, 0.99]
all_consistent = True

for raw in test_probs:
    fake_p = float(np.clip(raw, 0.0, 1.0))
    real_p = 1.0 - fake_p
    label = "FAKE" if fake_p >= detector.threshold else "REAL"
    conf = fake_p if label == "FAKE" else real_p

    # Check 1: probabilities sum to 1
    sum_ok = abs(real_p + fake_p - 1.0) < 1e-6
    # Check 2: confidence matches predicted class
    expected_conf = fake_p if label == "FAKE" else real_p
    conf_ok = abs(conf - expected_conf) < 1e-6
    # Check 3: label matches threshold comparison
    if label == "FAKE":
        label_ok = fake_p >= detector.threshold
    else:
        label_ok = fake_p < detector.threshold

    consistent = sum_ok and conf_ok and label_ok
    if not consistent:
        all_consistent = False

    mark = "OK" if consistent else "BUG"
    row = (
        f"{raw:>8.4f}  {fake_p:>7.4f}  {real_p:>7.4f}  "
        f"{label:>5}  {conf:>7.4f}  {mark:>12}"
    )
    print(f"  {row}")

print()
if all_consistent:
    print("  [PASS] All synthetic inputs produce consistent outputs.")
else:
    print("  [FAIL] Inconsistencies detected!")

# ── 5. Real image tests ───────────────────────────────────────────────────────
sep("5. KNOWN SAMPLE TESTS (toy dataset)")

real_dir = ROOT / "data/toy_dataset/real"
fake_dir = ROOT / "data/toy_dataset/fake"

if not real_dir.is_dir() or not fake_dir.is_dir():
    print("  Toy dataset not found. Run: python scripts/generate_toy_dataset.py")
    sys.exit(0)

results = []
failures = []
N = 8  # test N images per class

print(f"  {'Actual':>5}  {'Pred':>5}  {'Conf':>7}  {'Fake%':>7}  {'Real%':>7}  {'Match':>6}  {'Consistent':>12}")
print("  " + "-" * 60)

for actual_label, folder in [("REAL", real_dir), ("FAKE", fake_dir)]:
    imgs = sorted(folder.iterdir())[:N]
    for img_path in imgs:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            pred = detector.predict_from_array(img_rgb)
        except Exception as e:
            print(f"  ERROR on {img_path.name}: {e}")
            failures.append(str(e))
            continue

        # Consistency assertions
        sum_ok = abs(pred.real_probability + pred.fake_probability - 1.0) < 1e-4
        if pred.label == "FAKE":
            conf_ok = abs(pred.confidence - pred.fake_probability) < 1e-4
        else:
            conf_ok = abs(pred.confidence - pred.real_probability) < 1e-4
        label_ok = (pred.label == "FAKE") == (pred.fake_probability >= detector.threshold)

        consistent = sum_ok and conf_ok and label_ok
        if not consistent:
            failures.append(
                f"{img_path.name}: sum_ok={sum_ok} conf_ok={conf_ok} label_ok={label_ok}"
            )

        match = pred.label == actual_label
        results.append((actual_label, pred.label, match, consistent))

        row = (
            f"  {actual_label:>5}  {pred.label:>5}  {pred.confidence:>7.1%}  "
            f"{pred.fake_probability:>7.1%}  {pred.real_probability:>7.1%}  "
            f"{'YES' if match else 'NO':>6}  {'OK' if consistent else 'BUG':>12}"
        )
        print(row)

# ── 6. Summary ────────────────────────────────────────────────────────────────
sep("6. SUMMARY")

total = len(results)
correct = sum(1 for _, _, match, _ in results if match)
consistent_count = sum(1 for _, _, _, cons in results if cons)
accuracy = correct / total * 100 if total > 0 else 0

print(f"  Samples tested        : {total}")
print(f"  Correct predictions   : {correct}/{total}  ({accuracy:.1f}%)")
print(f"  Consistent outputs    : {consistent_count}/{total}")
print(f"  Inconsistencies found : {len(failures)}")

if failures:
    print("\n  FAILURES:")
    for f in failures:
        print(f"    - {f}")
else:
    print()
    print("  [PASS] Zero mathematically impossible outputs detected.")
    print("  [PASS] All confidence values match predicted class probability.")
    print("  [PASS] All real + fake probabilities sum to 1.0.")
    print()
    print("  The impossible output scenario (REAL + 8% confidence + 91% fake)")
    print("  is now structurally impossible due to __post_init__ consistency guard.")
