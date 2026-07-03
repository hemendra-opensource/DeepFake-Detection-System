"""
evaluation/evaluator.py
========================
Model evaluation orchestrator.

Loads each trained model, runs inference on the test split,
computes all metrics, generates a comparison table, identifies
the best model, and saves the best model path to a registry file.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.metrics import (
    EvaluationResult,
    compare_models,
    compute_metrics,
    identify_best_model,
)
from models.model_factory import ModelFactory
from training.data_loader import DeepFakeDataLoader
from utils.file_utils import ensure_dir, load_yaml_config
from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class Evaluator:
    """
    Runs evaluation for one or more trained DeepFake detection models.

    Args:
        config_path: Path to central config YAML.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.cfg = load_yaml_config(config_path)
        paths = self.cfg.get("paths", {})
        self.weights_dir = Path(paths.get("weights", "outputs/weights"))
        self.metadata_dir = Path(paths.get("metadata", "data/metadata"))
        self.outputs_dir = Path(paths.get("outputs", "outputs"))
        ensure_dir(self.outputs_dir)

        model_cfg = self.cfg.get("models", {})
        self.image_size: tuple = tuple(model_cfg.get("input_shape", [299, 299, 3])[:2])
        self.batch_size: int = self.cfg.get("training", {}).get("batch_size", 16)
        self.threshold: float = self.cfg.get("inference", {}).get(
            "confidence_threshold", 0.5
        )

    def evaluate_all(
        self,
        model_names: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate all specified models on the test split.

        Args:
            model_names: Model identifiers to evaluate. ``None`` evaluates all
                         models that have a saved ``_final.keras`` checkpoint.

        Returns:
            List of :class:`EvaluationResult` objects, one per model.
        """
        if model_names is None:
            model_names = ModelFactory.list_available()

        results: List[EvaluationResult] = []

        for name in model_names:
            checkpoint = self.weights_dir / f"{name}_final.keras"
            if not checkpoint.is_file():
                # Also look for phase checkpoints
                for phase in ["_phase_3", "_phase_2", "_phase_1"]:
                    alt = self.weights_dir / f"{name}{phase}_best.keras"
                    if alt.is_file():
                        checkpoint = alt
                        break

            if not checkpoint.is_file():
                logger.warning("No checkpoint found for '%s', skipping.", name)
                continue

            result = self.evaluate_model(name, str(checkpoint))
            if result:
                results.append(result)

        if results:
            comparison = compare_models(results)
            self._save_comparison(comparison)
            best = identify_best_model(results)

            # Find the best checkpoint path
            best_checkpoint_path = self.weights_dir / f"{best.model_name}_final.keras"
            if not best_checkpoint_path.is_file():
                for phase in ["_phase_3", "_phase_2", "_phase_1"]:
                    alt = self.weights_dir / f"{best.model_name}{phase}_best.keras"
                    if alt.is_file():
                        best_checkpoint_path = alt
                        break

            if best_checkpoint_path.is_file():
                # Save registry
                self._save_best_model_registry(best.model_name, str(best_checkpoint_path))

                # Copy to models/weights/best_model.keras
                best_weights_dir = Path("models/weights")
                best_weights_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = best_weights_dir / "best_model.keras"
                import shutil
                shutil.copy2(best_checkpoint_path, best_model_path)
                logger.info("Saved best model to %s", best_model_path)

                # Fit Platt scaling & Threshold Tuning on Validation set
                try:
                    best_model_loaded = ModelFactory.load(best_checkpoint_path, compile_model=False)
                    loader = DeepFakeDataLoader(
                        metadata_dir=str(self.metadata_dir),
                        image_size=self.image_size,
                        batch_size=self.batch_size,
                        augment_train=False,
                    )
                    val_ds = loader.get_dataset("val")

                    y_val_true_list = []
                    y_val_prob_list = []
                    for batch_images, batch_labels in val_ds:
                        preds = best_model_loaded.predict(batch_images, verbose=0)
                        y_val_true_list.extend(batch_labels.numpy().tolist())
                        y_val_prob_list.extend(preds.flatten().tolist())

                    y_val_true = np.array(y_val_true_list)
                    y_val_prob = np.array(y_val_prob_list)

                    # Fit Platt scaling (Logistic Regression on Logits)
                    epsilon = 1e-7
                    y_val_prob_clipped = np.clip(y_val_prob, epsilon, 1.0 - epsilon)
                    logits = np.log(y_val_prob_clipped / (1.0 - y_val_prob_clipped))

                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression()
                    clf.fit(logits.reshape(-1, 1), y_val_true)

                    calibration_params = {
                        "slope": float(clf.coef_[0][0]),
                        "intercept": float(clf.intercept_[0])
                    }

                    # Optimal Threshold using G-mean on ROC curve
                    from sklearn.metrics import roc_curve
                    fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
                    gmeans = np.sqrt(tpr * (1.0 - fpr))
                    ix = np.argmax(gmeans)
                    best_threshold = float(thresholds[ix]) if len(thresholds) > 0 else 0.5
                    if best_threshold < 0.05 or best_threshold > 0.95 or np.isnan(best_threshold):
                        best_threshold = 0.5

                    import json
                    Path("outputs/weights").mkdir(parents=True, exist_ok=True)
                    for out_dir in [Path("outputs/weights"), best_weights_dir]:
                        with open(out_dir / "calibration.json", "w") as f:
                            json.dump(calibration_params, f, indent=4)
                        with open(out_dir / "threshold.json", "w") as f:
                            json.dump({"threshold": best_threshold}, f, indent=4)

                    logger.info("Saved calibration params: %s", calibration_params)
                    logger.info("Saved optimal threshold: %.4f", best_threshold)
                except Exception as exc:
                    logger.warning("Platt calibration and threshold tuning failed: %s", exc)
            else:
                logger.warning("Could not find file path for best model checkpoint: %s", best.model_name)

        return results

    def evaluate_model(
        self,
        model_name: str,
        checkpoint_path: str,
    ) -> Optional[EvaluationResult]:
        """
        Evaluate a single model on the test split.

        Args:
            model_name:      Human-readable model name.
            checkpoint_path: Path to saved Keras model file.

        Returns:
            :class:`EvaluationResult` or ``None`` on failure.
        """
        logger.info("Evaluating model: %s from %s", model_name, checkpoint_path)

        try:
            model = ModelFactory.load(checkpoint_path, compile_model=False)
        except Exception as exc:
            logger.error("Failed to load model %s: %s", checkpoint_path, exc)
            return None

        try:
            loader = DeepFakeDataLoader(
                metadata_dir=str(self.metadata_dir),
                image_size=self.image_size,
                batch_size=self.batch_size,
                augment_train=False,
            )
            test_ds = loader.get_dataset("test")
        except FileNotFoundError as exc:
            logger.error("Test split not available: %s", exc)
            return None

        # Collect predictions
        y_true_list: List[float] = []
        y_prob_list: List[float] = []

        for batch_images, batch_labels in test_ds:
            preds = model.predict(batch_images, verbose=0)
            preds_flat = preds.flatten()
            y_true_list.extend(batch_labels.numpy().tolist())
            y_prob_list.extend(preds_flat.tolist())

        y_true = np.array(y_true_list)
        y_prob = np.array(y_prob_list)

        result = compute_metrics(
            model_name=model_name,
            y_true=y_true,
            y_prob=y_prob,
            threshold=self.threshold,
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_comparison(self, comparison: pd.DataFrame) -> None:
        """Save the model comparison table as CSV."""
        out_path = self.outputs_dir / "model_comparison.csv"
        comparison.to_csv(out_path, index=False)
        logger.info("Model comparison saved to: %s", out_path)
        logger.info("\n%s", comparison.to_string(index=False))

    def _save_best_model_registry(
        self, model_name: str, checkpoint_path: str
    ) -> None:
        """Write the best model name and path to a simple text registry."""
        registry_path = self.outputs_dir / "best_model.txt"
        with open(registry_path, "w", encoding="utf-8") as fh:
            fh.write(f"model_name={model_name}\n")
            fh.write(f"checkpoint={checkpoint_path}\n")
        logger.info("Best model registry saved: %s", registry_path)
