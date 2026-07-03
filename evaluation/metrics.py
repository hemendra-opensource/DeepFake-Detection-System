"""
evaluation/metrics.py
======================
Evaluation metrics for DeepFake detection models.

Computes:
- Accuracy, Precision, Recall, F1 Score
- ROC AUC
- ROC curve (fpr, tpr, thresholds)
- Confusion matrix
- Classification report (per-class)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics for a single model."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    fpr: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr: np.ndarray = field(default_factory=lambda: np.array([]))
    roc_thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    classification_report: str = ""
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_prob: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict:
        """Return scalar metrics as a dictionary (for comparison tables)."""
        return {
            "model": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "auc": round(self.auc_score, 4),
        }

    def summary(self) -> str:
        """Return a formatted summary string."""
        return (
            f"Model: {self.model_name}\n"
            f"  Accuracy : {self.accuracy:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall   : {self.recall:.4f}\n"
            f"  F1 Score : {self.f1_score:.4f}\n"
            f"  AUC      : {self.auc_score:.4f}\n"
        )


def compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationResult:
    """
    Compute all evaluation metrics from ground-truth labels and predicted probabilities.

    Args:
        model_name: Human-readable model name.
        y_true:     Ground truth binary labels (0 = real, 1 = fake).
        y_prob:     Predicted probabilities for the fake class (sigmoid output).
        threshold:  Decision threshold for converting probabilities to labels.

    Returns:
        Populated :class:`EvaluationResult` instance.
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc_score = roc_auc_score(y_true, y_prob)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    except ValueError as exc:
        logger.warning("ROC AUC computation failed: %s", exc)
        auc_score = 0.0
        fpr, tpr, thresholds = np.array([0, 1]), np.array([0, 1]), np.array([0.5])

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=["Real", "Fake"], zero_division=0
    )

    result = EvaluationResult(
        model_name=model_name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        auc_score=auc_score,
        fpr=fpr,
        tpr=tpr,
        roc_thresholds=thresholds,
        confusion_matrix=cm,
        classification_report=report,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    logger.info(result.summary())
    return result


def compare_models(results: List[EvaluationResult]) -> pd.DataFrame:
    """
    Generate a comparison table for multiple model results.

    Args:
        results: List of :class:`EvaluationResult` objects.

    Returns:
        DataFrame sorted by AUC (descending).
    """
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def identify_best_model(results: List[EvaluationResult]) -> EvaluationResult:
    """
    Identify the best-performing model by AUC score.

    Args:
        results: List of evaluation results.

    Returns:
        The :class:`EvaluationResult` with the highest AUC.
    """
    best = max(results, key=lambda r: r.auc_score)
    logger.info("Best model: %s (AUC=%.4f)", best.model_name, best.auc_score)
    return best
