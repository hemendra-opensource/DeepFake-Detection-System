"""
training/trainer.py
====================
Core training loop for DeepFake detection models.

Features:
- Configurable callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Training history logging
- Best model auto-save
- CPU/GPU-agnostic
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from utils.file_utils import ensure_dir, load_yaml_config
from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class Trainer:
    """
    Manages the training loop for a single DeepFake detection model.

    Args:
        model_name:    Model identifier (for naming saved weights).
        config_path:   Path to central config YAML.
        output_dir:    Directory for model checkpoints and logs.
    """

    def __init__(
        self,
        model_name: str,
        config_path: str = "configs/config.yaml",
        output_dir: str = "outputs/weights",
    ) -> None:
        self.model_name = model_name
        self.cfg = load_yaml_config(config_path)
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

        train_cfg = self.cfg.get("training", {})
        self.batch_size: int = train_cfg.get("batch_size", 16)
        self.initial_lr: float = train_cfg.get("initial_learning_rate", 1e-4)
        self.min_lr: float = train_cfg.get("min_learning_rate", 1e-6)
        self.max_epochs: int = train_cfg.get("max_epochs", 50)
        self.es_patience: int = train_cfg.get("early_stopping_patience", 10)
        self.rlr_patience: int = train_cfg.get("reduce_lr_patience", 5)
        self.rlr_factor: float = train_cfg.get("reduce_lr_factor", 0.5)

        self.history: Optional[Dict] = None
        self.best_weights_path: Optional[Path] = None

    def build_callbacks(self, phase_tag: str = "") -> list:
        """
        Build the standard callback set for a training run.

        Args:
            phase_tag: Optional suffix to distinguish multi-phase checkpoints
                       (e.g. ``"_phase1"``).

        Returns:
            List of Keras callbacks.
        """
        import tensorflow as tf

        checkpoint_path = (
            self.output_dir / f"{self.model_name}{phase_tag}_best.keras"
        )
        self.best_weights_path = checkpoint_path

        callbacks = [
            # Save the best model based on validation AUC
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            # Stop early if val_auc stops improving
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=self.es_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.rlr_factor,
                patience=self.rlr_patience,
                min_lr=self.min_lr,
                verbose=1,
            ),
            # CSV logger
            tf.keras.callbacks.CSVLogger(
                filename=str(self.output_dir / f"{self.model_name}{phase_tag}_history.csv"),
                append=True,
            ),
        ]

        logger.info(
            "Callbacks ready | checkpoint=%s | ES patience=%d | ReduceLR patience=%d",
            checkpoint_path.name, self.es_patience, self.rlr_patience,
        )
        return callbacks

    def train(
        self,
        model: "tf.keras.Model",  # type: ignore[name-defined]
        train_dataset: "tf.data.Dataset",  # type: ignore[name-defined]
        val_dataset: "tf.data.Dataset",   # type: ignore[name-defined]
        epochs: int = 20,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        phase_tag: str = "",
    ) -> Dict:
        """
        Execute the training loop.

        Args:
            model:             Compiled Keras model.
            train_dataset:     Training tf.data.Dataset.
            val_dataset:       Validation tf.data.Dataset.
            epochs:            Maximum number of epochs.
            steps_per_epoch:   Override steps per epoch (None = inferred).
            validation_steps:  Override validation steps (None = inferred).
            phase_tag:         Suffix for checkpoint naming.

        Returns:
            Training history dictionary.
        """
        callbacks = self.build_callbacks(phase_tag=phase_tag)

        logger.info(
            "Starting training | model=%s | epochs=%d | batch=%d",
            self.model_name, epochs, self.batch_size,
        )

        start_time = time.time()

        import numpy as np
        class_weights = None
        train_csv_path = Path("data/metadata/train.csv")
        if train_csv_path.is_file():
            try:
                df = pd.read_csv(train_csv_path)
                if "label_int" in df.columns:
                    labels = df["label_int"].values
                    total = len(labels)
                    fake_count = int(np.sum(labels))
                    real_count = total - fake_count
                    if fake_count > 0 and real_count > 0:
                        weight_for_0 = (1 / real_count) * (total / 2.0)
                        weight_for_1 = (1 / fake_count) * (total / 2.0)
                        class_weights = {0: float(weight_for_0), 1: float(weight_for_1)}
                        logger.info("Computed class weights: %s", class_weights)
            except Exception as exc:
                logger.warning("Could not compute class weights: %s", exc)

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        elapsed = time.time() - start_time
        logger.info(
            "Training complete in %.1f seconds. Best weights: %s",
            elapsed, self.best_weights_path,
        )

        self.history = history.history
        return self.history

    def get_history_df(self) -> Optional[pd.DataFrame]:
        """Return training history as a pandas DataFrame."""
        if self.history is None:
            return None
        return pd.DataFrame(self.history)
