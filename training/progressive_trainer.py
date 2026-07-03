"""
training/progressive_trainer.py
================================
3-Phase progressive training strategy.

Phase 1: Train on FaceForensics++ only (frozen base)
Phase 2: Fine-tune on FF++ + Celeb-DF (partial unfreeze)
Phase 3: Fine-tune on FF++ + Celeb-DF + DFDC (full fine-tune)

This approach improves model stability and cross-dataset generalisation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from models.model_factory import ModelFactory
from training.data_loader import DeepFakeDataLoader
from training.trainer import Trainer
from utils.file_utils import ensure_dir, load_yaml_config
from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class ProgressiveTrainer:
    """
    Orchestrates the 3-phase progressive training strategy.

    Args:
        model_name:  Model to train (e.g. ``"xceptionnet"``).
        config_path: Path to central config YAML.
    """

    def __init__(
        self,
        model_name: str = "xceptionnet",
        config_path: str = "configs/config.yaml",
    ) -> None:
        self.model_name = model_name
        self.cfg = load_yaml_config(config_path)
        self.prog_cfg = self.cfg.get("training", {}).get("progressive", {})
        self.model_cfg = self.cfg.get("models", {})
        self.paths_cfg = self.cfg.get("paths", {})

        self.weights_dir = Path(self.paths_cfg.get("weights", "outputs/weights"))
        self.metadata_dir = Path(self.cfg.get("paths", {}).get("metadata", "data/metadata"))
        ensure_dir(self.weights_dir)

        self._model = None

    def run(
        self,
        phases: Optional[List[str]] = None,
        skip_if_exists: bool = True,
    ) -> Path:
        """
        Execute all training phases.

        Args:
            phases:          Which phases to run (``["phase_1", "phase_2", "phase_3"]``).
                             ``None`` runs all.
            skip_if_exists:  Skip a phase if its checkpoint already exists.

        Returns:
            Path to the final best model checkpoint.
        """
        all_phases = ["phase_1", "phase_2", "phase_3"]
        phases_to_run = phases or all_phases

        for phase_key in phases_to_run:
            phase_cfg = self.prog_cfg.get(phase_key, {})
            epochs = phase_cfg.get("epochs", 20)
            freeze_base = phase_cfg.get("freeze_base", True)

            phase_tag = f"_{phase_key}"
            checkpoint = self.weights_dir / f"{self.model_name}{phase_tag}_best.keras"

            if skip_if_exists and checkpoint.is_file():
                logger.info("Checkpoint exists, skipping %s: %s", phase_key, checkpoint)
                # Load the saved model to pass to next phase
                self._model = ModelFactory.load(checkpoint)
                continue

            logger.info("=" * 60)
            logger.info("Starting %s | model=%s | freeze=%s | epochs=%d",
                        phase_key.upper(), self.model_name, freeze_base, epochs)

            # ── Build or update model ──────────────────────────────────────
            if self._model is None:
                # Phase 1: build fresh
                self._model = ModelFactory.build(
                    self.model_name,
                    input_shape=tuple(self.model_cfg.get("input_shape", [299, 299, 3])),
                    num_classes=self.model_cfg.get("num_classes", 2),
                    dropout_rate=self.model_cfg.get("dropout_rate", 0.5),
                    weights=self.model_cfg.get("weights", "imagenet"),
                    freeze_base=freeze_base,
                    learning_rate=self.cfg["training"]["initial_learning_rate"],
                )
            else:
                # Subsequent phases: adjust trainability and recompile
                self._adjust_trainability(self._model, freeze_base)
                self._model = ModelFactory.compile(
                    self._model,
                    learning_rate=self.cfg["training"]["initial_learning_rate"],
                )

            # ── Build data loaders for this phase ─────────────────────────
            # We reuse the global metadata dir (all datasets merged during preprocessing)
            loader = DeepFakeDataLoader(
                metadata_dir=str(self.metadata_dir),
                image_size=tuple(self.model_cfg.get("input_shape", [299, 299, 3])[:2]),
                batch_size=self.cfg["training"]["batch_size"],
            )

            try:
                train_ds = loader.get_dataset("train")
                val_ds = loader.get_dataset("val")
            except FileNotFoundError as exc:
                logger.error(
                    "Cannot load dataset for %s: %s — skipping phase.", phase_key, exc
                )
                continue

            # ── Train ─────────────────────────────────────────────────────
            trainer = Trainer(
                model_name=self.model_name,
                config_path="configs/config.yaml",
                output_dir=str(self.weights_dir),
            )
            trainer.train(
                model=self._model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=epochs,
                phase_tag=phase_tag,
            )

            logger.info("%s complete.", phase_key.upper())

        # ── Save final best model ──────────────────────────────────────────
        final_path = self.weights_dir / f"{self.model_name}_final.keras"
        if self._model is not None:
            ModelFactory.save(self._model, final_path)
            logger.info("Final model saved to: %s", final_path)

        return final_path

    def _adjust_trainability(
        self,
        model: "tf.keras.Model",  # type: ignore[name-defined]
        freeze_base: bool,
        trainable_layers: int = 20,
    ) -> None:
        """Adjust base model layer trainability between phases."""
        # The base model is the first layer of the model
        base = model.layers[1] if len(model.layers) > 1 else model

        if freeze_base:
            base.trainable = False
        else:
            base.trainable = True
            for layer in base.layers[:-trainable_layers]:
                layer.trainable = False

        trainable = sum(1 for l in model.layers if l.trainable)
        logger.info("Trainability adjusted: %d trainable layers.", trainable)
