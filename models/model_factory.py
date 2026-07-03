"""
models/model_factory.py
========================
Factory pattern for building and loading DeepFake detection models.

Usage::

    from models.model_factory import ModelFactory

    # Build a new model
    model = ModelFactory.build("xceptionnet", freeze_base=True)

    # Load saved weights
    model = ModelFactory.load("outputs/weights/best_model.keras")

    # Save a model
    ModelFactory.save(model, "outputs/weights/xceptionnet_phase1.keras")
"""

import logging
from pathlib import Path
from typing import Literal, Optional

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

ModelName = Literal["xceptionnet", "efficientnet_b0", "resnet50"]

# Map model names to their builder functions (lazy import to avoid TF startup cost)
_MODEL_REGISTRY: dict[str, str] = {
    "xceptionnet":    "models.xceptionnet:build_xceptionnet",
    "efficientnet_b0": "models.efficientnet:build_efficientnet_b0",
    "resnet50":       "models.resnet50:build_resnet50",
}

# Grad-CAM target layer names per model
GRADCAM_LAYERS: dict[str, str] = {
    "xceptionnet":    "block14_sepconv2_act",
    "efficientnet_b0": "top_activation",
    "resnet50":       "conv5_block3_out",
}


class ModelFactory:
    """Factory for creating, saving, and loading Keras DeepFake models."""

    @staticmethod
    def build(
        model_name: str,
        input_shape: tuple[int, int, int] = (299, 299, 3),
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        weights: str = "imagenet",
        freeze_base: bool = True,
        trainable_layers: int = 20,
        learning_rate: float = 1e-4,
    ) -> "tf.keras.Model":  # type: ignore[name-defined]
        """
        Build and compile a DeepFake detection model.

        Args:
            model_name:       One of ``"xceptionnet"``, ``"efficientnet_b0"``,
                              ``"resnet50"``.
            input_shape:      Input tensor shape (H, W, C).
            num_classes:      Number of output classes.
            dropout_rate:     Dropout probability.
            weights:          Pre-training weights (``"imagenet"`` or ``None``).
            freeze_base:      Freeze the convolutional base.
            trainable_layers: Layers to unfreeze when not fully frozen.
            learning_rate:    Initial Adam learning rate.

        Returns:
            Compiled Keras Model.

        Raises:
            ValueError: If ``model_name`` is not in the registry.
        """
        model_name_lower = model_name.lower().replace("-", "_")

        if model_name_lower not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(_MODEL_REGISTRY.keys())}"
            )

        # Lazy-import the builder function
        module_path, func_name = _MODEL_REGISTRY[model_name_lower].rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        builder_fn = getattr(module, func_name)

        model = builder_fn(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            weights=weights,
            freeze_base=freeze_base,
            trainable_layers=trainable_layers,
        )

        model = ModelFactory.compile(model, learning_rate=learning_rate)
        return model

    @staticmethod
    def compile(
        model: "tf.keras.Model",  # type: ignore[name-defined]
        learning_rate: float = 1e-4,
    ) -> "tf.keras.Model":  # type: ignore[name-defined]
        """
        Compile a model with standard settings for binary DeepFake detection.

        Args:
            model:         Keras model to compile.
            learning_rate: Adam learning rate.

        Returns:
            The compiled model (in-place).
        """
        import tensorflow as tf

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        logger.info("Model compiled with lr=%.6f", learning_rate)
        return model

    @staticmethod
    def save(
        model: "tf.keras.Model",  # type: ignore[name-defined]
        path: str | Path,
    ) -> None:
        """
        Save a Keras model to disk.

        Args:
            model: Trained Keras model.
            path:  Destination path (use ``.keras`` or ``.h5`` extension).
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        logger.info("Model saved to: %s", save_path)

    @staticmethod
    def load(
        path: str | Path,
        compile_model: bool = True,
        learning_rate: float = 1e-4,
    ) -> "tf.keras.Model":  # type: ignore[name-defined]
        """
        Load a saved Keras model from disk.

        Args:
            path:          Path to the saved model file.
            compile_model: Whether to recompile after loading.
            learning_rate: Learning rate if recompiling.

        Returns:
            Loaded Keras Model.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        import tensorflow as tf

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        logger.info("Loading model from: %s", model_path)
        model = tf.keras.models.load_model(str(model_path))

        if compile_model:
            model = ModelFactory.compile(model, learning_rate=learning_rate)

        logger.info("Model loaded successfully.")
        return model

    @staticmethod
    def get_gradcam_layer(model_name: str) -> str:
        """
        Return the recommended Grad-CAM target layer name for a model.

        Args:
            model_name: Model identifier string.

        Returns:
            Keras layer name string.
        """
        name = model_name.lower().replace("-", "_")
        layer = GRADCAM_LAYERS.get(name, "")
        if not layer:
            logger.warning("No Grad-CAM layer registered for model: %s", model_name)
        return layer

    @staticmethod
    def list_available() -> list[str]:
        """Return a list of all registered model names."""
        return list(_MODEL_REGISTRY.keys())
