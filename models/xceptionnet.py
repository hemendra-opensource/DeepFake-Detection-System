"""
models/xceptionnet.py
======================
XceptionNet model builder for DeepFake detection.

Architecture:
- Pre-trained XceptionNet base (ImageNet weights)
- Global Average Pooling
- Dropout regularisation
- Dense classification head (sigmoid output for binary classification)

References:
    Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable
    Convolutions. CVPR 2017.
"""

import logging
from typing import Optional, Tuple

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def build_xceptionnet(
    input_shape: Tuple[int, int, int] = (299, 299, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.5,
    weights: str = "imagenet",
    freeze_base: bool = True,
    trainable_layers: int = 20,
) -> "tf.keras.Model":  # type: ignore[name-defined]
    """
    Build and return an XceptionNet model for DeepFake detection.

    Args:
        input_shape:      Model input dimensions (H, W, C).
        num_classes:      Number of output classes (2 for binary).
        dropout_rate:     Fraction of units to drop before the output layer.
        weights:          Pre-trained weights — ``"imagenet"`` or ``None``.
        freeze_base:      If ``True``, freeze the XceptionNet base entirely.
        trainable_layers: When ``freeze_base=False``, unfreeze only the last
                          *N* layers of the base model.

    Returns:
        Compiled Keras Model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model  # type: ignore
    from tensorflow.keras.applications import Xception  # type: ignore

    logger.info(
        "Building XceptionNet | input=%s | classes=%d | dropout=%.2f | "
        "weights=%s | freeze_base=%s",
        input_shape, num_classes, dropout_rate, weights, freeze_base,
    )

    # ── Base model ────────────────────────────────────────────────────────────
    base_model = Xception(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )

    # Freeze strategy
    if freeze_base:
        base_model.trainable = False
        logger.info("XceptionNet base frozen (classifier-only training).")
    else:
        # Freeze all except the last N layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        trainable_count = sum(1 for l in base_model.layers if l.trainable)
        logger.info(
            "XceptionNet base partially unfrozen: %d trainable layers.",
            trainable_count,
        )

    # ── Classification head ───────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    if num_classes == 2:
        # Binary classification — single sigmoid output
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="XceptionNet_DeepFake")

    _log_model_summary(model)
    return model


def _log_model_summary(model: "tf.keras.Model") -> None:  # type: ignore[name-defined]
    """Log the total and trainable parameter counts."""
    total_params = model.count_params()
    trainable_params = sum(
        tf_var.numpy().size
        for tf_var in model.trainable_variables
    )
    logger.info(
        "XceptionNet built — Total params: %s | Trainable: %s",
        f"{total_params:,}", f"{trainable_params:,}",
    )
