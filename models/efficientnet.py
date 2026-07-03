"""
models/efficientnet.py
=======================
EfficientNet-B0 model builder for DeepFake detection.

Architecture:
- Pre-trained EfficientNet-B0 base (ImageNet weights)
- Global Average Pooling
- Dropout regularisation
- Dense classification head

References:
    Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks. ICML 2019.
"""

import logging
from typing import Tuple

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def build_efficientnet_b0(
    input_shape: Tuple[int, int, int] = (299, 299, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.5,
    weights: str = "imagenet",
    freeze_base: bool = True,
    trainable_layers: int = 20,
) -> "tf.keras.Model":  # type: ignore[name-defined]
    """
    Build and return an EfficientNet-B0 model for DeepFake detection.

    Args:
        input_shape:      Model input dimensions (H, W, C).
        num_classes:      Number of output classes.
        dropout_rate:     Dropout rate before the output layer.
        weights:          ``"imagenet"`` or ``None``.
        freeze_base:      Freeze the entire EfficientNet base.
        trainable_layers: Number of base layers to unfreeze (when not frozen).

    Returns:
        Compiled-ready Keras Model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model  # type: ignore
    from tensorflow.keras.applications import EfficientNetB0  # type: ignore

    logger.info(
        "Building EfficientNet-B0 | input=%s | classes=%d | dropout=%.2f | "
        "weights=%s | freeze_base=%s",
        input_shape, num_classes, dropout_rate, weights, freeze_base,
    )

    # ── Base model ────────────────────────────────────────────────────────────
    base_model = EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )

    if freeze_base:
        base_model.trainable = False
        logger.info("EfficientNet-B0 base frozen.")
    else:
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        trainable_count = sum(1 for l in base_model.layers if l.trainable)
        logger.info("EfficientNet-B0 partially unfrozen: %d trainable layers.", trainable_count)

    # ── Classification head ───────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_DeepFake")
    logger.info("EfficientNet-B0 built. Params: %s", f"{model.count_params():,}")
    return model
