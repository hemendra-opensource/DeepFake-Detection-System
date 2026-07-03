"""
training/data_loader.py
========================
Efficient tf.data input pipeline for DeepFake detection training.

Provides:
- CSV-manifest-based dataset loader
- On-the-fly image decoding and preprocessing
- Augmentation toggle (train=True / val=False)
- Prefetching and caching for CPU efficiency
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class DeepFakeDataLoader:
    """
    tf.data-based data loader for DeepFake binary classification.

    Reads split manifests (``train.csv``, ``val.csv``, ``test.csv``) produced
    by the preprocessing pipeline and returns compiled ``tf.data.Dataset``
    objects.

    Args:
        metadata_dir:  Directory containing split CSV files.
        image_size:    ``(height, width)`` for model input.
        batch_size:    Number of samples per batch.
        augment_train: Apply augmentation to training batches.
        seed:          Random seed for shuffling.
    """

    def __init__(
        self,
        metadata_dir: str = "data/metadata",
        image_size: Tuple[int, int] = (299, 299),
        batch_size: int = 16,
        augment_train: bool = True,
        seed: int = 42,
    ) -> None:
        self.metadata_dir = Path(metadata_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment_train = augment_train
        self.seed = seed

    # ── Public interface ──────────────────────────────────────────────────────

    def get_dataset(self, split: str) -> "tf.data.Dataset":  # type: ignore[name-defined]
        """
        Return a ``tf.data.Dataset`` for a given split.

        Args:
            split: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            Batched, prefetched ``tf.data.Dataset`` of ``(image, label)`` pairs.

        Raises:
            FileNotFoundError: If the split CSV does not exist.
        """
        import tensorflow as tf

        csv_path = self.metadata_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(
            "Loading '%s' split: %d samples (real=%d, fake=%d)",
            split,
            len(df),
            (df["label_int"] == 0).sum(),
            (df["label_int"] == 1).sum(),
        )

        paths = df["processed_path"].tolist()
        labels = df["label_int"].tolist()

        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, labels)
        )

        is_training = split == "train"

        if is_training:
            dataset = dataset.shuffle(
                buffer_size=min(len(paths), 5000),
                seed=self.seed,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.map(
            lambda p, l: self._load_and_preprocess(p, l, augment=is_training and self.augment_train),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.batch(self.batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_all_splits(self) -> Tuple["tf.data.Dataset", "tf.data.Dataset", "tf.data.Dataset"]:  # type: ignore[name-defined]
        """Return (train_ds, val_ds, test_ds) datasets."""
        return (
            self.get_dataset("train"),
            self.get_dataset("val"),
            self.get_dataset("test"),
        )

    def get_steps(self, split: str) -> int:
        """
        Compute the number of batches per epoch.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            Number of steps (batches).
        """
        csv_path = self.metadata_dir / f"{split}.csv"
        if not csv_path.is_file():
            return 0
        n = len(pd.read_csv(csv_path))
        return max(1, n // self.batch_size)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_and_preprocess(
        self,
        path: "tf.Tensor",   # type: ignore[name-defined]
        label: "tf.Tensor",  # type: ignore[name-defined]
        augment: bool = False,
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:  # type: ignore[name-defined]
        """
        Load a single image file and apply preprocessing.

        Args:
            path:    String tensor — path to the image file.
            label:   Integer tensor — 0 (real) or 1 (fake).
            augment: Whether to apply training augmentation.

        Returns:
            ``(image_tensor, label_tensor)`` tuple.
        """
        import tensorflow as tf

        # Read and decode
        raw = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0  # Normalise to [0, 1]

        if augment:
            image = self._augment_tensor(image)

        label = tf.cast(label, tf.float32)
        return image, label

    def _augment_tensor(self, image: "tf.Tensor") -> "tf.Tensor":  # type: ignore[name-defined]
        """Apply TensorFlow-native augmentations to a float32 image tensor."""
        import tensorflow as tf

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
