"""
preprocessing/pipeline.py
==========================
Orchestrates the full preprocessing pipeline.

Steps:
1. Validate each dataset directory
2. Remove invalid and duplicate files
3. Extract & crop faces from every valid file
4. Apply augmentation to training-split images
5. Generate ``data/metadata/metadata.csv``
6. Generate train / validation / test split manifests

Usage::

    from preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline(config_path="configs/config.yaml")
    pipeline.run(datasets=["celeb_df"])
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocessing.augmentor import Augmentor
from preprocessing.dataset_validator import DatasetValidator, FileRecord
from preprocessing.face_extractor import FaceExtractor
from utils.file_utils import ensure_dir, load_yaml_config, safe_copy
from utils.image_utils import save_image, load_image_rgb, preprocess_for_model
from utils.logger import get_logger
from utils.video_utils import extract_frames

logger: logging.Logger = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class PreprocessingPipeline:
    """
    End-to-end preprocessing orchestrator.

    Args:
        config_path: Path to ``configs/config.yaml``.
        dry_run:     If ``True``, validate and report without writing files.
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        dry_run: bool = False,
    ) -> None:
        self.cfg = load_yaml_config(config_path)
        self.dry_run = dry_run

        # Preprocessing config
        pre_cfg = self.cfg.get("preprocessing", {})
        self.image_size: Tuple[int, int] = tuple(pre_cfg.get("image_size", [299, 299]))  # type: ignore
        self.face_margin: float = pre_cfg.get("face_margin", 0.3)
        self.min_face_conf: float = pre_cfg.get("min_face_confidence", 0.7)
        self.seed: int = pre_cfg.get("random_seed", 42)
        split_ratios = pre_cfg.get("split_ratios", {})
        self.train_ratio: float = split_ratios.get("train", 0.70)
        self.val_ratio: float = split_ratios.get("val", 0.15)
        self.test_ratio: float = split_ratios.get("test", 0.15)

        aug_cfg = self.cfg.get("augmentation", {})
        self.augmentor = Augmentor(
            horizontal_flip=aug_cfg.get("horizontal_flip", True),
            brightness_limit=aug_cfg.get("brightness_limit", 0.2),
            contrast_limit=aug_cfg.get("contrast_limit", 0.2),
            rotation_limit=aug_cfg.get("rotation_limit", 10),
            noise_probability=aug_cfg.get("noise_probability", 0.3),
            blur_probability=aug_cfg.get("blur_probability", 0.2),
            seed=self.seed,
        )

        self.face_extractor = FaceExtractor(
            target_size=self.image_size,
            margin=self.face_margin,
            min_confidence=self.min_face_conf,
            max_faces=1,
        )

        # Paths
        paths = self.cfg.get("paths", {})
        self.processed_dir = Path(paths.get("data_processed", "data/processed"))
        self.metadata_dir = Path(paths.get("metadata", "data/metadata"))

        if not dry_run:
            ensure_dir(self.processed_dir)
            ensure_dir(self.metadata_dir)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, datasets: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute the full preprocessing pipeline.

        Args:
            datasets: List of dataset keys to process (e.g. ``["celeb_df"]``).
                      ``None`` means process all configured datasets.

        Returns:
            Merged metadata DataFrame with all processed samples.
        """
        dataset_cfgs = self.cfg.get("dataset", {})
        if datasets is None:
            datasets = list(dataset_cfgs.keys())

        all_records: List[Dict] = []

        for dataset_key in datasets:
            ds_cfg = dataset_cfgs.get(dataset_key)
            if ds_cfg is None:
                logger.warning("Dataset key not found in config: %s", dataset_key)
                continue

            ds_name = ds_cfg.get("name", dataset_key)
            raw_dir = Path(ds_cfg.get("raw_dir", f"data/raw/{dataset_key}"))

            if not raw_dir.is_dir():
                logger.warning(
                    "Raw directory for '%s' not found: %s — skipping.",
                    ds_name, raw_dir,
                )
                continue

            logger.info("=" * 60)
            logger.info("Processing dataset: %s", ds_name)
            logger.info("Raw directory: %s", raw_dir)

            records = self._process_dataset(dataset_key, ds_name, raw_dir)
            all_records.extend(records)

        if not all_records:
            logger.warning("No records processed. Check dataset paths.")
            return pd.DataFrame()

        metadata = pd.DataFrame(all_records)
        metadata = self._generate_splits(metadata)
        self._save_metadata(metadata)
        self._save_split_manifests(metadata)

        logger.info(
            "Pipeline complete. Total samples: %d (train=%d, val=%d, test=%d)",
            len(metadata),
            (metadata["split"] == "train").sum(),
            (metadata["split"] == "val").sum(),
            (metadata["split"] == "test").sum(),
        )
        return metadata

    # ── Per-dataset processing ────────────────────────────────────────────────

    def _process_dataset(
        self,
        dataset_key: str,
        dataset_name: str,
        raw_dir: Path,
    ) -> List[Dict]:
        """
        Process a single dataset directory.

        Args:
            dataset_key:  Config key (e.g. ``"celeb_df"``).
            dataset_name: Human-readable name.
            raw_dir:      Root of the raw dataset.

        Returns:
            List of record dicts for the metadata CSV.
        """
        # Step 1: Validate
        validator = DatasetValidator(dataset_name=dataset_name, compute_hashes=True)
        report = validator.validate_directory(raw_dir)
        logger.info(report.summary())

        # Step 2: Filter valid, non-duplicate files
        valid_records: List[FileRecord] = [
            r for r in report.records if r.is_valid and not r.is_duplicate
        ]
        logger.info("Using %d valid non-duplicate files.", len(valid_records))

        if not valid_records:
            return []

        # Step 3: Face extraction & saving
        processed: List[Dict] = []
        output_dir = self.processed_dir / dataset_key

        for record in tqdm(valid_records, desc=f"  Extracting faces ({dataset_name})"):
            result = self._process_file(record, output_dir, dataset_key)
            if result:
                processed.append(result)

        logger.info(
            "Dataset '%s': %d / %d files successfully processed.",
            dataset_name, len(processed), len(valid_records),
        )
        return processed

    def _process_file(
        self,
        record: FileRecord,
        output_dir: Path,
        dataset_key: str,
    ) -> Optional[Dict]:
        """
        Process a single image or video file.

        For images: detect face → crop → save.
        For videos: sample frames → detect face per frame → save.

        Returns:
            Metadata dict or ``None`` if no face was found.
        """
        src_path = Path(record.path)
        label_int = 0 if record.label == "real" else 1

        if record.file_type == "image":
            return self._process_image_file(
                src_path, label_int, record.label, output_dir, dataset_key
            )
        else:
            return self._process_video_file(
                src_path, label_int, record.label, output_dir, dataset_key
            )

    def _process_image_file(
        self,
        path: Path,
        label_int: int,
        label_str: str,
        output_dir: Path,
        dataset_key: str,
    ) -> Optional[Dict]:
        """Process a single image: load → face extract → save."""
        img = load_image_rgb(path)
        if img is None:
            return None

        detection = self.face_extractor.extract_largest(img)
        face_img = detection.cropped_face if detection else cv2.resize(
            img, self.image_size, interpolation=cv2.INTER_AREA
        )

        save_path = output_dir / label_str / f"{path.stem}.jpg"

        if not self.dry_run:
            ensure_dir(save_path.parent)
            save_image(face_img, save_path)

        return {
            "original_path": str(path),
            "processed_path": str(save_path),
            "label": label_str,
            "label_int": label_int,
            "dataset": dataset_key,
            "file_type": "image",
            "face_detected": detection is not None,
            "face_confidence": round(detection.confidence, 4) if detection else 0.0,
        }

    def _process_video_file(
        self,
        path: Path,
        label_int: int,
        label_str: str,
        output_dir: Path,
        dataset_key: str,
    ) -> Optional[Dict]:
        """
        Process a video: sample frames → face extract per frame → save.

        Returns a single record pointing to the first successful frame.
        """
        sample_rate = self.cfg.get("inference", {}).get("video_frame_sample_rate", 5)
        max_frames = 30  # Limit frames extracted during preprocessing

        frames = extract_frames(path, sample_rate=sample_rate, max_frames=max_frames)
        if not frames:
            logger.debug("No frames extracted from: %s", path)
            return None

        saved_count = 0
        first_save_path: Optional[Path] = None
        first_conf: float = 0.0

        for frame_result in frames:
            detection = self.face_extractor.extract_largest(frame_result.image)
            face_img = (
                detection.cropped_face
                if detection
                else cv2.resize(frame_result.image, self.image_size)
            )

            frame_name = f"{path.stem}_frame{frame_result.frame_index:05d}.jpg"
            save_path = output_dir / label_str / frame_name

            if not self.dry_run:
                ensure_dir(save_path.parent)
                save_image(face_img, save_path)
                saved_count += 1

            if first_save_path is None:
                first_save_path = save_path
                first_conf = detection.confidence if detection else 0.0

        if saved_count == 0:
            return None

        return {
            "original_path": str(path),
            "processed_path": str(first_save_path),
            "label": label_str,
            "label_int": label_int,
            "dataset": dataset_key,
            "file_type": "video",
            "face_detected": True,
            "face_confidence": round(first_conf, 4),
            "frames_saved": saved_count,
        }

    # ── Splitting ──────────────────────────────────────────────────────────────

    def _generate_splits(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Assign train / val / test split labels to each record.

        Uses stratified splitting to preserve label balance.

        Args:
            metadata: Full metadata DataFrame.

        Returns:
            DataFrame with a new ``split`` column.
        """
        metadata = metadata.copy()
        metadata["split"] = "train"

        val_test_ratio = self.val_ratio + self.test_ratio

        try:
            train_idx, val_test_idx = train_test_split(
                metadata.index,
                test_size=val_test_ratio,
                stratify=metadata["label_int"],
                random_state=self.seed,
            )
            val_ratio_adj = self.val_ratio / val_test_ratio
            val_idx, test_idx = train_test_split(
                val_test_idx,
                test_size=1 - val_ratio_adj,
                stratify=metadata.loc[val_test_idx, "label_int"],
                random_state=self.seed,
            )
        except ValueError as exc:
            logger.warning("Stratified split failed (%s); using random split.", exc)
            indices = list(metadata.index)
            np.random.default_rng(self.seed).shuffle(indices)
            n = len(indices)
            n_val = int(n * self.val_ratio)
            n_test = int(n * self.test_ratio)
            val_idx = indices[:n_val]
            test_idx = indices[n_val : n_val + n_test]
            train_idx = indices[n_val + n_test :]

        metadata.loc[val_idx, "split"] = "val"
        metadata.loc[test_idx, "split"] = "test"

        return metadata

    # ── Saving ────────────────────────────────────────────────────────────────

    def _save_metadata(self, metadata: pd.DataFrame) -> None:
        """Save the full metadata DataFrame to CSV."""
        out_path = self.metadata_dir / "metadata.csv"
        metadata.to_csv(out_path, index=False)
        logger.info("Metadata saved to: %s", out_path)

    def _save_split_manifests(self, metadata: pd.DataFrame) -> None:
        """Save individual CSV files for each split."""
        for split in ["train", "val", "test"]:
            split_df = metadata[metadata["split"] == split]
            out_path = self.metadata_dir / f"{split}.csv"
            split_df.to_csv(out_path, index=False)
            logger.info("  %s split: %d samples → %s", split, len(split_df), out_path)
