"""
preprocessing/dataset_validator.py
====================================
Dataset validation and quality assurance.

Responsibilities:
- Check image and video files are readable / not corrupted
- Detect and report duplicate files via MD5 hashing
- Validate directory structure and label consistency
- Generate a per-file validation report as a DataFrame
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from utils.file_utils import compute_md5, list_files
from utils.image_utils import is_valid_image
from utils.logger import get_logger
from utils.video_utils import is_valid_video

logger: logging.Logger = get_logger(__name__)

# Supported extensions (lower-case)
IMAGE_EXTS: list[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
VIDEO_EXTS: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FileRecord:
    """Validation result for a single file."""

    path: str
    label: str           # "real" | "fake"
    file_type: str       # "image" | "video"
    size_mb: float
    md5: Optional[str]
    is_valid: bool
    is_duplicate: bool = False
    error_message: str = ""


@dataclass
class ValidationReport:
    """Aggregated validation statistics for a dataset."""

    dataset_name: str
    total_files: int
    valid_files: int
    invalid_files: int
    duplicate_files: int
    real_count: int
    fake_count: int
    records: List[FileRecord] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        """Percentage of valid files."""
        return (self.valid_files / self.total_files * 100) if self.total_files else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all file records to a pandas DataFrame."""
        return pd.DataFrame(
            [
                {
                    "path": r.path,
                    "label": r.label,
                    "file_type": r.file_type,
                    "size_mb": round(r.size_mb, 3),
                    "md5": r.md5,
                    "is_valid": r.is_valid,
                    "is_duplicate": r.is_duplicate,
                    "error": r.error_message,
                }
                for r in self.records
            ]
        )

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"Dataset: {self.dataset_name}\n"
            f"  Total  : {self.total_files}\n"
            f"  Valid  : {self.valid_files} ({self.validity_rate:.1f}%)\n"
            f"  Invalid: {self.invalid_files}\n"
            f"  Dupes  : {self.duplicate_files}\n"
            f"  Real   : {self.real_count} | Fake: {self.fake_count}\n"
        )


# ── Core validator ────────────────────────────────────────────────────────────

class DatasetValidator:
    """
    Validates a dataset directory that follows the structure::

        root/
          real/   ← images or videos labelled as real
          fake/   ← images or videos labelled as fake

    Args:
        dataset_name: Human-readable identifier for the dataset.
        compute_hashes: Whether to compute MD5 hashes (slower but enables
                        duplicate detection). Disable for very large datasets.
    """

    def __init__(self, dataset_name: str, compute_hashes: bool = True) -> None:
        self.dataset_name = dataset_name
        self.compute_hashes = compute_hashes
        self._hash_registry: Dict[str, List[str]] = {}

    def validate_directory(
        self,
        root_dir: str | Path,
        label_map: Optional[Dict[str, str]] = None,
    ) -> ValidationReport:
        """
        Validate all files under *root_dir*.

        Args:
            root_dir:  Root path that contains ``real/`` and ``fake/``
                       sub-directories (or custom subdirs defined by *label_map*).
            label_map: Mapping of sub-directory name → label string.
                       Defaults to ``{"real": "real", "fake": "fake"}``.

        Returns:
            Completed :class:`ValidationReport`.
        """
        if label_map is None:
            label_map = {"real": "real", "fake": "fake"}

        root = Path(root_dir)
        if not root.is_dir():
            logger.error("Dataset directory not found: %s", root)
            return ValidationReport(
                dataset_name=self.dataset_name,
                total_files=0, valid_files=0, invalid_files=0,
                duplicate_files=0, real_count=0, fake_count=0,
            )

        records: List[FileRecord] = []
        all_exts = IMAGE_EXTS + VIDEO_EXTS

        for subdir_name, label in label_map.items():
            subdir = root / subdir_name
            if not subdir.is_dir():
                logger.warning("Sub-directory not found, skipping: %s", subdir)
                continue

            files = list(list_files(subdir, extensions=all_exts, recursive=True))
            logger.info(
                "Validating %d files in '%s' (label=%s)…",
                len(files), subdir_name, label,
            )

            for file_path in tqdm(files, desc=f"  {label}", unit="file", leave=False):
                record = self._validate_file(file_path, label)
                records.append(record)

        # Mark duplicates
        self._mark_duplicates(records)

        # Aggregate stats
        valid = [r for r in records if r.is_valid]
        invalid = [r for r in records if not r.is_valid]
        dupes = [r for r in records if r.is_duplicate]
        real_cnt = sum(1 for r in records if r.label == "real" and r.is_valid)
        fake_cnt = sum(1 for r in records if r.label == "fake" and r.is_valid)

        report = ValidationReport(
            dataset_name=self.dataset_name,
            total_files=len(records),
            valid_files=len(valid),
            invalid_files=len(invalid),
            duplicate_files=len(dupes),
            real_count=real_cnt,
            fake_count=fake_cnt,
            records=records,
        )

        logger.info(report.summary())
        return report

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _validate_file(self, path: Path, label: str) -> FileRecord:
        """Validate a single file and return its record."""
        file_type = "image" if path.suffix.lower() in IMAGE_EXTS else "video"
        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        md5: Optional[str] = None
        error_msg = ""

        # Integrity check
        if file_type == "image":
            is_valid = is_valid_image(path)
        else:
            is_valid = is_valid_video(path)

        if not is_valid:
            error_msg = "File is corrupted or unreadable"
            logger.debug("Invalid file: %s", path)

        # Hash computation (if valid and requested)
        if is_valid and self.compute_hashes:
            try:
                md5 = compute_md5(path)
                self._hash_registry.setdefault(md5, []).append(str(path))
            except OSError as exc:
                logger.warning("Cannot hash %s: %s", path, exc)

        return FileRecord(
            path=str(path),
            label=label,
            file_type=file_type,
            size_mb=size_mb,
            md5=md5,
            is_valid=is_valid,
            error_message=error_msg,
        )

    def _mark_duplicates(self, records: List[FileRecord]) -> None:
        """Flag duplicate files in-place based on matching MD5 hashes."""
        # Build set of paths that are duplicates
        duplicate_paths: set[str] = set()
        for paths in self._hash_registry.values():
            if len(paths) > 1:
                # Keep first occurrence; mark rest as duplicates
                for dup_path in paths[1:]:
                    duplicate_paths.add(dup_path)

        for record in records:
            if record.path in duplicate_paths:
                record.is_duplicate = True
                logger.debug("Duplicate detected: %s", record.path)

    def validate_file(self, path: str | Path, label: str = "unknown") -> FileRecord:
        """
        Validate a single file in isolation.

        Useful for quick ad-hoc checks without running a full directory scan.

        Args:
            path:  File path.
            label: Label to assign (``"real"`` or ``"fake"``).

        Returns:
            :class:`FileRecord` for the file.
        """
        return self._validate_file(Path(path), label)
