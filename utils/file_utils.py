"""
utils/file_utils.py
===================
File system helpers used throughout the project.

Provides utilities for:
- Directory creation
- File validation
- MD5-based duplicate detection
- Safe file copying / moving
- Recursive file listing by extension
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Generator, Optional

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


# ── Directory helpers ─────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory (and all parents) if it does not already exist.

    Args:
        path: Target directory path.

    Returns:
        Resolved :class:`pathlib.Path` of the created directory.
    """
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def clean_dir(path: str | Path, confirm: bool = False) -> None:
    """
    Remove all contents from a directory without deleting the directory itself.

    Args:
        path:    Directory to clean.
        confirm: Safety flag — must be ``True`` to actually perform the operation.

    Raises:
        ValueError: If ``confirm`` is ``False``.
        FileNotFoundError: If the directory does not exist.
    """
    if not confirm:
        raise ValueError("Pass confirm=True to clean a directory.")
    dir_path = Path(path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    for item in dir_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    logger.debug("Cleaned directory: %s", dir_path)


# ── File listing ─────────────────────────────────────────────────────────────

def list_files(
    directory: str | Path,
    extensions: Optional[list[str]] = None,
    recursive: bool = True,
) -> Generator[Path, None, None]:
    """
    Yield all files in a directory, optionally filtered by extension.

    Args:
        directory:  Root directory to search.
        extensions: List of lowercase extensions including the dot, e.g.
                    ``[".jpg", ".png"]``. ``None`` means all files.
        recursive:  Whether to search subdirectories.

    Yields:
        :class:`pathlib.Path` objects for each matching file.
    """
    root = Path(directory)
    if not root.is_dir():
        logger.warning("list_files: directory does not exist: %s", root)
        return

    pattern = "**/*" if recursive else "*"
    for file_path in root.glob(pattern):
        if not file_path.is_file():
            continue
        if extensions is None or file_path.suffix.lower() in extensions:
            yield file_path


def count_files(
    directory: str | Path,
    extensions: Optional[list[str]] = None,
    recursive: bool = True,
) -> int:
    """
    Count files in a directory matching the given extensions.

    Args:
        directory:  Root directory.
        extensions: Extension filter (see :func:`list_files`).
        recursive:  Recurse into subdirectories.

    Returns:
        Total number of matching files.
    """
    return sum(1 for _ in list_files(directory, extensions, recursive))


# ── File validation ───────────────────────────────────────────────────────────

def file_exists(path: str | Path) -> bool:
    """Return ``True`` if *path* points to an existing regular file."""
    return Path(path).is_file()


def get_file_size_mb(path: str | Path) -> float:
    """
    Return file size in megabytes.

    Args:
        path: Path to the file.

    Returns:
        Size in MB as a float.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return p.stat().st_size / (1024 * 1024)


# ── Duplicate detection ───────────────────────────────────────────────────────

def compute_md5(path: str | Path, chunk_size: int = 65536) -> str:
    """
    Compute the MD5 hash of a file.

    Args:
        path:       File path.
        chunk_size: Read chunk size in bytes.

    Returns:
        Hex-encoded MD5 digest string.
    """
    md5 = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def find_duplicates(
    files: list[Path],
) -> dict[str, list[Path]]:
    """
    Identify duplicate files by MD5 hash.

    Args:
        files: List of file paths to check.

    Returns:
        Dictionary mapping each duplicated MD5 hash to the list of
        file paths that share that hash. Non-duplicated files are
        excluded from the result.
    """
    hash_map: dict[str, list[Path]] = {}
    for file_path in files:
        try:
            digest = compute_md5(file_path)
            hash_map.setdefault(digest, []).append(file_path)
        except OSError as exc:
            logger.warning("Cannot hash %s: %s", file_path, exc)

    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


# ── Safe copy / move ─────────────────────────────────────────────────────────

def safe_copy(src: str | Path, dst: str | Path) -> Path:
    """
    Copy *src* to *dst*, creating parent directories as needed.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        Path to the copied file.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    shutil.copy2(src_path, dst_path)
    logger.debug("Copied %s → %s", src_path, dst_path)
    return dst_path


def safe_move(src: str | Path, dst: str | Path) -> Path:
    """
    Move *src* to *dst*, creating parent directories as needed.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        Path to the moved file.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    shutil.move(str(src_path), str(dst_path))
    logger.debug("Moved %s → %s", src_path, dst_path)
    return dst_path


# ── Config loader ─────────────────────────────────────────────────────────────

def load_yaml_config(path: str | Path) -> dict:
    """
    Load a YAML configuration file into a Python dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    import yaml  # lazy import to keep module lightweight

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    logger.debug("Loaded config from %s", config_path)
    return cfg or {}
