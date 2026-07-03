"""
utils/logger.py
===============
Centralized logging configuration for the DeepFake Detection project.

Provides a factory function `get_logger()` that returns a properly
configured logger for any module in the project.
"""

import logging
import logging.config
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

import yaml


# ── Module-level cache to avoid re-configuring ───────────────────────────────
_logging_initialized: bool = False


def _ensure_log_dir(log_dir: str = "logs") -> None:
    """Create the logs directory if it does not exist."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)


def setup_logging(
    config_path: Optional[str] = None,
    log_dir: str = "logs",
    level: int = logging.DEBUG,
) -> None:
    """
    Initialize the project-wide logging configuration.

    Reads from ``configs/logging_config.yaml`` if available, otherwise
    falls back to a sensible programmatic configuration.

    Args:
        config_path: Explicit path to a YAML logging config file.
                     Defaults to ``configs/logging_config.yaml``.
        log_dir:     Directory where log files will be written.
        level:       Root logging level used for the fallback config.
    """
    global _logging_initialized

    if _logging_initialized:
        return

    _ensure_log_dir(log_dir)

    # Resolve default config path relative to project root
    if config_path is None:
        config_path = str(
            Path(__file__).resolve().parent.parent / "configs" / "logging_config.yaml"
        )

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                log_cfg = yaml.safe_load(fh)

            # Patch log file paths to be absolute so they always resolve
            for handler_name, handler_cfg in log_cfg.get("handlers", {}).items():
                if "filename" in handler_cfg:
                    handler_cfg["filename"] = str(
                        Path(log_dir) / Path(handler_cfg["filename"]).name
                    )

            logging.config.dictConfig(log_cfg)
        except Exception as exc:  # pragma: no cover
            _fallback_logging(level, log_dir)
            logging.getLogger(__name__).warning(
                "Could not load logging config from %s: %s. Using fallback.",
                config_path,
                exc,
            )
    else:
        _fallback_logging(level, log_dir)

    _logging_initialized = True


def _fallback_logging(level: int, log_dir: str) -> None:
    """
    Configure logging programmatically when the YAML config is missing.

    Args:
        level:   Root logging level.
        log_dir: Directory for log files.
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, "deepfake_detection.log"),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        ),
    ]

    for handler in handlers:
        handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    logging.basicConfig(level=level, handlers=handlers)


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Retrieve a named logger, ensuring logging is initialised first.

    Args:
        name:    Module name, typically ``__name__``.
        log_dir: Log directory (used only on first call).

    Returns:
        Configured :class:`logging.Logger` instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module loaded successfully.")
    """
    setup_logging(log_dir=log_dir)
    return logging.getLogger(name)
