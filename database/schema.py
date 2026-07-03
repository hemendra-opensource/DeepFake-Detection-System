"""
database/schema.py
==================
SQLite database schema definitions for DeepFake detection history.

Table: detections
- Stores every detection result for the detection history page
- Indexed on timestamp and prediction for fast filtering
"""

import logging
import sqlite3
from pathlib import Path

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# ── SQL statements ────────────────────────────────────────────────────────────

CREATE_DETECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS detections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name        TEXT    NOT NULL,
    file_type        TEXT    NOT NULL DEFAULT 'image',   -- 'image' | 'video'
    prediction       TEXT    NOT NULL,                   -- 'FAKE' | 'REAL'
    confidence       REAL    NOT NULL,
    fake_probability REAL    NOT NULL DEFAULT 0.0,
    model_name       TEXT    NOT NULL DEFAULT 'xceptionnet',
    processing_time  REAL    NOT NULL DEFAULT 0.0,       -- milliseconds
    num_frames       INTEGER NOT NULL DEFAULT 0,
    fake_frames      INTEGER NOT NULL DEFAULT 0,
    real_frames      INTEGER NOT NULL DEFAULT 0,
    report_path      TEXT,
    timestamp        TEXT    NOT NULL,                   -- ISO 8601
    notes            TEXT    DEFAULT ''
);
"""

CREATE_TIMESTAMP_INDEX = """
CREATE INDEX IF NOT EXISTS idx_detections_timestamp
ON detections(timestamp DESC);
"""

CREATE_PREDICTION_INDEX = """
CREATE INDEX IF NOT EXISTS idx_detections_prediction
ON detections(prediction);
"""


def initialise_database(db_path: str | Path) -> sqlite3.Connection:
    """
    Create the SQLite database and all tables if they don't exist.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Open :class:`sqlite3.Connection` to the initialised database.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Enable dict-like row access
    conn.execute("PRAGMA journal_mode=WAL;")  # Better concurrency

    with conn:
        conn.execute(CREATE_DETECTIONS_TABLE)
        conn.execute(CREATE_TIMESTAMP_INDEX)
        conn.execute(CREATE_PREDICTION_INDEX)

    logger.info("Database initialised at: %s", db_path)
    return conn
