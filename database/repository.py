"""
database/repository.py
=======================
Data access layer for the DeepFake detection history database.

Provides a clean CRUD interface over the SQLite ``detections`` table,
keeping all SQL out of the dashboard and inference modules.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from database.schema import initialise_database
from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class DetectionRepository:
    """
    CRUD repository for DeepFake detection records.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path = "database/detections.db") -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()

    # ── Connection management ─────────────────────────────────────────────────

    def _connect(self) -> None:
        """Open the database connection, initialising schema if needed."""
        self._conn = initialise_database(self.db_path)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DetectionRepository":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Create ────────────────────────────────────────────────────────────────

    def insert_detection(
        self,
        file_name: str,
        prediction: str,
        confidence: float,
        fake_probability: float = 0.0,
        file_type: str = "image",
        model_name: str = "xceptionnet",
        processing_time: float = 0.0,
        num_frames: int = 0,
        fake_frames: int = 0,
        real_frames: int = 0,
        report_path: Optional[str] = None,
        notes: str = "",
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Insert a new detection record.

        Args:
            file_name:       Source file name.
            prediction:      ``"FAKE"`` or ``"REAL"``.
            confidence:      Prediction confidence (0–1).
            fake_probability: Raw sigmoid output.
            file_type:       ``"image"`` or ``"video"``.
            model_name:      Model used.
            processing_time: Time in milliseconds.
            num_frames:      Total frames analysed (videos).
            fake_frames:     Frames labelled FAKE.
            real_frames:     Frames labelled REAL.
            report_path:     Path to generated PDF report.
            notes:           Optional notes.
            timestamp:       Datetime of detection (defaults to now).

        Returns:
            ``rowid`` of the inserted record.
        """
        if timestamp is None:
            timestamp = datetime.now()

        assert self._conn is not None

        cursor = self._conn.execute(
            """
            INSERT INTO detections (
                file_name, file_type, prediction, confidence,
                fake_probability, model_name, processing_time,
                num_frames, fake_frames, real_frames,
                report_path, timestamp, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_name, file_type, prediction, round(confidence, 6),
                round(fake_probability, 6), model_name, round(processing_time, 2),
                num_frames, fake_frames, real_frames,
                report_path, timestamp.isoformat(), notes,
            ),
        )
        self._conn.commit()
        record_id = cursor.lastrowid
        logger.debug("Inserted detection #%d: %s → %s", record_id, file_name, prediction)
        return record_id

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_all(self, limit: int = 500) -> pd.DataFrame:
        """
        Retrieve all detection records as a DataFrame (newest first).

        Args:
            limit: Maximum number of records to return.

        Returns:
            :class:`pandas.DataFrame` of detection records.
        """
        assert self._conn is not None
        query = """
            SELECT id, file_name, file_type, prediction, confidence,
                   fake_probability, model_name, processing_time,
                   num_frames, fake_frames, real_frames,
                   report_path, timestamp, notes
            FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, self._conn, params=(limit,))
        return df

    def get_by_id(self, record_id: int) -> Optional[Dict]:
        """
        Retrieve a single detection record by ID.

        Args:
            record_id: Row ID to look up.

        Returns:
            Dict of row values, or ``None`` if not found.
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT * FROM detections WHERE id = ?", (record_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_statistics(self) -> Dict:
        """
        Compute summary statistics for the dashboard KPI cards.

        Returns:
            Dict with keys: total, fake_count, real_count, avg_confidence,
            avg_processing_time, today_count.
        """
        assert self._conn is not None
        today_str = datetime.now().strftime("%Y-%m-%d")

        stats = {}
        cur = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN prediction='FAKE' THEN 1 ELSE 0 END) AS fake_count,
                SUM(CASE WHEN prediction='REAL' THEN 1 ELSE 0 END) AS real_count,
                AVG(confidence) AS avg_confidence,
                AVG(processing_time) AS avg_processing_time,
                SUM(CASE WHEN timestamp LIKE ? THEN 1 ELSE 0 END) AS today_count
            FROM detections
            """,
            (f"{today_str}%",),
        )
        row = cur.fetchone()
        if row:
            stats = {
                "total": row["total"] or 0,
                "fake_count": row["fake_count"] or 0,
                "real_count": row["real_count"] or 0,
                "avg_confidence": round(row["avg_confidence"] or 0, 4),
                "avg_processing_time": round(row["avg_processing_time"] or 0, 2),
                "today_count": row["today_count"] or 0,
            }
        return stats

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_by_id(self, record_id: int) -> bool:
        """
        Delete a single detection record.

        Args:
            record_id: Row ID to delete.

        Returns:
            ``True`` if a row was deleted.
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            "DELETE FROM detections WHERE id = ?", (record_id,)
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted detection #%d", record_id)
        return deleted

    def delete_all(self) -> int:
        """
        Delete all records from the detections table.

        Returns:
            Number of rows deleted.
        """
        assert self._conn is not None
        cursor = self._conn.execute("DELETE FROM detections")
        self._conn.commit()
        logger.warning("All %d detection records deleted.", cursor.rowcount)
        return cursor.rowcount
