"""
tests/test_database.py
=======================
Unit tests for the SQLite database layer.

Tests CRUD operations on the DetectionRepository.
"""

import sys
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


class TestDetectionRepository(unittest.TestCase):

    def setUp(self):
        """Create a fresh in-memory-like temp database for each test."""
        self.tmpdir = tempfile.mkdtemp()
        db_path = Path(self.tmpdir) / "test.db"
        from database.repository import DetectionRepository
        self.repo = DetectionRepository(db_path=db_path)

    def tearDown(self):
        self.repo.close()

    def _insert_sample(self, prediction: str = "FAKE", file_name: str = "test.jpg") -> int:
        return self.repo.insert_detection(
            file_name=file_name,
            prediction=prediction,
            confidence=0.92,
            fake_probability=0.92,
            file_type="image",
            model_name="xceptionnet",
            processing_time=150.5,
            timestamp=datetime.now(),
        )

    # ── INSERT ────────────────────────────────────────────────────────────────

    def test_insert_returns_positive_id(self):
        """insert_detection should return a positive integer row ID."""
        rid = self._insert_sample()
        self.assertIsInstance(rid, int)
        self.assertGreater(rid, 0)

    def test_insert_multiple_records(self):
        """Multiple insertions should each get unique IDs."""
        id1 = self._insert_sample("FAKE", "a.jpg")
        id2 = self._insert_sample("REAL", "b.jpg")
        self.assertNotEqual(id1, id2)

    # ── READ ──────────────────────────────────────────────────────────────────

    def test_get_all_returns_dataframe(self):
        """get_all should return a pandas DataFrame."""
        import pandas as pd
        self._insert_sample()
        df = self.repo.get_all()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_get_all_ordered_newest_first(self):
        """Records should be returned newest first."""
        self._insert_sample("FAKE", "first.jpg")
        self._insert_sample("REAL", "second.jpg")
        df = self.repo.get_all()
        self.assertGreaterEqual(len(df), 2)
        # ID of first row should be >= second row (newest first)
        self.assertGreaterEqual(df.iloc[0]["id"], df.iloc[1]["id"])

    def test_get_by_id_found(self):
        """get_by_id should return the correct record."""
        rid = self._insert_sample("FAKE", "myfile.jpg")
        record = self.repo.get_by_id(rid)
        self.assertIsNotNone(record)
        self.assertEqual(record["file_name"], "myfile.jpg")
        self.assertEqual(record["prediction"], "FAKE")

    def test_get_by_id_not_found(self):
        """get_by_id with a non-existent ID should return None."""
        result = self.repo.get_by_id(99999)
        self.assertIsNone(result)

    def test_confidence_stored_correctly(self):
        """Confidence value must be stored and retrieved correctly."""
        rid = self.repo.insert_detection(
            file_name="conf_test.jpg",
            prediction="REAL",
            confidence=0.8765,
            fake_probability=0.1235,
            processing_time=100.0,
        )
        record = self.repo.get_by_id(rid)
        self.assertAlmostEqual(record["confidence"], 0.8765, places=4)

    # ── STATISTICS ────────────────────────────────────────────────────────────

    def test_get_statistics_empty(self):
        """Statistics on an empty DB should return zeros."""
        stats = self.repo.get_statistics()
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["fake_count"], 0)

    def test_get_statistics_counts(self):
        """Statistics should correctly count FAKE vs REAL."""
        self._insert_sample("FAKE")
        self._insert_sample("FAKE")
        self._insert_sample("REAL")
        stats = self.repo.get_statistics()
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["fake_count"], 2)
        self.assertEqual(stats["real_count"], 1)

    # ── DELETE ────────────────────────────────────────────────────────────────

    def test_delete_by_id_removes_record(self):
        """delete_by_id should remove the record so get_by_id returns None."""
        rid = self._insert_sample()
        deleted = self.repo.delete_by_id(rid)
        self.assertTrue(deleted)
        self.assertIsNone(self.repo.get_by_id(rid))

    def test_delete_by_id_returns_false_if_missing(self):
        """delete_by_id on a non-existent ID should return False."""
        result = self.repo.delete_by_id(99999)
        self.assertFalse(result)

    def test_delete_all(self):
        """delete_all should remove every record."""
        self._insert_sample("FAKE")
        self._insert_sample("REAL")
        self._insert_sample("FAKE")
        deleted = self.repo.delete_all()
        self.assertEqual(deleted, 3)
        df = self.repo.get_all()
        self.assertTrue(df.empty)

    # ── VIDEO FIELDS ──────────────────────────────────────────────────────────

    def test_video_fields_stored(self):
        """Video-specific fields (num_frames, fake_frames, etc.) should persist."""
        rid = self.repo.insert_detection(
            file_name="video.mp4",
            prediction="FAKE",
            confidence=0.75,
            fake_probability=0.75,
            file_type="video",
            num_frames=50,
            fake_frames=35,
            real_frames=15,
            processing_time=2000.0,
        )
        record = self.repo.get_by_id(rid)
        self.assertEqual(record["num_frames"], 50)
        self.assertEqual(record["fake_frames"], 35)
        self.assertEqual(record["real_frames"], 15)
        self.assertEqual(record["file_type"], "video")


if __name__ == "__main__":
    unittest.main(verbosity=2)
