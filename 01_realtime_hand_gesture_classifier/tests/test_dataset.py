"""
Tests for dataset save/load utilities.
Uses a temporary directory so tests never touch real training data.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
import tempfile
import os
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGetClassCounts:
    def test_returns_all_10_classes(self):
        """Should always return counts for all 10 digit classes."""
        from components.dataset import get_class_counts
        counts = get_class_counts()
        assert set(counts.keys()) == {str(i) for i in range(10)}

    def test_counts_are_non_negative(self):
        from components.dataset import get_class_counts
        counts = get_class_counts()
        assert all(v >= 0 for v in counts.values())


class TestSaveSample:
    def test_save_creates_image_file(self, tmp_path):
        """Saving a sample should create a JPEG on disk."""
        from PIL import Image
        import numpy as np

        # Monkey-patch the paths to use tmp_path
        import components.dataset as ds
        original_raw  = ds.config.RAW_DIR
        original_csv  = ds.config.LANDMARKS_CSV
        original_data = ds.config.DATA_DIR

        try:
            ds.config.DATA_DIR      = tmp_path
            ds.config.RAW_DIR       = tmp_path / "raw"
            ds.config.LANDMARKS_CSV = tmp_path / "landmarks.csv"

            # Create a fake RGB frame and landmarks
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            landmarks = np.random.randn(63).astype(np.float32)

            from components.dataset import save_sample
            saved_path = save_sample(frame, landmarks, "3")

            assert saved_path.exists()
            assert saved_path.suffix == ".jpg"
            assert (tmp_path / "landmarks.csv").exists()

        finally:
            ds.config.DATA_DIR      = original_data
            ds.config.RAW_DIR       = original_raw
            ds.config.LANDMARKS_CSV = original_csv

    def test_csv_row_has_correct_columns(self, tmp_path):
        """CSV row must have label + 63 landmark columns."""
        import pandas as pd
        import components.dataset as ds

        original_raw  = ds.config.RAW_DIR
        original_csv  = ds.config.LANDMARKS_CSV
        original_data = ds.config.DATA_DIR

        try:
            ds.config.DATA_DIR      = tmp_path
            ds.config.RAW_DIR       = tmp_path / "raw"
            ds.config.LANDMARKS_CSV = tmp_path / "landmarks.csv"

            frame     = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            landmarks = np.random.randn(63).astype(np.float32)

            from components.dataset import save_sample
            save_sample(frame, landmarks, "5")

            df = pd.read_csv(tmp_path / "landmarks.csv")
            assert "label" in df.columns
            assert len([c for c in df.columns if c.startswith("lm_")]) == 63

        finally:
            ds.config.DATA_DIR      = original_data
            ds.config.RAW_DIR       = original_raw
            ds.config.LANDMARKS_CSV = original_csv
