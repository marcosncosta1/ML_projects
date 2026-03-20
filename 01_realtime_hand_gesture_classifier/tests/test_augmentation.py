"""
Tests for the landmark augmentation pipeline.
Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from components.augmentation import augment_batch, _augment_array


class TestAugmentBatch:
    def test_output_shape_with_n_copies(self):
        """Total rows = original + n_copies × original."""
        X = np.random.randn(10, 63).astype(np.float32)
        y = np.arange(10)
        X_aug, y_aug = augment_batch(X, y, n_copies=4)
        assert X_aug.shape == (10 * 5, 63)
        assert y_aug.shape == (10 * 5,)

    def test_labels_preserved(self):
        """Each original label must appear n_copies+1 times."""
        X = np.random.randn(5, 63).astype(np.float32)
        y = np.array([0, 1, 2, 3, 4])
        _, y_aug = augment_batch(X, y, n_copies=3)
        for label in range(5):
            assert (y_aug == label).sum() == 4  # 1 original + 3 copies

    def test_feature_dimension_unchanged(self):
        """Augmentation must never change the 63-feature structure."""
        X = np.random.randn(20, 63).astype(np.float32)
        y = np.zeros(20, dtype=int)
        X_aug, _ = augment_batch(X, y, n_copies=5)
        assert X_aug.shape[1] == 63

    def test_augmented_differs_from_original(self):
        """Augmented samples should not be identical to the originals."""
        X = np.random.randn(10, 63).astype(np.float32)
        y = np.zeros(10, dtype=int)
        X_aug, _ = augment_batch(X, y, n_copies=1)
        originals = X_aug[:10]
        augmented = X_aug[10:]
        assert not np.allclose(originals, augmented)

    def test_n_copies_zero(self):
        """n_copies=0 should return only the original data."""
        X = np.random.randn(8, 63).astype(np.float32)
        y = np.arange(8)
        X_aug, y_aug = augment_batch(X, y, n_copies=0)
        assert X_aug.shape == (8, 63)
        np.testing.assert_array_equal(y_aug, y)


class TestAugmentArray:
    def test_output_shape(self):
        X = np.random.randn(15, 63).astype(np.float32)
        result = _augment_array(X)
        assert result.shape == (15, 63)

    def test_dtype_preserved(self):
        X = np.random.randn(5, 63).astype(np.float32)
        result = _augment_array(X)
        assert result.dtype == np.float32

    def test_values_in_reasonable_range(self):
        """
        After augmentation (scale 0.85-1.15, translation ±0.12, rotation ±25°),
        values shouldn't explode. Normalised landmarks are typically in [-3, 3].
        """
        X = np.random.randn(50, 63).astype(np.float32) * 0.5
        result = _augment_array(X)
        assert np.abs(result).max() < 10.0
