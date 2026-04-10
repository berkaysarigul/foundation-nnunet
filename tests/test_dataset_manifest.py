"""Tests for trusted processed-dataset manifest helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.dataset_manifest import compute_split_fingerprint, fingerprint_directory, summarize_mask_directory, summarize_splits


class TestDatasetManifestHelpers(unittest.TestCase):
    def test_split_fingerprint_is_deterministic(self) -> None:
        splits_a = {"train": ["b", "a"], "val": ["c"], "test": []}
        splits_b = {"val": ["c"], "train": ["a", "b"], "test": []}
        self.assertEqual(compute_split_fingerprint(splits_a), compute_split_fingerprint(splits_b))

    def test_mask_summary_and_split_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            mask_dir = root / "original_masks"
            mask_dir.mkdir()

            negative = np.zeros((4, 4), dtype=np.uint8)
            positive = negative.copy()
            positive[1:3, 1:3] = 255

            Image.fromarray(negative).save(mask_dir / "neg.png")
            Image.fromarray(positive).save(mask_dir / "pos.png")

            stats, positive_ids = summarize_mask_directory(mask_dir, image_size=4)
            self.assertEqual(stats["image_count"], 2)
            self.assertEqual(stats["positive_image_count"], 1)
            self.assertTrue(stats["binary_unique_values_ok"])
            self.assertEqual(positive_ids, {"pos"})

            split_summary = summarize_splits(
                {"train": ["pos"], "val": ["neg"], "test": []},
                positive_ids,
            )
            self.assertEqual(split_summary["train"]["positive_image_count"], 1)
            self.assertEqual(split_summary["val"]["positive_image_count"], 0)

    def test_directory_fingerprint_changes_with_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(root / "a.png")
            first = fingerprint_directory(root)["fingerprint"]

            Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(root / "a.png")
            second = fingerprint_directory(root)["fingerprint"]

            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
