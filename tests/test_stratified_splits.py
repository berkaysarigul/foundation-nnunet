"""Tests for the deterministic publication-facing stratified split policy."""

from __future__ import annotations

import unittest

from src.data.preprocess import create_splits


class TestStratifiedSplitPolicy(unittest.TestCase):
    def test_create_splits_is_deterministic_disjoint_and_sorted(self) -> None:
        image_ids = [f"img_{idx:03d}" for idx in range(200)]
        positive_ids = set(image_ids[:40])

        splits_a = create_splits(image_ids, positive_ids, seed=42)
        splits_b = create_splits(list(reversed(image_ids)), positive_ids, seed=42)

        self.assertEqual(splits_a, splits_b)
        self.assertEqual(
            len(splits_a["train"]) + len(splits_a["val"]) + len(splits_a["test"]),
            len(image_ids),
        )
        self.assertLessEqual(abs(len(splits_a["test"]) - 30), 0)
        self.assertLessEqual(abs(len(splits_a["val"]) - 30), 1)
        self.assertLessEqual(abs(len(splits_a["train"]) - 140), 1)

        for split_name in ("train", "val", "test"):
            self.assertEqual(splits_a[split_name], sorted(splits_a[split_name]))

        train_ids = set(splits_a["train"])
        val_ids = set(splits_a["val"])
        test_ids = set(splits_a["test"])
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(train_ids | val_ids | test_ids, set(image_ids))

    def test_create_splits_preserves_binary_class_ratio(self) -> None:
        image_ids = [f"img_{idx:03d}" for idx in range(200)]
        positive_ids = set(image_ids[:40])
        splits = create_splits(image_ids, positive_ids, seed=42)

        overall_ratio = len(positive_ids) / len(image_ids)
        for split_name in ("train", "val", "test"):
            split_ids = splits[split_name]
            positive_count = sum(1 for image_id in split_ids if image_id in positive_ids)
            split_ratio = positive_count / len(split_ids)
            self.assertLessEqual(abs(split_ratio - overall_ratio), 0.01)


if __name__ == "__main__":
    unittest.main()
