"""Tests for the deterministic publication-facing stratified split policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.data.preprocess import create_splits
from src.data.repeated_splits import (
    build_repeated_split_instances,
    load_processed_dataset_binary_labels,
)


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

    def test_load_processed_dataset_binary_labels_reads_sorted_ids_and_positives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            images_dir = dataset_root / "images"
            masks_dir = dataset_root / "original_masks"
            images_dir.mkdir()
            masks_dir.mkdir()

            for image_id, is_positive in (("img_b", False), ("img_a", True), ("img_c", False)):
                Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(images_dir / f"{image_id}.png")
                mask = np.zeros((4, 4), dtype=np.uint8)
                if is_positive:
                    mask[0, 0] = 255
                Image.fromarray(mask).save(masks_dir / f"{image_id}.png")

            image_ids, positive_ids = load_processed_dataset_binary_labels(dataset_root)

            self.assertEqual(image_ids, ["img_a", "img_b", "img_c"])
            self.assertEqual(positive_ids, {"img_a"})

    def test_build_repeated_split_instances_uses_canonical_ids_and_unique_seeds(self) -> None:
        image_ids = [f"img_{idx:03d}" for idx in range(200)]
        positive_ids = set(image_ids[:40])

        split_instances = build_repeated_split_instances(
            image_ids,
            positive_ids,
            split_seeds=[42, 43],
        )

        self.assertEqual(
            [instance["split_instance_id"] for instance in split_instances],
            ["split_001", "split_002"],
        )
        self.assertEqual(
            [instance["split_seed"] for instance in split_instances],
            [42, 43],
        )
        self.assertTrue(split_instances[0]["train_ids"])
        self.assertTrue(split_instances[0]["val_ids"])
        self.assertTrue(split_instances[0]["test_ids"])

        with self.assertRaisesRegex(ValueError, "split_seeds must be unique"):
            build_repeated_split_instances(
                image_ids,
                positive_ids,
                split_seeds=[42, 42],
            )

    def test_build_repeated_split_instances_rejects_duplicate_split_fingerprints(self) -> None:
        image_ids = [f"img_{idx:03d}" for idx in range(200)]
        positive_ids = set(image_ids[:40])
        duplicate_splits = {
            "train": sorted(image_ids[:140]),
            "val": sorted(image_ids[140:170]),
            "test": sorted(image_ids[170:200]),
        }

        with patch("src.data.repeated_splits.create_splits", return_value=duplicate_splits):
            with self.assertRaisesRegex(ValueError, "distinct split instances"):
                build_repeated_split_instances(
                    image_ids,
                    positive_ids,
                    split_seeds=[42, 43],
                )


if __name__ == "__main__":
    unittest.main()
