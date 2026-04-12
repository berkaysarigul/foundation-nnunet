"""Regression tests for the fixed D-031 train-only ROI crop policy."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.dataset import PneumothoraxDataset


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise IOError(f"Failed to write fixture PNG: {path}")


class TestTrainROICropPolicy(unittest.TestCase):
    def build_processed_dataset(self, root: Path) -> None:
        splits = {"train": ["pos_train", "neg_train"], "val": ["pos_val"], "test": ["neg_test"]}
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "original_masks").mkdir(parents=True, exist_ok=True)
        (root / "dilated_masks").mkdir(parents=True, exist_ok=True)
        (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")

        base_image = np.tile(np.arange(512, dtype=np.uint8), (512, 1))
        empty_mask = np.zeros((512, 512), dtype=np.uint8)

        pos_mask = np.zeros((512, 512), dtype=np.uint8)
        pos_mask[220:240, 300:320] = 255

        for image_id, mask in {
            "pos_train": pos_mask,
            "neg_train": empty_mask,
            "pos_val": pos_mask,
            "neg_test": empty_mask,
        }.items():
            write_png(root / "images" / f"{image_id}.png", base_image)
            write_png(root / "original_masks" / f"{image_id}.png", mask)
            write_png(root / "dilated_masks" / f"{image_id}.png", mask)

    def test_positive_train_crop_enlarges_the_roi_after_resize_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "processed"
            self.build_processed_dataset(dataset_root)

            uncropped = PneumothoraxDataset(
                str(dataset_root),
                split="train",
                img_size=512,
                transform=None,
                mask_variant="dilated_masks",
            )
            cropped = PneumothoraxDataset(
                str(dataset_root),
                split="train",
                img_size=512,
                transform=None,
                mask_variant="dilated_masks",
                train_crop={"mode": "roi_train_only", "crop_size": 384},
            )

            random.seed(0)
            _, uncropped_mask = uncropped[0]
            random.seed(0)
            _, cropped_mask = cropped[0]

            self.assertEqual(tuple(cropped_mask.shape), (1, 512, 512))
            self.assertGreater(float(cropped_mask.sum().item()), float(uncropped_mask.sum().item()))

    def test_negative_train_crop_preserves_empty_mask_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "processed"
            self.build_processed_dataset(dataset_root)
            cropped = PneumothoraxDataset(
                str(dataset_root),
                split="train",
                img_size=512,
                transform=None,
                mask_variant="dilated_masks",
                train_crop={"mode": "roi_train_only", "crop_size": 384},
            )

            random.seed(0)
            image, mask = cropped[1]

            self.assertEqual(tuple(image.shape), (1, 512, 512))
            self.assertEqual(tuple(mask.shape), (1, 512, 512))
            self.assertEqual(float(mask.sum().item()), 0.0)

    def test_train_crop_is_rejected_outside_the_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "processed"
            self.build_processed_dataset(dataset_root)

            with self.assertRaisesRegex(ValueError, "split='train'"):
                PneumothoraxDataset(
                    str(dataset_root),
                    split="val",
                    img_size=512,
                    transform=None,
                    mask_variant="original_masks",
                    train_crop={"mode": "roi_train_only", "crop_size": 384},
                )

    def test_train_crop_requires_input_size_512(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "processed"
            self.build_processed_dataset(dataset_root)

            with self.assertRaisesRegex(ValueError, "input_size=512"):
                PneumothoraxDataset(
                    str(dataset_root),
                    split="train",
                    img_size=256,
                    transform=None,
                    mask_variant="dilated_masks",
                    train_crop={"mode": "roi_train_only", "crop_size": 384},
                )


if __name__ == "__main__":
    unittest.main()
