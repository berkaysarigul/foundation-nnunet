"""Tests for Foundation X prior refiner dataset loading."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.fx_prior_refiner_dataset import FoundationXPriorRefinerDataset


def _save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _build_fixture(root: Path) -> Path:
    image_path = root / "nnUNet_raw" / "Dataset101_Pneumothorax" / "imagesTr" / "siim_000001_0000.png"
    label_path = root / "nnUNet_raw" / "Dataset101_Pneumothorax" / "labelsTr" / "siim_000001.png"
    prob_path = root / "artifacts" / "priors" / "siim_000001.png"
    manifest = root / "manifest.csv"

    image = np.full((512, 512), 96, dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[64:128, 96:160] = 255
    prior = np.linspace(0, 255, 224 * 224, dtype=np.uint8).reshape(224, 224)
    _save_png(image_path, image)
    _save_png(label_path, mask)
    _save_png(prob_path, prior)

    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "split",
                "image_path",
                "label_path",
                "probability_map_path",
                "gt_is_positive",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "siim_000001",
                "split": "train",
                "image_path": str(image_path),
                "label_path": str(label_path),
                "probability_map_path": str(prob_path),
                "gt_is_positive": "True",
            }
        )
    return manifest


class TestFoundationXPriorRefinerDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = _build_fixture(self.root)

    def _load(self, input_mode: str):
        ds = FoundationXPriorRefinerDataset(
            self.manifest,
            input_mode=input_mode,
            target_size=512,
            repo_root=self.root,
        )
        return ds[0]

    def test_image_prior_shape(self) -> None:
        x, y, metadata = self._load("image_prior")
        self.assertEqual(tuple(x.shape), (2, 512, 512))
        self.assertEqual(tuple(y.shape), (1, 512, 512))
        self.assertEqual(metadata["case_id"], "siim_000001")

    def test_image_only_shape(self) -> None:
        x, y, _metadata = self._load("image_only")
        self.assertEqual(tuple(x.shape), (1, 512, 512))
        self.assertEqual(tuple(y.shape), (1, 512, 512))

    def test_prior_only_shape(self) -> None:
        x, y, _metadata = self._load("prior_only")
        self.assertEqual(tuple(x.shape), (1, 512, 512))
        self.assertEqual(tuple(y.shape), (1, 512, 512))

    def test_probability_map_is_upsampled_to_target_size(self) -> None:
        x, _y, _metadata = self._load("image_prior")
        prior = x[1]
        self.assertEqual(tuple(prior.shape), (512, 512))
        self.assertGreater(float(prior.max()), float(prior.min()))

    def test_mask_is_binary(self) -> None:
        _x, y, _metadata = self._load("image_prior")
        unique = torch.unique(y)
        self.assertTrue(set(unique.tolist()).issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
