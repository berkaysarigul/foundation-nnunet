"""Smoke tests for scripts/train_fx_prior_refiner.py."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_fx_prior_refiner as train_refiner  # noqa: E402


def _save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _build_manifest(root: Path) -> Path:
    images = root / "nnUNet_raw" / "Dataset101_Pneumothorax" / "imagesTr"
    labels = root / "nnUNet_raw" / "Dataset101_Pneumothorax" / "labelsTr"
    priors = root / "artifacts" / "priors"
    manifest = root / "train_manifest.csv"
    rng = np.random.default_rng(123)

    rows = []
    for idx in range(4):
        case_id = f"siim_{idx + 1:06d}"
        positive = idx < 2
        image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        if positive:
            mask[8:16, 8:16] = 255
        prior = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)

        image_path = images / f"{case_id}_0000.png"
        label_path = labels / f"{case_id}.png"
        prob_path = priors / f"{case_id}.png"
        _save_png(image_path, image)
        _save_png(label_path, mask)
        _save_png(prob_path, prior)
        rows.append(
            {
                "case_id": case_id,
                "split": "train",
                "image_path": str(image_path),
                "label_path": str(label_path),
                "probability_map_path": str(prob_path),
                "gt_is_positive": "True" if positive else "False",
            }
        )

    with manifest.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "split",
            "image_path",
            "label_path",
            "probability_map_path",
            "gt_is_positive",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


class TestTrainFxPriorRefinerSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = _build_manifest(self.root)

    def test_train_script_runs_tiny_smoke(self) -> None:
        out_dir = self.root / "artifacts" / "refiner" / "runs" / "local_smoke"
        code = train_refiner.main(
            [
                "--train-manifest",
                str(self.manifest),
                "--out",
                str(out_dir),
                "--input-mode",
                "image_prior",
                "--target-size",
                "32",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--lr",
                "0.001",
                "--val-split",
                "0.5",
                "--seed",
                "42",
                "--device",
                "cpu",
                "--amp",
                "false",
                "--num-workers",
                "0",
                "--early-stopping-patience",
                "1",
            ]
        )
        self.assertEqual(code, 0)
        for name in [
            "config.yaml",
            "split_train.csv",
            "split_val.csv",
            "train_log.csv",
            "val_metrics.csv",
            "best_model.pth",
            "report.md",
        ]:
            self.assertTrue((out_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
