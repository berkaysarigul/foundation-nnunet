"""Smoke tests for scripts/evaluate_fx_prior_refiner.py."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evaluate_fx_prior_refiner as eval_refiner  # noqa: E402
from src.models.refiner_unet import FoundationXPriorRefinerUNet  # noqa: E402


def _save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _build_manifest(root: Path) -> Path:
    images = root / "nnUNet_raw" / "Dataset101_Pneumothorax" / "imagesTs"
    labels = root / "nnUNet_raw" / "Dataset101_Pneumothorax" / "heldout_labelsTs"
    priors = root / "artifacts" / "priors"
    manifest = root / "test_manifest.csv"
    rng = np.random.default_rng(456)
    rows = []
    for idx in range(2):
        case_id = f"siim_{idx + 10:06d}"
        positive = idx == 0
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
                "split": "test",
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


class TestEvaluateFxPriorRefinerSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = _build_manifest(self.root)
        self.checkpoint = self.root / "checkpoint.pth"
        model = FoundationXPriorRefinerUNet(in_channels=2, base_channels=4, norm="group")
        torch.save(
            {
                "schema_version": 1,
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_mode": "image_prior",
                    "target_size": 32,
                    "model": {
                        "out_channels": 1,
                        "base_channels": 4,
                        "norm": "group",
                    },
                },
            },
            self.checkpoint,
        )

    def test_evaluate_script_runs_tiny_smoke(self) -> None:
        out_dir = self.root / "artifacts" / "refiner" / "runs" / "eval_smoke" / "test_eval"
        code = eval_refiner.main(
            [
                "--manifest",
                str(self.manifest),
                "--checkpoint",
                str(self.checkpoint),
                "--input-mode",
                "image_prior",
                "--out",
                str(out_dir),
                "--device",
                "cpu",
                "--batch-size",
                "2",
                "--threshold",
                "0.5",
                "--save-visuals",
                "true",
            ]
        )
        self.assertEqual(code, 0)
        for name in ["summary.yaml", "test_metrics.csv", "per_case_metrics.csv", "report.md"]:
            self.assertTrue((out_dir / name).exists(), name)
        self.assertTrue((out_dir / "visual_overlays" / "siim_000010.png").exists())


if __name__ == "__main__":
    unittest.main()
