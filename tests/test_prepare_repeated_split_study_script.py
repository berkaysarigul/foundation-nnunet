"""Regression tests for the repeated-split study manifest runner."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_repeated_split_study.py"


class TestPrepareRepeatedSplitStudyScript(unittest.TestCase):
    def _write_processed_dataset(self, dataset_root: Path) -> None:
        images_dir = dataset_root / "images"
        masks_dir = dataset_root / "original_masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        image_ids = [f"img_{idx:03d}" for idx in range(200)]
        positive_ids = set(image_ids[:40])
        for image_id in image_ids:
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images_dir / f"{image_id}.png")
            mask = np.zeros((8, 8), dtype=np.uint8)
            if image_id in positive_ids:
                mask[0, 0] = 255
            Image.fromarray(mask).save(masks_dir / f"{image_id}.png")

        (dataset_root / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "dataset_fingerprint": "dataset-fp",
                    "fingerprints": {"splits": "trusted-single-split-fp"},
                }
            ),
            encoding="utf-8",
        )

    def test_script_writes_canonical_split_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "data" / "processed" / "trusted_v1"
            study_root = root / "artifacts" / "repeated_splits"
            self._write_processed_dataset(dataset_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--study_id",
                    "study_alpha",
                    "--dataset_dir",
                    str(dataset_root),
                    "--split_seeds",
                    "42,43",
                    "--study_root",
                    str(study_root),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            split_manifest_path = Path(result.stdout.strip())
            self.assertTrue(split_manifest_path.exists())

            payload = yaml.safe_load(split_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["study_id"], "study_alpha")
            self.assertEqual(payload["split_count"], 2)
            self.assertEqual(payload["dataset_fingerprint"], "dataset-fp")
            self.assertEqual(payload["base_split_fingerprint"], "trusted-single-split-fp")
            self.assertEqual(
                [instance["split_instance_id"] for instance in payload["split_instances"]],
                ["split_001", "split_002"],
            )
            self.assertEqual(
                [instance["split_seed"] for instance in payload["split_instances"]],
                [42, 43],
            )


if __name__ == "__main__":
    unittest.main()
