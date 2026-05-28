"""Focused tests for PR-11 data pipeline scripts."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_ptx498_manifest as ptx_manifest  # noqa: E402
import build_siim_acr_png_manifest as siim_manifest  # noqa: E402
import extract_raw_datasets as extract_raw  # noqa: E402
import validate_fx_prior_manifests as validate_fx  # noqa: E402


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class PR11PipelineScriptTests(unittest.TestCase):
    def test_siim_manifest_pairs_csv_and_mask_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "SIIM-ACR"
            _write_png(root / "png_images" / "0_train_0_.png", np.zeros((4, 4), dtype=np.uint8))
            _write_png(root / "png_masks" / "0_train_0_.png", np.zeros((4, 4), dtype=np.uint8))
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[1:3, 1:3] = 255
            _write_png(root / "png_images" / "1_test_1_.png", np.zeros((4, 4), dtype=np.uint8))
            _write_png(root / "png_masks" / "1_test_1_.png", mask)
            _write_csv(
                root / "stage_1_train_images.csv",
                [{"new_filename": "0_train_0_.png", "ImageId": "train-id", "has_pneumo": 0}],
            )
            _write_csv(
                root / "stage_1_test_images.csv",
                [{"new_filename": "1_test_1_.png", "ImageId": "test-id", "has_pneumo": 1}],
            )

            rows, conflicts, unmatched_images, unmatched_masks, summary = siim_manifest.build_manifest(
                root,
                hash_files=False,
            )

            self.assertEqual(len(rows), 2)
            self.assertFalse(conflicts)
            self.assertFalse(unmatched_images)
            self.assertFalse(unmatched_masks)
            self.assertEqual(summary["positive_count"], 1)
            self.assertEqual(summary["stage_counts"], {"stage1_test": 1, "stage1_train": 1})

    def test_ptx_manifest_pairs_site_case_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "PTX-498"
            site = root / "SiteA"
            _write_png(site / "1.1.img.png", np.zeros((4, 4), dtype=np.uint8))
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[0, 0] = 255
            _write_png(site / "1.2.mask.png", mask)
            _write_png(site / "1.3.merge.png", np.zeros((4, 4), dtype=np.uint8))
            (site / "1.4.img.nii.gz").write_bytes(b"nii")
            (site / "1.5.mask.nii.gz").write_bytes(b"nii")

            rows, conflicts, unmatched_images, unmatched_masks, empty_masks, summary = ptx_manifest.build_manifest(
                root,
                hash_files=False,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case_id"], "ptx498_sitea_1")
            self.assertTrue(rows[0]["mask_is_positive"])
            self.assertFalse(conflicts)
            self.assertFalse(unmatched_images)
            self.assertFalse(unmatched_masks)
            self.assertFalse(empty_masks)
            self.assertEqual(summary["site_counts"], {"SiteA": 1})

    def test_extract_ptx_wrapper_preserves_site_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ptx_zip = root / "PTX-498.zip"
            with zipfile.ZipFile(ptx_zip, "w") as zf:
                zf.writestr("PTX-498-v2-fix/SiteA/1.1.img.png", b"image")
                zf.writestr("PTX-498-v2-fix/SiteA/1.2.mask.png", b"mask")
                zf.writestr("older/SiteA/old.txt", b"old")

            rc = extract_raw.main(
                [
                    "--ptx-zip",
                    str(ptx_zip),
                    "--out-root",
                    str(root / "extracted"),
                    "--force",
                    "false",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue((root / "extracted" / "PTX-498" / "SiteA" / "1.1.img.png").is_file())
            self.assertFalse((root / "extracted" / "PTX-498" / "PTX-498-v2-fix").exists())
            self.assertFalse((root / "extracted" / "PTX-498" / "older").exists())

    def test_single_fx_prior_manifest_rejects_visual_head4_probability_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prob = root / "visual_overlays" / "teacher_model" / "official_siim_224" / "head_4" / "case001.png"
            _write_png(prob, np.zeros((4, 4), dtype=np.uint8))
            manifest = root / "fx_prior_manifest.csv"
            _write_csv(
                manifest,
                [
                    {
                        "case_id": "case001",
                        "probability_map_path": str(prob),
                        "aligned_probability_map_path": "",
                        "binary_mask_path": "",
                        "state_key": "teacher_model",
                        "preprocess_variant": "official_siim_224",
                        "head_key": "head_5",
                    }
                ],
            )

            payload = validate_fx.run_single_manifest(
                manifest,
                dataset_root=root,
                out_dir=root / "audit",
                strict=False,
                sample_size=1,
                repo_root=root,
                sample_seed=42,
                expected_state_key="teacher_model",
                expected_preprocess_variant="official_siim_224",
                expected_head_key="head_5",
            )

            self.assertEqual(payload["verdict"], "FAIL")
            self.assertGreater(payload["error_count"], 0)
            self.assertTrue(any("visual_overlays" in err or "head_4" in err for err in payload["errors"]))


if __name__ == "__main__":
    unittest.main()
