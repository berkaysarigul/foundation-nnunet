"""Focused tests for PR-11B PTX-498 manifest building."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_ptx498_manifest as manifest_builder  # noqa: E402


def _write_png(path: Path, arr: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr is None:
        arr = np.zeros((4, 4), dtype=np.uint8)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def test_manifest_pairs_pngs_and_records_merge_and_nii_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "PTX-498"
        site = root / "SiteA"
        _write_png(site / "7.1.img.png")
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        _write_png(site / "7.2.mask.png", mask)
        _write_png(site / "7.3.merge.png")
        (site / "7.4.img.nii.gz").write_bytes(b"nii-image")
        (site / "7.5.mask.nii.gz").write_bytes(b"nii-mask")

        rows, conflicts, unmatched_images, unmatched_masks, empty_masks, summary = manifest_builder.build_manifest(
            root,
            hash_files=False,
        )

        assert len(rows) == 1
        assert conflicts == []
        assert unmatched_images == []
        assert unmatched_masks == []
        assert empty_masks == []
        row = rows[0]
        assert row["case_id"] == "ptx498_sitea_7"
        assert row["site"] == "SiteA"
        assert row["merge_path"].endswith("7.3.merge.png")
        assert row["nii_image_path"].endswith("7.4.img.nii.gz")
        assert row["nii_mask_path"].endswith("7.5.mask.nii.gz")
        assert row["gt_foreground_pixels"] == 4
        assert row["mask_is_positive"] is True
        assert summary["positive_count"] == 1


def test_manifest_detects_unmatched_images_and_masks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "PTX-498"
        _write_png(root / "SiteA" / "1.1.img.png")
        _write_png(root / "SiteA" / "2.2.mask.png")

        rows, conflicts, unmatched_images, unmatched_masks, empty_masks, summary = manifest_builder.build_manifest(
            root,
            hash_files=False,
        )

        assert rows == []
        assert empty_masks == []
        assert len(unmatched_images) == 1
        assert len(unmatched_masks) == 1
        assert {row["conflict_type"] for row in conflicts} == {
            "unmatched_image_missing_mask",
            "unmatched_mask_missing_image",
        }
        assert summary["unmatched_images"] == 1
        assert summary["unmatched_masks"] == 1
        assert summary["conflicts"] == 2


def test_manifest_records_empty_masks_without_assuming_positive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "PTX-498"
        _write_png(root / "SiteB" / "3.1.img.png")
        _write_png(root / "SiteB" / "3.2.mask.png", np.zeros((4, 4), dtype=np.uint8))

        rows, conflicts, unmatched_images, unmatched_masks, empty_masks, summary = manifest_builder.build_manifest(
            root,
            hash_files=False,
        )

        assert len(rows) == 1
        assert conflicts == []
        assert unmatched_images == []
        assert unmatched_masks == []
        assert len(empty_masks) == 1
        assert rows[0]["mask_is_positive"] is False
        assert summary["empty_mask_count"] == 1
