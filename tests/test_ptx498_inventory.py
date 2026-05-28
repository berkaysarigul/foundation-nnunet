"""Focused tests for PR-11B PTX-498 inventory."""

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

import inventory_ptx498 as inventory  # noqa: E402


def _write_png(path: Path, arr: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr is None:
        arr = np.zeros((4, 4), dtype=np.uint8)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def test_inventory_detects_site_directories_and_counts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "PTX-498"
        for site in ["SiteA", "SiteB", "SiteC"]:
            _write_png(root / site / "1.1.img.png")
            _write_png(root / site / "1.2.mask.png")
            _write_png(root / site / "1.3.merge.png")
            (root / site / "1.4.img.nii.gz").write_bytes(b"nii-image")
            (root / site / "1.5.mask.nii.gz").write_bytes(b"nii-mask")

        summary = inventory.build_inventory(root)

        assert summary["site_dirs"] == ["SiteA", "SiteB", "SiteC"]
        assert summary["counts"]["png_images"] == 3
        assert summary["counts"]["png_masks"] == 3
        assert summary["counts"]["merge_pngs"] == 3
        assert summary["counts"]["nii_images"] == 3
        assert summary["counts"]["nii_masks"] == 3
        assert summary["every_image_has_matching_mask"] is True
        assert summary["merge_files_exist"] is True
        assert summary["first_10_cases"][0]["case_id"] == "ptx498_sitea_1"
