"""Focused tests for PR-11B PTX-498 manifest audit."""

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

import audit_ptx498_manifest as audit  # noqa: E402
import build_ptx498_manifest as manifest_builder  # noqa: E402
from pr11_utils import write_csv  # noqa: E402


def _write_png(path: Path, arr: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr is None:
        arr = np.zeros((4, 4), dtype=np.uint8)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _fake_manifest(root: Path, manifest_path: Path) -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 255
    _write_png(root / "SiteA" / "1.1.img.png", np.full((4, 4), 80, dtype=np.uint8))
    _write_png(root / "SiteA" / "1.2.mask.png", mask)
    _write_png(root / "SiteA" / "2.1.img.png", np.full((4, 4), 120, dtype=np.uint8))
    _write_png(root / "SiteA" / "2.2.mask.png", np.zeros((4, 4), dtype=np.uint8))
    rows, conflicts, *_ = manifest_builder.build_manifest(root, hash_files=False)
    assert conflicts == []
    write_csv(manifest_path, rows, manifest_builder.MANIFEST_COLUMNS)


def test_audit_writes_positive_empty_and_site_overlays() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        manifest_path = work / "ptx498_manifest.csv"
        _fake_manifest(work / "PTX-498", manifest_path)

        rc = audit.main(
            [
                "--manifest",
                str(manifest_path),
                "--out",
                str(work / "audit"),
                "--sample-count",
                "4",
                "--seed",
                "42",
                "--expected-row-count",
                "2",
            ]
        )

        assert rc == 0
        assert (work / "audit" / "summary.yaml").is_file()
        assert list((work / "audit" / "positive_samples").glob("*.png"))
        assert list((work / "audit" / "empty_mask_samples").glob("*.png"))
        assert list((work / "audit" / "site_samples").rglob("*.png"))


def test_audit_detects_missing_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        manifest_path = work / "ptx498_manifest.csv"
        write_csv(
            manifest_path,
            [
                {
                    "case_id": "ptx498_sitea_missing",
                    "source_dataset": "PTX498",
                    "site": "SiteA",
                    "image_path": str(work / "missing.img.png"),
                    "mask_path": str(work / "missing.mask.png"),
                    "merge_path": "",
                    "nii_image_path": "",
                    "nii_mask_path": "",
                    "width": 0,
                    "height": 0,
                    "gt_foreground_pixels": 0,
                    "mask_is_positive": False,
                    "image_hash": "",
                    "mask_hash": "",
                }
            ],
            manifest_builder.MANIFEST_COLUMNS,
        )

        audit_rows, summary, conflicts = audit.audit_manifest(manifest_path)

        assert summary["status"] == "FAIL"
        assert len(audit_rows) == 1
        assert "missing_image_path" in audit_rows[0]["issues"]
        assert "missing_mask_path" in audit_rows[0]["issues"]
        assert conflicts
