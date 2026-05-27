"""Tests for PR-10B manifest validator (scripts/validate_fx_prior_manifests.py).

These tests construct tiny synthetic manifests and PNG fixtures under a temp
directory, then exercise the validator's public API (`run`) and CLI (`main`).
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

# Ensure repo root and scripts/ are importable in the test process.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import validate_fx_prior_manifests as vfx  # noqa: E402


# --- fixture builders --------------------------------------------------------


def _make_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _build_dataset(
    root: Path,
    train_case_ids: list[str],
    test_case_ids: list[str],
    image_size: tuple[int, int] = (64, 64),
    prob_size: tuple[int, int] | None = None,
    degenerate_prob: bool = False,
    nonbinary_label: bool = False,
) -> tuple[Path, Path, Path]:
    """Create a minimal Dataset101-like layout + prior export under `root`.

    Returns (dataset_root, train_manifest_path, test_manifest_path).
    """
    dataset_root = root / "nnUNet_raw" / "Dataset101_Pneumothorax"
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"
    heldout = dataset_root / "heldout_labelsTs"
    prob_tr_dir = (
        root
        / "artifacts"
        / "diagnostics"
        / "foundation_x_official_direct"
        / "teacher_head5_official224_export_imagesTr"
        / "probability_maps"
        / "teacher_model"
        / "official_siim_224"
        / "head_5"
    )
    prob_ts_dir = (
        root
        / "artifacts"
        / "diagnostics"
        / "foundation_x_official_direct"
        / "teacher_head5_official224_export_full1602"
        / "probability_maps"
        / "teacher_model"
        / "official_siim_224"
        / "head_5"
    )
    binary_tr_dir = prob_tr_dir.parent.parent.parent / "binary_masks" / "teacher_model" / "official_siim_224" / "head_5"
    binary_ts_dir = prob_ts_dir.parent.parent.parent / "binary_masks" / "teacher_model" / "official_siim_224" / "head_5"
    for d in (images_tr, labels_tr, images_ts, heldout, prob_tr_dir, prob_ts_dir, binary_tr_dir, binary_ts_dir):
        d.mkdir(parents=True, exist_ok=True)

    H, W = image_size
    pH, pW = prob_size or (H, W)
    rng = np.random.default_rng(0)
    image_arr = (rng.integers(0, 256, size=(H, W))).astype(np.uint8)

    def label_arr(positive: bool) -> np.ndarray:
        a = np.zeros((H, W), dtype=np.uint8)
        if positive:
            a[H // 4 : H // 2, W // 4 : W // 2] = 255
        if nonbinary_label:
            a[0, 0] = 17  # not in {0, 1, 255}
        return a

    def prob_arr() -> np.ndarray:
        if degenerate_prob:
            return np.zeros((pH, pW), dtype=np.uint8)
        return (rng.integers(0, 256, size=(pH, pW))).astype(np.uint8)

    # train files
    for i, cid in enumerate(train_case_ids):
        is_pos = (i % 2 == 0)
        _make_png(images_tr / f"{cid}_0000.png", image_arr)
        _make_png(labels_tr / f"{cid}.png", label_arr(is_pos))
        _make_png(prob_tr_dir / f"{cid}.png", prob_arr())
        _make_png(binary_tr_dir / f"{cid}.png", (label_arr(is_pos) > 0).astype(np.uint8) * 255)
    # test files
    for i, cid in enumerate(test_case_ids):
        is_pos = (i % 2 == 0)
        _make_png(images_ts / f"{cid}_0000.png", image_arr)
        _make_png(heldout / f"{cid}.png", label_arr(is_pos))
        _make_png(prob_ts_dir / f"{cid}.png", prob_arr())
        _make_png(binary_ts_dir / f"{cid}.png", (label_arr(is_pos) > 0).astype(np.uint8) * 255)

    train_manifest = root / "train_manifest.csv"
    test_manifest = root / "test_manifest.csv"
    _write_manifest(train_manifest, train_case_ids, kind="train",
                    images_dir_rel=images_tr.relative_to(root),
                    labels_dir_rel=labels_tr.relative_to(root),
                    prob_dir_rel=prob_tr_dir.relative_to(root),
                    binary_dir_rel=binary_tr_dir.relative_to(root))
    _write_manifest(test_manifest, test_case_ids, kind="test",
                    images_dir_rel=images_ts.relative_to(root),
                    labels_dir_rel=heldout.relative_to(root),
                    prob_dir_rel=prob_ts_dir.relative_to(root),
                    binary_dir_rel=binary_ts_dir.relative_to(root))
    return dataset_root, train_manifest, test_manifest


def _write_manifest(
    path: Path,
    case_ids: list[str],
    *,
    kind: str,
    images_dir_rel: Path,
    labels_dir_rel: Path,
    prob_dir_rel: Path,
    binary_dir_rel: Path,
    split_value: str | None = None,
    provenance_override: dict[str, str] | None = None,
) -> None:
    split = split_value or ("train" if kind == "train" else "test")
    provenance = {
        "state_key": "teacher_model",
        "preprocess_variant": "official_siim_224",
        "head_key": "head_5",
    }
    if provenance_override:
        provenance.update(provenance_override)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=vfx.MANIFEST_COLUMNS)
        writer.writeheader()
        for i, cid in enumerate(case_ids):
            is_pos = (i % 2 == 0)
            writer.writerow({
                "case_id": cid,
                "split": split,
                # Manifests in PR-10A use Windows backslashes; reproduce that here
                # so the validator's path normalizer is exercised.
                "image_path": str(images_dir_rel / f"{cid}_0000.png").replace("/", "\\"),
                "label_path": str(labels_dir_rel / f"{cid}.png").replace("/", "\\"),
                "probability_map_path": str(prob_dir_rel / f"{cid}.png").replace("/", "\\"),
                "binary_mask_path": str(binary_dir_rel / f"{cid}.png").replace("/", "\\"),
                "state_key": provenance["state_key"],
                "preprocess_variant": provenance["preprocess_variant"],
                "head_key": provenance["head_key"],
                "threshold": "0.5",
                "gt_is_positive": "True" if is_pos else "False",
                "pred_is_positive": "False",
                "dice": "1.0",
                "iou": "1.0",
                "precision": "1.0",
                "recall": "1.0",
                "specificity": "1.0",
                "gt_foreground_pixels": "0",
                "pred_foreground_pixels": "0",
            })


def _check_by_name(payload: dict[str, Any], section: str, name: str) -> dict[str, Any]:
    checks = payload[section]["checks"] if section in {"train", "test"} else [payload["cross"]]
    for c in checks:
        if c["name"] == name:
            return c
    raise AssertionError(f"check {name!r} not found in {section}")


# --- tests -------------------------------------------------------------------


class TestValidateFxPriorManifests(unittest.TestCase):
    def _run(self, **overrides: Any) -> tuple[Path, dict[str, Any]]:
        tmp = Path(self._tmp.name)
        kwargs: dict[str, Any] = dict(
            train_manifest=self.train_manifest,
            test_manifest=self.test_manifest,
            dataset_root=self.dataset_root,
            out_dir=tmp / "out",
            repo_root=tmp,
            sample_size=8,
        )
        kwargs.update(overrides)
        payload = vfx.run(**kwargs)
        return kwargs["out_dir"], payload

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.dataset_root, self.train_manifest, self.test_manifest = _build_dataset(
            tmp,
            train_case_ids=[f"siim_{i:06d}" for i in range(1, 9)],
            test_case_ids=[f"siim_{i:06d}" for i in range(101, 105)],
        )

    # -- 1. happy path -------------------------------------------------------

    def test_happy_path_passes(self) -> None:
        out_dir, payload = self._run()
        self.assertEqual(payload["verdict"], "PASS", payload["failed_checks"])
        self.assertTrue((out_dir / "summary.json").exists())
        self.assertTrue((out_dir / "report.md").exists())
        self.assertTrue((out_dir / "manifest_audit.csv").exists())
        # report.md mentions verdict
        text = (out_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("PASS", text)
        # audit csv has one row per manifest row
        with (out_dir / "manifest_audit.csv").open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 8 + 4)

    def test_expected_row_count_match(self) -> None:
        _, payload = self._run(expected_train_rows=8, expected_test_rows=4)
        self.assertEqual(payload["verdict"], "PASS")
        chk = _check_by_name(payload, "train", "train_row_count")
        self.assertTrue(chk["passed"])
        self.assertEqual(chk["severity"], "error")

    def test_expected_row_count_mismatch_fails(self) -> None:
        _, payload = self._run(expected_train_rows=9999)
        self.assertEqual(payload["verdict"], "FAIL")
        self.assertIn("train:train_row_count", payload["failed_checks"])

    # -- 2. schema -----------------------------------------------------------

    def test_missing_column_fails(self) -> None:
        # Drop a required column by rewriting train manifest without it.
        with self.train_manifest.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        bad_cols = [c for c in vfx.MANIFEST_COLUMNS if c != "probability_map_path"]
        with self.train_manifest.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=bad_cols)
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r[c] for c in bad_cols})
        _, payload = self._run()
        chk = _check_by_name(payload, "train", "schema_header")
        self.assertFalse(chk["passed"])
        self.assertEqual(payload["verdict"], "FAIL")

    # -- 3. uniqueness -------------------------------------------------------

    def test_duplicate_case_id_in_train_fails(self) -> None:
        # Inject a duplicate row.
        with self.train_manifest.open("r", encoding="utf-8", newline="") as fh:
            text = fh.read()
        first_data_line = text.splitlines()[1]
        with self.train_manifest.open("a", encoding="utf-8") as fh:
            fh.write(first_data_line + "\n")
        _, payload = self._run()
        chk = _check_by_name(payload, "train", "train_case_id_unique")
        self.assertFalse(chk["passed"])
        self.assertGreaterEqual(chk["offenders_count"], 1)

    # -- 4. file existence ---------------------------------------------------

    def test_missing_image_fails(self) -> None:
        target = self.dataset_root / "imagesTr" / "siim_000001_0000.png"
        target.unlink()
        _, payload = self._run()
        chk = _check_by_name(payload, "train", "train_image_paths_exist")
        self.assertFalse(chk["passed"])
        self.assertIn("siim_000001", chk["offenders_sample"])
        self.assertEqual(payload["verdict"], "FAIL")

    def test_missing_prob_map_fails(self) -> None:
        target = (
            Path(self._tmp.name)
            / "artifacts"
            / "diagnostics"
            / "foundation_x_official_direct"
            / "teacher_head5_official224_export_imagesTr"
            / "probability_maps"
            / "teacher_model"
            / "official_siim_224"
            / "head_5"
            / "siim_000002.png"
        )
        target.unlink()
        _, payload = self._run()
        chk = _check_by_name(payload, "train", "train_probability_map_paths_exist")
        self.assertFalse(chk["passed"])
        self.assertEqual(payload["verdict"], "FAIL")

    # -- 5. provenance -------------------------------------------------------

    def test_wrong_head_key_fails(self) -> None:
        # Rewrite test manifest with head_key=head_4
        tmp = Path(self._tmp.name)
        _write_manifest(
            self.test_manifest,
            case_ids=[f"siim_{i:06d}" for i in range(101, 105)],
            kind="test",
            images_dir_rel=(self.dataset_root / "imagesTs").relative_to(tmp),
            labels_dir_rel=(self.dataset_root / "heldout_labelsTs").relative_to(tmp),
            prob_dir_rel=(tmp / "artifacts" / "diagnostics" / "foundation_x_official_direct"
                          / "teacher_head5_official224_export_full1602" / "probability_maps"
                          / "teacher_model" / "official_siim_224" / "head_5").relative_to(tmp),
            binary_dir_rel=(tmp / "artifacts" / "diagnostics" / "foundation_x_official_direct"
                            / "teacher_head5_official224_export_full1602" / "binary_masks"
                            / "teacher_model" / "official_siim_224" / "head_5").relative_to(tmp),
            provenance_override={"head_key": "head_4"},
        )
        _, payload = self._run()
        chk = _check_by_name(payload, "test", "test_provenance")
        self.assertFalse(chk["passed"])

    # -- 6. split values -----------------------------------------------------

    def test_train_split_must_be_train(self) -> None:
        tmp = Path(self._tmp.name)
        _write_manifest(
            self.train_manifest,
            case_ids=[f"siim_{i:06d}" for i in range(1, 9)],
            kind="train",
            images_dir_rel=(self.dataset_root / "imagesTr").relative_to(tmp),
            labels_dir_rel=(self.dataset_root / "labelsTr").relative_to(tmp),
            prob_dir_rel=(tmp / "artifacts" / "diagnostics" / "foundation_x_official_direct"
                          / "teacher_head5_official224_export_imagesTr" / "probability_maps"
                          / "teacher_model" / "official_siim_224" / "head_5").relative_to(tmp),
            binary_dir_rel=(tmp / "artifacts" / "diagnostics" / "foundation_x_official_direct"
                            / "teacher_head5_official224_export_imagesTr" / "binary_masks"
                            / "teacher_model" / "official_siim_224" / "head_5").relative_to(tmp),
            split_value="test",
        )
        _, payload = self._run()
        chk = _check_by_name(payload, "train", "train_split_value")
        self.assertFalse(chk["passed"])

    def test_test_split_eval_allowed(self) -> None:
        tmp = Path(self._tmp.name)
        _write_manifest(
            self.test_manifest,
            case_ids=[f"siim_{i:06d}" for i in range(101, 105)],
            kind="test",
            images_dir_rel=(self.dataset_root / "imagesTs").relative_to(tmp),
            labels_dir_rel=(self.dataset_root / "heldout_labelsTs").relative_to(tmp),
            prob_dir_rel=(tmp / "artifacts" / "diagnostics" / "foundation_x_official_direct"
                          / "teacher_head5_official224_export_full1602" / "probability_maps"
                          / "teacher_model" / "official_siim_224" / "head_5").relative_to(tmp),
            binary_dir_rel=(tmp / "artifacts" / "diagnostics" / "foundation_x_official_direct"
                            / "teacher_head5_official224_export_full1602" / "binary_masks"
                            / "teacher_model" / "official_siim_224" / "head_5").relative_to(tmp),
            split_value="eval",
        )
        _, payload = self._run()
        chk = _check_by_name(payload, "test", "test_split_value")
        self.assertTrue(chk["passed"], chk)

    # -- 7. leakage ----------------------------------------------------------

    def test_train_with_imagesTs_segment_fails(self) -> None:
        # Manually rewrite one row's image_path to point into imagesTs/.
        text = self.train_manifest.read_text(encoding="utf-8")
        text = text.replace("\\imagesTr\\", "\\imagesTs\\", 1)
        self.train_manifest.write_text(text, encoding="utf-8")
        _, payload = self._run()
        chk = _check_by_name(payload, "train", "train_leakage_imagesTs")
        self.assertFalse(chk["passed"])
        self.assertEqual(payload["verdict"], "FAIL")

    def test_train_with_heldout_labelsTs_fails(self) -> None:
        text = self.train_manifest.read_text(encoding="utf-8")
        text = text.replace("\\labelsTr\\", "\\heldout_labelsTs\\", 1)
        self.train_manifest.write_text(text, encoding="utf-8")
        _, payload = self._run()
        chk = _check_by_name(payload, "train", "train_leakage_heldout_labelsTs")
        self.assertFalse(chk["passed"])

    def test_test_with_imagesTr_fails_by_default(self) -> None:
        text = self.test_manifest.read_text(encoding="utf-8")
        text = text.replace("\\imagesTs\\", "\\imagesTr\\", 1)
        self.test_manifest.write_text(text, encoding="utf-8")
        _, payload = self._run()
        chk = _check_by_name(payload, "test", "test_leakage_imagesTr")
        self.assertFalse(chk["passed"])

    def test_test_with_imagesTr_allowed_with_flag(self) -> None:
        text = self.test_manifest.read_text(encoding="utf-8")
        text = text.replace("\\imagesTs\\", "\\imagesTr\\", 1)
        self.test_manifest.write_text(text, encoding="utf-8")
        _, payload = self._run(allow_imagestr_in_test=True)
        chk = _check_by_name(payload, "test", "test_leakage_imagesTr")
        self.assertTrue(chk["passed"])

    def test_case_id_overlap_train_test_fails(self) -> None:
        # Make one test case id collide with a train case id.
        tmp = Path(self._tmp.name)
        overlap_id = "siim_000001"  # also present in train
        _write_manifest(
            self.test_manifest,
            case_ids=[overlap_id, "siim_000102"],
            kind="test",
            images_dir_rel=(self.dataset_root / "imagesTs").relative_to(tmp),
            labels_dir_rel=(self.dataset_root / "heldout_labelsTs").relative_to(tmp),
            prob_dir_rel=(tmp / "artifacts" / "diagnostics" / "foundation_x_official_direct"
                          / "teacher_head5_official224_export_full1602" / "probability_maps"
                          / "teacher_model" / "official_siim_224" / "head_5").relative_to(tmp),
            binary_dir_rel=(tmp / "artifacts" / "diagnostics" / "foundation_x_official_direct"
                            / "teacher_head5_official224_export_full1602" / "binary_masks"
                            / "teacher_model" / "official_siim_224" / "head_5").relative_to(tmp),
        )
        # Create the matching test files for both ids so existence passes.
        H, W = 64, 64
        arr = np.zeros((H, W), dtype=np.uint8)
        _make_png(self.dataset_root / "imagesTs" / f"{overlap_id}_0000.png", arr)
        _make_png(self.dataset_root / "heldout_labelsTs" / f"{overlap_id}.png", arr)
        prob_dir = (tmp / "artifacts" / "diagnostics" / "foundation_x_official_direct"
                    / "teacher_head5_official224_export_full1602" / "probability_maps"
                    / "teacher_model" / "official_siim_224" / "head_5")
        binary_dir = (tmp / "artifacts" / "diagnostics" / "foundation_x_official_direct"
                      / "teacher_head5_official224_export_full1602" / "binary_masks"
                      / "teacher_model" / "official_siim_224" / "head_5")
        _make_png(prob_dir / f"{overlap_id}.png", arr)
        _make_png(binary_dir / f"{overlap_id}.png", arr)
        _make_png(self.dataset_root / "imagesTs" / "siim_000102_0000.png", arr)
        _make_png(self.dataset_root / "heldout_labelsTs" / "siim_000102.png", arr)
        _make_png(prob_dir / "siim_000102.png", arr)
        _make_png(binary_dir / "siim_000102.png", arr)
        _, payload = self._run()
        self.assertFalse(payload["cross"]["passed"])
        self.assertEqual(payload["verdict"], "FAIL")

    # -- 8. prob-map sanity --------------------------------------------------

    def test_degenerate_prob_map_warns_only_by_default(self) -> None:
        # Rebuild dataset with degenerate prob maps.
        tmp = Path(self._tmp.name)
        # wipe and rebuild
        import shutil
        shutil.rmtree(tmp / "artifacts")
        shutil.rmtree(tmp / "nnUNet_raw")
        (tmp / "train_manifest.csv").unlink(missing_ok=True)
        (tmp / "test_manifest.csv").unlink(missing_ok=True)
        self.dataset_root, self.train_manifest, self.test_manifest = _build_dataset(
            tmp,
            train_case_ids=[f"siim_{i:06d}" for i in range(1, 9)],
            test_case_ids=[f"siim_{i:06d}" for i in range(101, 105)],
            degenerate_prob=True,
        )
        _, payload = self._run()
        chk = _check_by_name(payload, "train", "train_prob_nondegenerate_sampled")
        self.assertFalse(chk["passed"])
        self.assertEqual(chk["severity"], "warning")
        # Verdict should still pass (warning, not error) in non-strict mode
        self.assertEqual(payload["verdict"], "PASS")

    def test_degenerate_prob_map_fails_in_strict_mode(self) -> None:
        tmp = Path(self._tmp.name)
        import shutil
        shutil.rmtree(tmp / "artifacts")
        shutil.rmtree(tmp / "nnUNet_raw")
        (tmp / "train_manifest.csv").unlink(missing_ok=True)
        (tmp / "test_manifest.csv").unlink(missing_ok=True)
        self.dataset_root, self.train_manifest, self.test_manifest = _build_dataset(
            tmp,
            train_case_ids=[f"siim_{i:06d}" for i in range(1, 9)],
            test_case_ids=[f"siim_{i:06d}" for i in range(101, 105)],
            degenerate_prob=True,
        )
        _, payload = self._run(strict=True)
        self.assertEqual(payload["verdict"], "FAIL")

    def test_nonbinary_label_fails(self) -> None:
        tmp = Path(self._tmp.name)
        import shutil
        shutil.rmtree(tmp / "artifacts")
        shutil.rmtree(tmp / "nnUNet_raw")
        (tmp / "train_manifest.csv").unlink(missing_ok=True)
        (tmp / "test_manifest.csv").unlink(missing_ok=True)
        self.dataset_root, self.train_manifest, self.test_manifest = _build_dataset(
            tmp,
            train_case_ids=[f"siim_{i:06d}" for i in range(1, 9)],
            test_case_ids=[f"siim_{i:06d}" for i in range(101, 105)],
            nonbinary_label=True,
        )
        _, payload = self._run()
        chk = _check_by_name(payload, "train", "train_label_binarity_sampled")
        self.assertFalse(chk["passed"])

    def test_prob_size_mismatch_warning_only(self) -> None:
        tmp = Path(self._tmp.name)
        import shutil
        shutil.rmtree(tmp / "artifacts")
        shutil.rmtree(tmp / "nnUNet_raw")
        (tmp / "train_manifest.csv").unlink(missing_ok=True)
        (tmp / "test_manifest.csv").unlink(missing_ok=True)
        self.dataset_root, self.train_manifest, self.test_manifest = _build_dataset(
            tmp,
            train_case_ids=[f"siim_{i:06d}" for i in range(1, 9)],
            test_case_ids=[f"siim_{i:06d}" for i in range(101, 105)],
            image_size=(64, 64),
            prob_size=(32, 32),
        )
        _, payload = self._run()
        # image_label alignment still ok; prob size differs - recorded in sampled_prob_sizes
        self.assertEqual(payload["verdict"], "PASS")
        self.assertIn("32x32", payload["train"]["sampled_prob_sizes"])

    # -- 9. CLI exit codes ---------------------------------------------------

    def test_cli_main_exit_zero_on_pass(self) -> None:
        tmp = Path(self._tmp.name)
        out = tmp / "cli_out"
        rc = vfx.main([
            "--train-manifest", str(self.train_manifest),
            "--test-manifest", str(self.test_manifest),
            "--dataset-root", str(self.dataset_root),
            "--out", str(out),
            "--repo-root", str(tmp),
            "--sample-size", "4",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue((out / "summary.json").exists())

    def test_cli_main_exit_one_on_fail(self) -> None:
        # Inject leakage and confirm CLI returns 1.
        text = self.train_manifest.read_text(encoding="utf-8")
        text = text.replace("\\imagesTr\\", "\\imagesTs\\", 1)
        self.train_manifest.write_text(text, encoding="utf-8")
        tmp = Path(self._tmp.name)
        rc = vfx.main([
            "--train-manifest", str(self.train_manifest),
            "--test-manifest", str(self.test_manifest),
            "--dataset-root", str(self.dataset_root),
            "--out", str(tmp / "cli_out_fail"),
            "--repo-root", str(tmp),
            "--sample-size", "4",
        ])
        self.assertEqual(rc, 1)

    def test_cli_missing_manifest_returns_two(self) -> None:
        tmp = Path(self._tmp.name)
        rc = vfx.main([
            "--train-manifest", str(tmp / "does_not_exist.csv"),
            "--test-manifest", str(self.test_manifest),
            "--dataset-root", str(self.dataset_root),
            "--out", str(tmp / "cli_out_missing"),
            "--repo-root", str(tmp),
        ])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
