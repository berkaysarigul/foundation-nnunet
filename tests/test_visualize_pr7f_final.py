"""
Tests for scripts/visualize_pr7f_final.py.

These tests stay light-weight: they cover CLI parsing, selection logic,
output-path computation, helper math, and the BLOCKED path for missing
inputs. They never load the HybridFoundationUNet model and never touch
the (non-existent locally) PR-7F diagnostic checkpoint. The script is
intended to be executed in Colab against real Drive paths.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import visualize_pr7f_final as vpf


def _write_selection_log(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "case_id",
        "is_positive",
        "label_max",
        "selection_bucket",
        "foreground_pixels",
        "total_pixels",
        "foreground_ratio",
        "image_path",
        "label_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


class TestCliParsing(unittest.TestCase):
    def test_required_args_enforced(self):
        with self.assertRaises(SystemExit):
            vpf.parse_args([])

    def test_full_args_parsed(self):
        args = vpf.parse_args(
            [
                "--run_folder", "/tmp/run",
                "--checkpoint", "/tmp/ckpt.pth",
                "--selection_log", "/tmp/sel.csv",
                "--images_dir", "/tmp/images",
                "--labels_dir", "/tmp/labels",
                "--foundation_checkpoint", "/tmp/foundation.pth",
                "--img_size", "512",
                "--device", "cpu",
                "--num_positive", "4",
                "--num_negative", "4",
                "--step_label", "0400",
            ],
        )
        self.assertEqual(args.run_folder, Path("/tmp/run"))
        self.assertEqual(args.img_size, 512)
        self.assertEqual(args.num_positive, 4)
        self.assertEqual(args.num_negative, 4)
        self.assertEqual(args.step_label, "0400")
        self.assertEqual(args.device, "cpu")
        self.assertFalse(args.strict)

    def test_img_size_choices_enforced(self):
        with self.assertRaises(SystemExit):
            vpf.parse_args(
                [
                    "--run_folder", "/tmp/run",
                    "--checkpoint", "/tmp/ckpt.pth",
                    "--selection_log", "/tmp/sel.csv",
                    "--images_dir", "/tmp/images",
                    "--labels_dir", "/tmp/labels",
                    "--foundation_checkpoint", "/tmp/foundation.pth",
                    "--img_size", "777",
                ],
            )


class TestOutputSubdir(unittest.TestCase):
    def test_default_subdir_uses_step_label(self):
        args = vpf.parse_args(
            [
                "--run_folder", "/tmp/run",
                "--checkpoint", "/tmp/ckpt.pth",
                "--selection_log", "/tmp/sel.csv",
                "--images_dir", "/tmp/images",
                "--labels_dir", "/tmp/labels",
                "--foundation_checkpoint", "/tmp/foundation.pth",
                "--step_label", "0400",
            ],
        )
        self.assertEqual(vpf.output_subdir(args), Path("visuals") / "val_step_0400")

    def test_explicit_subdir_override(self):
        args = vpf.parse_args(
            [
                "--run_folder", "/tmp/run",
                "--checkpoint", "/tmp/ckpt.pth",
                "--selection_log", "/tmp/sel.csv",
                "--images_dir", "/tmp/images",
                "--labels_dir", "/tmp/labels",
                "--foundation_checkpoint", "/tmp/foundation.pth",
                "--step_label", "0400",
                "--output_subdir", "custom/visuals/final",
            ],
        )
        self.assertEqual(vpf.output_subdir(args), Path("custom/visuals/final"))


class TestNormalizeCaseId(unittest.TestCase):
    def test_strip_suffix_variants(self):
        self.assertEqual(vpf._normalize_case_id("siim_000004_0000"), "siim_000004")
        self.assertEqual(vpf._normalize_case_id("siim_000004_0000.png"), "siim_000004")
        self.assertEqual(vpf._normalize_case_id("siim_000004"), "siim_000004")
        self.assertEqual(vpf._normalize_case_id("siim_000004.png"), "siim_000004")


class TestCaseSelection(unittest.TestCase):
    def test_diverse_picks_across_foreground_ratio(self):
        rows = [
            {"split": "val", "case_id": "p1", "is_positive": "True", "foreground_ratio": "0.001"},
            {"split": "val", "case_id": "p2", "is_positive": "True", "foreground_ratio": "0.005"},
            {"split": "val", "case_id": "p3", "is_positive": "True", "foreground_ratio": "0.01"},
            {"split": "val", "case_id": "p4", "is_positive": "True", "foreground_ratio": "0.02"},
            {"split": "val", "case_id": "p5", "is_positive": "True", "foreground_ratio": "0.05"},
            {"split": "val", "case_id": "p6", "is_positive": "True", "foreground_ratio": "0.08"},
            {"split": "val", "case_id": "n1", "is_positive": "False", "foreground_ratio": "0.0"},
            {"split": "val", "case_id": "n2", "is_positive": "False", "foreground_ratio": "0.0"},
            {"split": "val", "case_id": "n3", "is_positive": "False", "foreground_ratio": "0.0"},
            {"split": "val", "case_id": "n4", "is_positive": "False", "foreground_ratio": "0.0"},
            {"split": "val", "case_id": "n5", "is_positive": "False", "foreground_ratio": "0.0"},
            {"split": "train", "case_id": "train_skip", "is_positive": "True", "foreground_ratio": "0.005"},
        ]
        picked = vpf.pick_diverse_cases(rows, num_positive=4, num_negative=4)
        ids = [c["case_id"] for c in picked]
        self.assertEqual(len(ids), 8)
        self.assertNotIn("train_skip", ids)
        positives = [c for c in picked if vpf._coerce_bool(c["is_positive"])]
        negatives = [c for c in picked if not vpf._coerce_bool(c["is_positive"])]
        self.assertEqual(len(positives), 4)
        self.assertEqual(len(negatives), 4)
        # smallest and largest foreground ratios must be picked at the extremes
        self.assertEqual(positives[0]["case_id"], "p1")
        self.assertEqual(positives[-1]["case_id"], "p6")

    def test_falls_back_when_foreground_ratio_missing(self):
        rows = [
            {"split": "val", "case_id": "p1", "is_positive": "True"},
            {"split": "val", "case_id": "p2", "is_positive": "True"},
            {"split": "val", "case_id": "n1", "is_positive": "False"},
            {"split": "val", "case_id": "n2", "is_positive": "False"},
        ]
        picked = vpf.pick_diverse_cases(rows, num_positive=2, num_negative=2)
        ids = [c["case_id"] for c in picked]
        self.assertEqual(set(ids), {"p1", "p2", "n1", "n2"})

    def test_respects_availability_caps(self):
        rows = [
            {"split": "val", "case_id": "p1", "is_positive": "True", "foreground_ratio": "0.01"},
            {"split": "val", "case_id": "n1", "is_positive": "False", "foreground_ratio": "0.0"},
        ]
        picked = vpf.pick_diverse_cases(rows, num_positive=4, num_negative=4)
        ids = sorted(c["case_id"] for c in picked)
        self.assertEqual(ids, ["n1", "p1"])

    def test_excludes_train_split(self):
        rows = [
            {"split": "train", "case_id": "t1", "is_positive": "True", "foreground_ratio": "0.01"},
            {"split": "val", "case_id": "p1", "is_positive": "True", "foreground_ratio": "0.005"},
        ]
        picked = vpf.pick_diverse_cases(rows, num_positive=2, num_negative=0)
        self.assertEqual([c["case_id"] for c in picked], ["p1"])


class TestNormalizeUint8(unittest.TestCase):
    def test_uniform_input(self):
        arr = np.full((4, 4), 0.5, dtype=np.float32)
        out = vpf._normalize_to_uint8(arr)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (4, 4))

    def test_spread_input(self):
        arr = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        out = vpf._normalize_to_uint8(arr)
        self.assertGreater(int(out.max()) - int(out.min()), 100)


class TestPerImageDice(unittest.TestCase):
    def test_empty_both_match_yields_one(self):
        zero = np.zeros((8, 8), dtype=np.float32)
        self.assertEqual(vpf._per_image_dice(zero, zero), 1.0)

    def test_pred_empty_target_positive_yields_zero(self):
        pred = np.zeros((8, 8), dtype=np.float32)
        gt = np.zeros((8, 8), dtype=np.float32)
        gt[2:5, 2:5] = 1.0
        self.assertEqual(vpf._per_image_dice(pred, gt), 0.0)

    def test_perfect_overlap_yields_one(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        gt[1:4, 1:4] = 1.0
        self.assertEqual(vpf._per_image_dice(gt, gt), 1.0)

    def test_partial_overlap_between_zero_and_one(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        gt[1:4, 1:4] = 1.0
        pred = np.zeros((8, 8), dtype=np.float32)
        pred[2:5, 2:5] = 1.0
        value = vpf._per_image_dice(pred, gt)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)


class TestOverlay(unittest.TestCase):
    def test_overlay_colors_present(self):
        image = (np.ones((6, 6), dtype=np.uint8) * 80)
        gt = np.zeros((6, 6), dtype=np.uint8)
        gt[0:2, :] = 255
        pred = np.zeros((6, 6), dtype=np.uint8)
        pred[1:3, :] = 255
        overlay = vpf._build_overlay(image, gt, pred)
        self.assertEqual(overlay.shape, (6, 6, 3))
        # gt-only row (row 0) → reddish
        self.assertGreater(int(overlay[0, 0, 0]), int(overlay[0, 0, 1]))
        # pred-only row (row 2) → greenish
        self.assertGreater(int(overlay[2, 0, 1]), int(overlay[2, 0, 0]))
        # overlap row (row 1) → yellowish (red + green dominant, blue low)
        self.assertGreater(int(overlay[1, 0, 0]), int(overlay[1, 0, 2]))
        self.assertGreater(int(overlay[1, 0, 1]), int(overlay[1, 0, 2]))


class TestValidateInputs(unittest.TestCase):
    def test_missing_paths_reported(self):
        args = vpf.parse_args(
            [
                "--run_folder", "/nonexistent/run",
                "--checkpoint", "/nonexistent/ckpt.pth",
                "--selection_log", "/nonexistent/sel.csv",
                "--images_dir", "/nonexistent/images",
                "--labels_dir", "/nonexistent/labels",
                "--foundation_checkpoint", "/nonexistent/foundation.pth",
            ],
        )
        failures = vpf._validate_inputs(args)
        self.assertGreaterEqual(len(failures), 6)
        text = "\n".join(failures).lower()
        for fragment in ("run_folder", "checkpoint", "selection_log", "images_dir", "labels_dir", "foundation_checkpoint"):
            self.assertIn(fragment, text)


class TestRunVisualizationBlockedWhenInputsMissing(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.run_folder = self.tmpdir / "run"
        self.run_folder.mkdir(parents=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_blocked_summary_written_when_paths_missing(self):
        argv = [
            "--run_folder", str(self.run_folder),
            "--checkpoint", str(self.tmpdir / "nope_ckpt.pth"),
            "--selection_log", str(self.tmpdir / "nope_sel.csv"),
            "--images_dir", str(self.tmpdir / "nope_images"),
            "--labels_dir", str(self.tmpdir / "nope_labels"),
            "--foundation_checkpoint", str(self.tmpdir / "nope_foundation.pth"),
            "--step_label", "0400",
        ]
        code = vpf.main(argv)
        self.assertEqual(code, 2)
        summary_path = self.run_folder / "visuals" / "val_step_0400" / "final_visualization_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertEqual(summary["exit_code"], 2)
        self.assertTrue(summary["failures"])


class TestSelectionLogReader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reads_full_round_trip(self):
        sel = self.tmpdir / "sel.csv"
        _write_selection_log(
            sel,
            [
                {"split": "val", "case_id": "p1", "is_positive": "True", "foreground_ratio": "0.01"},
                {"split": "val", "case_id": "n1", "is_positive": "False", "foreground_ratio": "0.0"},
            ],
        )
        rows = vpf._read_selection_log(sel)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["case_id"], "p1")


class TestPathResolution(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.images_dir = self.tmpdir / "imagesTr"
        self.labels_dir = self.tmpdir / "labelsTr"
        self.images_dir.mkdir()
        self.labels_dir.mkdir()
        (self.images_dir / "siim_000004_0000.png").write_bytes(b"")
        (self.labels_dir / "siim_000004.png").write_bytes(b"")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resolves_image_with_suffix(self):
        img, lbl = vpf._resolve_image_label_paths("siim_000004", self.images_dir, self.labels_dir)
        self.assertEqual(img.name, "siim_000004_0000.png")
        self.assertEqual(lbl.name, "siim_000004.png")


if __name__ == "__main__":
    unittest.main()
