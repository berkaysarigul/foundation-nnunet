"""
Tests for scripts/evaluate_hybrid_heldout.py.

Light tests only: CLI parsing, guardrail, synthetic confusion-matrix /
aggregate metric math, and BLOCKED-summary writing when paths missing.
No HybridFoundationUNet load, no checkpoint required.
"""

from __future__ import annotations

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

import evaluate_hybrid_heldout as ehh


class TestCliParsing(unittest.TestCase):
    def test_required_args_enforced(self):
        with self.assertRaises(SystemExit):
            ehh.parse_args([])

    def test_default_threshold_and_img_size(self):
        args = ehh.parse_args(
            [
                "--images_dir", "/tmp/imagesTs",
                "--labels_dir", "/tmp/heldout_labelsTs",
                "--hybrid_checkpoint", "/tmp/ckpt.pth",
                "--foundation_checkpoint", "/tmp/foundation.pth",
                "--output_dir", "artifacts/diagnostics/hybrid_heldout_eval/test",
            ],
        )
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(args.img_size, 512)
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.save_prob_maps)
        self.assertFalse(args.save_logit_maps)
        self.assertTrue(args.save_pred_masks)
        self.assertFalse(args.strict)

    def test_threshold_override(self):
        args = ehh.parse_args(
            [
                "--images_dir", "/tmp/imagesTs",
                "--labels_dir", "/tmp/heldout_labelsTs",
                "--hybrid_checkpoint", "/tmp/ckpt.pth",
                "--foundation_checkpoint", "/tmp/foundation.pth",
                "--output_dir", "artifacts/diagnostics/hybrid_heldout_eval/test",
                "--threshold", "0.4",
            ],
        )
        self.assertEqual(args.threshold, 0.4)

    def test_img_size_choices_enforced(self):
        with self.assertRaises(SystemExit):
            ehh.parse_args(
                [
                    "--images_dir", "/tmp/imagesTs",
                    "--labels_dir", "/tmp/heldout_labelsTs",
                    "--hybrid_checkpoint", "/tmp/ckpt.pth",
                    "--foundation_checkpoint", "/tmp/foundation.pth",
                    "--output_dir", "artifacts/diagnostics/hybrid_heldout_eval/test",
                    "--img_size", "999",
                ],
            )


class TestOutputGuardrail(unittest.TestCase):
    def test_accepts_allowed_anchor(self):
        ok, msg = ehh.validate_output_guardrail(
            Path("artifacts/diagnostics/hybrid_heldout_eval/pr8_best_thr05"),
        )
        self.assertTrue(ok, msg)
        self.assertEqual(msg, "")

    def test_rejects_artifacts_runs(self):
        ok, msg = ehh.validate_output_guardrail(Path("artifacts/runs/smoke"))
        self.assertFalse(ok)
        self.assertIn("artifacts/diagnostics/hybrid_heldout_eval", msg)

    def test_rejects_nnunet_raw(self):
        ok, msg = ehh.validate_output_guardrail(
            Path("nnUNet_raw/Dataset101_Pneumothorax/anywhere"),
        )
        self.assertFalse(ok)

    def test_rejects_pr8_controlled_train_dir(self):
        ok, msg = ehh.validate_output_guardrail(
            Path("artifacts/diagnostics/hybrid_controlled_short_train/larger_lr3e5_1200step_bnfix"),
        )
        self.assertFalse(ok)

    def test_rejects_pr4_smoke_dir(self):
        ok, msg = ehh.validate_output_guardrail(
            Path("artifacts/diagnostics/foundation_x_smoke"),
        )
        self.assertFalse(ok)


class TestPerCaseMetrics(unittest.TestCase):
    def test_both_empty_returns_one(self):
        zero = np.zeros((8, 8), dtype=np.float32)
        m = ehh.per_case_metrics(zero, zero)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["iou"], 1.0)
        self.assertEqual(m["pred_foreground_pixels"], 0)
        self.assertEqual(m["gt_foreground_pixels"], 0)

    def test_perfect_overlap(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        gt[2:5, 2:5] = 1.0
        m = ehh.per_case_metrics(gt, gt)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["iou"], 1.0)
        self.assertEqual(m["pred_foreground_pixels"], 9)
        self.assertEqual(m["gt_foreground_pixels"], 9)

    def test_no_overlap_yields_zero(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        gt[0:2, 0:2] = 1.0
        pred = np.zeros((8, 8), dtype=np.float32)
        pred[6:8, 6:8] = 1.0
        m = ehh.per_case_metrics(pred, gt)
        self.assertEqual(m["dice"], 0.0)
        self.assertEqual(m["iou"], 0.0)

    def test_partial_overlap(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        gt[1:4, 1:4] = 1.0
        pred = np.zeros((8, 8), dtype=np.float32)
        pred[2:5, 2:5] = 1.0
        m = ehh.per_case_metrics(pred, gt)
        self.assertGreater(m["dice"], 0.0)
        self.assertLess(m["dice"], 1.0)
        self.assertEqual(m["intersection_pixels"], 4)


class TestAggregateMetrics(unittest.TestCase):
    def _mk(self, *, gt_pos: bool, pred_pos: bool, dice: float = 0.5, gt_px: int = 100, pred_px: int = 100, iou: float = 0.4, precision: float = 0.6, recall: float = 0.4) -> dict:
        return {
            "label_available": True,
            "gt_is_positive": gt_pos,
            "pred_is_positive": pred_pos,
            "gt_foreground_pixels": gt_px,
            "pred_foreground_pixels": pred_px,
            "total_pixels": 262144,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
        }

    def test_confusion_matrix_counts(self):
        rows = [
            self._mk(gt_pos=True, pred_pos=True, dice=0.5),   # TP
            self._mk(gt_pos=True, pred_pos=True, dice=0.6),   # TP
            self._mk(gt_pos=True, pred_pos=False, dice=0.0, pred_px=0),  # FN
            self._mk(gt_pos=False, pred_pos=True, dice=0.0, gt_px=0),    # FP
            self._mk(gt_pos=False, pred_pos=False, dice=1.0, gt_px=0, pred_px=0),  # TN
            self._mk(gt_pos=False, pred_pos=False, dice=1.0, gt_px=0, pred_px=0),  # TN
        ]
        agg = ehh.aggregate_case_metrics(rows)
        self.assertEqual(agg["tp"], 2)
        self.assertEqual(agg["fn"], 1)
        self.assertEqual(agg["fp"], 1)
        self.assertEqual(agg["tn"], 2)
        self.assertEqual(agg["gt_positive_cases"], 3)
        self.assertEqual(agg["gt_negative_cases"], 3)
        self.assertEqual(agg["pred_positive_cases"], 3)
        self.assertAlmostEqual(agg["case_level_precision"], 2.0 / 3.0)
        self.assertAlmostEqual(agg["case_level_recall"], 2.0 / 3.0)
        self.assertAlmostEqual(agg["case_level_specificity"], 2.0 / 3.0)
        self.assertAlmostEqual(agg["case_level_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(agg["negative_case_false_positive_rate"], 1.0 / 3.0)

    def test_mean_dice_detected_excludes_undetected_positives(self):
        rows = [
            self._mk(gt_pos=True, pred_pos=True, dice=0.6),
            self._mk(gt_pos=True, pred_pos=True, dice=0.8),
            self._mk(gt_pos=True, pred_pos=False, dice=0.0, pred_px=0),
            self._mk(gt_pos=False, pred_pos=False, dice=1.0, gt_px=0, pred_px=0),
        ]
        agg = ehh.aggregate_case_metrics(rows)
        # mean dice over positives = (0.6 + 0.8 + 0.0) / 3 = 0.4666...
        self.assertAlmostEqual(agg["mean_dice_positive_cases"], (0.6 + 0.8 + 0.0) / 3.0)
        # mean dice over detected positives only = (0.6 + 0.8) / 2 = 0.7
        self.assertAlmostEqual(agg["mean_dice_detected_positives_only"], 0.7)
        self.assertEqual(agg["detected_positive_count"], 2)

    def test_all_negative_cases(self):
        rows = [
            self._mk(gt_pos=False, pred_pos=False, dice=1.0, gt_px=0, pred_px=0),
            self._mk(gt_pos=False, pred_pos=False, dice=1.0, gt_px=0, pred_px=0),
        ]
        agg = ehh.aggregate_case_metrics(rows)
        self.assertEqual(agg["gt_positive_cases"], 0)
        self.assertEqual(agg["tp"], 0)
        self.assertEqual(agg["fn"], 0)
        self.assertEqual(agg["case_level_recall"], 0.0)  # safe-div
        self.assertEqual(agg["negative_case_false_positive_rate"], 0.0)


class TestNormalizeCaseId(unittest.TestCase):
    def test_strip_suffix_variants(self):
        self.assertEqual(ehh._normalize_case_id("siim_009074_0000"), "siim_009074")
        self.assertEqual(ehh._normalize_case_id("siim_009074_0000.png"), "siim_009074")
        self.assertEqual(ehh._normalize_case_id("siim_009074"), "siim_009074")
        self.assertEqual(ehh._normalize_case_id("siim_009074.png"), "siim_009074")


class TestListHeldoutCases(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.images_dir = self.tmpdir / "imagesTs"
        self.labels_dir = self.tmpdir / "heldout_labelsTs"
        self.images_dir.mkdir()
        self.labels_dir.mkdir()
        for idx in (1, 2, 3):
            (self.images_dir / f"siim_{idx:06d}_0000.png").write_bytes(b"")
        (self.labels_dir / "siim_000001.png").write_bytes(b"")
        # Skip label for case 2 to test label_exists flag.
        (self.labels_dir / "siim_000003.png").write_bytes(b"")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_lists_cases_sorted_and_label_existence(self):
        cases = ehh.list_heldout_cases(self.images_dir, self.labels_dir)
        ids = [c["case_id"] for c in cases]
        self.assertEqual(ids, ["siim_000001", "siim_000002", "siim_000003"])
        self.assertTrue(cases[0]["label_exists"])
        self.assertFalse(cases[1]["label_exists"])
        self.assertTrue(cases[2]["label_exists"])


class TestRunEvaluationBlockedPath(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # Construct an output dir that satisfies the guardrail anchor.
        self.output_dir = (
            self.tmpdir / "artifacts" / "diagnostics" / "hybrid_heldout_eval" / "test_blocked"
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_blocked_when_inputs_missing(self):
        argv = [
            "--images_dir", str(self.tmpdir / "nope_imagesTs"),
            "--labels_dir", str(self.tmpdir / "nope_labels"),
            "--hybrid_checkpoint", str(self.tmpdir / "nope_ckpt.pth"),
            "--foundation_checkpoint", str(self.tmpdir / "nope_foundation.pth"),
            "--output_dir", str(self.output_dir),
        ]
        code = ehh.main(argv)
        self.assertEqual(code, 2)
        summary_path = self.output_dir / "hybrid_heldout_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertEqual(summary["exit_code"], 2)
        self.assertTrue(summary["failures"])

    def test_blocked_when_output_under_forbidden_path(self):
        argv = [
            "--images_dir", str(self.tmpdir / "nope_imagesTs"),
            "--labels_dir", str(self.tmpdir / "nope_labels"),
            "--hybrid_checkpoint", str(self.tmpdir / "nope_ckpt.pth"),
            "--foundation_checkpoint", str(self.tmpdir / "nope_foundation.pth"),
            "--output_dir", str(self.tmpdir / "artifacts" / "runs" / "smoke"),
        ]
        code = ehh.main(argv)
        self.assertEqual(code, 2)


class TestSanitizeForJson(unittest.TestCase):
    def test_nan_becomes_none(self):
        self.assertIsNone(ehh._sanitize_for_json(float("nan")))
        self.assertIsNone(ehh._sanitize_for_json(float("inf")))

    def test_numpy_types_serialised(self):
        self.assertEqual(ehh._sanitize_for_json(np.int64(7)), 7)
        self.assertEqual(ehh._sanitize_for_json(np.float64(1.5)), 1.5)
        self.assertTrue(ehh._sanitize_for_json(np.bool_(True)))


class TestTinySubgroup(unittest.TestCase):
    def test_buckets_split_into_terciles(self):
        rows = [
            {"label_available": True, "gt_is_positive": True, "pred_is_positive": True,
             "dice": 0.1, "gt_foreground_pixels": 50, "total_pixels": 1000},
            {"label_available": True, "gt_is_positive": True, "pred_is_positive": True,
             "dice": 0.3, "gt_foreground_pixels": 200, "total_pixels": 1000},
            {"label_available": True, "gt_is_positive": True, "pred_is_positive": True,
             "dice": 0.6, "gt_foreground_pixels": 400, "total_pixels": 1000},
            {"label_available": True, "gt_is_positive": False, "pred_is_positive": False,
             "dice": 1.0, "gt_foreground_pixels": 0, "total_pixels": 1000},
        ]
        out = ehh._compute_tiny_subgroup(rows)
        self.assertIn("tiny", out)
        self.assertIn("medium", out)
        self.assertIn("large", out)
        # 3 positives split into 3 terciles -> each bucket non-empty.
        self.assertGreaterEqual(out["tiny"]["count"] + out["medium"]["count"] + out["large"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
