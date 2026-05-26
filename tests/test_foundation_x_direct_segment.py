"""
Lightweight validation tests for scripts/foundation_x_direct_segment.py.

These tests do NOT require the real 2.62 GiB Foundation X checkpoint. They
exercise:
  - the output guardrail
  - the local-run guardrail (CPU + no --max-cases is refused)
  - the case-listing / case-id normalization on a synthetic nnUNet_raw-like tree
  - per-case metric computation on hand-crafted prediction/target pairs
  - the threshold grid helper
  - the artifact directory layout produced by an end-to-end run with a
    stubbed FoundationXFullNet (no checkpoint, no real Swin-B)

Run:
    py -3 -m unittest tests.test_foundation_x_direct_segment -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Stub timm if missing so foundation_x_full importable
if "timm" not in sys.modules:
    sys.modules["timm"] = types.SimpleNamespace(
        create_model=lambda *a, **kw: None,
    )

import foundation_x_direct_segment as fxds  # type: ignore


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _write_grayscale_png(path: Path, size: int = 32, positive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.RandomState(0).randint(20, 200, (size, size), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(str(path))


def _write_label_png(path: Path, size: int = 32, positive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size, size), dtype=np.uint8)
    if positive:
        arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
    Image.fromarray(arr, mode="L").save(str(path))


def _make_synthetic_dataset_root(root: Path, n_pos: int = 1, n_neg: int = 1, size: int = 32) -> Path:
    images = root / "imagesTs"
    labels = root / "heldout_labelsTs"
    idx = 0
    for _ in range(n_pos):
        cid = f"siim_{idx:06d}"
        _write_grayscale_png(images / f"{cid}_0000.png", size=size)
        _write_label_png(labels / f"{cid}.png", size=size, positive=True)
        idx += 1
    for _ in range(n_neg):
        cid = f"siim_{idx:06d}"
        _write_grayscale_png(images / f"{cid}_0000.png", size=size)
        _write_label_png(labels / f"{cid}.png", size=size, positive=False)
        idx += 1
    return root


# ─── Guardrail tests ─────────────────────────────────────────────────────────


class TestGuardrails(unittest.TestCase):
    def test_output_guardrail_accepts_allowed_anchor(self):
        ok, _ = fxds.validate_output_guardrail(
            Path("artifacts/diagnostics/foundation_x_direct/dryrun_cpu"),
        )
        self.assertTrue(ok)

    def test_output_guardrail_rejects_outside_anchor(self):
        ok, why = fxds.validate_output_guardrail(Path("artifacts/runs/foo"))
        self.assertFalse(ok)
        self.assertIn("artifacts/diagnostics/foundation_x_direct", why)

    def test_output_guardrail_rejects_forbidden_segment(self):
        ok, _ = fxds.validate_output_guardrail(
            Path("artifacts/diagnostics/foundation_x_direct/nnunet_raw_subdir"),
        )
        self.assertFalse(ok)

    def test_local_run_guardrail_refuses_full_cpu_sweep(self):
        ok, why = fxds.validate_local_run_guardrail(device="cpu", max_cases=0)
        self.assertFalse(ok)
        self.assertIn("--max-cases", why)

    def test_local_run_guardrail_allows_cpu_with_max_cases(self):
        ok, _ = fxds.validate_local_run_guardrail(device="cpu", max_cases=2)
        self.assertTrue(ok)

    def test_local_run_guardrail_allows_cuda_without_max_cases(self):
        ok, _ = fxds.validate_local_run_guardrail(device="cuda", max_cases=0)
        self.assertTrue(ok)


# ─── Case ID normalization / dataset listing ────────────────────────────────


class TestCaseListing(unittest.TestCase):
    def test_normalize_case_id_strips_nnunet_suffix(self):
        self.assertEqual(fxds._normalize_case_id("siim_009074_0000"), "siim_009074")
        self.assertEqual(fxds._normalize_case_id("siim_009074_0000.png"), "siim_009074")
        self.assertEqual(fxds._normalize_case_id("siim_009074.png"), "siim_009074")
        self.assertEqual(fxds._normalize_case_id("siim_009074"), "siim_009074")

    def test_list_heldout_cases_matches_image_and_label(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_synthetic_dataset_root(Path(td), n_pos=2, n_neg=1)
            cases = fxds.list_heldout_cases(
                images_dir=root / "imagesTs",
                labels_dir=root / "heldout_labelsTs",
            )
        self.assertEqual(len(cases), 3)
        self.assertTrue(all(c["label_exists"] for c in cases))
        self.assertEqual(cases[0]["case_id"], "siim_000000")

    def test_select_cases_dry_run_picks_head_of_list(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_synthetic_dataset_root(Path(td), n_pos=3, n_neg=3)
            cases = fxds.list_heldout_cases(
                root / "imagesTs", root / "heldout_labelsTs",
            )
            picked = fxds.select_cases(cases, max_cases=2, seed=42)
        self.assertEqual([c["case_id"] for c in picked], ["siim_000000", "siim_000001"])


# ─── Threshold grid ─────────────────────────────────────────────────────────


class TestThresholdGrid(unittest.TestCase):
    def test_default_grid_has_19_steps(self):
        grid = fxds.threshold_grid(None)
        self.assertEqual(len(grid), 19)
        self.assertAlmostEqual(min(grid), 0.05)
        self.assertAlmostEqual(max(grid), 0.95)

    def test_custom_grid_is_sorted_and_unique(self):
        grid = fxds.threshold_grid([0.5, 0.5, 0.3, 0.7])
        self.assertEqual(grid, [0.3, 0.5, 0.7])

    def test_grid_rejects_out_of_range(self):
        grid = fxds.threshold_grid([0.0, 1.0, 0.5])
        self.assertEqual(grid, [0.5])


# ─── Per-case metrics ───────────────────────────────────────────────────────


class TestPerCaseMetrics(unittest.TestCase):
    def test_empty_empty_returns_ones(self):
        empty = np.zeros((8, 8), dtype=np.uint8)
        m = fxds.per_case_metrics(empty, empty)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["iou"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)

    def test_pred_empty_target_positive_returns_zero_dice(self):
        gt = np.zeros((8, 8), dtype=np.uint8)
        gt[2:6, 2:6] = 1
        pred = np.zeros((8, 8), dtype=np.uint8)
        m = fxds.per_case_metrics(pred, gt)
        self.assertEqual(m["dice"], 0.0)
        self.assertEqual(m["recall"], 0.0)

    def test_perfect_overlap(self):
        gt = np.zeros((8, 8), dtype=np.uint8)
        gt[2:6, 2:6] = 1
        m = fxds.per_case_metrics(gt, gt)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["iou"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)

    def test_aggregate_case_metrics_matches_baseline_shape(self):
        rows = [
            {  # positive case, perfect prediction
                "dice": 0.8, "iou": 0.7, "precision": 0.9, "recall": 0.85,
                "gt_foreground_pixels": 100, "pred_foreground_pixels": 100,
                "gt_is_positive": True, "pred_is_positive": True,
                "label_available": True,
            },
            {  # positive case, missed
                "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0,
                "gt_foreground_pixels": 100, "pred_foreground_pixels": 0,
                "gt_is_positive": True, "pred_is_positive": False,
                "label_available": True,
            },
            {  # negative case, correctly predicted negative
                "dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0,
                "gt_foreground_pixels": 0, "pred_foreground_pixels": 0,
                "gt_is_positive": False, "pred_is_positive": False,
                "label_available": True,
            },
            {  # negative case, false positive
                "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0,
                "gt_foreground_pixels": 0, "pred_foreground_pixels": 50,
                "gt_is_positive": False, "pred_is_positive": True,
                "label_available": True,
            },
        ]
        agg = fxds.aggregate_case_metrics(rows)
        self.assertEqual(agg["tp"], 1)
        self.assertEqual(agg["fn"], 1)
        self.assertEqual(agg["fp"], 1)
        self.assertEqual(agg["tn"], 1)
        self.assertAlmostEqual(agg["case_level_precision"], 0.5)
        self.assertAlmostEqual(agg["case_level_recall"], 0.5)
        self.assertAlmostEqual(agg["case_level_f1"], 0.5)
        self.assertAlmostEqual(agg["negative_case_false_positive_rate"], 0.5)
        self.assertAlmostEqual(agg["mean_dice_positive_cases"], 0.4)


# ─── Checkpoint-key detection utility ──────────────────────────────────────


class TestStateKeyDetection(unittest.TestCase):
    def test_load_checkpoint_state_dict_selects_named_key(self):
        with tempfile.TemporaryDirectory() as td:
            import torch

            ckpt_path = Path(td) / "fake.pth"
            torch.save(
                {
                    "model": {"backbone.0.dummy.weight": torch.zeros(1)},
                    "teacher_model": {"backbone.0.dummy.weight": torch.ones(1)},
                    "epoch": 896,
                },
                str(ckpt_path),
            )
            state, info = fxds.load_checkpoint_state_dict(ckpt_path, state_key="model")
        self.assertEqual(info["selected_state_key"], "model")
        self.assertIn("model", info["top_level_keys"])
        self.assertIn("teacher_model", info["top_level_keys"])
        self.assertEqual(info["state_dict_len"], 1)
        self.assertEqual(list(state.keys()), ["backbone.0.dummy.weight"])

    def test_load_checkpoint_state_dict_raises_on_missing_key(self):
        with tempfile.TemporaryDirectory() as td:
            import torch

            ckpt_path = Path(td) / "fake.pth"
            torch.save({"model": {"k": torch.zeros(1)}}, str(ckpt_path))
            with self.assertRaises(KeyError):
                fxds.load_checkpoint_state_dict(ckpt_path, state_key="teacher_model")


# ─── Output artifact emission with stubbed model ────────────────────────────


class _StubFoundationXFullNet:
    """Stub that mimics FoundationXFullNet's public surface without timm/Swin/PPN/FPN."""

    def __init__(self, img_size: int = 32, **_kw):
        self.img_size = int(img_size)

    def load_foundation_x_state_dict(self, state_dict):
        # Pretend everything routed cleanly to swin/heads.
        return {
            "missing": [],
            "unexpected": [],
            "route_stats": {
                "raw_total": len(state_dict),
                "raw_with_prefix": len(state_dict),
                "routed_swin": len(state_dict),
                "routed_segnorm": 0,
                "routed_locnorm": 0,
                "routed_ppn": 0,
                "routed_fpn": 0,
                "routed_heads": 0,
                "skipped_non_backbone": 0,
                "skipped_classification_heads": 0,
                "skipped_swin_head": 0,
                "skipped_other": 0,
            },
            "remapped_total": len(state_dict),
        }

    def to(self, device):
        return self

    def eval(self):
        return self

    def forward_segmentation(self, x, head_idx: int, upsample_to_input: bool = True):
        import torch

        b, _, h, w = x.shape
        # Deterministic non-trivial logits that depend on head_idx so the metric
        # values vary head-to-head and threshold-to-threshold.
        base = torch.zeros(b, 1, h, w, dtype=x.dtype, device=x.device)
        base[..., h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = float(head_idx + 1) * 0.1
        return base


class TestEndToEndStub(unittest.TestCase):
    def test_main_emits_required_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_synthetic_dataset_root(Path(td) / "ds", n_pos=1, n_neg=1, size=64)
            ckpt = Path(td) / "fake.pth"
            import torch
            torch.save({"model": {"backbone.0.dummy.weight": torch.zeros(1)}}, str(ckpt))
            out_dir = Path(td) / "artifacts" / "diagnostics" / "foundation_x_direct" / "dryrun_cpu"

            with patch.object(
                __import__("src.models.foundation_x_full", fromlist=["FoundationXFullNet"]),
                "FoundationXFullNet",
                _StubFoundationXFullNet,
            ):
                rc = fxds.main(
                    [
                        "--dataset-root", str(root),
                        "--checkpoint", str(ckpt),
                        "--state-key", "model",
                        "--heads", "0", "1",
                        "--device", "cpu",
                        "--max-cases", "2",
                        "--thresholds", "0.3", "0.5", "0.7",
                        "--img-size", "256",
                        "--out", str(out_dir),
                    ],
                )
            self.assertEqual(rc, 0, msg="main should succeed with stubbed model")
            self.assertTrue((out_dir / "summary.yaml").exists() or (out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "per_case_metrics.csv").exists())
            self.assertTrue((out_dir / "report.md").exists())
            self.assertTrue((out_dir / "run_metadata.json").exists())
            self.assertTrue((out_dir / "command.txt").exists())


if __name__ == "__main__":
    unittest.main()
