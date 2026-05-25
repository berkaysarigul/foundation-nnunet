"""
Tests for scripts/controlled_train_hybrid_short.py.

All tests run with a stubbed hybrid model, so no real Foundation X checkpoint
or timm installation is required.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Allow src.models.backbone import when timm is missing.
if "timm" not in sys.modules:
    sys.modules["timm"] = types.SimpleNamespace(create_model=lambda *args, **kwargs: None)

import controlled_train_hybrid_short as cts


def _save_gray_png(path: Path, size: int = 128, positive: bool | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if positive is None:
        arr = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    else:
        arr = np.zeros((size, size), dtype=np.uint8)
        if positive:
            arr[8:28, 8:28] = 255
    Image.fromarray(arr, mode="L").save(str(path))


def _build_dataset_fixture(
    root: Path,
    image_dir_name: str = "imagesTr",
    label_dir_name: str = "labelsTr",
    num_positive: int = 20,
    num_negative: int = 20,
) -> tuple[Path, Path]:
    images_dir = root / image_dir_name
    labels_dir = root / label_dir_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(num_positive):
        case_id = f"siim_{100000 + idx:06d}"
        _save_gray_png(images_dir / f"{case_id}_0000.png", size=128, positive=None)
        _save_gray_png(labels_dir / f"{case_id}.png", size=128, positive=True)

    for idx in range(num_negative):
        case_id = f"siim_{200000 + idx:06d}"
        _save_gray_png(images_dir / f"{case_id}_0000.png", size=128, positive=None)
        _save_gray_png(labels_dir / f"{case_id}.png", size=128, positive=False)

    return images_dir, labels_dir


def _build_fake_checkpoint(root: Path) -> Path:
    checkpoint = root / "fake_foundation_x.pth"
    checkpoint.write_bytes(b"fake-checkpoint")
    return checkpoint


class DummyFoundationX(torch.nn.Module):
    def __init__(self, frozen: bool):
        super().__init__()
        self.frozen = frozen
        self.backbone = torch.nn.Identity()
        self.fx0 = torch.nn.Conv2d(1, 128, kernel_size=1, bias=False)
        self.fx1 = torch.nn.Conv2d(1, 256, kernel_size=1, bias=False)
        self.fx2 = torch.nn.Conv2d(1, 512, kernel_size=1, bias=False)
        self.fx3 = torch.nn.Conv2d(1, 1024, kernel_size=1, bias=False)
        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [
            self.fx0(F.avg_pool2d(x, 4)),
            self.fx1(F.avg_pool2d(x, 8)),
            self.fx2(F.avg_pool2d(x, 16)),
            self.fx3(F.avg_pool2d(x, 32)),
        ]


class DummyHybrid(torch.nn.Module):
    def __init__(
        self,
        backbone_checkpoint: str,
        in_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 64,
        frozen_backbone: bool = True,
        img_size: int = 512,
    ):
        super().__init__()
        del backbone_checkpoint, in_channels, num_classes, base_filters, img_size
        self.frozen_backbone = frozen_backbone
        self.foundation_x = DummyFoundationX(frozen=frozen_backbone)

        self.p0 = torch.nn.Conv2d(128, 8, kernel_size=1)
        self.p1 = torch.nn.Conv2d(256, 8, kernel_size=1)
        self.p2 = torch.nn.Conv2d(512, 8, kernel_size=1)
        self.p3 = torch.nn.Conv2d(1024, 8, kernel_size=1)
        self.merge = torch.nn.Conv2d(32, 8, kernel_size=1)
        self.final = torch.nn.Conv2d(8, 1, kernel_size=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_backbone:
            self.foundation_x.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.foundation_x(x)
        u0 = F.interpolate(self.p0(fx[0]), size=x.shape[2:], mode="nearest")
        u1 = F.interpolate(self.p1(fx[1]), size=x.shape[2:], mode="nearest")
        u2 = F.interpolate(self.p2(fx[2]), size=x.shape[2:], mode="nearest")
        u3 = F.interpolate(self.p3(fx[3]), size=x.shape[2:], mode="nearest")
        merged = self.merge(torch.cat([u0, u1, u2, u3], dim=1))
        logits = self.final(merged)
        # Keep the default stub path non-degenerate for PASS-path diagnostics.
        ramp = torch.linspace(-0.75, 0.75, x.shape[-1], device=x.device).view(1, 1, 1, -1)
        logits = logits + ramp
        return torch.sigmoid(logits)


class DummyHybridEmptyPred(DummyHybrid):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return torch.zeros_like(out)


class DummyHybridFullPred(DummyHybrid):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return torch.ones_like(out)


class DummyHybridTransientFullPred(DummyHybrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_forward_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if not self.training:
            self.eval_forward_count += int(x.shape[0])
            if self.eval_forward_count <= 16:
                return torch.ones_like(out)
        return out


class TestCliParsing(unittest.TestCase):
    def test_defaults(self):
        args = cts.parse_args([])
        self.assertEqual(args.img_size, 512)
        self.assertEqual(args.num_train_cases, 256)
        self.assertEqual(args.num_val_cases, 128)
        self.assertEqual(args.max_steps, 200)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.val_every, 25)
        self.assertEqual(args.bce_loss_weight, 1.0)
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.strict)
        self.assertFalse(args.unfrozen_backbone)
        self.assertFalse(args.save_diagnostic_checkpoint)

    def test_checkpoint_alias(self):
        args = cts.parse_args(["--checkpoint", "ckpt.pth"])
        self.assertEqual(args.checkpoint, Path("ckpt.pth"))

    def test_flag_parsing(self):
        args = cts.parse_args(
            [
                "--strict",
                "--no_visuals",
                "--unfrozen_backbone",
                "--save_diagnostic_checkpoint",
                "--dry_run",
                "--bce_loss_weight",
                "0.1",
            ]
        )
        self.assertTrue(args.strict)
        self.assertTrue(args.no_visuals)
        self.assertTrue(args.unfrozen_backbone)
        self.assertTrue(args.save_diagnostic_checkpoint)
        self.assertTrue(args.dry_run)
        self.assertEqual(args.bce_loss_weight, 0.1)


class TestSelectionDeterminism(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.input_dir, self.labels_dir = _build_dataset_fixture(self.tmpdir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_deterministic_train_val_selection(self):
        s1 = cts.select_train_val_cases(self.input_dir, self.labels_dir, 32, 16)
        s2 = cts.select_train_val_cases(self.input_dir, self.labels_dir, 32, 16)
        self.assertEqual([c["case_id"] for c in s1["train"]], [c["case_id"] for c in s2["train"]])
        self.assertEqual([c["case_id"] for c in s1["val"]], [c["case_id"] for c in s2["val"]])

    def test_train_order_alternates_by_class(self):
        selected = cts.select_train_val_cases(self.input_dir, self.labels_dir, 32, 16)
        train_flags = [case["is_positive"] for case in selected["train"]]
        self.assertEqual(train_flags[:8], [True, False, True, False, True, False, True, False])

    def test_train_val_are_disjoint(self):
        selected = cts.select_train_val_cases(self.input_dir, self.labels_dir, 32, 16)
        train_ids = {c["case_id"] for c in selected["train"]}
        val_ids = {c["case_id"] for c in selected["val"]}
        self.assertEqual(len(train_ids & val_ids), 0)


class TestNoHeldoutUsage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.images_ts, self.labels_ts = _build_dataset_fixture(
            self.tmpdir / "heldout",
            image_dir_name="imagesTs",
            label_dir_name="heldout_labelsTs",
            num_positive=4,
            num_negative=4,
        )
        self.checkpoint = _build_fake_checkpoint(self.tmpdir)
        self.output_dir = self.tmpdir / "artifacts" / "diagnostics" / "hybrid_controlled_short_train"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rejects_imagesTs_and_heldout_labelsTs(self):
        code = cts.main(
            [
                "--input_dir",
                str(self.images_ts),
                "--labels_dir",
                str(self.labels_ts),
                "--foundation_checkpoint",
                str(self.checkpoint),
                "--output_dir",
                str(self.output_dir),
                "--dry_run",
            ]
        )
        self.assertEqual(code, 2)
        summary = json.loads(
            (self.output_dir / "hybrid_controlled_short_train_summary.json").read_text(
                encoding="utf-8"
            )
        )
        text = json.dumps(summary).lower()
        self.assertIn("imagests", text)
        self.assertIn("heldout_labelsts", text)


class TestOptimizerFiltering(unittest.TestCase):
    def test_frozen_backbone_excluded_from_optimizer(self):
        model = DummyHybrid(backbone_checkpoint="unused.pth", frozen_backbone=True, img_size=256)
        optimizer, trainable_names, frozen_backbone_count = cts.build_optimizer_for_trainable_params(
            model,
            lr=1e-4,
            weight_decay=0.01,
        )
        self.assertIsNotNone(optimizer)
        self.assertGreater(frozen_backbone_count, 0)
        self.assertTrue(trainable_names)
        self.assertTrue(all(not name.startswith("foundation_x.") for name in trainable_names))


class TestValidationSchedule(unittest.TestCase):
    def test_schedule_includes_zero_interval_and_final(self):
        sched = cts._validation_schedule(25, 5)
        self.assertEqual(sched, [0, 5, 10, 15, 20, 25])


class TestStubControlledRun(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.input_dir, self.labels_dir = _build_dataset_fixture(self.tmpdir, num_positive=24, num_negative=24)
        self.checkpoint = _build_fake_checkpoint(self.tmpdir)
        self.output_dir = self.tmpdir / "artifacts" / "diagnostics" / "hybrid_controlled_short_train"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, model_cls, extra_args: list[str] | None = None) -> dict[str, Any]:
        argv = [
            "--input_dir",
            str(self.input_dir),
            "--labels_dir",
            str(self.labels_dir),
            "--foundation_checkpoint",
            str(self.checkpoint),
            "--output_dir",
            str(self.output_dir),
            "--img_size",
            "256",
            "--device",
            "cpu",
            "--num_train_cases",
            "32",
            "--num_val_cases",
            "16",
            "--max_steps",
            "25",
            "--batch_size",
            "1",
            "--lr",
            "1e-4",
            "--val_every",
            "5",
            "--no_visuals",
        ]
        if extra_args:
            argv.extend(extra_args)

        with patch("controlled_train_hybrid_short.HybridFoundationUNet", model_cls):
            code = cts.main(argv)

        summary_path = self.output_dir / "hybrid_controlled_short_train_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["_exit_code_from_main"] = code
        return summary

    def test_stubbed_run_passes(self):
        summary = self._run(DummyHybrid)
        self.assertEqual(summary["_exit_code_from_main"], 0)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["audit_name"], "hybrid_controlled_short_train")

    def test_case_counts_and_exact_steps(self):
        summary = self._run(DummyHybrid)
        self.assertEqual(summary["setup"]["num_train_cases_processed"], 32)
        self.assertEqual(summary["setup"]["num_val_cases_processed"], 16)
        self.assertEqual(summary["training"]["steps_completed"], 25)
        self.assertEqual(len(summary["training"]["loss_total_per_step"]), 25)

    def test_validation_scheduling(self):
        summary = self._run(DummyHybrid)
        self.assertEqual(summary["validation"]["scheduled_steps"], [0, 5, 10, 15, 20, 25])
        self.assertEqual(summary["validation"]["executed_steps"], [0, 5, 10, 15, 20, 25])
        self.assertTrue(summary["pass_criteria"]["validation_schedule_complete"])

    def test_finite_losses_and_gradients(self):
        summary = self._run(DummyHybrid)
        self.assertTrue(all(np.isfinite(v) for v in summary["training"]["loss_total_per_step"]))
        self.assertTrue(summary["gradients"]["frozen_backbone_no_grad"])
        self.assertTrue(summary["gradients"]["trainable_gradients_present"])
        self.assertTrue(summary["gradients"]["trainable_gradients_finite"])

    def test_csv_outputs_written(self):
        self._run(DummyHybrid)
        train_csv = self.output_dir / "train_steps.csv"
        val_csv = self.output_dir / "validation_steps.csv"
        grad_csv = self.output_dir / "gradient_stats.csv"
        sel_csv = self.output_dir / "selection_log.csv"
        self.assertTrue(train_csv.exists())
        self.assertTrue(val_csv.exists())
        self.assertTrue(grad_csv.exists())
        self.assertTrue(sel_csv.exists())

        with train_csv.open(encoding="utf-8") as handle:
            train_rows = list(csv.DictReader(handle))
        with val_csv.open(encoding="utf-8") as handle:
            val_rows = list(csv.DictReader(handle))
        self.assertEqual(len(train_rows), 25)
        self.assertEqual(len(val_rows), 6)
        for key in (
            "prob_min",
            "prob_max",
            "prob_mean",
            "prob_std",
            "logits_min",
            "logits_max",
            "logits_mean",
            "logits_std",
            "prob_foreground_ratio_thr05",
        ):
            self.assertIn(key, val_rows[0])
            self.assertTrue(np.isfinite(float(val_rows[0][key])))

    def test_bce_loss_weight_changes_total_loss_but_logs_unweighted_bce(self):
        summary = self._run(DummyHybrid, extra_args=["--bce_loss_weight", "0.1"])
        self.assertEqual(summary["setup"]["bce_loss_weight"], 0.1)
        with (self.output_dir / "train_steps.csv").open(encoding="utf-8") as handle:
            first_row = next(csv.DictReader(handle))
        expected = float(first_row["loss_dice_focal"]) + 0.1 * float(first_row["loss_bce_with_logits"])
        self.assertAlmostEqual(float(first_row["loss_total"]), expected, places=5)
        self.assertGreater(float(first_row["loss_bce_with_logits"]), 0.0)

    def test_checkpoint_saved_only_if_enabled(self):
        summary = self._run(DummyHybrid)
        self.assertFalse(summary["outputs"]["diagnostic_checkpoint_saved"])

        summary2 = self._run(DummyHybrid, extra_args=["--save_diagnostic_checkpoint"])
        self.assertTrue(summary2["outputs"]["diagnostic_checkpoint_saved"])
        ckpt_path = Path(summary2["outputs"]["diagnostic_checkpoint_path"])
        self.assertTrue(ckpt_path.exists())

    def test_degeneracy_detection_empty_predictions(self):
        summary = self._run(DummyHybridEmptyPred, extra_args=["--strict"])
        self.assertEqual(summary["_exit_code_from_main"], 2)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertTrue(summary["degeneracy"]["persistent_collapse_from_start"])

    def test_degeneracy_detection_full_predictions(self):
        summary = self._run(DummyHybridFullPred, extra_args=["--strict"])
        self.assertEqual(summary["_exit_code_from_main"], 2)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertTrue(summary["degeneracy"]["persistent_collapse_from_start"])

    def test_strict_transient_degeneracy_is_partial_pass(self):
        summary = self._run(DummyHybridTransientFullPred, extra_args=["--strict"])
        self.assertEqual(summary["_exit_code_from_main"], 0)
        self.assertEqual(summary["status"], "PARTIAL_PASS")
        self.assertTrue(summary["degeneracy"]["any_degenerate_validation_point"])
        self.assertFalse(summary["degeneracy"]["persistent_collapse_from_start"])
        self.assertIn(
            "Degenerate validation points observed, but no persistent collapse; classified as PARTIAL_PASS.",
            summary["warnings"],
        )

    def test_summary_schema_and_d039(self):
        summary = self._run(DummyHybrid)
        for key in (
            "setup",
            "selection",
            "training",
            "validation",
            "gradients",
            "degeneracy",
            "pass_criteria",
            "outputs",
        ):
            self.assertIn(key, summary)
        self.assertIn(summary["status"], {"PASS", "PARTIAL_PASS", "BLOCKED"})
        raw_json = json.dumps(summary).lower()
        self.assertNotIn("hausdorff", raw_json)
        for key in (
            "prob_min_per_step",
            "prob_max_per_step",
            "prob_mean_per_step",
            "prob_std_per_step",
            "logits_min_per_step",
            "logits_max_per_step",
            "logits_mean_per_step",
            "logits_std_per_step",
            "prob_foreground_ratio_thr05_per_step",
        ):
            self.assertIn(key, summary["validation"])
            self.assertEqual(len(summary["validation"][key]), len(summary["validation"]["executed_steps"]))
        report_text = (self.output_dir / "hybrid_controlled_short_train_report.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("hausdorff", report_text.lower())


class TestGuardrails(unittest.TestCase):
    def test_forbidden_output_root_guardrail(self):
        ok, msg = cts._validate_output_guardrail(Path("artifacts/runs/smoke"))
        self.assertFalse(ok)
        self.assertIn("forbidden", msg.lower())


if __name__ == "__main__":
    unittest.main()
