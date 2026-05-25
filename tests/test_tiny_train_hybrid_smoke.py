"""
Tests for scripts/tiny_train_hybrid_smoke.py.

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

import tiny_train_hybrid_smoke as tts


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
    num_positive: int = 8,
    num_negative: int = 8,
) -> tuple[Path, Path]:
    images_dir = root / "imagesTr"
    labels_dir = root / "labelsTr"
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
        return torch.sigmoid(logits)


class TestCliParsing(unittest.TestCase):
    def test_defaults(self):
        args = tts.parse_args([])
        self.assertEqual(args.img_size, 512)
        self.assertEqual(args.num_train_cases, 8)
        self.assertEqual(args.num_val_cases, 4)
        self.assertEqual(args.max_steps, 5)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.strict)
        self.assertFalse(args.unfrozen_backbone)

    def test_checkpoint_alias(self):
        args = tts.parse_args(["--checkpoint", "ckpt.pth"])
        self.assertEqual(args.checkpoint, Path("ckpt.pth"))

    def test_flag_parsing(self):
        args = tts.parse_args(["--strict", "--no_visuals", "--unfrozen_backbone", "--dry_run"])
        self.assertTrue(args.strict)
        self.assertTrue(args.no_visuals)
        self.assertTrue(args.unfrozen_backbone)
        self.assertTrue(args.dry_run)


class TestSelectionDeterminism(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.input_dir, self.labels_dir = _build_dataset_fixture(self.tmpdir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_deterministic_train_val_selection(self):
        s1 = tts.select_train_val_cases(self.input_dir, self.labels_dir, 8, 4)
        s2 = tts.select_train_val_cases(self.input_dir, self.labels_dir, 8, 4)
        self.assertEqual([c["case_id"] for c in s1["train"]], [c["case_id"] for c in s2["train"]])
        self.assertEqual([c["case_id"] for c in s1["val"]], [c["case_id"] for c in s2["val"]])

    def test_expected_pos_neg_balance(self):
        selected = tts.select_train_val_cases(self.input_dir, self.labels_dir, 8, 4)
        self.assertEqual(selected["counts"]["train_total"], 8)
        self.assertEqual(selected["counts"]["val_total"], 4)
        self.assertGreaterEqual(selected["counts"]["train_positive"], 4)
        self.assertGreaterEqual(selected["counts"]["train_negative"], 4)
        self.assertGreaterEqual(selected["counts"]["val_positive"], 2)
        self.assertGreaterEqual(selected["counts"]["val_negative"], 2)


class TestOptimizerFiltering(unittest.TestCase):
    def test_frozen_backbone_excluded_from_optimizer(self):
        model = DummyHybrid(backbone_checkpoint="unused.pth", frozen_backbone=True, img_size=256)
        optimizer, trainable_names, frozen_backbone_count = tts.build_optimizer_for_trainable_params(
            model,
            lr=1e-4,
            weight_decay=0.01,
        )
        self.assertIsNotNone(optimizer)
        self.assertGreater(frozen_backbone_count, 0)
        self.assertTrue(trainable_names)
        self.assertTrue(all(not name.startswith("foundation_x.") for name in trainable_names))


class TestStubTinyTrainingRun(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.input_dir, self.labels_dir = _build_dataset_fixture(self.tmpdir, num_positive=10, num_negative=10)
        self.checkpoint = _build_fake_checkpoint(self.tmpdir)
        self.output_dir = (
            self.tmpdir / "artifacts" / "diagnostics" / "hybrid_tiny_train_smoke"
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, extra_args: list[str] | None = None) -> dict[str, Any]:
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
            "8",
            "--num_val_cases",
            "4",
            "--max_steps",
            "5",
            "--batch_size",
            "1",
            "--lr",
            "1e-4",
            "--no_visuals",
        ]
        if extra_args:
            argv.extend(extra_args)

        with patch("tiny_train_hybrid_smoke.HybridFoundationUNet", DummyHybrid):
            code = tts.main(argv)

        summary_path = self.output_dir / "hybrid_tiny_train_smoke_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["_exit_code_from_main"] = code
        return summary

    def test_stubbed_run_passes(self):
        summary = self._run(extra_args=["--strict"])
        self.assertEqual(summary["_exit_code_from_main"], 0)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["audit_name"], "hybrid_tiny_train_smoke")

    def test_train_val_case_counts(self):
        summary = self._run()
        self.assertEqual(summary["setup"]["num_train_cases_processed"], 8)
        self.assertEqual(summary["setup"]["num_val_cases_processed"], 4)

    def test_exact_step_count_and_losses(self):
        summary = self._run()
        self.assertEqual(summary["training_smoke"]["steps_completed"], 5)
        losses = summary["training_smoke"]["loss_total_per_step"]
        self.assertEqual(len(losses), 5)
        self.assertTrue(all(np.isfinite(loss) for loss in losses))

    def test_gradient_routing_and_finiteness(self):
        summary = self._run()
        self.assertTrue(summary["gradients"]["frozen_backbone_no_grad"])
        self.assertTrue(summary["gradients"]["trainable_gradients_present"])
        self.assertTrue(summary["gradients"]["trainable_gradients_finite"])
        self.assertTrue(summary["pass_criteria"]["optimizer_excludes_frozen_backbone"])

    def test_validation_forward_success(self):
        summary = self._run()
        self.assertTrue(summary["pass_criteria"]["validation_forward_success"])
        self.assertEqual(summary["validation_smoke"]["num_cases_processed"], 4)

    def test_csv_outputs_written(self):
        self._run()
        train_csv = self.output_dir / "train_steps.csv"
        grad_csv = self.output_dir / "gradient_stats.csv"
        sel_csv = self.output_dir / "selection_log.csv"
        self.assertTrue(train_csv.exists())
        self.assertTrue(grad_csv.exists())
        self.assertTrue(sel_csv.exists())

        with train_csv.open(encoding="utf-8") as handle:
            train_rows = list(csv.DictReader(handle))
        with grad_csv.open(encoding="utf-8") as handle:
            grad_rows = list(csv.DictReader(handle))
        with sel_csv.open(encoding="utf-8") as handle:
            sel_rows = list(csv.DictReader(handle))
        self.assertEqual(len(train_rows), 5)
        self.assertGreater(len(grad_rows), 0)
        self.assertEqual(len(sel_rows), 12)

    def test_summary_schema(self):
        summary = self._run()
        for key in (
            "setup",
            "selection",
            "training_smoke",
            "validation_smoke",
            "gradients",
            "pass_criteria",
            "outputs",
        ):
            self.assertIn(key, summary)
        self.assertIn(summary["status"], {"PASS", "PARTIAL_PASS", "BLOCKED"})

    def test_d039_no_hausdorff(self):
        summary = self._run()
        raw_json = json.dumps(summary).lower()
        self.assertNotIn("hausdorff", raw_json)
        report_text = (self.output_dir / "hybrid_tiny_train_smoke_report.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("hausdorff", report_text.lower())


class TestGuardrails(unittest.TestCase):
    def test_forbidden_output_root_guardrail(self):
        ok, msg = tts._validate_output_guardrail(Path("artifacts/runs/smoke"))
        self.assertFalse(ok)
        self.assertIn("forbidden", msg.lower())


if __name__ == "__main__":
    unittest.main()
