"""
Tests for scripts/sanity_hybrid_adapter.py.

All tests run with stubbed hybrid model so no real Foundation X checkpoint
or timm dependency is required.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
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

# Allow src.models.backbone import when timm is not installed.
if "timm" not in sys.modules:
    sys.modules["timm"] = types.SimpleNamespace(create_model=lambda *args, **kwargs: None)

import sanity_hybrid_adapter as sha


PRIORITY_IDS = sha.PRIORITY_FN_IDS + sha.PRIORITY_FP_IDS


def _save_gray_png(path: Path, size: int = 256, positive: bool | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if positive is None:
        arr = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    else:
        arr = np.zeros((size, size), dtype=np.uint8)
        if positive:
            arr[8:20, 8:20] = 255
    Image.fromarray(arr, mode="L").save(str(path))


def _build_dataset_fixture(tmpdir: Path, extra: int = 8) -> tuple[Path, Path]:
    input_dir = tmpdir / "imagesTs"
    labels_dir = tmpdir / "heldout_labelsTs"
    input_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for case_id in PRIORITY_IDS:
        _save_gray_png(input_dir / f"{case_id}_0000.png", size=256, positive=None)
        _save_gray_png(
            labels_dir / f"{case_id}.png",
            size=256,
            positive=(case_id in sha.PRIORITY_FN_IDS),
        )

    for i in range(extra):
        case_id = f"siim_{900000 + i:06d}"
        _save_gray_png(input_dir / f"{case_id}_0000.png", size=256, positive=None)
        _save_gray_png(labels_dir / f"{case_id}.png", size=256, positive=(i % 2 == 0))

    return input_dir, labels_dir


def _build_checkpoint_fixture(tmpdir: Path) -> Path:
    ckpt = tmpdir / "fake_foundation_x.pth"
    ckpt.write_bytes(b"fake")
    return ckpt


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
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [
            self.fx0(F.avg_pool2d(x, 4)),
            self.fx1(F.avg_pool2d(x, 8)),
            self.fx2(F.avg_pool2d(x, 16)),
            self.fx3(F.avg_pool2d(x, 32)),
        ]


class SimpleFusion(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))


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
        self.fusion_e3 = SimpleFusion(128, 1)
        self.fusion_e4 = SimpleFusion(256, 1)
        self.h16_fx_context = SimpleFusion(512, 1)
        self.h32_context_head = SimpleFusion(1024, 1)
        self.h32_to_h16 = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.context_merge = SimpleFusion(2, 1)
        self.dec4 = torch.nn.Identity()
        self.dec3 = torch.nn.Identity()
        self.dec2 = torch.nn.Identity()
        self.dec1 = torch.nn.Identity()
        self.final = torch.nn.Conv2d(1, 1, kernel_size=1, bias=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_backbone:
            self.foundation_x.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.foundation_x(x)
        f0 = F.interpolate(self.fusion_e3(fx[0]), size=x.shape[2:], mode="nearest")
        f1 = F.interpolate(self.fusion_e4(fx[1]), size=x.shape[2:], mode="nearest")
        f2 = F.interpolate(self.h16_fx_context(fx[2]), size=x.shape[2:], mode="nearest")
        h32 = self.h32_context_head(fx[3])
        h16 = F.interpolate(self.h32_to_h16(h32), size=x.shape[2:], mode="nearest")
        merged = self.context_merge(torch.cat([f2, h16], dim=1))
        out = f0 + f1 + merged
        logits = self.final(out)
        return torch.sigmoid(logits)


class DummyHybridNaN(DummyHybrid):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        out[:, :, :1, :1] = torch.nan
        return out


class TestCliParsing(unittest.TestCase):
    def test_defaults(self):
        args = sha.parse_args([])
        self.assertEqual(args.img_size, 512)
        self.assertEqual(args.num_cases, 8)
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.strict)
        self.assertFalse(args.unfrozen_backbone)

    def test_checkpoint_alias(self):
        args = sha.parse_args(["--checkpoint", "ckpt.pth"])
        self.assertEqual(args.checkpoint, Path("ckpt.pth"))

    def test_strict_and_unfrozen_flags(self):
        args = sha.parse_args(["--strict", "--unfrozen_backbone"])
        self.assertTrue(args.strict)
        self.assertTrue(args.unfrozen_backbone)


class TestDeterministicSelection(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.input_dir, self.labels_dir = _build_dataset_fixture(self.tmpdir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_same_selection_order(self):
        s1 = sha.select_cases(self.input_dir, self.labels_dir, 8, None)
        s2 = sha.select_cases(self.input_dir, self.labels_dir, 8, None)
        self.assertEqual([x["case_id"] for x in s1], [x["case_id"] for x in s2])

    def test_priority_cases_present(self):
        selected = sha.select_cases(self.input_dir, self.labels_dir, 8, None)
        ids = {x["case_id"] for x in selected}
        for cid in PRIORITY_IDS:
            self.assertIn(cid, ids)


class TestSanityRunWithStub(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.input_dir, self.labels_dir = _build_dataset_fixture(self.tmpdir)
        self.ckpt = _build_checkpoint_fixture(self.tmpdir)
        self.output_dir = self.tmpdir / "artifacts" / "diagnostics" / "hybrid_adapter_sanity"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_with_model(self, model_cls, strict: bool = False) -> dict:
        argv = [
            "--input_dir",
            str(self.input_dir),
            "--labels_dir",
            str(self.labels_dir),
            "--foundation_checkpoint",
            str(self.ckpt),
            "--output_dir",
            str(self.output_dir),
            "--img_size",
            "256",
            "--device",
            "cpu",
            "--num_cases",
            "8",
            "--no_visuals",
            "--backward_batch_size",
            "1",
        ]
        if strict:
            argv.append("--strict")

        with patch("sanity_hybrid_adapter.HybridFoundationUNet", model_cls):
            code = sha.main(argv)

        summary_path = self.output_dir / "hybrid_adapter_sanity_summary.json"
        self.assertTrue(summary_path.exists())
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data["_exit_code_from_main"] = code
        return data

    def test_stubbed_run_passes(self):
        summary = self._run_with_model(DummyHybrid, strict=False)
        self.assertEqual(summary["_exit_code_from_main"], 0)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["audit_name"], "hybrid_adapter_sanity")

    def test_forward_stats_written(self):
        self._run_with_model(DummyHybrid, strict=False)
        path = self.output_dir / "per_case_forward_stats.csv"
        self.assertTrue(path.exists())
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 8)

    def test_gradient_stats_written(self):
        self._run_with_model(DummyHybrid, strict=False)
        path = self.output_dir / "gradient_stats.csv"
        self.assertTrue(path.exists())
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(len(rows) > 0)

    def test_loss_is_finite(self):
        summary = self._run_with_model(DummyHybrid, strict=False)
        total_loss = summary["losses"]["backward_total_loss"]
        self.assertIsInstance(total_loss, float)
        self.assertTrue(np.isfinite(total_loss))
        self.assertTrue(summary["pass_criteria"]["loss_finite"])

    def test_gradient_routing_for_frozen_backbone(self):
        summary = self._run_with_model(DummyHybrid, strict=False)
        self.assertTrue(summary["gradients"]["frozen_backbone_no_grad"])
        self.assertTrue(summary["gradients"]["trainable_gradients_present"])
        self.assertTrue(summary["gradients"]["trainable_gradients_finite"])

    def test_summary_has_required_keys(self):
        summary = self._run_with_model(DummyHybrid, strict=False)
        self.assertIn("setup", summary)
        self.assertIn("forward", summary)
        self.assertIn("losses", summary)
        self.assertIn("gradients", summary)
        self.assertIn("outputs", summary)
        self.assertIn("pass_criteria", summary)
        self.assertIn(summary["status"], {"PASS", "PARTIAL_PASS", "BLOCKED"})

    def test_d039_no_hausdorff(self):
        summary = self._run_with_model(DummyHybrid, strict=False)
        raw_json = json.dumps(summary).lower()
        self.assertNotIn("hausdorff", raw_json)
        report_text = (self.output_dir / "hybrid_adapter_sanity_report.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("hausdorff", report_text.lower())

    def test_nonfinite_outputs_block_in_strict_mode(self):
        summary = self._run_with_model(DummyHybridNaN, strict=True)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertEqual(summary["_exit_code_from_main"], 2)
        self.assertFalse(summary["pass_criteria"]["output_finite"])


if __name__ == "__main__":
    unittest.main()
