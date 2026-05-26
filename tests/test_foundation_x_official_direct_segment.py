"""Lightweight tests for the official Foundation_X direct diagnostic script."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import foundation_x_official_direct_segment as fxods  # type: ignore  # noqa: E402


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _make_dataset(root: Path) -> Path:
    images = root / "imagesTs"
    labels = root / "heldout_labelsTs"
    image = np.full((32, 32), 128, dtype=np.uint8)

    pos_mask = np.zeros((32, 32), dtype=np.uint8)
    pos_mask[8:24, 8:24] = 1
    _write_png(images / "siim_000001_0000.png", image)
    _write_png(labels / "siim_000001.png", pos_mask)

    neg_mask = np.zeros((32, 32), dtype=np.uint8)
    _write_png(images / "siim_000002_0000.png", image)
    _write_png(labels / "siim_000002.png", neg_mask)
    return root


class _StubOfficialModel:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def load_checkpoint(self, checkpoint: Path, state_key: str):
        return {
            "status": "loaded",
            "state_key": state_key,
            "shape_compatible_key_count": 1,
            "missing_count": 0,
            "unexpected_before_shape_filter_count": 0,
            "shape_mismatch_count": 0,
        }

    def run_synthetic_probe(self, image_size: int, heads, head4_channel: int):
        outputs = {}
        for head in list(heads) + [4]:
            key = f"head_{head}" if head != 4 else f"head_4_ch{head4_channel}"
            channels = 13 if head == 4 else 1
            raw = torch.zeros(1, channels, image_size, image_size)
            selected = raw[:, :1]
            outputs[key] = {
                "raw_head_logits": {"shape": list(raw.shape)},
                "selected_logits": {"shape": list(selected.shape)},
                "probability": {"shape": list(selected.shape)},
            }
        return {"synthetic_input": {"shape": [1, 3, image_size, image_size]}, "outputs": outputs}

    def predict_probability(self, image_tensor: torch.Tensor, head_idx: int, channel=None):
        b, _, h, w = image_tensor.shape
        logits = torch.full((b, 1, h, w), -4.0, dtype=torch.float32, device=image_tensor.device)
        logits[..., h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 4.0 + float(head_idx) * 0.01
        return logits, torch.sigmoid(logits)


class TestFoundationXOfficialDirectSegment(unittest.TestCase):
    def test_balanced_case_sampling_returns_positive_and_negative(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_dataset(Path(td) / "ds")
            cases = fxods.list_dataset101_cases(root)
            picked = fxods.select_cases(
                cases,
                positive_cases=1,
                negative_cases=1,
                max_cases=2,
                case_sampling="balanced",
                seed=-1,
            )
            self.assertEqual(len(picked), 2)
            self.assertTrue(any(c["label_is_positive"] for c in picked))
            self.assertTrue(any(not c["label_is_positive"] for c in picked))

    def test_stub_run_emits_required_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_dataset(Path(td) / "ds")
            checkpoint = Path(td) / "fake.pth"
            checkpoint.write_bytes(b"stub")
            out_dir = (
                Path(td)
                / "artifacts"
                / "diagnostics"
                / "foundation_x_official_direct"
                / "stub"
            )
            args = fxods.parse_args(
                [
                    "--dataset-root", str(root),
                    "--checkpoint", str(checkpoint),
                    "--state-keys", "model", "teacher_model",
                    "--heads", "5", "2",
                    "--head4-channel", "12",
                    "--preprocess-variants", "official_siim_224",
                    "--device", "cpu",
                    "--max-cases", "2",
                    "--positive-cases", "1",
                    "--negative-cases", "1",
                    "--case-sampling", "balanced",
                    "--out", str(out_dir),
                    "--save-visuals", "true",
                    "--save-histograms", "false",
                ],
            )
            rc = fxods.run_diagnostics(args, model_factory=_StubOfficialModel)
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "load_diagnostics.json").exists())
            self.assertTrue((out_dir / "tensor_stats.json").exists())
            self.assertTrue((out_dir / "summary.yaml").exists() or (out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "report.md").exists())
            self.assertTrue((out_dir / "run_metadata.json").exists())
            self.assertTrue((out_dir / "per_case_metrics.csv").exists())
            self.assertTrue(any((out_dir / "visual_overlays").rglob("*.png")))
            self.assertTrue(any((out_dir / "probability_maps").rglob("*.png")))
            self.assertTrue(any((out_dir / "binary_masks").rglob("*.png")))


if __name__ == "__main__":
    unittest.main()
