"""Lightweight tests for the official Foundation_X direct export script."""

from __future__ import annotations

import csv
import json
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


def _write_case(images_dir: Path, labels_dir: Path, case_id: str, positive: bool = False) -> None:
    image = np.full((32, 32), 128, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    if positive:
        mask[8:24, 8:24] = 1
    _write_png(images_dir / f"{case_id}_0000.png", image)
    _write_png(labels_dir / f"{case_id}.png", mask)


def _make_dataset(root: Path) -> Path:
    images = root / "imagesTs"
    labels = root / "heldout_labelsTs"
    _write_case(images, labels, "siim_000001", positive=True)
    _write_case(images, labels, "siim_000002", positive=False)
    return root


def _make_train_dataset(root: Path) -> Path:
    images = root / "imagesTr"
    labels = root / "labelsTr"
    _write_case(images, labels, "siim_000001", positive=True)
    _write_case(images, labels, "siim_000002", positive=False)
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

    def run_synthetic_probe(self, image_size: int, heads, head4_channel: int, include_head4: bool = False):
        outputs = {}
        for spec in fxods.build_requested_head_specs(heads, head4_channel, include_head4):
            channels = 13 if spec.head_idx == fxods.HEAD_CHESTX_DET else 1
            raw = torch.zeros(1, channels, image_size, image_size)
            selected = raw[:, :1]
            outputs[spec.key] = {
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
    def test_case_id_parsing_strips_modality_suffix(self):
        self.assertEqual(fxods._normalize_case_id("siim_000001_0000.png"), "siim_000001")
        self.assertEqual(fxods._normalize_case_id("siim_000001_0000"), "siim_000001")
        self.assertEqual(fxods._normalize_case_id("siim_000001"), "siim_000001")

    def test_train_style_matching_pairs_images_and_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_train_dataset(Path(td) / "ds")
            cases = fxods.list_cases_from_dirs(root / "imagesTr", root / "labelsTr", split_name="train")
            self.assertEqual([c["case_id"] for c in cases], ["siim_000001", "siim_000002"])
            self.assertEqual(cases[0]["image_path"].name, "siim_000001_0000.png")
            self.assertEqual(cases[0]["label_path"].name, "siim_000001.png")
            self.assertEqual(cases[0]["split"], "train")

    def test_duplicate_and_unmatched_cases_fail_clearly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dup"
            images = root / "imagesTr"
            labels = root / "labelsTr"
            _write_case(images, labels, "siim_000001", positive=True)
            _write_png(images / "siim_000001_0001.png", np.full((32, 32), 128, dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "duplicate image case IDs"):
                fxods.list_cases_from_dirs(images, labels, split_name="train")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "unmatched"
            images = root / "imagesTr"
            labels = root / "labelsTr"
            _write_png(images / "siim_000001_0000.png", np.full((32, 32), 128, dtype=np.uint8))
            _write_png(labels / "siim_000002.png", np.zeros((32, 32), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "images without labels"):
                fxods.list_cases_from_dirs(images, labels, split_name="train")

    def test_dataset_root_backward_compatibility_resolves_heldout_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_dataset(Path(td) / "ds")
            args = fxods.parse_args(
                [
                    "--dataset-root", str(root),
                    "--checkpoint", str(Path(td) / "fake.pth"),
                    "--out", str(Path(td) / "artifacts/diagnostics/foundation_x_official_direct/stub"),
                ],
            )
            images_dir, labels_dir = fxods.resolve_case_dirs(args)
            self.assertEqual(images_dir, root / "imagesTs")
            self.assertEqual(labels_dir, root / "heldout_labelsTs")
            self.assertEqual(len(fxods.list_dataset101_cases(root)), 2)

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

    def _run_stub_export(
        self,
        *,
        td: str,
        extra_args: list[str] | None = None,
        out_name: str = "stub",
    ) -> Path:
        root = _make_train_dataset(Path(td) / "ds")
        checkpoint = Path(td) / "fake.pth"
        checkpoint.write_bytes(b"stub")
        out_dir = Path(td) / "artifacts" / "diagnostics" / "foundation_x_official_direct" / out_name
        args_list = [
            "--dataset-root", str(root),
            "--images-dir", str(root / "imagesTr"),
            "--labels-dir", str(root / "labelsTr"),
            "--split-name", "train",
            "--checkpoint", str(checkpoint),
            "--state-keys", "teacher_model",
            "--heads", "5",
            "--preprocess-variants", "official_siim_224",
            "--device", "cpu",
            "--max-cases", "2",
            "--case-sampling", "sequential",
            "--stage", "both",
            "--out", str(out_dir),
            "--save-visuals", "false",
            "--save-histograms", "false",
            "--threshold", "0.5",
        ]
        if extra_args:
            args_list.extend(extra_args)
        args = fxods.parse_args(args_list)
        rc = fxods.run_diagnostics(args, model_factory=_StubOfficialModel)
        self.assertEqual(rc, 0)
        return out_dir

    def test_train_stub_export_writes_manifest_with_required_columns(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = self._run_stub_export(td=td, extra_args=["--save-prob-maps", "true"])
            manifest = out_dir / "teacher_head5_prior_manifest_train.csv"
            self.assertTrue(manifest.exists())
            with open(manifest, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, fxods.MANIFEST_COLUMNS)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["split"] == "train" for r in rows))
            self.assertTrue(all(r["probability_map_path"] for r in rows))
            self.assertTrue(all(r["binary_mask_path"] == "" for r in rows))
            self.assertTrue(all(Path(r["probability_map_path"]).exists() for r in rows))
            self.assertFalse(any((out_dir / "binary_masks").rglob("*.png")))

    def test_heads_5_does_not_export_head4_without_opt_in(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = self._run_stub_export(td=td, extra_args=["--save-prob-maps", "true"])
            metrics = (out_dir / "per_case_metrics.csv").read_text(encoding="utf-8")
            tensor_stats = json.loads((out_dir / "tensor_stats.json").read_text(encoding="utf-8"))
            self.assertNotIn("head_4_ch12", metrics)
            self.assertNotIn(
                "head_4_ch12",
                tensor_stats["stage_a_synthetic"]["teacher_model"]["outputs"],
            )

    def test_include_head4_opt_in_exports_head4_channel(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = self._run_stub_export(
                td=td,
                extra_args=["--save-prob-maps", "true", "--include-head4", "true"],
                out_name="stub_head4",
            )
            metrics = (out_dir / "per_case_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("head_4_ch12", metrics)

    def test_metrics_only_skips_probability_and_binary_pngs(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = self._run_stub_export(
                td=td,
                extra_args=[
                    "--metrics-only", "true",
                    "--save-prob-maps", "true",
                    "--save-binary-masks", "true",
                ],
                out_name="metrics_only",
            )
            self.assertFalse((out_dir / "probability_maps").exists())
            self.assertFalse((out_dir / "binary_masks").exists())
            with open(out_dir / "teacher_head5_prior_manifest_train.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["probability_map_path"] == "" for r in rows))
            self.assertTrue(all(r["binary_mask_path"] == "" for r in rows))

    def test_train_validation_rejects_heldout_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_dataset(Path(td) / "ds")
            checkpoint = Path(td) / "fake.pth"
            checkpoint.write_bytes(b"stub")
            out_dir = Path(td) / "artifacts" / "diagnostics" / "foundation_x_official_direct" / "bad_train"
            prob_path = out_dir / "probability_maps" / "teacher_model" / "official_siim_224" / "head_5" / "siim_000001.png"
            _write_png(prob_path, np.zeros((32, 32), dtype=np.uint8))
            args = fxods.parse_args(
                [
                    "--dataset-root", str(root),
                    "--images-dir", str(root / "imagesTs"),
                    "--labels-dir", str(root / "heldout_labelsTs"),
                    "--split-name", "train",
                    "--checkpoint", str(checkpoint),
                    "--state-keys", "teacher_model",
                    "--heads", "5",
                    "--preprocess-variants", "official_siim_224",
                    "--device", "cpu",
                    "--max-cases", "1",
                    "--case-sampling", "sequential",
                    "--out", str(out_dir),
                ],
            )
            rows = [
                {
                    "case_id": "siim_000001",
                    "split": "train",
                    "image_path": str(root / "imagesTs" / "siim_000001_0000.png"),
                    "label_path": str(root / "heldout_labelsTs" / "siim_000001.png"),
                    "probability_map_path": str(prob_path),
                    "binary_mask_path": "",
                    "state_key": "teacher_model",
                    "preprocess_variant": "official_siim_224",
                    "head_key": "head_5",
                    "threshold": 0.5,
                    "gt_is_positive": True,
                    "pred_is_positive": False,
                    "dice": 0.0,
                    "iou": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "specificity": 1.0,
                    "gt_foreground_pixels": 1,
                    "pred_foreground_pixels": 0,
                }
            ]
            validation = fxods.validate_prior_export(
                manifest_rows=rows,
                args=args,
                selected_cases=[{"case_id": "siim_000001"}],
                loaded_state_keys=["teacher_model"],
                head_specs=fxods.build_requested_head_specs([5], 12, include_head4=False),
            )
            self.assertEqual(validation["status"], "failed")
            self.assertTrue(any("imagesTs" in e for e in validation["errors"]))
            self.assertTrue(any("heldout_labelsTs" in e for e in validation["errors"]))

    def test_save_binary_masks_false_leaves_manifest_binary_path_blank(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = self._run_stub_export(
                td=td,
                extra_args=["--save-prob-maps", "true", "--save-binary-masks", "false"],
                out_name="no_binary",
            )
            with open(out_dir / "teacher_head5_prior_manifest_train.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(r["probability_map_path"] for r in rows))
            self.assertTrue(all(r["binary_mask_path"] == "" for r in rows))
            self.assertFalse(any((out_dir / "binary_masks").rglob("*.png")))

    def test_stub_run_emits_required_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = _make_dataset(Path(td) / "ds")
            checkpoint = Path(td) / "fake.pth"
            checkpoint.write_bytes(b"stub")
            out_dir = Path(td) / "artifacts" / "diagnostics" / "foundation_x_official_direct" / "stub"
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
                    "--save-prob-maps", "true",
                    "--save-binary-masks", "true",
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
            self.assertTrue((out_dir / "teacher_head5_prior_manifest.csv").exists())
            self.assertTrue(any((out_dir / "visual_overlays").rglob("*.png")))
            self.assertTrue(any((out_dir / "probability_maps").rglob("*.png")))
            self.assertTrue(any((out_dir / "binary_masks").rglob("*.png")))


if __name__ == "__main__":
    unittest.main()
