"""Regression tests for authoritative evaluation-side run outputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from src.evaluation.evaluate import evaluate, save_selection_state, select_threshold_and_save
from src.training.run_artifacts import EVALUATION_CSV_COLUMNS


class DummyDataset:
    def __init__(self, image_ids: list[str]) -> None:
        self.image_ids = image_ids


class TestEvaluationRunOutputs(unittest.TestCase):
    def make_cfg(self, processed_dir: str) -> dict:
        return {
            "selection": {
                "metric": "val_dice_pos_mean",
                "threshold_candidates": [0.5],
                "postprocess": "none",
            },
            "data": {
                "processed_dir": processed_dir,
                "eval_mask_variant": "original_masks",
                "input_size": 2,
                "num_workers": 0,
            },
            "device": "cpu",
        }

    def make_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        positive_image = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
        positive_mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
        negative_image = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        negative_mask = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        return [
            (positive_image, positive_mask),
            (negative_image, negative_mask),
        ]

    def make_selection_result(self) -> dict:
        return {
            "split": "val",
            "selection_metric": "val_dice_pos_mean",
            "selected_threshold": 0.5,
            "selected_postprocess": "none",
            "threshold_summary": [
                {
                    "threshold": 0.5,
                    "val_dice_pos_mean": 1.0,
                    "val_dice_mean": 1.0,
                    "val_iou_mean": 1.0,
                    "positive_image_count": 1,
                }
            ],
        }

    def write_run_metadata(self, run_dir: Path) -> Path:
        run_metadata_path = run_dir / "metadata" / "run_metadata.yaml"
        run_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with run_metadata_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "run_id": run_dir.name,
                    "selection_metric": "val_dice_pos_mean",
                    "selected_threshold": None,
                    "selected_postprocess": "none",
                },
                handle,
                sort_keys=False,
            )
        return run_metadata_path

    def test_select_threshold_and_save_writes_validation_qualitative_under_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = tmp_path / "artifacts" / "runs" / "run_001"
            selection_path = run_dir / "selection" / "selection_state.yaml"
            checkpoint_path = run_dir / "checkpoints" / "best_checkpoint.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")
            run_metadata_path = self.write_run_metadata(run_dir)
            cfg = self.make_cfg(str(tmp_path / "data" / "processed" / "trusted_v1"))
            dataset = DummyDataset(["pos_001", "neg_001"])

            with patch(
                "src.evaluation.evaluate.load_model_for_evaluation",
                return_value=torch.nn.Identity(),
            ), patch(
                "src.evaluation.evaluate.build_eval_dataloader",
                return_value=(dataset, self.make_loader()),
            ):
                payload = select_threshold_and_save(
                    cfg,
                    checkpoint_path=str(checkpoint_path),
                    model_type="baseline",
                    selection_state_path=selection_path,
                )

            manifest_path = run_dir / "qualitative" / "validation_samples" / "manifest.yaml"
            self.assertTrue(selection_path.exists())
            self.assertTrue(manifest_path.exists())
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["split"], "val")
            self.assertEqual(
                [sample["image_id"] for sample in manifest["samples"]],
                ["pos_001", "neg_001"],
            )
            self.assertEqual(
                [sample["subset_tag"] for sample in manifest["samples"]],
                ["positive", "negative"],
            )

            run_metadata = yaml.safe_load(run_metadata_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["selected_threshold"], 0.5)
            self.assertAlmostEqual(run_metadata["selected_threshold"], 0.5)
            self.assertEqual(run_metadata["selected_postprocess"], "none")

    def test_evaluate_writes_reports_and_test_qualitative_under_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = tmp_path / "artifacts" / "runs" / "run_001"
            selection_path = run_dir / "selection" / "selection_state.yaml"
            checkpoint_path = run_dir / "checkpoints" / "best_checkpoint.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")
            run_metadata_path = self.write_run_metadata(run_dir)
            cfg = self.make_cfg(str(tmp_path / "data" / "processed" / "trusted_v1"))
            dataset = DummyDataset(["pos_001", "neg_001"])

            save_selection_state(
                selection_path,
                self.make_selection_result(),
                cfg,
                checkpoint_path=str(checkpoint_path),
                model_type="baseline",
            )

            with patch(
                "src.evaluation.evaluate.load_model_for_evaluation",
                return_value=torch.nn.Identity(),
            ), patch(
                "src.evaluation.evaluate.build_eval_dataloader",
                return_value=(dataset, self.make_loader()),
            ):
                df = evaluate(
                    cfg,
                    checkpoint_path=str(checkpoint_path),
                    model_type="baseline",
                    selection_state_path=selection_path,
                )

            report_path = run_dir / "reports" / "test_metrics.csv"
            summary_path = run_dir / "reports" / "test_summary.yaml"
            manifest_path = run_dir / "qualitative" / "test_samples" / "manifest.yaml"
            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(manifest_path.exists())

            self.assertEqual(
                list(df.columns[: len(EVALUATION_CSV_COLUMNS)]),
                list(EVALUATION_CSV_COLUMNS),
            )
            self.assertEqual(df["image_id"].tolist(), ["pos_001", "neg_001"])
            self.assertEqual(df["subset_tag"].tolist(), ["positive", "negative"])
            self.assertEqual(df["split"].tolist(), ["test", "test"])
            self.assertEqual(df["eval_mask_variant"].tolist(), ["original_masks", "original_masks"])
            self.assertEqual(df["selected_postprocess"].tolist(), ["none", "none"])
            self.assertEqual(df["selected_threshold"].tolist(), [0.5, 0.5])

            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["split"], "test")
            self.assertAlmostEqual(summary["selected_threshold"], 0.5)
            self.assertEqual(summary["selected_postprocess"], "none")

            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["split"], "test")
            self.assertEqual(len(manifest["samples"]), 2)
            self.assertEqual(
                [sample["image_id"] for sample in manifest["samples"]],
                ["pos_001", "neg_001"],
            )
            self.assertEqual(
                [sample["subset_tag"] for sample in manifest["samples"]],
                ["positive", "negative"],
            )

            run_metadata = yaml.safe_load(run_metadata_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(run_metadata["selected_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
