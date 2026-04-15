"""Regression tests for authoritative training run artifact helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.training.run_artifacts import (
    EVALUATION_CSV_COLUMNS,
    HISTORY_CSV_COLUMNS,
    build_best_checkpoint_metadata,
    build_run_metadata,
    compute_code_fingerprint,
    compute_config_hash,
    make_run_id,
    prepare_run_artifacts,
    resolve_initial_checkpoint_reference,
    write_config_snapshot,
    write_evaluation_csv,
    write_history_csv,
    write_yaml,
)


class TestRunArtifacts(unittest.TestCase):
    def test_make_run_id_is_deterministic_for_fixed_time(self) -> None:
        now = datetime(2026, 4, 11, 9, 30, 0, tzinfo=timezone.utc)
        self.assertEqual(
            make_run_id("pretrained_resnet34_unet", now=now),
            "20260411T093000Z_pretrained_resnet34_unet",
        )

    def test_prepare_run_artifacts_creates_expected_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "artifacts" / "runs"
            artifacts = prepare_run_artifacts(
                "baseline",
                run_dir=None,
                run_root=run_root,
                now=datetime(2026, 4, 11, 9, 30, 0, tzinfo=timezone.utc),
            )

            self.assertTrue(artifacts.run_dir.exists())
            self.assertTrue(artifacts.metadata_dir.exists())
            self.assertTrue(artifacts.metrics_dir.exists())
            self.assertTrue(artifacts.checkpoints_dir.exists())
            self.assertTrue(artifacts.selection_dir.exists())
            self.assertTrue(artifacts.reports_dir.exists())
            self.assertTrue(artifacts.qualitative_validation_dir.exists())
            self.assertTrue(artifacts.qualitative_test_dir.exists())
            self.assertEqual(artifacts.run_id, "20260411T093000Z_baseline")
            self.assertEqual(artifacts.selection_state_path, artifacts.selection_dir / "selection_state.yaml")
            self.assertEqual(artifacts.test_metrics_path, artifacts.reports_dir / "test_metrics.csv")
            self.assertEqual(artifacts.test_summary_path, artifacts.reports_dir / "test_summary.yaml")

    def test_config_hash_is_deterministic(self) -> None:
        cfg_a = {"model": {"type": "baseline"}, "seed": 42}
        cfg_b = {"seed": 42, "model": {"type": "baseline"}}
        self.assertEqual(compute_config_hash(cfg_a), compute_config_hash(cfg_b))

    def test_code_fingerprint_changes_with_scope_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            file_a = root / "src" / "a.py"
            file_a.parent.mkdir(parents=True)
            file_a.write_text("print('a')\n", encoding="utf-8")

            first = compute_code_fingerprint([file_a], repo_root=root)
            file_a.write_text("print('b')\n", encoding="utf-8")
            second = compute_code_fingerprint([file_a], repo_root=root)

            self.assertNotEqual(first, second)

    def test_build_run_metadata_uses_dataset_manifest_and_checkpoint_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "data" / "processed" / "trusted_v1"
            dataset_root.mkdir(parents=True)
            manifest = {
                "dataset_fingerprint": "dataset-fp",
                "fingerprints": {"splits": "split-fp"},
            }
            (dataset_root / "dataset_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            (root / "configs").mkdir()
            (root / "configs" / "config.yaml").write_text("model:\n  type: pretrained_resnet34_unet\n", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "placeholder.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "requirements.txt").write_text("torch\n", encoding="utf-8")

            cfg = {
                "model": {
                    "type": "pretrained_resnet34_unet",
                    "in_channels": 1,
                    "num_classes": 1,
                    "base_filters": 64,
                },
                "data": {
                    "processed_dir": "data/processed/trusted_v1",
                    "input_size": 512,
                    "train_mask_variant": "dilated_masks",
                    "eval_mask_variant": "original_masks",
                },
                "selection": {
                    "metric": "val_dice_pos_mean",
                    "postprocess": "none",
                },
                "seed": 42,
            }

            metadata = build_run_metadata(
                cfg=cfg,
                config_path="configs/config.yaml",
                repo_root=root,
                run_id="run_001",
                resume_checkpoint_path=None,
                started_at="2026-04-11T09:30:00Z",
            )

            self.assertEqual(metadata["run_id"], "run_001")
            self.assertEqual(metadata["dataset_fingerprint"], "dataset-fp")
            self.assertEqual(metadata["split_fingerprint"], "split-fp")
            self.assertEqual(metadata["initial_checkpoint_path"], "torchvision://resnet34_imagenet1k_v1")
            self.assertIsNone(metadata["resume_checkpoint_path"])
            self.assertEqual(metadata["selection_metric"], "val_dice_pos_mean")
            self.assertEqual(metadata["selected_postprocess"], "none")

    def test_write_helpers_persist_yaml_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            yaml_path = root / "metadata" / "run_metadata.yaml"
            history_path = root / "metrics" / "history.csv"
            snapshot_path = root / "metadata" / "config_snapshot.yaml"

            write_yaml(yaml_path, {"run_id": "run_001"})
            write_config_snapshot(snapshot_path, {"model": {"type": "baseline"}})
            write_history_csv(
                history_path,
                {
                    "train_loss": [1.0],
                    "val_loss": [0.5],
                    "val_dice": [0.25],
                    "val_dice_pos": [0.4],
                    "val_iou": [0.2],
                },
            )

            self.assertEqual(yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["run_id"], "run_001")
            self.assertIn("baseline", snapshot_path.read_text(encoding="utf-8"))
            history_df = pd.read_csv(history_path)
            self.assertEqual(list(history_df.columns), list(HISTORY_CSV_COLUMNS))
            self.assertEqual(history_df["epoch"].tolist(), [1])
            self.assertEqual(history_df["val_dice_mean"].tolist(), [0.25])
            self.assertEqual(history_df["val_dice_pos_mean"].tolist(), [0.4])
            self.assertEqual(history_df["val_iou_mean"].tolist(), [0.2])

    def test_write_evaluation_csv_orders_required_columns_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            report_path = root / "reports" / "test_metrics.csv"

            df = write_evaluation_csv(
                report_path,
                [
                    {
                        "image_id": "img_001",
                        "split": "test",
                        "model_type": "baseline",
                        "checkpoint_path": "checkpoints/best_checkpoint.pth",
                        "eval_mask_variant": "original_masks",
                        "selection_metric": "val_dice_pos_mean",
                        "selected_threshold": 0.5,
                        "selected_postprocess": "none",
                        "positive": True,
                        "dice": 0.8,
                        "iou": 0.7,
                        "hausdorff": 1.0,
                        "precision": 0.75,
                        "recall": 0.85,
                        "f1": 0.8,
                        "extra_debug_field": "kept",
                    }
                ],
            )

            self.assertEqual(
                list(df.columns[: len(EVALUATION_CSV_COLUMNS)]),
                list(EVALUATION_CSV_COLUMNS),
            )
            self.assertEqual(df["extra_debug_field"].tolist(), ["kept"])

    def test_best_checkpoint_metadata_records_training_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = root / "artifacts" / "runs" / "run_001" / "checkpoints" / "best_checkpoint.pth"
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"checkpoint")

            cfg = {
                "model": {"type": "baseline"},
                "data": {
                    "input_size": 512,
                    "train_mask_variant": "dilated_masks",
                    "eval_mask_variant": "original_masks",
                },
                "selection": {"metric": "val_dice_pos_mean"},
            }
            payload = build_best_checkpoint_metadata(
                checkpoint_path=checkpoint_path,
                cfg=cfg,
                repo_root=root,
                epoch=3,
                best_metric_value=0.42,
                training_components={
                    "loss": "dice_focal",
                    "optimizer": "AdamW",
                    "scheduler": "ReduceLROnPlateau",
                },
            )

            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(payload["selection_metric"], "val_dice_pos_mean")
            self.assertEqual(payload["training_components"]["optimizer"], "AdamW")
            self.assertTrue(payload["checkpoint_path"].endswith("best_checkpoint.pth"))

    def test_initial_checkpoint_reference_matches_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cfg = {"model": {"type": "baseline"}}
            self.assertEqual(resolve_initial_checkpoint_reference(cfg, repo_root=root), "random_init")

            hybrid_cfg = {
                "model": {"type": "hybrid"},
                "foundation_x": {"checkpoint_path": "checkpoints/foundation_x.pth"},
            }
            resolved = resolve_initial_checkpoint_reference(hybrid_cfg, repo_root=root)
            self.assertTrue(
                resolved.endswith("checkpoints\\foundation_x.pth")
                or resolved.endswith("checkpoints/foundation_x.pth")
            )


if __name__ == "__main__":
    unittest.main()
