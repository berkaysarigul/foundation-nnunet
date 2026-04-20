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
    PAIRED_DELTA_CSV_COLUMNS,
    SPLIT_LEVEL_CSV_COLUMNS,
    build_best_checkpoint_metadata,
    build_final_repeated_split_summary_payload,
    build_paired_delta_records,
    build_split_level_records_from_authoritative_runs,
    build_repeated_split_manifest,
    build_run_metadata,
    compute_code_fingerprint,
    compute_config_hash,
    make_run_id,
    prepare_repeated_split_study_artifacts,
    prepare_run_artifacts,
    resolve_initial_checkpoint_reference,
    write_config_snapshot,
    write_evaluation_csv,
    write_final_repeated_split_summary,
    write_history_csv,
    write_paired_delta_csv,
    write_split_level_csv,
    write_yaml,
)


class TestRunArtifacts(unittest.TestCase):
    def write_authoritative_run_package(
        self,
        *,
        run_dir: Path,
        run_id: str,
        model_type: str,
        split_fingerprint: str,
        dataset_root: Path,
        selection_metric: str = "val_dice_pos_mean",
        selected_threshold: float = 0.5,
        selected_postprocess: str = "none",
        train_mask_variant: str = "dilated_masks",
        eval_mask_variant: str = "original_masks",
        test_dice_mean: float = 0.4,
        test_dice_pos_mean: float = 0.5,
        test_iou_mean: float = 0.3,
        test_iou_pos_mean: float = 0.35,
    ) -> None:
        metadata_dir = run_dir / "metadata"
        reports_dir = run_dir / "reports"
        selection_dir = run_dir / "selection"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        selection_dir.mkdir(parents=True, exist_ok=True)

        run_metadata = {
            "run_id": run_id,
            "split_fingerprint": split_fingerprint,
        }
        with (metadata_dir / "run_metadata.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(run_metadata, handle, sort_keys=False)

        selection_state_path = selection_dir / "selection_state.yaml"
        selection_state_path.write_text("selection_state: placeholder\n", encoding="utf-8")

        summary = {
            "split": "test",
            "model_type": model_type,
            "checkpoint_path": str((run_dir / "checkpoints" / "best_checkpoint.pth").resolve()),
            "dataset_root": str(dataset_root.resolve()),
            "selection_state_path": str(selection_state_path.resolve()),
            "train_mask_variant": train_mask_variant,
            "eval_mask_variant": eval_mask_variant,
            "selection_metric": selection_metric,
            "selected_threshold": selected_threshold,
            "selected_postprocess": selected_postprocess,
            "subsets": {
                "all": {
                    "count": 10,
                    "dice": {"mean": test_dice_mean},
                    "iou": {"mean": test_iou_mean},
                },
                "positive": {
                    "count": 4,
                    "dice": {"mean": test_dice_pos_mean},
                    "iou": {"mean": test_iou_pos_mean},
                },
            },
        }
        with (reports_dir / "test_summary.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(summary, handle, sort_keys=False)

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

    def test_prepare_repeated_split_study_artifacts_creates_expected_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            study_root = Path(tmp_dir) / "artifacts" / "repeated_splits"
            artifacts = prepare_repeated_split_study_artifacts(
                "study_alpha",
                study_root=study_root,
            )

            self.assertTrue(artifacts.study_dir.exists())
            self.assertTrue(artifacts.metadata_dir.exists())
            self.assertTrue(artifacts.aggregations_dir.exists())
            self.assertTrue(artifacts.comparisons_dir.exists())
            self.assertTrue(artifacts.summary_dir.exists())
            self.assertEqual(
                artifacts.split_manifest_path,
                artifacts.metadata_dir / "split_manifest.yaml",
            )
            self.assertEqual(
                artifacts.split_level_table_path,
                artifacts.aggregations_dir / "split_level_metrics.csv",
            )
            self.assertEqual(
                artifacts.final_summary_path,
                artifacts.summary_dir / "final_summary.yaml",
            )
            self.assertEqual(
                artifacts.paired_delta_table_path("baseline vs hybrid"),
                artifacts.comparisons_dir / "baseline_vs_hybrid_paired_deltas.csv",
            )

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

    def test_build_repeated_split_manifest_records_dataset_context_and_sorted_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "data" / "processed" / "trusted_v1"
            dataset_root.mkdir(parents=True)
            (dataset_root / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_fingerprint": "dataset-fp",
                        "fingerprints": {"splits": "trusted-single-split-fp"},
                    }
                ),
                encoding="utf-8",
            )

            payload = build_repeated_split_manifest(
                study_id="study_alpha",
                dataset_root=dataset_root,
                repo_root=root,
                split_instances=[
                    {
                        "split_instance_id": "split_b",
                        "split_seed": 200,
                        "train_ids": ["img_3", "img_1"],
                        "val_ids": ["img_4"],
                        "test_ids": ["img_6", "img_5"],
                    },
                    {
                        "split_instance_id": "split_a",
                        "split_seed": 100,
                        "train_ids": ["img_2", "img_0"],
                        "val_ids": ["img_7"],
                        "test_ids": ["img_8"],
                    },
                ],
            )

            self.assertEqual(payload["study_id"], "study_alpha")
            self.assertEqual(payload["dataset_fingerprint"], "dataset-fp")
            self.assertEqual(payload["base_split_fingerprint"], "trusted-single-split-fp")
            self.assertEqual(payload["selection_metric"], "val_dice_pos_mean")
            self.assertEqual(payload["primary_test_metric"], "test_positive_only_dice_mean")
            self.assertEqual(payload["split_count"], 2)
            self.assertEqual(
                [instance["split_instance_id"] for instance in payload["split_instances"]],
                ["split_a", "split_b"],
            )
            self.assertEqual(payload["split_instances"][1]["train_ids"], ["img_1", "img_3"])
            self.assertEqual(payload["split_instances"][1]["test_ids"], ["img_5", "img_6"])
            self.assertEqual(payload["split_instances"][1]["counts"]["train"], 2)
            self.assertTrue(payload["split_instances"][1]["split_fingerprint"])

    def test_build_repeated_split_manifest_rejects_overlap_between_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "data" / "processed" / "trusted_v1"
            dataset_root.mkdir(parents=True)
            (dataset_root / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_fingerprint": "dataset-fp",
                        "fingerprints": {"splits": "trusted-single-split-fp"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must keep train/val/test disjoint"):
                build_repeated_split_manifest(
                    study_id="study_alpha",
                    dataset_root=dataset_root,
                    repo_root=root,
                    split_instances=[
                        {
                            "split_instance_id": "split_a",
                            "split_seed": 123,
                            "train_ids": ["img_1", "img_2"],
                            "val_ids": ["img_2"],
                            "test_ids": ["img_3"],
                        }
                    ],
                )

    def test_build_split_level_records_from_authoritative_runs_consumes_manifest_and_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "data" / "processed" / "trusted_v1"
            dataset_root.mkdir(parents=True)
            (dataset_root / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_fingerprint": "dataset-fp",
                        "fingerprints": {"splits": "trusted-single-split-fp"},
                    }
                ),
                encoding="utf-8",
            )

            split_manifest = build_repeated_split_manifest(
                study_id="study_alpha",
                dataset_root=dataset_root,
                repo_root=root,
                split_instances=[
                    {
                        "split_instance_id": "split_a",
                        "split_seed": 100,
                        "train_ids": ["img_2", "img_0"],
                        "val_ids": ["img_7"],
                        "test_ids": ["img_8"],
                    }
                ],
            )

            baseline_run_dir = root / "artifacts" / "runs" / "baseline_split_a"
            hybrid_run_dir = root / "artifacts" / "runs" / "hybrid_split_a"
            self.write_authoritative_run_package(
                run_dir=baseline_run_dir,
                run_id="baseline_split_a",
                model_type="pretrained_resnet34_unet",
                split_fingerprint="run-split-fp-a",
                dataset_root=dataset_root,
                test_dice_mean=0.41,
                test_dice_pos_mean=0.50,
                test_iou_mean=0.31,
                test_iou_pos_mean=0.37,
            )
            self.write_authoritative_run_package(
                run_dir=hybrid_run_dir,
                run_id="hybrid_split_a",
                model_type="hybrid",
                split_fingerprint="run-split-fp-a",
                dataset_root=dataset_root,
                test_dice_mean=0.43,
                test_dice_pos_mean=0.53,
                test_iou_mean=0.33,
                test_iou_pos_mean=0.39,
            )

            records = build_split_level_records_from_authoritative_runs(
                split_manifest=split_manifest,
                model_runs=[
                    {
                        "split_instance_id": "split_a",
                        "model_name": "baseline",
                        "run_dir": baseline_run_dir,
                    },
                    {
                        "split_instance_id": "split_a",
                        "model_name": "hybrid",
                        "run_dir": hybrid_run_dir,
                    },
                ],
                repo_root=root,
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["split_instance_id"], "split_a")
            self.assertEqual(records[0]["dataset_fingerprint"], "dataset-fp")
            self.assertEqual(records[0]["base_split_fingerprint"], "trusted-single-split-fp")
            self.assertEqual(records[0]["study_split_fingerprint"], split_manifest["split_instances"][0]["split_fingerprint"])
            self.assertEqual(records[0]["selection_metric"], "val_dice_pos_mean")
            self.assertAlmostEqual(records[0]["selected_threshold"], 0.5)
            self.assertAlmostEqual(records[0]["test_dice_pos_mean"], 0.5)

            with tempfile.TemporaryDirectory() as csv_tmp_dir:
                csv_path = Path(csv_tmp_dir) / "split_level_metrics.csv"
                df = write_split_level_csv(csv_path, records)
                self.assertEqual(
                    list(df.columns[: len(SPLIT_LEVEL_CSV_COLUMNS)]),
                    list(SPLIT_LEVEL_CSV_COLUMNS),
                )
                self.assertEqual(df["model_name"].tolist(), ["baseline", "hybrid"])

    def test_build_paired_delta_records_uses_shared_split_instances_and_metric_delta(self) -> None:
        split_level_records = [
            {
                "study_id": "study_alpha",
                "split_instance_id": "split_a",
                "split_seed": 100,
                "model_name": "baseline",
                "model_type": "pretrained_resnet34_unet",
                "run_id": "baseline_split_a",
                "run_dir": "/tmp/baseline_split_a",
                "dataset_fingerprint": "dataset-fp",
                "base_split_fingerprint": "trusted-single-split-fp",
                "study_split_fingerprint": "study-fp-a",
                "run_split_fingerprint": "run-fp-a",
                "selection_metric": "val_dice_pos_mean",
                "selected_threshold": 0.5,
                "selected_postprocess": "none",
                "selection_state_path": "/tmp/baseline_split_a/selection/selection_state.yaml",
                "checkpoint_path": "/tmp/baseline_split_a/checkpoints/best_checkpoint.pth",
                "train_mask_variant": "dilated_masks",
                "eval_mask_variant": "original_masks",
                "test_dice_mean": 0.40,
                "test_dice_pos_mean": 0.50,
                "test_iou_mean": 0.30,
                "test_iou_pos_mean": 0.35,
            },
            {
                "study_id": "study_alpha",
                "split_instance_id": "split_a",
                "split_seed": 100,
                "model_name": "hybrid",
                "model_type": "hybrid",
                "run_id": "hybrid_split_a",
                "run_dir": "/tmp/hybrid_split_a",
                "dataset_fingerprint": "dataset-fp",
                "base_split_fingerprint": "trusted-single-split-fp",
                "study_split_fingerprint": "study-fp-a",
                "run_split_fingerprint": "run-fp-a",
                "selection_metric": "val_dice_pos_mean",
                "selected_threshold": 0.55,
                "selected_postprocess": "none",
                "selection_state_path": "/tmp/hybrid_split_a/selection/selection_state.yaml",
                "checkpoint_path": "/tmp/hybrid_split_a/checkpoints/best_checkpoint.pth",
                "train_mask_variant": "dilated_masks",
                "eval_mask_variant": "original_masks",
                "test_dice_mean": 0.43,
                "test_dice_pos_mean": 0.53,
                "test_iou_mean": 0.33,
                "test_iou_pos_mean": 0.39,
            },
        ]

        records = build_paired_delta_records(
            split_level_records=split_level_records,
            comparison_name="baseline_vs_hybrid",
            reference_model="baseline",
            candidate_model="hybrid",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["split_instance_id"], "split_a")
        self.assertAlmostEqual(records[0]["reference_value"], 0.50)
        self.assertAlmostEqual(records[0]["candidate_value"], 0.53)
        self.assertAlmostEqual(records[0]["delta"], 0.03)

        with tempfile.TemporaryDirectory() as csv_tmp_dir:
            csv_path = Path(csv_tmp_dir) / "baseline_vs_hybrid_paired_deltas.csv"
            df = write_paired_delta_csv(csv_path, records)
            self.assertEqual(
                list(df.columns[: len(PAIRED_DELTA_CSV_COLUMNS)]),
                list(PAIRED_DELTA_CSV_COLUMNS),
            )
            self.assertEqual(df["comparison_name"].tolist(), ["baseline_vs_hybrid"])

    def test_build_final_repeated_split_summary_payload_reports_means_cis_and_counts(self) -> None:
        split_manifest = {
            "study_id": "study_alpha",
            "dataset_fingerprint": "dataset-fp",
            "base_split_fingerprint": "trusted-single-split-fp",
            "split_policy": "repeated_stratified_train_val_test",
            "selection_metric": "val_dice_pos_mean",
            "split_instances": [
                {"split_instance_id": "split_a"},
                {"split_instance_id": "split_b"},
            ],
        }
        split_level_records = [
            {
                "study_id": "study_alpha",
                "split_instance_id": "split_a",
                "split_seed": 100,
                "model_name": "baseline",
                "model_type": "pretrained_resnet34_unet",
                "run_id": "baseline_split_a",
                "run_dir": "/tmp/baseline_split_a",
                "dataset_fingerprint": "dataset-fp",
                "base_split_fingerprint": "trusted-single-split-fp",
                "study_split_fingerprint": "study-fp-a",
                "run_split_fingerprint": "run-fp-a",
                "selection_metric": "val_dice_pos_mean",
                "selected_threshold": 0.5,
                "selected_postprocess": "none",
                "selection_state_path": "/tmp/baseline_split_a/selection/selection_state.yaml",
                "checkpoint_path": "/tmp/baseline_split_a/checkpoints/best_checkpoint.pth",
                "train_mask_variant": "dilated_masks",
                "eval_mask_variant": "original_masks",
                "test_dice_mean": 0.40,
                "test_dice_pos_mean": 0.50,
                "test_iou_mean": 0.30,
                "test_iou_pos_mean": 0.35,
            },
            {
                "study_id": "study_alpha",
                "split_instance_id": "split_b",
                "split_seed": 101,
                "model_name": "baseline",
                "model_type": "pretrained_resnet34_unet",
                "run_id": "baseline_split_b",
                "run_dir": "/tmp/baseline_split_b",
                "dataset_fingerprint": "dataset-fp",
                "base_split_fingerprint": "trusted-single-split-fp",
                "study_split_fingerprint": "study-fp-b",
                "run_split_fingerprint": "run-fp-b",
                "selection_metric": "val_dice_pos_mean",
                "selected_threshold": 0.6,
                "selected_postprocess": "none",
                "selection_state_path": "/tmp/baseline_split_b/selection/selection_state.yaml",
                "checkpoint_path": "/tmp/baseline_split_b/checkpoints/best_checkpoint.pth",
                "train_mask_variant": "dilated_masks",
                "eval_mask_variant": "original_masks",
                "test_dice_mean": 0.42,
                "test_dice_pos_mean": 0.60,
                "test_iou_mean": 0.32,
                "test_iou_pos_mean": 0.40,
            },
            {
                "study_id": "study_alpha",
                "split_instance_id": "split_a",
                "split_seed": 100,
                "model_name": "hybrid",
                "model_type": "hybrid",
                "run_id": "hybrid_split_a",
                "run_dir": "/tmp/hybrid_split_a",
                "dataset_fingerprint": "dataset-fp",
                "base_split_fingerprint": "trusted-single-split-fp",
                "study_split_fingerprint": "study-fp-a",
                "run_split_fingerprint": "run-fp-a",
                "selection_metric": "val_dice_pos_mean",
                "selected_threshold": 0.55,
                "selected_postprocess": "none",
                "selection_state_path": "/tmp/hybrid_split_a/selection/selection_state.yaml",
                "checkpoint_path": "/tmp/hybrid_split_a/checkpoints/best_checkpoint.pth",
                "train_mask_variant": "dilated_masks",
                "eval_mask_variant": "original_masks",
                "test_dice_mean": 0.43,
                "test_dice_pos_mean": 0.53,
                "test_iou_mean": 0.33,
                "test_iou_pos_mean": 0.39,
            },
            {
                "study_id": "study_alpha",
                "split_instance_id": "split_b",
                "split_seed": 101,
                "model_name": "hybrid",
                "model_type": "hybrid",
                "run_id": "hybrid_split_b",
                "run_dir": "/tmp/hybrid_split_b",
                "dataset_fingerprint": "dataset-fp",
                "base_split_fingerprint": "trusted-single-split-fp",
                "study_split_fingerprint": "study-fp-b",
                "run_split_fingerprint": "run-fp-b",
                "selection_metric": "val_dice_pos_mean",
                "selected_threshold": 0.65,
                "selected_postprocess": "none",
                "selection_state_path": "/tmp/hybrid_split_b/selection/selection_state.yaml",
                "checkpoint_path": "/tmp/hybrid_split_b/checkpoints/best_checkpoint.pth",
                "train_mask_variant": "dilated_masks",
                "eval_mask_variant": "original_masks",
                "test_dice_mean": 0.47,
                "test_dice_pos_mean": 0.67,
                "test_iou_mean": 0.37,
                "test_iou_pos_mean": 0.45,
            },
        ]
        paired_delta_records = build_paired_delta_records(
            split_level_records=split_level_records,
            comparison_name="baseline_vs_hybrid",
            reference_model="baseline",
            candidate_model="hybrid",
        )

        payload = build_final_repeated_split_summary_payload(
            split_manifest=split_manifest,
            split_level_records=split_level_records,
            paired_delta_records=paired_delta_records,
            bootstrap_samples=2000,
            bootstrap_seed=7,
        )

        self.assertEqual(payload["study_id"], "study_alpha")
        self.assertEqual(payload["primary_metric"], "test_dice_pos_mean")
        self.assertEqual(len(payload["model_summaries"]), 2)
        self.assertEqual(len(payload["paired_comparisons"]), 1)

        baseline_summary = next(
            item for item in payload["model_summaries"] if item["model_name"] == "baseline"
        )
        self.assertAlmostEqual(baseline_summary["mean"], 0.55)
        self.assertEqual(baseline_summary["contributing_split_count"], 2)
        self.assertEqual(
            baseline_summary["contributing_split_instance_ids"],
            ["split_a", "split_b"],
        )
        self.assertLessEqual(baseline_summary["ci_lower"], baseline_summary["mean"])
        self.assertGreaterEqual(baseline_summary["ci_upper"], baseline_summary["mean"])

        comparison_summary = payload["paired_comparisons"][0]
        self.assertEqual(comparison_summary["comparison_name"], "baseline_vs_hybrid")
        self.assertEqual(comparison_summary["metric_name"], "test_dice_pos_mean")
        self.assertAlmostEqual(comparison_summary["mean_delta"], 0.05)
        self.assertEqual(comparison_summary["contributing_split_count"], 2)
        self.assertEqual(
            comparison_summary["contributing_split_instance_ids"],
            ["split_a", "split_b"],
        )
        self.assertLessEqual(comparison_summary["ci_lower"], comparison_summary["mean_delta"])
        self.assertGreaterEqual(comparison_summary["ci_upper"], comparison_summary["mean_delta"])

    def test_write_final_repeated_split_summary_persists_yaml(self) -> None:
        split_manifest = {
            "study_id": "study_alpha",
            "dataset_fingerprint": "dataset-fp",
            "base_split_fingerprint": "trusted-single-split-fp",
            "split_policy": "repeated_stratified_train_val_test",
            "selection_metric": "val_dice_pos_mean",
            "split_instances": [
                {"split_instance_id": "split_a"},
            ],
        }
        split_level_records = [
            {
                "study_id": "study_alpha",
                "split_instance_id": "split_a",
                "split_seed": 100,
                "model_name": "baseline",
                "model_type": "pretrained_resnet34_unet",
                "run_id": "baseline_split_a",
                "run_dir": "/tmp/baseline_split_a",
                "dataset_fingerprint": "dataset-fp",
                "base_split_fingerprint": "trusted-single-split-fp",
                "study_split_fingerprint": "study-fp-a",
                "run_split_fingerprint": "run-fp-a",
                "selection_metric": "val_dice_pos_mean",
                "selected_threshold": 0.5,
                "selected_postprocess": "none",
                "selection_state_path": "/tmp/baseline_split_a/selection/selection_state.yaml",
                "checkpoint_path": "/tmp/baseline_split_a/checkpoints/best_checkpoint.pth",
                "train_mask_variant": "dilated_masks",
                "eval_mask_variant": "original_masks",
                "test_dice_mean": 0.40,
                "test_dice_pos_mean": 0.50,
                "test_iou_mean": 0.30,
                "test_iou_pos_mean": 0.35,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary" / "final_summary.yaml"
            payload = write_final_repeated_split_summary(
                summary_path,
                split_manifest=split_manifest,
                split_level_records=split_level_records,
                paired_delta_records=[],
                bootstrap_samples=100,
                bootstrap_seed=11,
            )

            self.assertTrue(summary_path.exists())
            written = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(written["study_id"], "study_alpha")
            self.assertEqual(written["bootstrap_samples"], 100)
            self.assertEqual(payload["model_summaries"][0]["contributing_split_count"], 1)

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
