"""Regression tests for repeated-split study finalization."""

from __future__ import annotations

import tempfile
import runpy
import unittest
from pathlib import Path

import pandas as pd
import yaml


def load_runner_module() -> dict:
    return runpy.run_path("scripts/finalize_repeated_split_pretrained_study.py")


class TestFinalizeRepeatedSplitPretrainedStudyRunner(unittest.TestCase):
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
        checkpoints_dir = run_dir / "checkpoints"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        selection_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / "best_checkpoint.pth").write_bytes(b"checkpoint")

        with (metadata_dir / "run_metadata.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "run_id": run_id,
                    "split_fingerprint": split_fingerprint,
                },
                handle,
                sort_keys=False,
            )

        selection_state_path = selection_dir / "selection_state.yaml"
        selection_state_path.write_text("selection_state: placeholder\n", encoding="utf-8")

        with (reports_dir / "test_summary.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "split": "test",
                    "model_type": model_type,
                    "checkpoint_path": str((checkpoints_dir / "best_checkpoint.pth").resolve()),
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
                },
                handle,
                sort_keys=False,
            )

    def test_finalize_repeated_split_study_writes_split_level_comparison_and_summary(self) -> None:
        module = load_runner_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            study_dir = root / "artifacts" / "repeated_splits" / "study_alpha"
            metadata_dir = study_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            dataset_root = root / "data" / "processed" / "trusted_v1"
            dataset_root.mkdir(parents=True, exist_ok=True)

            split_manifest_path = metadata_dir / "split_manifest.yaml"
            split_manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "study_id": "study_alpha",
                        "dataset_root": str(dataset_root.resolve()),
                        "dataset_fingerprint": "dataset-fp",
                        "base_split_fingerprint": "trusted-base-split-fp",
                        "split_policy": "repeated_stratified_train_val_test",
                        "selection_metric": "val_dice_pos_mean",
                        "split_instances": [
                            {
                                "split_instance_id": "split_001",
                                "split_seed": 42,
                                "split_fingerprint": "study-fp-001",
                                "train_ids": ["img_001", "img_002"],
                                "val_ids": ["img_003"],
                                "test_ids": ["img_004"],
                            },
                            {
                                "split_instance_id": "split_002",
                                "split_seed": 43,
                                "split_fingerprint": "study-fp-002",
                                "train_ids": ["img_005", "img_006"],
                                "val_ids": ["img_007"],
                                "test_ids": ["img_008"],
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            runs_root = root / "artifacts" / "runs"
            baseline_run_a = runs_root / "study_alpha__split_001__pretrained_resnet34_unet"
            baseline_run_b = runs_root / "study_alpha__split_002__pretrained_resnet34_unet"
            hybrid_run_a = runs_root / "study_alpha__split_001__hybrid"
            hybrid_run_b = runs_root / "study_alpha__split_002__hybrid"

            self.write_authoritative_run_package(
                run_dir=baseline_run_a,
                run_id="baseline_split_001",
                model_type="pretrained_resnet34_unet",
                split_fingerprint="study-fp-001",
                dataset_root=dataset_root,
                test_dice_mean=0.40,
                test_dice_pos_mean=0.50,
                test_iou_mean=0.30,
                test_iou_pos_mean=0.35,
            )
            self.write_authoritative_run_package(
                run_dir=baseline_run_b,
                run_id="baseline_split_002",
                model_type="pretrained_resnet34_unet",
                split_fingerprint="study-fp-002",
                dataset_root=dataset_root,
                test_dice_mean=0.42,
                test_dice_pos_mean=0.60,
                test_iou_mean=0.32,
                test_iou_pos_mean=0.40,
            )
            self.write_authoritative_run_package(
                run_dir=hybrid_run_a,
                run_id="hybrid_split_001",
                model_type="hybrid",
                split_fingerprint="study-fp-001",
                dataset_root=dataset_root,
                selected_threshold=0.55,
                test_dice_mean=0.43,
                test_dice_pos_mean=0.53,
                test_iou_mean=0.33,
                test_iou_pos_mean=0.39,
            )
            self.write_authoritative_run_package(
                run_dir=hybrid_run_b,
                run_id="hybrid_split_002",
                model_type="hybrid",
                split_fingerprint="study-fp-002",
                dataset_root=dataset_root,
                selected_threshold=0.65,
                test_dice_mean=0.47,
                test_dice_pos_mean=0.67,
                test_iou_mean=0.37,
                test_iou_pos_mean=0.45,
            )

            pretrained_inventory_path = metadata_dir / "pretrained_resnet34_run_inventory.yaml"
            pretrained_inventory_path.write_text(
                yaml.safe_dump(
                    {
                        "study_id": "study_alpha",
                        "study_manifest_path": str(split_manifest_path.resolve()),
                        "base_config_path": str((root / "configs" / "pretrained.yaml").resolve()),
                        "stage": "select_test",
                        "model_name": "pretrained_resnet34_unet",
                        "entries": [
                            {
                                "split_instance_id": "split_001",
                                "split_seed": 42,
                                "split_override_path": str((metadata_dir / "split_overrides" / "split_001.json").resolve()),
                                "config_override_path": str((metadata_dir / "config_overrides" / "split_001.yaml").resolve()),
                                "run_dir": str(baseline_run_a.resolve()),
                            },
                            {
                                "split_instance_id": "split_002",
                                "split_seed": 43,
                                "split_override_path": str((metadata_dir / "split_overrides" / "split_002.json").resolve()),
                                "config_override_path": str((metadata_dir / "config_overrides" / "split_002.yaml").resolve()),
                                "run_dir": str(baseline_run_b.resolve()),
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            hybrid_inventory_path = metadata_dir / "hybrid_run_inventory.yaml"
            hybrid_inventory_path.write_text(
                yaml.safe_dump(
                    {
                        "study_id": "study_alpha",
                        "study_manifest_path": str(split_manifest_path.resolve()),
                        "base_config_path": str((root / "configs" / "hybrid.yaml").resolve()),
                        "stage": "select_test",
                        "model_name": "hybrid",
                        "entries": [
                            {
                                "split_instance_id": "split_001",
                                "split_seed": 42,
                                "split_override_path": str((metadata_dir / "split_overrides" / "split_001.json").resolve()),
                                "config_override_path": str((metadata_dir / "config_overrides" / "split_001_hybrid.yaml").resolve()),
                                "run_dir": str(hybrid_run_a.resolve()),
                            },
                            {
                                "split_instance_id": "split_002",
                                "split_seed": 43,
                                "split_override_path": str((metadata_dir / "split_overrides" / "split_002.json").resolve()),
                                "config_override_path": str((metadata_dir / "config_overrides" / "split_002_hybrid.yaml").resolve()),
                                "run_dir": str(hybrid_run_b.resolve()),
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            payload = module["finalize_repeated_split_study"](
                run_inventory_paths=[pretrained_inventory_path, hybrid_inventory_path],
                comparison_specs=[
                    "baseline_vs_hybrid:pretrained_resnet34_unet:hybrid",
                ],
                bootstrap_samples=2000,
                bootstrap_seed=7,
            )

            split_level_path = study_dir / "aggregations" / "split_level_metrics.csv"
            comparison_path = study_dir / "comparisons" / "baseline_vs_hybrid_paired_deltas.csv"
            summary_path = study_dir / "summary" / "final_summary.yaml"

            self.assertEqual(payload["split_level_table_path"], str(split_level_path))
            self.assertEqual(payload["final_summary_path"], str(summary_path))
            self.assertEqual(payload["split_level_row_count"], 4)
            self.assertEqual(payload["summary_model_count"], 2)
            self.assertEqual(payload["summary_comparison_count"], 1)
            self.assertEqual(payload["paired_delta_table_paths"], [str(comparison_path)])

            split_level_df = pd.read_csv(split_level_path)
            self.assertEqual(split_level_df["model_name"].tolist(), [
                "hybrid",
                "pretrained_resnet34_unet",
                "hybrid",
                "pretrained_resnet34_unet",
            ])
            self.assertEqual(sorted(split_level_df["split_instance_id"].unique().tolist()), ["split_001", "split_002"])

            comparison_df = pd.read_csv(comparison_path)
            self.assertEqual(comparison_df["comparison_name"].tolist(), ["baseline_vs_hybrid", "baseline_vs_hybrid"])
            self.assertAlmostEqual(comparison_df["delta"].iloc[0], 0.03)
            self.assertAlmostEqual(comparison_df["delta"].iloc[1], 0.07)

            summary_payload = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["study_id"], "study_alpha")
            self.assertEqual(summary_payload["bootstrap_samples"], 2000)
            self.assertEqual(len(summary_payload["model_summaries"]), 2)
            self.assertEqual(len(summary_payload["paired_comparisons"]), 1)
            self.assertAlmostEqual(summary_payload["paired_comparisons"][0]["mean_delta"], 0.05)


if __name__ == "__main__":
    unittest.main()
