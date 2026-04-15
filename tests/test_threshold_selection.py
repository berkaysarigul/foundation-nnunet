"""Regression tests for threshold selection, persistence, and reuse."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluation.evaluate import (
    load_selection_state,
    resolve_test_evaluation_selection,
    resolve_threshold_selection_config,
    save_selection_state,
    tune_threshold_on_validation_predictions,
)


class TestThresholdSelection(unittest.TestCase):
    def make_cfg(
        self,
        *,
        metric: str = "val_dice_pos_mean",
        threshold_candidates: list[float] | None = None,
        postprocess: str = "none",
        processed_dir: str = "data/processed/pneumothorax_trusted_v1",
        train_mask_variant: str = "dilated_masks",
        eval_mask_variant: str = "original_masks",
        input_size: int = 512,
    ) -> dict:
        return {
            "selection": {
                "metric": metric,
                "threshold_candidates": (
                    threshold_candidates if threshold_candidates is not None else [0.3, 0.5, 0.7]
                ),
                "postprocess": postprocess,
            },
            "data": {
                "processed_dir": processed_dir,
                "train_mask_variant": train_mask_variant,
                "eval_mask_variant": eval_mask_variant,
                "input_size": input_size,
            },
        }

    def make_selection_result(self, cfg: dict | None = None) -> dict:
        preds = torch.tensor(
            [
                [[[0.60, 0.20], [0.20, 0.20]]],
                [[[0.60, 0.60], [0.60, 0.60]]],
                [[[0.60, 0.60], [0.60, 0.60]]],
                [[[0.60, 0.60], [0.60, 0.60]]],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        return tune_threshold_on_validation_predictions(
            preds,
            masks,
            cfg if cfg is not None else self.make_cfg(threshold_candidates=[0.5, 0.7]),
        )

    def test_resolve_threshold_selection_config_normalizes_and_sorts(self) -> None:
        cfg = self.make_cfg(
            metric="val_dice_pos_mean",
            threshold_candidates=[0.7, 0.5, 0.3, 0.5],
            postprocess="off",
        )

        self.assertEqual(
            resolve_threshold_selection_config(cfg),
            {
                "metric": "val_dice_pos_mean",
                "postprocess": "none",
                "threshold_candidates": [0.3, 0.5, 0.7],
            },
        )

    def test_threshold_selection_rejects_unsupported_surface(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection.metric"):
            resolve_threshold_selection_config(self.make_cfg(metric="val_iou_mean"))

        with self.assertRaisesRegex(ValueError, "selection.postprocess"):
            resolve_threshold_selection_config(self.make_cfg(postprocess="min_area"))

        with self.assertRaisesRegex(ValueError, "must include 0.5"):
            resolve_threshold_selection_config(
                self.make_cfg(threshold_candidates=[0.2, 0.4, 0.6])
            )

    def test_threshold_selection_requires_validation_split(self) -> None:
        preds = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        masks = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "validation data only"):
            tune_threshold_on_validation_predictions(preds, masks, self.make_cfg(), split="test")

    def test_threshold_selection_optimizes_positive_only_dice(self) -> None:
        preds = torch.tensor(
            [
                [[[0.60, 0.20], [0.20, 0.20]]],
                [[[0.60, 0.60], [0.60, 0.60]]],
                [[[0.60, 0.60], [0.60, 0.60]]],
                [[[0.60, 0.60], [0.60, 0.60]]],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        cfg = self.make_cfg(threshold_candidates=[0.5, 0.7])

        selection = tune_threshold_on_validation_predictions(preds, masks, cfg)
        summary_by_threshold = {
            row["threshold"]: row for row in selection["threshold_summary"]
        }

        self.assertEqual(selection["selection_metric"], "val_dice_pos_mean")
        self.assertEqual(selection["selected_postprocess"], "none")
        self.assertAlmostEqual(selection["selected_threshold"], 0.5)
        self.assertGreater(
            summary_by_threshold[0.5]["val_dice_pos_mean"],
            summary_by_threshold[0.7]["val_dice_pos_mean"],
        )
        self.assertLess(
            summary_by_threshold[0.5]["val_dice_mean"],
            summary_by_threshold[0.7]["val_dice_mean"],
        )

    def test_threshold_selection_prefers_legacy_default_on_exact_tie(self) -> None:
        preds = torch.tensor(
            [
                [[[0.60, 0.60], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor(
            [
                [[[1.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        cfg = self.make_cfg(threshold_candidates=[0.4, 0.5, 0.6])

        selection = tune_threshold_on_validation_predictions(preds, masks, cfg)

        self.assertAlmostEqual(selection["selected_threshold"], 0.5)

    def test_selection_state_round_trip_and_test_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            selection_path = tmp_path / "run_001" / "selection" / "selection_state.yaml"
            checkpoint_path = tmp_path / "checkpoints" / "best_baseline.pth"
            processed_dir = tmp_path / "data" / "processed" / "trusted_v1"
            cfg = self.make_cfg(processed_dir=str(processed_dir))
            selection_result = self.make_selection_result(cfg)

            saved_state = save_selection_state(
                selection_path,
                selection_result,
                cfg,
                checkpoint_path=str(checkpoint_path),
                model_type="baseline",
            )
            loaded_state = load_selection_state(selection_path)
            selected_threshold, resolved_state = resolve_test_evaluation_selection(
                cfg,
                checkpoint_path=str(checkpoint_path),
                model_type="baseline",
                selection_state_path=selection_path,
            )

            self.assertEqual(loaded_state["selection_metric"], "val_dice_pos_mean")
            self.assertEqual(loaded_state["selected_postprocess"], "none")
            self.assertAlmostEqual(saved_state["selected_threshold"], 0.5)
            self.assertAlmostEqual(selected_threshold, saved_state["selected_threshold"])
            self.assertEqual(resolved_state["dataset_root"], loaded_state["dataset_root"])
            self.assertEqual(
                loaded_state["selection_state_path"],
                str(selection_path.resolve()),
            )
            self.assertEqual(loaded_state["train_mask_variant"], "dilated_masks")

    def test_selection_state_requires_authoritative_path_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bad_path = tmp_path / "selection_state.yaml"
            cfg = self.make_cfg()
            selection_result = self.make_selection_result(cfg)

            with self.assertRaisesRegex(ValueError, "selection"):
                save_selection_state(
                    bad_path,
                    selection_result,
                    cfg,
                    checkpoint_path=str(tmp_path / "best_baseline.pth"),
                    model_type="baseline",
                )

    def test_test_reuse_rejects_mismatched_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            selection_path = tmp_path / "run_001" / "selection" / "selection_state.yaml"
            checkpoint_path = tmp_path / "checkpoints" / "best_baseline.pth"
            cfg = self.make_cfg(processed_dir=str(tmp_path / "data" / "processed" / "trusted_v1"))
            selection_result = self.make_selection_result(cfg)
            save_selection_state(
                selection_path,
                selection_result,
                cfg,
                checkpoint_path=str(checkpoint_path),
                model_type="baseline",
            )

            mismatched_cfg = self.make_cfg(
                processed_dir=str(tmp_path / "data" / "processed" / "other_v1")
            )

            with self.assertRaisesRegex(ValueError, "dataset_root"):
                resolve_test_evaluation_selection(
                    mismatched_cfg,
                    checkpoint_path=str(checkpoint_path),
                    model_type="baseline",
                    selection_state_path=selection_path,
                )

            mismatched_mask_cfg = self.make_cfg(
                processed_dir=str(tmp_path / "data" / "processed" / "trusted_v1"),
                train_mask_variant="original_masks",
            )

            with self.assertRaisesRegex(ValueError, "train_mask_variant"):
                resolve_test_evaluation_selection(
                    mismatched_mask_cfg,
                    checkpoint_path=str(checkpoint_path),
                    model_type="baseline",
                    selection_state_path=selection_path,
                )

            with self.assertRaisesRegex(ValueError, "requires --selection_state_input"):
                resolve_test_evaluation_selection(
                    cfg,
                    checkpoint_path=str(checkpoint_path),
                    model_type="baseline",
                    selection_state_path=None,
                )


if __name__ == "__main__":
    unittest.main()
