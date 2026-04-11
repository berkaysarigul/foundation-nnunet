"""Regression tests for validation-only threshold selection policy."""

from __future__ import annotations

import unittest

import torch

from src.evaluation.evaluate import (
    resolve_threshold_selection_config,
    tune_threshold_on_validation_predictions,
)


class TestThresholdSelection(unittest.TestCase):
    def make_cfg(
        self,
        *,
        metric: str = "val_dice_pos_mean",
        threshold_candidates: list[float] | None = None,
        postprocess: str = "none",
    ) -> dict:
        return {
            "selection": {
                "metric": metric,
                "threshold_candidates": (
                    threshold_candidates if threshold_candidates is not None else [0.3, 0.5, 0.7]
                ),
                "postprocess": postprocess,
            }
        }

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


if __name__ == "__main__":
    unittest.main()
