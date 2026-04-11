"""Tests for trainer-side validation aggregation over per-image overlap metrics."""

from __future__ import annotations

import unittest

import torch

from src.training.metrics import dice_score, iou_score
from src.training.trainer import compute_validation_overlap_totals


class TestTrainerValidationAggregation(unittest.TestCase):
    def test_overlap_totals_match_per_image_none_reduction(self) -> None:
        preds = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )

        totals = compute_validation_overlap_totals(preds, masks)
        expected_dice = dice_score(preds, masks, reduction="none")
        expected_iou = iou_score(preds, masks, reduction="none")

        self.assertEqual(totals["image_count"], 2)
        self.assertAlmostEqual(totals["dice_sum"], expected_dice.sum().item(), places=6)
        self.assertAlmostEqual(totals["iou_sum"], expected_iou.sum().item(), places=6)

    def test_aggregated_means_follow_per_image_mean_not_batch_micro(self) -> None:
        preds = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )

        totals = compute_validation_overlap_totals(preds, masks)
        dice_mean = totals["dice_sum"] / totals["image_count"]
        iou_mean = totals["iou_sum"] / totals["image_count"]

        self.assertAlmostEqual(dice_mean, dice_score(preds, masks, reduction="mean").item(), places=6)
        self.assertAlmostEqual(iou_mean, iou_score(preds, masks, reduction="mean").item(), places=6)
        self.assertNotAlmostEqual(dice_mean, dice_score(preds, masks, reduction="micro").item(), places=6)


if __name__ == "__main__":
    unittest.main()
