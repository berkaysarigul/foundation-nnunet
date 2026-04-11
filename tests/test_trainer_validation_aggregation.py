"""Tests for trainer-side validation aggregation over per-image overlap metrics."""

from __future__ import annotations

import unittest

import torch

from src.training.metrics import dice_score, iou_score
from src.training.trainer import (
    compute_positive_validation_dice_totals,
    compute_validation_overlap_totals,
)


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

    def test_positive_validation_totals_follow_positive_image_mean(self) -> None:
        batch1_preds = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        batch1_masks = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        batch2_preds = torch.tensor(
            [
                [[[0.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        batch2_masks = torch.tensor(
            [
                [[[0.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )

        totals1 = compute_positive_validation_dice_totals(batch1_preds, batch1_masks)
        totals2 = compute_positive_validation_dice_totals(batch2_preds, batch2_masks)
        combined_sum = totals1["dice_sum"] + totals2["dice_sum"]
        combined_count = totals1["positive_image_count"] + totals2["positive_image_count"]
        aggregated_positive_mean = combined_sum / combined_count

        all_preds = torch.cat([batch1_preds, batch2_preds], dim=0)
        all_masks = torch.cat([batch1_masks, batch2_masks], dim=0)
        expected_positive_mean = dice_score(all_preds, all_masks, reduction="positive_mean").item()

        old_batch_mean = (
            dice_score(batch1_preds[batch1_masks.sum(dim=(1, 2, 3)) > 0], batch1_masks[batch1_masks.sum(dim=(1, 2, 3)) > 0]).item()
            + dice_score(batch2_preds[batch2_masks.sum(dim=(1, 2, 3)) > 0], batch2_masks[batch2_masks.sum(dim=(1, 2, 3)) > 0]).item()
        ) / 2.0

        self.assertEqual(combined_count, 3)
        self.assertAlmostEqual(aggregated_positive_mean, expected_positive_mean, places=6)
        self.assertNotAlmostEqual(aggregated_positive_mean, old_batch_mean, places=6)


if __name__ == "__main__":
    unittest.main()
