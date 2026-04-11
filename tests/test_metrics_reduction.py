"""Unit tests for segmentation metric reductions and empty-mask policy."""

from __future__ import annotations

import math
import unittest

import torch

from src.training.metrics import (
    compute_binary_segmentation_stats,
    dice_score,
    f1_score,
    iou_score,
    precision_score,
    recall_score,
)


class TestMetricReductions(unittest.TestCase):
    def setUp(self) -> None:
        self.pred = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        self.target = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )

    def test_confusion_stats_and_positive_mask(self) -> None:
        stats = compute_binary_segmentation_stats(self.pred, self.target)
        self.assertTrue(torch.equal(stats["tp"], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.equal(stats["fp"], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.equal(stats["fn"], torch.tensor([0.0, 1.0, 1.0])))
        self.assertTrue(torch.equal(stats["positive_target_mask"], torch.tensor([False, True, True])))

    def test_dice_reductions_are_distinct_and_correct(self) -> None:
        per_image = dice_score(self.pred, self.target, reduction="none")
        self.assertTrue(torch.allclose(per_image, torch.tensor([1.0, 0.0, 0.5])))
        self.assertAlmostEqual(dice_score(self.pred, self.target, reduction="mean").item(), 0.5, places=6)
        self.assertAlmostEqual(
            dice_score(self.pred, self.target, reduction="positive_mean").item(),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(dice_score(self.pred, self.target, reduction="micro").item(), 0.4, places=6)

    def test_iou_precision_recall_and_f1_match_expected_policy(self) -> None:
        self.assertAlmostEqual(iou_score(self.pred, self.target, reduction="mean").item(), (1.0 + 0.0 + (1.0 / 3.0)) / 3.0, places=6)
        self.assertAlmostEqual(precision_score(self.pred, self.target, reduction="mean").item(), 0.5, places=6)
        self.assertAlmostEqual(recall_score(self.pred, self.target, reduction="mean").item(), 0.5, places=6)
        self.assertAlmostEqual(f1_score(self.pred, self.target, reduction="mean").item(), 0.5, places=6)
        self.assertAlmostEqual(precision_score(self.pred, self.target, reduction="micro").item(), 0.5, places=6)
        self.assertAlmostEqual(recall_score(self.pred, self.target, reduction="micro").item(), 1.0 / 3.0, places=6)

    def test_empty_mask_policy_is_exact_match_for_all_overlap_metrics(self) -> None:
        empty = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        positive = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)

        for metric in [dice_score, iou_score, precision_score, recall_score, f1_score]:
            self.assertEqual(metric(empty, empty, reduction="mean").item(), 1.0)
            self.assertEqual(metric(empty, positive, reduction="mean").item(), 0.0)
            self.assertEqual(metric(positive, empty, reduction="mean").item(), 0.0)

    def test_positive_mean_returns_nan_when_no_positive_targets_exist(self) -> None:
        empty = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
        result = dice_score(empty, empty, reduction="positive_mean").item()
        self.assertTrue(math.isnan(result))

    def test_invalid_reduction_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown reduction"):
            dice_score(self.pred, self.target, reduction="batch_mean")


if __name__ == "__main__":
    unittest.main()
