"""Sanity tests for PR-10C losses and metrics."""

from __future__ import annotations

import unittest

import torch

from src.training.losses import BCEDiceLoss, DiceLoss
from src.training.metrics import (
    aggregate_per_case_binary_metrics,
    case_level_f1_score,
    compute_per_case_binary_metrics,
    negative_false_positive_rate,
    specificity_score,
)


class TestFxPriorRefinerLosses(unittest.TestCase):
    def test_logits_dice_loss_is_low_for_good_prediction(self) -> None:
        logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        loss = DiceLoss()(logits, target)
        self.assertLess(float(loss.item()), 0.01)

    def test_bce_dice_loss_runs(self) -> None:
        logits = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
        target = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
        loss = BCEDiceLoss()(logits, target)
        self.assertGreater(float(loss.item()), 0.0)


class TestFxPriorRefinerMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.pred = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],  # TP case
                [[[0.0, 0.0], [0.0, 0.0]]],  # FN case
                [[[1.0, 0.0], [0.0, 0.0]]],  # FP case
                [[[0.0, 0.0], [0.0, 0.0]]],  # TN case
            ],
            dtype=torch.float32,
        )
        self.target = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )

    def test_specificity_case_f1_and_negative_fpr(self) -> None:
        self.assertAlmostEqual(float(specificity_score(self.pred, self.target).item()), 13.0 / 14.0)
        self.assertAlmostEqual(case_level_f1_score(self.pred, self.target), 0.5)
        self.assertAlmostEqual(negative_false_positive_rate(self.pred, self.target), 0.5)

    def test_per_case_aggregate_metrics(self) -> None:
        rows = compute_per_case_binary_metrics(self.pred, self.target)
        agg = aggregate_per_case_binary_metrics(rows)
        self.assertEqual(agg["tp"], 1)
        self.assertEqual(agg["fn"], 1)
        self.assertEqual(agg["fp"], 1)
        self.assertEqual(agg["tn"], 1)
        self.assertAlmostEqual(agg["case_level_f1"], 0.5)
        self.assertAlmostEqual(agg["negative_case_false_positive_rate"], 0.5)
        self.assertAlmostEqual(agg["positive_dice"], 0.5)


if __name__ == "__main__":
    unittest.main()
