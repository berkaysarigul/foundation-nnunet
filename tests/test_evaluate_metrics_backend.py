"""Tests for evaluator-side wiring to the shared metric backend."""

from __future__ import annotations

import unittest

import torch

from src.evaluation.evaluate import compute_per_image_metrics
from src.training.metrics import (
    dice_score,
    f1_score,
    hausdorff_distance,
    iou_score,
    precision_score,
    recall_score,
)


class TestEvaluateMetricBackend(unittest.TestCase):
    def test_compute_per_image_metrics_uses_none_reduction(self) -> None:
        pred = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32)
        mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)

        metrics = compute_per_image_metrics(pred, mask, threshold=0.5)

        self.assertAlmostEqual(
            metrics["dice"],
            dice_score(pred, mask, threshold=0.5, reduction="none").squeeze(0).item(),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["iou"],
            iou_score(pred, mask, threshold=0.5, reduction="none").squeeze(0).item(),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["precision"],
            precision_score(pred, mask, threshold=0.5, reduction="none").squeeze(0).item(),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["recall"],
            recall_score(pred, mask, threshold=0.5, reduction="none").squeeze(0).item(),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["f1"],
            f1_score(pred, mask, threshold=0.5, reduction="none").squeeze(0).item(),
            places=6,
        )
        self.assertEqual(metrics["hausdorff"], hausdorff_distance(pred, mask, threshold=0.5))

    def test_compute_per_image_metrics_preserves_empty_mask_policy(self) -> None:
        empty = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        metrics = compute_per_image_metrics(empty, empty)

        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
