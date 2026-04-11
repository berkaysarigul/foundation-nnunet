"""Regression tests proving trainer/evaluator parity on the same saved predictions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluation.evaluate import compute_per_image_metrics
from src.training.trainer import (
    compute_positive_validation_dice_totals,
    compute_validation_overlap_totals,
)


class TestTrainerEvaluatorParity(unittest.TestCase):
    def test_trainer_and_evaluator_match_on_same_saved_predictions(self) -> None:
        preds = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture_path = Path(tmp_dir) / "saved_predictions.pt"
            torch.save({"preds": preds, "masks": masks}, fixture_path)
            saved = torch.load(fixture_path, weights_only=False)

        saved_preds = saved["preds"]
        saved_masks = saved["masks"]

        trainer_dice_sum = 0.0
        trainer_iou_sum = 0.0
        trainer_image_count = 0
        trainer_positive_dice_sum = 0.0
        trainer_positive_count = 0

        for batch_preds, batch_masks in (
            (saved_preds[:3], saved_masks[:3]),
            (saved_preds[3:], saved_masks[3:]),
        ):
            overlap_totals = compute_validation_overlap_totals(batch_preds, batch_masks)
            positive_totals = compute_positive_validation_dice_totals(batch_preds, batch_masks)
            trainer_dice_sum += overlap_totals["dice_sum"]
            trainer_iou_sum += overlap_totals["iou_sum"]
            trainer_image_count += overlap_totals["image_count"]
            trainer_positive_dice_sum += positive_totals["dice_sum"]
            trainer_positive_count += positive_totals["positive_image_count"]

        evaluator_records = [
            compute_per_image_metrics(
                saved_preds[idx : idx + 1],
                saved_masks[idx : idx + 1],
            )
            for idx in range(saved_preds.shape[0])
        ]
        evaluator_positive_mask = [
            bool(saved_masks[idx].sum().item() > 0) for idx in range(saved_masks.shape[0])
        ]

        evaluator_dice_mean = sum(record["dice"] for record in evaluator_records) / len(evaluator_records)
        evaluator_iou_mean = sum(record["iou"] for record in evaluator_records) / len(evaluator_records)
        evaluator_positive_dice = [
            record["dice"]
            for record, is_positive in zip(evaluator_records, evaluator_positive_mask)
            if is_positive
        ]
        evaluator_positive_mean = sum(evaluator_positive_dice) / len(evaluator_positive_dice)

        self.assertEqual(trainer_image_count, len(evaluator_records))
        self.assertEqual(trainer_positive_count, len(evaluator_positive_dice))
        self.assertAlmostEqual(
            trainer_dice_sum / trainer_image_count,
            evaluator_dice_mean,
            places=6,
        )
        self.assertAlmostEqual(
            trainer_iou_sum / trainer_image_count,
            evaluator_iou_mean,
            places=6,
        )
        self.assertAlmostEqual(
            trainer_positive_dice_sum / trainer_positive_count,
            evaluator_positive_mean,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
