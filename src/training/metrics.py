"""
metrics.py - Segmentation evaluation metrics.

Overlap metrics support explicit reduction modes:
- "micro": aggregate TP/FP/FN across the batch first
- "mean": compute per-image scores then average across all images
- "positive_mean": compute per-image scores then average only across images
  whose target mask contains foreground
- "none": return the per-image metric tensor directly

Empty-mask policy for overlap metrics:
- pred empty and target empty -> score 1.0
- pred empty and target positive -> score 0.0
- pred positive and target empty -> score 0.0

Current trainer/evaluator alignment work is tracked separately in recovery tasks.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

_REDUCTIONS = {"micro", "mean", "positive_mean", "none"}


def _validate_reduction(reduction: str) -> str:
    if reduction not in _REDUCTIONS:
        raise ValueError(
            f"Unknown reduction '{reduction}'. Expected one of {sorted(_REDUCTIONS)}."
        )
    return reduction


def compute_binary_segmentation_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return per-image TP/FP/FN/TN counts for binary segmentation tensors."""
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"Expected pred/target with shape (B, 1, H, W); got {tuple(pred.shape)} and {tuple(target.shape)}."
        )
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape; got {tuple(pred.shape)} vs {tuple(target.shape)}."
        )

    pred_binary = (pred > threshold).to(dtype=torch.float32)
    target_binary = (target > 0.5).to(dtype=torch.float32)

    pred_flat = pred_binary.reshape(pred_binary.shape[0], -1)
    target_flat = target_binary.reshape(target_binary.shape[0], -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    fp = pred_sum - tp
    fn = target_sum - tp
    total = torch.full_like(tp, pred_flat.shape[1], dtype=torch.float32)
    tn = total - tp - fp - fn

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_sum": pred_sum,
        "target_sum": target_sum,
        "positive_target_mask": target_sum > 0,
    }


def _per_image_overlap_metric(
    stats: dict[str, torch.Tensor],
    metric_name: str,
) -> torch.Tensor:
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    pred_sum = stats["pred_sum"]
    target_sum = stats["target_sum"]

    if metric_name in {"dice", "f1"}:
        numerator = 2.0 * tp
        denominator = pred_sum + target_sum
    elif metric_name == "iou":
        numerator = tp
        denominator = tp + fp + fn
    elif metric_name == "precision":
        numerator = tp
        denominator = pred_sum
    elif metric_name == "recall":
        numerator = tp
        denominator = target_sum
    else:
        raise ValueError(f"Unsupported overlap metric '{metric_name}'.")

    result = torch.zeros_like(numerator, dtype=torch.float32)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    empty_match = (pred_sum == 0) & (target_sum == 0)
    result[empty_match] = 1.0
    return result


def _reduce_overlap_metric(
    per_image: torch.Tensor,
    stats: dict[str, torch.Tensor],
    metric_name: str,
    reduction: str,
) -> torch.Tensor:
    reduction = _validate_reduction(reduction)

    if reduction == "none":
        return per_image

    if reduction == "mean":
        return per_image.mean()

    if reduction == "positive_mean":
        positive_mask = stats["positive_target_mask"]
        if positive_mask.any():
            return per_image[positive_mask].mean()
        return torch.full((), float("nan"), dtype=per_image.dtype, device=per_image.device)

    micro_stats = {
        key: value.sum()
        for key, value in stats.items()
        if key in {"tp", "fp", "fn", "tn", "pred_sum", "target_sum"}
    }
    return _per_image_overlap_metric(
        {
            "tp": micro_stats["tp"].reshape(1),
            "fp": micro_stats["fp"].reshape(1),
            "fn": micro_stats["fn"].reshape(1),
            "tn": micro_stats["tn"].reshape(1),
            "pred_sum": micro_stats["pred_sum"].reshape(1),
            "target_sum": micro_stats["target_sum"].reshape(1),
            "positive_target_mask": micro_stats["target_sum"].reshape(1) > 0,
        },
        metric_name,
    ).squeeze(0)


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    reduction: str = "micro",
) -> torch.Tensor:
    """Dice similarity coefficient with explicit reduction control."""
    stats = compute_binary_segmentation_stats(pred, target, threshold=threshold)
    per_image = _per_image_overlap_metric(stats, metric_name="dice")
    return _reduce_overlap_metric(per_image, stats, metric_name="dice", reduction=reduction)


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    reduction: str = "micro",
) -> torch.Tensor:
    """Intersection over Union (Jaccard index) with explicit reduction control."""
    stats = compute_binary_segmentation_stats(pred, target, threshold=threshold)
    per_image = _per_image_overlap_metric(stats, metric_name="iou")
    return _reduce_overlap_metric(per_image, stats, metric_name="iou", reduction=reduction)


def precision_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    reduction: str = "micro",
) -> torch.Tensor:
    """Pixel precision with exact-match empty-mask handling."""
    stats = compute_binary_segmentation_stats(pred, target, threshold=threshold)
    per_image = _per_image_overlap_metric(stats, metric_name="precision")
    return _reduce_overlap_metric(per_image, stats, metric_name="precision", reduction=reduction)


def recall_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    reduction: str = "micro",
) -> torch.Tensor:
    """Pixel recall with exact-match empty-mask handling."""
    stats = compute_binary_segmentation_stats(pred, target, threshold=threshold)
    per_image = _per_image_overlap_metric(stats, metric_name="recall")
    return _reduce_overlap_metric(per_image, stats, metric_name="recall", reduction=reduction)


def f1_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    reduction: str = "micro",
) -> torch.Tensor:
    """Pixel F1 score with exact-match empty-mask handling."""
    stats = compute_binary_segmentation_stats(pred, target, threshold=threshold)
    per_image = _per_image_overlap_metric(stats, metric_name="f1")
    return _reduce_overlap_metric(per_image, stats, metric_name="f1", reduction=reduction)


def specificity_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    reduction: str = "micro",
) -> torch.Tensor:
    """Pixel specificity (true negative rate) with explicit reduction control."""
    stats = compute_binary_segmentation_stats(pred, target, threshold=threshold)
    tn = stats["tn"]
    fp = stats["fp"]
    denominator = tn + fp
    per_image = torch.ones_like(tn, dtype=torch.float32)
    valid = denominator > 0
    per_image[valid] = tn[valid] / denominator[valid]

    reduction = _validate_reduction(reduction)
    if reduction == "none":
        return per_image
    if reduction == "mean":
        return per_image.mean()
    if reduction == "positive_mean":
        positive_mask = stats["positive_target_mask"]
        if positive_mask.any():
            return per_image[positive_mask].mean()
        return torch.full((), float("nan"), dtype=per_image.dtype, device=per_image.device)

    tn_sum = tn.sum()
    fp_sum = fp.sum()
    denom_sum = tn_sum + fp_sum
    if denom_sum <= 0:
        return torch.ones((), dtype=torch.float32, device=pred.device)
    return tn_sum / denom_sum


def positive_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Mean Dice over target-positive images only."""
    return dice_score(pred, target, threshold=threshold, reduction="positive_mean")


def case_level_counts(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, int]:
    """Return case-level TP/FN/FP/TN counts from binary segmentation tensors."""
    stats = compute_binary_segmentation_stats(pred, target, threshold=threshold)
    pred_positive = stats["pred_sum"] > 0
    target_positive = stats["target_sum"] > 0
    return {
        "tp": int((pred_positive & target_positive).sum().item()),
        "fn": int((~pred_positive & target_positive).sum().item()),
        "fp": int((pred_positive & ~target_positive).sum().item()),
        "tn": int((~pred_positive & ~target_positive).sum().item()),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def case_level_f1_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    counts = case_level_counts(pred, target, threshold=threshold)
    precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
    recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
    return _safe_div(2.0 * precision * recall, precision + recall)


def negative_false_positive_rate(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    counts = case_level_counts(pred, target, threshold=threshold)
    return _safe_div(counts["fp"], counts["fp"] + counts["tn"])


def compute_per_case_binary_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Compute per-case pixel and case-level metrics from probabilities."""
    stats = compute_binary_segmentation_stats(pred, target, threshold=threshold)
    dice = _per_image_overlap_metric(stats, "dice")
    iou = _per_image_overlap_metric(stats, "iou")
    precision = _per_image_overlap_metric(stats, "precision")
    recall = _per_image_overlap_metric(stats, "recall")

    tn = stats["tn"]
    fp = stats["fp"]
    specificity = torch.ones_like(tn, dtype=torch.float32)
    spec_den = tn + fp
    spec_valid = spec_den > 0
    specificity[spec_valid] = tn[spec_valid] / spec_den[spec_valid]

    rows: list[dict[str, Any]] = []
    for idx in range(pred.shape[0]):
        rows.append(
            {
                "gt_is_positive": bool(stats["target_sum"][idx].item() > 0),
                "pred_is_positive": bool(stats["pred_sum"][idx].item() > 0),
                "gt_foreground_pixels": int(stats["target_sum"][idx].item()),
                "pred_foreground_pixels": int(stats["pred_sum"][idx].item()),
                "intersection_pixels": int(stats["tp"][idx].item()),
                "false_positive_pixels": int(stats["fp"][idx].item()),
                "false_negative_pixels": int(stats["fn"][idx].item()),
                "true_negative_pixels": int(stats["tn"][idx].item()),
                "total_pixels": int(
                    stats["tp"][idx].item()
                    + stats["fp"][idx].item()
                    + stats["fn"][idx].item()
                    + stats["tn"][idx].item()
                ),
                "dice": float(dice[idx].item()),
                "iou": float(iou[idx].item()),
                "precision": float(precision[idx].item()),
                "recall": float(recall[idx].item()),
                "specificity": float(specificity[idx].item()),
            }
        )
    return rows


def aggregate_per_case_binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case segmentation rows into the PR-10C reporting schema."""
    if not rows:
        return {
            "total_cases": 0,
            "gt_positive_cases": 0,
            "gt_negative_cases": 0,
            "pred_positive_cases": 0,
            "tp": 0,
            "fn": 0,
            "fp": 0,
            "tn": 0,
            "case_level_precision": 0.0,
            "case_level_recall": 0.0,
            "case_level_specificity": 0.0,
            "case_level_f1": 0.0,
            "negative_case_false_positive_rate": 0.0,
            "mean_dice_all_cases": float("nan"),
            "mean_dice_positive_cases": float("nan"),
            "positive_dice": float("nan"),
            "mean_iou_all_cases": float("nan"),
            "mean_precision_all_cases": float("nan"),
            "mean_recall_all_cases": float("nan"),
            "mean_specificity_all_cases": float("nan"),
        }

    gt_pos_rows = [row for row in rows if row["gt_is_positive"]]
    gt_neg_rows = [row for row in rows if not row["gt_is_positive"]]
    pred_pos_rows = [row for row in rows if row["pred_is_positive"]]

    tp = sum(1 for row in rows if row["gt_is_positive"] and row["pred_is_positive"])
    fn = sum(1 for row in rows if row["gt_is_positive"] and not row["pred_is_positive"])
    fp = sum(1 for row in rows if (not row["gt_is_positive"]) and row["pred_is_positive"])
    tn = sum(1 for row in rows if (not row["gt_is_positive"]) and (not row["pred_is_positive"]))

    def _mean(values: list[float]) -> float:
        return float(statistics.fmean(values)) if values else float("nan")

    case_precision = _safe_div(tp, tp + fp)
    case_recall = _safe_div(tp, tp + fn)
    case_specificity = _safe_div(tn, tn + fp)
    case_f1 = _safe_div(2.0 * case_precision * case_recall, case_precision + case_recall)
    positive_dice = _mean([float(row["dice"]) for row in gt_pos_rows])

    return {
        "total_cases": len(rows),
        "gt_positive_cases": len(gt_pos_rows),
        "gt_negative_cases": len(gt_neg_rows),
        "pred_positive_cases": len(pred_pos_rows),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "case_level_precision": case_precision,
        "case_level_recall": case_recall,
        "case_level_specificity": case_specificity,
        "case_level_f1": case_f1,
        "negative_case_false_positive_rate": _safe_div(fp, fp + tn),
        "mean_dice_all_cases": _mean([float(row["dice"]) for row in rows]),
        "mean_dice_positive_cases": positive_dice,
        "positive_dice": positive_dice,
        "mean_iou_all_cases": _mean([float(row["iou"]) for row in rows]),
        "mean_iou_positive_cases": _mean([float(row["iou"]) for row in gt_pos_rows]),
        "mean_precision_all_cases": _mean([float(row["precision"]) for row in rows]),
        "mean_precision_positive_cases": _mean([float(row["precision"]) for row in gt_pos_rows]),
        "mean_recall_all_cases": _mean([float(row["recall"]) for row in rows]),
        "mean_recall_positive_cases": _mean([float(row["recall"]) for row in gt_pos_rows]),
        "mean_specificity_all_cases": _mean([float(row["specificity"]) for row in rows]),
    }


def hausdorff_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    95th-percentile Hausdorff distance between predicted and ground-truth boundaries.

    Uses scipy distance_transform_edt. Returns float('nan') if either mask is empty.
    Averages over the batch dimension.
    """
    pred_np = (pred > threshold).squeeze(1).cpu().numpy().astype(bool)
    target_np = (target > 0.5).squeeze(1).cpu().numpy().astype(bool)

    distances = []
    for p, t in zip(pred_np, target_np):
        if not p.any() or not t.any():
            distances.append(float("nan"))
            continue

        dist_p_to_t = distance_transform_edt(~p)
        dist_t_to_p = distance_transform_edt(~t)

        hd_p_to_t = dist_p_to_t[t].max()
        hd_t_to_p = dist_t_to_p[p].max()
        distances.append(float(max(hd_p_to_t, hd_t_to_p)))

    valid = [d for d in distances if not math.isnan(d)]
    return float(np.mean(valid)) if valid else float("nan")
