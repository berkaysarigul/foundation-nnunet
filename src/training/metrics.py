"""
metrics.py — Segmentation evaluation metrics.

All functions accept (batch, 1, H, W) tensors and return Python floats.
hausdorff_distance returns float('nan') when either mask is empty.
"""

import math

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Dice similarity coefficient."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    return (2.0 * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Intersection over Union (Jaccard index)."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


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
    pred_np = (pred > threshold).squeeze(1).cpu().numpy().astype(bool)  # (B, H, W)
    target_np = target.squeeze(1).cpu().numpy().astype(bool)             # (B, H, W)

    distances = []
    for p, t in zip(pred_np, target_np):
        if not p.any() or not t.any():
            distances.append(float("nan"))
            continue

        # Distance from each target pixel to nearest pred pixel
        dist_p_to_t = distance_transform_edt(~p)
        dist_t_to_p = distance_transform_edt(~t)

        hd_p_to_t = dist_p_to_t[t].max()
        hd_t_to_p = dist_t_to_p[p].max()
        distances.append(float(max(hd_p_to_t, hd_t_to_p)))

    valid = [d for d in distances if not math.isnan(d)]
    return float(np.mean(valid)) if valid else float("nan")


def precision_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Precision: TP / (TP + FP)."""
    pred_binary = (pred > threshold).float()
    tp = (pred_binary * target).sum()
    return (tp + smooth) / (pred_binary.sum() + smooth)


def recall_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Recall (sensitivity): TP / (TP + FN)."""
    pred_binary = (pred > threshold).float()
    tp = (pred_binary * target).sum()
    return (tp + smooth) / (target.sum() + smooth)


def f1_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """F1 score (harmonic mean of precision and recall)."""
    p = precision_score(pred, target, threshold)
    r = recall_score(pred, target, threshold)
    return 2.0 * p * r / (p + r + 1e-6)
