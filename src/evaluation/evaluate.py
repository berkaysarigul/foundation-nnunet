"""
evaluate.py — Test set evaluation for baseline UNet and HybridFoundationUNet.

Computes per-image Dice, IoU, Hausdorff, Precision, Recall, F1.
Reports overall and stratified by pneumothorax-positive / negative.

Usage:
    python -m src.evaluation.evaluate \
        --config configs/config.yaml \
        --checkpoint checkpoints/best_baseline.pth \
        --model_type baseline
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PneumothoraxDataset
from src.training.metrics import (
    dice_score,
    f1_score,
    hausdorff_distance,
    iou_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


SELECTION_METRIC_ALIASES = {
    "valdiceposmean": "val_dice_pos_mean",
}

POSTPROCESS_ALIASES = {
    "none": "none",
    "off": "none",
    "disabled": "none",
}

DEFAULT_THRESHOLD_CANDIDATES = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

DEFAULT_SELECTION_THRESHOLD = 0.5
SELECTION_SCORE_TOLERANCE = 1e-12


def normalize_component_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def resolve_component_choice(
    raw_value: str,
    aliases: dict[str, str],
    field_name: str,
) -> str:
    normalized = normalize_component_name(raw_value)
    canonical = aliases.get(normalized)
    if canonical is None:
        supported_values = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unsupported {field_name}: {raw_value!r}. Supported values: {supported_values}."
        )
    return canonical


def resolve_threshold_selection_config(cfg: dict[str, Any]) -> dict[str, Any]:
    selection_cfg = cfg.get("selection", {})
    metric_name = resolve_component_choice(
        selection_cfg.get("metric", "val_dice_pos_mean"),
        SELECTION_METRIC_ALIASES,
        "selection.metric",
    )
    postprocess = resolve_component_choice(
        selection_cfg.get("postprocess", "none"),
        POSTPROCESS_ALIASES,
        "selection.postprocess",
    )

    raw_thresholds = selection_cfg.get("threshold_candidates", DEFAULT_THRESHOLD_CANDIDATES)
    if not raw_thresholds:
        raise ValueError("selection.threshold_candidates must contain at least one candidate.")

    thresholds = sorted({float(value) for value in raw_thresholds})
    if any(threshold <= 0.0 or threshold >= 1.0 for threshold in thresholds):
        raise ValueError(
            "selection.threshold_candidates must stay strictly inside (0, 1)."
        )
    if DEFAULT_SELECTION_THRESHOLD not in thresholds:
        raise ValueError(
            "selection.threshold_candidates must include 0.5 so threshold tuning can "
            "compare against the current default inference threshold."
        )

    return {
        "metric": metric_name,
        "postprocess": postprocess,
        "threshold_candidates": thresholds,
    }


def validate_threshold_selection_split(split: str) -> str:
    if split != "val":
        raise ValueError(
            f"Threshold selection must use validation data only; received split={split!r}."
        )
    return split


def summarize_threshold_candidates(
    preds: torch.Tensor,
    masks: torch.Tensor,
    thresholds: list[float],
) -> list[dict[str, float]]:
    summary = []
    positive_image_count = int((masks > 0.5).reshape(masks.shape[0], -1).any(dim=1).sum().item())

    for threshold in thresholds:
        summary.append(
            {
                "threshold": float(threshold),
                "val_dice_pos_mean": float(
                    dice_score(preds, masks, threshold=threshold, reduction="positive_mean").item()
                ),
                "val_dice_mean": float(
                    dice_score(preds, masks, threshold=threshold, reduction="mean").item()
                ),
                "val_iou_mean": float(
                    iou_score(preds, masks, threshold=threshold, reduction="mean").item()
                ),
                "positive_image_count": positive_image_count,
            }
        )

    return summary


def select_best_threshold(
    threshold_summary: list[dict[str, float]],
    metric_name: str,
) -> dict[str, float]:
    if not threshold_summary:
        raise ValueError("threshold_summary must contain at least one evaluated threshold.")

    valid_rows = [
        row for row in threshold_summary if not math.isnan(float(row[metric_name]))
    ]
    if not valid_rows:
        raise ValueError(
            f"All threshold candidates produced NaN for selection metric {metric_name!r}."
        )

    best_score = max(float(row[metric_name]) for row in valid_rows)
    tied_rows = [
        row
        for row in valid_rows
        if math.isclose(float(row[metric_name]), best_score, abs_tol=SELECTION_SCORE_TOLERANCE)
    ]
    return min(
        tied_rows,
        key=lambda row: (
            abs(float(row["threshold"]) - DEFAULT_SELECTION_THRESHOLD),
            float(row["threshold"]),
        ),
    )


def tune_threshold_on_validation_predictions(
    preds: torch.Tensor,
    masks: torch.Tensor,
    cfg: dict[str, Any],
    *,
    split: str = "val",
) -> dict[str, Any]:
    validate_threshold_selection_split(split)
    selection_config = resolve_threshold_selection_config(cfg)
    threshold_summary = summarize_threshold_candidates(
        preds,
        masks,
        selection_config["threshold_candidates"],
    )
    best_row = select_best_threshold(
        threshold_summary,
        metric_name=selection_config["metric"],
    )
    return {
        "split": split,
        "selection_metric": selection_config["metric"],
        "selected_threshold": float(best_row["threshold"]),
        "selected_postprocess": selection_config["postprocess"],
        "threshold_summary": threshold_summary,
    }


def build_model(cfg: dict, model_type: str) -> torch.nn.Module:
    in_ch  = cfg["model"]["in_channels"]
    num_cls = cfg["model"]["num_classes"]
    base_f  = cfg["model"]["base_filters"]

    if model_type == "baseline":
        from src.models.unet import UNet
        return UNet(in_channels=in_ch, num_classes=num_cls, base_filters=base_f)

    if model_type == "hybrid":
        from src.models.hybrid import HybridFoundationUNet
        return HybridFoundationUNet(
            backbone_checkpoint=cfg["foundation_x"]["checkpoint_path"],
            in_channels=in_ch,
            num_classes=num_cls,
            base_filters=base_f,
            frozen_backbone=cfg["foundation_x"]["frozen"],
            img_size=cfg["data"]["input_size"],
        )

    raise ValueError(f"Unknown model type: '{model_type}'. Use 'baseline' or 'hybrid'.")


def _stat(values: list[float]) -> str:
    """Return 'mean ± std' string, ignoring NaNs."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return "nan ± nan"
    m = np.mean(clean)
    s = np.std(clean)
    return f"{m:.4f} ± {s:.4f}"


def compute_per_image_metrics(
    pred: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return per-image metrics via the shared overlap-metric backend."""
    return {
        "dice": float(dice_score(pred, mask, threshold=threshold, reduction="none").squeeze(0).item()),
        "iou": float(iou_score(pred, mask, threshold=threshold, reduction="none").squeeze(0).item()),
        "hausdorff": hausdorff_distance(pred, mask, threshold=threshold),
        "precision": float(
            precision_score(pred, mask, threshold=threshold, reduction="none").squeeze(0).item()
        ),
        "recall": float(
            recall_score(pred, mask, threshold=threshold, reduction="none").squeeze(0).item()
        ),
        "f1": float(f1_score(pred, mask, threshold=threshold, reduction="none").squeeze(0).item()),
    }


def print_summary(df: pd.DataFrame, model_type: str) -> None:
    metrics = ["dice", "iou", "hausdorff", "precision", "recall", "f1"]

    pos_df  = df[df["positive"] == True]
    neg_df  = df[df["positive"] == False]

    print(f"\n{'='*60}")
    print(f"  Test Results: {model_type}")
    print(f"{'='*60}")
    print(f"  Samples — All: {len(df)} | Positive: {len(pos_df)} | Negative: {len(neg_df)}")
    print(f"{'='*60}")

    header = f"{'Subset':<12}" + "".join(f"{m:>22}" for m in metrics)
    print(header)
    print("-" * len(header))

    for label, subset in [("All", df), ("Positive", pos_df), ("Negative", neg_df)]:
        if len(subset) == 0:
            continue
        row = f"{label:<12}"
        for m in metrics:
            row += f"{_stat(subset[m].tolist()):>22}"
        print(row)

    print(f"{'='*60}\n")


def evaluate(cfg: dict, checkpoint_path: str, model_type: str) -> pd.DataFrame:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu"
    )
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(cfg, model_type)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    # ------------------------------------------------------------------
    # Test DataLoader (batch_size=1 for per-image metrics)
    # ------------------------------------------------------------------
    test_ds = PneumothoraxDataset(
        cfg["data"]["processed_dir"],
        split="test",
        img_size=cfg["data"]["input_size"],
        transform=None,
        mask_variant=cfg["data"].get("eval_mask_variant", "original_masks"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    logger.info("Test set: %d samples | eval_mask_variant=%s", len(test_ds), test_ds.mask_variant)

    # ------------------------------------------------------------------
    # Per-image evaluation
    # ------------------------------------------------------------------
    records = []

    with torch.no_grad():
        for idx, (image, mask) in enumerate(tqdm(test_loader, desc="Evaluating")):
            image, mask = image.to(device), mask.to(device)
            pred = model(image)

            is_positive = mask.sum().item() > 0
            metrics = compute_per_image_metrics(pred, mask)

            records.append({
                "image_id":  test_ds.image_ids[idx],
                "positive":  is_positive,
                **metrics,
            })

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    Path("results").mkdir(exist_ok=True)
    out_path = f"results/test_metrics_{model_type}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved per-image metrics to %s", out_path)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print_summary(df, model_type)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Foundation-nnU-Net on test set")
    parser.add_argument("--config",     default="configs/config.yaml",          help="Path to config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_baseline.pth", help="Path to model checkpoint")
    parser.add_argument("--model_type", default=None, help="Override config model type: 'baseline' or 'hybrid'")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_type = args.model_type if args.model_type else cfg["model"]["type"]
    evaluate(cfg, args.checkpoint, model_type)


if __name__ == "__main__":
    main()
