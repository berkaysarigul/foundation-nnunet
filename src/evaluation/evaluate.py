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

            records.append({
                "image_id":  test_ds.image_ids[idx],
                "positive":  is_positive,
                "dice":      dice_score(pred, mask).item(),
                "iou":       iou_score(pred, mask).item(),
                "hausdorff": hausdorff_distance(pred, mask),
                "precision": precision_score(pred, mask).item(),
                "recall":    recall_score(pred, mask).item(),
                "f1":        f1_score(pred, mask).item(),
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
