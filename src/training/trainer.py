"""
trainer.py — Main training loop for baseline UNet and HybridFoundationUNet.

Usage:
    python -m src.training.trainer --config configs/config.yaml
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.augmentations import get_train_transforms
from src.data.dataset import PneumothoraxDataset
from src.models.unet import UNet
from src.training.losses import DiceFocalLoss
from src.training.metrics import dice_score, iou_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_sampler(dataset: PneumothoraxDataset) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler giving positive samples higher draw probability.

    Scans mask filenames already stored in dataset.image_ids — no extra I/O beyond
    what's needed to check pixel sums. Positive and negative class weights are
    balanced to produce a ~50/50 mix per epoch regardless of dataset imbalance.
    """
    masks_dir = dataset.mask_dir
    import cv2

    labels = []
    for image_id in dataset.image_ids:
        mask = cv2.imread(str(masks_dir / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        labels.append(1 if (mask is not None and mask.max() > 0) else 0)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    # weight per class = 1 / class_count  → equal expected frequency
    weight_pos = 1.0 / n_pos if n_pos > 0 else 1.0
    weight_neg = 1.0 / n_neg if n_neg > 0 else 1.0
    sample_weights = [weight_pos if l == 1 else weight_neg for l in labels]

    logger.info(
        "Sampler — positive: %d (%.1f%%), negative: %d (%.1f%%)",
        n_pos, 100 * n_pos / len(labels),
        n_neg, 100 * n_neg / len(labels),
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_model(cfg: dict) -> torch.nn.Module:
    model_type = cfg["model"]["type"]
    in_ch = cfg["model"]["in_channels"]
    num_cls = cfg["model"]["num_classes"]
    base_f = cfg["model"]["base_filters"]

    if model_type == "baseline":
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


def train(cfg: dict) -> float:
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seeds(cfg["seed"])
    device = resolve_device(cfg["device"])
    logger.info("Device: %s", device)

    model_type   = cfg["model"]["type"]
    data_dir     = cfg["data"]["processed_dir"]
    input_size   = cfg["data"]["input_size"]
    num_workers  = cfg["data"]["num_workers"]
    train_mask_variant = cfg["data"].get("train_mask_variant", "dilated_masks")
    eval_mask_variant = cfg["data"].get("eval_mask_variant", "original_masks")
    batch_size   = cfg["training"]["batch_size"]
    epochs       = cfg["training"]["epochs"]
    lr           = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    patience     = cfg["training"]["early_stopping_patience"]

    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    train_ds = PneumothoraxDataset(
        data_dir, split="train", img_size=input_size,
        transform=get_train_transforms(),
        mask_variant=train_mask_variant,
    )
    val_ds = PneumothoraxDataset(
        data_dir, split="val", img_size=input_size, transform=None,
        mask_variant=eval_mask_variant,
    )

    sampler = build_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    logger.info(
        "Train: %d samples | Val: %d samples | train_mask_variant=%s | eval_mask_variant=%s",
        len(train_ds),
        len(val_ds),
        train_ds.mask_variant,
        val_ds.mask_variant,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s | Trainable: %s / %s params", model_type, f"{trainable:,}", f"{total:,}")

    # ------------------------------------------------------------------
    # Loss, Optimizer, Scheduler
    # ------------------------------------------------------------------
    criterion = DiceFocalLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint if available
    # ------------------------------------------------------------------
    best_dice = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_dice_pos": [], "val_iou": []}
    start_epoch = 1

    resume_path = Path(f"checkpoints/last_{model_type}.pth")
    if resume_path.exists():
        logger.info("Resuming from %s", resume_path)
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch      = resume["epoch"] + 1
        best_dice        = resume["best_dice"]
        patience_counter = resume["patience_counter"]
        history          = resume["history"]
        logger.info(
            "Resumed at epoch %d | Best Dice so far: %.4f", start_epoch - 1, best_dice
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):
        # === TRAIN ===
        model.train()
        # Keep Foundation X backbone in eval() at all times
        if hasattr(model, "foundation_x"):
            model.foundation_x.backbone.eval()

        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss  = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # === VALIDATION ===
        model.eval()
        val_loss = val_dice_sum = val_iou_sum = 0.0
        val_dice_pos_sum = 0.0
        val_dice_pos_count = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_loss     += criterion(preds, masks).item()
                val_dice_sum += dice_score(preds, masks).item()
                val_iou_sum  += iou_score(preds, masks).item()

                # Per-batch positive-only Dice: select images that have GT foreground
                is_pos = masks.sum(dim=(1, 2, 3)) > 0
                if is_pos.any():
                    val_dice_pos_sum   += dice_score(preds[is_pos], masks[is_pos]).item()
                    val_dice_pos_count += 1

        val_loss         /= len(val_loader)
        val_dice_mean     = val_dice_sum / len(val_loader)
        val_iou_mean      = val_iou_sum  / len(val_loader)
        val_dice_pos_mean = val_dice_pos_sum / max(val_dice_pos_count, 1)

        # ReduceLROnPlateau monitors positive-only Dice (maximize)
        scheduler.step(val_dice_pos_mean)

        # === LOG ===
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice_mean)
        history["val_dice_pos"].append(val_dice_pos_mean)
        history["val_iou"].append(val_iou_mean)

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Dice: %.4f | Val Dice (pos): %.4f | Val IoU: %.4f",
            epoch, epochs, train_loss, val_loss, val_dice_mean, val_dice_pos_mean, val_iou_mean,
        )

        # === CHECKPOINT — based on positive-only Dice ===
        if val_dice_pos_mean > best_dice:
            best_dice = val_dice_pos_mean
            patience_counter = 0
            ckpt_path = f"checkpoints/best_{model_type}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  → Best model saved (Dice pos: %.4f) → %s", best_dice, ckpt_path)
        else:
            patience_counter += 1

        # === RESUME STATE (overwritten every epoch) ===
        torch.save(
            {
                "epoch":            epoch,
                "model":            model.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "scheduler":        scheduler.state_dict(),
                "best_dice":        best_dice,
                "patience_counter": patience_counter,
                "history":          history,
            },
            f"checkpoints/last_{model_type}.pth",
        )

        # === EARLY STOPPING ===
        if patience_counter >= patience:
            logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, patience)
            break

    # ------------------------------------------------------------------
    # Save history
    # ------------------------------------------------------------------
    history_path = f"results/{model_type}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    logger.info("Training history saved to %s", history_path)
    logger.info("Best Val Dice: %.4f", best_dice)

    return best_dice


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Foundation-nnU-Net")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
