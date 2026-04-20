"""
trainer.py — Main training loop for baseline UNet and HybridFoundationUNet.

Usage:
    python -m src.training.trainer --config configs/config.yaml
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.augmentations import get_train_transforms
from src.data.dataset import PneumothoraxDataset
from src.models.unet import UNet
from src.training.losses import DiceFocalLoss
from src.training.metrics import dice_score, iou_score
from src.training.run_artifacts import (
    build_best_checkpoint_metadata,
    build_run_metadata,
    canonicalize_history,
    prepare_run_artifacts,
    write_config_snapshot,
    write_history_csv,
    write_yaml,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


LOSS_ALIASES = {
    "dicefocal": "dice_focal",
    "dicefocalloss": "dice_focal",
}

OPTIMIZER_ALIASES = {
    "adamw": "AdamW",
    "adam": "Adam",
}

SCHEDULER_ALIASES = {
    "reducelronplateau": "ReduceLROnPlateau",
    "plateau": "ReduceLROnPlateau",
    "none": "none",
    "disabled": "none",
    "off": "none",
}


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


def resolve_training_component_config(cfg: dict[str, Any]) -> dict[str, str]:
    return {
        "loss": resolve_component_choice(
            cfg["loss"]["type"],
            LOSS_ALIASES,
            "loss.type",
        ),
        "optimizer": resolve_component_choice(
            cfg["training"]["optimizer"],
            OPTIMIZER_ALIASES,
            "training.optimizer",
        ),
        "scheduler": resolve_component_choice(
            cfg["training"].get("scheduler", "none"),
            SCHEDULER_ALIASES,
            "training.scheduler",
        ),
    }


def build_loss(training_components: dict[str, str]) -> torch.nn.Module:
    loss_name = training_components["loss"]
    if loss_name == "dice_focal":
        return DiceFocalLoss()
    raise ValueError(f"Unsupported resolved loss: {loss_name!r}")


def build_optimizer(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    training_components: dict[str, str],
) -> torch.optim.Optimizer:
    optimizer_name = training_components["optimizer"]
    parameters = [param for param in model.parameters() if param.requires_grad]
    learning_rate = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]

    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "Adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported resolved optimizer: {optimizer_name!r}")


def apply_foundation_x_backbone_train_mode_policy(model: torch.nn.Module) -> None:
    """Apply the trainer-side Foundation X backbone mode policy.

    Frozen backbones stay in eval mode during training. Unfrozen backbones must
    not be silently forced back to eval mode here.
    """
    foundation_x = getattr(model, "foundation_x", None)
    if foundation_x is None or not hasattr(foundation_x, "backbone"):
        return

    frozen_backbone = getattr(
        model,
        "frozen_backbone",
        getattr(foundation_x, "frozen", False),
    )
    if frozen_backbone:
        foundation_x.backbone.eval()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_components: dict[str, str],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = training_components["scheduler"]
    if scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unsupported resolved scheduler: {scheduler_name!r}")


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    monitor_value: float,
) -> None:
    if scheduler is None:
        return
    scheduler.step(monitor_value)


def validate_resume_training_components(
    resume_state: dict[str, Any],
    expected_training_components: dict[str, str],
) -> None:
    resume_training_components = resume_state.get("training_components")
    if resume_training_components is None:
        raise ValueError(
            "Legacy resume checkpoint is missing training_components metadata. "
            "Delete checkpoints/last_*.pth or start a fresh authoritative run."
        )

    normalized_resume_training_components = {
        "loss": resolve_component_choice(
            resume_training_components["loss"],
            LOSS_ALIASES,
            "resume.training_components.loss",
        ),
        "optimizer": resolve_component_choice(
            resume_training_components["optimizer"],
            OPTIMIZER_ALIASES,
            "resume.training_components.optimizer",
        ),
        "scheduler": resolve_component_choice(
            resume_training_components["scheduler"],
            SCHEDULER_ALIASES,
            "resume.training_components.scheduler",
        ),
    }
    if normalized_resume_training_components != expected_training_components:
        raise ValueError(
            "Resume checkpoint training_components do not match the current config. "
            f"checkpoint={normalized_resume_training_components}, "
            f"config={expected_training_components}"
        )


def compute_validation_overlap_totals(
    preds: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return summed per-image overlap scores for trainer-side validation aggregation."""
    per_image_dice = dice_score(preds, masks, threshold=threshold, reduction="none")
    per_image_iou = iou_score(preds, masks, threshold=threshold, reduction="none")
    return {
        "dice_sum": float(per_image_dice.sum().item()),
        "iou_sum": float(per_image_iou.sum().item()),
        "image_count": int(per_image_dice.numel()),
    }


def compute_positive_validation_dice_totals(
    preds: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return summed per-image Dice over positive-target images only."""
    per_image_dice = dice_score(preds, masks, threshold=threshold, reduction="none")
    positive_mask = (masks > 0.5).reshape(masks.shape[0], -1).any(dim=1)

    if not positive_mask.any():
        return {"dice_sum": 0.0, "positive_image_count": 0}

    positive_dice = per_image_dice[positive_mask]
    return {
        "dice_sum": float(positive_dice.sum().item()),
        "positive_image_count": int(positive_mask.sum().item()),
    }


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

    if model_type == "pretrained_resnet34_unet":
        from src.models.resnet34_unet import PretrainedResNet34UNet

        return PretrainedResNet34UNet(
            in_channels=in_ch,
            num_classes=num_cls,
            base_filters=base_f,
        )

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

    raise ValueError(
        "Unknown model type: "
        f"'{model_type}'. Use 'baseline', 'pretrained_resnet34_unet', or 'hybrid'."
    )


def train(
    cfg: dict,
    *,
    config_path: str | Path = "configs/config.yaml",
    run_dir: str | Path | None = None,
) -> float:
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seeds(cfg["seed"])
    device = resolve_device(cfg["device"])
    logger.info("Device: %s", device)
    repo_root = Path(__file__).resolve().parents[2]

    model_type   = cfg["model"]["type"]
    data_dir     = cfg["data"]["processed_dir"]
    input_size   = cfg["data"]["input_size"]
    num_workers  = cfg["data"]["num_workers"]
    splits_path = cfg["data"].get("splits_path")
    train_mask_variant = cfg["data"].get("train_mask_variant", "dilated_masks")
    eval_mask_variant = cfg["data"].get("eval_mask_variant", "original_masks")
    train_crop = cfg["data"].get("train_crop")
    batch_size   = cfg["training"]["batch_size"]
    epochs       = cfg["training"]["epochs"]
    patience     = cfg["training"]["early_stopping_patience"]
    run_artifacts = prepare_run_artifacts(
        model_type,
        run_dir=run_dir,
        run_root=repo_root / "artifacts" / "runs",
    )
    logger.info("Authoritative run directory: %s", run_artifacts.run_dir)

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    train_ds = PneumothoraxDataset(
        data_dir, split="train", img_size=input_size,
        transform=get_train_transforms(),
        mask_variant=train_mask_variant,
        train_crop=train_crop,
        splits_path=splits_path,
    )
    val_ds = PneumothoraxDataset(
        data_dir, split="val", img_size=input_size, transform=None,
        mask_variant=eval_mask_variant,
        splits_path=splits_path,
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
        "Train: %d samples | Val: %d samples | train_mask_variant=%s | eval_mask_variant=%s | train_crop=%s",
        len(train_ds),
        len(val_ds),
        train_ds.mask_variant,
        val_ds.mask_variant,
        train_ds.train_crop["mode"] if train_ds.train_crop is not None else "none",
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
    training_components = resolve_training_component_config(cfg)
    logger.info(
        "Training components | loss=%s | optimizer=%s | scheduler=%s",
        training_components["loss"],
        training_components["optimizer"],
        training_components["scheduler"],
    )
    criterion = build_loss(training_components)
    optimizer = build_optimizer(cfg, model, training_components)
    scheduler = build_scheduler(optimizer, training_components)

    # ------------------------------------------------------------------
    # Resume from checkpoint if available
    # ------------------------------------------------------------------
    best_dice = float("-inf")
    patience_counter = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice_mean": [],
        "val_dice_pos_mean": [],
        "val_iou_mean": [],
    }
    start_epoch = 1

    resume_path = run_artifacts.last_checkpoint_path
    run_metadata = build_run_metadata(
        cfg=cfg,
        config_path=config_path,
        repo_root=repo_root,
        run_id=run_artifacts.run_id,
        resume_checkpoint_path=resume_path if resume_path.exists() else None,
    )
    write_yaml(run_artifacts.run_metadata_path, run_metadata)
    write_config_snapshot(run_artifacts.config_snapshot_path, cfg)

    if resume_path.exists():
        logger.info("Resuming from %s", resume_path)
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume_training_components(resume, training_components)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None:
            scheduler_state = resume.get("scheduler")
            if scheduler_state is None:
                raise ValueError(
                    "Resume checkpoint is missing scheduler state for the configured scheduler."
                )
            scheduler.load_state_dict(scheduler_state)
        start_epoch      = resume["epoch"] + 1
        best_dice        = resume["best_dice"]
        patience_counter = resume["patience_counter"]
        history          = canonicalize_history(resume["history"])
        logger.info(
            "Resumed at epoch %d | Best Dice so far: %.4f", start_epoch - 1, best_dice
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):
        # === TRAIN ===
        model.train()
        apply_foundation_x_backbone_train_mode_policy(model)

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
        val_metric_image_count = 0
        val_dice_pos_sum = 0.0
        val_dice_pos_count = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_loss     += criterion(preds, masks).item()
                overlap_totals = compute_validation_overlap_totals(preds, masks)
                val_dice_sum += overlap_totals["dice_sum"]
                val_iou_sum  += overlap_totals["iou_sum"]
                val_metric_image_count += overlap_totals["image_count"]

                positive_dice_totals = compute_positive_validation_dice_totals(preds, masks)
                val_dice_pos_sum += positive_dice_totals["dice_sum"]
                val_dice_pos_count += positive_dice_totals["positive_image_count"]

        val_loss         /= len(val_loader)
        val_dice_mean     = val_dice_sum / max(val_metric_image_count, 1)
        val_iou_mean      = val_iou_sum  / max(val_metric_image_count, 1)
        val_dice_pos_mean = val_dice_pos_sum / max(val_dice_pos_count, 1)

        # ReduceLROnPlateau monitors positive-only Dice (maximize)
        step_scheduler(scheduler, val_dice_pos_mean)

        # === LOG ===
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice_mean"].append(val_dice_mean)
        history["val_dice_pos_mean"].append(val_dice_pos_mean)
        history["val_iou_mean"].append(val_iou_mean)

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Dice: %.4f | Val Dice (pos): %.4f | Val IoU: %.4f",
            epoch, epochs, train_loss, val_loss, val_dice_mean, val_dice_pos_mean, val_iou_mean,
        )

        # === CHECKPOINT — based on positive-only Dice ===
        if val_dice_pos_mean > best_dice:
            best_dice = val_dice_pos_mean
            patience_counter = 0
            ckpt_path = run_artifacts.best_checkpoint_path
            torch.save(model.state_dict(), ckpt_path)
            best_checkpoint_metadata = build_best_checkpoint_metadata(
                checkpoint_path=ckpt_path,
                cfg=cfg,
                repo_root=repo_root,
                epoch=epoch,
                best_metric_value=best_dice,
                training_components=training_components,
            )
            write_yaml(run_artifacts.best_checkpoint_metadata_path, best_checkpoint_metadata)
            logger.info("  → Best model saved (Dice pos: %.4f) → %s", best_dice, ckpt_path)
        else:
            patience_counter += 1

        # === RESUME STATE (overwritten every epoch) ===
        torch.save(
            {
                "epoch":            epoch,
                "model":            model.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "scheduler":        scheduler.state_dict() if scheduler is not None else None,
                "training_components": training_components,
                "best_dice":        best_dice,
                "patience_counter": patience_counter,
                "history":          history,
            },
            run_artifacts.last_checkpoint_path,
        )

        # === EARLY STOPPING ===
        if patience_counter >= patience:
            logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, patience)
            break

    # ------------------------------------------------------------------
    # Save history
    # ------------------------------------------------------------------
    write_history_csv(run_artifacts.history_path, history)
    logger.info("Training history saved to %s", run_artifacts.history_path)
    logger.info("Best Val Dice: %.4f", best_dice)

    return best_dice


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Foundation-nnU-Net")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Optional authoritative run directory. If omitted, trainer creates a new directory under artifacts/runs/.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, config_path=args.config, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
