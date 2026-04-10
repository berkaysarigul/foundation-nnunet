"""
visualize.py — Training curves, prediction visuals, and model comparison plots.

Usage:
    python -m src.evaluation.visualize --results_dir results/

    # With prediction visuals:
    python -m src.evaluation.visualize --results_dir results/ \
        --config configs/config.yaml \
        --checkpoint checkpoints/best_baseline.pth \
        --model_type baseline
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.training.metrics import dice_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    baseline_csv: str | None,
    hybrid_csv: str | None,
    save_dir: str = "results/",
) -> None:
    """Plot loss and Dice curves for baseline and/or hybrid model.

    Missing CSV files are skipped with a warning.
    Saved as: {save_dir}/training_curves.png
    """
    curves: dict[str, pd.DataFrame] = {}
    for name, path in [("Baseline", baseline_csv), ("Hybrid", hybrid_csv)]:
        if path and Path(path).exists():
            curves[name] = pd.read_csv(path)
        elif path:
            logger.warning("Training CSV not found, skipping %s: %s", name, path)

    if not curves:
        logger.warning("No training history CSVs found — skipping training curves.")
        return

    colors = {"Baseline": "#2196F3", "Hybrid": "#FF5722"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    for name, df in curves.items():
        epochs = range(1, len(df) + 1)
        c = colors[name]
        ax1.plot(epochs, df["train_loss"], color=c, linestyle="-",  label=f"{name} Train")
        ax1.plot(epochs, df["val_loss"],   color=c, linestyle="--", label=f"{name} Val")
        ax2.plot(epochs, df["val_dice"],   color=c, linestyle="-",  label=name)

    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title("Validation Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curves → %s", out)


# ---------------------------------------------------------------------------
# 2. Prediction visuals
# ---------------------------------------------------------------------------

def plot_predictions(
    model: torch.nn.Module,
    dataset,
    checkpoint: str,
    config: dict,
    num_samples: int = 8,
    save_dir: str = "results/",
) -> None:
    """Show best 4 + worst 4 predictions (3 columns: image | GT mask | pred mask).

    Saved as: {save_dir}/predictions_{model_type}.png
    """
    model_type = config["model"]["type"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["device"] != "cpu" else "cpu"
    )

    # Load checkpoint
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()

    # Collect per-image Dice on entire dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dices: list[float] = []

    with torch.no_grad():
        for image, mask in loader:
            image, mask = image.to(device), mask.to(device)
            pred = model(image)
            dices.append(dice_score(pred, mask).item())

    # Best 4 + worst 4 by Dice
    n_each = num_samples // 2
    sorted_idx = np.argsort(dices)
    worst_idx  = sorted_idx[:n_each].tolist()
    best_idx   = sorted_idx[-n_each:][::-1].tolist()
    selected   = worst_idx + best_idx
    labels     = [f"Worst #{i+1}" for i in range(n_each)] + \
                 [f"Best #{i+1}"  for i in range(n_each)]

    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3 * len(selected)))
    fig.suptitle(f"Predictions — {model_type.capitalize()}", fontsize=14, fontweight="bold")

    col_titles = ["Image", "Ground Truth", "Prediction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    with torch.no_grad():
        for row, (idx, label) in enumerate(zip(selected, labels)):
            image, mask = dataset[idx]
            image_t = image.unsqueeze(0).to(device)
            pred_t  = model(image_t)

            img_np  = (image.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_np   = mask.squeeze().numpy()
            pred_np = (pred_t.squeeze().cpu().numpy() > 0.5).astype(np.float32)
            d       = dices[idx]

            axes[row, 0].imshow(img_np,   cmap="gray", vmin=0, vmax=255)
            axes[row, 1].imshow(gt_np,    cmap="gray", vmin=0, vmax=1)
            axes[row, 2].imshow(pred_np,  cmap="gray", vmin=0, vmax=1)

            for col in range(3):
                axes[row, col].axis("off")

            axes[row, 0].set_ylabel(
                f"{label}\nDice: {d:.4f}", fontsize=8, rotation=0,
                labelpad=70, va="center",
            )

    plt.tight_layout()
    out = Path(save_dir) / f"predictions_{model_type}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved prediction visuals → %s", out)


# ---------------------------------------------------------------------------
# 3. Model comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    baseline_csv: str | None,
    hybrid_csv: str | None,
    save_dir: str = "results/",
) -> None:
    """Box plots of Dice and IoU for baseline vs hybrid, split by positive/negative.

    Saved as: {save_dir}/model_comparison.png
    """
    dfs: dict[str, pd.DataFrame] = {}
    for name, path in [("Baseline", baseline_csv), ("Hybrid", hybrid_csv)]:
        if path and Path(path).exists():
            dfs[name] = pd.read_csv(path)
        elif path:
            logger.warning("Metrics CSV not found, skipping %s: %s", name, path)

    if not dfs:
        logger.warning("No metrics CSVs found — skipping model comparison.")
        return

    groups = [
        ("All",      lambda df: df),
        ("Positive", lambda df: df[df["positive"] == True]),
        ("Negative", lambda df: df[df["positive"] == False]),
    ]
    colors = {"Baseline": "#2196F3", "Hybrid": "#FF5722"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Baseline vs Hybrid — Test Set Metrics", fontsize=14, fontweight="bold")

    for ax, metric in [(ax1, "dice"), (ax2, "iou")]:
        group_labels = []
        positions    = []
        all_data     = []
        all_colors   = []

        pos = 1
        for g_name, g_fn in groups:
            for m_name, df in dfs.items():
                vals = g_fn(df)[metric].dropna().tolist()
                group_labels.append(f"{g_name}\n{m_name}")
                positions.append(pos)
                all_data.append(vals)
                all_colors.append(colors[m_name])
                pos += 1
            pos += 0.5  # gap between groups

        bp = ax.boxplot(
            all_data,
            positions=positions,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, color in zip(bp["boxes"], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels, fontsize=8)
        ax.set_title(metric.capitalize())
        ax.set_ylabel(metric.capitalize() + " Score")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=colors[n], alpha=0.7, label=n) for n in dfs]
        ax.legend(handles=handles, loc="lower right")

    plt.tight_layout()
    out = Path(save_dir) / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved model comparison → %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Foundation-nnU-Net visualizations")
    parser.add_argument("--results_dir", default="results/",           help="Directory with CSV results")
    parser.add_argument("--config",      default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint",  default=None,                  help="Checkpoint for prediction visuals")
    parser.add_argument("--model_type",  default=None,                  help="'baseline' or 'hybrid'")
    args = parser.parse_args()

    r = Path(args.results_dir)
    r.mkdir(exist_ok=True)

    baseline_history = str(r / "baseline_history.csv")
    hybrid_history   = str(r / "hybrid_history.csv")
    baseline_metrics = str(r / "test_metrics_baseline.csv")
    hybrid_metrics   = str(r / "test_metrics_hybrid.csv")

    # 1. Training curves
    plot_training_curves(baseline_history, hybrid_history, save_dir=str(r))

    # 2. Prediction visuals (only if checkpoint provided)
    if args.checkpoint:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        model_type = args.model_type or cfg["model"]["type"]
        cfg["model"]["type"] = model_type

        # Build model
        if model_type == "baseline":
            from src.models.unet import UNet
            model = UNet(
                in_channels=cfg["model"]["in_channels"],
                num_classes=cfg["model"]["num_classes"],
                base_filters=cfg["model"]["base_filters"],
            )
        else:
            from src.models.hybrid import HybridFoundationUNet
            model = HybridFoundationUNet(
                backbone_checkpoint=cfg["foundation_x"]["checkpoint_path"],
                in_channels=cfg["model"]["in_channels"],
                num_classes=cfg["model"]["num_classes"],
                base_filters=cfg["model"]["base_filters"],
                frozen_backbone=cfg["foundation_x"]["frozen"],
                img_size=cfg["data"]["input_size"],
            )

        from src.data.dataset import PneumothoraxDataset
        dataset = PneumothoraxDataset(
            cfg["data"]["processed_dir"],
            split="test",
            img_size=cfg["data"]["input_size"],
            transform=None,
        )
        plot_predictions(model, dataset, args.checkpoint, cfg, num_samples=8, save_dir=str(r))

    # 3. Model comparison
    plot_comparison(baseline_metrics, hybrid_metrics, save_dir=str(r))


if __name__ == "__main__":
    main()
