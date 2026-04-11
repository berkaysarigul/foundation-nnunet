"""
evaluate.py - Validation threshold selection and test-set evaluation.

Computes per-image Dice, IoU, Hausdorff, Precision, Recall, F1.
Can either:
- run validation-only threshold selection and save `selection_state.yaml`
- evaluate the test set using a previously saved `selection_state.yaml`

Usage:
    python -m src.evaluation.evaluate \
        --config configs/config.yaml \
        --checkpoint checkpoints/best_baseline.pth \
        --model_type baseline \
        --selection_state_output artifacts/runs/<run_id>/selection/selection_state.yaml

    python -m src.evaluation.evaluate \
        --config configs/config.yaml \
        --checkpoint checkpoints/best_baseline.pth \
        --model_type baseline \
        --selection_state_input artifacts/runs/<run_id>/selection/selection_state.yaml
"""

from __future__ import annotations

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
SELECTION_STATE_FILENAME = "selection_state.yaml"
SELECTION_STATE_DIRNAME = "selection"
REQUIRED_SELECTION_STATE_KEYS = {
    "selection_split",
    "selection_metric",
    "selected_threshold",
    "selected_postprocess",
    "threshold_candidates",
    "threshold_summary",
    "model_type",
    "checkpoint_path",
    "dataset_root",
    "eval_mask_variant",
    "input_size",
}


def normalize_component_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def canonicalize_path(path_like: str | Path) -> str:
    return str(Path(path_like).resolve())


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


def validate_selection_state_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.name != SELECTION_STATE_FILENAME:
        raise ValueError(
            f"selection state path must end with {SELECTION_STATE_FILENAME!r}; got {path.name!r}."
        )
    if path.parent.name != SELECTION_STATE_DIRNAME:
        raise ValueError(
            f"selection state path must live inside a {SELECTION_STATE_DIRNAME!r} directory."
        )
    return path


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


def summarize_threshold_candidates_from_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
) -> list[dict[str, float]]:
    threshold_totals = {
        float(threshold): {
            "dice_sum": 0.0,
            "iou_sum": 0.0,
            "image_count": 0,
            "positive_dice_sum": 0.0,
            "positive_image_count": 0,
        }
        for threshold in thresholds
    }

    with torch.no_grad():
        for image, mask in tqdm(data_loader, desc="Selecting threshold"):
            image, mask = image.to(device), mask.to(device)
            pred = model(image)
            positive_mask = (mask > 0.5).reshape(mask.shape[0], -1).any(dim=1)

            for threshold in thresholds:
                per_image_dice = dice_score(pred, mask, threshold=threshold, reduction="none")
                per_image_iou = iou_score(pred, mask, threshold=threshold, reduction="none")
                totals = threshold_totals[float(threshold)]
                totals["dice_sum"] += float(per_image_dice.sum().item())
                totals["iou_sum"] += float(per_image_iou.sum().item())
                totals["image_count"] += int(per_image_dice.numel())
                if positive_mask.any():
                    totals["positive_dice_sum"] += float(per_image_dice[positive_mask].sum().item())
                    totals["positive_image_count"] += int(positive_mask.sum().item())

    summary = []
    for threshold in thresholds:
        totals = threshold_totals[float(threshold)]
        positive_count = totals["positive_image_count"]
        summary.append(
            {
                "threshold": float(threshold),
                "val_dice_pos_mean": (
                    totals["positive_dice_sum"] / positive_count
                    if positive_count > 0
                    else float("nan")
                ),
                "val_dice_mean": totals["dice_sum"] / max(totals["image_count"], 1),
                "val_iou_mean": totals["iou_sum"] / max(totals["image_count"], 1),
                "positive_image_count": positive_count,
            }
        )

    return summary


def build_selection_state_payload(
    selection_result: dict[str, Any],
    cfg: dict[str, Any],
    checkpoint_path: str,
    model_type: str,
) -> dict[str, Any]:
    return {
        "selection_split": selection_result["split"],
        "selection_metric": selection_result["selection_metric"],
        "selected_threshold": float(selection_result["selected_threshold"]),
        "selected_postprocess": selection_result["selected_postprocess"],
        "threshold_candidates": [
            float(row["threshold"]) for row in selection_result["threshold_summary"]
        ],
        "threshold_summary": selection_result["threshold_summary"],
        "model_type": model_type,
        "checkpoint_path": canonicalize_path(checkpoint_path),
        "dataset_root": canonicalize_path(cfg["data"]["processed_dir"]),
        "eval_mask_variant": cfg["data"].get("eval_mask_variant", "original_masks"),
        "input_size": int(cfg["data"]["input_size"]),
    }


def save_selection_state(
    selection_state_path: str | Path,
    selection_result: dict[str, Any],
    cfg: dict[str, Any],
    checkpoint_path: str,
    model_type: str,
) -> dict[str, Any]:
    path = validate_selection_state_path(selection_state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_selection_state_payload(
        selection_result,
        cfg,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
    )
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return payload


def load_selection_state(selection_state_path: str | Path) -> dict[str, Any]:
    path = validate_selection_state_path(selection_state_path)
    with path.open(encoding="utf-8") as handle:
        state = yaml.safe_load(handle) or {}

    missing_keys = REQUIRED_SELECTION_STATE_KEYS.difference(state)
    if missing_keys:
        raise ValueError(
            f"selection state is missing required keys: {sorted(missing_keys)}"
        )

    state["selection_split"] = validate_threshold_selection_split(state["selection_split"])
    state["selection_metric"] = resolve_component_choice(
        state["selection_metric"],
        SELECTION_METRIC_ALIASES,
        "selection_state.selection_metric",
    )
    state["selected_postprocess"] = resolve_component_choice(
        state["selected_postprocess"],
        POSTPROCESS_ALIASES,
        "selection_state.selected_postprocess",
    )
    state["threshold_candidates"] = sorted(
        {float(threshold) for threshold in state["threshold_candidates"]}
    )
    state["selected_threshold"] = float(state["selected_threshold"])
    if state["selected_threshold"] not in state["threshold_candidates"]:
        raise ValueError(
            "selection state selected_threshold must be present in threshold_candidates."
        )
    state["checkpoint_path"] = canonicalize_path(state["checkpoint_path"])
    state["dataset_root"] = canonicalize_path(state["dataset_root"])
    state["input_size"] = int(state["input_size"])
    return state


def resolve_test_evaluation_selection(
    cfg: dict[str, Any],
    checkpoint_path: str,
    model_type: str,
    selection_state_path: str | Path | None,
) -> tuple[float, dict[str, Any]]:
    if selection_state_path is None:
        raise ValueError(
            "Test evaluation requires --selection_state_input pointing to "
            "<run_dir>/selection/selection_state.yaml."
        )

    state = load_selection_state(selection_state_path)
    expected_checkpoint_path = canonicalize_path(checkpoint_path)
    expected_dataset_root = canonicalize_path(cfg["data"]["processed_dir"])
    expected_eval_mask_variant = cfg["data"].get("eval_mask_variant", "original_masks")
    expected_input_size = int(cfg["data"]["input_size"])

    if state["model_type"] != model_type:
        raise ValueError(
            f"selection state model_type {state['model_type']!r} does not match "
            f"current model_type {model_type!r}."
        )
    if state["checkpoint_path"] != expected_checkpoint_path:
        raise ValueError(
            "selection state checkpoint_path does not match the current evaluation checkpoint."
        )
    if state["dataset_root"] != expected_dataset_root:
        raise ValueError(
            "selection state dataset_root does not match the current evaluation dataset_root."
        )
    if state["eval_mask_variant"] != expected_eval_mask_variant:
        raise ValueError(
            "selection state eval_mask_variant does not match the current evaluation mask variant."
        )
    if state["input_size"] != expected_input_size:
        raise ValueError(
            "selection state input_size does not match the current evaluation input_size."
        )

    return float(state["selected_threshold"]), state


def build_model(cfg: dict, model_type: str) -> torch.nn.Module:
    in_ch = cfg["model"]["in_channels"]
    num_cls = cfg["model"]["num_classes"]
    base_f = cfg["model"]["base_filters"]

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
    """Return 'mean +/- std' string, ignoring NaNs."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return "nan +/- nan"
    mean = np.mean(clean)
    std = np.std(clean)
    return f"{mean:.4f} +/- {std:.4f}"


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

    pos_df = df[df["positive"] == True]
    neg_df = df[df["positive"] == False]

    print(f"\n{'=' * 60}")
    print(f"  Test Results: {model_type}")
    print(f"{'=' * 60}")
    print(f"  Samples - All: {len(df)} | Positive: {len(pos_df)} | Negative: {len(neg_df)}")
    print(f"{'=' * 60}")

    header = f"{'Subset':<12}" + "".join(f"{metric:>22}" for metric in metrics)
    print(header)
    print("-" * len(header))

    for label, subset in [("All", df), ("Positive", pos_df), ("Negative", neg_df)]:
        if len(subset) == 0:
            continue
        row = f"{label:<12}"
        for metric in metrics:
            row += f"{_stat(subset[metric].tolist()):>22}"
        print(row)

    print(f"{'=' * 60}\n")


def resolve_device(cfg: dict[str, Any]) -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu"
    )


def load_model_for_evaluation(
    cfg: dict[str, Any],
    checkpoint_path: str,
    model_type: str,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(cfg, model_type)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint: %s", checkpoint_path)
    return model


def build_eval_dataloader(
    cfg: dict[str, Any],
    *,
    split: str,
) -> tuple[PneumothoraxDataset, DataLoader]:
    dataset = PneumothoraxDataset(
        cfg["data"]["processed_dir"],
        split=split,
        img_size=cfg["data"]["input_size"],
        transform=None,
        mask_variant=cfg["data"].get("eval_mask_variant", "original_masks"),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    logger.info(
        "%s set: %d samples | eval_mask_variant=%s",
        split.capitalize(),
        len(dataset),
        dataset.mask_variant,
    )
    return dataset, loader


def select_threshold_and_save(
    cfg: dict[str, Any],
    checkpoint_path: str,
    model_type: str,
    selection_state_path: str | Path,
) -> dict[str, Any]:
    device = resolve_device(cfg)
    logger.info("Device: %s", device)
    model = load_model_for_evaluation(cfg, checkpoint_path, model_type, device)
    _, val_loader = build_eval_dataloader(cfg, split="val")
    selection_config = resolve_threshold_selection_config(cfg)
    threshold_summary = summarize_threshold_candidates_from_model(
        model,
        val_loader,
        device,
        selection_config["threshold_candidates"],
    )
    best_row = select_best_threshold(
        threshold_summary,
        metric_name=selection_config["metric"],
    )
    selection_result = {
        "split": "val",
        "selection_metric": selection_config["metric"],
        "selected_threshold": float(best_row["threshold"]),
        "selected_postprocess": selection_config["postprocess"],
        "threshold_summary": threshold_summary,
    }
    payload = save_selection_state(
        selection_state_path,
        selection_result,
        cfg,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
    )
    logger.info(
        "Saved selection state to %s | selected_threshold=%.2f | selection_metric=%s",
        validate_selection_state_path(selection_state_path),
        payload["selected_threshold"],
        payload["selection_metric"],
    )
    return payload


def evaluate(
    cfg: dict[str, Any],
    checkpoint_path: str,
    model_type: str,
    *,
    selection_state_path: str | Path | None,
) -> pd.DataFrame:
    device = resolve_device(cfg)
    logger.info("Device: %s", device)
    model = load_model_for_evaluation(cfg, checkpoint_path, model_type, device)
    test_ds, test_loader = build_eval_dataloader(cfg, split="test")
    threshold, selection_state = resolve_test_evaluation_selection(
        cfg,
        checkpoint_path,
        model_type,
        selection_state_path,
    )
    logger.info(
        "Using selected threshold %.2f from %s",
        threshold,
        validate_selection_state_path(selection_state_path) if selection_state_path is not None else "",
    )

    records = []
    with torch.no_grad():
        for idx, (image, mask) in enumerate(tqdm(test_loader, desc="Evaluating")):
            image, mask = image.to(device), mask.to(device)
            pred = model(image)
            is_positive = mask.sum().item() > 0
            metrics = compute_per_image_metrics(pred, mask, threshold=threshold)

            records.append(
                {
                    "image_id": test_ds.image_ids[idx],
                    "positive": is_positive,
                    **metrics,
                }
            )

    df = pd.DataFrame(records)

    Path("results").mkdir(exist_ok=True)
    out_path = f"results/test_metrics_{model_type}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved per-image metrics to %s", out_path)

    print_summary(df, model_type)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select validation threshold or evaluate Foundation-nnU-Net on test set"
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_baseline.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        help="Override config model type: 'baseline' or 'hybrid'",
    )
    parser.add_argument(
        "--selection_state_input",
        default=None,
        help="Path to <run_dir>/selection/selection_state.yaml used for test evaluation",
    )
    parser.add_argument(
        "--selection_state_output",
        default=None,
        help="Path to <run_dir>/selection/selection_state.yaml written by validation-only threshold selection",
    )
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    model_type = args.model_type if args.model_type else cfg["model"]["type"]
    if args.selection_state_output is not None:
        select_threshold_and_save(
            cfg,
            args.checkpoint,
            model_type,
            selection_state_path=args.selection_state_output,
        )
        return

    evaluate(
        cfg,
        args.checkpoint,
        model_type,
        selection_state_path=args.selection_state_input,
    )


if __name__ == "__main__":
    main()
