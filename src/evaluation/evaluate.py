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
from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any

import cv2
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
from src.training.run_artifacts import prepare_run_artifacts, write_evaluation_csv, write_yaml

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
    "train_mask_variant",
    "eval_mask_variant",
    "input_size",
}
METRIC_COLUMNS = ["dice", "iou", "precision", "recall", "f1"]
QUALITATIVE_SAMPLE_LIMIT_PER_CLASS = 4
QUALITATIVE_SELECTION_POLICY = "first_n_per_class_in_dataset_order"


@dataclass(frozen=True)
class QualitativeSample:
    image_id: str
    positive: bool
    subset_tag: str
    metrics: dict[str, float]
    image_uint8: np.ndarray
    target_mask_uint8: np.ndarray
    pred_mask_uint8: np.ndarray


def normalize_component_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def canonicalize_path(path_like: str | Path) -> str:
    return str(Path(path_like).resolve())


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prepare_evaluation_run_artifacts(model_type: str, selection_state_path: str | Path):
    validated_selection_state_path = validate_selection_state_path(selection_state_path)
    run_dir = validated_selection_state_path.parent.parent
    return prepare_run_artifacts(
        model_type,
        run_dir=run_dir,
        run_root=resolve_repo_root() / "artifacts" / "runs",
    )


def sync_run_metadata_with_selection_state(
    run_metadata_path: Path,
    selection_state: dict[str, Any],
) -> None:
    if not run_metadata_path.exists():
        logger.warning(
            "Run metadata not found while syncing selected threshold: %s",
            run_metadata_path,
        )
        return

    with run_metadata_path.open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}

    metadata["selection_metric"] = selection_state["selection_metric"]
    metadata["selected_threshold"] = float(selection_state["selected_threshold"])
    metadata["selected_postprocess"] = selection_state["selected_postprocess"]
    write_yaml(run_metadata_path, metadata)


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
    *,
    selection_state_path: str | Path,
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
        "selection_state_path": canonicalize_path(selection_state_path),
        "train_mask_variant": cfg["data"].get("train_mask_variant", "dilated_masks"),
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
        selection_state_path=path,
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
    state["selection_state_path"] = canonicalize_path(path)
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
    expected_train_mask_variant = cfg["data"].get("train_mask_variant", "dilated_masks")
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
    if state["train_mask_variant"] != expected_train_mask_variant:
        raise ValueError(
            "selection state train_mask_variant does not match the current evaluation mask variant policy."
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


def summarize_metric_values(values: list[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    if not clean:
        return {"mean": None, "std": None}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
    }


def resolve_subset_tag(*, positive: bool) -> str:
    return "positive" if positive else "negative"


def build_test_summary_payload(
    df: pd.DataFrame,
    selection_state: dict[str, Any],
    checkpoint_path: str,
    model_type: str,
) -> dict[str, Any]:
    subsets = {
        "all": df,
        "positive": df[df["positive"] == True],
        "negative": df[df["positive"] == False],
    }
    subset_summary: dict[str, Any] = {}
    for subset_name, subset_df in subsets.items():
        metrics_summary: dict[str, Any] = {"count": int(len(subset_df))}
        for metric_name in METRIC_COLUMNS:
            metric_summary = summarize_metric_values(subset_df[metric_name].tolist())
            metrics_summary[metric_name] = metric_summary
        subset_summary[subset_name] = metrics_summary

    return {
        "split": "test",
        "model_type": model_type,
        "checkpoint_path": canonicalize_path(checkpoint_path),
        "dataset_root": selection_state["dataset_root"],
        "selection_state_path": selection_state["selection_state_path"],
        "train_mask_variant": selection_state["train_mask_variant"],
        "eval_mask_variant": selection_state["eval_mask_variant"],
        "input_size": int(selection_state["input_size"]),
        "selection_metric": selection_state["selection_metric"],
        "selected_threshold": float(selection_state["selected_threshold"]),
        "selected_postprocess": selection_state["selected_postprocess"],
        "subsets": subset_summary,
    }


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    return np.clip(np.rint(image.squeeze().detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)


def tensor_mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    return ((mask.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255)


def build_overlay_image(
    image_uint8: np.ndarray,
    target_mask_uint8: np.ndarray,
    pred_mask_uint8: np.ndarray,
) -> np.ndarray:
    overlay = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    target_mask = target_mask_uint8 > 0
    pred_mask = pred_mask_uint8 > 0
    both_mask = target_mask & pred_mask

    if target_mask.any():
        overlay[target_mask] = (0.6 * overlay[target_mask] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
    if pred_mask.any():
        overlay[pred_mask] = (0.6 * overlay[pred_mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    if both_mask.any():
        overlay[both_mask] = (0.4 * overlay[both_mask] + 0.6 * np.array([0, 255, 255])).astype(np.uint8)

    return overlay


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write PNG artifact: {path}")


def build_qualitative_manifest(
    *,
    split: str,
    model_type: str,
    checkpoint_path: str,
    selection_state: dict[str, Any],
    samples: list[QualitativeSample],
) -> dict[str, Any]:
    manifest_samples = []
    for index, sample in enumerate(samples, start=1):
        sample_prefix = f"{index:02d}_{sample.image_id}"
        manifest_samples.append(
            {
                "image_id": sample.image_id,
                "positive": bool(sample.positive),
                "subset_tag": sample.subset_tag,
                "metrics": {name: float(value) for name, value in sample.metrics.items()},
                "files": {
                    "image": f"{sample_prefix}_image.png",
                    "target_mask": f"{sample_prefix}_target_mask.png",
                    "prediction_mask": f"{sample_prefix}_prediction_mask.png",
                    "overlay": f"{sample_prefix}_overlay.png",
                },
            }
        )

    return {
        "split": split,
        "model_type": model_type,
        "checkpoint_path": canonicalize_path(checkpoint_path),
        "dataset_root": selection_state["dataset_root"],
        "selection_state_path": selection_state["selection_state_path"],
        "train_mask_variant": selection_state["train_mask_variant"],
        "eval_mask_variant": selection_state["eval_mask_variant"],
        "input_size": int(selection_state["input_size"]),
        "selection_metric": selection_state["selection_metric"],
        "selected_threshold": float(selection_state["selected_threshold"]),
        "selected_postprocess": selection_state["selected_postprocess"],
        "sample_selection_policy": {
            "type": QUALITATIVE_SELECTION_POLICY,
            "positive_limit": QUALITATIVE_SAMPLE_LIMIT_PER_CLASS,
            "negative_limit": QUALITATIVE_SAMPLE_LIMIT_PER_CLASS,
        },
        "samples": manifest_samples,
    }


def write_qualitative_package(
    output_dir: Path,
    *,
    split: str,
    model_type: str,
    checkpoint_path: str,
    selection_state: dict[str, Any],
    samples: list[QualitativeSample],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_qualitative_manifest(
        split=split,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        selection_state=selection_state,
        samples=samples,
    )

    for index, sample in enumerate(samples, start=1):
        sample_prefix = f"{index:02d}_{sample.image_id}"
        write_png(output_dir / f"{sample_prefix}_image.png", sample.image_uint8)
        write_png(output_dir / f"{sample_prefix}_target_mask.png", sample.target_mask_uint8)
        write_png(output_dir / f"{sample_prefix}_prediction_mask.png", sample.pred_mask_uint8)
        write_png(
            output_dir / f"{sample_prefix}_overlay.png",
            build_overlay_image(
                sample.image_uint8,
                sample.target_mask_uint8,
                sample.pred_mask_uint8,
            ),
        )

    write_yaml(output_dir / "manifest.yaml", manifest)
    return manifest


def collect_split_records_and_samples(
    model: torch.nn.Module,
    dataset: PneumothoraxDataset,
    data_loader: DataLoader,
    device: torch.device,
    *,
    split: str,
    checkpoint_path: str,
    model_type: str,
    selection_state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[QualitativeSample]]:
    records: list[dict[str, Any]] = []
    samples: list[QualitativeSample] = []
    positive_sample_count = 0
    negative_sample_count = 0
    threshold = float(selection_state["selected_threshold"])

    with torch.no_grad():
        for idx, (image, mask) in enumerate(tqdm(data_loader, desc=f"Evaluating {split}")):
            image, mask = image.to(device), mask.to(device)
            pred = model(image)
            is_positive = bool(mask.sum().item() > 0)
            subset_tag = resolve_subset_tag(positive=is_positive)
            all_metrics = compute_per_image_metrics(pred, mask, threshold=threshold)
            reported_metrics = {
                metric_name: all_metrics[metric_name] for metric_name in METRIC_COLUMNS
            }
            image_id = dataset.image_ids[idx]

            records.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "subset_tag": subset_tag,
                    "model_type": model_type,
                    "checkpoint_path": canonicalize_path(checkpoint_path),
                    "selection_state_path": selection_state["selection_state_path"],
                    "train_mask_variant": selection_state["train_mask_variant"],
                    "eval_mask_variant": selection_state["eval_mask_variant"],
                    "selection_metric": selection_state["selection_metric"],
                    "selected_threshold": threshold,
                    "selected_postprocess": selection_state["selected_postprocess"],
                    "positive": is_positive,
                    **reported_metrics,
                }
            )

            should_capture = False
            if is_positive and positive_sample_count < QUALITATIVE_SAMPLE_LIMIT_PER_CLASS:
                positive_sample_count += 1
                should_capture = True
            elif not is_positive and negative_sample_count < QUALITATIVE_SAMPLE_LIMIT_PER_CLASS:
                negative_sample_count += 1
                should_capture = True

            if should_capture:
                pred_mask_uint8 = ((pred > threshold).squeeze().detach().cpu().numpy().astype(np.uint8) * 255)
                samples.append(
                    QualitativeSample(
                        image_id=image_id,
                        positive=is_positive,
                        subset_tag=subset_tag,
                        metrics=reported_metrics,
                        image_uint8=tensor_image_to_uint8(image),
                        target_mask_uint8=tensor_mask_to_uint8(mask),
                        pred_mask_uint8=pred_mask_uint8,
                    )
                )

    return records, samples


def print_summary(df: pd.DataFrame, model_type: str) -> None:
    metrics = METRIC_COLUMNS

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
    run_artifacts = prepare_evaluation_run_artifacts(model_type, selection_state_path)
    device = resolve_device(cfg)
    logger.info("Device: %s", device)
    model = load_model_for_evaluation(cfg, checkpoint_path, model_type, device)
    val_ds, val_loader = build_eval_dataloader(cfg, split="val")
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
        run_artifacts.selection_state_path,
        selection_result,
        cfg,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
    )
    sync_run_metadata_with_selection_state(run_artifacts.run_metadata_path, payload)
    _, qualitative_samples = collect_split_records_and_samples(
        model,
        val_ds,
        val_loader,
        device,
        split="val",
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        selection_state=payload,
    )
    write_qualitative_package(
        run_artifacts.qualitative_validation_dir,
        split="val",
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        selection_state=payload,
        samples=qualitative_samples,
    )
    logger.info(
        "Saved selection state to %s | selected_threshold=%.2f | selection_metric=%s | validation qualitative=%s",
        run_artifacts.selection_state_path,
        payload["selected_threshold"],
        payload["selection_metric"],
        run_artifacts.qualitative_validation_dir,
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
    run_artifacts = prepare_evaluation_run_artifacts(model_type, selection_state_path)
    sync_run_metadata_with_selection_state(run_artifacts.run_metadata_path, selection_state)
    logger.info(
        "Using selected threshold %.2f from %s",
        threshold,
        validate_selection_state_path(selection_state_path) if selection_state_path is not None else "",
    )

    records, qualitative_samples = collect_split_records_and_samples(
        model,
        test_ds,
        test_loader,
        device,
        split="test",
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        selection_state=selection_state,
    )
    df = write_evaluation_csv(run_artifacts.test_metrics_path, records)
    write_yaml(
        run_artifacts.test_summary_path,
        build_test_summary_payload(df, selection_state, checkpoint_path, model_type),
    )
    write_qualitative_package(
        run_artifacts.qualitative_test_dir,
        split="test",
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        selection_state=selection_state,
        samples=qualitative_samples,
    )
    logger.info(
        "Saved evaluation reports to %s and %s | test qualitative=%s",
        run_artifacts.test_metrics_path,
        run_artifacts.test_summary_path,
        run_artifacts.qualitative_test_dir,
    )

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
        help="Override config model type: 'baseline', 'pretrained_resnet34_unet', or 'hybrid'",
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
