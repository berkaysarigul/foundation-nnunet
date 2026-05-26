"""
PR-9 hybrid heldout inference + evaluation.

Runs the PR-8 best HybridFoundationUNet checkpoint over
`nnUNet_raw/Dataset101_Pneumothorax/imagesTs` and evaluates against
`heldout_labelsTs`. Produces per-case and aggregate metrics matching
the nnU-Net baseline schema, plus failure-analysis CSVs and visual
diagnostic grids.

Hard constraints:
  - No training, no optimizer, no backward pass.
  - All forwards inside `torch.no_grad()`.
  - Foundation X stays frozen and in eval().
  - Output dir MUST be under
    `artifacts/diagnostics/hybrid_heldout_eval/`. The script refuses
    `artifacts/runs/`, `nnUNet_raw`, `nnunet_results*`, or any PR-7/8
    diagnostic directory.

Example (Colab, threshold 0.5):

    py scripts/evaluate_hybrid_heldout.py \\
      --images_dir nnUNet_raw/Dataset101_Pneumothorax/imagesTs \\
      --labels_dir nnUNet_raw/Dataset101_Pneumothorax/heldout_labelsTs \\
      --hybrid_checkpoint artifacts/diagnostics/hybrid_controlled_short_train/larger_lr3e5_1200step_bnfix/diagnostic_checkpoint_best_dice_pos_mean_thr_050.pth \\
      --foundation_checkpoint checkpoints/foundation_x.pth \\
      --output_dir artifacts/diagnostics/hybrid_heldout_eval/pr8_best_thr05 \\
      --threshold 0.5 --img_size 512 --device auto \\
      --save_pred_masks --num_visuals_per_grid 8 --strict
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import yaml as _yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


ALLOWED_OUTPUT_ANCHOR = "artifacts/diagnostics/hybrid_heldout_eval"
FORBIDDEN_OUTPUT_SEGMENTS = (
    "artifacts/runs",
    "nnunet_results",
    "nnunet_results_smoke",
    "nnunet_raw",
    "artifacts/diagnostics/foundation_x_smoke",
    "artifacts/diagnostics/hybrid_adapter_sanity",
    "artifacts/diagnostics/hybrid_tiny_train_smoke",
    "artifacts/diagnostics/hybrid_controlled_short_train",
)

NNUNET_BASELINE = {
    "total_heldout_cases": 1602,
    "positive_cases": 357,
    "negative_cases": 1245,
    "tp": 247,
    "fn": 110,
    "fp": 73,
    "tn": 1172,
    "case_level_precision": 0.7719,
    "case_level_recall": 0.6919,
    "case_level_specificity": 0.9414,
    "case_level_f1": 0.7297,
    "mean_dice_positive_cases": 0.3722,
    "mean_dice_detected_positives_only": 0.5380,
    "negative_case_false_positive_rate": 0.0586,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inference-only heldout evaluation of the PR-8 hybrid best "
            "checkpoint. Compares case-level and pixel-level metrics "
            "against the pure nnU-Net heldout baseline."
        ),
    )
    parser.add_argument("--images_dir", type=Path, required=True,
                        help="Directory of siim_NNNNNN_0000.png heldout images.")
    parser.add_argument("--labels_dir", type=Path, required=True,
                        help="Directory of siim_NNNNNN.png heldout labels.")
    parser.add_argument("--hybrid_checkpoint", type=Path, required=True,
                        help="PR-8 best hybrid checkpoint (.pth).")
    parser.add_argument("--foundation_checkpoint", type=Path, required=True,
                        help="Foundation X checkpoint (.pth).")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory under artifacts/diagnostics/hybrid_heldout_eval/.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binary prediction.")
    parser.add_argument("--img_size", type=int, default=512, choices=[256, 512],
                        help="Runtime input size. Must match the PR-8 training img_size.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto|cuda|cpu.")
    parser.add_argument("--save_pred_masks", action="store_true", default=True,
                        help="Save per-case binary prediction PNGs (default: on).")
    parser.add_argument("--no_save_pred_masks", action="store_true",
                        help="Disable saving binary prediction PNGs.")
    parser.add_argument("--save_prob_maps", action="store_true",
                        help="Save per-case probability map PNGs (uint8 percentile).")
    parser.add_argument("--save_logit_maps", action="store_true",
                        help="Save per-case raw logit map PNGs (uint8 percentile).")
    parser.add_argument("--num_visuals_per_grid", type=int, default=8,
                        help="Number of cases per visual diagnostic grid.")
    parser.add_argument("--strict", action="store_true",
                        help="Promote non-empty failures list to BLOCKED.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic seed.")
    parser.add_argument("--max_cases", type=int, default=0,
                        help="If > 0, limit inference to this many cases (debug only).")
    return parser.parse_args(argv)


def _resolve_device(device_arg: str) -> Any:
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def validate_output_guardrail(output_dir: Path) -> tuple[bool, str]:
    normalized = output_dir.resolve().as_posix().lower()
    if ALLOWED_OUTPUT_ANCHOR not in normalized:
        return False, f"Output dir must be under {ALLOWED_OUTPUT_ANCHOR}: {output_dir}"
    for segment in FORBIDDEN_OUTPUT_SEGMENTS:
        if segment in normalized:
            return False, f"Output dir falls into forbidden segment '{segment}': {output_dir}"
    return True, ""


def _normalize_case_id(raw: str) -> str:
    case_id = raw.strip()
    if case_id.endswith(".png"):
        case_id = case_id[: -len(".png")]
    if case_id.endswith("_0000"):
        case_id = case_id[: -len("_0000")]
    return case_id


def list_heldout_cases(images_dir: Path, labels_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for image_path in sorted(images_dir.glob("siim_*_0000.png")):
        case_id = _normalize_case_id(image_path.stem)
        label_path = labels_dir / f"{case_id}.png"
        cases.append(
            {
                "case_id": case_id,
                "image_path": image_path,
                "label_path": label_path,
                "label_exists": label_path.exists(),
            },
        )
    return cases


def _load_image_label(
    image_path: Path,
    label_path: Path,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    from PIL import Image

    pil_img = Image.open(image_path).convert("L")
    if pil_img.size != (img_size, img_size):
        pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
    image_arr = np.array(pil_img, dtype=np.float32) / 255.0

    label_exists = label_path.exists()
    if label_exists:
        pil_lbl = Image.open(label_path).convert("L")
        if pil_lbl.size != (img_size, img_size):
            pil_lbl = pil_lbl.resize((img_size, img_size), Image.NEAREST)
        mask_arr = (np.array(pil_lbl, dtype=np.float32) > 0.5).astype(np.float32)
    else:
        mask_arr = np.zeros_like(image_arr, dtype=np.float32)
    return image_arr, mask_arr, label_exists


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _save_uint8_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def per_case_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
) -> dict[str, float]:
    """Pixel-level metrics for a single case."""
    pred = pred_binary.astype(np.float32)
    gt = gt_binary.astype(np.float32)
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    inter = float((pred * gt).sum())
    union = pred_sum + gt_sum - inter

    if pred_sum == 0.0 and gt_sum == 0.0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    elif pred_sum + gt_sum == 0.0:
        dice = 0.0
        iou = 0.0
        precision = 0.0
        recall = 0.0
    else:
        dice = (2.0 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
        iou = inter / union if union > 0 else 0.0
        precision = inter / pred_sum if pred_sum > 0 else 0.0
        recall = inter / gt_sum if gt_sum > 0 else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "intersection_pixels": int(inter),
        "pred_foreground_pixels": int(pred_sum),
        "gt_foreground_pixels": int(gt_sum),
        "total_pixels": int(pred_binary.size),
    }


def aggregate_case_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the full aggregate-metric schema from per-case rows.

    Each row must contain: dice, iou, precision, recall,
    gt_foreground_pixels, pred_foreground_pixels, gt_is_positive,
    pred_is_positive, label_available.
    """
    n = len(rows)
    rows_with_label = [r for r in rows if r.get("label_available")]
    n_with_label = len(rows_with_label)

    gt_pos_rows = [r for r in rows_with_label if r["gt_is_positive"]]
    gt_neg_rows = [r for r in rows_with_label if not r["gt_is_positive"]]

    pred_pos_rows = [r for r in rows_with_label if r["pred_is_positive"]]

    tp = sum(1 for r in rows_with_label if r["gt_is_positive"] and r["pred_is_positive"])
    fn = sum(1 for r in rows_with_label if r["gt_is_positive"] and not r["pred_is_positive"])
    fp = sum(1 for r in rows_with_label if (not r["gt_is_positive"]) and r["pred_is_positive"])
    tn = sum(1 for r in rows_with_label if (not r["gt_is_positive"]) and (not r["pred_is_positive"]))

    def _safe_div(num: float, den: float) -> float:
        return float(num) / float(den) if den > 0 else 0.0

    case_level_precision = _safe_div(tp, tp + fp)
    case_level_recall = _safe_div(tp, tp + fn)
    case_level_specificity = _safe_div(tn, tn + fp)
    case_level_f1 = _safe_div(2 * case_level_precision * case_level_recall,
                              case_level_precision + case_level_recall)

    def _mean(vs: list[float]) -> float:
        return float(statistics.fmean(vs)) if vs else float("nan")

    def _median(vs: list[float]) -> float:
        return float(statistics.median(vs)) if vs else float("nan")

    dice_all = [r["dice"] for r in rows_with_label]
    iou_all = [r["iou"] for r in rows_with_label]
    prec_all = [r["precision"] for r in rows_with_label]
    rec_all = [r["recall"] for r in rows_with_label]

    dice_pos = [r["dice"] for r in gt_pos_rows]
    iou_pos = [r["iou"] for r in gt_pos_rows]
    prec_pos = [r["precision"] for r in gt_pos_rows]
    rec_pos = [r["recall"] for r in gt_pos_rows]

    detected = [r for r in gt_pos_rows if r["pred_is_positive"]]
    dice_detected = [r["dice"] for r in detected]

    neg_fp = [r for r in gt_neg_rows if r["pred_is_positive"]]
    negative_case_fpr = _safe_div(len(neg_fp), len(gt_neg_rows))

    return {
        "total_cases": n,
        "cases_with_label": n_with_label,
        "gt_positive_cases": len(gt_pos_rows),
        "gt_negative_cases": len(gt_neg_rows),
        "pred_positive_cases": len(pred_pos_rows),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "case_level_precision": case_level_precision,
        "case_level_recall": case_level_recall,
        "case_level_specificity": case_level_specificity,
        "case_level_f1": case_level_f1,
        "mean_dice_all_cases": _mean(dice_all),
        "median_dice_all_cases": _median(dice_all),
        "mean_iou_all_cases": _mean(iou_all),
        "mean_precision_all_cases": _mean(prec_all),
        "mean_recall_all_cases": _mean(rec_all),
        "mean_dice_positive_cases": _mean(dice_pos),
        "median_dice_positive_cases": _median(dice_pos),
        "mean_iou_positive_cases": _mean(iou_pos),
        "mean_precision_positive_cases": _mean(prec_pos),
        "mean_recall_positive_cases": _mean(rec_pos),
        "mean_dice_detected_positives_only": _mean(dice_detected),
        "median_dice_detected_positives_only": _median(dice_detected),
        "negative_case_false_positive_rate": negative_case_fpr,
        "detected_positive_count": len(detected),
    }


def _instantiate_model_and_hook(
    foundation_checkpoint: Path,
    hybrid_checkpoint: Path,
    img_size: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    from src.models.hybrid import HybridFoundationUNet  # noqa: WPS433

    model = HybridFoundationUNet(
        backbone_checkpoint=str(foundation_checkpoint),
        in_channels=1,
        num_classes=1,
        base_filters=64,
        frozen_backbone=True,
        img_size=img_size,
    )

    diag_payload = torch.load(str(hybrid_checkpoint), map_location="cpu", weights_only=False)
    state_dict = diag_payload.get("model_state_dict") if isinstance(diag_payload, dict) else None
    if state_dict is None:
        raise KeyError("hybrid checkpoint missing 'model_state_dict'")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    if getattr(model, "frozen_backbone", False):
        model.foundation_x.backbone.eval()

    hook_state: dict[str, Any] = {"raw_logits": None}

    def _hook(_module: Any, _inputs: Any, output: Any) -> None:
        hook_state["raw_logits"] = output

    handle = model.final.register_forward_hook(_hook)
    hook_state["handle"] = handle
    hook_state["missing_keys"] = [str(k) for k in missing]
    hook_state["unexpected_keys"] = [str(k) for k in unexpected]
    hook_state["diagnostic_payload_meta"] = {
        "step": diag_payload.get("step") if isinstance(diag_payload, dict) else None,
        "status": diag_payload.get("status") if isinstance(diag_payload, dict) else None,
        "selection_metric": diag_payload.get("selection_metric") if isinstance(diag_payload, dict) else None,
        "metric_value": diag_payload.get("metric_value") if isinstance(diag_payload, dict) else None,
        "audit_name": diag_payload.get("audit_name") if isinstance(diag_payload, dict) else None,
        "schema_version": diag_payload.get("schema_version") if isinstance(diag_payload, dict) else None,
    }
    return model, hook_state


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, bool):
        return bool(obj)
    return obj


def _validate_inputs(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for attr, label in (
        ("images_dir", "images_dir"),
        ("labels_dir", "labels_dir"),
        ("hybrid_checkpoint", "hybrid_checkpoint"),
        ("foundation_checkpoint", "foundation_checkpoint"),
    ):
        path = getattr(args, attr)
        if not path.exists():
            failures.append(f"{label} not found: {path}")
    if args.threshold < 0.0 or args.threshold > 1.0:
        failures.append(f"threshold must be in [0, 1]; got {args.threshold}")
    if args.num_visuals_per_grid <= 0:
        failures.append(f"num_visuals_per_grid must be > 0; got {args.num_visuals_per_grid}")
    return failures


def _build_overlay_uint8(
    image_uint8: np.ndarray,
    gt_uint8: np.ndarray,
    pred_uint8: np.ndarray,
) -> np.ndarray:
    rgb = np.stack([image_uint8, image_uint8, image_uint8], axis=-1).astype(np.float32)
    gt = gt_uint8 > 0
    pred = pred_uint8 > 0
    both = gt & pred
    if gt.any():
        rgb[gt] = 0.6 * rgb[gt] + 0.4 * np.array([255, 0, 0], dtype=np.float32)
    if pred.any():
        rgb[pred] = 0.6 * rgb[pred] + 0.4 * np.array([0, 255, 0], dtype=np.float32)
    if both.any():
        rgb[both] = 0.4 * rgb[both] + 0.6 * np.array([255, 255, 0], dtype=np.float32)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _render_grid(
    title: str,
    samples: list[dict[str, Any]],
    output_path: Path,
    img_size: int,
) -> bool:
    if not samples:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return False

    n = len(samples)
    cols = 1
    rows = n
    fig, axes = plt.subplots(rows, cols, figsize=(6, 3 * rows))
    if rows == 1:
        axes = [axes]
    for ax, sample in zip(axes, samples):
        overlay = sample["overlay"]
        ax.imshow(overlay)
        ax.set_title(
            f"{sample['case_id']}  dice={sample['dice']:.3f}  "
            f"pred_px={sample['pred_foreground_pixels']}  gt_px={sample['gt_foreground_pixels']}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return True


def _write_case_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "case_id", "label_available", "gt_is_positive", "pred_is_positive",
        "gt_foreground_pixels", "pred_foreground_pixels", "total_pixels",
        "intersection_pixels",
        "dice", "iou", "precision", "recall",
        "prob_min", "prob_max", "prob_mean", "prob_std",
        "logits_min", "logits_max", "logits_mean", "logits_std",
        "latency_ms",
        "image_path", "label_path", "pred_mask_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_subset_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "case_id", "gt_is_positive", "pred_is_positive",
        "gt_foreground_pixels", "pred_foreground_pixels",
        "dice", "iou", "precision", "recall",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _build_report_markdown(
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> str:
    metrics = summary.get("aggregate_metrics", {})
    baseline = NNUNET_BASELINE
    lines: list[str] = [
        "# Hybrid Heldout Evaluation Report",
        "",
        f"- Hybrid checkpoint: `{summary['setup']['hybrid_checkpoint']}`",
        f"- Foundation X: `{summary['setup']['foundation_checkpoint']}`",
        f"- imagesTs: `{summary['setup']['images_dir']}`",
        f"- heldout_labelsTs: `{summary['setup']['labels_dir']}`",
        f"- threshold: `{summary['setup']['threshold']}`",
        f"- img_size: `{summary['setup']['img_size']}`",
        f"- device: `{summary['setup']['device']}`",
        f"- generated: `{summary['generated_at_utc']}`",
        "",
        f"## Status: `{summary['status']}` (exit code {summary['exit_code']})",
        "",
        "## Comparison vs pure nnU-Net heldout baseline",
        "",
        "| metric | nnU-Net baseline | hybrid PR-8 best | Δ (hybrid − baseline) | beats baseline? |",
        "|---|---:|---:|---:|---|",
    ]

    def _add_row(label: str, key: str, baseline_value: float, higher_is_better: bool) -> None:
        hv = metrics.get(key)
        if hv is None or (isinstance(hv, float) and (math.isnan(hv) or math.isinf(hv))):
            lines.append(f"| {label} | {baseline_value:.4f} | n/a | n/a | n/a |")
            return
        delta = hv - baseline_value
        if higher_is_better:
            beats = "YES" if hv >= baseline_value else "NO"
        else:
            beats = "YES" if hv <= baseline_value else "NO"
        lines.append(
            f"| {label} | {baseline_value:.4f} | {hv:.4f} | {delta:+.4f} | {beats} |",
        )

    _add_row("case-level precision", "case_level_precision", baseline["case_level_precision"], True)
    _add_row("case-level recall", "case_level_recall", baseline["case_level_recall"], True)
    _add_row("case-level specificity", "case_level_specificity", baseline["case_level_specificity"], True)
    _add_row("case-level F1", "case_level_f1", baseline["case_level_f1"], True)
    _add_row("mean Dice positive cases", "mean_dice_positive_cases", baseline["mean_dice_positive_cases"], True)
    _add_row("mean Dice detected positives only", "mean_dice_detected_positives_only", baseline["mean_dice_detected_positives_only"], True)
    _add_row("negative case FPR (lower is better)", "negative_case_false_positive_rate", baseline["negative_case_false_positive_rate"], False)

    lines.extend([
        "",
        "## Heldout aggregate",
        "",
        f"- total cases processed: {summary['inference']['processed_cases']}",
        f"- cases with label: {metrics.get('cases_with_label')}",
        f"- gt positive cases: {metrics.get('gt_positive_cases')}",
        f"- gt negative cases: {metrics.get('gt_negative_cases')}",
        f"- pred positive cases: {metrics.get('pred_positive_cases')}",
        f"- TP / FN / FP / TN: {metrics.get('tp')} / {metrics.get('fn')} / {metrics.get('fp')} / {metrics.get('tn')}",
        f"- mean Dice (all cases): {metrics.get('mean_dice_all_cases')}",
        f"- median Dice (all cases): {metrics.get('median_dice_all_cases')}",
        f"- mean IoU (all cases): {metrics.get('mean_iou_all_cases')}",
        f"- mean precision (all cases): {metrics.get('mean_precision_all_cases')}",
        f"- mean recall (all cases): {metrics.get('mean_recall_all_cases')}",
        f"- detected positive count: {metrics.get('detected_positive_count')}",
        "",
        "## Tiny-positive subgroup",
        "",
    ])
    tg = summary.get("tiny_positive_subgroup")
    if tg:
        for bucket_name in ("tiny", "medium", "large"):
            bucket = tg.get(bucket_name, {})
            lines.append(
                f"- {bucket_name} (gt_foreground_ratio range "
                f"[{bucket.get('range_low')}, {bucket.get('range_high')}]): "
                f"n={bucket.get('count')}, mean_dice={bucket.get('mean_dice')}, "
                f"detection_rate={bucket.get('detection_rate')}"
            )
    else:
        lines.append("- (not computed; no positive cases)")

    lines.extend([
        "",
        "## Checkpoint metadata",
        "",
        f"- state_dict missing_keys_count: {summary['state_dict']['missing_keys_count']}",
        f"- state_dict unexpected_keys_count: {summary['state_dict']['unexpected_keys_count']}",
        f"- PR-8 step (per ckpt payload): {summary['checkpoint_meta'].get('step')}",
        f"- PR-8 selection_metric: {summary['checkpoint_meta'].get('selection_metric')}",
        f"- PR-8 metric_value: {summary['checkpoint_meta'].get('metric_value')}",
        "",
        "## Outputs",
        "",
        f"- summary_json: `{summary['outputs']['summary_json']}`",
        f"- summary_yaml: `{summary['outputs']['summary_yaml']}`",
        f"- report_md: `{summary['outputs']['report_md']}`",
        f"- case_csv: `{summary['outputs']['case_csv']}`",
        f"- pred_masks_dir: `{summary['outputs']['pred_masks_dir']}` (saved={summary['outputs']['pred_masks_saved']})",
        f"- visuals_dir: `{summary['outputs']['visuals_dir']}`",
        f"- visual_grids_written: {summary['outputs']['visual_grids_written']}",
        "",
        "## Failures",
        "",
    ])
    if summary.get("failures"):
        for f in summary["failures"]:
            lines.append(f"- {f}")
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def _select_grid_samples(
    rows: list[dict[str, Any]],
    cases_data: dict[str, dict[str, Any]],
    img_size: int,
    n: int,
) -> dict[str, list[dict[str, Any]]]:
    """Return rows grouped per grid kind. Each sample dict carries an `overlay`."""

    def _materialize(rs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in rs[:n]:
            data = cases_data.get(r["case_id"])
            if data is None:
                continue
            overlay = _build_overlay_uint8(data["image_uint8"], data["gt_uint8"], data["pred_uint8"])
            out.append({
                "case_id": r["case_id"],
                "dice": r["dice"],
                "pred_foreground_pixels": r["pred_foreground_pixels"],
                "gt_foreground_pixels": r["gt_foreground_pixels"],
                "overlay": overlay,
            })
        return out

    with_label = [r for r in rows if r.get("label_available")]
    pos = [r for r in with_label if r["gt_is_positive"]]
    neg = [r for r in with_label if not r["gt_is_positive"]]
    pos_detected = [r for r in pos if r["pred_is_positive"]]
    pos_missed = [r for r in pos if not r["pred_is_positive"]]
    neg_clean = [r for r in neg if not r["pred_is_positive"]]

    largest_fp = sorted(
        [r for r in neg if r["pred_is_positive"]],
        key=lambda r: r["pred_foreground_pixels"], reverse=True,
    )
    largest_fn = sorted(pos_missed, key=lambda r: r["gt_foreground_pixels"], reverse=True)
    worst_pos = sorted(pos, key=lambda r: r["dice"])
    best_pos = sorted(pos, key=lambda r: r["dice"], reverse=True)
    if pos_detected:
        sorted_detected = sorted(pos_detected, key=lambda r: r["dice"])
        median_idx = len(sorted_detected) // 2
        rep_tp = sorted_detected[max(0, median_idx - n // 2): max(0, median_idx - n // 2) + n]
    else:
        rep_tp = []
    rep_tn = neg_clean[:n]

    return {
        "grid_largest_false_positives.png": _materialize(largest_fp),
        "grid_largest_false_negatives.png": _materialize(largest_fn),
        "grid_worst_positive_dice.png": _materialize(worst_pos),
        "grid_best_positive_dice.png": _materialize(best_pos),
        "grid_representative_true_positives.png": _materialize(rep_tp),
        "grid_representative_true_negatives.png": _materialize(rep_tn),
    }


def _compute_tiny_subgroup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in rows if r.get("label_available") and r["gt_is_positive"]]
    if not pos:
        return {}
    ratios = sorted(r["gt_foreground_pixels"] / max(1, r["total_pixels"]) for r in pos)
    n = len(ratios)
    t1 = ratios[n // 3] if n >= 3 else ratios[0]
    t2 = ratios[(2 * n) // 3] if n >= 3 else ratios[-1]

    def _bucket_stats(low: float, high: float, inclusive_high: bool) -> dict[str, Any]:
        if inclusive_high:
            bucket = [r for r in pos if low <= r["gt_foreground_pixels"] / max(1, r["total_pixels"]) <= high]
        else:
            bucket = [r for r in pos if low <= r["gt_foreground_pixels"] / max(1, r["total_pixels"]) < high]
        if not bucket:
            return {"count": 0, "mean_dice": None, "detection_rate": None,
                    "range_low": float(low), "range_high": float(high)}
        dice_mean = statistics.fmean(r["dice"] for r in bucket)
        det_rate = sum(1 for r in bucket if r["pred_is_positive"]) / len(bucket)
        return {
            "count": len(bucket),
            "mean_dice": float(dice_mean),
            "detection_rate": float(det_rate),
            "range_low": float(low),
            "range_high": float(high),
        }

    return {
        "tiny":   _bucket_stats(ratios[0], t1, inclusive_high=False),
        "medium": _bucket_stats(t1, t2, inclusive_high=False),
        "large":  _bucket_stats(t2, ratios[-1], inclusive_high=True),
    }


def _new_tensor_aggregate() -> dict[str, Any]:
    return {"count": 0, "sum": 0.0, "sum_sq": 0.0,
            "min": float("inf"), "max": float("-inf"),
            "nan_count": 0, "inf_count": 0}


def _update_tensor_aggregate(agg: dict[str, Any], arr: np.ndarray) -> None:
    finite = arr[np.isfinite(arr)]
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    agg["nan_count"] += nan_count
    agg["inf_count"] += inf_count
    if finite.size == 0:
        return
    agg["count"] += int(finite.size)
    agg["sum"] += float(finite.sum())
    agg["sum_sq"] += float((finite * finite).sum())
    agg["min"] = min(float(agg["min"]), float(finite.min()))
    agg["max"] = max(float(agg["max"]), float(finite.max()))


def _finalize_tensor_aggregate(agg: dict[str, Any]) -> dict[str, Any]:
    count = int(agg["count"])
    if count == 0:
        return {"min": None, "max": None, "mean": None, "std": None,
                "nan_count": int(agg["nan_count"]), "inf_count": int(agg["inf_count"])}
    mean = agg["sum"] / count
    var = (agg["sum_sq"] - (agg["sum"] * agg["sum"] / count)) / count if count > 1 else 0.0
    std = math.sqrt(max(0.0, var))
    return {"min": float(agg["min"]), "max": float(agg["max"]),
            "mean": float(mean), "std": float(std),
            "nan_count": int(agg["nan_count"]), "inf_count": int(agg["inf_count"])}


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.no_save_pred_masks:
        args.save_pred_masks = False

    output_dir = args.output_dir
    guardrail_ok, guardrail_msg = validate_output_guardrail(output_dir)
    if not guardrail_ok:
        output_dir.mkdir(parents=True, exist_ok=True)
        blocked = {
            "schema_version": 1,
            "audit_name": "hybrid_heldout_eval",
            "status": "BLOCKED",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "failures": [guardrail_msg],
            "exit_code": 2,
        }
        (output_dir / "hybrid_heldout_summary.json").write_text(
            json.dumps(blocked, indent=2), encoding="utf-8",
        )
        return blocked

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_masks_dir = output_dir / "pred_masks"
    prob_maps_dir = output_dir / "prob_maps"
    logit_maps_dir = output_dir / "logit_maps"
    visuals_dir = output_dir / "visuals"

    failures = _validate_inputs(args)
    if failures:
        blocked = {
            "schema_version": 1,
            "audit_name": "hybrid_heldout_eval",
            "status": "BLOCKED",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "failures": failures,
            "exit_code": 2,
            "setup": {
                "images_dir": str(args.images_dir),
                "labels_dir": str(args.labels_dir),
                "hybrid_checkpoint": str(args.hybrid_checkpoint),
                "foundation_checkpoint": str(args.foundation_checkpoint),
                "output_dir": str(args.output_dir),
                "threshold": float(args.threshold),
                "img_size": int(args.img_size),
                "device": args.device,
            },
            "outputs": {
                "summary_json": str(output_dir / "hybrid_heldout_summary.json"),
                "summary_yaml": str(output_dir / "hybrid_heldout_summary.yaml"),
                "report_md": str(output_dir / "hybrid_heldout_report.md"),
                "case_csv": str(output_dir / "hybrid_heldout_case_metrics.csv"),
                "pred_masks_dir": str(pred_masks_dir),
                "pred_masks_saved": False,
                "visuals_dir": str(visuals_dir),
                "visual_grids_written": 0,
            },
        }
        (output_dir / "hybrid_heldout_summary.json").write_text(
            json.dumps(blocked, indent=2), encoding="utf-8",
        )
        return blocked

    device = _resolve_device(args.device)

    cases = list_heldout_cases(args.images_dir, args.labels_dir)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        failures.append("No heldout images found in images_dir.")
        blocked = {
            "schema_version": 1,
            "audit_name": "hybrid_heldout_eval",
            "status": "BLOCKED",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "failures": failures,
            "exit_code": 2,
            "setup": {
                "images_dir": str(args.images_dir),
                "labels_dir": str(args.labels_dir),
                "threshold": float(args.threshold),
                "img_size": int(args.img_size),
                "device": str(device),
            },
        }
        (output_dir / "hybrid_heldout_summary.json").write_text(
            json.dumps(blocked, indent=2), encoding="utf-8",
        )
        return blocked

    model, hook_state = _instantiate_model_and_hook(
        foundation_checkpoint=args.foundation_checkpoint,
        hybrid_checkpoint=args.hybrid_checkpoint,
        img_size=int(args.img_size),
        device=device,
    )

    prob_aggregate = _new_tensor_aggregate()
    logits_aggregate = _new_tensor_aggregate()

    case_rows: list[dict[str, Any]] = []
    visuals_cache: dict[str, dict[str, Any]] = {}
    cases_visuals_kept = 0
    visuals_keep_max = max(1, args.num_visuals_per_grid) * 8  # rough upper bound to cover all grids

    pred_masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_prob_maps:
        prob_maps_dir.mkdir(parents=True, exist_ok=True)
    if args.save_logit_maps:
        logit_maps_dir.mkdir(parents=True, exist_ok=True)

    processed_cases = 0
    missing_predictions = 0
    shape_mismatch_cases = 0
    prob_nan_total = 0
    prob_inf_total = 0
    logits_nan_total = 0
    logits_inf_total = 0

    threshold = float(args.threshold)

    try:
        with torch.no_grad():
            for case in cases:
                case_id = case["case_id"]
                image_path = case["image_path"]
                label_path = case["label_path"]
                label_available = bool(case["label_exists"])

                try:
                    image_arr, mask_arr, _ = _load_image_label(
                        image_path=image_path,
                        label_path=label_path,
                        img_size=int(args.img_size),
                    )
                except Exception as exc:
                    failures.append(f"load failed for {case_id}: {exc}")
                    continue

                image_tensor = (
                    torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to(device)
                )
                hook_state["raw_logits"] = None
                t0 = time.time()
                probs = model(image_tensor)
                raw_logits = hook_state["raw_logits"]
                latency_ms = (time.time() - t0) * 1000.0

                if raw_logits is None:
                    failures.append(f"raw logits hook capture is None for case {case_id}")
                    missing_predictions += 1
                    continue

                if list(probs.shape) != [1, 1, args.img_size, args.img_size]:
                    failures.append(f"shape mismatch for {case_id}: {list(probs.shape)}")
                    shape_mismatch_cases += 1
                    continue

                prob_arr = probs[0, 0].detach().float().cpu().numpy()
                logits_arr = raw_logits[0, 0].detach().float().cpu().numpy()

                prob_nan = int(np.isnan(prob_arr).sum())
                prob_inf = int(np.isinf(prob_arr).sum())
                logits_nan = int(np.isnan(logits_arr).sum())
                logits_inf = int(np.isinf(logits_arr).sum())
                prob_nan_total += prob_nan
                prob_inf_total += prob_inf
                logits_nan_total += logits_nan
                logits_inf_total += logits_inf

                if prob_nan or prob_inf or logits_nan or logits_inf:
                    failures.append(
                        f"non-finite values for {case_id}: "
                        f"prob_nan={prob_nan} prob_inf={prob_inf} "
                        f"logits_nan={logits_nan} logits_inf={logits_inf}",
                    )

                _update_tensor_aggregate(prob_aggregate, prob_arr)
                _update_tensor_aggregate(logits_aggregate, logits_arr)

                gt_binary = (mask_arr > 0.5).astype(np.float32) if label_available else None
                pred_binary = (prob_arr >= threshold).astype(np.float32)

                if label_available:
                    metrics = per_case_metrics(pred_binary, gt_binary)
                else:
                    metrics = {
                        "dice": float("nan"),
                        "iou": float("nan"),
                        "precision": float("nan"),
                        "recall": float("nan"),
                        "intersection_pixels": 0,
                        "pred_foreground_pixels": int(pred_binary.sum()),
                        "gt_foreground_pixels": 0,
                        "total_pixels": int(pred_binary.size),
                    }

                pred_is_positive = bool(metrics["pred_foreground_pixels"] > 0)
                gt_is_positive = bool(metrics["gt_foreground_pixels"] > 0) if label_available else False

                pred_mask_path = ""
                if args.save_pred_masks:
                    pred_mask_path = str(pred_masks_dir / f"{case_id}.png")
                    _save_uint8_png((pred_binary.astype(np.uint8) * 255), Path(pred_mask_path))

                if args.save_prob_maps:
                    _save_uint8_png(_normalize_to_uint8(prob_arr), prob_maps_dir / f"{case_id}.png")
                if args.save_logit_maps:
                    _save_uint8_png(_normalize_to_uint8(logits_arr), logit_maps_dir / f"{case_id}.png")

                row = {
                    "case_id": case_id,
                    "label_available": bool(label_available),
                    "gt_is_positive": gt_is_positive,
                    "pred_is_positive": pred_is_positive,
                    "gt_foreground_pixels": int(metrics["gt_foreground_pixels"]),
                    "pred_foreground_pixels": int(metrics["pred_foreground_pixels"]),
                    "total_pixels": int(metrics["total_pixels"]),
                    "intersection_pixels": int(metrics["intersection_pixels"]),
                    "dice": float(metrics["dice"]),
                    "iou": float(metrics["iou"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "prob_min": float(prob_arr.min()),
                    "prob_max": float(prob_arr.max()),
                    "prob_mean": float(prob_arr.mean()),
                    "prob_std": float(prob_arr.std()),
                    "logits_min": float(logits_arr.min()),
                    "logits_max": float(logits_arr.max()),
                    "logits_mean": float(logits_arr.mean()),
                    "logits_std": float(logits_arr.std()),
                    "latency_ms": float(latency_ms),
                    "image_path": str(image_path),
                    "label_path": str(label_path) if label_available else "",
                    "pred_mask_path": pred_mask_path,
                }
                case_rows.append(row)
                processed_cases += 1

                if cases_visuals_kept < visuals_keep_max:
                    image_uint8 = (np.clip(image_arr * 255.0, 0.0, 255.0)).astype(np.uint8)
                    gt_uint8 = ((gt_binary > 0.5).astype(np.uint8) * 255) if label_available else np.zeros_like(image_uint8)
                    pred_uint8 = (pred_binary.astype(np.uint8) * 255)
                    visuals_cache[case_id] = {
                        "image_uint8": image_uint8,
                        "gt_uint8": gt_uint8,
                        "pred_uint8": pred_uint8,
                    }
                    cases_visuals_kept += 1
    finally:
        handle = hook_state.get("handle")
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass

    _write_case_csv(case_rows, output_dir / "hybrid_heldout_case_metrics.csv")

    agg = aggregate_case_metrics(case_rows) if case_rows else {}

    # Failure-analysis subset CSVs
    with_label = [r for r in case_rows if r.get("label_available")]
    pos = [r for r in with_label if r["gt_is_positive"]]
    neg = [r for r in with_label if not r["gt_is_positive"]]
    largest_fp = sorted([r for r in neg if r["pred_is_positive"]],
                         key=lambda r: r["pred_foreground_pixels"], reverse=True)
    largest_fn = sorted([r for r in pos if not r["pred_is_positive"]],
                         key=lambda r: r["gt_foreground_pixels"], reverse=True)
    worst_pos = sorted(pos, key=lambda r: r["dice"])
    best_pos = sorted(pos, key=lambda r: r["dice"], reverse=True)

    top_n = max(1, args.num_visuals_per_grid) * 2
    _write_subset_csv(largest_fp[:top_n], output_dir / "largest_false_positive_cases.csv")
    _write_subset_csv(largest_fn[:top_n], output_dir / "largest_false_negative_cases.csv")
    _write_subset_csv(worst_pos[:top_n], output_dir / "worst_positive_cases.csv")
    _write_subset_csv(best_pos[:top_n], output_dir / "best_positive_cases.csv")

    # Visual grids — use cached samples where possible. Fall back to re-loading
    # for cases not in cache.
    def _ensure_visual(row: dict[str, Any]) -> dict[str, Any] | None:
        case_id = row["case_id"]
        if case_id in visuals_cache:
            return visuals_cache[case_id]
        case = next((c for c in cases if c["case_id"] == case_id), None)
        if case is None:
            return None
        try:
            image_arr, mask_arr, label_available = _load_image_label(
                case["image_path"], case["label_path"], int(args.img_size),
            )
        except Exception:
            return None
        image_uint8 = (np.clip(image_arr * 255.0, 0.0, 255.0)).astype(np.uint8)
        gt_uint8 = ((mask_arr > 0.5).astype(np.uint8) * 255) if label_available else np.zeros_like(image_uint8)
        pred_mask_path = Path(row.get("pred_mask_path", ""))
        if pred_mask_path.exists():
            from PIL import Image
            pred_pil = Image.open(pred_mask_path).convert("L")
            if pred_pil.size != (args.img_size, args.img_size):
                pred_pil = pred_pil.resize((args.img_size, args.img_size), Image.NEAREST)
            pred_uint8 = (np.array(pred_pil, dtype=np.uint8) > 127).astype(np.uint8) * 255
        else:
            pred_uint8 = np.zeros_like(image_uint8)
        cached = {"image_uint8": image_uint8, "gt_uint8": gt_uint8, "pred_uint8": pred_uint8}
        visuals_cache[case_id] = cached
        return cached

    visuals_dir.mkdir(parents=True, exist_ok=True)

    n_per = max(1, args.num_visuals_per_grid)
    grid_sources = {
        "grid_largest_false_positives.png": largest_fp[:n_per],
        "grid_largest_false_negatives.png": largest_fn[:n_per],
        "grid_worst_positive_dice.png": worst_pos[:n_per],
        "grid_best_positive_dice.png": best_pos[:n_per],
    }
    pos_detected = [r for r in pos if r["pred_is_positive"]]
    if pos_detected:
        sorted_detected = sorted(pos_detected, key=lambda r: r["dice"])
        median_idx = len(sorted_detected) // 2
        start = max(0, median_idx - n_per // 2)
        grid_sources["grid_representative_true_positives.png"] = sorted_detected[start: start + n_per]
    else:
        grid_sources["grid_representative_true_positives.png"] = []
    grid_sources["grid_representative_true_negatives.png"] = [r for r in neg if not r["pred_is_positive"]][:n_per]

    visual_grids_written = 0
    for grid_name, source_rows in grid_sources.items():
        samples: list[dict[str, Any]] = []
        for row in source_rows:
            cached = _ensure_visual(row)
            if cached is None:
                continue
            overlay = _build_overlay_uint8(
                cached["image_uint8"], cached["gt_uint8"], cached["pred_uint8"],
            )
            samples.append({
                "case_id": row["case_id"],
                "dice": float(row.get("dice", float("nan"))) if row.get("dice") is not None else float("nan"),
                "pred_foreground_pixels": int(row.get("pred_foreground_pixels", 0)),
                "gt_foreground_pixels": int(row.get("gt_foreground_pixels", 0)),
                "overlay": overlay,
            })
        title = grid_name.replace(".png", "").replace("_", " ").title()
        ok = _render_grid(title, samples, visuals_dir / grid_name, int(args.img_size))
        if ok:
            visual_grids_written += 1

    tiny_subgroup = _compute_tiny_subgroup(case_rows)

    pass_criteria = {
        "checkpoint_loads": True,
        "foundation_loads": True,
        "no_forbidden_writes": guardrail_ok,
        "case_csv_written": (output_dir / "hybrid_heldout_case_metrics.csv").exists(),
        "summary_written": True,
        "report_written": True,
        "all_six_visual_grids_written": visual_grids_written == 6,
        "all_cases_processed": processed_cases == len(cases),
        "no_missing_predictions": missing_predictions == 0,
        "no_shape_mismatches": shape_mismatch_cases == 0,
        "outputs_finite": (
            prob_nan_total == 0 and prob_inf_total == 0
            and logits_nan_total == 0 and logits_inf_total == 0
        ),
        "no_state_dict_drift": len(hook_state.get("missing_keys", [])) == 0,
    }

    hard_blockers = [
        "no_forbidden_writes", "case_csv_written", "summary_written",
        "all_cases_processed", "no_missing_predictions",
        "no_shape_mismatches", "outputs_finite",
    ]
    has_blocker = any(not pass_criteria[k] for k in hard_blockers) or bool(failures)
    if has_blocker:
        status = "BLOCKED"
        exit_code = 2
    else:
        # PARTIAL_PASS if performance is weak vs baseline; PASS otherwise.
        mdpc = agg.get("mean_dice_positive_cases")
        nfpr = agg.get("negative_case_false_positive_rate")
        partial = False
        if isinstance(mdpc, float) and not (math.isnan(mdpc) or math.isinf(mdpc)):
            if mdpc < NNUNET_BASELINE["mean_dice_positive_cases"]:
                partial = True
        if isinstance(nfpr, float) and not (math.isnan(nfpr) or math.isinf(nfpr)):
            if nfpr > NNUNET_BASELINE["negative_case_false_positive_rate"]:
                partial = True
        status = "PARTIAL_PASS" if partial else "PASS"
        exit_code = 0
    if args.strict and failures:
        status = "BLOCKED"
        exit_code = 2

    summary = {
        "schema_version": 1,
        "audit_name": "hybrid_heldout_eval",
        "status": status,
        "exit_code": exit_code,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "setup": {
            "images_dir": str(args.images_dir),
            "labels_dir": str(args.labels_dir),
            "hybrid_checkpoint": str(args.hybrid_checkpoint),
            "foundation_checkpoint": str(args.foundation_checkpoint),
            "output_dir": str(args.output_dir),
            "threshold": float(args.threshold),
            "img_size": int(args.img_size),
            "device": str(device),
            "save_pred_masks": bool(args.save_pred_masks),
            "save_prob_maps": bool(args.save_prob_maps),
            "save_logit_maps": bool(args.save_logit_maps),
            "num_visuals_per_grid": int(args.num_visuals_per_grid),
            "max_cases": int(args.max_cases),
            "seed": int(args.seed),
            "strict": bool(args.strict),
        },
        "inference": {
            "processed_cases": int(processed_cases),
            "missing_predictions": int(missing_predictions),
            "shape_mismatch_cases": int(shape_mismatch_cases),
            "prob_nan_total": int(prob_nan_total),
            "prob_inf_total": int(prob_inf_total),
            "logits_nan_total": int(logits_nan_total),
            "logits_inf_total": int(logits_inf_total),
            "prob_aggregate": _finalize_tensor_aggregate(prob_aggregate),
            "logits_aggregate": _finalize_tensor_aggregate(logits_aggregate),
        },
        "aggregate_metrics": agg,
        "tiny_positive_subgroup": tiny_subgroup,
        "checkpoint_meta": hook_state.get("diagnostic_payload_meta", {}),
        "state_dict": {
            "missing_keys_count": len(hook_state.get("missing_keys", [])),
            "unexpected_keys_count": len(hook_state.get("unexpected_keys", [])),
        },
        "comparison_baseline": NNUNET_BASELINE,
        "pass_criteria": pass_criteria,
        "outputs": {
            "summary_json": str(output_dir / "hybrid_heldout_summary.json"),
            "summary_yaml": str(output_dir / "hybrid_heldout_summary.yaml"),
            "report_md": str(output_dir / "hybrid_heldout_report.md"),
            "case_csv": str(output_dir / "hybrid_heldout_case_metrics.csv"),
            "largest_false_positive_cases_csv": str(output_dir / "largest_false_positive_cases.csv"),
            "largest_false_negative_cases_csv": str(output_dir / "largest_false_negative_cases.csv"),
            "worst_positive_cases_csv": str(output_dir / "worst_positive_cases.csv"),
            "best_positive_cases_csv": str(output_dir / "best_positive_cases.csv"),
            "pred_masks_dir": str(pred_masks_dir),
            "pred_masks_saved": bool(args.save_pred_masks),
            "prob_maps_dir": str(prob_maps_dir),
            "logit_maps_dir": str(logit_maps_dir),
            "visuals_dir": str(visuals_dir),
            "visual_grids_written": int(visual_grids_written),
        },
        "failures": failures,
    }

    safe = _sanitize_for_json(summary)
    (output_dir / "hybrid_heldout_summary.json").write_text(
        json.dumps(safe, indent=2), encoding="utf-8",
    )
    if _HAS_YAML:
        with (output_dir / "hybrid_heldout_summary.yaml").open("w", encoding="utf-8") as h:
            _yaml.dump(safe, h, default_flow_style=False, allow_unicode=True)
    (output_dir / "hybrid_heldout_report.md").write_text(
        _build_report_markdown(args, safe), encoding="utf-8",
    )
    return safe


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_evaluation(args)
    return int(summary.get("exit_code", 2))


if __name__ == "__main__":
    sys.exit(main())
