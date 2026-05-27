"""Evaluate binary segmentation prediction folders for PR-11 reports."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from pr11_utils import (
    REPO_ROOT,
    percentile_summary,
    read_csv,
    resolve_existing_path,
    write_csv,
    write_markdown_report,
    write_yaml,
)


DEFAULT_THRESHOLDS = [round(v, 2) for v in np.arange(0.05, 1.00, 0.05)]
DEFAULT_MIN_PIXELS = [0, 16, 64, 256, 1024, 4096]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--case-id-col", default="case_id")
    parser.add_argument("--label-col", default="mask_path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--component-min-pixels", default=",".join(str(v) for v in DEFAULT_MIN_PIXELS))
    parser.add_argument("--selected-threshold", type=float, default=0.5)
    parser.add_argument("--selected-min-pixels", type=int, default=0)
    parser.add_argument("--size-bin-source-manifest", type=Path, default=None)
    return parser.parse_args(argv)


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _load_prob(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    return arr / 255.0 if arr.max(initial=0.0) > 1.0 else arr


def _load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        return (np.asarray(image) > 0).astype(np.uint8)


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 0 or not mask.any():
        return mask.astype(np.uint8)
    labeled, count = ndimage.label(mask > 0)
    keep = np.zeros_like(mask, dtype=bool)
    for label_idx in range(1, count + 1):
        component = labeled == label_idx
        if int(component.sum()) >= int(min_pixels):
            keep |= component
    return keep.astype(np.uint8)


def _case_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    def div(num: float, den: float) -> float:
        if den <= 0:
            return 1.0 if num == 0 else 0.0
        return float(num / den)

    if pred_sum == 0 and gt_sum == 0:
        dice = iou = precision = recall = 1.0
    else:
        dice = div(2 * tp, pred_sum + gt_sum)
        iou = div(tp, tp + fp + fn)
        precision = div(tp, tp + fp)
        recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    return {
        "gt_is_positive": gt_sum > 0,
        "pred_is_positive": pred_sum > 0,
        "gt_foreground_pixels": gt_sum,
        "pred_foreground_pixels": pred_sum,
        "intersection_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def aggregate(rows: list[dict[str, Any]], bins: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    gt_pos = [r for r in rows if r["gt_is_positive"]]
    gt_neg = [r for r in rows if not r["gt_is_positive"]]
    detected = [r for r in gt_pos if r["pred_is_positive"]]
    tp = sum(1 for r in rows if r["gt_is_positive"] and r["pred_is_positive"])
    fn = sum(1 for r in rows if r["gt_is_positive"] and not r["pred_is_positive"])
    fp = sum(1 for r in rows if (not r["gt_is_positive"]) and r["pred_is_positive"])
    tn = sum(1 for r in rows if (not r["gt_is_positive"]) and not r["pred_is_positive"])
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    payload = {
        "total_cases": len(rows),
        "gt_positive_cases": len(gt_pos),
        "gt_negative_cases": len(gt_neg),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "case_level_precision": precision,
        "case_level_recall": recall,
        "case_level_specificity": specificity,
        "case_level_f1": f1,
        "negative_case_false_positive_rate": _safe_div(fp, fp + tn),
        "mean_dice_all_cases": _mean([float(r["dice"]) for r in rows]),
        "positive_dice": _mean([float(r["dice"]) for r in gt_pos]),
        "detected_positive_dice": _mean([float(r["dice"]) for r in detected]),
        "mean_iou_all_cases": _mean([float(r["iou"]) for r in rows]),
        "mean_precision_all_cases": _mean([float(r["precision"]) for r in rows]),
        "mean_recall_all_cases": _mean([float(r["recall"]) for r in rows]),
        "mean_specificity_all_cases": _mean([float(r["specificity"]) for r in rows]),
    }
    if bins:
        for name, (lo, hi) in bins.items():
            subset = [r for r in gt_pos if lo <= float(r["gt_foreground_pixels"]) <= hi]
            payload[f"{name}_positive_dice"] = _mean([float(r["dice"]) for r in subset])
            payload[f"{name}_positive_cases"] = len(subset)
    return payload


def _cases_from_manifest(path: Path, case_col: str, label_col: str) -> list[dict[str, str]]:
    rows = read_csv(path)
    return [
        {"case_id": row[case_col], "label_path": row[label_col]}
        for row in rows
        if row.get(case_col) and row.get(label_col)
    ]


def _cases_from_label_dir(label_dir: Path) -> list[dict[str, str]]:
    return [{"case_id": p.stem, "label_path": str(p)} for p in sorted(label_dir.glob("*.png"))]


def _pred_path(pred_dir: Path, case_id: str) -> Path | None:
    candidates = [pred_dir / f"{case_id}.png", pred_dir / f"{case_id}_0000.png"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(pred_dir.glob(f"{case_id}*.png"))
    return matches[0] if matches else None


def _size_bins(source_manifest: Path | None, eval_cases: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    if source_manifest is not None:
        rows = read_csv(source_manifest)
        values = [
            int(float(r.get("gt_foreground_pixels", "0") or 0))
            for r in rows
            if int(float(r.get("gt_foreground_pixels", "0") or 0)) > 0
        ]
    else:
        values = []
        for case in eval_cases:
            label = resolve_existing_path(case["label_path"], REPO_ROOT)
            if label.is_file():
                values.append(int(_load_mask(label).sum()))
        values = [v for v in values if v > 0]
    if len(values) < 3:
        return {}
    q1, q2 = np.percentile(np.asarray(values, dtype=np.float64), [33.3333, 66.6667])
    return {
        "small": (1, float(q1)),
        "medium": (float(q1) + 1e-6, float(q2)),
        "large": (float(q2) + 1e-6, float("inf")),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest is not None:
        cases = _cases_from_manifest(args.manifest, args.case_id_col, args.label_col)
    elif args.label_dir is not None:
        cases = _cases_from_label_dir(args.label_dir)
    else:
        raise ValueError("Either --manifest or --label-dir is required.")
    thresholds = _parse_float_list(args.thresholds)
    min_pixels_values = _parse_int_list(args.component_min_pixels)
    bins = _size_bins(args.size_bin_source_manifest, cases)

    loaded: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for case in cases:
        pred_path = _pred_path(args.pred_dir, case["case_id"])
        label_path = resolve_existing_path(case["label_path"], REPO_ROOT)
        if pred_path is None or not label_path.is_file():
            missing.append({"case_id": case["case_id"], "missing_prediction": str(pred_path is None), "label_path": str(label_path)})
            continue
        prob = _load_prob(pred_path)
        gt = _load_mask(label_path, size=(prob.shape[1], prob.shape[0]))
        loaded.append({"case_id": case["case_id"], "prob": prob, "gt": gt, "pred_path": str(pred_path), "label_path": str(label_path)})

    sweep_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for min_pixels in min_pixels_values:
            per_case = []
            for item in loaded:
                pred = _remove_small_components(item["prob"] >= threshold, min_pixels)
                metrics = _case_metrics(pred, item["gt"])
                per_case.append({"case_id": item["case_id"], **metrics})
            agg = aggregate(per_case, bins)
            sweep_rows.append({"threshold": threshold, "min_component_pixels": min_pixels, **agg})
            if abs(threshold - float(args.selected_threshold)) < 1e-9 and min_pixels == int(args.selected_min_pixels):
                selected_rows = [
                    {
                        "case_id": item["case_id"],
                        "prediction_path": item["pred_path"],
                        "label_path": item["label_path"],
                        "threshold": threshold,
                        "min_component_pixels": min_pixels,
                        **_case_metrics(_remove_small_components(item["prob"] >= threshold, min_pixels), item["gt"]),
                    }
                    for item in loaded
                ]

    selected_agg = aggregate(selected_rows, bins) if selected_rows else {}
    summary = {
        "schema_version": 1,
        "prediction_dir": str(args.pred_dir),
        "cases_requested": len(cases),
        "cases_evaluated": len(loaded),
        "missing_cases": len(missing),
        "thresholds": thresholds,
        "component_min_pixels": min_pixels_values,
        "selected_threshold": float(args.selected_threshold),
        "selected_min_component_pixels": int(args.selected_min_pixels),
        "lesion_size_bins": bins,
        "selected_metrics": selected_agg,
        "foreground_pixel_distribution_evaluated": percentile_summary(
            [int(row["gt_foreground_pixels"]) for row in selected_rows if row["gt_is_positive"]]
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "threshold_sweep.csv", sweep_rows)
    write_csv(args.out / "per_case_metrics.csv", selected_rows)
    write_csv(args.out / "missing_cases.csv", missing)
    write_yaml(args.out / "summary.yaml", summary)
    write_report(args.out / "report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("selected_metrics", {})
    body = [
        f"- cases requested: `{summary['cases_requested']}`",
        f"- cases evaluated: `{summary['cases_evaluated']}`",
        f"- missing cases: `{summary['missing_cases']}`",
        f"- selected threshold: `{summary['selected_threshold']}`",
        f"- selected min component pixels: `{summary['selected_min_component_pixels']}`",
        f"- positive Dice: `{metrics.get('positive_dice')}`",
        f"- detected-positive Dice: `{metrics.get('detected_positive_dice')}`",
        f"- all-case Dice: `{metrics.get('mean_dice_all_cases')}`",
        f"- case-level F1: `{metrics.get('case_level_f1')}`",
        f"- negative FPR: `{metrics.get('negative_case_false_positive_rate')}`",
    ]
    write_markdown_report(path, "PR-11 Segmentation Evaluation Report", [("Summary", body)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = evaluate(args)
    print(f"[done] evaluated {summary['cases_evaluated']} cases under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

