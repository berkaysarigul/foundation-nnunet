"""Export Foundation X probability priors from a generic image/mask manifest."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pr11_utils import (
    REPO_ROOT,
    load_grayscale,
    parse_bool,
    read_csv,
    repo_relative,
    resolve_existing_path,
    save_uint8,
    write_csv,
    write_markdown_report,
    write_yaml,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.foundation_x_official_preprocess import preprocess_image_for_foundation_x
from src.models.foundation_x_official import OfficialFoundationXSegmentationModel, SegmentationHeadSpec


MANIFEST_COLUMNS = [
    "case_id",
    "split",
    "image_path",
    "label_path",
    "probability_map_path",
    "aligned_probability_map_path",
    "binary_mask_path",
    "visual_overlay_path",
    "state_key",
    "preprocess_variant",
    "head_key",
    "threshold",
    "gt_is_positive",
    "pred_is_positive",
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "gt_foreground_pixels",
    "pred_foreground_pixels",
    "prob_width",
    "prob_height",
    "aligned_width",
    "aligned_height",
    "prob_min",
    "prob_max",
    "prob_mean",
    "prob_std",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Foundation X priors from a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-col", default="image_path")
    parser.add_argument("--label-col", default="mask_path")
    parser.add_argument("--case-id-col", default="case_id")
    parser.add_argument("--split-col", default="source_split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-key", default="teacher_model")
    parser.add_argument("--head", type=int, default=5)
    parser.add_argument("--preprocess-variant", default="official_siim_224")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-prob-maps", type=parse_bool, default=True)
    parser.add_argument("--save-aligned-prob-maps", type=parse_bool, default=True)
    parser.add_argument("--save-binary-masks", type=parse_bool, default=False)
    parser.add_argument("--save-visuals", type=parse_bool, default=False)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _validate_out(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
        resolved = out.resolve()
        try:
            resolved.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"Refusing to remove output outside repo: {out}") from exc
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def _prob_to_uint8(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob, 0.0, 1.0).astype(np.float32) * 255.0


def _load_label_resized(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        return (np.asarray(image) > 0).astype(np.uint8)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    def div(num: float, den: float) -> float:
        if den == 0:
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
        "gt_is_positive": bool(gt_sum > 0),
        "pred_is_positive": bool(pred_sum > 0),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "gt_foreground_pixels": gt_sum,
        "pred_foreground_pixels": pred_sum,
    }


def _save_overlay(image_path: Path, gt: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    with Image.open(image_path) as image:
        image = image.convert("L").resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR)
        base = np.asarray(image, dtype=np.uint8)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    rgb[gt > 0, 1] = 255
    rgb[gt > 0, 0] *= 0.35
    rgb[pred > 0, 0] = 255
    rgb[pred > 0, 1] *= 0.35
    rgb[(gt > 0) & (pred > 0)] = np.array([255, 220, 0], dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out_path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    _validate_out(args.out, bool(args.overwrite))
    rows = read_csv(args.manifest)
    if args.max_cases > 0:
        rows = rows[: int(args.max_cases)]
    device = _resolve_device(args.device)
    spec = SegmentationHeadSpec(int(args.head))
    model = OfficialFoundationXSegmentationModel(device=device)
    load_diagnostics = model.load_checkpoint(args.checkpoint, args.state_key)

    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    head_dir = spec.key
    for row in tqdm(rows, desc="export_fx_priors"):
        case_id = str(row.get(args.case_id_col, "")).strip()
        split = str(row.get(args.split_col, "custom") or "custom")
        try:
            image_path = resolve_existing_path(row[args.image_col], REPO_ROOT)
            label_raw = row.get(args.label_col, "")
            label_path = resolve_existing_path(label_raw, REPO_ROOT) if label_raw else Path("")
            prep = preprocess_image_for_foundation_x(image_path, args.preprocess_variant)
            _, prob_tensor = model.predict_probability(prep.tensor.to(device), head_idx=spec.head_idx, channel=spec.channel)
            prob = prob_tensor.detach().squeeze().float().cpu().numpy().astype(np.float32)
            if prob.ndim != 2:
                raise ValueError(f"Expected 2D probability map, got shape {prob.shape}")

            prob_uint8 = _prob_to_uint8(prob).astype(np.uint8)
            prob_path = (
                args.out
                / "probability_maps"
                / args.state_key
                / args.preprocess_variant
                / head_dir
                / f"{case_id}.png"
            )
            aligned_path = (
                args.out
                / "probability_maps_aligned"
                / args.state_key
                / args.preprocess_variant
                / head_dir
                / f"{case_id}.png"
            )
            binary_path = args.out / "binary_masks" / args.state_key / args.preprocess_variant / head_dir / f"{case_id}.png"
            visual_path = args.out / "visual_overlays" / args.state_key / args.preprocess_variant / head_dir / f"{case_id}.png"

            probability_map_path = ""
            aligned_probability_map_path = ""
            binary_mask_path = ""
            visual_overlay_path = ""
            if args.save_prob_maps:
                save_uint8(prob_uint8, prob_path)
                probability_map_path = repo_relative(prob_path)

            with Image.open(image_path) as image:
                target_size = image.size
            aligned_uint8 = np.asarray(
                Image.fromarray(prob_uint8).resize(target_size, Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
            if args.save_aligned_prob_maps:
                save_uint8(aligned_uint8, aligned_path)
                aligned_probability_map_path = repo_relative(aligned_path)

            pred = (prob >= float(args.threshold)).astype(np.uint8)
            metric_values = {
                "gt_is_positive": "",
                "pred_is_positive": bool(pred.sum() > 0),
                "dice": math.nan,
                "iou": math.nan,
                "precision": math.nan,
                "recall": math.nan,
                "specificity": math.nan,
                "gt_foreground_pixels": "",
                "pred_foreground_pixels": int(pred.sum()),
            }
            gt = np.zeros_like(pred, dtype=np.uint8)
            if label_path and label_path.is_file():
                gt = _load_label_resized(label_path, size=(prob.shape[1], prob.shape[0]))
                metric_values = _metrics(pred, gt)

            if args.save_binary_masks:
                save_uint8((pred * 255).astype(np.uint8), binary_path)
                binary_mask_path = repo_relative(binary_path)
            if args.save_visuals and label_path and label_path.is_file():
                _save_overlay(image_path, gt, pred, visual_path)
                visual_overlay_path = repo_relative(visual_path)

            manifest_rows.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "image_path": repo_relative(image_path),
                    "label_path": repo_relative(label_path) if label_path and label_path.is_file() else "",
                    "probability_map_path": probability_map_path,
                    "aligned_probability_map_path": aligned_probability_map_path,
                    "binary_mask_path": binary_mask_path,
                    "visual_overlay_path": visual_overlay_path,
                    "state_key": args.state_key,
                    "preprocess_variant": args.preprocess_variant,
                    "head_key": spec.key,
                    "threshold": float(args.threshold),
                    "prob_width": int(prob.shape[1]),
                    "prob_height": int(prob.shape[0]),
                    "aligned_width": int(target_size[0]),
                    "aligned_height": int(target_size[1]),
                    "prob_min": float(prob.min()),
                    "prob_max": float(prob.max()),
                    "prob_mean": float(prob.mean()),
                    "prob_std": float(prob.std()),
                    **metric_values,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=5),
                }
            )

    positives = [r for r in manifest_rows if str(r.get("gt_is_positive", "")).lower() == "true"]
    summary = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "state_key": args.state_key,
        "preprocess_variant": args.preprocess_variant,
        "head_key": spec.key,
        "threshold": float(args.threshold),
        "rows_requested": len(rows),
        "rows_exported": len(manifest_rows),
        "failures": len(failures),
        "positive_dice": float(np.mean([float(r["dice"]) for r in positives])) if positives else math.nan,
        "save_prob_maps": bool(args.save_prob_maps),
        "save_aligned_prob_maps": bool(args.save_aligned_prob_maps),
        "save_binary_masks": bool(args.save_binary_masks),
        "save_visuals": bool(args.save_visuals),
        "load_diagnostics": load_diagnostics,
        "status": "PASS" if not failures else "FAIL",
    }
    write_csv(args.out / "fx_prior_manifest.csv", manifest_rows, MANIFEST_COLUMNS)
    write_csv(args.out / "per_case_metrics.csv", manifest_rows)
    write_csv(args.out / "failures.csv", failures)
    write_yaml(args.out / "summary.yaml", summary)
    write_report(args.out / "report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    body = [
        f"- status: `{summary['status']}`",
        f"- rows requested: `{summary['rows_requested']}`",
        f"- rows exported: `{summary['rows_exported']}`",
        f"- failures: `{summary['failures']}`",
        f"- state key: `{summary['state_key']}`",
        f"- preprocess variant: `{summary['preprocess_variant']}`",
        f"- head key: `{summary['head_key']}`",
        f"- positive Dice: `{summary['positive_dice']}`",
    ]
    write_markdown_report(path, "PR-11 Foundation X Prior Export Report", [("Summary", body)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = export(args)
    print(f"[{summary['status']}] Foundation X prior export written under {args.out}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
