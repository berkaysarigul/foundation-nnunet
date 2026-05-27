"""Evaluate a trained Foundation X prior refiner on a heldout/test manifest."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fx_prior_refiner_dataset import (  # noqa: E402
    INPUT_MODES,
    FoundationXPriorRefinerDataset,
    input_channels_for_mode,
)
from src.models.refiner_unet import FoundationXPriorRefinerUNet  # noqa: E402
from src.training.metrics import (  # noqa: E402
    aggregate_per_case_binary_metrics,
    compute_per_case_binary_metrics,
)


BASELINE_POSITIVE_DICE = 0.5235
SUCCESS_POSITIVE_DICE = 0.5335
SUCCESS_ABLATION_DELTA = 0.005


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the PR-10C Foundation X prior refiner.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-mode", choices=sorted(INPUT_MODES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-visuals", type=parse_bool, default=False)
    return parser.parse_args(argv)


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _validate_out_dir(out_dir: Path) -> Path:
    resolved = _resolve_path(out_dir).resolve()
    normalized = resolved.as_posix().lower()
    if "artifacts/refiner/runs" not in normalized:
        raise ValueError(f"--out must be under artifacts/refiner/runs; got {out_dir}")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty eval directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _metadata_at(metadata: dict[str, Any], idx: int) -> dict[str, str]:
    item: dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, (list, tuple)):
            item[key] = str(value[idx])
        else:
            item[key] = str(value)
    return item


def _overlay_image(image_path: str, gt: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    with Image.open(image_path) as image:
        image = image.convert("L").resize((gt.shape[1], gt.shape[0]), Image.BILINEAR)
        base = np.asarray(image, dtype=np.uint8)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    gt_mask = gt > 0.5
    pred_mask = pred > 0.5
    rgb[gt_mask, 1] = 255
    rgb[gt_mask, 0] *= 0.35
    rgb[pred_mask, 0] = 255
    rgb[pred_mask, 1] *= 0.35
    both = gt_mask & pred_mask
    rgb[both] = np.array([255, 220, 0], dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out_path)


def _build_report(summary: dict[str, Any]) -> str:
    metrics = summary["aggregate_metrics"]
    positive_dice = metrics.get("positive_dice", float("nan"))
    positive_text = "nan" if math.isnan(float(positive_dice)) else f"{float(positive_dice):.6f}"
    return "\n".join(
        [
            "# PR-10C Foundation X Prior Refiner Evaluation Report",
            "",
            f"- input mode: `{summary['config']['input_mode']}`",
            f"- manifest: `{summary['config']['manifest']}`",
            f"- checkpoint: `{summary['config']['checkpoint']}`",
            f"- threshold: `{summary['config']['threshold']}`",
            f"- heldout positive Dice: `{positive_text}`",
            f"- case-level F1: `{metrics.get('case_level_f1', 0.0):.6f}`",
            f"- negative FPR: `{metrics.get('negative_case_false_positive_rate', 0.0):.6f}`",
            "",
            "## Baseline Targets",
            "",
            (
                "- Foundation X direct teacher/head5 official_siim_224 baseline "
                f"positive Dice ~= {BASELINE_POSITIVE_DICE:.4f}"
            ),
            f"- Success: image_prior heldout positive Dice >= {SUCCESS_POSITIVE_DICE:.4f}",
            (
                "- Success: image_prior beats image_only by at least "
                f"{SUCCESS_ABLATION_DELTA:.3f} absolute positive Dice"
            ),
        ]
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = _validate_out_dir(args.out)
    manifest = _resolve_path(args.manifest).resolve()
    checkpoint = _resolve_path(args.checkpoint).resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("checkpoint must contain a model_state_dict.")
    ckpt_config = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
    ckpt_mode = ckpt_config.get("input_mode")
    if ckpt_mode is not None and ckpt_mode != args.input_mode:
        raise ValueError(
            f"Checkpoint input_mode={ckpt_mode!r} does not match CLI input_mode={args.input_mode!r}."
        )

    target_size = int(ckpt_config.get("target_size", 512))
    model_cfg = ckpt_config.get("model", {}) if isinstance(ckpt_config.get("model", {}), dict) else {}
    device = _resolve_device(args.device)
    dataset = FoundationXPriorRefinerDataset(
        manifest,
        input_mode=args.input_mode,
        target_size=target_size,
        augment=False,
        normalization="zero_one",
        repo_root=REPO_ROOT,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = FoundationXPriorRefinerUNet(
        in_channels=input_channels_for_mode(args.input_mode),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        norm=str(model_cfg.get("norm", "batch")),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    per_case_rows: list[dict[str, Any]] = []
    visuals_dir = out_dir / "visual_overlays"
    with torch.no_grad():
        for x, y, metadata in tqdm(loader, desc="evaluate", leave=False):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            logits = model(x)
            probs = torch.sigmoid(logits)
            batch_rows = compute_per_case_binary_metrics(
                probs.detach().cpu(),
                y.detach().cpu(),
                threshold=float(args.threshold),
            )
            pred_binary = (probs.detach().cpu().numpy() > float(args.threshold)).astype(np.float32)
            gt_binary = y.detach().cpu().numpy().astype(np.float32)
            for idx, row in enumerate(batch_rows):
                meta = _metadata_at(metadata, idx)
                case_row = {
                    "case_id": meta["case_id"],
                    "gt_is_positive": bool(row["gt_is_positive"]),
                    "pred_is_positive": bool(row["pred_is_positive"]),
                    "gt_foreground_pixels": int(row["gt_foreground_pixels"]),
                    "pred_foreground_pixels": int(row["pred_foreground_pixels"]),
                    "intersection_pixels": int(row["intersection_pixels"]),
                    "false_positive_pixels": int(row["false_positive_pixels"]),
                    "false_negative_pixels": int(row["false_negative_pixels"]),
                    "true_negative_pixels": int(row["true_negative_pixels"]),
                    "total_pixels": int(row["total_pixels"]),
                    "dice": float(row["dice"]),
                    "iou": float(row["iou"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "specificity": float(row["specificity"]),
                    "image_path": meta["image_path"],
                    "label_path": meta["label_path"],
                    "probability_map_path": meta["probability_map_path"],
                }
                per_case_rows.append(case_row)
                if args.save_visuals:
                    _overlay_image(
                        meta["image_path"],
                        gt_binary[idx, 0],
                        pred_binary[idx, 0],
                        visuals_dir / f"{meta['case_id']}.png",
                    )

    aggregate = aggregate_per_case_binary_metrics(per_case_rows)
    summary = {
        "schema_version": 1,
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config": {
            "manifest": str(manifest),
            "checkpoint": str(checkpoint),
            "input_mode": args.input_mode,
            "out": str(out_dir),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "threshold": float(args.threshold),
            "save_visuals": bool(args.save_visuals),
            "target_size": target_size,
        },
        "aggregate_metrics": aggregate,
        "baseline_targets": {
            "foundation_x_direct_teacher_head5_official_siim_224_positive_dice": BASELINE_POSITIVE_DICE,
            "image_prior_success_positive_dice": SUCCESS_POSITIVE_DICE,
            "image_prior_vs_image_only_required_delta": SUCCESS_ABLATION_DELTA,
        },
        "outputs": {
            "summary_yaml": str(out_dir / "summary.yaml"),
            "test_metrics_csv": str(out_dir / "test_metrics.csv"),
            "per_case_metrics_csv": str(out_dir / "per_case_metrics.csv"),
            "report_md": str(out_dir / "report.md"),
            "visual_overlays": str(visuals_dir) if args.save_visuals else None,
        },
    }

    _write_dict_csv(out_dir / "per_case_metrics.csv", per_case_rows)
    _write_dict_csv(out_dir / "test_metrics.csv", [aggregate])
    _save_yaml(out_dir / "summary.yaml", summary)
    (out_dir / "report.md").write_text(_build_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
