"""
PR-7F inference-only final visualization for a completed controlled
medium training run.

This script does NOT train. It only loads the diagnostic checkpoint
produced by `scripts/controlled_train_hybrid_short.py` (saved when
`--save_diagnostic_checkpoint` is set) and renders per-case visual
diagnostics for a small balanced subset of validation cases at
thresholds 0.3 / 0.4 / 0.5.

Intended runtime: Colab (where the PR-7F artifacts and checkpoint
live on Google Drive). The script does not bind to a default Drive
path; every input path must be supplied explicitly.

Example (Colab):

    py scripts/visualize_pr7f_final.py \\
      --run_folder /content/drive/MyDrive/Foundation-nnUNet/artifacts/diagnostics/hybrid_controlled_short_train/medium_lr3e5_400step_bnfix \\
      --checkpoint /content/drive/MyDrive/Foundation-nnUNet/artifacts/diagnostics/hybrid_controlled_short_train/medium_lr3e5_400step_bnfix/diagnostic_checkpoint_final_step.pth \\
      --selection_log /content/drive/MyDrive/Foundation-nnUNet/artifacts/diagnostics/hybrid_controlled_short_train/medium_lr3e5_400step_bnfix/selection_log.csv \\
      --images_dir /content/nnUNet_raw/Dataset101_Pneumothorax/imagesTr \\
      --labels_dir /content/nnUNet_raw/Dataset101_Pneumothorax/labelsTr \\
      --foundation_checkpoint checkpoints/foundation_x.pth \\
      --img_size 512 \\
      --device auto \\
      --num_positive 4 \\
      --num_negative 4 \\
      --step_label 0400

Hard constraints:
  - No training, no optimizer, no backward.
  - All forwards are inside `torch.no_grad()`.
  - Exit code 0 only if all expected visual files are written for every
    selected case.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


VISUAL_THRESHOLDS = (0.30, 0.40, 0.50)
EXPECTED_VISUAL_BASENAMES = (
    "input.png",
    "gt_mask.png",
    "probability_map.png",
    "raw_logits_map.png",
    "prediction_mask_thr03.png",
    "prediction_mask_thr04.png",
    "prediction_mask_thr05.png",
    "overlay_input_gt_pred_thr03.png",
    "overlay_input_gt_pred_thr04.png",
    "overlay_input_gt_pred_thr05.png",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inference-only PR-7F final visualization. Renders prediction "
            "overlays and probability/logit maps at thresholds 0.3/0.4/0.5 "
            "for a balanced subset of validation cases. Does not train."
        ),
    )
    parser.add_argument(
        "--run_folder",
        type=Path,
        required=True,
        help="Root of the PR-7F controlled training run (used to anchor outputs).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to diagnostic_checkpoint_final_step.pth produced by PR-7F.",
    )
    parser.add_argument(
        "--selection_log",
        type=Path,
        required=True,
        help="Path to selection_log.csv produced by the PR-7F training run.",
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        required=True,
        help="Directory containing siim_NNNNNN_0000.png training images.",
    )
    parser.add_argument(
        "--labels_dir",
        type=Path,
        required=True,
        help="Directory containing siim_NNNNNN.png training labels.",
    )
    parser.add_argument(
        "--foundation_checkpoint",
        type=Path,
        required=True,
        help="Path to Foundation X checkpoint (foundation_x.pth).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        choices=[256, 512],
        help="Runtime input size. Must match the PR-7F training img_size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto|cuda|cpu.",
    )
    parser.add_argument(
        "--num_positive",
        type=int,
        default=4,
        help="Number of positive validation cases to visualize.",
    )
    parser.add_argument(
        "--num_negative",
        type=int,
        default=4,
        help="Number of negative validation cases to visualize.",
    )
    parser.add_argument(
        "--step_label",
        type=str,
        default="0400",
        help="Used to name the visuals subfolder (val_step_{step_label}).",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help=(
            "Override output subdirectory under run_folder. "
            "Defaults to 'visuals/val_step_{step_label}'."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat partial visual writes as BLOCKED in the summary.",
    )
    return parser.parse_args(argv)


def _resolve_device(device_arg: str) -> Any:
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _normalize_case_id(raw: str) -> str:
    case_id = raw.strip()
    if case_id.endswith(".png"):
        case_id = case_id[: -len(".png")]
    if case_id.endswith("_0000"):
        case_id = case_id[: -len("_0000")]
    return case_id


def _read_selection_log(selection_log: Path) -> list[dict[str, Any]]:
    if not selection_log.exists():
        raise FileNotFoundError(f"selection_log not found: {selection_log}")
    with selection_log.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _coerce_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pick_diverse_cases(
    rows: list[dict[str, Any]],
    num_positive: int,
    num_negative: int,
) -> list[dict[str, Any]]:
    """Return up to num_positive + num_negative validation cases.

    Positives are chosen to be diverse across foreground_ratio when that
    column is present. Otherwise the script falls back to the order in
    the selection log.
    """

    val_rows = [r for r in rows if str(r.get("split", "")).strip().lower() == "val"]
    pos_rows = [r for r in val_rows if _coerce_bool(r.get("is_positive"))]
    neg_rows = [r for r in val_rows if not _coerce_bool(r.get("is_positive"))]

    has_fg_ratio = any("foreground_ratio" in r for r in pos_rows)
    if has_fg_ratio and pos_rows:
        sorted_pos = sorted(
            pos_rows,
            key=lambda r: _coerce_float(r.get("foreground_ratio"), 0.0),
        )
        picked_pos: list[dict[str, Any]] = []
        n_avail = len(sorted_pos)
        target = min(num_positive, n_avail)
        if target > 0:
            seen_ids: set[str] = set()
            for i in range(target):
                if n_avail == 1:
                    idx = 0
                else:
                    idx = int(round(i * (n_avail - 1) / max(1, target - 1)))
                candidate = sorted_pos[idx]
                cid = str(candidate.get("case_id", ""))
                offset = 0
                while cid in seen_ids and offset + 1 < n_avail:
                    offset += 1
                    next_idx = (idx + offset) % n_avail
                    candidate = sorted_pos[next_idx]
                    cid = str(candidate.get("case_id", ""))
                seen_ids.add(cid)
                picked_pos.append(candidate)
    else:
        picked_pos = pos_rows[:num_positive]

    picked_neg = neg_rows[:num_negative]

    out: list[dict[str, Any]] = []
    for row in picked_pos + picked_neg:
        out.append({**row, "_picked_split": "val"})
    return out


def _resolve_image_label_paths(
    case_id: str,
    images_dir: Path,
    labels_dir: Path,
) -> tuple[Path, Path]:
    image_candidate_with_suffix = images_dir / f"{case_id}_0000.png"
    image_candidate_plain = images_dir / f"{case_id}.png"
    if image_candidate_with_suffix.exists():
        image_path = image_candidate_with_suffix
    elif image_candidate_plain.exists():
        image_path = image_candidate_plain
    else:
        image_path = image_candidate_with_suffix  # default; existence checked later

    label_path = labels_dir / f"{case_id}.png"
    return image_path, label_path


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _build_overlay(
    image_uint8: np.ndarray,
    gt_mask_uint8: np.ndarray,
    pred_mask_uint8: np.ndarray,
) -> np.ndarray:
    rgb = np.stack([image_uint8, image_uint8, image_uint8], axis=-1).astype(np.float32)
    gt = gt_mask_uint8 > 0
    pred = pred_mask_uint8 > 0
    both = gt & pred
    if gt.any():
        rgb[gt] = 0.6 * rgb[gt] + 0.4 * np.array([255, 0, 0], dtype=np.float32)
    if pred.any():
        rgb[pred] = 0.6 * rgb[pred] + 0.4 * np.array([0, 255, 0], dtype=np.float32)
    if both.any():
        rgb[both] = 0.4 * rgb[both] + 0.6 * np.array([255, 255, 0], dtype=np.float32)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _threshold_suffix(threshold: float) -> str:
    return f"thr{int(round(threshold * 10)):02d}"


def output_subdir(args: argparse.Namespace) -> Path:
    if args.output_subdir:
        return Path(args.output_subdir)
    return Path("visuals") / f"val_step_{args.step_label}"


def _resolve_output_root(args: argparse.Namespace) -> Path:
    subdir = output_subdir(args)
    if subdir.is_absolute():
        return subdir
    return args.run_folder / subdir


def _save_uint8_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def _load_case_arrays(
    image_path: Path,
    label_path: Path,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    from PIL import Image

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    image_pil = Image.open(image_path).convert("L")
    if image_pil.size != (img_size, img_size):
        image_pil = image_pil.resize((img_size, img_size), Image.BILINEAR)
    image_arr = np.array(image_pil, dtype=np.float32) / 255.0

    label_available = label_path.exists()
    if label_available:
        label_pil = Image.open(label_path).convert("L")
        if label_pil.size != (img_size, img_size):
            label_pil = label_pil.resize((img_size, img_size), Image.NEAREST)
        mask_arr = (np.array(label_pil, dtype=np.float32) > 0.5).astype(np.float32)
    else:
        mask_arr = np.zeros_like(image_arr, dtype=np.float32)
    return image_arr, mask_arr, label_available


def _per_image_dice(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
) -> float:
    pred = pred_binary.astype(np.float32)
    gt = gt_binary.astype(np.float32)
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    intersection = float((pred * gt).sum())
    if pred_sum == 0.0 and gt_sum == 0.0:
        return 1.0
    if pred_sum + gt_sum == 0.0:
        return 0.0
    return (2.0 * intersection) / (pred_sum + gt_sum)


def _tensor_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _instantiate_model_and_hook(
    foundation_checkpoint: Path,
    diagnostic_checkpoint: Path,
    img_size: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Build HybridFoundationUNet, load Foundation X + diagnostic weights,
    register a forward hook on `model.final` to capture raw logits."""

    import torch

    from src.models.hybrid import HybridFoundationUNet  # noqa: WPS433

    if not foundation_checkpoint.exists():
        raise FileNotFoundError(f"foundation checkpoint not found: {foundation_checkpoint}")
    if not diagnostic_checkpoint.exists():
        raise FileNotFoundError(f"diagnostic checkpoint not found: {diagnostic_checkpoint}")

    model = HybridFoundationUNet(
        backbone_checkpoint=str(foundation_checkpoint),
        in_channels=1,
        num_classes=1,
        base_filters=64,
        frozen_backbone=True,
        img_size=img_size,
    )

    diag_payload = torch.load(str(diagnostic_checkpoint), map_location="cpu", weights_only=False)
    state_dict = diag_payload.get("model_state_dict") if isinstance(diag_payload, dict) else None
    if state_dict is None:
        raise KeyError(
            "diagnostic checkpoint missing 'model_state_dict'; cannot restore PR-7F weights",
        )
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
        "audit_name": diag_payload.get("audit_name") if isinstance(diag_payload, dict) else None,
        "schema_version": diag_payload.get("schema_version") if isinstance(diag_payload, dict) else None,
    }
    return model, hook_state


def _write_case_visuals(
    *,
    case_dir: Path,
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    prob_arr: np.ndarray,
    logits_arr: np.ndarray,
) -> list[Path]:
    case_dir.mkdir(parents=True, exist_ok=True)

    image_uint8 = (np.clip(image_arr * 255.0, 0.0, 255.0)).astype(np.uint8)
    gt_uint8 = (mask_arr > 0.5).astype(np.uint8) * 255
    prob_uint8 = _normalize_to_uint8(prob_arr)
    logits_uint8 = _normalize_to_uint8(logits_arr)

    written: list[Path] = []
    base_writes = (
        ("input.png", image_uint8),
        ("gt_mask.png", gt_uint8),
        ("probability_map.png", prob_uint8),
        ("raw_logits_map.png", logits_uint8),
    )
    for name, arr in base_writes:
        path = case_dir / name
        _save_uint8_png(arr, path)
        written.append(path)

    for threshold in VISUAL_THRESHOLDS:
        pred_bin = (prob_arr >= threshold).astype(np.uint8)
        pred_uint8 = pred_bin * 255
        overlay = _build_overlay(image_uint8, gt_uint8, pred_uint8)
        suffix = _threshold_suffix(threshold)
        for name, arr in (
            (f"prediction_mask_{suffix}.png", pred_uint8),
            (f"overlay_input_gt_pred_{suffix}.png", overlay),
        ):
            path = case_dir / name
            _save_uint8_png(arr, path)
            written.append(path)
    return written


def _per_case_metrics(
    prob_arr: np.ndarray,
    mask_arr: np.ndarray,
    label_available: bool,
) -> dict[str, Any]:
    prob_stats = _tensor_stats(prob_arr)
    gt_binary = (mask_arr > 0.5).astype(np.float32) if label_available else None

    per_threshold: dict[str, dict[str, float | bool]] = {}
    for threshold in VISUAL_THRESHOLDS:
        suffix = _threshold_suffix(threshold)
        pred_binary = (prob_arr >= threshold).astype(np.float32)
        pred_pos_ratio = float(pred_binary.mean())
        non_empty = bool(pred_binary.sum() > 0)
        if label_available and gt_binary is not None:
            dice_value = _per_image_dice(pred_binary, gt_binary)
        else:
            dice_value = float("nan")
        per_threshold[suffix] = {
            "dice": float(dice_value),
            "pred_pos_pixel_ratio": pred_pos_ratio,
            "prediction_non_empty": non_empty,
        }
    return {
        "prob_stats": prob_stats,
        "per_threshold": per_threshold,
    }


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
    return obj


def _write_cases_csv(
    cases: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fieldnames = [
        "case_id",
        "split",
        "is_positive",
        "foreground_pixels",
        "total_pixels",
        "foreground_ratio",
        "label_available",
        "prob_min",
        "prob_max",
        "prob_mean",
        "prob_std",
        "logits_min",
        "logits_max",
        "logits_mean",
        "logits_std",
        "dice_thr_030",
        "dice_thr_040",
        "dice_thr_050",
        "pred_pos_pixel_ratio_thr_030",
        "pred_pos_pixel_ratio_thr_040",
        "pred_pos_pixel_ratio_thr_050",
        "prediction_non_empty_thr_030",
        "prediction_non_empty_thr_040",
        "prediction_non_empty_thr_050",
        "visuals_dir",
        "visual_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({k: case.get(k, "") for k in fieldnames})


def _validate_inputs(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for attr, label in (
        ("run_folder", "run_folder"),
        ("checkpoint", "checkpoint"),
        ("selection_log", "selection_log"),
        ("images_dir", "images_dir"),
        ("labels_dir", "labels_dir"),
        ("foundation_checkpoint", "foundation_checkpoint"),
    ):
        path = getattr(args, attr)
        if path is None:
            failures.append(f"{label} is required")
            continue
        if not path.exists():
            failures.append(f"{label} not found: {path}")
    if args.num_positive < 0:
        failures.append("num_positive must be >= 0")
    if args.num_negative < 0:
        failures.append("num_negative must be >= 0")
    if args.num_positive + args.num_negative == 0:
        failures.append("at least one of num_positive / num_negative must be > 0")
    return failures


def run_visualization(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    failures = _validate_inputs(args)
    output_root = _resolve_output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)

    if failures:
        summary = {
            "schema_version": 1,
            "audit_name": "pr7f_final_visualization",
            "status": "BLOCKED",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "failures": failures,
            "warnings": [],
            "exit_code": 2,
            "setup": {
                "run_folder": str(args.run_folder),
                "checkpoint": str(args.checkpoint),
                "selection_log": str(args.selection_log),
                "images_dir": str(args.images_dir),
                "labels_dir": str(args.labels_dir),
                "foundation_checkpoint": str(args.foundation_checkpoint),
                "img_size": int(args.img_size),
                "device": args.device,
                "num_positive": int(args.num_positive),
                "num_negative": int(args.num_negative),
                "step_label": args.step_label,
            },
            "outputs": {
                "visuals_dir": str(output_root),
                "summary_json": str(output_root / "final_visualization_summary.json"),
                "cases_csv": str(output_root / "final_visualization_cases.csv"),
                "visual_files_written": 0,
            },
        }
        (output_root / "final_visualization_summary.json").write_text(
            json.dumps(_sanitize_for_json(summary), indent=2),
            encoding="utf-8",
        )
        return summary

    rows = _read_selection_log(args.selection_log)
    picked = pick_diverse_cases(rows, args.num_positive, args.num_negative)
    if not picked:
        failures.append("No validation cases selected from selection_log.")
        summary = {
            "schema_version": 1,
            "audit_name": "pr7f_final_visualization",
            "status": "BLOCKED",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "failures": failures,
            "warnings": [],
            "exit_code": 2,
            "setup": {
                "run_folder": str(args.run_folder),
                "img_size": int(args.img_size),
                "device": args.device,
            },
            "outputs": {
                "visuals_dir": str(output_root),
                "summary_json": str(output_root / "final_visualization_summary.json"),
                "cases_csv": str(output_root / "final_visualization_cases.csv"),
                "visual_files_written": 0,
            },
        }
        (output_root / "final_visualization_summary.json").write_text(
            json.dumps(_sanitize_for_json(summary), indent=2),
            encoding="utf-8",
        )
        return summary

    device = _resolve_device(args.device)

    warnings: list[str] = []
    model, hook_state = _instantiate_model_and_hook(
        foundation_checkpoint=args.foundation_checkpoint,
        diagnostic_checkpoint=args.checkpoint,
        img_size=int(args.img_size),
        device=device,
    )

    case_rows_out: list[dict[str, Any]] = []
    total_visuals_written = 0
    all_visuals_ok = True

    try:
        with torch.no_grad():
            for row in picked:
                case_id = _normalize_case_id(str(row.get("case_id", "")))
                if not case_id:
                    failures.append("Skipped row with empty case_id.")
                    all_visuals_ok = False
                    continue

                image_path_raw = row.get("image_path")
                label_path_raw = row.get("label_path")
                if image_path_raw:
                    image_path = Path(image_path_raw)
                else:
                    image_path, _ = _resolve_image_label_paths(case_id, args.images_dir, args.labels_dir)
                if label_path_raw:
                    label_path = Path(label_path_raw)
                else:
                    _, label_path = _resolve_image_label_paths(case_id, args.images_dir, args.labels_dir)

                try:
                    image_arr, mask_arr, label_available = _load_case_arrays(
                        image_path=image_path,
                        label_path=label_path,
                        img_size=int(args.img_size),
                    )
                except Exception as exc:
                    failures.append(f"Failed to load case {case_id}: {exc}")
                    all_visuals_ok = False
                    continue

                image_tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to(device)
                hook_state["raw_logits"] = None
                probs = model(image_tensor)
                raw_logits = hook_state["raw_logits"]
                if raw_logits is None:
                    failures.append(f"raw logits hook capture is None for case {case_id}")
                    all_visuals_ok = False
                    continue

                prob_arr = probs[0, 0].detach().float().cpu().numpy()
                logits_arr = raw_logits[0, 0].detach().float().cpu().numpy()

                if not np.all(np.isfinite(prob_arr)) or not np.all(np.isfinite(logits_arr)):
                    failures.append(f"non-finite probs/logits for case {case_id}")
                    all_visuals_ok = False
                    continue

                case_dir = output_root / case_id
                try:
                    written = _write_case_visuals(
                        case_dir=case_dir,
                        image_arr=image_arr,
                        mask_arr=mask_arr,
                        prob_arr=prob_arr,
                        logits_arr=logits_arr,
                    )
                except Exception as exc:
                    failures.append(f"Failed to write visuals for case {case_id}: {exc}")
                    all_visuals_ok = False
                    continue

                actual_basenames = {p.name for p in written}
                missing_visuals = [name for name in EXPECTED_VISUAL_BASENAMES if name not in actual_basenames]
                if missing_visuals:
                    failures.append(
                        f"Case {case_id} is missing visuals: {missing_visuals}",
                    )
                    all_visuals_ok = False

                total_visuals_written += len(written)
                metrics = _per_case_metrics(prob_arr, mask_arr, label_available)
                logits_stats = _tensor_stats(logits_arr)

                row_out = {
                    "case_id": case_id,
                    "split": row.get("split", "val"),
                    "is_positive": _coerce_bool(row.get("is_positive")),
                    "foreground_pixels": row.get("foreground_pixels", ""),
                    "total_pixels": row.get("total_pixels", ""),
                    "foreground_ratio": row.get("foreground_ratio", ""),
                    "label_available": bool(label_available),
                    "prob_min": metrics["prob_stats"]["min"],
                    "prob_max": metrics["prob_stats"]["max"],
                    "prob_mean": metrics["prob_stats"]["mean"],
                    "prob_std": metrics["prob_stats"]["std"],
                    "logits_min": logits_stats["min"],
                    "logits_max": logits_stats["max"],
                    "logits_mean": logits_stats["mean"],
                    "logits_std": logits_stats["std"],
                    "dice_thr_030": metrics["per_threshold"]["thr03"]["dice"],
                    "dice_thr_040": metrics["per_threshold"]["thr04"]["dice"],
                    "dice_thr_050": metrics["per_threshold"]["thr05"]["dice"],
                    "pred_pos_pixel_ratio_thr_030": metrics["per_threshold"]["thr03"]["pred_pos_pixel_ratio"],
                    "pred_pos_pixel_ratio_thr_040": metrics["per_threshold"]["thr04"]["pred_pos_pixel_ratio"],
                    "pred_pos_pixel_ratio_thr_050": metrics["per_threshold"]["thr05"]["pred_pos_pixel_ratio"],
                    "prediction_non_empty_thr_030": metrics["per_threshold"]["thr03"]["prediction_non_empty"],
                    "prediction_non_empty_thr_040": metrics["per_threshold"]["thr04"]["prediction_non_empty"],
                    "prediction_non_empty_thr_050": metrics["per_threshold"]["thr05"]["prediction_non_empty"],
                    "visuals_dir": str(case_dir),
                    "visual_count": len(written),
                }
                case_rows_out.append(row_out)
    finally:
        handle = hook_state.get("handle")
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                warnings.append("Failed to remove forward hook cleanly.")

    cases_csv_path = output_root / "final_visualization_cases.csv"
    _write_cases_csv(case_rows_out, cases_csv_path)

    if not case_rows_out:
        all_visuals_ok = False
        failures.append("No cases produced visuals.")

    status = "PASS" if all_visuals_ok and not failures else (
        "BLOCKED" if args.strict or not case_rows_out else "PARTIAL_PASS"
    )
    exit_code = 0 if status in {"PASS", "PARTIAL_PASS"} else 2
    if args.strict and not all_visuals_ok:
        status = "BLOCKED"
        exit_code = 2

    summary = {
        "schema_version": 1,
        "audit_name": "pr7f_final_visualization",
        "status": status,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "setup": {
            "run_folder": str(args.run_folder),
            "checkpoint": str(args.checkpoint),
            "selection_log": str(args.selection_log),
            "images_dir": str(args.images_dir),
            "labels_dir": str(args.labels_dir),
            "foundation_checkpoint": str(args.foundation_checkpoint),
            "img_size": int(args.img_size),
            "device": str(device),
            "num_positive": int(args.num_positive),
            "num_negative": int(args.num_negative),
            "step_label": args.step_label,
            "strict": bool(args.strict),
        },
        "selection": {
            "selected_case_ids": [r["case_id"] for r in case_rows_out],
            "counts": {
                "selected_total": len(case_rows_out),
                "selected_positive": sum(1 for r in case_rows_out if r["is_positive"]),
                "selected_negative": sum(1 for r in case_rows_out if not r["is_positive"]),
            },
        },
        "checkpoint_meta": hook_state.get("diagnostic_payload_meta", {}),
        "state_dict": {
            "missing_keys_count": len(hook_state.get("missing_keys", [])),
            "unexpected_keys_count": len(hook_state.get("unexpected_keys", [])),
        },
        "per_case": case_rows_out,
        "outputs": {
            "visuals_dir": str(output_root),
            "summary_json": str(output_root / "final_visualization_summary.json"),
            "cases_csv": str(cases_csv_path),
            "visual_files_written": int(total_visuals_written),
            "expected_per_case_visuals": list(EXPECTED_VISUAL_BASENAMES),
        },
        "warnings": warnings,
        "failures": failures,
        "exit_code": int(exit_code),
    }
    summary_path = output_root / "final_visualization_summary.json"
    summary_path.write_text(json.dumps(_sanitize_for_json(summary), indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_visualization(args)
    return int(summary.get("exit_code", 2))


if __name__ == "__main__":
    sys.exit(main())
