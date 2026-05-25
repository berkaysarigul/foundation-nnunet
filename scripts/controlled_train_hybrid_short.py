"""
Controlled short hybrid training for Dataset101 train split PNGs.

Diagnostic-only constraints:
- deterministic train/val subsets from imagesTr/labelsTr
- exactly max_steps optimizer steps
- no full-dataset inference
- outputs only under artifacts/diagnostics/hybrid_controlled_short_train
- optional diagnostic checkpoint only when explicitly requested

Usage:
    py scripts/controlled_train_hybrid_short.py \
      --input_dir nnUNet_raw/Dataset101_Pneumothorax/imagesTr \
      --labels_dir nnUNet_raw/Dataset101_Pneumothorax/labelsTr \
      --foundation_checkpoint checkpoints/foundation_x.pth \
      --output_dir artifacts/diagnostics/hybrid_controlled_short_train \
      --img_size 512 \
      --device auto \
      --num_train_cases 256 \
      --num_val_cases 128 \
      --max_steps 200 \
      --batch_size 1 \
      --lr 1e-4 \
      --bce_loss_weight 1.0 \
      --val_every 25 \
      --save_diagnostic_checkpoint \
      --strict
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import random
import sys
import time
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

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

from src.models.hybrid import HybridFoundationUNet
from src.training.losses import DiceFocalLoss
from src.training.metrics import dice_score


STAGE_CHANNEL_EXPECT = [128, 256, 512, 1024]
STAGE_STRIDE_EXPECT = [4, 8, 16, 32]

FORBIDDEN_OUTPUT_SEGMENTS = [
    "artifacts/runs",
    "nnunet_results",
    "nnunet_results_smoke",
    "nnunet_raw/dataset101_pneumothorax",
    "artifacts/diagnostics/foundation_x_smoke",
    "artifacts/diagnostics/hybrid_adapter_sanity",
    "artifacts/diagnostics/hybrid_tiny_train_smoke",
]

ALLOWED_OUTPUT_ANCHOR = "artifacts/diagnostics/hybrid_controlled_short_train"
CHECKPOINT_FILENAME = "diagnostic_checkpoint_final_step.pth"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled short hybrid training diagnostics. Runs deterministic, "
            "bounded training and scheduled validation checks."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("nnUNet_raw/Dataset101_Pneumothorax/imagesTr"),
        help="Directory of siim_NNNNNN_0000.png training images.",
    )
    parser.add_argument(
        "--labels_dir",
        type=Path,
        default=Path("nnUNet_raw/Dataset101_Pneumothorax/labelsTr"),
        help="Directory of siim_NNNNNN.png training labels.",
    )
    parser.add_argument(
        "--foundation_checkpoint",
        type=Path,
        default=None,
        help="Path to Foundation X checkpoint (.pth).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Alias for --foundation_checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/diagnostics/hybrid_controlled_short_train"),
        help="Diagnostic output root.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        choices=[256, 512],
        help="Runtime input size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto|cuda|cpu.",
    )
    parser.add_argument(
        "--num_train_cases",
        type=int,
        default=256,
        help="Deterministic controlled training subset size.",
    )
    parser.add_argument(
        "--num_val_cases",
        type=int,
        default=128,
        help="Deterministic controlled validation subset size.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Exact number of optimizer steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--bce_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to BCEWithLogitsLoss in total loss. Unweighted BCE is still logged.",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=25,
        help="Run validation every N training steps (plus step 0 and final step).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Promote suspicious non-blocking diagnostics to BLOCKED.",
    )
    parser.add_argument(
        "--no_visuals",
        action="store_true",
        help="Skip optional visual diagnostics.",
    )
    parser.add_argument(
        "--save_diagnostic_checkpoint",
        action="store_true",
        help="Save a diagnostic-only checkpoint under the output directory.",
    )
    parser.add_argument(
        "--unfrozen_backbone",
        action="store_true",
        help="Diagnostic override: allow Foundation X backbone gradients.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate paths and deterministic subset selection.",
    )
    return parser.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _normalize_case_id(raw: str) -> str:
    case_id = raw.strip()
    if case_id.endswith("_0000"):
        case_id = case_id[: -len("_0000")]
    if case_id.endswith(".png"):
        case_id = case_id[: -len(".png")]
    return case_id


def _resolve_paths(args: argparse.Namespace) -> None:
    for attr in ("input_dir", "labels_dir", "output_dir"):
        value = getattr(args, attr)
        if value is not None and not value.is_absolute():
            setattr(args, attr, REPO_ROOT / value)

    if args.foundation_checkpoint is None and args.checkpoint is None:
        args.foundation_checkpoint = REPO_ROOT / Path("checkpoints/foundation_x.pth")
    elif args.foundation_checkpoint is None and args.checkpoint is not None:
        args.foundation_checkpoint = args.checkpoint
    elif args.foundation_checkpoint is not None and args.checkpoint is not None:
        if args.foundation_checkpoint != args.checkpoint:
            raise ValueError(
                "Conflicting checkpoint arguments: --foundation_checkpoint and --checkpoint differ.",
            )

    if args.foundation_checkpoint is not None and not args.foundation_checkpoint.is_absolute():
        args.foundation_checkpoint = REPO_ROOT / args.foundation_checkpoint


def _validate_dataset_split_roots(input_dir: Path, labels_dir: Path) -> list[str]:
    failures: list[str] = []
    input_norm = input_dir.resolve().as_posix().lower()
    labels_norm = labels_dir.resolve().as_posix().lower()

    if "/imagests" in input_norm or input_dir.name.lower() == "imagests":
        failures.append("input_dir must use imagesTr, not imagesTs.")
    if "/heldout_labelsts" in labels_norm or labels_dir.name.lower() == "heldout_labelsts":
        failures.append("labels_dir must use labelsTr, not heldout_labelsTs.")
    if input_dir.name.lower() != "imagestr":
        failures.append("input_dir leaf directory must be imagesTr for PR-7.")
    if labels_dir.name.lower() != "labelstr":
        failures.append("labels_dir leaf directory must be labelsTr for PR-7.")

    return failures


def _validate_paths(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if not args.input_dir.exists():
        failures.append(f"input_dir not found: {args.input_dir}")
    if not args.labels_dir.exists():
        failures.append(f"labels_dir not found: {args.labels_dir}")
    if args.foundation_checkpoint is None:
        failures.append("foundation checkpoint argument resolved to None")
    elif not args.foundation_checkpoint.exists():
        failures.append(f"foundation checkpoint not found: {args.foundation_checkpoint}")
    if args.num_train_cases <= 0:
        failures.append("num_train_cases must be > 0")
    if args.num_val_cases <= 0:
        failures.append("num_val_cases must be > 0")
    if args.max_steps <= 0:
        failures.append("max_steps must be > 0")
    if args.batch_size <= 0:
        failures.append("batch_size must be > 0")
    if args.val_every <= 0:
        failures.append("val_every must be > 0")
    if args.bce_loss_weight < 0:
        failures.append("bce_loss_weight must be >= 0")

    failures.extend(_validate_dataset_split_roots(args.input_dir, args.labels_dir))
    return failures


def _validate_output_guardrail(output_dir: Path) -> tuple[bool, str]:
    normalized = output_dir.resolve().as_posix().lower()
    for segment in FORBIDDEN_OUTPUT_SEGMENTS:
        if segment in normalized and ALLOWED_OUTPUT_ANCHOR not in normalized:
            return (
                False,
                f"Output dir falls into forbidden location segment '{segment}': {output_dir}",
            )
    if ALLOWED_OUTPUT_ANCHOR not in normalized:
        return (
            False,
            f"Output dir must be under {ALLOWED_OUTPUT_ANCHOR}",
        )
    return True, ""


def _list_labeled_cases(input_dir: Path, labels_dir: Path) -> list[dict[str, Any]]:
    all_cases: list[dict[str, Any]] = []
    for image_path in sorted(input_dir.glob("siim_*_0000.png")):
        case_id = _normalize_case_id(image_path.stem)
        label_path = labels_dir / f"{case_id}.png"
        if not label_path.exists():
            continue
        label_max = int(np.array(Image.open(label_path).convert("L")).max())
        is_positive = label_max > 0
        all_cases.append(
            {
                "case_id": case_id,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "label_max": label_max,
                "is_positive": is_positive,
            },
        )
    return all_cases


def _interleave_fill(
    pos_pool: list[dict[str, Any]],
    neg_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pos_case, neg_case in zip_longest(pos_pool, neg_pool):
        if pos_case is not None:
            out.append(pos_case)
        if neg_case is not None:
            out.append(neg_case)
    return out


def _target_counts(
    total: int,
    pos_avail: int,
    neg_avail: int,
) -> tuple[int, int]:
    # Semi-balanced by default, but always bounded by availability.
    target_pos = min(total // 2, pos_avail)
    target_neg = min(total - target_pos, neg_avail)

    if target_pos + target_neg < total:
        remaining = total - (target_pos + target_neg)
        extra_pos = min(remaining, max(0, pos_avail - target_pos))
        target_pos += extra_pos
        remaining -= extra_pos
        if remaining > 0:
            extra_neg = min(remaining, max(0, neg_avail - target_neg))
            target_neg += extra_neg

    return target_pos, target_neg


def _choose_split(
    *,
    pos_pool: list[dict[str, Any]],
    neg_pool: list[dict[str, Any]],
    total: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if total <= 0:
        return [], pos_pool, neg_pool

    target_pos, target_neg = _target_counts(total, len(pos_pool), len(neg_pool))

    selected = pos_pool[:target_pos] + neg_pool[:target_neg]
    pos_rem = pos_pool[target_pos:]
    neg_rem = neg_pool[target_neg:]

    remaining = total - len(selected)
    if remaining > 0:
        filler = _interleave_fill(pos_rem, neg_rem)
        fill_slice = filler[:remaining]
        selected.extend(fill_slice)

        used_case_ids = {item["case_id"] for item in fill_slice}
        pos_rem = [item for item in pos_rem if item["case_id"] not in used_case_ids]
        neg_rem = [item for item in neg_rem if item["case_id"] not in used_case_ids]

    return selected[:total], pos_rem, neg_rem


def _alternate_by_class(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [case for case in cases if case["is_positive"]]
    negatives = [case for case in cases if not case["is_positive"]]
    ordered: list[dict[str, Any]] = []
    for pos_case, neg_case in zip_longest(positives, negatives):
        if pos_case is not None:
            ordered.append(pos_case)
        if neg_case is not None:
            ordered.append(neg_case)
    return ordered


def select_train_val_cases(
    input_dir: Path,
    labels_dir: Path,
    num_train_cases: int,
    num_val_cases: int,
) -> dict[str, Any]:
    all_cases = _list_labeled_cases(input_dir, labels_dir)
    pos_cases = [case for case in all_cases if case["is_positive"]]
    neg_cases = [case for case in all_cases if not case["is_positive"]]

    train_cases, pos_remaining, neg_remaining = _choose_split(
        pos_pool=pos_cases,
        neg_pool=neg_cases,
        total=num_train_cases,
    )
    train_cases = _alternate_by_class(train_cases)
    val_cases, _, _ = _choose_split(
        pos_pool=pos_remaining,
        neg_pool=neg_remaining,
        total=num_val_cases,
    )

    for case in train_cases:
        case["split"] = "train"
        case["selection_bucket"] = "train_positive" if case["is_positive"] else "train_negative"
    for case in val_cases:
        case["split"] = "val"
        case["selection_bucket"] = "val_positive" if case["is_positive"] else "val_negative"

    train_ids = {case["case_id"] for case in train_cases}
    val_ids = {case["case_id"] for case in val_cases}
    overlap = sorted(train_ids & val_ids)

    train_pos = sum(1 for case in train_cases if case["is_positive"])
    train_neg = len(train_cases) - train_pos
    val_pos = sum(1 for case in val_cases if case["is_positive"])
    val_neg = len(val_cases) - val_pos

    warnings: list[str] = []
    if overlap:
        warnings.append(f"Train/val overlap detected: {overlap[:5]} ...")
    if train_pos == 0 or train_neg == 0:
        warnings.append("Training subset is single-class; semi-balanced selection unmet.")
    if val_pos == 0 or val_neg == 0:
        warnings.append("Validation subset is single-class; semi-balanced selection unmet.")

    return {
        "train": train_cases,
        "val": val_cases,
        "counts": {
            "source_total": len(all_cases),
            "source_positive": len(pos_cases),
            "source_negative": len(neg_cases),
            "train_total": len(train_cases),
            "train_positive": train_pos,
            "train_negative": train_neg,
            "val_total": len(val_cases),
            "val_positive": val_pos,
            "val_negative": val_neg,
            "overlap_count": len(overlap),
        },
        "overlap_case_ids": overlap,
        "warnings": warnings,
    }


def _load_case_tensor(
    case: dict[str, Any],
    img_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(case["image_path"]).convert("L")
    mask = Image.open(case["label_path"]).convert("L")

    if image.size != (img_size, img_size):
        image = image.resize((img_size, img_size), Image.BILINEAR)
    if mask.size != (img_size, img_size):
        mask = mask.resize((img_size, img_size), Image.NEAREST)

    image_arr = np.array(image, dtype=np.float32) / 255.0
    mask_arr = (np.array(mask, dtype=np.float32) > 0.5).astype(np.float32)

    image_tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0).to(device)
    return image_tensor, mask_tensor


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    tf = tensor.float()
    nan_mask = torch.isnan(tf)
    inf_mask = torch.isinf(tf)
    finite = tf[~(nan_mask | inf_mask)]
    if finite.numel() == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "nan_count": int(nan_mask.sum().item()),
            "inf_count": int(inf_mask.sum().item()),
        }
    return {
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std().item()),
        "nan_count": int(nan_mask.sum().item()),
        "inf_count": int(inf_mask.sum().item()),
    }


def _new_tensor_aggregate() -> dict[str, Any]:
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "nan_count": 0,
        "inf_count": 0,
    }


def _update_tensor_aggregate(aggregate: dict[str, Any], tensor: torch.Tensor) -> None:
    tf = tensor.detach().float()
    nan_mask = torch.isnan(tf)
    inf_mask = torch.isinf(tf)
    finite = tf[~(nan_mask | inf_mask)].double()

    aggregate["nan_count"] += int(nan_mask.sum().item())
    aggregate["inf_count"] += int(inf_mask.sum().item())
    if finite.numel() == 0:
        return

    aggregate["count"] += int(finite.numel())
    aggregate["sum"] += float(finite.sum().item())
    aggregate["sum_sq"] += float((finite * finite).sum().item())
    aggregate["min"] = min(float(aggregate["min"]), float(finite.min().item()))
    aggregate["max"] = max(float(aggregate["max"]), float(finite.max().item()))


def _finalize_tensor_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    count = int(aggregate["count"])
    if count == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "nan_count": int(aggregate["nan_count"]),
            "inf_count": int(aggregate["inf_count"]),
        }

    total = float(aggregate["sum"])
    mean = total / float(count)
    if count > 1:
        variance = (float(aggregate["sum_sq"]) - (total * total / float(count))) / float(count - 1)
        std = math.sqrt(max(0.0, variance))
    else:
        std = 0.0

    return {
        "min": float(aggregate["min"]),
        "max": float(aggregate["max"]),
        "mean": mean,
        "std": float(std),
        "nan_count": int(aggregate["nan_count"]),
        "inf_count": int(aggregate["inf_count"]),
    }


def _shape_of(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [_shape_of(item) for item in obj]
    return str(type(obj))


def _stage_contract_ok(
    stage_shapes: list[list[int]],
    img_size: int,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if len(stage_shapes) != 4:
        return False, [f"Foundation X expected 4 stage maps, got {len(stage_shapes)}"]

    for idx, shape in enumerate(stage_shapes):
        if len(shape) != 4:
            errors.append(f"stage_{idx} is not 4D: {shape}")
            continue
        _, channels, height, width = shape
        exp_channels = STAGE_CHANNEL_EXPECT[idx]
        exp_spatial = img_size // STAGE_STRIDE_EXPECT[idx]
        if channels != exp_channels:
            errors.append(
                f"stage_{idx} channel mismatch: expected {exp_channels}, got {channels}",
            )
        if height != exp_spatial or width != exp_spatial:
            errors.append(
                f"stage_{idx} spatial mismatch: expected {exp_spatial}x{exp_spatial}, got {height}x{width}",
            )
    return len(errors) == 0, errors


def _positive_ratio_and_non_empty_rate(probabilities: torch.Tensor) -> tuple[float, float]:
    pred_bin = (probabilities >= 0.5).float()
    pos_ratio = float(pred_bin.mean().item())
    per_case_non_empty = pred_bin.reshape(pred_bin.shape[0], -1).sum(dim=1) > 0
    non_empty_rate = float(per_case_non_empty.float().mean().item())
    return pos_ratio, non_empty_rate


class HookRecorder:
    def __init__(self):
        self.handles: list[Any] = []
        self.last_records: dict[str, dict[str, Any]] = {}
        self.raw_logits: torch.Tensor | None = None

    def add_hook(self, model: torch.nn.Module, module_name: str, alias: str | None = None) -> None:
        target = getattr(model, module_name, None)
        if target is None:
            return
        hook_name = alias if alias is not None else module_name

        def _hook(_module, inputs, output):
            self.last_records[hook_name] = {
                "input_shape": _shape_of(inputs),
                "output_shape": _shape_of(output),
            }
            if hook_name == "final":
                self.raw_logits = output if isinstance(output, torch.Tensor) else None

        self.handles.append(target.register_forward_hook(_hook))

    def clear(self) -> None:
        self.last_records = {}
        self.raw_logits = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _register_hooks(model: HybridFoundationUNet) -> HookRecorder:
    recorder = HookRecorder()
    recorder.add_hook(model, "foundation_x")
    recorder.add_hook(model, "final")
    return recorder


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


def _write_visuals(
    output_dir: Path,
    split: str,
    case_id: str,
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    probs_tensor: torch.Tensor,
    logits_tensor: torch.Tensor,
) -> list[str]:
    case_dir = output_dir / "visuals" / split / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    img = (image_tensor[0, 0].detach().cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    gt = (mask_tensor[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    prob = probs_tensor[0, 0].detach().cpu().numpy()
    logits = logits_tensor[0, 0].detach().cpu().numpy()
    pred_mask = (prob >= 0.5).astype(np.uint8) * 255
    prob_u8 = _normalize_to_uint8(prob)
    logits_u8 = _normalize_to_uint8(logits)
    overlay = _build_overlay(img, gt, pred_mask)

    written: list[str] = []
    for name, arr in (
        ("input.png", img),
        ("gt_mask.png", gt),
        ("probability_map.png", prob_u8),
        ("raw_logits_map.png", logits_u8),
        ("prediction_mask_thr05.png", pred_mask),
        ("overlay_input_gt_pred.png", overlay),
    ):
        path = case_dir / name
        Image.fromarray(arr).save(path)
        written.append(str(path))
    return written


def _group_for_param(name: str) -> str:
    if name.startswith("foundation_x."):
        return "foundation_backbone"
    if name.startswith("fusion_") or name.startswith("h16_fx_context") or name.startswith("h32_") or name.startswith("context_merge"):
        return "adapter_fusion"
    if name.startswith("dec") or name.startswith("up") or name.startswith("final"):
        return "decoder_head"
    if name.startswith("enc"):
        return "encoder_cnn"
    return "other_trainable"


def build_optimizer_for_trainable_params(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, list[str], int]:
    trainable_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        [param for _, param in trainable_named],
        lr=lr,
        weight_decay=weight_decay,
    )
    all_frozen_backbone = sum(
        1 for name, param in model.named_parameters() if name.startswith("foundation_x.") and not param.requires_grad
    )
    return optimizer, [name for name, _ in trainable_named], all_frozen_backbone


def _collect_gradient_rows(
    model: torch.nn.Module,
    step_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = {
        "frozen_backbone_params": 0,
        "frozen_backbone_grad_present": 0,
        "trainable_non_backbone_params": 0,
        "trainable_non_backbone_grad_present": 0,
        "trainable_non_backbone_grad_finite": 0,
        "trainable_non_backbone_grad_nonzero": 0,
        "grad_norm_l2_total": 0.0,
        "grad_nonfinite_count": 0,
    }
    sum_norm_sq = 0.0

    for name, param in model.named_parameters():
        group = _group_for_param(name)
        grad_present = param.grad is not None
        grad_finite = False
        grad_nonzero = False
        grad_norm_l2 = 0.0
        if grad_present:
            grad_tensor = param.grad.detach()
            grad_finite = bool(torch.isfinite(grad_tensor).all().item())
            grad_nonzero = bool(float(grad_tensor.abs().sum().item()) > 0.0)
            grad_norm_l2 = float(torch.norm(grad_tensor).item())
            sum_norm_sq += grad_norm_l2 * grad_norm_l2
            if not grad_finite:
                summary["grad_nonfinite_count"] += 1

        rows.append(
            {
                "step": int(step_index),
                "param_name": name,
                "requires_grad": bool(param.requires_grad),
                "group": group,
                "grad_present": grad_present,
                "grad_finite": grad_finite,
                "grad_nonzero": grad_nonzero,
                "grad_norm_l2": grad_norm_l2,
            },
        )

        if group == "foundation_backbone":
            if not param.requires_grad:
                summary["frozen_backbone_params"] += 1
                if grad_present:
                    summary["frozen_backbone_grad_present"] += 1
        else:
            if param.requires_grad:
                summary["trainable_non_backbone_params"] += 1
                if grad_present:
                    summary["trainable_non_backbone_grad_present"] += 1
                if grad_finite:
                    summary["trainable_non_backbone_grad_finite"] += 1
                if grad_nonzero:
                    summary["trainable_non_backbone_grad_nonzero"] += 1

    summary["grad_norm_l2_total"] = float(sum_norm_sq ** 0.5)
    return rows, summary


def _write_selection_log(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "selection_log.csv"
    fieldnames = [
        "split",
        "case_id",
        "is_positive",
        "label_max",
        "selection_bucket",
        "image_path",
        "label_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return path


def _write_train_steps(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "train_steps.csv"
    fieldnames = [
        "step",
        "case_ids",
        "loss_total",
        "loss_dice_focal",
        "loss_bce_with_logits",
        "prob_min",
        "prob_max",
        "prob_mean",
        "prob_std",
        "prob_nan_count",
        "prob_inf_count",
        "logits_min",
        "logits_max",
        "logits_mean",
        "logits_std",
        "logits_nan_count",
        "logits_inf_count",
        "pred_pos_pixel_ratio",
        "pred_non_empty_rate",
        "grad_norm_l2_total",
        "grad_nonfinite_count",
        "frozen_backbone_grad_present",
        "trainable_non_backbone_grad_present",
        "trainable_non_backbone_grad_nonzero",
        "step_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return path


def _write_validation_steps(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "validation_steps.csv"
    fieldnames = [
        "step",
        "num_cases",
        "loss_total_mean",
        "loss_dice_focal_mean",
        "loss_bce_with_logits_mean",
        "dice_mean",
        "dice_pos_mean",
        "prob_min",
        "prob_max",
        "prob_mean",
        "prob_std",
        "logits_min",
        "logits_max",
        "logits_mean",
        "logits_std",
        "prob_foreground_ratio_thr05",
        "prediction_non_empty_rate",
        "pred_pos_pixel_ratio_mean",
        "all_background_collapse",
        "all_foreground_collapse",
        "degenerate",
        "prob_nan_total",
        "prob_inf_total",
        "logits_nan_total",
        "logits_inf_total",
        "val_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return path


def _write_gradient_stats(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "gradient_stats.csv"
    fieldnames = [
        "step",
        "param_name",
        "requires_grad",
        "group",
        "grad_present",
        "grad_finite",
        "grad_nonzero",
        "grad_norm_l2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return path


def _save_progress_plot(
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> str | None:
    if not train_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    train_steps = [int(row["step"]) for row in train_rows]
    train_losses = [float(row["loss_total"]) for row in train_rows]
    val_steps = [int(row["step"]) for row in val_rows]
    val_losses = [float(row["loss_total_mean"]) for row in val_rows]

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_steps, train_losses, label="train_loss_total", color="#1f77b4")
    if val_steps:
        ax.plot(val_steps, val_losses, label="val_loss_total_mean", color="#ff7f0e")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Controlled Short Hybrid Training Progress")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    path = output_dir / "progress.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def _write_report(summary: dict[str, Any], output_dir: Path) -> Path:
    setup = summary["setup"]
    training = summary["training"]
    validation = summary["validation"]
    gradients = summary["gradients"]
    degeneracy = summary["degeneracy"]

    lines: list[str] = [
        "# Controlled Short Hybrid Training Report",
        "",
        "## Repository findings",
        "",
        "- Existing hybrid path reused: `src/models/hybrid.py::HybridFoundationUNet`.",
        "- Existing backbone path reused: `src/models/backbone.py::FoundationXBackbone`.",
        "- Existing PR-6 diagnostics were extended for longer bounded training and scheduled validation.",
        "- The full single-split runner was intentionally not reused because it writes to `artifacts/runs/` and exceeds PR-7 scope.",
        "",
        "## Setup",
        "",
        "```",
        setup["command"],
        "```",
        "",
        f"- Device: `{setup['device']}`",
        f"- Train cases: {setup['num_train_cases_processed']} (pos={setup['num_train_pos']}, neg={setup['num_train_neg']})",
        f"- Val cases: {setup['num_val_cases_processed']} (pos={setup['num_val_pos']}, neg={setup['num_val_neg']})",
        f"- Max steps: {setup['max_steps']}",
        f"- Validation interval: {setup['val_every']}",
        f"- Batch size: {setup['batch_size']}",
        f"- Learning rate: {setup['learning_rate']}",
        f"- BCE loss weight: {setup['bce_loss_weight']}",
        f"- Frozen backbone: {setup['frozen_backbone']}",
        "",
        "## Dataset selection",
        "",
        f"- Train/val disjoint overlap count: {summary['selection']['counts']['overlap_count']}",
        f"- Train case IDs count: {len(summary['selection']['train_case_ids'])}",
        f"- Val case IDs count: {len(summary['selection']['val_case_ids'])}",
        "",
        "## Training results",
        "",
        f"- Steps completed: {training['steps_completed']} / {setup['max_steps']}",
        f"- Train total loss series: {training['loss_total_per_step']}",
        f"- Train predicted positive pixel ratio series: {training['pred_pos_pixel_ratio_per_step']}",
        f"- Train non-empty prediction rate series: {training['pred_non_empty_rate_per_step']}",
        f"- Probability output shape sample: `{training['prob_output_shape_sample']}`",
        f"- Raw logits shape sample: `{training['raw_logits_shape_sample']}`",
        f"- NaN/Inf totals: prob_nan={training['prob_nan_total']}, prob_inf={training['prob_inf_total']}, logits_nan={training['logits_nan_total']}, logits_inf={training['logits_inf_total']}",
        "",
        "## Validation results",
        "",
        f"- Validation schedule steps: {validation['scheduled_steps']}",
        f"- Validation executed steps: {validation['executed_steps']}",
        f"- Validation mean total loss per step: {validation['loss_total_mean_per_step']}",
        f"- Validation Dice mean per step: {validation['dice_mean_per_step']}",
        f"- Validation probability min per step: {validation['prob_min_per_step']}",
        f"- Validation probability max per step: {validation['prob_max_per_step']}",
        f"- Validation probability mean per step: {validation['prob_mean_per_step']}",
        f"- Validation probability std per step: {validation['prob_std_per_step']}",
        f"- Validation raw logit min per step: {validation['logits_min_per_step']}",
        f"- Validation raw logit max per step: {validation['logits_max_per_step']}",
        f"- Validation raw logit mean per step: {validation['logits_mean_per_step']}",
        f"- Validation raw logit std per step: {validation['logits_std_per_step']}",
        f"- Validation probability foreground ratio @0.5 per step: {validation['prob_foreground_ratio_thr05_per_step']}",
        f"- Validation non-empty rate per step: {validation['prediction_non_empty_rate_per_step']}",
        "",
        "## Gradient and frozen-backbone checks",
        "",
        f"- Frozen backbone no-grad check: {gradients['frozen_backbone_no_grad']}",
        f"- Trainable non-backbone gradient present check: {gradients['trainable_gradients_present']}",
        f"- Trainable non-backbone finite gradient check: {gradients['trainable_gradients_finite']}",
        f"- Optimizer excludes frozen backbone: {gradients['optimizer_excludes_frozen_backbone']}",
        "",
        "## Degeneracy checks",
        "",
        f"- Any degenerate validation point: {degeneracy['any_degenerate_validation_point']}",
        f"- Persistent collapse from start: {degeneracy['persistent_collapse_from_start']}",
        f"- All-background validation points: {degeneracy['all_background_point_count']}",
        f"- All-foreground validation points: {degeneracy['all_foreground_point_count']}",
        "",
        "## Artifact outputs",
        "",
        f"- Summary JSON: `{summary['outputs']['summary_json']}`",
        f"- Summary YAML: `{summary['outputs']['summary_yaml']}`",
        f"- Report: `{summary['outputs']['report_md']}`",
        f"- Train steps CSV: `{summary['outputs']['train_steps_csv']}`",
        f"- Validation steps CSV: `{summary['outputs']['validation_steps_csv']}`",
        f"- Gradient stats CSV: `{summary['outputs']['gradient_stats_csv']}`",
        f"- Selection log CSV: `{summary['outputs']['selection_log_csv']}`",
        f"- Progress plot: `{summary['outputs']['progress_png']}`",
        f"- Visual file count: {summary['outputs']['visual_count']}",
        "",
        "## Failure cases",
        "",
    ]
    failures = summary.get("failures", [])
    if failures:
        lines.extend([f"- {item}" for item in failures])
    else:
        lines.append("None.")

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Status: `{summary['status']}`",
            "",
            "## Next recommended PR",
            "",
            f"- `{summary['next_recommended_pr']}`",
        ],
    )

    path = output_dir / "hybrid_controlled_short_train_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_model(args: argparse.Namespace) -> HybridFoundationUNet:
    frozen_backbone = not args.unfrozen_backbone
    return HybridFoundationUNet(
        backbone_checkpoint=str(args.foundation_checkpoint),
        in_channels=1,
        num_classes=1,
        base_filters=64,
        frozen_backbone=frozen_backbone,
        img_size=args.img_size,
    )


def _validation_schedule(max_steps: int, val_every: int) -> list[int]:
    steps = {0, max_steps}
    for step in range(val_every, max_steps + 1, val_every):
        steps.add(step)
    return sorted(steps)


def _compute_decision(
    *,
    pass_criteria: dict[str, bool],
    suspicious_signals: dict[str, bool],
    strict: bool,
) -> tuple[str, int]:
    hard_blockers = [
        "model_instantiates",
        "checkpoint_loads",
        "train_selection_ok",
        "val_selection_ok",
        "train_val_disjoint",
        "max_steps_completed",
        "optimizer_step_success",
        "losses_finite",
        "outputs_finite",
        "gradients_finite",
        "frozen_backbone_no_grad",
        "trainable_gradients_present",
        "validation_schedule_complete",
        "validation_finite",
        "no_persistent_degeneracy",
        "no_forbidden_writes",
    ]
    if any(not pass_criteria.get(key, False) for key in hard_blockers):
        return "BLOCKED", 2

    if any(suspicious_signals.values()):
        return ("BLOCKED", 2) if strict else ("PARTIAL_PASS", 0)

    return "PASS", 0


def _evaluate_validation_point(
    *,
    model: HybridFoundationUNet,
    hook_recorder: HookRecorder,
    val_cases: list[dict[str, Any]],
    img_size: int,
    device: torch.device,
    criterion_dice: DiceFocalLoss,
    criterion_bce_logits: torch.nn.BCEWithLogitsLoss,
    bce_loss_weight: float,
    step_index: int,
    output_dir: Path,
    no_visuals: bool,
    visual_paths: list[str],
) -> tuple[dict[str, Any], list[str]]:
    t0 = time.time()
    rows: list[dict[str, Any]] = []

    total_losses: list[float] = []
    dice_losses: list[float] = []
    bce_losses: list[float] = []
    dice_values: list[float] = []
    dice_pos_values: list[float] = []
    pos_ratios: list[float] = []
    non_empty_rates: list[float] = []

    prob_aggregate = _new_tensor_aggregate()
    logits_aggregate = _new_tensor_aggregate()

    model.eval()
    with torch.no_grad():
        for case in val_cases:
            hook_recorder.clear()
            image_tensor, mask_tensor = _load_case_tensor(case, img_size, device)
            probs = model(image_tensor)
            raw_logits = hook_recorder.raw_logits
            if raw_logits is None:
                raise RuntimeError(f"Validation case {case['case_id']}: raw logits hook capture is None.")

            if list(probs.shape) != list(mask_tensor.shape) or list(raw_logits.shape) != list(mask_tensor.shape):
                raise RuntimeError(
                    f"Validation case {case['case_id']}: output shape mismatch "
                    f"prob={list(probs.shape)} logits={list(raw_logits.shape)} mask={list(mask_tensor.shape)}",
                )

            prob_stats = _tensor_stats(probs)
            logits_stats = _tensor_stats(raw_logits)
            _update_tensor_aggregate(prob_aggregate, probs)
            _update_tensor_aggregate(logits_aggregate, raw_logits)

            loss_dice = criterion_dice(probs, mask_tensor)
            loss_bce = criterion_bce_logits(raw_logits, mask_tensor)
            total_loss = loss_dice + float(bce_loss_weight) * loss_bce

            dice_all = float(dice_score(probs, mask_tensor, threshold=0.5, reduction="mean").item())
            dice_pos = dice_score(probs, mask_tensor, threshold=0.5, reduction="positive_mean")
            dice_pos_val = float(dice_pos.item()) if bool(torch.isfinite(dice_pos).item()) else float("nan")

            pos_ratio, non_empty_rate = _positive_ratio_and_non_empty_rate(probs)

            total_losses.append(float(total_loss.item()))
            dice_losses.append(float(loss_dice.item()))
            bce_losses.append(float(loss_bce.item()))
            dice_values.append(dice_all)
            dice_pos_values.append(dice_pos_val)
            pos_ratios.append(pos_ratio)
            non_empty_rates.append(non_empty_rate)

            rows.append(
                {
                    "case_id": case["case_id"],
                    "loss_total": float(total_loss.item()),
                    "loss_dice_focal": float(loss_dice.item()),
                    "loss_bce_with_logits": float(loss_bce.item()),
                    "dice_mean": dice_all,
                    "dice_pos_mean": dice_pos_val,
                    "prob_min": prob_stats["min"],
                    "prob_max": prob_stats["max"],
                    "prob_mean": prob_stats["mean"],
                    "prob_std": prob_stats["std"],
                    "logits_min": logits_stats["min"],
                    "logits_max": logits_stats["max"],
                    "logits_mean": logits_stats["mean"],
                    "logits_std": logits_stats["std"],
                    "pred_pos_pixel_ratio": pos_ratio,
                    "prediction_non_empty_rate": non_empty_rate,
                    "prob_shape": list(probs.shape),
                    "raw_logits_shape": list(raw_logits.shape),
                },
            )

            if not no_visuals and len(visual_paths) < 24:
                written = _write_visuals(
                    output_dir,
                    f"val_step_{step_index:04d}",
                    case["case_id"],
                    image_tensor,
                    mask_tensor,
                    probs,
                    raw_logits,
                )
                visual_paths.extend(written)

    pred_non_empty_rate_mean = float(np.mean(non_empty_rates)) if non_empty_rates else 0.0
    pred_pos_ratio_mean = float(np.mean(pos_ratios)) if pos_ratios else 0.0
    all_background = pred_non_empty_rate_mean == 0.0
    all_foreground = pred_pos_ratio_mean >= 0.95
    degenerate = all_background or all_foreground
    prob_summary = _finalize_tensor_aggregate(prob_aggregate)
    logits_summary = _finalize_tensor_aggregate(logits_aggregate)

    out = {
        "step": int(step_index),
        "num_cases": len(val_cases),
        "loss_total_mean": float(np.mean(total_losses)) if total_losses else float("nan"),
        "loss_dice_focal_mean": float(np.mean(dice_losses)) if dice_losses else float("nan"),
        "loss_bce_with_logits_mean": float(np.mean(bce_losses)) if bce_losses else float("nan"),
        "dice_mean": float(np.mean(dice_values)) if dice_values else float("nan"),
        "dice_pos_mean": float(np.nanmean(dice_pos_values)) if dice_pos_values else float("nan"),
        "prob_min": prob_summary["min"],
        "prob_max": prob_summary["max"],
        "prob_mean": prob_summary["mean"],
        "prob_std": prob_summary["std"],
        "logits_min": logits_summary["min"],
        "logits_max": logits_summary["max"],
        "logits_mean": logits_summary["mean"],
        "logits_std": logits_summary["std"],
        "prob_foreground_ratio_thr05": pred_pos_ratio_mean,
        "prediction_non_empty_rate": pred_non_empty_rate_mean,
        "pred_pos_pixel_ratio_mean": pred_pos_ratio_mean,
        "all_background_collapse": bool(all_background),
        "all_foreground_collapse": bool(all_foreground),
        "degenerate": bool(degenerate),
        "prob_nan_total": int(prob_summary["nan_count"]),
        "prob_inf_total": int(prob_summary["inf_count"]),
        "logits_nan_total": int(logits_summary["nan_count"]),
        "logits_inf_total": int(logits_summary["inf_count"]),
        "val_ms": (time.time() - t0) * 1000.0,
        "per_case_rows": rows,
    }
    return out, visual_paths


def run_controlled_short_training(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    _resolve_paths(args)

    output_guardrail_ok, guardrail_error = _validate_output_guardrail(args.output_dir)
    if not output_guardrail_ok:
        blocked = {
            "schema_version": 1,
            "audit_name": "hybrid_controlled_short_train",
            "status": "BLOCKED",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "failures": [guardrail_error],
            "exit_code": 2,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "hybrid_controlled_short_train_summary.json").write_text(
            json.dumps(blocked, indent=2),
            encoding="utf-8",
        )
        return blocked

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = _validate_paths(args)
    if failures:
        blocked = {
            "schema_version": 1,
            "audit_name": "hybrid_controlled_short_train",
            "status": "BLOCKED",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "failures": failures,
            "exit_code": 2,
        }
        (args.output_dir / "hybrid_controlled_short_train_summary.json").write_text(
            json.dumps(blocked, indent=2),
            encoding="utf-8",
        )
        return blocked

    selection = select_train_val_cases(
        args.input_dir,
        args.labels_dir,
        args.num_train_cases,
        args.num_val_cases,
    )
    selection_rows = selection["train"] + selection["val"]
    _write_selection_log(selection_rows, args.output_dir)
    failures.extend(selection.get("warnings", []))

    counts = selection["counts"]
    if counts["train_total"] < args.num_train_cases:
        failures.append(
            f"Selected train cases {counts['train_total']} < requested {args.num_train_cases}.",
        )
    if counts["val_total"] < args.num_val_cases:
        failures.append(
            f"Selected val cases {counts['val_total']} < requested {args.num_val_cases}.",
        )
    if counts["overlap_count"] > 0:
        failures.append(f"Train/val overlap detected ({counts['overlap_count']} cases).")

    if args.dry_run:
        summary = {
            "schema_version": 1,
            "audit_name": "hybrid_controlled_short_train",
            "status": "PARTIAL_PASS",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dry_run": True,
            "selection_counts": counts,
            "selected_train_case_ids": [case["case_id"] for case in selection["train"]],
            "selected_val_case_ids": [case["case_id"] for case in selection["val"]],
            "exit_code": 0,
        }
        (args.output_dir / "hybrid_controlled_short_train_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    device = _resolve_device(args.device)
    ckpt_sha = _compute_sha256(args.foundation_checkpoint)
    ckpt_size = args.foundation_checkpoint.stat().st_size

    model: HybridFoundationUNet | None = None
    model_instantiates = False
    checkpoint_loads = False
    model_init_error = ""
    hook_recorder: HookRecorder | None = None

    try:
        model = _build_model(args).to(device)
        model_instantiates = True
        checkpoint_loads = True
        hook_recorder = _register_hooks(model)
    except Exception as exc:
        model_init_error = str(exc)
        failures.append(f"Model instantiation/checkpoint load failed: {exc}")

    train_step_rows: list[dict[str, Any]] = []
    validation_step_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    visual_paths: list[str] = []

    foundation_feature_shapes_sample: list[list[int]] = []
    prob_output_shape_sample: list[int] = []
    raw_logits_shape_sample: list[int] = []

    optimizer_trainable_param_names: list[str] = []
    optimizer_trainable_param_count = 0
    optimizer_excludes_frozen_backbone = False
    frozen_backbone_param_count = 0

    total_train_ms = 0.0
    steps_completed = 0
    optimizer_step_success = True
    losses_finite = True
    outputs_finite = True
    gradients_finite = True
    feature_stage_contract_ok = True

    prob_nan_total = 0
    prob_inf_total = 0
    logits_nan_total = 0
    logits_inf_total = 0
    grad_nonfinite_total = 0
    grad_norm_l2_per_step: list[float] = []
    loss_total_per_step: list[float] = []
    pred_pos_ratio_per_step: list[float] = []
    pred_non_empty_rate_per_step: list[float] = []

    frozen_backbone_no_grad = True
    trainable_gradients_present = False
    trainable_gradients_finite = True
    latest_gradient_summary: dict[str, Any] = {}

    criterion_dice = DiceFocalLoss()
    criterion_bce_logits = torch.nn.BCEWithLogitsLoss()
    scheduled_val_steps = _validation_schedule(int(args.max_steps), int(args.val_every))
    executed_val_steps: list[int] = []

    if model is not None and hook_recorder is not None and selection["train"] and selection["val"]:
        model.train()
        if model.frozen_backbone:
            model.foundation_x.backbone.eval()

        optimizer, optimizer_trainable_param_names, frozen_backbone_param_count = (
            build_optimizer_for_trainable_params(
                model,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
            )
        )
        optimizer_trainable_param_count = len(optimizer_trainable_param_names)
        optimizer_excludes_frozen_backbone = all(
            not name.startswith("foundation_x.")
            for name in optimizer_trainable_param_names
        )
        if not optimizer_excludes_frozen_backbone:
            failures.append("Optimizer includes foundation_x parameters unexpectedly.")

        # Baseline validation at step 0.
        try:
            val_row, visual_paths = _evaluate_validation_point(
                model=model,
                hook_recorder=hook_recorder,
                val_cases=selection["val"],
                img_size=args.img_size,
                device=device,
                criterion_dice=criterion_dice,
                criterion_bce_logits=criterion_bce_logits,
                bce_loss_weight=float(args.bce_loss_weight),
                step_index=0,
                output_dir=args.output_dir,
                no_visuals=bool(args.no_visuals),
                visual_paths=visual_paths,
            )
            validation_step_rows.append(val_row)
            executed_val_steps.append(0)
        except Exception as exc:
            failures.append(f"Validation step 0 failed: {exc}")
            optimizer_step_success = False

        train_cases = selection["train"]
        batch_size = int(args.batch_size)

        for step_number in range(1, int(args.max_steps) + 1):
            if not optimizer_step_success:
                break

            start_index = ((step_number - 1) * batch_size) % len(train_cases)
            batch_indices = [(start_index + offset) % len(train_cases) for offset in range(batch_size)]
            batch_cases = [train_cases[idx] for idx in batch_indices]

            batch_x_tensors: list[torch.Tensor] = []
            batch_y_tensors: list[torch.Tensor] = []
            for case in batch_cases:
                image_tensor, mask_tensor = _load_case_tensor(case, args.img_size, device)
                batch_x_tensors.append(image_tensor)
                batch_y_tensors.append(mask_tensor)
            batch_x = torch.cat(batch_x_tensors, dim=0)
            batch_y = torch.cat(batch_y_tensors, dim=0)

            optimizer.zero_grad(set_to_none=True)
            hook_recorder.clear()
            step_start = time.time()

            probs = model(batch_x)
            raw_logits = hook_recorder.raw_logits
            if raw_logits is None:
                failures.append(f"Step {step_number}: raw logits hook capture is None.")
                optimizer_step_success = False
                break

            if not prob_output_shape_sample:
                prob_output_shape_sample = list(probs.shape)
            if not raw_logits_shape_sample:
                raw_logits_shape_sample = list(raw_logits.shape)

            foundation_output_shape = hook_recorder.last_records.get("foundation_x", {}).get(
                "output_shape",
                [],
            )
            stage_shapes: list[list[int]] = []
            if isinstance(foundation_output_shape, list):
                for item in foundation_output_shape:
                    if isinstance(item, list):
                        stage_shapes.append(item)
            if not foundation_feature_shapes_sample and stage_shapes:
                foundation_feature_shapes_sample = stage_shapes
            stage_ok, stage_errors = _stage_contract_ok(stage_shapes, args.img_size)
            if not stage_ok:
                feature_stage_contract_ok = False
                failures.extend([f"Step {step_number}: {err}" for err in stage_errors])

            if list(probs.shape) != list(batch_y.shape) or list(raw_logits.shape) != list(batch_y.shape):
                failures.append(
                    f"Step {step_number}: output/mask shape mismatch. "
                    f"prob={list(probs.shape)} logits={list(raw_logits.shape)} mask={list(batch_y.shape)}",
                )
                optimizer_step_success = False
                break

            prob_stats = _tensor_stats(probs)
            logits_stats = _tensor_stats(raw_logits)
            prob_nan_total += int(prob_stats["nan_count"])
            prob_inf_total += int(prob_stats["inf_count"])
            logits_nan_total += int(logits_stats["nan_count"])
            logits_inf_total += int(logits_stats["inf_count"])

            pos_ratio, non_empty_rate = _positive_ratio_and_non_empty_rate(probs)
            pred_pos_ratio_per_step.append(pos_ratio)
            pred_non_empty_rate_per_step.append(non_empty_rate)

            if (
                prob_stats["nan_count"] > 0
                or prob_stats["inf_count"] > 0
                or logits_stats["nan_count"] > 0
                or logits_stats["inf_count"] > 0
            ):
                outputs_finite = False
                losses_finite = False
                optimizer_step_success = False
                failures.append(
                    f"Step {step_number}: non-finite outputs prob_nan={prob_stats['nan_count']} "
                    f"prob_inf={prob_stats['inf_count']} logits_nan={logits_stats['nan_count']} "
                    f"logits_inf={logits_stats['inf_count']}",
                )
                break

            loss_dice = criterion_dice(probs, batch_y)
            loss_bce = criterion_bce_logits(raw_logits, batch_y)
            total_loss = loss_dice + float(args.bce_loss_weight) * loss_bce
            total_loss_value = float(total_loss.item())
            loss_total_per_step.append(total_loss_value)

            if not bool(torch.isfinite(total_loss).item()):
                losses_finite = False
                optimizer_step_success = False
                failures.append(f"Step {step_number}: total loss is non-finite ({total_loss_value}).")
                break

            total_loss.backward()
            step_gradient_rows, step_gradient_summary = _collect_gradient_rows(model, step_index=step_number)
            gradient_rows.extend(step_gradient_rows)
            latest_gradient_summary = step_gradient_summary
            grad_nonfinite_total += int(step_gradient_summary["grad_nonfinite_count"])
            grad_norm_l2_per_step.append(float(step_gradient_summary["grad_norm_l2_total"]))

            if step_gradient_summary["frozen_backbone_grad_present"] != 0 and model.frozen_backbone:
                frozen_backbone_no_grad = False
                failures.append(f"Step {step_number}: frozen backbone gradients detected.")

            if step_gradient_summary["trainable_non_backbone_grad_nonzero"] > 0:
                trainable_gradients_present = True
            if (
                step_gradient_summary["trainable_non_backbone_grad_finite"]
                < step_gradient_summary["trainable_non_backbone_grad_present"]
            ):
                gradients_finite = False
                trainable_gradients_finite = False
                failures.append(f"Step {step_number}: non-finite trainable gradients detected.")

            if step_gradient_summary["grad_nonfinite_count"] > 0:
                gradients_finite = False
                losses_finite = False
                optimizer_step_success = False
                failures.append(f"Step {step_number}: non-finite parameter gradients detected.")
                break

            optimizer.step()
            steps_completed += 1
            step_ms = (time.time() - step_start) * 1000.0
            total_train_ms += step_ms

            train_step_rows.append(
                {
                    "step": int(step_number),
                    "case_ids": "|".join(case["case_id"] for case in batch_cases),
                    "loss_total": total_loss_value,
                    "loss_dice_focal": float(loss_dice.item()),
                    "loss_bce_with_logits": float(loss_bce.item()),
                    "prob_min": prob_stats["min"],
                    "prob_max": prob_stats["max"],
                    "prob_mean": prob_stats["mean"],
                    "prob_std": prob_stats["std"],
                    "prob_nan_count": prob_stats["nan_count"],
                    "prob_inf_count": prob_stats["inf_count"],
                    "logits_min": logits_stats["min"],
                    "logits_max": logits_stats["max"],
                    "logits_mean": logits_stats["mean"],
                    "logits_std": logits_stats["std"],
                    "logits_nan_count": logits_stats["nan_count"],
                    "logits_inf_count": logits_stats["inf_count"],
                    "pred_pos_pixel_ratio": pos_ratio,
                    "pred_non_empty_rate": non_empty_rate,
                    "grad_norm_l2_total": step_gradient_summary["grad_norm_l2_total"],
                    "grad_nonfinite_count": step_gradient_summary["grad_nonfinite_count"],
                    "frozen_backbone_grad_present": step_gradient_summary["frozen_backbone_grad_present"],
                    "trainable_non_backbone_grad_present": step_gradient_summary["trainable_non_backbone_grad_present"],
                    "trainable_non_backbone_grad_nonzero": step_gradient_summary["trainable_non_backbone_grad_nonzero"],
                    "step_ms": round(step_ms, 3),
                },
            )

            if not args.no_visuals and len(visual_paths) < 12:
                try:
                    for vis_index, case in enumerate(batch_cases):
                        if len(visual_paths) >= 12:
                            break
                        written = _write_visuals(
                            args.output_dir,
                            "train",
                            case["case_id"],
                            batch_x[vis_index : vis_index + 1],
                            batch_y[vis_index : vis_index + 1],
                            probs[vis_index : vis_index + 1],
                            raw_logits[vis_index : vis_index + 1],
                        )
                        visual_paths.extend(written)
                except Exception as exc:
                    failures.append(f"Step {step_number}: visual write failed: {exc}")

            if step_number in scheduled_val_steps and step_number != 0:
                try:
                    val_row, visual_paths = _evaluate_validation_point(
                        model=model,
                        hook_recorder=hook_recorder,
                        val_cases=selection["val"],
                        img_size=args.img_size,
                        device=device,
                        criterion_dice=criterion_dice,
                        criterion_bce_logits=criterion_bce_logits,
                        bce_loss_weight=float(args.bce_loss_weight),
                        step_index=step_number,
                        output_dir=args.output_dir,
                        no_visuals=bool(args.no_visuals),
                        visual_paths=visual_paths,
                    )
                    validation_step_rows.append(val_row)
                    executed_val_steps.append(step_number)
                except Exception as exc:
                    failures.append(f"Validation step {step_number} failed: {exc}")
                    optimizer_step_success = False
                    break
    else:
        failures.append("Training skipped due to missing model or selected train/val cases.")
        optimizer_step_success = False

    _write_train_steps(train_step_rows, args.output_dir)
    _write_gradient_stats(gradient_rows, args.output_dir)

    # Flatten validation rows for csv output.
    validation_csv_rows: list[dict[str, Any]] = []
    for row in validation_step_rows:
        flat = {k: v for k, v in row.items() if k != "per_case_rows"}
        validation_csv_rows.append(flat)
    _write_validation_steps(validation_csv_rows, args.output_dir)
    progress_path = _save_progress_plot(args.output_dir, train_step_rows, validation_csv_rows)

    if hook_recorder is not None:
        hook_recorder.close()

    val_deg_flags = [bool(row["degenerate"]) for row in validation_csv_rows]
    any_degenerate = any(val_deg_flags)
    first_degenerate = bool(val_deg_flags[0]) if val_deg_flags else False
    persistent_collapse_from_start = bool(val_deg_flags) and first_degenerate and all(val_deg_flags)
    all_background_count = sum(int(row["all_background_collapse"]) for row in validation_csv_rows)
    all_foreground_count = sum(int(row["all_foreground_collapse"]) for row in validation_csv_rows)

    suspicious_loss_behavior = False
    if len(loss_total_per_step) >= 5 and loss_total_per_step[-1] > loss_total_per_step[0] * 1.2:
        suspicious_loss_behavior = True
    suspicious_signals = {
        "any_degenerate_validation_point": any_degenerate,
        "suspicious_loss_behavior": suspicious_loss_behavior,
    }

    pass_criteria = {
        "model_instantiates": model_instantiates,
        "checkpoint_loads": checkpoint_loads,
        "train_selection_ok": counts["train_total"] >= args.num_train_cases,
        "val_selection_ok": counts["val_total"] >= args.num_val_cases,
        "train_val_disjoint": counts["overlap_count"] == 0,
        "feature_stage_contract_ok": feature_stage_contract_ok,
        "max_steps_completed": steps_completed == int(args.max_steps),
        "optimizer_step_success": optimizer_step_success,
        "losses_finite": losses_finite and all(np.isfinite(loss_total_per_step)),
        "outputs_finite": outputs_finite and prob_nan_total == 0 and prob_inf_total == 0 and logits_nan_total == 0 and logits_inf_total == 0,
        "gradients_finite": gradients_finite and grad_nonfinite_total == 0,
        "frozen_backbone_no_grad": frozen_backbone_no_grad if (model is not None and model.frozen_backbone) else True,
        "trainable_gradients_present": trainable_gradients_present,
        "trainable_gradients_finite": trainable_gradients_finite,
        "optimizer_excludes_frozen_backbone": optimizer_excludes_frozen_backbone,
        "validation_schedule_complete": executed_val_steps == scheduled_val_steps,
        "validation_finite": all(
            row["prob_nan_total"] == 0
            and row["prob_inf_total"] == 0
            and row["logits_nan_total"] == 0
            and row["logits_inf_total"] == 0
            and np.isfinite(float(row["loss_total_mean"]))
            for row in validation_csv_rows
        ) if validation_csv_rows else False,
        "no_persistent_degeneracy": not persistent_collapse_from_start,
        "no_forbidden_writes": output_guardrail_ok,
    }
    status, exit_code = _compute_decision(
        pass_criteria=pass_criteria,
        suspicious_signals=suspicious_signals,
        strict=bool(args.strict),
    )

    if status == "PASS":
        next_recommended_pr = "PR-8 Larger Hybrid Training Run"
    elif status == "PARTIAL_PASS":
        next_recommended_pr = "Blocking fix PR"
    else:
        next_recommended_pr = "Blocking fix PR"

    command_render = (
        "py scripts/controlled_train_hybrid_short.py "
        f"--input_dir {args.input_dir} "
        f"--labels_dir {args.labels_dir} "
        f"--foundation_checkpoint {args.foundation_checkpoint} "
        f"--output_dir {args.output_dir} "
        f"--img_size {args.img_size} "
        f"--device {args.device} "
        f"--num_train_cases {args.num_train_cases} "
        f"--num_val_cases {args.num_val_cases} "
        f"--max_steps {args.max_steps} "
        f"--batch_size {args.batch_size} "
        f"--lr {args.lr} "
        f"--bce_loss_weight {args.bce_loss_weight} "
        f"--val_every {args.val_every} "
        + ("--strict " if args.strict else "")
        + ("--no_visuals " if args.no_visuals else "")
        + ("--save_diagnostic_checkpoint " if args.save_diagnostic_checkpoint else "")
        + ("--unfrozen_backbone " if args.unfrozen_backbone else "")
    ).strip()

    checkpoint_saved = False
    checkpoint_path = ""
    if args.save_diagnostic_checkpoint and model is not None:
        checkpoint_path = str(args.output_dir / CHECKPOINT_FILENAME)
        torch.save(
            {
                "diagnostic_only": True,
                "note": "PR-7 controlled short train checkpoint. Not for final training continuation or claims.",
                "schema_version": 1,
                "audit_name": "hybrid_controlled_short_train",
                "status": status,
                "step": int(steps_completed),
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        checkpoint_saved = True

    summary = {
        "schema_version": 1,
        "audit_name": "hybrid_controlled_short_train",
        "status": status,
        "strict": bool(args.strict),
        "limited_run": True,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "setup": {
            "command": command_render,
            "device": str(device),
            "input_dir": str(args.input_dir),
            "labels_dir": str(args.labels_dir),
            "foundation_checkpoint": str(args.foundation_checkpoint),
            "foundation_checkpoint_sha256": ckpt_sha,
            "foundation_checkpoint_size_bytes": ckpt_size,
            "frozen_backbone": not args.unfrozen_backbone,
            "img_size": int(args.img_size),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "bce_loss_weight": float(args.bce_loss_weight),
            "max_steps": int(args.max_steps),
            "val_every": int(args.val_every),
            "num_train_cases_requested": int(args.num_train_cases),
            "num_val_cases_requested": int(args.num_val_cases),
            "num_train_cases_processed": int(counts["train_total"]),
            "num_val_cases_processed": int(counts["val_total"]),
            "num_train_pos": int(counts["train_positive"]),
            "num_train_neg": int(counts["train_negative"]),
            "num_val_pos": int(counts["val_positive"]),
            "num_val_neg": int(counts["val_negative"]),
            "seed": int(args.seed),
        },
        "selection": {
            "train_case_ids": [case["case_id"] for case in selection["train"]],
            "val_case_ids": [case["case_id"] for case in selection["val"]],
            "counts": counts,
        },
        "training": {
            "steps_completed": int(steps_completed),
            "loss_total_per_step": loss_total_per_step,
            "grad_norm_l2_per_step": grad_norm_l2_per_step,
            "pred_pos_pixel_ratio_per_step": pred_pos_ratio_per_step,
            "pred_non_empty_rate_per_step": pred_non_empty_rate_per_step,
            "total_train_ms": float(total_train_ms),
            "optimizer_trainable_param_count": int(optimizer_trainable_param_count),
            "optimizer_trainable_param_names": optimizer_trainable_param_names,
            "frozen_backbone_param_count": int(frozen_backbone_param_count),
            "foundation_feature_shapes_sample": foundation_feature_shapes_sample,
            "prob_output_shape_sample": prob_output_shape_sample,
            "raw_logits_shape_sample": raw_logits_shape_sample,
            "prob_nan_total": int(prob_nan_total),
            "prob_inf_total": int(prob_inf_total),
            "logits_nan_total": int(logits_nan_total),
            "logits_inf_total": int(logits_inf_total),
            "grad_nonfinite_total": int(grad_nonfinite_total),
        },
        "validation": {
            "scheduled_steps": scheduled_val_steps,
            "executed_steps": executed_val_steps,
            "num_points": len(validation_csv_rows),
            "loss_total_mean_per_step": [float(row["loss_total_mean"]) for row in validation_csv_rows],
            "dice_mean_per_step": [float(row["dice_mean"]) for row in validation_csv_rows],
            "dice_pos_mean_per_step": [float(row["dice_pos_mean"]) for row in validation_csv_rows],
            "prob_min_per_step": [float(row["prob_min"]) for row in validation_csv_rows],
            "prob_max_per_step": [float(row["prob_max"]) for row in validation_csv_rows],
            "prob_mean_per_step": [float(row["prob_mean"]) for row in validation_csv_rows],
            "prob_std_per_step": [float(row["prob_std"]) for row in validation_csv_rows],
            "logits_min_per_step": [float(row["logits_min"]) for row in validation_csv_rows],
            "logits_max_per_step": [float(row["logits_max"]) for row in validation_csv_rows],
            "logits_mean_per_step": [float(row["logits_mean"]) for row in validation_csv_rows],
            "logits_std_per_step": [float(row["logits_std"]) for row in validation_csv_rows],
            "prob_foreground_ratio_thr05_per_step": [
                float(row["prob_foreground_ratio_thr05"]) for row in validation_csv_rows
            ],
            "prediction_non_empty_rate_per_step": [float(row["prediction_non_empty_rate"]) for row in validation_csv_rows],
            "pred_pos_pixel_ratio_mean_per_step": [float(row["pred_pos_pixel_ratio_mean"]) for row in validation_csv_rows],
        },
        "gradients": {
            "summary_last_step": latest_gradient_summary,
            "frozen_backbone_no_grad": pass_criteria["frozen_backbone_no_grad"],
            "trainable_gradients_present": pass_criteria["trainable_gradients_present"],
            "trainable_gradients_finite": pass_criteria["trainable_gradients_finite"],
            "optimizer_excludes_frozen_backbone": pass_criteria["optimizer_excludes_frozen_backbone"],
        },
        "degeneracy": {
            "any_degenerate_validation_point": any_degenerate,
            "persistent_collapse_from_start": persistent_collapse_from_start,
            "all_background_point_count": int(all_background_count),
            "all_foreground_point_count": int(all_foreground_count),
        },
        "pass_criteria": pass_criteria,
        "suspicious_signals": suspicious_signals,
        "outputs": {
            "summary_json": str(args.output_dir / "hybrid_controlled_short_train_summary.json"),
            "summary_yaml": str(args.output_dir / "hybrid_controlled_short_train_summary.yaml"),
            "report_md": str(args.output_dir / "hybrid_controlled_short_train_report.md"),
            "selection_log_csv": str(args.output_dir / "selection_log.csv"),
            "train_steps_csv": str(args.output_dir / "train_steps.csv"),
            "validation_steps_csv": str(args.output_dir / "validation_steps.csv"),
            "gradient_stats_csv": str(args.output_dir / "gradient_stats.csv"),
            "visuals_dir": str(args.output_dir / "visuals"),
            "visual_count": len(visual_paths),
            "progress_png": progress_path,
            "diagnostic_checkpoint_saved": checkpoint_saved,
            "diagnostic_checkpoint_path": checkpoint_path,
        },
        "warnings": [],
        "failures": failures,
        "next_recommended_pr": next_recommended_pr,
        "exit_code": int(exit_code),
    }

    if model_init_error:
        summary["warnings"].append(f"Model initialization error detail: {model_init_error}")

    safe_summary = _sanitize_for_json(summary)
    summary_path = args.output_dir / "hybrid_controlled_short_train_summary.json"
    summary_path.write_text(json.dumps(safe_summary, indent=2), encoding="utf-8")

    if _HAS_YAML:
        with (args.output_dir / "hybrid_controlled_short_train_summary.yaml").open(
            "w",
            encoding="utf-8",
        ) as handle:
            _yaml.dump(safe_summary, handle, default_flow_style=False, allow_unicode=True)

    _write_report(safe_summary, args.output_dir)
    return safe_summary


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run_controlled_short_training(args)
    return int(summary.get("exit_code", 2))


if __name__ == "__main__":
    sys.exit(main())
