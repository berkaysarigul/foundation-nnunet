"""
Hybrid adapter sanity test for Dataset101 nnU-Net heldout PNGs.

This script is diagnostic-only:
- no training loop
- one optional backward pass for gradient routing sanity
- no optimizer.step()
- no weight save

Usage:
    python scripts/sanity_hybrid_adapter.py \
      --input_dir nnUNet_raw/Dataset101_Pneumothorax/imagesTs \
      --labels_dir nnUNet_raw/Dataset101_Pneumothorax/heldout_labelsTs \
      --foundation_checkpoint checkpoints/foundation_x.pth \
      --output_dir artifacts/diagnostics/hybrid_adapter_sanity \
      --img_size 512 \
      --device auto \
      --num_cases 8 \
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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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


PRIORITY_FN_IDS = ["siim_009589", "siim_009791", "siim_009082", "siim_010023"]
PRIORITY_FP_IDS = ["siim_009416", "siim_009693", "siim_009266"]

STAGE_CHANNEL_EXPECT = [128, 256, 512, 1024]
STAGE_STRIDE_EXPECT = [4, 8, 16, 32]

FORBIDDEN_OUTPUT_SEGMENTS = [
    "artifacts/runs",
    "nnunet_results",
    "nnunet_results_smoke",
    "nnunet_raw/dataset101_pneumothorax",
    "artifacts/diagnostics/foundation_x_smoke",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid adapter sanity test using deterministic small-sample Dataset101 "
            "forward/loss/backward diagnostics."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("nnUNet_raw/Dataset101_Pneumothorax/imagesTs"),
        help="Directory of siim_NNNNNN_0000.png input images",
    )
    parser.add_argument(
        "--labels_dir",
        type=Path,
        default=Path("nnUNet_raw/Dataset101_Pneumothorax/heldout_labelsTs"),
        help="Directory of siim_NNNNNN.png binary labels",
    )
    parser.add_argument(
        "--foundation_checkpoint",
        type=Path,
        default=None,
        help="Path to Foundation X checkpoint (.pth)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Alias of --foundation_checkpoint for compatibility",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/diagnostics/hybrid_adapter_sanity"),
        help="Diagnostic output root",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        choices=[256, 512],
        help="Input size to hybrid model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto|cuda|cpu",
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=8,
        help="Number of deterministic cases (8-12 recommended)",
    )
    parser.add_argument(
        "--case_list",
        type=Path,
        default=None,
        help="Optional one-case-id-per-line file to override deterministic selection",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat any unmet pass criterion as BLOCKED except explicit gradient-only partial path",
    )
    parser.add_argument(
        "--no_visuals",
        action="store_true",
        help="Skip visual diagnostics",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve paths and case selection only",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Forward batch size for per-case sweep (default: 1)",
    )
    parser.add_argument(
        "--backward_batch_size",
        type=int,
        default=1,
        help="Batch size for single backward sanity pass (default: 1)",
    )
    parser.add_argument(
        "--unfrozen_backbone",
        action="store_true",
        help="Override default frozen backbone for diagnostic comparison",
    )
    return parser.parse_args(argv)


def _set_seed(seed: int = 42) -> None:
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
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _is_positive_mask(mask_path: Path) -> bool:
    try:
        arr = np.array(Image.open(mask_path).convert("L"))
        return int(arr.max()) > 0
    except Exception:
        return False


def _normalize_case_id(raw: str) -> str:
    case_id = raw.strip()
    if case_id.endswith("_0000"):
        case_id = case_id[: -len("_0000")]
    if case_id.endswith(".png"):
        case_id = case_id[: -len(".png")]
    return case_id


def select_cases(
    input_dir: Path,
    labels_dir: Path,
    num_cases: int,
    case_list: Path | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(case_id: str, source: str) -> bool:
        cid = _normalize_case_id(case_id)
        if cid in seen:
            return False
        image_path = input_dir / f"{cid}_0000.png"
        label_path = labels_dir / f"{cid}.png"
        if not image_path.exists():
            return False
        if not label_path.exists():
            return False
        label_max = int(np.array(Image.open(label_path).convert("L")).max())
        selected.append(
            {
                "case_id": cid,
                "source": source,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "label_present": True,
                "label_max": label_max,
            }
        )
        seen.add(cid)
        return True

    if case_list is not None:
        with case_list.open("r", encoding="utf-8") as fh:
            for line in fh:
                cid = line.strip()
                if cid:
                    _add(cid, "case_list")
        return selected[:num_cases]

    for cid in PRIORITY_FN_IDS:
        _add(cid, "priority_fn")
    for cid in PRIORITY_FP_IDS:
        _add(cid, "priority_fp")

    remaining = num_cases - len(selected)
    if remaining > 0:
        image_paths = sorted(input_dir.glob("siim_*_0000.png"))
        pos_pool: list[str] = []
        neg_pool: list[str] = []
        for image_path in image_paths:
            cid = _normalize_case_id(image_path.stem)
            if cid in seen:
                continue
            label_path = labels_dir / f"{cid}.png"
            if not label_path.exists():
                continue
            if _is_positive_mask(label_path):
                pos_pool.append(cid)
            else:
                neg_pool.append(cid)

        n_pos = (remaining + 1) // 2
        n_neg = remaining - n_pos
        for cid in pos_pool[:n_pos]:
            _add(cid, "filler_positive")
        for cid in neg_pool[:n_neg]:
            _add(cid, "filler_negative")

    return selected[:num_cases]


def _resolve_paths(args: argparse.Namespace) -> None:
    for attr in ("input_dir", "labels_dir", "output_dir"):
        p = getattr(args, attr)
        if p is not None and not p.is_absolute():
            setattr(args, attr, REPO_ROOT / p)

    if args.foundation_checkpoint is None and args.checkpoint is None:
        args.foundation_checkpoint = REPO_ROOT / Path("checkpoints/foundation_x.pth")
    elif args.foundation_checkpoint is None and args.checkpoint is not None:
        args.foundation_checkpoint = args.checkpoint
    elif args.foundation_checkpoint is not None and args.checkpoint is not None:
        if args.foundation_checkpoint != args.checkpoint:
            raise ValueError(
                "Conflicting checkpoint arguments: --foundation_checkpoint and --checkpoint differ."
            )

    if args.foundation_checkpoint is not None and not args.foundation_checkpoint.is_absolute():
        args.foundation_checkpoint = REPO_ROOT / args.foundation_checkpoint

    if args.case_list is not None and not args.case_list.is_absolute():
        args.case_list = REPO_ROOT / args.case_list


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
    return failures


def _validate_output_guardrail(output_dir: Path) -> tuple[bool, str]:
    normalized = output_dir.resolve().as_posix().lower()
    for segment in FORBIDDEN_OUTPUT_SEGMENTS:
        if segment in normalized and "hybrid_adapter_sanity" not in normalized:
            return (
                False,
                f"Output dir falls into forbidden location segment '{segment}': {output_dir}",
            )
    allowed_anchor = "artifacts/diagnostics/hybrid_adapter_sanity"
    if allowed_anchor not in normalized:
        return (
            False,
            "Output dir must be under artifacts/diagnostics/hybrid_adapter_sanity",
        )
    return True, ""


def _load_case_tensor(case: dict[str, Any], img_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image = np.array(Image.open(case["image_path"]).convert("L"), dtype=np.float32) / 255.0
    mask = np.array(Image.open(case["label_path"]).convert("L"), dtype=np.float32)
    mask = (mask > 0.5).astype(np.float32)

    if image.shape != (img_size, img_size) or mask.shape != (img_size, img_size):
        import cv2

        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    return image_tensor, mask_tensor


def _tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    tf = t.float()
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


def _shape_of(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        out: list[Any] = []
        for item in obj:
            out.append(_shape_of(item))
        return out
    return str(type(obj))


class HookRecorder:
    def __init__(self):
        self.last_records: dict[str, dict[str, Any]] = {}
        self.raw_logits: torch.Tensor | None = None
        self.handles: list[Any] = []

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
                if isinstance(output, torch.Tensor):
                    self.raw_logits = output
                else:
                    self.raw_logits = None

        handle = target.register_forward_hook(_hook)
        self.handles.append(handle)

    def clear(self) -> None:
        self.last_records = {}
        self.raw_logits = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _register_model_hooks(model: HybridFoundationUNet) -> HookRecorder:
    recorder = HookRecorder()
    recorder.add_hook(model, "foundation_x")
    recorder.add_hook(model, "fusion_e3")
    recorder.add_hook(model, "fusion_e4")
    recorder.add_hook(model, "h16_fx_context")
    recorder.add_hook(model, "h32_context_head")
    recorder.add_hook(model, "h32_to_h16")
    recorder.add_hook(model, "context_merge")
    recorder.add_hook(model, "dec4")
    recorder.add_hook(model, "dec3")
    recorder.add_hook(model, "dec2")
    recorder.add_hook(model, "dec1")
    recorder.add_hook(model, "final")
    return recorder


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
        _, c, h, w = shape
        exp_c = STAGE_CHANNEL_EXPECT[idx]
        exp_hw = img_size // STAGE_STRIDE_EXPECT[idx]
        if c != exp_c:
            errors.append(f"stage_{idx} channel mismatch: expected {exp_c}, got {c}")
        if h != exp_hw or w != exp_hw:
            errors.append(
                f"stage_{idx} spatial mismatch: expected {exp_hw}x{exp_hw}, got {h}x{w}"
            )
    return len(errors) == 0, errors


def _save_selection_log(cases: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "selection_log.csv"
    fields = ["case_id", "source", "image_path", "label_path", "label_present", "label_max"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in cases:
            writer.writerow({k: row[k] for k in fields})
    return path


def _write_forward_stats_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "per_case_forward_stats.csv"
    fieldnames = [
        "case_id",
        "input_shape",
        "mask_shape",
        "foundation_stage_shapes",
        "stage_contract_ok",
        "prob_output_shape",
        "raw_logits_shape",
        "shape_compatible_with_mask",
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
        "device",
        "dtype",
        "forward_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_gradient_stats_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "gradient_stats.csv"
    fieldnames = [
        "param_name",
        "requires_grad",
        "group",
        "grad_present",
        "grad_finite",
        "grad_nonzero",
        "grad_norm_l2",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


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
    case_id: str,
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    probs_tensor: torch.Tensor,
    logits_tensor: torch.Tensor,
) -> list[str]:
    case_dir = output_dir / "visuals" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    img = (image_tensor[0, 0].detach().cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    gt = (mask_tensor[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    prob = probs_tensor[0, 0].detach().cpu().numpy()
    logits = logits_tensor[0, 0].detach().cpu().numpy()
    pred_mask = (prob >= 0.5).astype(np.uint8) * 255

    prob_u8 = _normalize_to_uint8(prob)
    logits_u8 = _normalize_to_uint8(logits)
    overlay = _build_overlay(img, gt, pred_mask)

    paths = []
    for name, arr in (
        ("input.png", img),
        ("gt_mask.png", gt),
        ("probability_map.png", prob_u8),
        ("raw_logits_map.png", logits_u8),
        ("prediction_mask_thr05.png", pred_mask),
    ):
        path = case_dir / name
        Image.fromarray(arr).save(path)
        paths.append(str(path))

    overlay_path = case_dir / "overlay_input_gt_pred.png"
    Image.fromarray(overlay).save(overlay_path)
    paths.append(str(overlay_path))
    return paths


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


def _collect_gradient_stats(model: torch.nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = {
        "frozen_backbone_params": 0,
        "frozen_backbone_grad_present": 0,
        "trainable_non_backbone_params": 0,
        "trainable_non_backbone_grad_present": 0,
        "trainable_non_backbone_grad_finite": 0,
        "trainable_non_backbone_grad_nonzero": 0,
    }
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

        rows.append(
            {
                "param_name": name,
                "requires_grad": bool(param.requires_grad),
                "group": group,
                "grad_present": grad_present,
                "grad_finite": grad_finite,
                "grad_nonzero": grad_nonzero,
                "grad_norm_l2": grad_norm_l2,
            }
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

    return rows, summary


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


def _compute_decision(pass_criteria: dict[str, bool], strict: bool) -> tuple[str, int]:
    hard_blocking_keys = [
        "model_instantiates",
        "checkpoint_loads",
        "num_cases_ok",
        "forward_all_cases",
        "output_shape_compatible",
        "output_finite",
        "loss_finite",
        "backward_success",
        "no_forbidden_writes",
    ]
    for key in hard_blocking_keys:
        if not pass_criteria.get(key, False):
            return "BLOCKED", 2

    if not pass_criteria.get("frozen_backbone_no_grad", False):
        return ("BLOCKED", 2) if strict else ("PARTIAL_PASS", 0)
    if not pass_criteria.get("trainable_gradients_present", False):
        return ("BLOCKED", 2) if strict else ("PARTIAL_PASS", 0)

    if all(pass_criteria.values()):
        return "PASS", 0

    return ("BLOCKED", 2) if strict else ("PARTIAL_PASS", 0)


def _write_report(summary: dict[str, Any], output_dir: Path) -> Path:
    setup = summary["setup"]
    fwd = summary["forward"]
    grad = summary["gradients"]
    losses = summary["losses"]

    def _fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.6f}"
        except Exception:
            return str(value)

    lines: list[str] = [
        "# Hybrid Adapter Sanity Test Report",
        "",
        "## Repository findings",
        "",
        "- Existing hybrid model class: `src/models/hybrid.py::HybridFoundationUNet`",
        "- Existing backbone loader: `src/models/backbone.py::FoundationXBackbone`",
        "- Existing scale contract helper: `assert_corrected_hybrid_scale_contract()`",
        "- Existing related tests: `tests/test_hybrid_gradient_flow.py`, `tests/test_hybrid_scale_contract.py`",
        "- Existing Foundation X smoke artifacts found under `artifacts/diagnostics/foundation_x_smoke/`",
        "",
        "## Setup",
        "",
        "```",
        setup["command"],
        "```",
        "",
        f"- Device: `{setup['device']}`",
        f"- Number of cases requested: {setup['num_cases_requested']}",
        f"- Number of cases processed: {setup['num_cases_processed']}",
        f"- Input dir: `{setup['input_dir']}`",
        f"- Labels dir: `{setup['labels_dir']}`",
        f"- Foundation checkpoint: `{setup['foundation_checkpoint']}`",
        f"- Foundation checkpoint SHA-256: `{setup['foundation_checkpoint_sha256']}`",
        f"- Model config: frozen_backbone={setup['frozen_backbone']}, img_size={setup['img_size']}",
        "",
        "## Forward pass results",
        "",
        f"- Input shape sample: `{fwd['input_shape_sample']}`",
        f"- Foundation X feature shapes sample: `{fwd['foundation_feature_shapes_sample']}`",
        f"- Adapter/fusion shapes sample: `{fwd['adapter_fusion_shapes_sample']}`",
        f"- Output probability shape sample: `{fwd['prob_output_shape_sample']}`",
        f"- Raw logits shape sample (from model.final hook): `{fwd['raw_logits_shape_sample']}`",
        (
            f"- Probability stats aggregate: min={_fmt(fwd['prob_min_across_cases'])}, "
            f"max={_fmt(fwd['prob_max_across_cases'])}, mean={_fmt(fwd['prob_mean_across_cases'])}, "
            f"std={_fmt(fwd['prob_std_across_cases'])}"
        ),
        (
            f"- Raw logits stats aggregate: min={_fmt(fwd['logits_min_across_cases'])}, "
            f"max={_fmt(fwd['logits_max_across_cases'])}, mean={_fmt(fwd['logits_mean_across_cases'])}, "
            f"std={_fmt(fwd['logits_std_across_cases'])}"
        ),
        f"- NaN/Inf status: prob_nan_total={fwd['prob_nan_total']}, prob_inf_total={fwd['prob_inf_total']}, logits_nan_total={fwd['logits_nan_total']}, logits_inf_total={fwd['logits_inf_total']}",
        "",
        "## Loss and gradient results",
        "",
        "- Loss functions used: `DiceFocalLoss` on sigmoid output and `BCEWithLogitsLoss` on raw logits",
        f"- DiceFocal loss value: {losses['dice_focal_loss']}",
        f"- BCEWithLogits loss value: {losses['bce_with_logits_loss']}",
        f"- Total backward loss value: {losses['backward_total_loss']}",
        f"- Frozen backbone gradient check: {grad['frozen_backbone_no_grad']}",
        f"- Trainable adapter/decoder gradient check: {grad['trainable_gradients_present']}",
        f"- Finite gradient check: {grad['trainable_gradients_finite']}",
        "",
        "## Visual diagnostics",
        "",
        f"- Visual files saved: {summary['outputs']['visual_count']}",
        "- Brief note: diagnostic overlays/probability/logit maps are saved for non-empty/non-degenerate sanity only.",
        "- No performance claims are made in this report.",
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
            f"- Next recommended PR: `{summary['next_recommended_pr']}`",
        ]
    )

    path = output_dir / "hybrid_adapter_sanity_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_sanity(args: argparse.Namespace) -> dict[str, Any]:
    import time

    _set_seed(42)
    _resolve_paths(args)

    output_guardrail_ok, guardrail_error = _validate_output_guardrail(args.output_dir)
    if not output_guardrail_ok:
        blocked = {
            "schema_version": 1,
            "audit_name": "hybrid_adapter_sanity",
            "status": "BLOCKED",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "failures": [guardrail_error],
            "exit_code": 2,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "hybrid_adapter_sanity_summary.json").write_text(
            json.dumps(blocked, indent=2),
            encoding="utf-8",
        )
        return blocked

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = _validate_paths(args)
    if failures:
        blocked = {
            "schema_version": 1,
            "audit_name": "hybrid_adapter_sanity",
            "status": "BLOCKED",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "failures": failures,
            "exit_code": 2,
        }
        (args.output_dir / "hybrid_adapter_sanity_summary.json").write_text(
            json.dumps(blocked, indent=2),
            encoding="utf-8",
        )
        return blocked

    cases = select_cases(args.input_dir, args.labels_dir, args.num_cases, args.case_list)
    _save_selection_log(cases, args.output_dir)
    if len(cases) < args.num_cases:
        failures.append(
            f"Selected only {len(cases)} cases but num_cases={args.num_cases}. Missing labels or images."
        )

    if args.dry_run:
        summary = {
            "schema_version": 1,
            "audit_name": "hybrid_adapter_sanity",
            "status": "PARTIAL_PASS",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dry_run": True,
            "selection_count": len(cases),
            "selected_cases": [c["case_id"] for c in cases],
            "exit_code": 0,
        }
        (args.output_dir / "hybrid_adapter_sanity_summary.json").write_text(
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
    hook_recorder: HookRecorder | None = None
    model_init_error = ""
    try:
        model = _build_model(args).to(device)
        model.eval()
        model_instantiates = True
        checkpoint_loads = True
        hook_recorder = _register_model_hooks(model)
    except Exception as exc:
        model_init_error = str(exc)
        failures.append(f"Model instantiation/checkpoint load failed: {exc}")

    per_case_rows: list[dict[str, Any]] = []
    visual_paths: list[str] = []
    forward_all_cases = False
    output_shape_compatible = True
    output_finite = True
    stage_contract_ok = True
    input_shape_sample: list[int] = []
    foundation_feature_shapes_sample: list[list[int]] = []
    adapter_fusion_shapes_sample: dict[str, Any] = {}
    prob_output_shape_sample: list[int] = []
    raw_logits_shape_sample: list[int] = []

    forward_tensors_for_backward: list[tuple[torch.Tensor, torch.Tensor]] = []
    total_forward_ms = 0.0

    if model is not None and hook_recorder is not None:
        for case in cases:
            hook_recorder.clear()
            image_tensor, mask_tensor = _load_case_tensor(case, args.img_size, device)
            forward_tensors_for_backward.append((image_tensor, mask_tensor))

            if not input_shape_sample:
                input_shape_sample = list(image_tensor.shape)

            t0 = time.time()
            with torch.no_grad():
                probs = model(image_tensor)
            forward_ms = (time.time() - t0) * 1000.0
            total_forward_ms += forward_ms

            raw_logits = hook_recorder.raw_logits
            if raw_logits is None:
                failures.append(f"{case['case_id']}: raw logits hook did not capture model.final output")
                output_shape_compatible = False
                continue

            foundation_output_shape = hook_recorder.last_records.get("foundation_x", {}).get(
                "output_shape", []
            )
            stage_shapes: list[list[int]] = []
            if isinstance(foundation_output_shape, list):
                for item in foundation_output_shape:
                    if isinstance(item, list):
                        stage_shapes.append(item)

            current_stage_ok, stage_errors = _stage_contract_ok(stage_shapes, args.img_size)
            if not current_stage_ok:
                stage_contract_ok = False
                failures.extend([f"{case['case_id']}: {err}" for err in stage_errors])

            prob_shape = list(probs.shape)
            logits_shape = list(raw_logits.shape)
            mask_shape = list(mask_tensor.shape)
            shape_ok = prob_shape == mask_shape and logits_shape == mask_shape
            if not shape_ok:
                output_shape_compatible = False
                failures.append(
                    f"{case['case_id']}: output/mask shape mismatch. prob={prob_shape}, logits={logits_shape}, mask={mask_shape}"
                )

            prob_stats = _tensor_stats(probs)
            logits_stats = _tensor_stats(raw_logits)
            if (
                prob_stats["nan_count"] > 0
                or prob_stats["inf_count"] > 0
                or logits_stats["nan_count"] > 0
                or logits_stats["inf_count"] > 0
            ):
                output_finite = False
                failures.append(
                    f"{case['case_id']}: non-finite outputs prob_nan={prob_stats['nan_count']} prob_inf={prob_stats['inf_count']} logits_nan={logits_stats['nan_count']} logits_inf={logits_stats['inf_count']}"
                )

            if not foundation_feature_shapes_sample and stage_shapes:
                foundation_feature_shapes_sample = stage_shapes
            if not prob_output_shape_sample:
                prob_output_shape_sample = prob_shape
            if not raw_logits_shape_sample:
                raw_logits_shape_sample = logits_shape
            if not adapter_fusion_shapes_sample:
                keys = [
                    "fusion_e3",
                    "fusion_e4",
                    "h16_fx_context",
                    "h32_context_head",
                    "h32_to_h16",
                    "context_merge",
                    "dec4",
                    "dec3",
                    "dec2",
                    "dec1",
                ]
                adapter_fusion_shapes_sample = {
                    k: hook_recorder.last_records.get(k, {})
                    for k in keys
                    if k in hook_recorder.last_records
                }

            row = {
                "case_id": case["case_id"],
                "input_shape": str(list(image_tensor.shape)),
                "mask_shape": str(mask_shape),
                "foundation_stage_shapes": str(stage_shapes),
                "stage_contract_ok": current_stage_ok,
                "prob_output_shape": str(prob_shape),
                "raw_logits_shape": str(logits_shape),
                "shape_compatible_with_mask": shape_ok,
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
                "device": str(probs.device),
                "dtype": str(probs.dtype),
                "forward_ms": round(forward_ms, 3),
            }
            per_case_rows.append(row)

            if not args.no_visuals and len(visual_paths) < 24:
                try:
                    written = _write_visuals(
                        args.output_dir,
                        case["case_id"],
                        image_tensor,
                        mask_tensor,
                        probs,
                        raw_logits,
                    )
                    visual_paths.extend(written)
                except Exception as exc:
                    failures.append(f"{case['case_id']}: visual write failed: {exc}")

        forward_all_cases = len(per_case_rows) == len(cases) and len(cases) > 0
    else:
        failures.append("Forward skipped because model failed to instantiate.")

    _write_forward_stats_csv(per_case_rows, args.output_dir)

    loss_finite = False
    backward_success = False
    frozen_backbone_no_grad = False
    trainable_gradients_present = False
    trainable_gradients_finite = False
    gradient_rows: list[dict[str, Any]] = []
    gradient_summary: dict[str, Any] = {}
    dice_focal_loss_value = float("nan")
    bce_with_logits_loss_value = float("nan")
    backward_total_loss_value = float("nan")

    if model is not None and hook_recorder is not None and forward_tensors_for_backward:
        model.train()
        if model.frozen_backbone:
            model.foundation_x.backbone.eval()
        model.zero_grad(set_to_none=True)

        bsz = max(1, int(args.backward_batch_size))
        batch_items = forward_tensors_for_backward[:bsz]
        batch_x = torch.cat([item[0] for item in batch_items], dim=0)
        batch_y = torch.cat([item[1] for item in batch_items], dim=0)

        hook_recorder.clear()
        probs_batch = model(batch_x)
        raw_logits_batch = hook_recorder.raw_logits
        if raw_logits_batch is None:
            failures.append("Backward phase failed: raw logits hook capture is None.")
        else:
            if not bool(torch.isfinite(probs_batch).all().item()) or not bool(
                torch.isfinite(raw_logits_batch).all().item()
            ):
                failures.append(
                    "Backward phase skipped: non-finite tensors detected in probability or raw-logit outputs."
                )
                loss_finite = False
            else:
                try:
                    criterion_dice = DiceFocalLoss()
                    criterion_bce = torch.nn.BCEWithLogitsLoss()
                    loss_dice = criterion_dice(probs_batch, batch_y)
                    loss_bce = criterion_bce(raw_logits_batch, batch_y)
                    total_loss = loss_dice + loss_bce

                    dice_focal_loss_value = float(loss_dice.item())
                    bce_with_logits_loss_value = float(loss_bce.item())
                    backward_total_loss_value = float(total_loss.item())
                    loss_finite = bool(torch.isfinite(total_loss).item())

                    if not loss_finite:
                        failures.append(
                            f"Loss is non-finite: dice_focal={dice_focal_loss_value}, bce_with_logits={bce_with_logits_loss_value}"
                        )
                    else:
                        total_loss.backward()
                        backward_success = True
                except Exception as exc:
                    loss_finite = False
                    failures.append(f"Backward phase loss computation failed: {exc}")

            gradient_rows, gradient_summary = _collect_gradient_stats(model)
            frozen_backbone_no_grad = gradient_summary.get("frozen_backbone_grad_present", 1) == 0
            trainable_gradients_present = (
                gradient_summary.get("trainable_non_backbone_grad_nonzero", 0) > 0
            )
            trainable_gradients_finite = (
                gradient_summary.get("trainable_non_backbone_grad_finite", 0)
                >= gradient_summary.get("trainable_non_backbone_grad_present", 0)
            )

            if model.frozen_backbone and not frozen_backbone_no_grad:
                failures.append("Frozen Foundation X backbone received gradients unexpectedly.")
            if loss_finite and not trainable_gradients_present:
                failures.append("No non-backbone trainable parameter received non-zero gradients.")
            if loss_finite and not trainable_gradients_finite:
                failures.append("Detected non-finite gradients in non-backbone trainable parameters.")
    else:
        failures.append("Backward sanity skipped due to missing model or selected cases.")

    _write_gradient_stats_csv(gradient_rows, args.output_dir)

    if hook_recorder is not None:
        hook_recorder.close()

    prob_vals = [float(row["prob_mean"]) for row in per_case_rows if row["prob_nan_count"] == 0 and row["prob_inf_count"] == 0]
    prob_mins = [float(row["prob_min"]) for row in per_case_rows if row["prob_nan_count"] == 0 and row["prob_inf_count"] == 0]
    prob_maxs = [float(row["prob_max"]) for row in per_case_rows if row["prob_nan_count"] == 0 and row["prob_inf_count"] == 0]
    prob_stds = [float(row["prob_std"]) for row in per_case_rows if row["prob_nan_count"] == 0 and row["prob_inf_count"] == 0]
    logits_vals = [float(row["logits_mean"]) for row in per_case_rows if row["logits_nan_count"] == 0 and row["logits_inf_count"] == 0]
    logits_mins = [float(row["logits_min"]) for row in per_case_rows if row["logits_nan_count"] == 0 and row["logits_inf_count"] == 0]
    logits_maxs = [float(row["logits_max"]) for row in per_case_rows if row["logits_nan_count"] == 0 and row["logits_inf_count"] == 0]
    logits_stds = [float(row["logits_std"]) for row in per_case_rows if row["logits_nan_count"] == 0 and row["logits_inf_count"] == 0]

    prob_nan_total = sum(int(row["prob_nan_count"]) for row in per_case_rows)
    prob_inf_total = sum(int(row["prob_inf_count"]) for row in per_case_rows)
    logits_nan_total = sum(int(row["logits_nan_count"]) for row in per_case_rows)
    logits_inf_total = sum(int(row["logits_inf_count"]) for row in per_case_rows)

    pass_criteria = {
        "model_instantiates": model_instantiates,
        "checkpoint_loads": checkpoint_loads,
        "num_cases_ok": len(cases) >= args.num_cases and len(per_case_rows) >= args.num_cases,
        "forward_all_cases": forward_all_cases,
        "feature_stage_contract_ok": stage_contract_ok,
        "output_shape_compatible": output_shape_compatible,
        "output_finite": output_finite and prob_nan_total == 0 and prob_inf_total == 0 and logits_nan_total == 0 and logits_inf_total == 0,
        "loss_finite": loss_finite,
        "backward_success": backward_success,
        "frozen_backbone_no_grad": frozen_backbone_no_grad if (model is not None and model.frozen_backbone) else True,
        "trainable_gradients_present": trainable_gradients_present,
        "trainable_gradients_finite": trainable_gradients_finite,
        "no_forbidden_writes": output_guardrail_ok,
    }
    status, exit_code = _compute_decision(pass_criteria, strict=args.strict)

    next_pr = (
        "PR-6 Tiny Hybrid Training Smoke"
        if status == "PASS"
        else "Blocking fix PR for hybrid adapter/model"
    )

    command_render = (
        "python scripts/sanity_hybrid_adapter.py "
        f"--input_dir {args.input_dir} "
        f"--labels_dir {args.labels_dir} "
        f"--foundation_checkpoint {args.foundation_checkpoint} "
        f"--output_dir {args.output_dir} "
        f"--img_size {args.img_size} "
        f"--device {args.device} "
        f"--num_cases {args.num_cases} "
        + ("--strict " if args.strict else "")
        + ("--no_visuals " if args.no_visuals else "")
        + ("--unfrozen_backbone " if args.unfrozen_backbone else "")
    ).strip()

    summary = {
        "schema_version": 1,
        "audit_name": "hybrid_adapter_sanity",
        "status": status,
        "strict": bool(args.strict),
        "limited_run": True,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "setup": {
            "command": command_render,
            "device": str(device),
            "num_cases_requested": int(args.num_cases),
            "num_cases_processed": len(per_case_rows),
            "batch_size_forward": int(args.batch_size),
            "batch_size_backward": int(args.backward_batch_size),
            "input_dir": str(args.input_dir),
            "labels_dir": str(args.labels_dir),
            "foundation_checkpoint": str(args.foundation_checkpoint),
            "foundation_checkpoint_sha256": ckpt_sha,
            "foundation_checkpoint_size_bytes": ckpt_size,
            "frozen_backbone": not args.unfrozen_backbone,
            "img_size": int(args.img_size),
        },
        "selection": {
            "priority_fn_ids": PRIORITY_FN_IDS,
            "priority_fp_ids": PRIORITY_FP_IDS,
            "selected_case_ids": [c["case_id"] for c in cases],
        },
        "forward": {
            "input_shape_sample": input_shape_sample,
            "foundation_feature_shapes_sample": foundation_feature_shapes_sample,
            "adapter_fusion_shapes_sample": adapter_fusion_shapes_sample,
            "prob_output_shape_sample": prob_output_shape_sample,
            "raw_logits_shape_sample": raw_logits_shape_sample,
            "prob_min_across_cases": float(np.min(prob_mins)) if prob_mins else float("nan"),
            "prob_max_across_cases": float(np.max(prob_maxs)) if prob_maxs else float("nan"),
            "prob_mean_across_cases": float(np.mean(prob_vals)) if prob_vals else float("nan"),
            "prob_std_across_cases": float(np.mean(prob_stds)) if prob_stds else float("nan"),
            "logits_min_across_cases": float(np.min(logits_mins)) if logits_mins else float("nan"),
            "logits_max_across_cases": float(np.max(logits_maxs)) if logits_maxs else float("nan"),
            "logits_mean_across_cases": float(np.mean(logits_vals)) if logits_vals else float("nan"),
            "logits_std_across_cases": float(np.mean(logits_stds)) if logits_stds else float("nan"),
            "prob_nan_total": int(prob_nan_total),
            "prob_inf_total": int(prob_inf_total),
            "logits_nan_total": int(logits_nan_total),
            "logits_inf_total": int(logits_inf_total),
            "total_forward_ms": float(total_forward_ms),
        },
        "losses": {
            "dice_focal_loss": dice_focal_loss_value,
            "bce_with_logits_loss": bce_with_logits_loss_value,
            "backward_total_loss": backward_total_loss_value,
        },
        "gradients": {
            "summary": gradient_summary,
            "frozen_backbone_no_grad": pass_criteria["frozen_backbone_no_grad"],
            "trainable_gradients_present": pass_criteria["trainable_gradients_present"],
            "trainable_gradients_finite": pass_criteria["trainable_gradients_finite"],
        },
        "pass_criteria": pass_criteria,
        "outputs": {
            "summary_json": str(args.output_dir / "hybrid_adapter_sanity_summary.json"),
            "summary_yaml": str(args.output_dir / "hybrid_adapter_sanity_summary.yaml"),
            "report_md": str(args.output_dir / "hybrid_adapter_sanity_report.md"),
            "selection_log_csv": str(args.output_dir / "selection_log.csv"),
            "forward_stats_csv": str(args.output_dir / "per_case_forward_stats.csv"),
            "gradient_stats_csv": str(args.output_dir / "gradient_stats.csv"),
            "visuals_dir": str(args.output_dir / "visuals"),
            "visual_count": len(visual_paths),
        },
        "warnings": [],
        "failures": failures,
        "next_recommended_pr": next_pr,
        "exit_code": exit_code,
    }

    if model_init_error:
        summary["warnings"].append(f"Model initialization error detail: {model_init_error}")

    safe_summary = _sanitize_for_json(summary)
    summary_path = args.output_dir / "hybrid_adapter_sanity_summary.json"
    summary_path.write_text(json.dumps(safe_summary, indent=2), encoding="utf-8")

    if _HAS_YAML:
        with (args.output_dir / "hybrid_adapter_sanity_summary.yaml").open(
            "w", encoding="utf-8"
        ) as fh:
            _yaml.dump(safe_summary, fh, default_flow_style=False, allow_unicode=True)

    _write_report(safe_summary, args.output_dir)
    return safe_summary


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run_sanity(args)
    return int(summary.get("exit_code", 2))


if __name__ == "__main__":
    sys.exit(main())
