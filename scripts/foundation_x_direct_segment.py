"""
PR-9A: Foundation X direct segmentation reproduction / head sweep on
Dataset101_Pneumothorax.

Purpose
-------
- Instantiate the full Foundation X model (Swin-B + PPN + FPN +
  segmentation_heads) as a best-effort reverse-engineered reconstruction
  from the checkpoint key inventory.
- Load the released Foundation X checkpoint (default state-dict key:
  ``model``; sanity-comparable to ``teacher_model``).
- Run direct segmentation inference for each candidate binary head and
  identify the head that best matches SIIM-ACR pneumothorax on the
  Dataset101 heldout split.
- Save a reproducible artifact bundle so the full run can later be
  executed on Colab GPU.

This script does NOT train. It runs forward passes only, inside
``torch.no_grad()``, with ``model.eval()``. No optimizer is built.

Safety / scope rules
~~~~~~~~~~~~~~~~~~~~
- Default device is ``cpu``. ``--device cuda`` must be set explicitly.
- A local CPU dry-run MUST pass ``--max-cases`` (e.g. ``--max-cases 2``).
  If ``--device cpu`` is selected without ``--max-cases``, the script
  refuses to run the full 1602-case sweep on CPU to protect the local
  machine.
- Outputs are written under ``artifacts/diagnostics/foundation_x_direct/``.

Framing boundary (D-035 / D-040–D-042)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Foundation X is a SIIM-exposed pretraining source and is deferred from
  the main paper path. Any number this script produces is leakage-aware
  diagnostic evidence and must be compared back to the trusted
  ``pretrained_resnet34_unet`` baseline (D-042: 0.4951).
- The head-to-task mapping is NOT officially confirmed in this script.
- This script is a diagnostic head sweep; ``best head + threshold`` is
  reported as the best fit on the evaluated cases, not as the official
  SIIM head.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
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
    import yaml as _yaml  # type: ignore

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


ALLOWED_OUTPUT_ANCHOR = "artifacts/diagnostics/foundation_x_direct"
DEFAULT_BINARY_HEAD_INDICES = (0, 1, 2, 3, 5)
KNOWN_MULTICLASS_HEAD_INDICES = (4,)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Direct Foundation X segmentation inference and head sweep on "
            "Dataset101 heldout. No training. Saves reproducible artifacts."
        ),
    )
    parser.add_argument(
        "--dataset-root", dest="dataset_root", type=Path, required=True,
        help="Path to nnUNet_raw/Dataset101_Pneumothorax (root containing imagesTs/, heldout_labelsTs/, ...).",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to Foundation X .pth checkpoint (e.g. checkpoints/foundation_x.pth or ckpt_E896_TH15.pth).",
    )
    parser.add_argument(
        "--state-key", dest="state_key", type=str, default="model",
        choices=("model", "teacher_model"),
        help="Which state-dict key to load from the checkpoint. Default: 'model' (per checkpoint README).",
    )
    parser.add_argument(
        "--heads", type=int, nargs="+", default=list(DEFAULT_BINARY_HEAD_INDICES),
        help="Indices of segmentation heads to evaluate. Default: 0 1 2 3 5.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=("cpu", "cuda", "auto"),
        help="Inference device. Default: cpu. 'auto' picks cuda if available.",
    )
    parser.add_argument(
        "--max-cases", dest="max_cases", type=int, default=0,
        help="If > 0, evaluate at most this many cases (required for CPU dry-runs).",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=None,
        help="Probability thresholds to sweep. Default: 0.05..0.95 step 0.05.",
    )
    parser.add_argument(
        "--save-prob-maps", dest="save_prob_maps", type=_strtobool, default=False,
        help="Save per-case probability maps (uint8 percentile-normalized PNG).",
    )
    parser.add_argument(
        "--save-binary-masks", dest="save_binary_masks", type=_strtobool, default=False,
        help="Save per-case binary masks at the per-head best threshold.",
    )
    parser.add_argument(
        "--save-visuals", dest="save_visuals", type=_strtobool, default=False,
        help="Save per-head qualitative grids (image + GT + prob + binary).",
    )
    parser.add_argument(
        "--img-size", dest="img_size", type=int, default=512, choices=(256, 512),
        help="Spatial size used by the model. Default: 512.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output directory. Must be under artifacts/diagnostics/foundation_x_direct/.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Deterministic seed for case ordering.",
    )
    parser.add_argument(
        "--num-visuals", dest="num_visuals", type=int, default=8,
        help="Number of cases to include per visual grid (positives + negatives mixed).",
    )
    parser.add_argument(
        "--print-keys", dest="print_keys", action="store_true",
        help="Print a short checkpoint key summary at startup (diagnostic).",
    )
    return parser.parse_args(argv)


def _strtobool(value: str) -> bool:
    """Permissive bool parser for CLI flags ('true'/'1'/'yes' or 'false'/'0'/'no')."""
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "on", "t"}:
        return True
    if s in {"false", "0", "no", "n", "off", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected truthy/falsy string, got: {value!r}")


# ─── Guardrails ──────────────────────────────────────────────────────────────


def validate_output_guardrail(output_dir: Path) -> tuple[bool, str]:
    """Reject outputs outside artifacts/diagnostics/foundation_x_direct/."""
    norm = output_dir.resolve().as_posix().lower()
    if ALLOWED_OUTPUT_ANCHOR not in norm:
        return False, f"Output dir must be under {ALLOWED_OUTPUT_ANCHOR}: {output_dir}"
    forbidden = ("artifacts/runs", "nnunet_results", "nnunet_raw")
    for seg in forbidden:
        if seg in norm:
            return False, f"Output dir falls into forbidden segment '{seg}': {output_dir}"
    return True, ""


def validate_local_run_guardrail(device: str, max_cases: int) -> tuple[bool, str]:
    """Refuse full CPU runs without explicit --max-cases."""
    if device == "cpu" and max_cases <= 0:
        return False, (
            "Refusing to run a full Foundation X CPU sweep without --max-cases. "
            "Pass --max-cases 1 or --max-cases 2 for a local dry-run, or use --device cuda."
        )
    return True, ""


def _resolve_device(device_arg: str) -> Any:
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# ─── Dataset loading (Dataset101 nnU-Net v2 convention) ─────────────────────


def _normalize_case_id(raw: str) -> str:
    case_id = raw.strip()
    if case_id.endswith(".png"):
        case_id = case_id[: -len(".png")]
    if case_id.endswith("_0000"):
        case_id = case_id[: -len("_0000")]
    return case_id


def list_heldout_cases(images_dir: Path, labels_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for image_path in sorted(images_dir.glob("*_0000.png")):
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
    if not cases:
        # Fallback for non-nnU-Net naming
        for image_path in sorted(images_dir.glob("*.png")):
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


def select_cases(cases: list[dict[str, Any]], max_cases: int, seed: int) -> list[dict[str, Any]]:
    """If max_cases <= 0, return all. Otherwise prefer a positive/negative mix when known."""
    if max_cases <= 0 or max_cases >= len(cases):
        return cases
    # Cheap pos/neg inference from the heldout label (read at selection time would be expensive;
    # so we just take the deterministic first N cases — caller may rely on dry-run sanity).
    rng = np.random.default_rng(seed)
    if max_cases <= 2:
        # For dry-runs prefer the head of the list for stable testing
        return cases[:max_cases]
    indices = rng.choice(len(cases), size=max_cases, replace=False)
    return [cases[int(i)] for i in sorted(indices.tolist())]


def _load_image_label(
    image_path: Path,
    label_path: Path,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, bool, tuple[int, int]]:
    """Returns (image_arr [0,1] HxW, mask_arr {0,1} HxW, label_exists, original_hw)."""
    from PIL import Image

    pil_img = Image.open(image_path).convert("L")
    original_hw = (pil_img.size[1], pil_img.size[0])  # (H, W)
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
    return image_arr, mask_arr, label_exists, original_hw


# ─── Metric helpers ─────────────────────────────────────────────────────────


def per_case_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray) -> dict[str, float]:
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
    n = len(rows)
    rows_with_label = [r for r in rows if r.get("label_available")]
    n_with_label = len(rows_with_label)

    gt_pos = [r for r in rows_with_label if r["gt_is_positive"]]
    gt_neg = [r for r in rows_with_label if not r["gt_is_positive"]]
    pred_pos = [r for r in rows_with_label if r["pred_is_positive"]]

    tp = sum(1 for r in rows_with_label if r["gt_is_positive"] and r["pred_is_positive"])
    fn = sum(1 for r in rows_with_label if r["gt_is_positive"] and not r["pred_is_positive"])
    fp = sum(1 for r in rows_with_label if (not r["gt_is_positive"]) and r["pred_is_positive"])
    tn = sum(1 for r in rows_with_label if (not r["gt_is_positive"]) and (not r["pred_is_positive"]))

    def _safe(a: float, b: float) -> float:
        return float(a) / float(b) if b > 0 else 0.0

    case_precision = _safe(tp, tp + fp)
    case_recall = _safe(tp, tp + fn)
    case_specificity = _safe(tn, tn + fp)
    case_f1 = _safe(2 * case_precision * case_recall, case_precision + case_recall)

    def _mean(vs: list[float]) -> float:
        return float(statistics.fmean(vs)) if vs else float("nan")

    dice_all = [r["dice"] for r in rows_with_label]
    iou_all = [r["iou"] for r in rows_with_label]
    dice_pos = [r["dice"] for r in gt_pos]
    detected = [r for r in gt_pos if r["pred_is_positive"]]
    dice_detected = [r["dice"] for r in detected]
    neg_fp = [r for r in gt_neg if r["pred_is_positive"]]

    return {
        "total_cases": n,
        "cases_with_label": n_with_label,
        "gt_positive_cases": len(gt_pos),
        "gt_negative_cases": len(gt_neg),
        "pred_positive_cases": len(pred_pos),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "case_level_precision": case_precision,
        "case_level_recall": case_recall,
        "case_level_specificity": case_specificity,
        "case_level_f1": case_f1,
        "mean_dice_all_cases": _mean(dice_all),
        "mean_iou_all_cases": _mean(iou_all),
        "mean_dice_positive_cases": _mean(dice_pos),
        "mean_dice_detected_positives_only": _mean(dice_detected),
        "negative_case_false_positive_rate": _safe(len(neg_fp), len(gt_neg)),
        "detected_positive_count": len(detected),
    }


# ─── Checkpoint loading ─────────────────────────────────────────────────────


def _sha256_of_file(path: Path, max_bytes: int = 0) -> str:
    """Best-effort SHA-256 of a file. If max_bytes > 0, only hashes the prefix."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if max_bytes > 0:
            h.update(f.read(max_bytes))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def load_checkpoint_state_dict(
    checkpoint_path: Path,
    state_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load the checkpoint and select the desired state dict.

    Returns (state_dict, ckpt_info). `ckpt_info` includes top_level_keys,
    selected_state_key, available state_dict keys when relevant.
    """
    import torch

    obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    info: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "top_level_type": type(obj).__name__,
    }
    if isinstance(obj, dict):
        info["top_level_keys"] = list(obj.keys())
        if state_key not in obj:
            raise KeyError(
                f"--state-key '{state_key}' not found in checkpoint top-level keys "
                f"{info['top_level_keys']}"
            )
        state = obj[state_key]
        info["selected_state_key"] = state_key
    else:
        # Raw state dict
        info["top_level_keys"] = []
        info["selected_state_key"] = "raw"
        state = obj

    if not isinstance(state, dict):
        raise TypeError(
            f"Selected state under '{state_key}' is not a dict; got {type(state).__name__}"
        )

    info["state_dict_len"] = len(state)
    # Lightweight prefix summary
    prefix_counter: dict[str, int] = {}
    for k in state:
        p = k.split(".")[0]
        prefix_counter[p] = prefix_counter.get(p, 0) + 1
    info["top_prefix_counts"] = prefix_counter
    info["selected_state_key_top_keys_sample"] = list(state.keys())[:10]
    return state, info


# ─── Inference ──────────────────────────────────────────────────────────────


def run_head_inference(
    model: Any,
    image_tensor: Any,
    head_idx: int,
) -> np.ndarray:
    """
    Forward pass for a single image and head. Returns a probability map
    in [0, 1] as a numpy array of shape (H, W).
    """
    import torch

    with torch.no_grad():
        logits = model.forward_segmentation(
            image_tensor,
            head_idx=head_idx,
            upsample_to_input=True,
        )
        if logits.shape[1] != 1:
            raise AssertionError(
                f"Head {head_idx} produced {logits.shape[1]}-channel output; "
                f"this head sweep expects binary heads only."
            )
        prob = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()
    return prob.astype(np.float32)


def threshold_grid(thresholds: list[float] | None) -> list[float]:
    if thresholds is None:
        return [round(0.05 + 0.05 * i, 2) for i in range(19)]  # 0.05..0.95
    return sorted({round(float(t), 4) for t in thresholds if 0.0 < float(t) < 1.0})


# ─── Visualization ──────────────────────────────────────────────────────────


def _save_uint8_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def _percentile_normalize(arr: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def save_visual_grid(
    grid_path: Path,
    cases: list[dict[str, Any]],
    head_idx: int,
    threshold: float,
    img_size: int,
) -> bool:
    """Write a compact grid PNG: rows = cases, cols = image / GT / prob / binary."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    n = len(cases)
    if n == 0:
        return False
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = np.array([axes])
    for row_idx, c in enumerate(cases):
        img = c["_image_for_visual"]
        gt = c["_gt_for_visual"]
        prob = c["_prob_for_visual"]
        binary = (prob > threshold).astype(np.uint8) * 255

        axes[row_idx, 0].imshow(img, cmap="gray")
        axes[row_idx, 0].set_title(f"{c['case_id']} image")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(gt, cmap="gray")
        axes[row_idx, 1].set_title("GT mask")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(prob, cmap="magma", vmin=0.0, vmax=1.0)
        axes[row_idx, 2].set_title(f"head {head_idx} prob")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(binary, cmap="gray")
        axes[row_idx, 3].set_title(f"binary @ {threshold:.2f}")
        axes[row_idx, 3].axis("off")
    fig.tight_layout()
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(grid_path), dpi=80)
    plt.close(fig)
    return True


# ─── Report writers ─────────────────────────────────────────────────────────


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


def _write_yaml_or_json(payload: dict[str, Any], yaml_path: Path, json_fallback: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_YAML:
        with open(yaml_path, "w", encoding="utf-8") as f:
            _yaml.safe_dump(_sanitize_for_json(payload), f, sort_keys=False)
    else:
        with open(json_fallback, "w", encoding="utf-8") as f:
            json.dump(_sanitize_for_json(payload), f, indent=2)


def write_per_case_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    # Stable ordering: identity columns first
    leading = ["case_id", "head_idx", "threshold", "label_available",
               "gt_is_positive", "pred_is_positive",
               "dice", "iou", "precision", "recall",
               "intersection_pixels", "pred_foreground_pixels",
               "gt_foreground_pixels", "total_pixels"]
    ordered = [k for k in leading if k in keys] + [k for k in keys if k not in leading]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in ordered})


def write_report_md(
    out_path: Path,
    summary: dict[str, Any],
    ckpt_info: dict[str, Any],
    load_diag: dict[str, Any],
    head_stats: dict[int, dict[str, Any]],
    skipped_heads: list[int],
    failure_log: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Foundation X Direct Segmentation — Head Sweep Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- This is **direct Foundation X segmentation inference**, not training.")
    lines.append("- This run is a **head sweep** to identify which segmentation head best matches")
    lines.append("  SIIM-ACR pneumothorax on Dataset101 heldout.")
    lines.append("- The official Foundation X README states that the best SIIM-ACR segmentation")
    lines.append("  model is `(Student) 896`, so `checkpoint[\"model\"]` is the primary state dict.")
    lines.append("- `checkpoint[\"teacher_model\"]` may be evaluated separately but should not")
    lines.append("  replace `model` unless it scores better and is explicitly documented.")
    lines.append("- Head 4 (`segmentation_heads.4.weight` shape `(13, 128, 3, 3)`) is multiclass")
    lines.append("  (likely ChestX-Det) and is **skipped** from the binary head sweep.")
    lines.append("- No claim is made that the best head is the official SIIM head. The mapping")
    lines.append("  must be confirmed from the Foundation X config or repository code before it")
    lines.append("  is treated as authoritative (D-035 framing boundary).")
    lines.append("")
    lines.append("## Framing boundary (D-035 / D-040–D-042)")
    lines.append("")
    lines.append("Foundation X is a SIIM-exposed pretraining source under the current checkpoint")
    lines.append("provenance. Any number below is leakage-aware diagnostic evidence and must be")
    lines.append("compared back to the trusted `pretrained_resnet34_unet` baseline")
    lines.append("(D-042 anchor: held-out positive-only Dice mean **0.4951**).")
    lines.append("")
    lines.append("## Run setup")
    lines.append("")
    lines.append(f"- checkpoint: `{ckpt_info.get('checkpoint_path')}`")
    if "checkpoint_sha256" in ckpt_info:
        lines.append(f"- checkpoint SHA-256: `{ckpt_info['checkpoint_sha256']}`")
    lines.append(f"- selected state key: `{ckpt_info.get('selected_state_key')}`")
    lines.append(f"- top-level keys: `{ckpt_info.get('top_level_keys')}`")
    lines.append(f"- dataset root: `{summary.get('dataset_root')}`")
    lines.append(f"- cases evaluated: {summary.get('cases_evaluated')}")
    lines.append(f"- heads evaluated: `{summary.get('heads_evaluated')}`")
    lines.append(f"- heads skipped (non-binary): `{skipped_heads}`")
    lines.append(f"- thresholds: `{summary.get('thresholds_evaluated')}`")
    lines.append(f"- device: `{summary.get('device')}`")
    lines.append(f"- img_size: `{summary.get('img_size')}`")
    lines.append("")
    lines.append("## Checkpoint load diagnostics")
    lines.append("")
    rs = load_diag.get("route_stats", {})
    lines.append(f"- raw_total: {rs.get('raw_total')}")
    lines.append(f"- raw_with_prefix `backbone.0.`: {rs.get('raw_with_prefix')}")
    lines.append(f"- routed_swin: {rs.get('routed_swin')}")
    lines.append(f"- routed_segnorm: {rs.get('routed_segnorm')}")
    lines.append(f"- routed_locnorm: {rs.get('routed_locnorm')}")
    lines.append(f"- routed_ppn: {rs.get('routed_ppn')}")
    lines.append(f"- routed_fpn: {rs.get('routed_fpn')}")
    lines.append(f"- routed_heads: {rs.get('routed_heads')}")
    lines.append(f"- skipped_non_backbone (e.g. teacher_model, optimizer, epoch): {rs.get('skipped_non_backbone')}")
    lines.append(f"- skipped_classification_heads: {rs.get('skipped_classification_heads')}")
    lines.append(f"- skipped_swin_head: {rs.get('skipped_swin_head')}")
    lines.append(f"- missing keys (model expected but checkpoint did not provide): {len(load_diag.get('missing', []))}")
    lines.append(f"- unexpected keys (checkpoint had, model does not declare): {len(load_diag.get('unexpected', []))}")
    if load_diag.get("missing"):
        lines.append("- sample missing keys:")
        for k in load_diag["missing"][:10]:
            lines.append(f"  - `{k}`")
    if load_diag.get("unexpected"):
        lines.append("- sample unexpected keys:")
        for k in load_diag["unexpected"][:10]:
            lines.append(f"  - `{k}`")
    lines.append("")
    lines.append("## Head-by-head results")
    lines.append("")
    lines.append("| head | best threshold | pos Dice (pos cases) | det. pos Dice | mean Dice all | case F1 | case recall | case spec | neg FPR |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for head_idx in sorted(head_stats):
        st = head_stats[head_idx]
        best = st.get("best_by_pos_dice", {})
        m = best.get("metrics", {})
        lines.append(
            "| {h} | {t} | {pd} | {dd} | {ma} | {f1} | {r} | {s} | {fpr} |".format(
                h=head_idx,
                t=f"{best.get('threshold', float('nan')):.2f}",
                pd=_fmt(m.get("mean_dice_positive_cases")),
                dd=_fmt(m.get("mean_dice_detected_positives_only")),
                ma=_fmt(m.get("mean_dice_all_cases")),
                f1=_fmt(m.get("case_level_f1")),
                r=_fmt(m.get("case_level_recall")),
                s=_fmt(m.get("case_level_specificity")),
                fpr=_fmt(m.get("negative_case_false_positive_rate")),
            )
        )
    lines.append("")
    best_overall = summary.get("best_overall")
    if best_overall:
        lines.append("## Best overall (by mean Dice on positive cases)")
        lines.append("")
        lines.append(f"- head: **{best_overall['head_idx']}**")
        lines.append(f"- threshold: **{best_overall['threshold']:.2f}**")
        lines.append(f"- positive-cases mean Dice: **{_fmt(best_overall.get('positive_dice_mean'))}**")
        lines.append("")
        lines.append("> The 'best head' label here is a fit to the evaluated subset only.")
        lines.append("> It is NOT confirmed to be the official SIIM-ACR pneumothorax head until")
        lines.append("> the head-to-task mapping is read from the Foundation X repository.")
        lines.append("")
    if failure_log:
        lines.append("## Failures / warnings")
        lines.append("")
        for line in failure_log:
            lines.append(f"- {line}")
        lines.append("")
    lines.append("## Local dry-run command")
    lines.append("")
    lines.append("```")
    lines.append("python scripts/foundation_x_direct_segment.py \\")
    lines.append("  --dataset-root nnUNet_raw/Dataset101_Pneumothorax \\")
    lines.append("  --checkpoint checkpoints/ckpt_E896_TH15.pth \\")
    lines.append("  --state-key model \\")
    lines.append("  --heads 0 1 2 3 5 \\")
    lines.append("  --device cpu \\")
    lines.append("  --max-cases 2 \\")
    lines.append("  --out artifacts/diagnostics/foundation_x_direct/dryrun_cpu \\")
    lines.append("  --save-visuals true")
    lines.append("```")
    lines.append("")
    lines.append("## Intended Colab full-run command (DO NOT run locally)")
    lines.append("")
    lines.append("```")
    lines.append("python scripts/foundation_x_direct_segment.py \\")
    lines.append("  --dataset-root nnUNet_raw/Dataset101_Pneumothorax \\")
    lines.append("  --checkpoint checkpoints/ckpt_E896_TH15.pth \\")
    lines.append("  --state-key model \\")
    lines.append("  --heads 0 1 2 3 5 \\")
    lines.append("  --device cuda \\")
    lines.append("  --out artifacts/diagnostics/foundation_x_direct/head_sweep \\")
    lines.append("  --save-prob-maps true \\")
    lines.append("  --save-binary-masks true \\")
    lines.append("  --save-visuals true")
    lines.append("```")
    lines.append("")
    lines.append("## Reverse-engineering caveats")
    lines.append("")
    lines.append("- `src/models/foundation_x_full.py` is a **best-effort reverse-engineered")
    lines.append("  reconstruction** of the Foundation X model from the checkpoint key shapes.")
    lines.append("  The forward graph for `segmentation_PPN` and `segmentation_FPN` is")
    lines.append("  inferred from the parameter shapes (PSPNet-style PPM + UPerNet-style top-down")
    lines.append("  FPN with conv_fusion) and may differ in details from the official")
    lines.append("  JLiangLab/Foundation_X implementation.")
    lines.append("- If the load diagnostic above reports a small `missing` set covering only")
    lines.append("  optional Swin/timm-versioning keys and a `routed_*` count matching the")
    lines.append("  inventory in `foundationx_checkpoint_inventory.txt`, the segmentation path")
    lines.append("  is loaded; otherwise the inference numbers should be treated as")
    lines.append("  diagnostic-only.")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return "n/a"
        return f"{x:.4f}"
    return str(x)


# ─── Main ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    ok, why = validate_output_guardrail(args.out)
    if not ok:
        print(f"[guardrail] {why}", file=sys.stderr)
        return 2

    ok, why = validate_local_run_guardrail(args.device, args.max_cases)
    if not ok:
        print(f"[guardrail] {why}", file=sys.stderr)
        return 2

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact command line for reproducibility.
    (out_dir / "command.txt").write_text(
        " ".join(sys.argv), encoding="utf-8",
    )

    images_dir = args.dataset_root / "imagesTs"
    labels_dir = args.dataset_root / "heldout_labelsTs"
    if not images_dir.exists() or not labels_dir.exists():
        print(f"[error] expected imagesTs and heldout_labelsTs under {args.dataset_root}", file=sys.stderr)
        return 3

    cases = list_heldout_cases(images_dir, labels_dir)
    cases = select_cases(cases, args.max_cases, args.seed)
    if not cases:
        print(f"[error] no cases found in {images_dir}", file=sys.stderr)
        return 3

    thresholds = threshold_grid(args.thresholds)

    # Split heads into binary and known-multiclass; reject anything else explicitly.
    selected_heads = list(args.heads)
    skipped_heads = [h for h in selected_heads if h in KNOWN_MULTICLASS_HEAD_INDICES]
    binary_heads = [h for h in selected_heads if h not in KNOWN_MULTICLASS_HEAD_INDICES]
    failure_log: list[str] = []
    if skipped_heads:
        failure_log.append(
            f"Skipped multiclass head(s) {skipped_heads}; the binary sweep only runs heads "
            f"{binary_heads}."
        )

    import torch

    device = _resolve_device(args.device)
    print(f"[info] device: {device}")
    print(f"[info] cases selected: {len(cases)}")
    print(f"[info] binary heads to sweep: {binary_heads}")
    print(f"[info] thresholds: {thresholds}")

    # Load checkpoint
    state_dict, ckpt_info = load_checkpoint_state_dict(args.checkpoint, args.state_key)
    try:
        ckpt_info["checkpoint_sha256"] = _sha256_of_file(args.checkpoint, max_bytes=64 * 1024 * 1024)
        ckpt_info["checkpoint_sha256_scope"] = "first_64MB_prefix"
    except Exception as exc:
        failure_log.append(f"Could not compute checkpoint SHA-256: {exc}")

    if args.print_keys:
        print(f"[ckpt] top-level keys: {ckpt_info.get('top_level_keys')}")
        print(f"[ckpt] selected state key: {ckpt_info.get('selected_state_key')}")
        print(f"[ckpt] state dict size: {ckpt_info.get('state_dict_len')}")

    # Build model
    from src.models.foundation_x_full import FoundationXFullNet  # noqa: WPS433

    try:
        model = FoundationXFullNet(img_size=args.img_size)
    except Exception as exc:
        failure_log.append(f"FoundationXFullNet instantiation failed: {exc}")
        _emit_failure_artifacts(out_dir, args, ckpt_info, failure_log)
        print(f"[error] model instantiation failed: {exc}", file=sys.stderr)
        return 4

    # Load weights
    load_diag = model.load_foundation_x_state_dict(state_dict)
    model.to(device).eval()
    print(
        f"[ckpt] routed: swin={load_diag['route_stats']['routed_swin']} "
        f"segnorm={load_diag['route_stats']['routed_segnorm']} "
        f"ppn={load_diag['route_stats']['routed_ppn']} "
        f"fpn={load_diag['route_stats']['routed_fpn']} "
        f"heads={load_diag['route_stats']['routed_heads']}"
    )
    print(
        f"[ckpt] missing={len(load_diag['missing'])} unexpected={len(load_diag['unexpected'])}"
    )

    # Forward sanity: shape probe at the configured img_size for each binary head.
    head_output_shapes: dict[int, list[int]] = {}
    with torch.no_grad():
        probe = torch.zeros(1, 1, args.img_size, args.img_size, dtype=torch.float32, device=device)
        for h in binary_heads:
            out = model.forward_segmentation(probe, head_idx=h, upsample_to_input=True)
            head_output_shapes[h] = list(out.shape)
    print(f"[shape] head output shapes (after upsample): {head_output_shapes}")

    # Run inference for each case and each binary head: cache the prob map and gt.
    per_case_rows: list[dict[str, Any]] = []
    head_case_predictions: dict[int, dict[str, dict[str, Any]]] = {h: {} for h in binary_heads}
    visual_buffer: dict[int, list[dict[str, Any]]] = {h: [] for h in binary_heads}
    inference_seconds = 0.0

    case_index = 0
    for case in cases:
        case_index += 1
        image_arr, gt_arr, label_exists, original_hw = _load_image_label(
            case["image_path"], case["label_path"], args.img_size,
        )
        tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to(device)
        gt_pos = bool(gt_arr.sum() > 0)
        case_dirs_prob = out_dir / "prob_maps"
        case_dirs_bin = out_dir / "binary_masks"
        for h in binary_heads:
            t0 = time.perf_counter()
            prob = run_head_inference(model, tensor, head_idx=h)
            inference_seconds += time.perf_counter() - t0

            if args.save_prob_maps:
                _save_uint8_png(
                    _percentile_normalize(prob),
                    case_dirs_prob / f"head_{h}" / f"{case['case_id']}.png",
                )

            head_case_predictions[h][case["case_id"]] = {
                "prob": prob,
                "gt": gt_arr,
                "label_exists": label_exists,
                "gt_is_positive": gt_pos,
            }
            if args.save_visuals and len(visual_buffer[h]) < args.num_visuals:
                visual_buffer[h].append(
                    {
                        "case_id": case["case_id"],
                        "_image_for_visual": image_arr,
                        "_gt_for_visual": gt_arr,
                        "_prob_for_visual": prob,
                    },
                )
        if case_index % 10 == 0:
            print(f"[infer] case {case_index}/{len(cases)}")

    # Per-head threshold sweep + per-case metric rows.
    head_stats: dict[int, dict[str, Any]] = {}
    for h in binary_heads:
        per_threshold_summary: dict[float, dict[str, Any]] = {}
        for thr in thresholds:
            rows = []
            for cid, pred_pack in head_case_predictions[h].items():
                pred_bin = (pred_pack["prob"] > thr).astype(np.uint8)
                gt_bin = (pred_pack["gt"] > 0.5).astype(np.uint8)
                pc = per_case_metrics(pred_bin, gt_bin)
                row = {
                    "case_id": cid,
                    "head_idx": h,
                    "threshold": thr,
                    "label_available": pred_pack["label_exists"],
                    "gt_is_positive": pred_pack["gt_is_positive"],
                    "pred_is_positive": int(pred_bin.sum()) > 0,
                    **pc,
                }
                rows.append(row)
                per_case_rows.append(row)
            agg = aggregate_case_metrics(rows)
            per_threshold_summary[thr] = agg

        # Best threshold by positive-cases mean Dice.
        def _score(t: float) -> float:
            v = per_threshold_summary[t].get("mean_dice_positive_cases")
            return float("-inf") if v is None or (isinstance(v, float) and math.isnan(v)) else v

        best_thr = max(thresholds, key=_score) if thresholds else 0.5
        head_stats[h] = {
            "thresholds_summary": {str(t): per_threshold_summary[t] for t in thresholds},
            "best_by_pos_dice": {
                "threshold": best_thr,
                "metrics": per_threshold_summary[best_thr],
            },
            "shape": head_output_shapes.get(h),
        }

        # Optional binary masks at best threshold per head
        if args.save_binary_masks:
            for cid, pred_pack in head_case_predictions[h].items():
                bin_mask = (pred_pack["prob"] > best_thr).astype(np.uint8) * 255
                _save_uint8_png(
                    bin_mask,
                    out_dir / "binary_masks" / f"head_{h}" / f"{cid}.png",
                )

        # Optional visual grid at the best threshold
        if args.save_visuals and visual_buffer[h]:
            save_visual_grid(
                out_dir / "visual_grids" / f"head_{h}_best_thr_{best_thr:.2f}.png",
                visual_buffer[h],
                head_idx=h,
                threshold=best_thr,
                img_size=args.img_size,
            )

    # Build summary.yaml payload
    best_overall = None
    if head_stats:
        best_overall_head = max(
            head_stats,
            key=lambda h: (head_stats[h]["best_by_pos_dice"]["metrics"].get("mean_dice_positive_cases") or -1.0),
        )
        best_overall_thr = head_stats[best_overall_head]["best_by_pos_dice"]["threshold"]
        best_overall = {
            "head_idx": int(best_overall_head),
            "threshold": float(best_overall_thr),
            "positive_dice_mean": head_stats[best_overall_head]["best_by_pos_dice"]["metrics"].get(
                "mean_dice_positive_cases"
            ),
        }

    summary_payload: dict[str, Any] = {
        "audit_name": "foundation_x_direct_head_sweep",
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "framing_notice": (
            "Foundation X is SIIM-exposed (D-006). Numbers are leakage-aware diagnostic "
            "evidence. Compare back to the trusted pretrained_resnet34_unet baseline "
            "(D-042 anchor: 0.4951). Head-to-task mapping not officially confirmed."
        ),
        "checkpoint_path": ckpt_info.get("checkpoint_path"),
        "checkpoint_sha256": ckpt_info.get("checkpoint_sha256"),
        "checkpoint_sha256_scope": ckpt_info.get("checkpoint_sha256_scope"),
        "selected_state_key": ckpt_info.get("selected_state_key"),
        "ckpt_top_level_keys": ckpt_info.get("top_level_keys"),
        "dataset_root": str(args.dataset_root),
        "cases_evaluated": len(cases),
        "thresholds_evaluated": thresholds,
        "heads_evaluated": binary_heads,
        "skipped_multiclass_heads": skipped_heads,
        "img_size": args.img_size,
        "device": str(device),
        "inference_seconds_total": inference_seconds,
        "head_output_shapes": head_output_shapes,
        "load_diagnostics": load_diag,
        "per_head_best_threshold": {
            int(h): {
                "threshold": head_stats[h]["best_by_pos_dice"]["threshold"],
                "metrics": head_stats[h]["best_by_pos_dice"]["metrics"],
            }
            for h in head_stats
        },
        "best_overall": best_overall,
        "failure_log": failure_log,
    }
    _write_yaml_or_json(
        summary_payload,
        out_dir / "summary.yaml",
        out_dir / "summary.json",
    )
    # Always emit a JSON copy too so downstream tools without yaml can parse it.
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(summary_payload), f, indent=2)

    write_per_case_csv(per_case_rows, out_dir / "per_case_metrics.csv")

    # run_metadata.json
    run_metadata = {
        "schema_version": 1,
        "tool": "scripts/foundation_x_direct_segment.py",
        "command": " ".join(sys.argv),
        "generated_at": summary_payload["generated_at"],
        "args": _sanitize_for_json(vars(args)),
        "load_route_stats": load_diag.get("route_stats"),
        "missing_keys_count": len(load_diag.get("missing", [])),
        "unexpected_keys_count": len(load_diag.get("unexpected", [])),
        "framing_notice": summary_payload["framing_notice"],
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(run_metadata), f, indent=2)

    # report.md
    write_report_md(
        out_dir / "report.md",
        summary=summary_payload,
        ckpt_info=ckpt_info,
        load_diag=load_diag,
        head_stats=head_stats,
        skipped_heads=skipped_heads,
        failure_log=failure_log,
    )

    print(f"[done] artifacts written under {out_dir}")
    return 0


def _emit_failure_artifacts(
    out_dir: Path,
    args: argparse.Namespace,
    ckpt_info: dict[str, Any],
    failure_log: list[str],
) -> None:
    """Emit minimal diagnostic artifacts when the model cannot be instantiated."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failure_payload = {
        "audit_name": "foundation_x_direct_head_sweep",
        "schema_version": 1,
        "status": "BLOCKED",
        "framing_notice": (
            "Model instantiation failed. No inference was run; no metrics were computed."
        ),
        "checkpoint_info": ckpt_info,
        "failure_log": failure_log,
        "args": _sanitize_for_json(vars(args)),
        "required_to_unblock": [
            "Confirm the timm version supports swin_base_patch4_window7_224 with the configured img_size.",
            "If the reverse-engineered FoundationXFullNet shape contract diverges from the released "
            "checkpoint, vendor the official JLiangLab/Foundation_X model code under third_party/ and "
            "swap it in via the same load_foundation_x_state_dict entrypoint.",
            "Verify the checkpoint state-dict key ('model' vs 'teacher_model').",
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(failure_payload), f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
