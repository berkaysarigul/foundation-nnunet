"""
Foundation X smoke test: load checkpoint, run inference on a small deterministic
sample of Dataset101_Pneumothorax images, and write diagnostic artifacts.

Usage:
    python scripts/smoke_foundation_x.py \\
      --input_dir nnUNet_raw/Dataset101_Pneumothorax/imagesTs \\
      --labels_dir nnUNet_raw/Dataset101_Pneumothorax/heldout_labelsTs \\
      --checkpoint checkpoints/foundation_x.pth \\
      --output_dir artifacts/diagnostics/foundation_x_smoke \\
      --img_size 512 \\
      --device auto \\
      --num_cases 12

Exit codes:
    0  PASS or PARTIAL_PASS
    2  BLOCKED (checkpoint missing, model fails to load, too few cases)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import hashlib
import io
import json
import math
import random
import sys
import warnings
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

from src.models.backbone import FoundationXBackbone, _remap_key


# ─── Constants ────────────────────────────────────────────────────────────────

PRIORITY_FN_IDS = ["siim_009589", "siim_009791", "siim_009082", "siim_010023"]
PRIORITY_FP_IDS = ["siim_009416", "siim_009693", "siim_009266"]

STAGE_CHANNEL_EXPECT = [128, 256, 512, 1024]
STAGE_STRIDE_EXPECT = [4, 8, 16, 32]

_BACKBONE_PREFIX = "backbone.0."


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Foundation X smoke test: load checkpoint, run inference on a small "
            "deterministic sample of Dataset101 images, write diagnostic artifacts."
        ),
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("nnUNet_raw/Dataset101_Pneumothorax/imagesTs"),
        metavar="DIR",
        help="Directory of siim_NNNNNN_0000.png images",
    )
    p.add_argument(
        "--labels_dir",
        type=Path,
        default=Path("nnUNet_raw/Dataset101_Pneumothorax/heldout_labelsTs"),
        metavar="DIR",
        help="Directory of siim_NNNNNN.png binary labels (0/1 uint8) for positive/negative balance",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/foundation_x.pth"),
        metavar="PTH",
        help="Path to Foundation X .pth checkpoint",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/diagnostics/foundation_x_smoke"),
        metavar="DIR",
        help="Directory where all diagnostic artifacts are written",
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=512,
        choices=[256, 512],
        help="Input size passed to FoundationXBackbone (256 or 512, default: 512)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', or 'cpu' (default: auto)",
    )
    p.add_argument(
        "--num_cases",
        type=int,
        default=12,
        help="Total number of cases to process (default: 12)",
    )
    p.add_argument(
        "--case_list",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional text file with one case_id per line; overrides heuristic selection",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Promote warnings to failures (unexpected_keys_count > 500 → BLOCKED)",
    )
    p.add_argument(
        "--no_visuals",
        action="store_true",
        help="Skip visual diagnostic generation",
    )
    p.add_argument(
        "--save_raw_features",
        action="store_true",
        help="Also save raw .npy feature tensors per stage per case (large output, off by default)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print case selection and checkpoint metadata, then exit without running inference",
    )
    return p.parse_args(argv)


# ─── Utilities ────────────────────────────────────────────────────────────────


def _set_seed() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


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
    """Recursively convert NaN/Inf floats to None for JSON safety (D-039: no hausdorff)."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _label_is_positive(label_path: Path) -> bool:
    try:
        return int(np.array(Image.open(label_path).convert("L")).max()) == 1
    except Exception:
        return False


# ─── Case selection ───────────────────────────────────────────────────────────


def select_cases(
    input_dir: Path,
    labels_dir: Path,
    num_cases: int,
    case_list: Path | None,
) -> list[dict[str, Any]]:
    """
    Deterministic case selection.
    Returns list of dicts: {case_id, source, image_path, label_path, label_present, label_max}.
    """
    selected: list[dict] = []
    seen: set[str] = set()

    def _add(case_id: str, source: str) -> bool:
        if case_id in seen:
            return False
        img_path = input_dir / f"{case_id}_0000.png"
        if not img_path.exists():
            print(f"[case-select] Priority ID not found, skipping: {img_path}", file=sys.stderr)
            return False
        label_path = labels_dir / f"{case_id}.png"
        label_present = label_path.exists()
        label_max = int(np.array(Image.open(label_path).convert("L")).max()) if label_present else -1
        seen.add(case_id)
        selected.append(
            {
                "case_id": case_id,
                "source": source,
                "image_path": str(img_path),
                "label_path": str(label_path) if label_present else "",
                "label_present": label_present,
                "label_max": label_max,
            }
        )
        return True

    if case_list is not None:
        with case_list.open("r", encoding="utf-8") as fh:
            ids = [line.strip() for line in fh if line.strip()]
        for cid in ids:
            _add(cid, "case_list")
        return selected[:num_cases]

    for cid in PRIORITY_FN_IDS:
        _add(cid, "priority_fn")
    for cid in PRIORITY_FP_IDS:
        _add(cid, "priority_fp")

    remaining = num_cases - len(selected)
    if remaining > 0:
        all_imgs = sorted(input_dir.glob("siim_*_0000.png"))
        pos_queue: list[Path] = []
        neg_queue: list[Path] = []
        for img_path in all_imgs:
            cid = img_path.name.replace("_0000.png", "")
            if cid in seen:
                continue
            label_path = labels_dir / f"{cid}.png"
            if label_path.exists() and _label_is_positive(label_path):
                pos_queue.append(img_path)
            else:
                neg_queue.append(img_path)

        n_pos_need = (remaining + 1) // 2
        n_neg_need = remaining - n_pos_need
        for img_path in pos_queue[:n_pos_need]:
            _add(img_path.name.replace("_0000.png", ""), "filler_positive")
        for img_path in neg_queue[:n_neg_need]:
            _add(img_path.name.replace("_0000.png", ""), "filler_negative")

    return selected[:num_cases]


# ─── Checkpoint metadata (first load — freed before model construction) ────────


def _load_checkpoint_metadata(checkpoint: Path) -> dict:
    print(f"[metadata] SHA-256 of {checkpoint.name} ({checkpoint.stat().st_size:,} bytes) ...")
    sha256 = _compute_sha256(checkpoint)
    print(f"[metadata] SHA-256: {sha256}")

    print("[metadata] Loading checkpoint for metadata extraction (map_location=cpu) ...")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    top_level_keys = sorted(ckpt.keys())
    model_weights = ckpt.get("model", {})
    state_dict_len = len(model_weights)
    prefix_keys = [k for k in model_weights if k.startswith(_BACKBONE_PREFIX)]
    prefix_match_count = len(prefix_keys)
    sample_prefix_keys = sorted(prefix_keys)[:5]
    remapped_keys = {_remap_key(k[len(_BACKBONE_PREFIX):]) for k in prefix_keys}

    del ckpt, model_weights
    gc.collect()
    print(f"[metadata] Freed. prefix_match_count={prefix_match_count}")

    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "top_level_keys": top_level_keys,
        "state_dict_len": state_dict_len,
        "backbone_prefix": _BACKBONE_PREFIX,
        "prefix_match_count": prefix_match_count,
        "sample_prefix_keys": sample_prefix_keys,
        "_remapped_keys": remapped_keys,  # popped before serialization
    }


# ─── Model loading ────────────────────────────────────────────────────────────


def _load_model_and_count_mismatches(
    checkpoint: Path,
    img_size: int,
    device: torch.device,
    remapped_keys: set[str],
) -> tuple[FoundationXBackbone, list[str], int, int]:
    """
    Construct FoundationXBackbone (re-loads checkpoint internally), capture warnings,
    and compute missing/unexpected key counts vs. the timm model state dict.
    """
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = FoundationXBackbone(str(checkpoint), frozen=True, img_size=img_size)
    captured_warnings = [str(warning.message) for warning in w]

    model = model.to(device)
    model.eval()

    timm_keys = set(model.backbone.state_dict().keys())
    missing_count = len(timm_keys - remapped_keys)
    unexpected_count = len(remapped_keys - timm_keys)

    return model, captured_warnings, missing_count, unexpected_count


# ─── Per-case inference ───────────────────────────────────────────────────────


def _load_image(image_path: Path, img_size: int, device: torch.device) -> torch.Tensor:
    """Load a grayscale PNG as (1, 1, img_size, img_size) float32 [0,1] tensor."""
    arr = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    if arr.shape != (img_size, img_size):
        try:
            import cv2
            arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            pil_img = Image.fromarray((arr * 255).astype(np.uint8)).resize(
                (img_size, img_size), Image.BILINEAR
            )
            arr = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def _stage_stats(feat: torch.Tensor, stage_idx: int, img_size: int, case_id: str) -> dict:
    """Compute diagnostic stats for a single (1, C, H, W) stage feature tensor."""
    C_expect = STAGE_CHANNEL_EXPECT[stage_idx]
    stride = STAGE_STRIDE_EXPECT[stage_idx]
    H_expect = img_size // stride
    shape = list(feat.shape)
    shapes_match = (
        len(shape) == 4
        and shape[1] == C_expect
        and shape[2] == H_expect
        and shape[3] == H_expect
    )

    f = feat.float()
    nan_count = int(torch.isnan(f).sum().item())
    inf_count = int(torch.isinf(f).sum().item())
    fin = f[~(torch.isnan(f) | torch.isinf(f))]
    if fin.numel() > 0:
        fmin = float(fin.min().item())
        fmax = float(fin.max().item())
        fmean = float(fin.mean().item())
        fstd = float(fin.std().item())
    else:
        fmin = fmax = fmean = fstd = float("nan")

    return {
        "case_id": case_id,
        "stage": stage_idx,
        "expected_shape": [1, C_expect, H_expect, H_expect],
        "observed_shape": shape,
        "shapes_match_contract": shapes_match,
        "min": fmin,
        "max": fmax,
        "mean": fmean,
        "std": fstd,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def _aggregate_stage_stats(all_case_stats: list[list[dict]]) -> dict:
    """Aggregate per-case per-stage stats across all successfully processed cases."""
    aggregated: dict = {}
    for stage_idx in range(4):
        stage_list = [cs[stage_idx] for cs in all_case_stats if len(cs) > stage_idx]
        all_means = [s["mean"] for s in stage_list if not math.isnan(s["mean"])]
        all_stds = [s["std"] for s in stage_list if not math.isnan(s["std"])]
        all_mins = [s["min"] for s in stage_list if not math.isnan(s["min"])]
        all_maxs = [s["max"] for s in stage_list if not math.isnan(s["max"])]
        shapes_ok = all(s["shapes_match_contract"] for s in stage_list) if stage_list else False
        sample_observed = stage_list[0]["observed_shape"] if stage_list else []
        expected_shape = stage_list[0]["expected_shape"] if stage_list else []
        aggregated[f"stage_{stage_idx}"] = {
            "expected_shape": expected_shape,
            "observed_shape_sample": sample_observed,
            "shapes_match_contract": shapes_ok,
            "mean_across_cases": float(np.mean(all_means)) if all_means else float("nan"),
            "std_across_cases": float(np.mean(all_stds)) if all_stds else float("nan"),
            "min_across_cases": float(np.min(all_mins)) if all_mins else float("nan"),
            "max_across_cases": float(np.max(all_maxs)) if all_maxs else float("nan"),
            "nan_count_total": sum(s["nan_count"] for s in stage_list),
            "inf_count_total": sum(s["inf_count"] for s in stage_list),
        }
    return aggregated


# ─── Decision ─────────────────────────────────────────────────────────────────


def _compute_decision(
    model_load_ok: bool,
    num_cases_processed: int,
    num_cases_requested: int,
    per_stage_agg: dict,
    unexpected_keys_count: int,
    strict: bool,
) -> tuple[str, int, dict[str, bool]]:
    pass_criteria: dict[str, bool] = {
        "checkpoint_loads": model_load_ok,
        "num_cases_ok": num_cases_processed >= num_cases_requested,
        "shapes_match_contract": all(
            v.get("shapes_match_contract", False) for v in per_stage_agg.values()
        ),
        "features_finite": all(
            v.get("nan_count_total", 1) == 0 and v.get("inf_count_total", 1) == 0
            for v in per_stage_agg.values()
        ),
        "features_non_degenerate": all(
            not (
                abs(v.get("mean_across_cases") or 0.0) < 1e-6
                and (v.get("std_across_cases") or 0.0) < 1e-4
            )
            for v in per_stage_agg.values()
        ),
    }

    if not pass_criteria["checkpoint_loads"] or not pass_criteria["num_cases_ok"]:
        return "BLOCKED", 2, pass_criteria
    if strict and unexpected_keys_count > 500:
        pass_criteria["strict_unexpected_keys_ok"] = False
        return "BLOCKED", 2, pass_criteria

    if all(pass_criteria.values()):
        return "PASS", 0, pass_criteria
    return "PARTIAL_PASS", 0, pass_criteria


# ─── Output writers ───────────────────────────────────────────────────────────


def _write_selection_log(cases: list[dict], output_dir: Path) -> Path:
    path = output_dir / "selection_log.csv"
    fields = ["case_id", "source", "image_path", "label_path", "label_present", "label_max"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in cases:
            w.writerow({k: c[k] for k in fields})
    return path


def _write_per_case_stats(
    all_case_stats: list[list[dict]],
    infer_ms_per_case: list[float],
    output_dir: Path,
) -> Path:
    path = output_dir / "per_case_stats.csv"
    fields = ["case_id", "stage", "observed_shape", "min", "max", "mean", "std",
              "nan_count", "inf_count", "infer_ms"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for case_idx, case_stats in enumerate(all_case_stats):
            if not case_stats:
                continue
            ms = infer_ms_per_case[case_idx]
            for s in case_stats:
                def _fmt(v: float) -> str:
                    return f"{v:.6f}" if not math.isnan(v) else "nan"
                w.writerow({
                    "case_id": s["case_id"],
                    "stage": s["stage"],
                    "observed_shape": str(s["observed_shape"]),
                    "min": _fmt(s["min"]),
                    "max": _fmt(s["max"]),
                    "mean": _fmt(s["mean"]),
                    "std": _fmt(s["std"]),
                    "nan_count": s["nan_count"],
                    "inf_count": s["inf_count"],
                    "infer_ms": f"{ms:.1f}",
                })
    return path


def _build_summary(
    args: argparse.Namespace,
    device: torch.device,
    cases: list[dict],
    ckpt_meta: dict,
    load_warnings: list[str],
    missing_count: int,
    unexpected_count: int,
    per_stage_agg: dict,
    total_wall_s: float,
    status: str,
    exit_code: int,
    pass_criteria: dict[str, bool],
    failures: list[str],
    warnings_list: list[str],
    output_dir: Path,
) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary = {
        "schema_version": 1,
        "audit_name": "foundation_x_smoke",
        "status": status,
        "strict": args.strict,
        "limited_run": True,
        "generated_at_utc": now,
        "inputs": {
            "input_dir": str(args.input_dir),
            "labels_dir": str(args.labels_dir),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": ckpt_meta["checkpoint_sha256"],
            "checkpoint_size_bytes": ckpt_meta["checkpoint_size_bytes"],
            "img_size": args.img_size,
            "num_cases_requested": args.num_cases,
            "num_cases_processed": len(cases),
            "device": str(device),
            "case_list_source": "csv_override" if args.case_list else "priority_plus_filler",
        },
        "checkpoint_metadata": {
            "top_level_keys": ckpt_meta["top_level_keys"],
            "state_dict_len": ckpt_meta["state_dict_len"],
            "backbone_prefix": ckpt_meta["backbone_prefix"],
            "prefix_match_count": ckpt_meta["prefix_match_count"],
            "missing_keys_count": missing_count,
            "unexpected_keys_count": unexpected_count,
            "sample_prefix_keys": ckpt_meta["sample_prefix_keys"],
        },
        "inference": {
            "total_wall_seconds": round(total_wall_s, 2),
            "per_stage": per_stage_agg,
        },
        "pass_criteria": pass_criteria,
        "outputs": {
            "report_md": str(output_dir / "foundation_x_smoke_report.md"),
            "summary_json": str(output_dir / "foundation_x_smoke_summary.json"),
            "summary_yaml": str(output_dir / "foundation_x_smoke_summary.yaml"),
            "per_case_stats_csv": str(output_dir / "per_case_stats.csv"),
            "selection_log_csv": str(output_dir / "selection_log.csv"),
            "load_metadata_json": str(output_dir / "load_metadata.json"),
            "visuals_dir": str(output_dir / "visuals"),
        },
        "load_warnings": load_warnings,
        "warnings": warnings_list,
        "failures": failures,
        "exit_code": exit_code,
        "decision_for_next_pr": (
            "PR-5_hybrid_adapter_sanity" if status == "PASS"
            else "blocking_fix_before_PR-5" if status == "BLOCKED"
            else "interpret_PARTIAL_PASS_before_PR-5"
        ),
    }
    return _sanitize_for_json(summary)


def _write_report_md(summary: dict, output_dir: Path) -> Path:
    status = summary["status"]
    inp = summary["inputs"]
    ckpt = summary["checkpoint_metadata"]
    inf = summary["inference"]

    lines: list[str] = [
        "# Foundation X Smoke Test Report",
        "",
        "## Repository findings",
        "",
        "- **Foundation X model class:** `FoundationXBackbone` in `src/models/backbone.py`",
        "- **Checkpoint:** `checkpoints/foundation_x.pth` (Swin-B, embed_dim=128, patch_size=4, window_size=7)",
        "- **Checkpoint top-level key:** `ckpt['model']`",
        f"- **Key prefix stripped:** `{ckpt['backbone_prefix']}` "
        "(note: `docs/foundation_nnunet_dev_guide.md` incorrectly lists `backbone.` — not authoritative per D-046)",
        "- **Key remapping:** `_remap_key()` in `src/models/backbone.py` — "
        "converts `layers.N.*` → `layers_N.*`, shifts downsample index to N+1",
        "- **Expected input:** `(B, 1, H, W)` grayscale float32 in `[0,1]`; H=W=img_size (256 or 512)",
        "- **Internal normalization:** grayscale→RGB repeat then ImageNet mean/std (D-062/D-070)",
        "- **Output:** list of 4 spatial feature maps `(B, {128,256,512,1024}, H/{4,8,16,32}, W/{4,8,16,32})`",
        "- **No standalone smoke script existed prior to PR-4** (all existing tests stub the checkpoint)",
        "- **Governance note:** PR-4 operates on the nnU-Net v2/Dataset101 track. "
        "Recovery memory updated in §P1.13 / D-073 / VALIDATION_CHECKLIST §20.",
        "",
        "## Smoke test setup",
        "",
        "```",
        "python scripts/smoke_foundation_x.py \\",
        f"  --input_dir {inp['input_dir']} \\",
        f"  --labels_dir {inp['labels_dir']} \\",
        f"  --checkpoint {inp['checkpoint']} \\",
        f"  --output_dir {output_dir} \\",
        f"  --img_size {inp['img_size']} \\",
        f"  --device {inp['device']} \\",
        f"  --num_cases {inp['num_cases_requested']}",
        "```",
        "",
        f"- **Device:** `{inp['device']}`",
        f"- **Cases requested:** {inp['num_cases_requested']}",
        f"- **Cases processed:** {inp['num_cases_processed']}",
        f"- **Case list source:** `{inp['case_list_source']}`",
        f"- **Checkpoint SHA-256:** `{inp['checkpoint_sha256']}`",
        f"- **Checkpoint size:** {inp['checkpoint_size_bytes']:,} bytes",
        f"- **Preprocessing:** grayscale uint8 PNG → /255.0 → float32 [0,1] → "
        f"(1,1,{inp['img_size']},{inp['img_size']}) → internal RGB repeat + ImageNet norm",
        "",
        "## Results",
        "",
        f"**Load status:** {'OK' if summary['pass_criteria'].get('checkpoint_loads') else 'FAILED'}",
        "",
        f"- Loaded {ckpt['prefix_match_count']} backbone keys (prefix `{ckpt['backbone_prefix']}`)",
        f"- Missing keys vs. timm state dict: **{ckpt['missing_keys_count']}**",
        f"- Unexpected keys vs. timm state dict: **{ckpt['unexpected_keys_count']}**",
        "",
        "**Per-stage inference summary** (aggregated across all processed cases):",
        "",
        "| Stage | Channels | Spatial | Shape OK | Mean | Std | NaN total | Inf total |",
        "|-------|----------|---------|----------|------|-----|-----------|-----------|",
    ]

    for s_key, sv in inf["per_stage"].items():
        s_idx = int(s_key.split("_")[1])
        ok = "✓" if sv.get("shapes_match_contract") else "✗"
        mean_v = sv.get("mean_across_cases")
        std_v = sv.get("std_across_cases")
        mean_s = f"{mean_v:.4f}" if mean_v is not None else "n/a"
        std_s = f"{std_v:.4f}" if std_v is not None else "n/a"
        lines.append(
            f"| {s_idx} | {STAGE_CHANNEL_EXPECT[s_idx]} | "
            f"H/{STAGE_STRIDE_EXPECT[s_idx]} | {ok} | {mean_s} | {std_s} | "
            f"{sv.get('nan_count_total', '?')} | {sv.get('inf_count_total', '?')} |"
        )

    lines += [
        "",
        f"**Total inference wall time:** {inf['total_wall_seconds']:.1f}s "
        f"for {inp['num_cases_processed']} cases",
        "",
        f"**Visual artifacts:** `{summary['outputs']['visuals_dir']}/`",
        "",
        "## Failure cases",
        "",
    ]
    if summary.get("failures"):
        for f_str in summary["failures"]:
            lines.append(f"- {f_str}")
    else:
        lines.append("None.")

    if summary.get("warnings"):
        lines += ["", "**Warnings:**", ""]
        for w_str in summary["warnings"]:
            lines.append(f"- {w_str}")

    lines += [
        "",
        "## Decision",
        "",
        f"**Status: `{status}`**",
        "",
        "Pass criteria:",
        "",
    ]
    for criterion, passed in summary["pass_criteria"].items():
        mark = "✓" if passed else "✗"
        lines.append(f"- [{mark}] `{criterion}`")

    lines += [
        "",
        "> **Framing boundary (D-035 / D-040–D-042):** This report documents only that the",
        "> Foundation X code path loads and produces finite, image-aligned feature maps on",
        "> Dataset101 imagery. It does not constitute evidence that Foundation X generalizes",
        "> to unseen data or is the superior model. The authoritative comparison anchor",
        "> remains `pretrained_resnet34_unet` at held-out positive-only Dice 0.4951.",
        "",
        "## Next recommended PR",
        "",
    ]

    if status == "PASS":
        lines += [
            "**PR-5: Hybrid adapter sanity test.**",
            "",
            "Foundation X loads cleanly and produces well-formed, finite, spatially-structured",
            "feature maps on Dataset101 imagery. The next step is to confirm that the",
            "`HybridFoundationUNet` (D-054–D-059 scale mapping) runs a correct forward pass",
            "end-to-end and that the fused outputs are valid segmentation logits.",
        ]
    elif status == "PARTIAL_PASS":
        lines += [
            "**Interpret PARTIAL_PASS before advancing to PR-5.**",
            "",
            "The model loads but one or more stages produced NaN/Inf or degenerate features.",
            "Inspect visual overlays (`visuals/`) and `per_case_stats.csv` to diagnose.",
            "If features appear spatially structured, advance to PR-5 cautiously.",
            "If outputs look like noise, open a blocking-fix PR first.",
        ]
    else:
        lines += [
            "**Blocking fix required before PR-5.**",
            "",
            "The checkpoint could not be loaded or produced invalid outputs.",
            f"Diagnose using `load_metadata.json`. Failures: {summary.get('failures', [])}",
        ]

    path = output_dir / "foundation_x_smoke_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ─── Main logic ───────────────────────────────────────────────────────────────


def run_smoke(args: argparse.Namespace) -> dict:
    """
    Execute the smoke test and return the summary dict.
    Calls sys.exit(2) on BLOCKED conditions; otherwise returns normally.
    """
    import time

    _set_seed()

    # Resolve relative paths against REPO_ROOT
    for attr in ("input_dir", "labels_dir", "checkpoint", "output_dir"):
        p = getattr(args, attr)
        if p is not None and not p.is_absolute():
            setattr(args, attr, REPO_ROOT / p)
    if args.case_list is not None and not args.case_list.is_absolute():
        args.case_list = REPO_ROOT / args.case_list

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    print(f"[smoke] torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={device}")
    if device.type == "cuda":
        print(f"[smoke] GPU: {torch.cuda.get_device_name(device)}")

    # ── Checkpoint presence gate ──────────────────────────────────────────────
    if not args.checkpoint.exists():
        msg = f"Checkpoint not found: {args.checkpoint}"
        print(f"ERROR: {msg}", file=sys.stderr)
        blocked = _sanitize_for_json({
            "schema_version": 1,
            "audit_name": "foundation_x_smoke",
            "status": "BLOCKED",
            "strict": args.strict,
            "limited_run": True,
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "failures": [msg],
            "warnings": [],
            "exit_code": 2,
        })
        (output_dir / "foundation_x_smoke_summary.json").write_text(
            json.dumps(blocked, indent=2), encoding="utf-8"
        )
        sys.exit(2)

    # ── Metadata (first load) ─────────────────────────────────────────────────
    ckpt_meta = _load_checkpoint_metadata(args.checkpoint)
    remapped_keys: set[str] = ckpt_meta.pop("_remapped_keys")

    # ── Case selection ────────────────────────────────────────────────────────
    print(f"[smoke] Selecting {args.num_cases} cases from {args.input_dir} ...")
    cases = select_cases(args.input_dir, args.labels_dir, args.num_cases, args.case_list)
    print(f"[smoke] Selected {len(cases)} cases "
          f"({sum(1 for c in cases if c['label_max'] == 1)} positive, "
          f"{sum(1 for c in cases if c['label_max'] == 0)} negative).")
    _write_selection_log(cases, output_dir)

    if args.dry_run:
        print("[smoke] --dry_run: stopping before model load.")
        for c in cases:
            print(f"  {c['case_id']}  source={c['source']}  label_max={c['label_max']}")
        sys.exit(0)

    if len(cases) < args.num_cases:
        print(f"WARNING: only {len(cases)}/{args.num_cases} cases available.", file=sys.stderr)

    # ── Model load (second checkpoint load, inside FoundationXBackbone) ───────
    print("[smoke] Loading FoundationXBackbone (re-reads checkpoint internally) ...")
    t_load = time.time()
    model, load_warnings, missing_count, unexpected_count = _load_model_and_count_mismatches(
        args.checkpoint, args.img_size, device, remapped_keys
    )
    print(f"[smoke] Model loaded in {time.time() - t_load:.1f}s  "
          f"missing={missing_count}  unexpected={unexpected_count}")

    (output_dir / "load_metadata.json").write_text(
        json.dumps(_sanitize_for_json({
            **{k: v for k, v in ckpt_meta.items()},
            "missing_keys_count": missing_count,
            "unexpected_keys_count": unexpected_count,
            "load_warnings": load_warnings,
        }), indent=2, default=str),
        encoding="utf-8",
    )

    # ── Inference loop ────────────────────────────────────────────────────────
    all_case_stats: list[list[dict]] = []
    infer_ms_per_case: list[float] = []
    all_feats_for_vis: list[tuple[str, np.ndarray, list]] = []
    failures: list[str] = []
    warnings_list: list[str] = []

    if missing_count > 100:
        warnings_list.append(
            f"Large missing_keys_count={missing_count}; backbone may be partially initialized."
        )
    if unexpected_count > 100:
        warnings_list.append(
            f"Large unexpected_keys_count={unexpected_count}; checkpoint has extra keys."
        )

    raw_features_dir: Path | None = output_dir / "raw_features" if args.save_raw_features else None
    if raw_features_dir:
        raw_features_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] Running inference on {len(cases)} cases ...")
    t_infer_total = time.time()

    for c in cases:
        try:
            img_tensor = _load_image(Path(c["image_path"]), args.img_size, device)
            t0 = time.time()
            with torch.no_grad():
                feats = model(img_tensor)
            infer_ms = (time.time() - t0) * 1000.0

            case_stats = [_stage_stats(feats[i], i, args.img_size, c["case_id"]) for i in range(4)]
            all_case_stats.append(case_stats)
            infer_ms_per_case.append(infer_ms)

            for s in case_stats:
                if s["nan_count"] > 0 or s["inf_count"] > 0:
                    failures.append(
                        f"{c['case_id']} stage {s['stage']}: "
                        f"nan={s['nan_count']} inf={s['inf_count']}"
                    )

            if not args.no_visuals:
                img_uint8 = (img_tensor[0, 0].cpu().numpy() * 255).astype(np.uint8)
                feats_cpu = [f[0].cpu() for f in feats]
                all_feats_for_vis.append((c["case_id"], img_uint8, feats_cpu))

            if raw_features_dir:
                for i, feat in enumerate(feats):
                    np.save(
                        str(raw_features_dir / f"{c['case_id']}_stage{i}.npy"),
                        feat.cpu().numpy(),
                    )

            print(f"  [{c['case_id']}] {infer_ms:.0f}ms  shapes={[list(f.shape) for f in feats]}")

        except Exception as exc:
            failures.append(f"{c['case_id']}: inference error: {exc}")
            print(f"  [{c['case_id']}] ERROR: {exc}", file=sys.stderr)
            all_case_stats.append([])
            infer_ms_per_case.append(0.0)

    total_wall_s = time.time() - t_infer_total

    # ── Visuals ───────────────────────────────────────────────────────────────
    if not args.no_visuals and all_feats_for_vis:
        print("[smoke] Rendering visuals ...")
        (output_dir / "visuals").mkdir(parents=True, exist_ok=True)
        try:
            from visualize_foundation_x_smoke import render_case
            for (case_id, img_uint8, feats_cpu) in all_feats_for_vis:
                try:
                    render_case(case_id, img_uint8, feats_cpu, output_dir)
                except Exception as exc:
                    warnings_list.append(f"Visual rendering failed for {case_id}: {exc}")
                    print(f"  [vis] WARNING {case_id}: {exc}", file=sys.stderr)
        except ImportError as exc:
            warnings_list.append(f"visualize_foundation_x_smoke import failed: {exc}")

    # ── Aggregate and decide ──────────────────────────────────────────────────
    valid_stats = [cs for cs in all_case_stats if cs]
    per_stage_agg = _aggregate_stage_stats(valid_stats) if valid_stats else {
        f"stage_{i}": {"shapes_match_contract": False, "nan_count_total": 1, "inf_count_total": 0}
        for i in range(4)
    }

    status, exit_code, pass_criteria = _compute_decision(
        model_load_ok=True,
        num_cases_processed=len(valid_stats),
        num_cases_requested=args.num_cases,
        per_stage_agg=per_stage_agg,
        unexpected_keys_count=unexpected_count,
        strict=args.strict,
    )

    # ── Write all outputs ─────────────────────────────────────────────────────
    _write_per_case_stats(all_case_stats, infer_ms_per_case, output_dir)

    summary = _build_summary(
        args=args,
        device=device,
        cases=valid_stats if valid_stats else cases,
        ckpt_meta=ckpt_meta,
        load_warnings=load_warnings,
        missing_count=missing_count,
        unexpected_count=unexpected_count,
        per_stage_agg=per_stage_agg,
        total_wall_s=total_wall_s,
        status=status,
        exit_code=exit_code,
        pass_criteria=pass_criteria,
        failures=failures,
        warnings_list=warnings_list,
        output_dir=output_dir,
    )

    (output_dir / "foundation_x_smoke_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if _HAS_YAML:
        with (output_dir / "foundation_x_smoke_summary.yaml").open("w", encoding="utf-8") as fh:
            _yaml.dump(summary, fh, default_flow_style=False, allow_unicode=True)

    _write_report_md(summary, output_dir)

    print(f"\n[smoke] -- STATUS: {status} --")
    print(f"[smoke] Report:  {output_dir / 'foundation_x_smoke_report.md'}")
    print(f"[smoke] Summary: {output_dir / 'foundation_x_smoke_summary.json'}")

    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run_smoke(args)
    return int(summary.get("exit_code", 0))


if __name__ == "__main__":
    sys.exit(main())
