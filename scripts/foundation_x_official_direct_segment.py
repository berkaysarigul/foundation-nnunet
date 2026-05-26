"""
PR-9A-FIX Stage A/B: official Foundation_X direct segmentation diagnostics.

This script uses the vendored official JLiangLab/Foundation_X Swin cyclic
segmentation path through `src.models.foundation_x_official`. It does not train,
does not run the full Dataset101 evaluation, and does not export priors.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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

import torch

from src.data.foundation_x_official_preprocess import (  # noqa: E402
    PREPROCESS_VARIANTS,
    preprocess_image_for_foundation_x,
)
from src.models.foundation_x_official import (  # noqa: E402
    CHESTX_DET_PNEUMOTHORAX_CHANNEL,
    HEAD_CANDID_PTX,
    HEAD_CHESTX_DET,
    HEAD_SIIM_ACR_PNEUMOTHORAX,
    OFFICIAL_FOUNDATION_X_COMMIT,
    OfficialFoundationXSegmentationModel,
    SegmentationHeadSpec,
    default_head_specs,
    tensor_stats as torch_tensor_stats,
)


ALLOWED_OUTPUT_ANCHOR = "artifacts/diagnostics/foundation_x_official_direct"
DEFAULT_THRESHOLD = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Official Foundation_X direct segmentation Stage A/B diagnostics "
            "for Dataset101_Pneumothorax. No training, no priors."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-keys", nargs="+", default=["model", "teacher_model"])
    parser.add_argument("--heads", type=int, nargs="+", default=[HEAD_SIIM_ACR_PNEUMOTHORAX, HEAD_CANDID_PTX])
    parser.add_argument("--head4-channel", type=int, default=CHESTX_DET_PNEUMOTHORAX_CHANNEL)
    parser.add_argument("--preprocess-variants", nargs="+", default=list(PREPROCESS_VARIANTS))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--positive-cases", type=int, default=1)
    parser.add_argument("--negative-cases", type=int, default=1)
    parser.add_argument("--case-sampling", choices=("balanced", "sequential"), default="balanced")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-visuals", type=_strtobool, default=False)
    parser.add_argument("--save-histograms", type=_strtobool, default=False)
    parser.add_argument("--stage", choices=("stage_a", "stage_b", "both"), default="both")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _strtobool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected truthy/falsy string, got {value!r}")


def validate_output_guardrail(output_dir: Path) -> tuple[bool, str]:
    norm = output_dir.resolve().as_posix().lower()
    if ALLOWED_OUTPUT_ANCHOR not in norm:
        return False, f"Output dir must be under {ALLOWED_OUTPUT_ANCHOR}: {output_dir}"
    forbidden = ("artifacts/runs", "nnunet_results", "nnunet_raw")
    for segment in forbidden:
        if segment in norm:
            return False, f"Output dir falls into forbidden segment {segment!r}: {output_dir}"
    return True, ""


def validate_local_run_guardrail(args: argparse.Namespace, device: torch.device) -> tuple[bool, str]:
    if device.type != "cpu":
        return True, ""
    if args.max_cases <= 0:
        return False, "Refusing CPU Stage B without --max-cases. Use --max-cases 2 locally."
    requested = int(args.positive_cases) + int(args.negative_cases)
    if requested > 2 or args.max_cases > 2:
        return False, (
            "Refusing CPU run above two cases. Local machine is for Stage A and "
            "max-2 Stage B dry-runs only."
        )
    return True, ""


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def _normalize_case_id(raw: str) -> str:
    case_id = raw.strip()
    if case_id.endswith(".png"):
        case_id = case_id[:-4]
    if case_id.endswith("_0000"):
        case_id = case_id[:-5]
    return case_id


def list_dataset101_cases(dataset_root: Path) -> list[dict[str, Any]]:
    images_dir = dataset_root / "imagesTs"
    labels_dir = dataset_root / "heldout_labelsTs"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected imagesTs/ and heldout_labelsTs/ under {dataset_root}"
        )

    cases: list[dict[str, Any]] = []
    image_paths = sorted(images_dir.glob("*_0000.png"))
    if not image_paths:
        image_paths = sorted(images_dir.glob("*.png"))
    for image_path in image_paths:
        case_id = _normalize_case_id(image_path.stem)
        label_path = labels_dir / f"{case_id}.png"
        label_exists = label_path.exists()
        label_positive = False
        label_foreground_pixels = 0
        label_unique: list[int] = []
        if label_exists:
            arr = np.asarray(Image.open(label_path).convert("L"))
            label_unique = sorted(int(x) for x in np.unique(arr).tolist())
            label_foreground_pixels = int((arr > 0).sum())
            label_positive = label_foreground_pixels > 0
        cases.append(
            {
                "case_id": case_id,
                "image_path": image_path,
                "label_path": label_path,
                "label_exists": label_exists,
                "label_is_positive": label_positive,
                "label_foreground_pixels": label_foreground_pixels,
                "label_unique_values": label_unique,
            },
        )
    return cases


def dataset101_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "imagesTs_count": len(cases),
        "heldout_labelsTs_count": sum(1 for c in cases if c["label_exists"]),
        "matched_cases": sum(1 for c in cases if c["label_exists"]),
        "positive_cases": sum(1 for c in cases if c["label_is_positive"]),
        "negative_cases": sum(1 for c in cases if c["label_exists"] and not c["label_is_positive"]),
    }


def select_cases(
    cases: list[dict[str, Any]],
    positive_cases: int,
    negative_cases: int,
    max_cases: int,
    case_sampling: str,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select Stage B cases, balanced by default."""
    if case_sampling == "sequential":
        selected = cases[: max(0, max_cases)]
        return selected

    positives = [c for c in cases if c["label_exists"] and c["label_is_positive"]]
    negatives = [c for c in cases if c["label_exists"] and not c["label_is_positive"]]
    if len(positives) < positive_cases:
        raise ValueError(f"Requested {positive_cases} positives, found {len(positives)}")
    if len(negatives) < negative_cases:
        raise ValueError(f"Requested {negative_cases} negatives, found {len(negatives)}")

    if seed >= 0:
        # Deterministic shuffle, still balanced.
        rng = np.random.default_rng(seed)
        pos_idx = sorted(rng.choice(len(positives), size=positive_cases, replace=False).tolist())
        neg_idx = sorted(rng.choice(len(negatives), size=negative_cases, replace=False).tolist())
        selected = [positives[int(i)] for i in pos_idx] + [negatives[int(i)] for i in neg_idx]
    else:
        selected = positives[:positive_cases] + negatives[:negative_cases]

    if max_cases > 0:
        selected = selected[:max_cases]
    return selected


def load_binary_label(label_path: Path, output_size: int) -> np.ndarray:
    label = Image.open(label_path).convert("L")
    if label.size != (output_size, output_size):
        label = label.resize((output_size, output_size), Image.NEAREST)
    return (np.asarray(label) > 0).astype(np.uint8)


def per_case_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray) -> dict[str, Any]:
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    pred_sum = tp + fp
    gt_sum = tp + fn
    union = tp + fp + fn

    if pred_sum == 0 and gt_sum == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    else:
        dice = (2.0 * tp) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
        iou = tp / union if union > 0 else 0.0
        precision = tp / pred_sum if pred_sum > 0 else 0.0
        recall = tp / gt_sum if gt_sum > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "tn_pixels": tn,
        "gt_foreground_pixels": int(gt_sum),
        "pred_foreground_pixels": int(pred_sum),
        "total_pixels": int(pred.size),
    }


def numpy_stats(arr: np.ndarray) -> dict[str, Any]:
    x = arr.astype(np.float32, copy=False)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def save_probability_map(prob: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(prob, 0.0, 1.0)
    Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(path)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_overlay(image_path: Path, gt: np.ndarray, pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = gt.shape
    image = Image.open(image_path).convert("L").resize((w, h), Image.BILINEAR)
    base = np.repeat(np.asarray(image, dtype=np.uint8)[:, :, None], 3, axis=2).astype(np.float32)
    color = np.zeros_like(base)
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    color[np.logical_and(gt_bool, np.logical_not(pred_bool))] = np.asarray([0, 255, 0])
    color[np.logical_and(pred_bool, np.logical_not(gt_bool))] = np.asarray([255, 0, 0])
    color[np.logical_and(pred_bool, gt_bool)] = np.asarray([255, 255, 0])
    mask = np.logical_or(gt_bool, pred_bool)
    alpha = 0.45
    base[mask] = (1.0 - alpha) * base[mask] + alpha * color[mask]
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").save(path)


def save_probability_histogram(prob: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
        ax.hist(prob.reshape(-1), bins=40, range=(0.0, 1.0), color="#3267a8")
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability")
        ax.set_ylabel("pixels")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        counts, bins = np.histogram(prob.reshape(-1), bins=40, range=(0.0, 1.0))
        fallback = path.with_suffix(".json")
        fallback.write_text(
            json.dumps({"counts": counts.tolist(), "bins": bins.tolist()}, indent=2),
            encoding="utf-8",
        )


def _head_dir_name(spec: SegmentationHeadSpec) -> str:
    return spec.key


def _build_model(model_factory: Any, device: torch.device) -> Any:
    try:
        return model_factory(device=device)
    except TypeError:
        return model_factory()


def _model_load_checkpoint(model: Any, checkpoint: Path, state_key: str) -> dict[str, Any]:
    if hasattr(model, "load_checkpoint"):
        return model.load_checkpoint(checkpoint, state_key)
    return {"status": "loaded", "state_key": state_key, "stub": True}


def _model_synthetic_probe(
    model: Any,
    image_size: int,
    heads: list[int],
    head4_channel: int,
) -> dict[str, Any]:
    if hasattr(model, "run_synthetic_probe"):
        return model.run_synthetic_probe(image_size=image_size, heads=heads, head4_channel=head4_channel)
    probe = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    outputs: dict[str, Any] = {"synthetic_input": torch_tensor_stats(probe), "outputs": {}}
    for spec in default_head_specs(heads, head4_channel):
        logits, prob = _model_predict_probability(model, probe, spec)
        outputs["outputs"][spec.key] = {
            "selected_logits": torch_tensor_stats(logits),
            "probability": torch_tensor_stats(prob),
        }
    return outputs


def _model_predict_probability(
    model: Any,
    tensor: torch.Tensor,
    spec: SegmentationHeadSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "predict_probability"):
        return model.predict_probability(tensor, head_idx=spec.head_idx, channel=spec.channel)
    if hasattr(model, "forward_logits"):
        logits = model.forward_logits(tensor, head_idx=spec.head_idx, channel=spec.channel)
        return logits, torch.sigmoid(logits)
    raise AttributeError("Model object must expose predict_probability() or forward_logits()")


def run_diagnostics(
    args: argparse.Namespace,
    model_factory: Any = OfficialFoundationXSegmentationModel,
) -> int:
    ok, why = validate_output_guardrail(args.out)
    if not ok:
        print(f"[guardrail] {why}", file=sys.stderr)
        return 2

    try:
        device = resolve_device(args.device)
    except Exception as exc:
        print(f"[guardrail] {exc}", file=sys.stderr)
        return 2

    ok, why = validate_local_run_guardrail(args, device)
    if not ok:
        print(f"[guardrail] {why}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    stage_a_enabled = args.stage in {"stage_a", "both"}
    stage_b_enabled = args.stage in {"stage_b", "both"}
    head_specs = default_head_specs(args.heads, args.head4_channel)

    cases: list[dict[str, Any]] = []
    selected_cases: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    if stage_b_enabled:
        cases = list_dataset101_cases(args.dataset_root)
        counts = dataset101_counts(cases)
        selected_cases = select_cases(
            cases,
            positive_cases=args.positive_cases,
            negative_cases=args.negative_cases,
            max_cases=args.max_cases,
            case_sampling=args.case_sampling,
            seed=args.seed,
        )
        if not selected_cases:
            print("[error] Stage B selected zero cases.", file=sys.stderr)
            return 3

    load_diagnostics: dict[str, Any] = {}
    tensor_stats: dict[str, Any] = {
        "stage_a_synthetic": {},
        "stage_b_preprocess": {},
        "stage_b_outputs": {},
    }
    per_case_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    loaded_state_keys: list[str] = []

    for state_key in args.state_keys:
        print(f"[state] loading official model with checkpoint[{state_key!r}]")
        try:
            model = _build_model(model_factory, device)
            load_diag = _model_load_checkpoint(model, args.checkpoint, state_key)
            load_diagnostics[state_key] = load_diag
            loaded_state_keys.append(state_key)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            failures.append(f"state_key={state_key}: {err}")
            load_diagnostics[state_key] = {
                "status": "failed",
                "error": err,
                "traceback": traceback.format_exc(limit=10),
            }
            continue

        if stage_a_enabled:
            try:
                tensor_stats["stage_a_synthetic"][state_key] = _model_synthetic_probe(
                    model,
                    image_size=224,
                    heads=args.heads,
                    head4_channel=args.head4_channel,
                )
            except Exception as exc:
                err = f"Stage A failed for {state_key}: {type(exc).__name__}: {exc}"
                failures.append(err)
                tensor_stats["stage_a_synthetic"][state_key] = {
                    "status": "failed",
                    "error": err,
                    "traceback": traceback.format_exc(limit=10),
                }

        if stage_b_enabled:
            for case in selected_cases:
                case_id = case["case_id"]
                for variant in args.preprocess_variants:
                    prep_key = f"{state_key}/{variant}/{case_id}"
                    try:
                        prep = preprocess_image_for_foundation_x(case["image_path"], variant)
                        image_tensor = prep.tensor.to(device)
                        output_size = int(image_tensor.shape[-1])
                        gt = load_binary_label(case["label_path"], output_size=output_size)
                        tensor_stats["stage_b_preprocess"][prep_key] = prep.stats
                    except Exception as exc:
                        err = f"Preprocess failed for {prep_key}: {type(exc).__name__}: {exc}"
                        failures.append(err)
                        tensor_stats["stage_b_preprocess"][prep_key] = {
                            "status": "failed",
                            "error": err,
                            "traceback": traceback.format_exc(limit=5),
                        }
                        continue

                    for spec in head_specs:
                        output_key = f"{state_key}/{variant}/{case_id}/{spec.key}"
                        try:
                            logits, prob_tensor = _model_predict_probability(model, image_tensor, spec)
                            prob = prob_tensor.detach().squeeze().float().cpu().numpy().astype(np.float32)
                            logits_np = logits.detach().squeeze().float().cpu().numpy().astype(np.float32)
                            if prob.ndim != 2:
                                raise AssertionError(
                                    f"Expected a single probability map for {output_key}, got shape {prob.shape}"
                                )
                            pred = (prob >= float(args.threshold)).astype(np.uint8)
                            metrics = per_case_metrics(pred, gt)

                            head_dir = _head_dir_name(spec)
                            prob_path = (
                                args.out
                                / "probability_maps"
                                / state_key
                                / variant
                                / head_dir
                                / f"{case_id}.png"
                            )
                            binary_path = (
                                args.out
                                / "binary_masks"
                                / state_key
                                / variant
                                / head_dir
                                / f"{case_id}.png"
                            )
                            overlay_path = (
                                args.out
                                / "visual_overlays"
                                / state_key
                                / variant
                                / head_dir
                                / f"{case_id}.png"
                            )
                            hist_path = (
                                args.out
                                / "probability_histograms"
                                / state_key
                                / variant
                                / head_dir
                                / f"{case_id}.png"
                            )

                            save_probability_map(prob, prob_path)
                            save_binary_mask(pred, binary_path)
                            if args.save_visuals:
                                save_overlay(case["image_path"], gt, pred, overlay_path)
                            if args.save_histograms:
                                save_probability_histogram(prob, hist_path)

                            tensor_stats["stage_b_outputs"][output_key] = {
                                "logits": numpy_stats(logits_np),
                                "probability": numpy_stats(prob),
                                "threshold": float(args.threshold),
                                "prediction": {
                                    "foreground_pixels": int(pred.sum()),
                                    "is_positive": bool(pred.sum() > 0),
                                },
                            }
                            per_case_rows.append(
                                {
                                    "case_id": case_id,
                                    "state_key": state_key,
                                    "preprocess_variant": variant,
                                    "head_idx": spec.head_idx,
                                    "channel": "" if spec.channel is None else spec.channel,
                                    "head_key": spec.key,
                                    "threshold": float(args.threshold),
                                    "gt_is_positive": bool(gt.sum() > 0),
                                    "pred_is_positive": bool(pred.sum() > 0),
                                    "logits_min": float(np.min(logits_np)),
                                    "logits_max": float(np.max(logits_np)),
                                    "logits_mean": float(np.mean(logits_np)),
                                    "logits_std": float(np.std(logits_np)),
                                    "prob_min": float(np.min(prob)),
                                    "prob_max": float(np.max(prob)),
                                    "prob_mean": float(np.mean(prob)),
                                    "prob_std": float(np.std(prob)),
                                    "probability_map_path": str(prob_path),
                                    "binary_mask_path": str(binary_path),
                                    "visual_overlay_path": str(overlay_path) if args.save_visuals else "",
                                    "probability_histogram_path": str(hist_path) if args.save_histograms else "",
                                    **metrics,
                                },
                            )
                        except Exception as exc:
                            err = f"Inference failed for {output_key}: {type(exc).__name__}: {exc}"
                            failures.append(err)
                            tensor_stats["stage_b_outputs"][output_key] = {
                                "status": "failed",
                                "error": err,
                                "traceback": traceback.format_exc(limit=5),
                            }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = build_summary(
        args=args,
        device=device,
        counts=counts,
        selected_cases=selected_cases,
        loaded_state_keys=loaded_state_keys,
        per_case_rows=per_case_rows,
        failures=failures,
        head_specs=head_specs,
    )

    (args.out / "load_diagnostics.json").write_text(
        json.dumps(_sanitize_for_json(load_diagnostics), indent=2),
        encoding="utf-8",
    )
    (args.out / "tensor_stats.json").write_text(
        json.dumps(_sanitize_for_json(tensor_stats), indent=2),
        encoding="utf-8",
    )
    write_per_case_csv(per_case_rows, args.out / "per_case_metrics.csv")
    write_yaml_or_json(summary, args.out / "summary.yaml", args.out / "summary.json")
    (args.out / "run_metadata.json").write_text(
        json.dumps(_sanitize_for_json(build_run_metadata(args, device)), indent=2),
        encoding="utf-8",
    )
    write_report(args.out / "report.md", summary, load_diagnostics, tensor_stats, per_case_rows)

    print(f"[done] artifacts written under {args.out}")
    if not loaded_state_keys:
        return 4
    return 0


def build_summary(
    args: argparse.Namespace,
    device: torch.device,
    counts: dict[str, int],
    selected_cases: list[dict[str, Any]],
    loaded_state_keys: list[str],
    per_case_rows: list[dict[str, Any]],
    failures: list[str],
    head_specs: list[SegmentationHeadSpec],
) -> dict[str, Any]:
    non_trivial_rows = [
        r for r in per_case_rows
        if 0 < int(r.get("pred_foreground_pixels", 0)) < int(r.get("total_pixels", 0))
    ]
    nonconstant_prob_rows = [
        r for r in per_case_rows
        if float(r.get("prob_std", 0.0)) > 1e-6
    ]
    by_combo = aggregate_rows(per_case_rows)
    return {
        "audit_name": "foundation_x_official_direct_stage_a_b",
        "schema_version": 1,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "official_foundation_x_commit": OFFICIAL_FOUNDATION_X_COMMIT,
        "do_not_use_as_priors": True,
        "framing_notice": (
            "PR-9A reverse-engineered predictions were trivial and must not be "
            "exported as nnU-Net priors. This run only diagnoses the official "
            "Foundation_X segmentation path."
        ),
        "stage": args.stage,
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "state_keys_requested": list(args.state_keys),
        "state_keys_loaded": loaded_state_keys,
        "heads_requested": list(args.heads),
        "head4_channel": int(args.head4_channel),
        "head_specs_evaluated": [
            {"head_idx": s.head_idx, "channel": s.channel, "key": s.key, "label": s.label}
            for s in head_specs
        ],
        "preprocess_variants": list(args.preprocess_variants),
        "device": str(device),
        "threshold": float(args.threshold),
        "dataset_counts": counts,
        "selected_cases": [
            {
                "case_id": c["case_id"],
                "label_is_positive": bool(c["label_is_positive"]),
                "label_foreground_pixels": int(c["label_foreground_pixels"]),
            }
            for c in selected_cases
        ],
        "stage_b_rows": len(per_case_rows),
        "non_trivial_mask_rows": len(non_trivial_rows),
        "nonconstant_probability_rows": len(nonconstant_prob_rows),
        "non_trivial_enough_to_consider_stage_c": bool(non_trivial_rows and nonconstant_prob_rows),
        "aggregates_by_state_preprocess_head": by_combo,
        "failures": failures,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["state_key"]), str(row["preprocess_variant"]), str(row["head_key"]))
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (state_key, variant, head_key), group in sorted(groups.items()):
        gt_pos = [r for r in group if r["gt_is_positive"]]
        gt_neg = [r for r in group if not r["gt_is_positive"]]
        pred_pos = [r for r in group if r["pred_is_positive"]]
        out.append(
            {
                "state_key": state_key,
                "preprocess_variant": variant,
                "head_key": head_key,
                "cases": len(group),
                "gt_positive_cases": len(gt_pos),
                "gt_negative_cases": len(gt_neg),
                "pred_positive_cases": len(pred_pos),
                "mean_dice": _mean([float(r["dice"]) for r in group]),
                "mean_positive_dice": _mean([float(r["dice"]) for r in gt_pos]),
                "mean_negative_specificity": _mean([float(r["specificity"]) for r in gt_neg]),
                "mean_prob_std": _mean([float(r["prob_std"]) for r in group]),
                "non_trivial_masks": sum(
                    1
                    for r in group
                    if 0 < int(r["pred_foreground_pixels"]) < int(r["total_pixels"])
                ),
            }
        )
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    good = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    if not good:
        return None
    return float(sum(good) / len(good))


def build_run_metadata(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tool": "scripts/foundation_x_official_direct_segment.py",
        "command": " ".join(sys.argv),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "official_foundation_x_commit": OFFICIAL_FOUNDATION_X_COMMIT,
        "constraints": [
            "no training",
            "no full Dataset101 evaluation",
            "no priors exported",
            "CPU guardrail max 2 cases",
        ],
    }


def write_per_case_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = [
            "case_id",
            "state_key",
            "preprocess_variant",
            "head_key",
            "dice",
            "pred_foreground_pixels",
            "gt_foreground_pixels",
        ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_sanitize_for_json(row))


def write_yaml_or_json(payload: dict[str, Any], yaml_path: Path, json_path: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_for_json(payload)
    if _HAS_YAML:
        yaml_path.write_text(
            _yaml.safe_dump(sanitized, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    else:
        yaml_path.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    json_path.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")


def write_report(
    path: Path,
    summary: dict[str, Any],
    load_diagnostics: dict[str, Any],
    tensor_stats: dict[str, Any],
    per_case_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# PR-9A-FIX Stage A/B Official Foundation_X Direct Segmentation")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- official commit: `{summary['official_foundation_x_commit']}`")
    lines.append(f"- state keys requested: `{', '.join(summary['state_keys_requested'])}`")
    lines.append(f"- state keys loaded: `{', '.join(summary['state_keys_loaded'])}`")
    lines.append(f"- selected cases: {len(summary['selected_cases'])}")
    lines.append(f"- Stage B rows: {summary['stage_b_rows']}")
    lines.append(f"- non-trivial mask rows: {summary['non_trivial_mask_rows']}")
    lines.append(
        f"- consider Stage C: {summary['non_trivial_enough_to_consider_stage_c']}"
    )
    lines.append("")
    lines.append("> Current reverse-engineered PR-9A predictions remain invalid as nnU-Net priors.")
    lines.append("")

    lines.append("## Checkpoint Loading")
    lines.append("")
    for state_key, diag in load_diagnostics.items():
        if diag.get("status") == "failed":
            lines.append(f"- `{state_key}`: FAILED - {diag.get('error')}")
            continue
        lines.append(
            "- `{state}`: loaded, compatible keys {compatible}, missing {missing}, "
            "unexpected {unexpected}, shape mismatches {mismatch}".format(
                state=state_key,
                compatible=diag.get("shape_compatible_key_count"),
                missing=diag.get("missing_count"),
                unexpected=diag.get("unexpected_before_shape_filter_count", diag.get("unexpected_count")),
                mismatch=diag.get("shape_mismatch_count"),
            )
        )
    lines.append("")

    lines.append("## Stage A Output Shapes")
    lines.append("")
    stage_a = tensor_stats.get("stage_a_synthetic", {})
    for state_key, state_stats in stage_a.items():
        lines.append(f"### {state_key}")
        outputs = state_stats.get("outputs", {})
        for head_key, stats in outputs.items():
            if stats.get("status") == "failed":
                lines.append(f"- `{head_key}`: FAILED - {stats.get('error')}")
                continue
            selected = stats.get("selected_logits", {})
            raw = stats.get("raw_head_logits", selected)
            lines.append(
                f"- `{head_key}` raw {raw.get('shape')} selected {selected.get('shape')}"
            )
    lines.append("")

    lines.append("## Stage B Metrics")
    lines.append("")
    lines.append("| state | preprocess | head | gt+ | pred+ | Dice | prob std | pred px |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in per_case_rows:
        lines.append(
            "| {state} | {variant} | {head} | {gt} | {pred} | {dice:.4f} | {std:.6f} | {px} |".format(
                state=row["state_key"],
                variant=row["preprocess_variant"],
                head=row["head_key"],
                gt=int(bool(row["gt_is_positive"])),
                pred=int(bool(row["pred_is_positive"])),
                dice=float(row["dice"]),
                std=float(row["prob_std"]),
                px=int(row["pred_foreground_pixels"]),
            )
        )
    lines.append("")

    if summary.get("failures"):
        lines.append("## Failures")
        lines.append("")
        for failure in summary["failures"]:
            lines.append(f"- {failure}")
        lines.append("")

    lines.append("## Colab Stage C Command")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/foundation_x_official_direct_segment.py \\")
    lines.append("  --dataset-root nnUNet_raw/Dataset101_Pneumothorax \\")
    lines.append("  --checkpoint checkpoints/foundation_x.pth \\")
    lines.append("  --state-keys model teacher_model \\")
    lines.append("  --heads 5 2 \\")
    lines.append("  --head4-channel 12 \\")
    lines.append("  --preprocess-variants official_siim_224 official_siim_512 current_pr9a_512 \\")
    lines.append("  --device cuda \\")
    lines.append("  --max-cases 20 \\")
    lines.append("  --positive-cases 10 \\")
    lines.append("  --negative-cases 10 \\")
    lines.append("  --case-sampling balanced \\")
    lines.append("  --out artifacts/diagnostics/foundation_x_official_direct/sanity_20case \\")
    lines.append("  --save-visuals true \\")
    lines.append("  --save-histograms true")
    lines.append("```")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return torch_tensor_stats(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_diagnostics(args)


if __name__ == "__main__":
    sys.exit(main())
