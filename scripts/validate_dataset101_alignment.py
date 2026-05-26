"""
Validate Dataset101_Pneumothorax image/heldout-label alignment.

This helper is diagnostic-only. It verifies counts, matching case IDs, binary
labels, positive/negative counts, and optional sample overlays.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import yaml as _yaml  # type: ignore

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Dataset101 heldout alignment.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-overlays", type=int, default=10)
    return parser.parse_args(argv)


def _normalize_case_id(raw: str) -> str:
    case_id = raw
    if case_id.endswith(".png"):
        case_id = case_id[:-4]
    if case_id.endswith("_0000"):
        case_id = case_id[:-5]
    return case_id


def scan_dataset(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images_dir = dataset_root / "imagesTs"
    labels_dir = dataset_root / "heldout_labelsTs"
    image_paths = sorted(images_dir.glob("*_0000.png"))
    if not image_paths:
        image_paths = sorted(images_dir.glob("*.png"))
    label_paths = sorted(labels_dir.glob("*.png"))
    labels_by_case = {_normalize_case_id(p.stem): p for p in label_paths}

    rows: list[dict[str, Any]] = []
    for image_path in image_paths:
        case_id = _normalize_case_id(image_path.stem)
        label_path = labels_by_case.get(case_id)
        image = Image.open(image_path)
        row: dict[str, Any] = {
            "case_id": case_id,
            "image_path": str(image_path),
            "label_path": str(label_path) if label_path else "",
            "image_shape": [image.size[1], image.size[0]],
            "label_exists": label_path is not None,
            "label_shape": None,
            "label_unique_values": [],
            "label_is_binary": False,
            "label_is_positive": False,
            "label_foreground_pixels": 0,
            "shape_matches": False,
        }
        if label_path is not None:
            label = Image.open(label_path).convert("L")
            arr = np.asarray(label)
            unique = sorted(int(x) for x in np.unique(arr).tolist())
            row.update(
                {
                    "label_shape": [label.size[1], label.size[0]],
                    "label_unique_values": unique,
                    "label_is_binary": set(unique).issubset({0, 1, 255}),
                    "label_foreground_pixels": int((arr > 0).sum()),
                    "label_is_positive": bool((arr > 0).any()),
                    "shape_matches": image.size == label.size,
                }
            )
        rows.append(row)

    image_case_ids = {_normalize_case_id(p.stem) for p in image_paths}
    label_case_ids = set(labels_by_case)
    summary = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "imagesTs_count": len(image_paths),
        "heldout_labelsTs_count": len(label_paths),
        "matched_cases": sum(1 for r in rows if r["label_exists"]),
        "missing_label_cases": sorted(image_case_ids - label_case_ids),
        "extra_label_cases": sorted(label_case_ids - image_case_ids),
        "binary_label_cases": sum(1 for r in rows if r["label_is_binary"]),
        "nonbinary_label_cases": [r["case_id"] for r in rows if r["label_exists"] and not r["label_is_binary"]],
        "shape_mismatch_cases": [r["case_id"] for r in rows if r["label_exists"] and not r["shape_matches"]],
        "positive_cases": sum(1 for r in rows if r["label_is_positive"]),
        "negative_cases": sum(1 for r in rows if r["label_exists"] and not r["label_is_positive"]),
    }
    return rows, summary


def save_overlay(row: dict[str, Any], path: Path) -> None:
    image_path = Path(row["image_path"])
    label_path = Path(row["label_path"])
    image = Image.open(image_path).convert("L")
    label = Image.open(label_path).convert("L")
    if label.size != image.size:
        label = label.resize(image.size, Image.NEAREST)
    base = np.repeat(np.asarray(image, dtype=np.uint8)[:, :, None], 3, axis=2).astype(np.float32)
    mask = np.asarray(label) > 0
    color = np.zeros_like(base)
    color[mask] = np.asarray([255, 0, 0])
    base[mask] = 0.55 * base[mask] + 0.45 * color[mask]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").save(path)


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path, num_overlays: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _sanitize(summary)
    if _HAS_YAML:
        (out_dir / "summary.yaml").write_text(
            _yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    positives = [r for r in rows if r["label_exists"] and r["label_is_positive"]]
    negatives = [r for r in rows if r["label_exists"] and not r["label_is_positive"]]
    selected = positives[: max(0, num_overlays // 2)] + negatives[: max(0, num_overlays - num_overlays // 2)]
    for row in selected:
        save_overlay(row, out_dir / "visual_overlays" / f"{row['case_id']}.png")

    lines = [
        "# Dataset101 Alignment Validation",
        "",
        f"- imagesTs count: {summary['imagesTs_count']}",
        f"- heldout_labelsTs count: {summary['heldout_labelsTs_count']}",
        f"- matched cases: {summary['matched_cases']}",
        f"- positive cases: {summary['positive_cases']}",
        f"- negative cases: {summary['negative_cases']}",
        f"- nonbinary label cases: {len(summary['nonbinary_label_cases'])}",
        f"- shape mismatch cases: {len(summary['shape_mismatch_cases'])}",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, summary = scan_dataset(args.dataset_root)
    write_outputs(rows, summary, args.out, args.num_overlays)
    print(f"[done] Dataset101 alignment artifacts written under {args.out}")
    if summary["missing_label_cases"] or summary["extra_label_cases"] or summary["nonbinary_label_cases"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
