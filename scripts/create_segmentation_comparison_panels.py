"""Create PR-11 visual comparison panels for segmentation predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from pr11_utils import REPO_ROOT, read_csv, resolve_existing_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create segmentation comparison panels.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fx-prior-manifest", type=Path, required=True)
    parser.add_argument("--image-only-preds", type=Path, required=True)
    parser.add_argument("--image-prior-preds", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cases-per-category", type=int, default=12)
    parser.add_argument("--panel-size", type=int, default=256)
    return parser.parse_args(argv)


def _row_map(path: Path, key: str = "case_id") -> dict[str, dict[str, str]]:
    return {row[key]: row for row in read_csv(path) if row.get(key)}


def _pred_path(root: Path, case_id: str) -> Path | None:
    for name in (f"{case_id}.png", f"{case_id}_0000.png"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    matches = sorted(root.glob(f"{case_id}*.png"))
    return matches[0] if matches else None


def _load_gray(path: Path | None, size: int) -> Image.Image:
    if path is None or not path.exists():
        return Image.new("L", (size, size), 0)
    with Image.open(path) as image:
        return image.convert("L").resize((size, size), Image.Resampling.BILINEAR)


def _mask(path: Path | None, size: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((size, size), dtype=np.uint8)
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return (np.asarray(image) > 0).astype(np.uint8)


def _overlay(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgb = np.asarray(base.convert("RGB"), dtype=np.float32)
    m = mask > 0
    rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(color, dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _error_map(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    rgb[(gt > 0) & (pred > 0)] = [255, 220, 0]
    rgb[(gt == 0) & (pred > 0)] = [255, 0, 0]
    rgb[(gt > 0) & (pred == 0)] = [0, 180, 255]
    return Image.fromarray(rgb)


def _caption(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 22), "white")
    canvas.paste(image.convert("RGB"), (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), text, fill=(0, 0, 0))
    return canvas


def _panel(case_id: str, manifest_row: dict[str, str], fx_row: dict[str, str], args: argparse.Namespace) -> Image.Image:
    size = int(args.panel_size)
    image_path = resolve_existing_path(manifest_row.get("image_path", ""), REPO_ROOT)
    mask_path = resolve_existing_path(manifest_row.get("mask_path", ""), REPO_ROOT)
    fx_path = resolve_existing_path(
        fx_row.get("aligned_probability_map_path") or fx_row.get("probability_map_path", ""),
        REPO_ROOT,
    )
    image = _load_gray(image_path, size)
    gt = _mask(mask_path, size)
    fx = _load_gray(fx_path, size)
    pred_image_only_path = _pred_path(args.image_only_preds, case_id)
    pred_image_prior_path = _pred_path(args.image_prior_preds, case_id)
    pred_image_only = _mask(pred_image_only_path, size)
    pred_image_prior = _mask(pred_image_prior_path, size)

    tiles = [
        _caption(image, "X-ray"),
        _caption(_overlay(image, gt, (0, 255, 0)), "GT"),
        _caption(fx, "Foundation X prob"),
        _caption(_overlay(image, pred_image_only, (255, 0, 0)), "nnU-Net image"),
        _caption(_overlay(image, pred_image_prior, (255, 0, 0)), "nnU-Net image+prior"),
        _caption(_error_map(gt, pred_image_prior), "Error"),
    ]
    canvas = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height), "white")
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width
    return canvas


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def _bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true", "yes"}


def _select_categories(metrics_rows: list[dict[str, str]], limit: int) -> dict[str, list[dict[str, str]]]:
    rows = [r for r in metrics_rows if r.get("case_id")]
    categories = {
        "best_improvements": sorted(
            rows,
            key=lambda r: _float(r, "image_prior_dice", _float(r, "dice")) - _float(r, "image_only_dice", 0.0),
            reverse=True,
        )[:limit],
        "worst_degradations": sorted(
            rows,
            key=lambda r: _float(r, "image_prior_dice", _float(r, "dice")) - _float(r, "image_only_dice", 0.0),
        )[:limit],
        "false_positives": [r for r in rows if not _bool(r, "gt_is_positive") and _bool(r, "pred_is_positive")][:limit],
        "false_negatives": [r for r in rows if _bool(r, "gt_is_positive") and not _bool(r, "pred_is_positive")][:limit],
        "small_lesions": sorted(
            [r for r in rows if _bool(r, "gt_is_positive")],
            key=lambda r: _float(r, "gt_foreground_pixels"),
        )[:limit],
        "large_lesions": sorted(
            [r for r in rows if _bool(r, "gt_is_positive")],
            key=lambda r: _float(r, "gt_foreground_pixels"),
            reverse=True,
        )[:limit],
    }
    return categories


def create_panels(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _row_map(args.manifest)
    fx_rows = _row_map(args.fx_prior_manifest)
    metrics_rows = read_csv(args.metrics)
    categories = _select_categories(metrics_rows, int(args.max_cases_per_category))
    args.out.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for category, rows in categories.items():
        count = 0
        for row in rows:
            case_id = row["case_id"]
            if case_id not in manifest or case_id not in fx_rows:
                continue
            panel = _panel(case_id, manifest[case_id], fx_rows[case_id], args)
            category_dir = args.out / category
            category_dir.mkdir(parents=True, exist_ok=True)
            panel.save(category_dir / f"{case_id}.png")
            count += 1
        written[category] = count
    summary = {"out": str(args.out), "categories": written}
    (args.out / "summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = create_panels(args)
    print(f"[done] visual panels written under {summary['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

