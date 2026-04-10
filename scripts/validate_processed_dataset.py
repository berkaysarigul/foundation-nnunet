"""Validate a regenerated processed SIIM dataset and optionally export previews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_manifest import (
    DATASET_MANIFEST_FILENAME,
    compute_split_fingerprint,
    fingerprint_directory,
    sha256_file,
    summarize_mask_directory,
    summarize_splits,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VALIDATION FAILED: {message}")


def export_overlay(image_path: Path, mask_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    overlay = np.array(image, dtype=np.uint8)
    overlay[mask > 0] = [255, 0, 0]
    blended = Image.blend(image, Image.fromarray(overlay), alpha=0.35)
    blended.save(output_path)


def main(dataset_dir: str, preview_dir: str | None, preview_limit: int) -> None:
    root = Path(dataset_dir)
    manifest_path = root / DATASET_MANIFEST_FILENAME
    splits_path = root / "splits.json"
    mask_variants_path = root / "mask_variants.json"
    images_dir = root / "images"
    original_masks_dir = root / "original_masks"
    dilated_masks_dir = root / "dilated_masks"

    for required in [manifest_path, splits_path, mask_variants_path, images_dir, original_masks_dir, dilated_masks_dir]:
        _check(required.exists(), f"Missing required processed-dataset path: {required}")

    manifest = _load_json(manifest_path)
    splits = _load_json(splits_path)
    mask_variants = _load_json(mask_variants_path)

    image_size = int(manifest["generation"]["image_size"])
    original_stats, positive_ids = summarize_mask_directory(original_masks_dir, image_size=image_size)
    dilated_stats, _ = summarize_mask_directory(dilated_masks_dir, image_size=image_size)
    split_summary = summarize_splits(splits, positive_ids)

    image_names = sorted(path.stem for path in images_dir.glob("*.png"))
    original_mask_names = sorted(path.stem for path in original_masks_dir.glob("*.png"))
    dilated_mask_names = sorted(path.stem for path in dilated_masks_dir.glob("*.png"))
    _check(image_names == original_mask_names == dilated_mask_names, "Image and mask filename sets differ")

    split_union = set()
    for split_name, image_ids in splits.items():
        split_set = set(image_ids)
        _check(len(split_set) == len(image_ids), f"Split '{split_name}' contains duplicate image IDs")
        _check(split_union.isdisjoint(split_set), f"Split '{split_name}' overlaps another split")
        split_union.update(split_set)
    _check(split_union == set(image_names), "Split union does not match processed image IDs")

    recomputed_fingerprints = {
        "images": fingerprint_directory(images_dir)["fingerprint"],
        "original_masks": fingerprint_directory(original_masks_dir)["fingerprint"],
        "dilated_masks": fingerprint_directory(dilated_masks_dir)["fingerprint"],
        "splits": compute_split_fingerprint(splits),
        "mask_variants": sha256_file(mask_variants_path),
        "splits_file": sha256_file(splits_path),
    }

    _check(manifest["mask_variants"] == mask_variants, "mask_variants.json does not match dataset manifest")
    _check(manifest["mask_statistics"]["original_masks"] == original_stats, "Original mask stats mismatch manifest")
    _check(manifest["mask_statistics"]["dilated_masks"] == dilated_stats, "Dilated mask stats mismatch manifest")
    _check(manifest["split_summary"] == split_summary, "Split summary mismatch manifest")
    _check(manifest["fingerprints"] == recomputed_fingerprints, "Fingerprint block mismatch manifest")
    _check(manifest["counts"]["images"] == len(image_names), "Image count mismatch manifest")
    _check(manifest["counts"]["positive_images"] == original_stats["positive_image_count"], "Positive count mismatch manifest")
    _check(original_stats["binary_unique_values_ok"], "Original masks are not binary")
    _check(dilated_stats["binary_unique_values_ok"], "Dilated masks are not binary")
    _check(mask_variants["default_train_mask_variant"] == "dilated_masks", "Unexpected default train mask variant")
    _check(mask_variants["default_eval_mask_variant"] == "original_masks", "Unexpected default eval mask variant")

    if preview_dir:
        out_dir = Path(preview_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        positive_preview_ids = sorted(positive_ids)[:preview_limit]
        negative_preview_ids = [image_id for image_id in image_names if image_id not in positive_ids][:preview_limit]
        for image_id in positive_preview_ids + negative_preview_ids:
            export_overlay(images_dir / f"{image_id}.png", original_masks_dir / f"{image_id}.png", out_dir / f"{image_id}_original_overlay.png")
            export_overlay(images_dir / f"{image_id}.png", dilated_masks_dir / f"{image_id}.png", out_dir / f"{image_id}_dilated_overlay.png")

    print(f"dataset_dir={root}")
    print(f"dataset_version={manifest['dataset_version']}")
    print(f"dataset_fingerprint={manifest['dataset_fingerprint']}")
    print(f"images={len(image_names)}")
    print(f"positive_images={original_stats['positive_image_count']}")
    print(f"negative_images={original_stats['negative_image_count']}")
    print(f"split_fingerprint={manifest['fingerprints']['splits']}")
    if preview_dir:
        print(f"preview_dir={preview_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a processed SIIM dataset")
    parser.add_argument("--dataset_dir", required=True, help="Processed dataset root to validate")
    parser.add_argument("--preview_dir", help="Optional directory for overlay previews")
    parser.add_argument("--preview_limit", type=int, default=3, help="Preview count per class subset")
    args = parser.parse_args()
    main(args.dataset_dir, args.preview_dir, args.preview_limit)
