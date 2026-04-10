"""Dataset manifest and fingerprint helpers for trusted processed SIIM outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

DEFAULT_DATASET_VERSION = "pneumothorax_trusted_v1"
DEFAULT_DATASET_ROOT = Path("data/processed") / DEFAULT_DATASET_VERSION
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"


def _to_posix(path: Path) -> str:
    return path.as_posix()


def repo_relative_or_posix(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return _to_posix(resolved.relative_to(repo_root))
    except ValueError:
        return _to_posix(resolved)


def repo_root_from_here() -> Path:
    """Return the repository root based on this module location."""
    return Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_split_fingerprint(splits: dict[str, list[str]]) -> str:
    normalized = {name: sorted(image_ids) for name, image_ids in sorted(splits.items())}
    return sha256_json(normalized)


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def build_code_identity(repo_root: Path, code_paths: Iterable[Path]) -> dict[str, object]:
    scope = sorted({_to_posix(path.relative_to(repo_root)) for path in code_paths})
    scope_payload = []
    for relative_path in scope:
        absolute_path = repo_root / relative_path
        scope_payload.append(
            {
                "path": relative_path,
                "sha256": sha256_file(absolute_path),
            }
        )

    return {
        "git_revision": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_output(repo_root, "status", "--porcelain")),
        "code_fingerprint": sha256_json(scope_payload),
        "code_fingerprint_scope": scope,
    }


def fingerprint_directory(directory: Path) -> dict[str, object]:
    entries = []
    for file_path in sorted(directory.glob("*.png")):
        entries.append(
            {
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return {
        "file_count": len(entries),
        "fingerprint": sha256_json(entries),
    }


def summarize_mask_directory(mask_dir: Path, image_size: int) -> tuple[dict[str, object], set[str]]:
    positive_ids: set[str] = set()
    positive_pixels: list[int] = []
    binary_unique_values_ok = True
    invalid_examples: list[str] = []
    total_foreground_pixels = 0
    image_count = 0

    for mask_path in sorted(mask_dir.glob("*.png")):
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        image_count += 1
        unique_values = np.unique(mask).tolist()
        if not set(unique_values).issubset({0, 255}):
            binary_unique_values_ok = False
            if len(invalid_examples) < 10:
                invalid_examples.append(mask_path.name)
        foreground_pixels = int((mask > 0).sum())
        total_foreground_pixels += foreground_pixels
        if foreground_pixels > 0:
            positive_ids.add(mask_path.stem)
            positive_pixels.append(foreground_pixels)

    total_pixels = image_count * image_size * image_size
    positive_pixels_array = np.array(positive_pixels, dtype=np.int64) if positive_pixels else np.array([], dtype=np.int64)
    stats = {
        "image_count": image_count,
        "positive_image_count": int(len(positive_pixels)),
        "negative_image_count": int(image_count - len(positive_pixels)),
        "positive_image_ratio": (len(positive_pixels) / image_count) if image_count else 0.0,
        "total_foreground_pixels": int(total_foreground_pixels),
        "foreground_fraction_all_pixels": (total_foreground_pixels / total_pixels) if total_pixels else 0.0,
        "mean_foreground_pixels_per_image": (total_foreground_pixels / image_count) if image_count else 0.0,
        "mean_foreground_pixels_positive": float(positive_pixels_array.mean()) if positive_pixels else 0.0,
        "median_foreground_pixels_positive": float(np.median(positive_pixels_array)) if positive_pixels else 0.0,
        "p10_foreground_pixels_positive": float(np.percentile(positive_pixels_array, 10)) if positive_pixels else 0.0,
        "p25_foreground_pixels_positive": float(np.percentile(positive_pixels_array, 25)) if positive_pixels else 0.0,
        "p75_foreground_pixels_positive": float(np.percentile(positive_pixels_array, 75)) if positive_pixels else 0.0,
        "max_foreground_pixels_positive": int(positive_pixels_array.max()) if positive_pixels else 0,
        "binary_unique_values_ok": binary_unique_values_ok,
        "invalid_binary_examples": invalid_examples,
    }
    return stats, positive_ids


def summarize_splits(splits: dict[str, list[str]], positive_ids: set[str]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for split_name, image_ids in sorted(splits.items()):
        positive_count = sum(1 for image_id in image_ids if image_id in positive_ids)
        total_count = len(image_ids)
        summary[split_name] = {
            "image_count": total_count,
            "positive_image_count": positive_count,
            "negative_image_count": total_count - positive_count,
            "positive_image_ratio": (positive_count / total_count) if total_count else 0.0,
        }
    return summary


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_dataset_manifest(
    *,
    dataset_version: str,
    dataset_root: Path,
    raw_dir: Path,
    annotation_csv_path: Path,
    dicom_dir: Path,
    image_size: int,
    seed: int,
    resolved_rle_mode: str,
    skipped_images: int,
    splits: dict[str, list[str]],
    mask_variant_manifest: dict[str, object],
    code_paths: Iterable[Path],
) -> dict[str, object]:
    repo_root = repo_root_from_here()
    images_dir = dataset_root / "images"
    original_masks_dir = dataset_root / "original_masks"
    dilated_masks_dir = dataset_root / "dilated_masks"
    splits_path = dataset_root / "splits.json"
    mask_variants_path = dataset_root / "mask_variants.json"

    code_identity = build_code_identity(repo_root, code_paths)
    images_fingerprint = fingerprint_directory(images_dir)
    original_mask_fingerprint = fingerprint_directory(original_masks_dir)
    dilated_mask_fingerprint = fingerprint_directory(dilated_masks_dir)
    original_mask_stats, positive_ids = summarize_mask_directory(original_masks_dir, image_size=image_size)
    dilated_mask_stats, _ = summarize_mask_directory(dilated_masks_dir, image_size=image_size)
    split_summary = summarize_splits(splits, positive_ids)
    split_fingerprint = compute_split_fingerprint(splits)

    manifest = {
        "dataset_version": dataset_version,
        "generated_at_utc": utc_now_iso(),
        "dataset_root": repo_relative_or_posix(dataset_root, repo_root),
        "raw_source": {
            "raw_dir": repo_relative_or_posix(raw_dir, repo_root),
            "annotation_csv_path": repo_relative_or_posix(annotation_csv_path, repo_root),
            "annotation_csv_sha256": sha256_file(annotation_csv_path),
            "dicom_dir": repo_relative_or_posix(dicom_dir, repo_root),
        },
        "generation": {
            "image_size": image_size,
            "seed": seed,
            "resolved_rle_mode": resolved_rle_mode,
            "split_policy": "random_unstratified_70_15_15",
            "skipped_images": skipped_images,
            "generator_code_revision": code_identity["git_revision"],
            "generator_git_dirty": code_identity["git_dirty"],
            "generator_code_fingerprint": code_identity["code_fingerprint"],
            "generator_code_fingerprint_scope": code_identity["code_fingerprint_scope"],
        },
        "processed_layout": {
            "images_dir": "images",
            "original_masks_dir": "original_masks",
            "dilated_masks_dir": "dilated_masks",
            "mask_variants_manifest": "mask_variants.json",
            "splits_manifest": "splits.json",
            "dataset_manifest": DATASET_MANIFEST_FILENAME,
        },
        "counts": {
            "images": images_fingerprint["file_count"],
            "original_masks": original_mask_fingerprint["file_count"],
            "dilated_masks": dilated_mask_fingerprint["file_count"],
            "positive_images": original_mask_stats["positive_image_count"],
            "negative_images": original_mask_stats["negative_image_count"],
        },
        "mask_variants": mask_variant_manifest,
        "mask_statistics": {
            "original_masks": original_mask_stats,
            "dilated_masks": dilated_mask_stats,
        },
        "split_summary": split_summary,
        "fingerprints": {
            "images": images_fingerprint["fingerprint"],
            "original_masks": original_mask_fingerprint["fingerprint"],
            "dilated_masks": dilated_mask_fingerprint["fingerprint"],
            "splits": split_fingerprint,
            "mask_variants": sha256_file(mask_variants_path),
            "splits_file": sha256_file(splits_path),
        },
    }
    manifest["dataset_fingerprint"] = sha256_json(manifest)
    return manifest
