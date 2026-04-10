"""Audit helper for the local SIIM DICOM intensity policy."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dicom_intensity import prepare_dicom_pixels_for_png, read_dicom_dataset


def _iter_dicom_files(dicom_root: Path) -> list[Path]:
    return list(dicom_root.rglob("*.dcm"))


def _metadata_audit(files: list[Path]) -> dict:
    counters = defaultdict(Counter)
    missing = Counter()
    for index, dicom_path in enumerate(files, 1):
        ds = read_dicom_dataset(dicom_path, stop_before_pixels=True)
        for key in (
            "PhotometricInterpretation",
            "Modality",
            "BitsStored",
            "BitsAllocated",
            "PixelRepresentation",
            "SamplesPerPixel",
            "RescaleSlope",
            "RescaleIntercept",
            "WindowCenter",
            "WindowWidth",
            "VOILUTFunction",
        ):
            value = getattr(ds, key, None)
            if value is None:
                missing[key] += 1
            else:
                counters[key][str(value)] += 1

        if index % 2000 == 0:
            print(f"metadata_progress={index}")

    return {
        "total_files": len(files),
        "counters": {key: counter.most_common(10) for key, counter in counters.items()},
        "missing": dict(missing),
    }


def _sample_pixel_audit(files: list[Path], sample_size: int) -> dict:
    rng = random.Random(42)
    sample = files if len(files) <= sample_size else rng.sample(files, sample_size)

    pixel_mins = []
    pixel_maxs = []
    mean_values = []
    p01_values = []
    p99_values = []
    max_lt_255 = 0
    examples = []

    for dicom_path in sample:
        ds = read_dicom_dataset(dicom_path, stop_before_pixels=False)
        pixels = ds.pixel_array.astype(np.float32)
        pixel_mins.append(float(pixels.min()))
        pixel_maxs.append(float(pixels.max()))
        mean_values.append(float(pixels.mean()))
        p01_values.append(float(np.percentile(pixels, 1)))
        p99_values.append(float(np.percentile(pixels, 99)))
        if float(pixels.max()) < 255.0:
            max_lt_255 += 1

        if len(examples) < 5:
            png_pixels, transforms = prepare_dicom_pixels_for_png(ds)
            examples.append(
                {
                    "file": dicom_path.name,
                    "min": float(pixels.min()),
                    "max": float(pixels.max()),
                    "mean": float(pixels.mean()),
                    "photometric": str(getattr(ds, "PhotometricInterpretation", None)),
                    "png_min": int(png_pixels.min()),
                    "png_max": int(png_pixels.max()),
                    "applied_transforms": transforms,
                }
            )

    return {
        "sample_size": len(sample),
        "max_lt_255": max_lt_255,
        "pixel_min_range": [float(min(pixel_mins)), float(max(pixel_mins))],
        "pixel_max_range": [float(min(pixel_maxs)), float(max(pixel_maxs))],
        "pixel_mean_median": float(np.median(mean_values)),
        "p01_median": float(np.median(p01_values)),
        "p99_median": float(np.median(p99_values)),
        "examples": examples,
    }


def _export_preview_images(files: list[Path], export_dir: Path, limit: int) -> list[str]:
    export_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for dicom_path in files[:limit]:
        ds = read_dicom_dataset(dicom_path, stop_before_pixels=False)
        png_pixels, transforms = prepare_dicom_pixels_for_png(ds)
        suffix = "preserved" if not transforms else "-".join(transforms)
        output_path = export_dir / f"{dicom_path.stem}_{suffix}.png"
        Image.fromarray(png_pixels).save(output_path)
        exported.append(str(output_path))
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local SIIM DICOM intensity metadata")
    parser.add_argument(
        "--dicom_root",
        default="data/raw/SIIM-ACR/dicom-images-train",
        help="Root directory containing training DICOM files",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=512,
        help="Sample size for pixel-stat audit",
    )
    parser.add_argument(
        "--preview_dir",
        default="",
        help="Optional directory for exported PNG preview images",
    )
    parser.add_argument(
        "--preview_limit",
        type=int,
        default=3,
        help="How many preview PNGs to export when preview_dir is set",
    )
    args = parser.parse_args()

    dicom_root = Path(args.dicom_root)
    files = _iter_dicom_files(dicom_root)
    if not files:
        raise FileNotFoundError(f"No DICOM files found under {dicom_root}")

    metadata = _metadata_audit(files)
    pixel_stats = _sample_pixel_audit(files, sample_size=args.sample_size)
    result = {
        "dicom_root": str(dicom_root),
        "metadata": metadata,
        "pixel_stats": pixel_stats,
    }

    if args.preview_dir:
        preview_paths = _export_preview_images(files, Path(args.preview_dir), args.preview_limit)
        result["preview_paths"] = preview_paths

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
