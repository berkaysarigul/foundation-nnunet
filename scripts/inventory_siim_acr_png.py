"""Inventory the extracted SIIM-ACR PNG/mask repackage for PR-11."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pr11_utils import (
    infer_siim_filename_parts,
    read_csv,
    repo_relative,
    size_distribution,
    write_json,
    write_markdown_report,
    write_yaml,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory extracted SIIM-ACR PNG dataset.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--size-sample-limit", type=int, default=0)
    return parser.parse_args(argv)


def _find_one(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _csv_info(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"path": "", "exists": False, "row_count": 0, "headers": []}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return {
        "path": repo_relative(path),
        "exists": True,
        "row_count": len(rows),
        "headers": reader.fieldnames or [],
        "target_column_present": "has_pneumo" in (reader.fieldnames or []),
        "image_id_column_present": "ImageId" in (reader.fieldnames or []),
    }


def _filename_pattern_summary(paths: list[Path]) -> dict[str, Any]:
    parsed = [infer_siim_filename_parts(path.name) for path in paths]
    target_encoded = sum(1 for item in parsed if item["target"] != "")
    split_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for item in parsed:
        if item["split"]:
            split_counts[item["split"]] = split_counts.get(item["split"], 0) + 1
        if item["target"]:
            target_counts[item["target"]] = target_counts.get(item["target"], 0) + 1
    return {
        "target_encoded_count": target_encoded,
        "target_encoded_all": target_encoded == len(paths),
        "split_counts_from_filename": split_counts,
        "target_counts_from_filename": target_counts,
        "examples": [path.name for path in paths[:20]],
    }


def build_inventory(root: Path, sample_count: int, size_sample_limit: int) -> dict[str, Any]:
    images_dir = _find_one(root, "png_images")
    masks_dir = _find_one(root, "png_masks")
    train_csv = _find_one(root, "stage_1_train_images.csv")
    test_csv = _find_one(root, "stage_1_test_images.csv")

    image_paths = sorted(images_dir.glob("*.png")) if images_dir else []
    mask_paths = sorted(masks_dir.glob("*.png")) if masks_dir else []
    image_names = {p.name for p in image_paths}
    mask_names = {p.name for p in mask_paths}
    paired_names = sorted(image_names & mask_names)

    samples = []
    for name in paired_names[:sample_count]:
        samples.append(
            {
                "filename": name,
                "image_path": repo_relative((images_dir / name) if images_dir else Path(name)),
                "mask_path": repo_relative((masks_dir / name) if masks_dir else Path(name)),
                "filename_parts": infer_siim_filename_parts(name),
            }
        )

    discovered = {
        "root": repo_relative(root),
        "png_images_dir": repo_relative(images_dir) if images_dir else "",
        "png_masks_dir": repo_relative(masks_dir) if masks_dir else "",
        "stage_1_train_images_csv": repo_relative(train_csv) if train_csv else "",
        "stage_1_test_images_csv": repo_relative(test_csv) if test_csv else "",
        "image_files": [repo_relative(p) for p in image_paths],
        "mask_files": [repo_relative(p) for p in mask_paths],
    }

    return {
        "schema_version": 1,
        "root": repo_relative(root),
        "discovered": discovered,
        "counts": {
            "images": len(image_paths),
            "masks": len(mask_paths),
            "paired_by_filename": len(paired_names),
            "unmatched_images": len(image_names - mask_names),
            "unmatched_masks": len(mask_names - image_names),
        },
        "extensions": {
            "images": sorted({p.suffix.lower() for p in image_paths}),
            "masks": sorted({p.suffix.lower() for p in mask_paths}),
        },
        "csv": {
            "stage1_train": _csv_info(train_csv),
            "stage1_test": _csv_info(test_csv),
        },
        "filename_patterns": _filename_pattern_summary(image_paths),
        "size_distributions": {
            "images": size_distribution(image_paths, max_items=size_sample_limit),
            "masks": size_distribution(mask_paths, max_items=size_sample_limit),
        },
        "samples": samples,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    csv_info = summary["csv"]
    body = [
        f"- root: `{summary['root']}`",
        f"- images: `{counts['images']}`",
        f"- masks: `{counts['masks']}`",
        f"- paired by filename: `{counts['paired_by_filename']}`",
        f"- unmatched images: `{counts['unmatched_images']}`",
        f"- unmatched masks: `{counts['unmatched_masks']}`",
        f"- train CSV rows: `{csv_info['stage1_train']['row_count']}`",
        f"- test CSV rows: `{csv_info['stage1_test']['row_count']}`",
        f"- filename target encoding all images: `{summary['filename_patterns']['target_encoded_all']}`",
    ]
    samples = [
        f"- `{row['filename']}` -> `{row['image_path']}` / `{row['mask_path']}`"
        for row in summary["samples"]
    ]
    write_markdown_report(
        path,
        "PR-11 SIIM-ACR PNG Inventory",
        [("Summary", body), ("Sample Pairs", samples)],
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_inventory(args.root, int(args.sample_count), int(args.size_sample_limit))
    args.out.mkdir(parents=True, exist_ok=True)
    write_yaml(args.out / "inventory_summary.yaml", summary)
    write_json(args.out / "discovered_files.json", summary["discovered"])
    write_report(args.out / "inventory_report.md", summary)
    print(f"[done] SIIM inventory written under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

