"""Build the PR-11 SIIM-ACR PNG/mask manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pr11_utils import (
    image_size,
    infer_siim_filename_parts,
    mask_stats,
    parse_bool,
    percentile_summary,
    read_csv,
    repo_relative,
    sanitize_identifier,
    sha256_file,
    write_csv,
    write_markdown_report,
    write_yaml,
)


MANIFEST_COLUMNS = [
    "case_id",
    "source_dataset",
    "source_split",
    "original_image_id",
    "image_filename",
    "mask_filename",
    "image_path",
    "mask_path",
    "target_from_csv",
    "target_from_filename",
    "mask_is_positive",
    "gt_foreground_pixels",
    "width",
    "height",
    "image_hash",
    "mask_hash",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SIIM-ACR PNG manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-conflicts", type=parse_bool, default=False)
    parser.add_argument("--hash-files", type=parse_bool, default=False)
    return parser.parse_args(argv)


def _find_one(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name!r} under {root}")
    return matches[0]


def _load_stage_csv(path: Path, split_name: str) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    by_filename: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = str(row.get("new_filename", "")).strip()
        if not filename:
            raise ValueError(f"{path} contains a row without new_filename")
        if filename in by_filename:
            raise ValueError(f"{path} contains duplicate new_filename {filename!r}")
        by_filename[filename] = {
            "source_split": split_name,
            "original_image_id": str(row.get("ImageId", "")).strip(),
            "target_from_csv": str(row.get("has_pneumo", "")).strip(),
        }
    return by_filename


def _target_to_bool(raw: str) -> bool | None:
    text = str(raw).strip()
    if text in {"0", "0.0", "false", "False"}:
        return False
    if text in {"1", "1.0", "true", "True"}:
        return True
    return None


def build_manifest(
    root: Path,
    *,
    hash_files: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    images_dir = _find_one(root, "png_images")
    masks_dir = _find_one(root, "png_masks")
    train_csv = _find_one(root, "stage_1_train_images.csv")
    test_csv = _find_one(root, "stage_1_test_images.csv")

    csv_rows: dict[str, dict[str, str]] = {}
    csv_rows.update(_load_stage_csv(train_csv, "stage1_train"))
    csv_rows.update(_load_stage_csv(test_csv, "stage1_test"))

    image_paths = {p.name: p for p in sorted(images_dir.glob("*.png"))}
    mask_paths = {p.name: p for p in sorted(masks_dir.glob("*.png"))}
    unmatched_images = [
        {"image_filename": name, "image_path": repo_relative(path)}
        for name, path in sorted(image_paths.items())
        if name not in mask_paths
    ]
    unmatched_masks = [
        {"mask_filename": name, "mask_path": repo_relative(path)}
        for name, path in sorted(mask_paths.items())
        if name not in image_paths
    ]

    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    foreground_pixels: list[int] = []
    for filename in sorted(image_paths):
        if filename not in mask_paths:
            continue
        image_path = image_paths[filename]
        mask_path = mask_paths[filename]
        csv_meta = csv_rows.get(filename, {})
        parts = infer_siim_filename_parts(filename)
        image_width, image_height = image_size(image_path)
        stats = mask_stats(mask_path)
        foreground_pixels.append(int(stats["foreground_pixels"]))
        target_from_csv = csv_meta.get("target_from_csv", "")
        target_from_filename = parts["target"]
        csv_bool = _target_to_bool(target_from_csv)
        filename_bool = _target_to_bool(target_from_filename)
        mask_bool = bool(stats["is_positive"])

        row_conflicts: list[str] = []
        if image_width != int(stats["width"]) or image_height != int(stats["height"]):
            row_conflicts.append("image_mask_size_mismatch")
        if not bool(stats["is_binary"]):
            row_conflicts.append("non_binary_mask")
        if csv_bool is not None and csv_bool != mask_bool:
            row_conflicts.append("csv_target_vs_mask_foreground")
        if filename_bool is not None and filename_bool != mask_bool:
            row_conflicts.append("filename_target_vs_mask_foreground")
        if filename not in csv_rows:
            row_conflicts.append("missing_csv_row")

        case_id = sanitize_identifier(filename, prefix="siimacr")
        row = {
            "case_id": case_id,
            "source_dataset": "SIIM_ACR_PNG",
            "source_split": csv_meta.get("source_split", "unknown"),
            "original_image_id": csv_meta.get("original_image_id", ""),
            "image_filename": filename,
            "mask_filename": filename,
            "image_path": repo_relative(image_path),
            "mask_path": repo_relative(mask_path),
            "target_from_csv": target_from_csv,
            "target_from_filename": target_from_filename,
            "mask_is_positive": mask_bool,
            "gt_foreground_pixels": int(stats["foreground_pixels"]),
            "width": image_width,
            "height": image_height,
            "image_hash": sha256_file(image_path) if hash_files else "",
            "mask_hash": sha256_file(mask_path) if hash_files else "",
        }
        rows.append(row)
        for conflict_type in row_conflicts:
            conflicts.append(
                {
                    "case_id": case_id,
                    "image_filename": filename,
                    "conflict_type": conflict_type,
                    "target_from_csv": target_from_csv,
                    "target_from_filename": target_from_filename,
                    "mask_is_positive": mask_bool,
                    "gt_foreground_pixels": int(stats["foreground_pixels"]),
                    "image_path": repo_relative(image_path),
                    "mask_path": repo_relative(mask_path),
                }
            )

    case_id_counts: dict[str, int] = {}
    for row in rows:
        case_id_counts[row["case_id"]] = case_id_counts.get(row["case_id"], 0) + 1
    duplicate_case_ids = sorted(case_id for case_id, count in case_id_counts.items() if count > 1)
    for case_id in duplicate_case_ids:
        conflicts.append({"case_id": case_id, "conflict_type": "duplicate_case_id"})

    summary = {
        "schema_version": 1,
        "root": repo_relative(root),
        "row_count": len(rows),
        "positive_count": sum(1 for r in rows if r["mask_is_positive"]),
        "negative_count": sum(1 for r in rows if not r["mask_is_positive"]),
        "stage_counts": _counts_by(rows, "source_split"),
        "unmatched_images": len(unmatched_images),
        "unmatched_masks": len(unmatched_masks),
        "conflicts": len(conflicts),
        "foreground_pixels": percentile_summary(foreground_pixels),
        "hash_files": bool(hash_files),
    }
    return rows, conflicts, unmatched_images, unmatched_masks, summary


def _counts_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def write_report(path: Path, summary: dict[str, Any], conflicts: list[dict[str, Any]]) -> None:
    body = [
        f"- rows: `{summary['row_count']}`",
        f"- positives: `{summary['positive_count']}`",
        f"- negatives: `{summary['negative_count']}`",
        f"- stage counts: `{summary['stage_counts']}`",
        f"- unmatched images: `{summary['unmatched_images']}`",
        f"- unmatched masks: `{summary['unmatched_masks']}`",
        f"- conflicts: `{summary['conflicts']}`",
    ]
    conflict_body = [
        f"- `{row.get('case_id', '')}`: `{row.get('conflict_type', '')}`"
        for row in conflicts[:25]
    ] or ["- none"]
    write_markdown_report(
        path,
        "PR-11 SIIM-ACR PNG Manifest Report",
        [("Summary", body), ("Conflict Sample", conflict_body)],
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, conflicts, unmatched_images, unmatched_masks, summary = build_manifest(
        args.root,
        hash_files=bool(args.hash_files),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "siim_acr_png_manifest.csv", rows, MANIFEST_COLUMNS)
    write_yaml(args.out / "siim_acr_png_manifest_summary.yaml", summary)
    write_report(args.out / "siim_acr_png_manifest_report.md", summary, conflicts)
    write_csv(args.out / "siim_acr_png_conflicts.csv", conflicts)
    write_csv(args.out / "siim_acr_png_unmatched_images.csv", unmatched_images)
    write_csv(args.out / "siim_acr_png_unmatched_masks.csv", unmatched_masks)
    if conflicts and not bool(args.allow_conflicts):
        print(
            f"[failed] wrote manifest artifacts, but found {len(conflicts)} conflict(s). "
            "Pass --allow-conflicts true only for diagnostic output.",
            file=sys.stderr,
        )
        return 1
    print(f"[done] SIIM manifest written to {args.out / 'siim_acr_png_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
