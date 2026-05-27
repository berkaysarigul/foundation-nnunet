"""Audit the PR-11 SIIM-ACR PNG manifest and export visual sanity overlays."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from pr11_utils import (
    REPO_ROOT,
    image_size,
    make_overlay,
    mask_stats,
    percentile_summary,
    read_csv,
    resolve_existing_path,
    write_csv,
    write_markdown_report,
    write_yaml,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SIIM-ACR PNG manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-row-count", type=int, default=12047)
    return parser.parse_args(argv)


def _bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes"}


def audit_manifest(manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    rows = read_csv(manifest)
    seen_case_ids: set[str] = set()
    duplicates: set[str] = set()
    audit_rows: list[dict[str, Any]] = []
    foreground: list[int] = []
    size_counts: dict[str, int] = {}
    mask_size_counts: dict[str, int] = {}
    positives = 0
    negatives = 0
    stage_counts: dict[str, int] = {}
    conflicts: list[dict[str, str]] = []

    for row in rows:
        case_id = row.get("case_id", "")
        if case_id in seen_case_ids:
            duplicates.add(case_id)
        seen_case_ids.add(case_id)
        image_path = resolve_existing_path(row.get("image_path", ""), REPO_ROOT)
        mask_path = resolve_existing_path(row.get("mask_path", ""), REPO_ROOT)
        issues: list[str] = []
        image_exists = image_path.is_file()
        mask_exists = mask_path.is_file()
        width = height = mask_width = mask_height = 0
        gt_pixels = int(float(row.get("gt_foreground_pixels", "0") or 0))
        mask_positive = _bool(row.get("mask_is_positive", "false"))
        if not image_exists:
            issues.append("missing_image_path")
        if not mask_exists:
            issues.append("missing_mask_path")
        if image_exists:
            width, height = image_size(image_path)
            size_counts[f"{width}x{height}"] = size_counts.get(f"{width}x{height}", 0) + 1
        if mask_exists:
            stats = mask_stats(mask_path)
            mask_width = int(stats["width"])
            mask_height = int(stats["height"])
            mask_size_counts[f"{mask_width}x{mask_height}"] = mask_size_counts.get(f"{mask_width}x{mask_height}", 0) + 1
            if not stats["is_binary"]:
                issues.append("non_binary_mask")
            if int(stats["foreground_pixels"]) != gt_pixels:
                issues.append("foreground_pixel_mismatch")
            if bool(stats["is_positive"]) != mask_positive:
                issues.append("mask_positive_mismatch")
        if image_exists and mask_exists and (width, height) != (mask_width, mask_height):
            issues.append("image_mask_size_mismatch")
        if mask_positive and gt_pixels <= 0:
            issues.append("positive_without_foreground")
        if (not mask_positive) and gt_pixels > 0:
            issues.append("negative_with_foreground")
        if case_id in duplicates:
            issues.append("duplicate_case_id")

        if mask_positive:
            positives += 1
        else:
            negatives += 1
        foreground.append(gt_pixels)
        split = row.get("source_split", "")
        stage_counts[split] = stage_counts.get(split, 0) + 1

        audit_row = {
            "case_id": case_id,
            "source_split": split,
            "image_path": row.get("image_path", ""),
            "mask_path": row.get("mask_path", ""),
            "image_exists": image_exists,
            "mask_exists": mask_exists,
            "width": width,
            "height": height,
            "mask_width": mask_width,
            "mask_height": mask_height,
            "mask_is_positive": mask_positive,
            "gt_foreground_pixels": gt_pixels,
            "issues": ";".join(issues),
        }
        audit_rows.append(audit_row)
        if issues:
            conflicts.append({"case_id": case_id, "issues": ";".join(issues)})

    summary = {
        "schema_version": 1,
        "manifest": str(manifest),
        "row_count": len(rows),
        "unique_case_ids": len(seen_case_ids),
        "duplicate_case_ids": sorted(duplicates),
        "positive_count": positives,
        "negative_count": negatives,
        "stage_counts": dict(sorted(stage_counts.items())),
        "image_size_distribution": dict(sorted(size_counts.items())),
        "mask_size_distribution": dict(sorted(mask_size_counts.items())),
        "foreground_pixels": percentile_summary(foreground),
        "conflict_count": len(conflicts),
        "status": "PASS" if not conflicts else "FAIL",
    }
    return audit_rows, summary, conflicts


def _sample(rows: list[dict[str, Any]], *, positive: bool, count: int, seed: int) -> list[dict[str, Any]]:
    subset = [r for r in rows if bool(r["mask_is_positive"]) is positive and not r["issues"]]
    rng = random.Random(seed + (1 if positive else 2))
    rng.shuffle(subset)
    return subset[:count]


def export_samples(audit_rows: list[dict[str, Any]], conflicts: list[dict[str, str]], out: Path, sample_count: int, seed: int) -> None:
    per_class = max(1, sample_count // 2)
    for folder, selected in [
        ("positive_samples", _sample(audit_rows, positive=True, count=per_class, seed=seed)),
        ("negative_samples", _sample(audit_rows, positive=False, count=per_class, seed=seed)),
    ]:
        for row in selected:
            make_overlay(
                resolve_existing_path(row["image_path"], REPO_ROOT),
                resolve_existing_path(row["mask_path"], REPO_ROOT),
                out / folder / f"{row['case_id']}.png",
            )
    conflict_ids = {row["case_id"] for row in conflicts[:sample_count]}
    for row in audit_rows:
        if row["case_id"] not in conflict_ids:
            continue
        image_path = resolve_existing_path(row["image_path"], REPO_ROOT)
        mask_path = resolve_existing_path(row["mask_path"], REPO_ROOT)
        if image_path.exists() and mask_path.exists():
            make_overlay(image_path, mask_path, out / "conflict_samples" / f"{row['case_id']}.png")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    body = [
        f"- status: `{summary['status']}`",
        f"- rows: `{summary['row_count']}`",
        f"- unique case IDs: `{summary['unique_case_ids']}`",
        f"- positives: `{summary['positive_count']}`",
        f"- negatives: `{summary['negative_count']}`",
        f"- stage counts: `{summary['stage_counts']}`",
        f"- conflicts: `{summary['conflict_count']}`",
        f"- image sizes: `{summary['image_size_distribution']}`",
        f"- mask sizes: `{summary['mask_size_distribution']}`",
    ]
    write_markdown_report(path, "PR-11 SIIM-ACR Manifest Audit", [("Summary", body)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit_rows, summary, conflicts = audit_manifest(args.manifest)
    if args.expected_row_count and int(summary["row_count"]) != int(args.expected_row_count):
        summary["status"] = "FAIL"
        summary["expected_row_count"] = int(args.expected_row_count)
        summary["row_count_matches_expected"] = False
    args.out.mkdir(parents=True, exist_ok=True)
    write_yaml(args.out / "summary.yaml", summary)
    write_csv(args.out / "manifest_audit.csv", audit_rows)
    write_report(args.out / "report.md", summary)
    export_samples(audit_rows, conflicts, args.out, int(args.sample_count), int(args.seed))
    print(f"[{summary['status']}] SIIM manifest audit written under {args.out}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

