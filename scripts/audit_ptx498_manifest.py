"""Audit the PR-11 PTX-498 manifest and export visual sanity overlays."""

from __future__ import annotations

import argparse
import random
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
    parser = argparse.ArgumentParser(description="Audit PTX-498 manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-row-count", type=int, default=498)
    return parser.parse_args(argv)


def _bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes"}


def audit_manifest(manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows = read_csv(manifest)
    audit_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    foreground: list[int] = []
    site_counts: dict[str, int] = {}
    image_size_counts: dict[str, int] = {}
    mask_size_counts: dict[str, int] = {}
    seen: set[str] = set()
    duplicates: set[str] = set()

    for row in rows:
        case_id = row.get("case_id", "")
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
        image_path = resolve_existing_path(row.get("image_path", ""), REPO_ROOT)
        mask_path = resolve_existing_path(row.get("mask_path", ""), REPO_ROOT)
        issues: list[str] = []
        image_exists = image_path.is_file()
        mask_exists = mask_path.is_file()
        width = height = mask_width = mask_height = 0
        if image_exists:
            width, height = image_size(image_path)
            image_size_counts[f"{width}x{height}"] = image_size_counts.get(f"{width}x{height}", 0) + 1
        else:
            issues.append("missing_image_path")
        fg = int(float(row.get("gt_foreground_pixels", "0") or 0))
        positive = _bool(row.get("mask_is_positive", "false"))
        if mask_exists:
            stats = mask_stats(mask_path)
            mask_width = int(stats["width"])
            mask_height = int(stats["height"])
            mask_size_counts[f"{mask_width}x{mask_height}"] = mask_size_counts.get(f"{mask_width}x{mask_height}", 0) + 1
            if not stats["is_binary"]:
                issues.append("non_binary_mask")
            if int(stats["foreground_pixels"]) != fg:
                issues.append("foreground_pixel_mismatch")
            if bool(stats["is_positive"]) != positive:
                issues.append("mask_positive_mismatch")
        else:
            issues.append("missing_mask_path")
        if image_exists and mask_exists and (width, height) != (mask_width, mask_height):
            issues.append("image_mask_size_mismatch")
        if case_id in duplicates:
            issues.append("duplicate_case_id")
        site = row.get("site", "")
        site_counts[site] = site_counts.get(site, 0) + 1
        foreground.append(fg)
        audit_row = {
            "case_id": case_id,
            "site": site,
            "image_path": row.get("image_path", ""),
            "mask_path": row.get("mask_path", ""),
            "image_exists": image_exists,
            "mask_exists": mask_exists,
            "width": width,
            "height": height,
            "mask_width": mask_width,
            "mask_height": mask_height,
            "mask_is_positive": positive,
            "gt_foreground_pixels": fg,
            "issues": ";".join(issues),
        }
        audit_rows.append(audit_row)
        if issues:
            conflicts.append({"case_id": case_id, "issues": ";".join(issues)})

    summary = {
        "schema_version": 1,
        "manifest": str(manifest),
        "row_count": len(rows),
        "unique_case_ids": len(seen),
        "duplicate_case_ids": sorted(duplicates),
        "positive_count": sum(1 for r in audit_rows if r["mask_is_positive"]),
        "negative_or_empty_mask_count": sum(1 for r in audit_rows if not r["mask_is_positive"]),
        "site_counts": dict(sorted(site_counts.items())),
        "image_size_distribution": dict(sorted(image_size_counts.items())),
        "mask_size_distribution": dict(sorted(mask_size_counts.items())),
        "foreground_pixels": percentile_summary(foreground),
        "conflict_count": len(conflicts),
        "status": "PASS" if not conflicts else "FAIL",
    }
    return audit_rows, summary, conflicts


def export_samples(audit_rows: list[dict[str, Any]], out: Path, sample_count: int, seed: int) -> None:
    candidates = [row for row in audit_rows if not row["issues"]]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    for row in candidates[:sample_count]:
        make_overlay(
            resolve_existing_path(row["image_path"], REPO_ROOT),
            resolve_existing_path(row["mask_path"], REPO_ROOT),
            out / "samples" / f"{row['case_id']}.png",
        )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    body = [
        f"- status: `{summary['status']}`",
        f"- rows: `{summary['row_count']}`",
        f"- positives: `{summary['positive_count']}`",
        f"- empty/negative masks: `{summary['negative_or_empty_mask_count']}`",
        f"- site counts: `{summary['site_counts']}`",
        f"- conflicts: `{summary['conflict_count']}`",
        f"- image sizes: `{summary['image_size_distribution']}`",
    ]
    write_markdown_report(path, "PR-11 PTX-498 Manifest Audit", [("Summary", body)])


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
    export_samples(audit_rows, args.out, int(args.sample_count), int(args.seed))
    print(f"[{summary['status']}] PTX manifest audit written under {args.out}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

