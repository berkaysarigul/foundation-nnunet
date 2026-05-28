"""Audit the PR-11B PTX-498 manifest and export visual sanity overlays."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from pr11_utils import (
    REPO_ROOT,
    ensure_dir,
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
from ptx498_utils import safe_token


REQUIRED_COLUMNS = {
    "case_id",
    "source_dataset",
    "site",
    "image_path",
    "mask_path",
    "merge_path",
    "nii_image_path",
    "nii_mask_path",
    "width",
    "height",
    "gt_foreground_pixels",
    "mask_is_positive",
    "image_hash",
    "mask_hash",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PTX-498 manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-row-count", type=int, default=498)
    parser.add_argument("--row-count-tolerance", type=int, default=0)
    return parser.parse_args(argv)


def _bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes"}


def _int(raw: Any) -> int:
    try:
        return int(float(str(raw).strip() or "0"))
    except ValueError:
        return 0


def _manifest_missing_summary(manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    conflict = {"case_id": "", "issues": "missing_manifest", "path": str(manifest)}
    summary = {
        "schema_version": 1,
        "manifest": str(manifest),
        "manifest_exists": False,
        "row_count": 0,
        "unique_case_ids": 0,
        "duplicate_case_ids": [],
        "positive_count": 0,
        "empty_mask_count": 0,
        "site_counts": {},
        "image_size_distribution": {},
        "mask_size_distribution": {},
        "foreground_pixels": percentile_summary([]),
        "strict_binary_mask_count": 0,
        "non_binary_binarizable_mask_count": 0,
        "conflict_count": 1,
        "status": "FAIL",
    }
    return [], summary, [conflict]


def audit_manifest(manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not manifest.is_file():
        return _manifest_missing_summary(manifest)

    rows = read_csv(manifest)
    audit_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    foreground: list[int] = []
    site_counts: dict[str, int] = {}
    image_size_counts: dict[str, int] = {}
    mask_size_counts: dict[str, int] = {}
    seen: set[str] = set()
    duplicates: set[str] = set()
    strict_binary_masks = 0
    non_binary_binarizable_masks = 0

    if rows:
        missing_columns = sorted(REQUIRED_COLUMNS - set(rows[0]))
        if missing_columns:
            conflicts.append({"case_id": "", "issues": "missing_required_columns", "columns": ";".join(missing_columns)})

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
        manifest_width = _int(row.get("width", "0"))
        manifest_height = _int(row.get("height", "0"))
        fg = _int(row.get("gt_foreground_pixels", "0"))
        positive = _bool(row.get("mask_is_positive", "false"))

        if image_exists:
            width, height = image_size(image_path)
            image_size_counts[f"{width}x{height}"] = image_size_counts.get(f"{width}x{height}", 0) + 1
            if manifest_width and manifest_height and (width, height) != (manifest_width, manifest_height):
                issues.append("manifest_image_size_mismatch")
        else:
            issues.append("missing_image_path")

        if mask_exists:
            stats = mask_stats(mask_path)
            mask_width = int(stats["width"])
            mask_height = int(stats["height"])
            mask_size_counts[f"{mask_width}x{mask_height}"] = mask_size_counts.get(f"{mask_width}x{mask_height}", 0) + 1
            if bool(stats["is_binary"]):
                strict_binary_masks += 1
            else:
                non_binary_binarizable_masks += 1
            if int(stats["foreground_pixels"]) != fg:
                issues.append("foreground_pixel_mismatch")
            if bool(stats["is_positive"]) != positive:
                issues.append("mask_positive_mismatch")
        else:
            issues.append("missing_mask_path")

        if image_exists and mask_exists and (width, height) != (mask_width, mask_height):
            issues.append("image_mask_size_mismatch")
        if positive and fg <= 0:
            issues.append("positive_without_foreground")
        if (not positive) and fg > 0:
            issues.append("empty_mask_flag_with_foreground")
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
        "manifest_exists": True,
        "row_count": len(rows),
        "unique_case_ids": len(seen),
        "duplicate_case_ids": sorted(duplicates),
        "positive_count": sum(1 for r in audit_rows if r["mask_is_positive"]),
        "empty_mask_count": sum(1 for r in audit_rows if not r["mask_is_positive"]),
        "site_counts": dict(sorted(site_counts.items())),
        "image_size_distribution": dict(sorted(image_size_counts.items())),
        "mask_size_distribution": dict(sorted(mask_size_counts.items())),
        "foreground_pixels": percentile_summary(foreground),
        "strict_binary_mask_count": strict_binary_masks,
        "non_binary_binarizable_mask_count": non_binary_binarizable_masks,
        "conflict_count": len(conflicts),
        "status": "PASS" if not conflicts else "FAIL",
    }
    return audit_rows, summary, conflicts


def _sample(rows: list[dict[str, Any]], *, positive: bool, count: int, seed: int) -> list[dict[str, Any]]:
    subset = [row for row in rows if bool(row["mask_is_positive"]) is positive and not row["issues"]]
    rng = random.Random(seed + (1 if positive else 2))
    rng.shuffle(subset)
    return subset[:count]


def _site_samples(rows: list[dict[str, Any]], sample_count: int, seed: int) -> list[dict[str, Any]]:
    clean = [row for row in rows if not row["issues"]]
    by_site: dict[str, list[dict[str, Any]]] = {}
    for row in clean:
        by_site.setdefault(str(row["site"]), []).append(row)
    rng = random.Random(seed + 3)
    selected: list[dict[str, Any]] = []
    for site in sorted(by_site):
        site_rows = by_site[site]
        rng.shuffle(site_rows)
        if site_rows:
            selected.append(site_rows[0])
    remaining = [row for row in clean if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_count - len(selected))])
    return selected[:sample_count]


def export_samples(audit_rows: list[dict[str, Any]], out: Path, sample_count: int, seed: int) -> None:
    ensure_dir(out / "positive_samples")
    ensure_dir(out / "empty_mask_samples")
    ensure_dir(out / "site_samples")
    per_class = max(1, sample_count // 2)
    for folder, selected in [
        ("positive_samples", _sample(audit_rows, positive=True, count=per_class, seed=seed)),
        ("empty_mask_samples", _sample(audit_rows, positive=False, count=per_class, seed=seed)),
    ]:
        for row in selected:
            make_overlay(
                resolve_existing_path(row["image_path"], REPO_ROOT),
                resolve_existing_path(row["mask_path"], REPO_ROOT),
                out / folder / f"{row['case_id']}.png",
            )
    for row in _site_samples(audit_rows, sample_count, seed):
        site = safe_token(str(row["site"]))
        make_overlay(
            resolve_existing_path(row["image_path"], REPO_ROOT),
            resolve_existing_path(row["mask_path"], REPO_ROOT),
            out / "site_samples" / site / f"{row['case_id']}.png",
        )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    body = [
        f"- status: `{summary['status']}`",
        f"- manifest exists: `{summary['manifest_exists']}`",
        f"- rows: `{summary['row_count']}`",
        f"- unique case IDs: `{summary['unique_case_ids']}`",
        f"- positives: `{summary['positive_count']}`",
        f"- empty masks: `{summary['empty_mask_count']}`",
        f"- site counts: `{summary['site_counts']}`",
        f"- conflicts: `{summary['conflict_count']}`",
        f"- image sizes: `{summary['image_size_distribution']}`",
        f"- mask sizes: `{summary['mask_size_distribution']}`",
        f"- strict binary masks: `{summary['strict_binary_mask_count']}`",
        f"- non-binary but binarizable masks: `{summary['non_binary_binarizable_mask_count']}`",
    ]
    write_markdown_report(path, "PR-11B PTX-498 Manifest Audit", [("Summary", body)])


def _apply_expected_row_count(summary: dict[str, Any], conflicts: list[dict[str, Any]], expected: int, tolerance: int) -> None:
    if expected <= 0:
        return
    row_count = int(summary["row_count"])
    matches = abs(row_count - expected) <= max(0, tolerance)
    summary["expected_row_count"] = expected
    summary["row_count_tolerance"] = max(0, tolerance)
    summary["row_count_matches_expected"] = matches
    if not matches:
        conflicts.append(
            {
                "case_id": "",
                "issues": "row_count_outside_expected_range",
                "expected": expected,
                "tolerance": max(0, tolerance),
                "actual": row_count,
            }
        )
        summary["conflict_count"] = len(conflicts)
        summary["status"] = "FAIL"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit_rows, summary, conflicts = audit_manifest(args.manifest)
    _apply_expected_row_count(summary, conflicts, int(args.expected_row_count), int(args.row_count_tolerance))
    args.out.mkdir(parents=True, exist_ok=True)
    write_yaml(args.out / "summary.yaml", summary)
    write_csv(args.out / "manifest_audit.csv", audit_rows)
    write_csv(args.out / "manifest_audit_conflicts.csv", conflicts)
    write_report(args.out / "report.md", summary)
    export_samples(audit_rows, args.out, int(args.sample_count), int(args.seed))
    print(f"[{summary['status']}] PTX manifest audit written under {args.out}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
