"""Build the PR-11 PTX-498 external-validation manifest."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from pr11_utils import (
    image_size,
    mask_stats,
    parse_bool,
    percentile_summary,
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
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PTX-498 manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hash-files", type=parse_bool, default=False)
    parser.add_argument("--allow-conflicts", type=parse_bool, default=False)
    return parser.parse_args(argv)


def _case_key_from_name(name: str, suffix: str) -> str | None:
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def _case_id(site: str, case_key: str) -> str:
    if re.fullmatch(r"\d+", case_key):
        return f"ptx498_{site.lower()}_{int(case_key):06d}"
    return sanitize_identifier(f"{site}_{case_key}", prefix="ptx498")


def build_manifest(root: Path, *, hash_files: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    foreground: list[int] = []
    site_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    for site_dir in site_dirs:
        site = site_dir.name
        img_by_key = {}
        for path in sorted(site_dir.glob("*.1.img.png")):
            key = _case_key_from_name(path.name, ".1.img.png")
            if key is not None:
                img_by_key[key] = path
        for key, image_path in sorted(img_by_key.items(), key=lambda item: item[0]):
            mask_path = site_dir / f"{key}.2.mask.png"
            merge_path = site_dir / f"{key}.3.merge.png"
            nii_image_path = site_dir / f"{key}.4.img.nii.gz"
            nii_mask_path = site_dir / f"{key}.5.mask.nii.gz"
            case_id = _case_id(site, key)
            row_conflicts: list[str] = []
            if not mask_path.is_file():
                row_conflicts.append("missing_mask_png")
            width = height = 0
            fg = 0
            positive = False
            if image_path.is_file():
                width, height = image_size(image_path)
            if mask_path.is_file():
                stats = mask_stats(mask_path)
                fg = int(stats["foreground_pixels"])
                positive = bool(stats["is_positive"])
                foreground.append(fg)
                if not stats["is_binary"]:
                    row_conflicts.append("non_binary_mask")
                if (width, height) != (int(stats["width"]), int(stats["height"])):
                    row_conflicts.append("image_mask_size_mismatch")
            for optional_path, conflict_name in [
                (merge_path, "missing_merge_png"),
                (nii_image_path, "missing_image_nii"),
                (nii_mask_path, "missing_mask_nii"),
            ]:
                if not optional_path.is_file():
                    row_conflicts.append(conflict_name)
            row = {
                "case_id": case_id,
                "source_dataset": "PTX498",
                "site": site,
                "image_path": repo_relative(image_path),
                "mask_path": repo_relative(mask_path) if mask_path.exists() else "",
                "merge_path": repo_relative(merge_path) if merge_path.exists() else "",
                "nii_image_path": repo_relative(nii_image_path) if nii_image_path.exists() else "",
                "nii_mask_path": repo_relative(nii_mask_path) if nii_mask_path.exists() else "",
                "width": width,
                "height": height,
                "gt_foreground_pixels": fg,
                "mask_is_positive": positive,
                "image_hash": sha256_file(image_path) if hash_files else "",
                "mask_hash": sha256_file(mask_path) if hash_files and mask_path.exists() else "",
            }
            rows.append(row)
            for conflict in row_conflicts:
                conflicts.append({"case_id": case_id, "site": site, "case_key": key, "conflict_type": conflict})

    summary = {
        "schema_version": 1,
        "root": repo_relative(root),
        "row_count": len(rows),
        "positive_count": sum(1 for r in rows if r["mask_is_positive"]),
        "negative_or_empty_mask_count": sum(1 for r in rows if not r["mask_is_positive"]),
        "site_counts": _counts_by(rows, "site"),
        "conflicts": len(conflicts),
        "foreground_pixels": percentile_summary(foreground),
        "expected_498_cases": len(rows) == 498,
        "hash_files": bool(hash_files),
    }
    if len(rows) != 498:
        conflicts.append({"case_id": "", "site": "", "case_key": "", "conflict_type": "expected_498_cases_not_met"})
        summary["conflicts"] = len(conflicts)
    return rows, conflicts, summary


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
        f"- empty/negative masks: `{summary['negative_or_empty_mask_count']}`",
        f"- site counts: `{summary['site_counts']}`",
        f"- conflicts: `{summary['conflicts']}`",
        f"- expected 498 cases: `{summary['expected_498_cases']}`",
    ]
    conflict_body = [
        f"- `{row.get('case_id', '')}`: `{row.get('conflict_type', '')}`"
        for row in conflicts[:25]
    ] or ["- none"]
    write_markdown_report(path, "PR-11 PTX-498 Manifest Report", [("Summary", body), ("Conflict Sample", conflict_body)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, conflicts, summary = build_manifest(args.root, hash_files=bool(args.hash_files))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "ptx498_manifest.csv", rows, MANIFEST_COLUMNS)
    write_csv(args.out / "ptx498_conflicts.csv", conflicts)
    write_yaml(args.out / "ptx498_manifest_summary.yaml", summary)
    write_report(args.out / "ptx498_manifest_report.md", summary, conflicts)
    if conflicts and not bool(args.allow_conflicts):
        print(f"[failed] wrote PTX manifest artifacts, but found {len(conflicts)} conflict(s).", file=sys.stderr)
        return 1
    print(f"[done] PTX manifest written to {args.out / 'ptx498_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

