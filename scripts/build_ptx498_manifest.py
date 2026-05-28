"""Build the PR-11B PTX-498 external-validation manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pr11_utils import (
    image_size,
    mask_stats,
    parse_bool,
    percentile_summary,
    repo_relative,
    sha256_file,
    write_csv,
    write_markdown_report,
    write_yaml,
)
from ptx498_utils import (
    collect_site_files,
    discover_site_dirs,
    duplicate_role_rows,
    first_path,
    natural_sort_key,
    ptx_case_id,
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


def _counts_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _conflict(
    conflict_type: str,
    *,
    site: str,
    case_stem: str = "",
    case_id: str = "",
    image_path: Path | None = None,
    mask_path: Path | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "site": site,
        "case_stem": case_stem,
        "conflict_type": conflict_type,
        "image_path": repo_relative(image_path) if image_path else "",
        "mask_path": repo_relative(mask_path) if mask_path else "",
        "detail": detail,
    }


def _duplicate_conflicts(
    grouped: dict[str, dict[str, list[Path]]],
    *,
    site: str,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for row in duplicate_role_rows(grouped, site=site, relpath=repo_relative):
        conflicts.append(
            _conflict(
                f"duplicate_{row['role']}",
                site=site,
                case_stem=row["case_stem"],
                case_id=ptx_case_id(site, row["case_stem"]),
                detail=row["paths"],
            )
        )
    return conflicts


def build_manifest(
    root: Path,
    *,
    hash_files: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unmatched_images: list[dict[str, Any]] = []
    unmatched_masks: list[dict[str, Any]] = []
    empty_masks: list[dict[str, Any]] = []
    foreground: list[int] = []
    seen_case_ids: dict[str, int] = {}
    strict_binary_mask_count = 0
    non_binary_binarizable_mask_count = 0

    for site_dir in discover_site_dirs(root):
        site = site_dir.name if site_dir != root else "root"
        grouped = collect_site_files(site_dir)
        conflicts.extend(_duplicate_conflicts(grouped, site=site))

        image_stems = set(grouped["image_png"])
        mask_stems = set(grouped["mask_png"])
        for stem in sorted(image_stems - mask_stems, key=natural_sort_key):
            image_path = first_path(grouped, "image_png", stem)
            case_id = ptx_case_id(site, stem)
            unmatched_images.append(
                {
                    "case_id": case_id,
                    "site": site,
                    "case_stem": stem,
                    "image_path": repo_relative(image_path) if image_path else "",
                }
            )
            conflicts.append(
                _conflict(
                    "unmatched_image_missing_mask",
                    site=site,
                    case_stem=stem,
                    case_id=case_id,
                    image_path=image_path,
                )
            )
        for stem in sorted(mask_stems - image_stems, key=natural_sort_key):
            mask_path = first_path(grouped, "mask_png", stem)
            case_id = ptx_case_id(site, stem)
            unmatched_masks.append(
                {
                    "case_id": case_id,
                    "site": site,
                    "case_stem": stem,
                    "mask_path": repo_relative(mask_path) if mask_path else "",
                }
            )
            conflicts.append(
                _conflict(
                    "unmatched_mask_missing_image",
                    site=site,
                    case_stem=stem,
                    case_id=case_id,
                    mask_path=mask_path,
                )
            )

        for stem in sorted(image_stems & mask_stems, key=natural_sort_key):
            image_path = first_path(grouped, "image_png", stem)
            mask_path = first_path(grouped, "mask_png", stem)
            if image_path is None or mask_path is None:
                continue
            merge_path = first_path(grouped, "merge_png", stem)
            nii_image_path = first_path(grouped, "image_nii", stem)
            nii_mask_path = first_path(grouped, "mask_nii", stem)
            case_id = ptx_case_id(site, stem)
            seen_case_ids[case_id] = seen_case_ids.get(case_id, 0) + 1

            row_conflicts: list[dict[str, Any]] = []
            width, height = image_size(image_path)
            stats = mask_stats(mask_path)
            mask_width = int(stats["width"])
            mask_height = int(stats["height"])
            fg = int(stats["foreground_pixels"])
            positive = bool(stats["is_positive"])
            if bool(stats["is_binary"]):
                strict_binary_mask_count += 1
            else:
                non_binary_binarizable_mask_count += 1
            foreground.append(fg)
            if (width, height) != (mask_width, mask_height):
                row_conflicts.append(
                    _conflict(
                        "image_mask_size_mismatch",
                        site=site,
                        case_stem=stem,
                        case_id=case_id,
                        image_path=image_path,
                        mask_path=mask_path,
                        detail=f"image={width}x{height};mask={mask_width}x{mask_height}",
                    )
                )

            row = {
                "case_id": case_id,
                "source_dataset": "PTX498",
                "site": site,
                "image_path": repo_relative(image_path),
                "mask_path": repo_relative(mask_path),
                "merge_path": repo_relative(merge_path) if merge_path else "",
                "nii_image_path": repo_relative(nii_image_path) if nii_image_path else "",
                "nii_mask_path": repo_relative(nii_mask_path) if nii_mask_path else "",
                "width": width,
                "height": height,
                "gt_foreground_pixels": fg,
                "mask_is_positive": positive,
                "image_hash": sha256_file(image_path) if hash_files else "",
                "mask_hash": sha256_file(mask_path) if hash_files else "",
            }
            rows.append(row)
            if not positive:
                empty_masks.append(
                    {
                        "case_id": case_id,
                        "site": site,
                        "case_stem": stem,
                        "mask_path": repo_relative(mask_path),
                        "gt_foreground_pixels": fg,
                    }
                )
            conflicts.extend(row_conflicts)

    duplicate_case_ids = sorted(case_id for case_id, count in seen_case_ids.items() if count > 1)
    for case_id in duplicate_case_ids:
        conflicts.append(_conflict("duplicate_case_id", site="", case_id=case_id))

    summary = {
        "schema_version": 1,
        "root": repo_relative(root),
        "row_count": len(rows),
        "positive_count": sum(1 for r in rows if r["mask_is_positive"]),
        "empty_mask_count": len(empty_masks),
        "site_counts": _counts_by(rows, "site"),
        "unmatched_images": len(unmatched_images),
        "unmatched_masks": len(unmatched_masks),
        "duplicate_case_ids": duplicate_case_ids,
        "conflicts": len(conflicts),
        "foreground_pixels": percentile_summary(foreground),
        "strict_binary_mask_count": strict_binary_mask_count,
        "non_binary_binarizable_mask_count": non_binary_binarizable_mask_count,
        "expected_498_cases": len(rows) == 498,
        "hash_files": bool(hash_files),
    }
    return rows, conflicts, unmatched_images, unmatched_masks, empty_masks, summary


def write_report(path: Path, summary: dict[str, Any], conflicts: list[dict[str, Any]]) -> None:
    body = [
        f"- rows: `{summary['row_count']}`",
        f"- positives: `{summary['positive_count']}`",
        f"- empty masks: `{summary['empty_mask_count']}`",
        f"- site counts: `{summary['site_counts']}`",
        f"- unmatched images: `{summary['unmatched_images']}`",
        f"- unmatched masks: `{summary['unmatched_masks']}`",
        f"- conflicts: `{summary['conflicts']}`",
        f"- strict binary masks: `{summary['strict_binary_mask_count']}`",
        f"- non-binary but binarizable masks: `{summary['non_binary_binarizable_mask_count']}`",
        f"- expected 498 cases: `{summary['expected_498_cases']}`",
    ]
    conflict_body = [
        f"- `{row.get('case_id', '')}`: `{row.get('conflict_type', '')}` {row.get('detail', '')}".rstrip()
        for row in conflicts[:25]
    ] or ["- none"]
    write_markdown_report(
        path,
        "PR-11B PTX-498 Manifest Report",
        [("Summary", body), ("Conflict Sample", conflict_body)],
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, conflicts, unmatched_images, unmatched_masks, empty_masks, summary = build_manifest(
        args.root,
        hash_files=bool(args.hash_files),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "ptx498_manifest.csv", rows, MANIFEST_COLUMNS)
    write_yaml(args.out / "ptx498_manifest_summary.yaml", summary)
    write_report(args.out / "ptx498_manifest_report.md", summary, conflicts)
    write_csv(args.out / "ptx498_unmatched_images.csv", unmatched_images)
    write_csv(args.out / "ptx498_unmatched_masks.csv", unmatched_masks)
    write_csv(args.out / "ptx498_empty_masks.csv", empty_masks)
    write_csv(args.out / "ptx498_conflicts.csv", conflicts)
    if conflicts and not bool(args.allow_conflicts):
        print(
            f"[failed] wrote PTX manifest artifacts, but found {len(conflicts)} conflict(s). "
            "Pass --allow-conflicts true only for diagnostic output.",
            file=sys.stderr,
        )
        return 1
    print(f"[done] PTX manifest written to {args.out / 'ptx498_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
