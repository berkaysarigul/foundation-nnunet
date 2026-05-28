"""Inventory the extracted PTX-498 pneumothorax dataset for PR-11B."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pr11_utils import repo_relative, size_distribution, write_json, write_markdown_report, write_yaml
from ptx498_utils import (
    EXPECTED_SITE_NAMES,
    collect_site_files,
    detect_v2_fix_evidence,
    discover_site_dirs,
    first_path,
    natural_sort_key,
    ptx_case_id,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory extracted PTX-498 dataset.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--size-sample-limit", type=int, default=0)
    return parser.parse_args(argv)


def _paths_for_role(grouped: dict[str, dict[str, list[Path]]], role: str) -> list[Path]:
    paths: list[Path] = []
    for stem in sorted(grouped.get(role, {}), key=natural_sort_key):
        paths.extend(grouped[role][stem])
    return paths


def build_inventory(root: Path, sample_count: int = 10, size_sample_limit: int = 0) -> dict[str, Any]:
    site_dirs = discover_site_dirs(root)
    per_site: dict[str, dict[str, Any]] = {}
    all_images: list[Path] = []
    all_masks: list[Path] = []
    all_merges: list[Path] = []
    all_nii_images: list[Path] = []
    all_nii_masks: list[Path] = []
    first_cases: list[dict[str, Any]] = []
    unmatched_images: list[dict[str, str]] = []
    unmatched_masks: list[dict[str, str]] = []
    paired_png_image_mask_count = 0

    for site_dir in site_dirs:
        site = site_dir.name if site_dir != root else "root"
        grouped = collect_site_files(site_dir)
        image_stems = set(grouped["image_png"])
        mask_stems = set(grouped["mask_png"])
        paired_stems = image_stems & mask_stems
        paired_png_image_mask_count += len(paired_stems)

        site_unmatched_images = sorted(image_stems - mask_stems, key=natural_sort_key)
        site_unmatched_masks = sorted(mask_stems - image_stems, key=natural_sort_key)
        for stem in site_unmatched_images:
            image_path = first_path(grouped, "image_png", stem)
            unmatched_images.append(
                {
                    "site": site,
                    "case_stem": stem,
                    "image_path": repo_relative(image_path) if image_path else "",
                }
            )
        for stem in site_unmatched_masks:
            mask_path = first_path(grouped, "mask_png", stem)
            unmatched_masks.append(
                {
                    "site": site,
                    "case_stem": stem,
                    "mask_path": repo_relative(mask_path) if mask_path else "",
                }
            )

        image_paths = _paths_for_role(grouped, "image_png")
        mask_paths = _paths_for_role(grouped, "mask_png")
        merge_paths = _paths_for_role(grouped, "merge_png")
        nii_image_paths = _paths_for_role(grouped, "image_nii")
        nii_mask_paths = _paths_for_role(grouped, "mask_nii")
        all_images.extend(image_paths)
        all_masks.extend(mask_paths)
        all_merges.extend(merge_paths)
        all_nii_images.extend(nii_image_paths)
        all_nii_masks.extend(nii_mask_paths)

        per_site[site] = {
            "site_dir": repo_relative(site_dir),
            "png_images": len(image_paths),
            "png_masks": len(mask_paths),
            "merge_pngs": len(merge_paths),
            "nii_images": len(nii_image_paths),
            "nii_masks": len(nii_mask_paths),
            "paired_png_images_masks": len(paired_stems),
            "unmatched_images": len(site_unmatched_images),
            "unmatched_masks": len(site_unmatched_masks),
            "merge_files_exist": bool(merge_paths),
            "nii_files_exist": bool(nii_image_paths or nii_mask_paths),
        }

        for stem in sorted(paired_stems, key=natural_sort_key):
            if len(first_cases) >= sample_count:
                break
            image_path = first_path(grouped, "image_png", stem)
            mask_path = first_path(grouped, "mask_png", stem)
            first_cases.append(
                {
                    "case_id": ptx_case_id(site, stem),
                    "site": site,
                    "case_stem": stem,
                    "image_path": repo_relative(image_path) if image_path else "",
                    "mask_path": repo_relative(mask_path) if mask_path else "",
                    "merge_path": repo_relative(first_path(grouped, "merge_png", stem))
                    if first_path(grouped, "merge_png", stem)
                    else "",
                    "nii_image_path": repo_relative(first_path(grouped, "image_nii", stem))
                    if first_path(grouped, "image_nii", stem)
                    else "",
                    "nii_mask_path": repo_relative(first_path(grouped, "mask_nii", stem))
                    if first_path(grouped, "mask_nii", stem)
                    else "",
                }
            )

    all_paths = all_images + all_masks + all_merges + all_nii_images + all_nii_masks
    discovered = {
        "root": repo_relative(root),
        "site_dirs": [repo_relative(path) for path in site_dirs],
        "png_images": [repo_relative(path) for path in all_images],
        "png_masks": [repo_relative(path) for path in all_masks],
        "merge_pngs": [repo_relative(path) for path in all_merges],
        "nii_images": [repo_relative(path) for path in all_nii_images],
        "nii_masks": [repo_relative(path) for path in all_nii_masks],
        "first_cases": first_cases,
    }

    site_names = {path.name for path in site_dirs}
    return {
        "schema_version": 1,
        "root": repo_relative(root),
        "site_dirs": [path.name for path in site_dirs],
        "expected_site_dirs_present": sorted(EXPECTED_SITE_NAMES & site_names),
        "counts": {
            "png_images": len(all_images),
            "png_masks": len(all_masks),
            "merge_pngs": len(all_merges),
            "nii_images": len(all_nii_images),
            "nii_masks": len(all_nii_masks),
            "paired_png_images_masks": paired_png_image_mask_count,
            "unmatched_images": len(unmatched_images),
            "unmatched_masks": len(unmatched_masks),
        },
        "counts_by_site": dict(sorted(per_site.items())),
        "image_size_distribution": size_distribution(all_images, max_items=size_sample_limit),
        "mask_size_distribution": size_distribution(all_masks, max_items=size_sample_limit),
        "first_10_cases": first_cases[:10],
        "every_image_has_matching_mask": not unmatched_images,
        "every_mask_has_matching_image": not unmatched_masks,
        "merge_files_exist": bool(all_merges),
        "nii_files_exist": bool(all_nii_images or all_nii_masks),
        "v2_fix_provenance": detect_v2_fix_evidence(root, all_paths),
        "unmatched_images": unmatched_images[:50],
        "unmatched_masks": unmatched_masks[:50],
        "discovered": discovered,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    body = [
        f"- root: `{summary['root']}`",
        f"- site dirs: `{', '.join(summary['site_dirs'])}`",
        f"- PNG images: `{counts['png_images']}`",
        f"- PNG masks: `{counts['png_masks']}`",
        f"- merge PNGs: `{counts['merge_pngs']}`",
        f"- NIfTI images: `{counts['nii_images']}`",
        f"- NIfTI masks: `{counts['nii_masks']}`",
        f"- every image has matching mask: `{summary['every_image_has_matching_mask']}`",
        f"- every mask has matching image: `{summary['every_mask_has_matching_image']}`",
        f"- merge files exist: `{summary['merge_files_exist']}`",
        f"- v2-fix provenance detectable: `{summary['v2_fix_provenance']['detectable']}`",
    ]
    per_site = [
        f"- `{site}`: images `{info['png_images']}`, masks `{info['png_masks']}`, "
        f"merges `{info['merge_pngs']}`, NIfTI `{info['nii_images']}/{info['nii_masks']}`"
        for site, info in summary["counts_by_site"].items()
    ] or ["- none"]
    samples = [
        f"- `{row['case_id']}` -> `{row['image_path']}` / `{row['mask_path']}`"
        for row in summary["first_10_cases"]
    ] or ["- none"]
    write_markdown_report(
        path,
        "PR-11B PTX-498 Inventory",
        [("Summary", body), ("Per-Site Counts", per_site), ("First 10 Cases", samples)],
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_inventory(args.root, int(args.sample_count), int(args.size_sample_limit))
    args.out.mkdir(parents=True, exist_ok=True)
    write_yaml(args.out / "inventory_summary.yaml", summary)
    write_json(args.out / "discovered_files.json", summary["discovered"])
    write_report(args.out / "inventory_report.md", summary)
    print(f"[done] PTX-498 inventory written under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
