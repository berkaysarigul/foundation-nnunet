"""Inventory the extracted PTX-498 pneumothorax dataset for PR-11."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pr11_utils import repo_relative, size_distribution, write_json, write_markdown_report, write_yaml


SUFFIX_RULES = {
    "img_png": ".1.img.png",
    "mask_png": ".2.mask.png",
    "merge_png": ".3.merge.png",
    "img_nii": ".4.img.nii.gz",
    "mask_nii": ".5.mask.nii.gz",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory extracted PTX-498 dataset.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size-sample-limit", type=int, default=0)
    return parser.parse_args(argv)


def _suffix_kind(path: Path) -> str:
    name = path.name
    for kind, suffix in SUFFIX_RULES.items():
        if name.endswith(suffix):
            return kind
    return "other"


def build_inventory(root: Path, size_sample_limit: int) -> dict[str, Any]:
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    site_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    by_kind: dict[str, list[Path]] = {}
    by_site: dict[str, dict[str, int]] = {}
    for path in files:
        site = path.relative_to(root).parts[0] if path.relative_to(root).parts else ""
        kind = _suffix_kind(path)
        by_kind.setdefault(kind, []).append(path)
        by_site.setdefault(site, {})
        by_site[site][kind] = by_site[site].get(kind, 0) + 1

    metadata_files = [
        p for p in files if p.suffix.lower() in {".csv", ".json", ".xlsx", ".xls", ".txt", ".md"}
    ]
    discovered = {
        "root": repo_relative(root),
        "site_dirs": [repo_relative(p) for p in site_dirs],
        "files": [repo_relative(p) for p in files],
        "metadata_files": [repo_relative(p) for p in metadata_files],
    }
    return {
        "schema_version": 1,
        "root": repo_relative(root),
        "site_dirs": [p.name for p in site_dirs],
        "counts_by_kind": {kind: len(paths) for kind, paths in sorted(by_kind.items())},
        "counts_by_site": {site: dict(sorted(counts.items())) for site, counts in sorted(by_site.items())},
        "image_size_distribution": size_distribution(by_kind.get("img_png", []), max_items=size_sample_limit),
        "mask_size_distribution": size_distribution(by_kind.get("mask_png", []), max_items=size_sample_limit),
        "metadata_files": [repo_relative(p) for p in metadata_files],
        "v2_fix_evidence": {
            "expected_sites_present": set(["SiteA", "SiteB", "SiteC"]).issubset({p.name for p in site_dirs}),
            "expected_png_image_count": len(by_kind.get("img_png", [])) == 498,
            "expected_png_mask_count": len(by_kind.get("mask_png", [])) == 498,
            "zenodo_record": "https://zenodo.org/records/8266529",
        },
        "discovered": discovered,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    body = [
        f"- root: `{summary['root']}`",
        f"- site dirs: `{', '.join(summary['site_dirs'])}`",
        f"- counts by kind: `{summary['counts_by_kind']}`",
        f"- counts by site: `{summary['counts_by_site']}`",
        f"- metadata files: `{len(summary['metadata_files'])}`",
        "- Zenodo record: `https://zenodo.org/records/8266529`",
    ]
    write_markdown_report(path, "PR-11 PTX-498 Inventory", [("Summary", body)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_inventory(args.root, int(args.size_sample_limit))
    args.out.mkdir(parents=True, exist_ok=True)
    write_yaml(args.out / "inventory_summary.yaml", summary)
    write_json(args.out / "discovered_files.json", summary["discovered"])
    write_report(args.out / "inventory_report.md", summary)
    print(f"[done] PTX-498 inventory written under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

