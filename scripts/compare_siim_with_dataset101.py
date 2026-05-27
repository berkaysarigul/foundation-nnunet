"""Compare SIIM-ACR PNG manifest cases with Dataset101 to prevent leakage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pr11_utils import (
    REPO_ROOT,
    normalized_thumbnail_hash,
    read_csv,
    repo_relative,
    resolve_existing_path,
    write_csv,
    write_markdown_report,
    write_yaml,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SIIM original manifest with Dataset101.")
    parser.add_argument("--siim-manifest", type=Path, required=True)
    parser.add_argument("--dataset101-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hash-mode", choices=["none", "normalized_thumbnail"], default="normalized_thumbnail")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _dataset101_rows(root: Path) -> list[dict[str, str]]:
    mapping = root / "case_mapping.csv"
    if not mapping.is_file():
        raise FileNotFoundError(f"Dataset101 case_mapping.csv not found: {mapping}")
    rows = read_csv(mapping)
    for row in rows:
        row["dataset101_root"] = repo_relative(root)
        row["dataset101_image_path"] = row.get("image_output_path", "")
        if row.get("image_output_path"):
            row["dataset101_image_abs"] = str(root / row["image_output_path"])
        else:
            row["dataset101_image_abs"] = ""
    return rows


def _index_dataset101(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        original_id = row.get("original_image_id", "").strip()
        if original_id:
            index[original_id] = row
    return index


def _hash_index_dataset101(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        image_abs = Path(row.get("dataset101_image_abs", ""))
        if not image_abs.is_file():
            continue
        key = normalized_thumbnail_hash(image_abs)
        index.setdefault(key, []).append(row)
    return index


def compare(siim_manifest: Path, dataset101_root: Path, hash_mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    siim_rows = read_csv(siim_manifest)
    d101_rows = _dataset101_rows(dataset101_root)
    by_original_id = _index_dataset101(d101_rows)
    hash_index = _hash_index_dataset101(d101_rows) if hash_mode == "normalized_thumbnail" else {}

    overlap_rows: list[dict[str, Any]] = []
    excluded_heldout: list[dict[str, Any]] = []
    allowed_train: list[dict[str, Any]] = []

    for row in siim_rows:
        matches: list[tuple[str, dict[str, str]]] = []
        original_id = row.get("original_image_id", "").strip()
        if original_id and original_id in by_original_id:
            matches.append(("original_image_id", by_original_id[original_id]))
        elif hash_index:
            image_path = resolve_existing_path(row.get("image_path", ""), REPO_ROOT)
            if image_path.is_file():
                key = normalized_thumbnail_hash(image_path)
                matches.extend(("normalized_thumbnail_hash", match) for match in hash_index.get(key, []))

        heldout_overlap = False
        for match_type, d101 in matches:
            d101_split = d101.get("nnunet_split", "")
            is_heldout = d101_split == "imagesTs" or bool(d101.get("heldout_label_output_path", ""))
            heldout_overlap = heldout_overlap or is_heldout
            overlap_rows.append(
                {
                    "siim_case_id": row.get("case_id", ""),
                    "siim_source_split": row.get("source_split", ""),
                    "siim_original_image_id": original_id,
                    "dataset101_case_id": d101.get("case_id", ""),
                    "dataset101_source_split": d101.get("source_split", ""),
                    "dataset101_nnunet_split": d101_split,
                    "match_type": match_type,
                    "is_dataset101_heldout": is_heldout,
                    "siim_image_path": row.get("image_path", ""),
                    "dataset101_image_path": d101.get("image_output_path", ""),
                }
            )
        if heldout_overlap:
            excluded_heldout.append(row)
        elif row.get("source_split") == "stage1_train":
            allowed_train.append(row)

    summary = {
        "schema_version": 1,
        "siim_manifest": str(siim_manifest),
        "dataset101_root": str(dataset101_root),
        "hash_mode": hash_mode,
        "siim_rows": len(siim_rows),
        "dataset101_rows": len(d101_rows),
        "overlap_rows": len(overlap_rows),
        "unique_siim_overlap_cases": len({r["siim_case_id"] for r in overlap_rows}),
        "dataset101_heldout_overlap_cases": len({r["siim_case_id"] for r in overlap_rows if r["is_dataset101_heldout"]}),
        "siim_train_allowed_cases": len(allowed_train),
        "siim_excluded_heldout_overlap_cases": len(excluded_heldout),
        "protocol_a_rule": "exclude every SIIM case overlapping Dataset101 heldout from training",
        "protocol_b_rule": "document overlap; SIIM stage test remains primary evaluation",
    }
    return overlap_rows, allowed_train, excluded_heldout, summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    body = [
        f"- SIIM rows: `{summary['siim_rows']}`",
        f"- Dataset101 rows: `{summary['dataset101_rows']}`",
        f"- overlap rows: `{summary['overlap_rows']}`",
        f"- unique SIIM overlap cases: `{summary['unique_siim_overlap_cases']}`",
        f"- Dataset101 heldout overlap cases: `{summary['dataset101_heldout_overlap_cases']}`",
        f"- Protocol A allowed train cases: `{summary['siim_train_allowed_cases']}`",
        f"- Protocol A excluded heldout-overlap cases: `{summary['siim_excluded_heldout_overlap_cases']}`",
    ]
    write_markdown_report(path, "PR-11 Dataset101 Overlap Report", [("Summary", body)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overlap, allowed, excluded, summary = compare(args.siim_manifest, args.dataset101_root, args.hash_mode)
    args.out.mkdir(parents=True, exist_ok=True)
    write_yaml(args.out / "overlap_summary.yaml", summary)
    write_report(args.out / "overlap_report.md", summary)
    write_csv(args.out / "overlap_cases.csv", overlap)
    write_csv(args.out / "siim_train_allowed_cases.csv", allowed)
    write_csv(args.out / "siim_excluded_heldout_overlap_cases.csv", excluded)
    print(f"[done] Dataset101 overlap report written under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

