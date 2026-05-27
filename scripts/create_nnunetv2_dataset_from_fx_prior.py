"""Create PR-11 nnU-Net v2 PNG datasets from SIIM/PTX manifests and FX priors."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pr11_utils import (
    REPO_ROOT,
    binarize_mask,
    parse_bool,
    read_csv,
    repo_relative,
    resolve_existing_path,
    save_binary_mask,
    save_uint8,
    write_csv,
    write_markdown_report,
    write_yaml,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create nnU-Net v2 PNG dataset from PR-11 manifests.")
    parser.add_argument("--siim-manifest", type=Path, default=None)
    parser.add_argument("--ptx-manifest", type=Path, default=None)
    parser.add_argument("--fx-prior-manifest", type=Path, default=None)
    parser.add_argument("--protocol", choices=["siim_stage", "dataset101_bridge", "ptx_external"], required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--channels", required=True, help="Comma list: image,fx_prior, or fx_prior only.")
    parser.add_argument("--split-source", default="stage1_train_test")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-cases-csv", type=Path, default=None)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    return parser.parse_args(argv)


def _dataset_dir(nnunet_raw: Path, dataset_id: int, dataset_name: str) -> Path:
    return nnunet_raw / f"Dataset{int(dataset_id):03d}_{dataset_name}"


def _validate_channels(channels: str) -> list[str]:
    parsed = [item.strip() for item in channels.split(",") if item.strip()]
    allowed = {"image", "fx_prior"}
    if not parsed or any(item not in allowed for item in parsed):
        raise ValueError(f"--channels must contain only {sorted(allowed)}, got {channels!r}")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"--channels contains duplicates: {channels!r}")
    return parsed


def _prepare_output(root: Path, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset already exists: {root}. Pass --overwrite true to replace.")
        resolved = root.resolve()
        try:
            resolved.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"Refusing to remove output outside repo: {root}") from exc
        shutil.rmtree(root)
    for name in ("imagesTr", "labelsTr", "imagesTs", "heldout_labelsTs"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _prior_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = read_csv(path)
    return {row["case_id"]: row for row in rows if row.get("case_id")}


def _excluded_cases(path: Path | None) -> set[str]:
    if path is None:
        return set()
    rows = read_csv(path)
    return {row.get("case_id", "") or row.get("siim_case_id", "") for row in rows}


def _split_train_val(rows: list[dict[str, str]], val_fraction: float, seed: int) -> dict[str, list[str]]:
    import random

    positives = [r["case_id"] for r in rows if str(r.get("mask_is_positive", "")).lower() == "true"]
    negatives = [r["case_id"] for r in rows if str(r.get("mask_is_positive", "")).lower() != "true"]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    val_pos = max(1, round(len(positives) * val_fraction)) if positives else 0
    val_neg = max(1, round(len(negatives) * val_fraction)) if negatives else 0
    val = sorted(positives[:val_pos] + negatives[:val_neg])
    train = sorted(positives[val_pos:] + negatives[val_neg:])
    return {"train": train, "val": val}


def _load_image_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _save_image_channel(src_path: Path, dst_path: Path) -> None:
    save_uint8(_load_image_uint8(src_path), dst_path)


def _save_prior_channel(prior_row: dict[str, str], dst_path: Path, target_size: tuple[int, int]) -> None:
    raw = prior_row.get("aligned_probability_map_path") or prior_row.get("probability_map_path")
    if not raw:
        raise ValueError("FX prior row lacks aligned_probability_map_path/probability_map_path")
    prior_path = resolve_existing_path(raw, REPO_ROOT)
    with Image.open(prior_path) as image:
        image = image.convert("L")
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.uint8)
    save_uint8(arr, dst_path)


def _materialize_case(
    row: dict[str, str],
    *,
    prior_rows: dict[str, dict[str, str]],
    channels: list[str],
    dataset_root: Path,
    image_subdir: str,
    label_subdir: str | None,
) -> dict[str, Any]:
    case_id = row["case_id"]
    image_path = resolve_existing_path(row["image_path"], REPO_ROOT)
    mask_path = resolve_existing_path(row.get("mask_path", ""), REPO_ROOT)
    with Image.open(image_path) as image:
        target_size = image.size
    output_images: list[str] = []
    for channel_idx, channel in enumerate(channels):
        dst = dataset_root / image_subdir / f"{case_id}_{channel_idx:04d}.png"
        if channel == "image":
            _save_image_channel(image_path, dst)
        elif channel == "fx_prior":
            if case_id not in prior_rows:
                raise KeyError(f"No Foundation X prior for case_id={case_id}")
            _save_prior_channel(prior_rows[case_id], dst, target_size)
        output_images.append(repo_relative(dst))
    label_out = ""
    if label_subdir is not None:
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found for case_id={case_id}: {mask_path}")
        label_dst = dataset_root / label_subdir / f"{case_id}.png"
        save_binary_mask(binarize_mask(mask_path), label_dst)
        label_out = repo_relative(label_dst)
    return {
        "case_id": case_id,
        "source_dataset": row.get("source_dataset", ""),
        "source_split": row.get("source_split", row.get("site", "")),
        "nnunet_split": image_subdir,
        "image_source_path": row.get("image_path", ""),
        "mask_source_path": row.get("mask_path", ""),
        "image_output_paths": output_images,
        "label_output_path": label_out,
    }


def _write_dataset_json(root: Path, dataset_id: int, dataset_name: str, channels: list[str], num_training: int, description: str) -> None:
    payload = {
        "channel_names": {str(idx): ("xray" if channel == "image" else "foundation_x_prob") for idx, channel in enumerate(channels)},
        "labels": {"background": 0, "pneumothorax": 1},
        "numTraining": int(num_training),
        "file_ending": ".png",
        "name": f"Dataset{int(dataset_id):03d}_{dataset_name}",
        "description": description,
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (root / "dataset.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_dataset(args: argparse.Namespace) -> dict[str, Any]:
    channels = _validate_channels(args.channels)
    dataset_root = _dataset_dir(args.nnunet_raw, args.dataset_id, args.dataset_name)
    _prepare_output(dataset_root, bool(args.overwrite))
    prior_rows = _prior_map(args.fx_prior_manifest)
    if "fx_prior" in channels and not prior_rows:
        raise ValueError("--fx-prior-manifest is required when channels include fx_prior")

    excluded = _excluded_cases(args.exclude_cases_csv)
    mapping_rows: list[dict[str, Any]] = []
    splits_payload: list[dict[str, list[str]]] = []

    if args.protocol in {"siim_stage", "dataset101_bridge"}:
        if args.siim_manifest is None:
            raise ValueError("--siim-manifest is required for SIIM protocols")
        rows = [row for row in read_csv(args.siim_manifest) if row.get("case_id") not in excluded]
        train_rows = [row for row in rows if row.get("source_split") == "stage1_train"]
        test_rows = [row for row in rows if row.get("source_split") == "stage1_test"]
        split = _split_train_val(train_rows, float(args.val_fraction), int(args.seed))
        splits_payload = [split]
        for row in train_rows:
            mapping_rows.append(
                _materialize_case(
                    row,
                    prior_rows=prior_rows,
                    channels=channels,
                    dataset_root=dataset_root,
                    image_subdir="imagesTr",
                    label_subdir="labelsTr",
                )
            )
        for row in test_rows:
            mapping_rows.append(
                _materialize_case(
                    row,
                    prior_rows=prior_rows,
                    channels=channels,
                    dataset_root=dataset_root,
                    image_subdir="imagesTs",
                    label_subdir="heldout_labelsTs",
                )
            )
        description = (
            "PR-11 SIIM-ACR original PNG protocol. Train/val from stage_1_train; "
            "stage_1_test masks are stored in heldout_labelsTs for external evaluation."
        )
        num_training = len(train_rows)
    else:
        if args.ptx_manifest is None:
            raise ValueError("--ptx-manifest is required for ptx_external")
        rows = read_csv(args.ptx_manifest)
        for row in rows:
            mapping_rows.append(
                _materialize_case(
                    row,
                    prior_rows=prior_rows,
                    channels=channels,
                    dataset_root=dataset_root,
                    image_subdir="imagesTs",
                    label_subdir="heldout_labelsTs",
                )
            )
        splits_payload = [{"train": [], "val": []}]
        description = "PR-11 PTX-498 external validation dataset. Not for training/model selection."
        num_training = 0

    _write_dataset_json(dataset_root, int(args.dataset_id), args.dataset_name, channels, num_training, description)
    write_csv(dataset_root / "case_mapping.csv", mapping_rows)
    (dataset_root / "splits_final_source.json").write_text(json.dumps(splits_payload, indent=2), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "dataset_root": repo_relative(dataset_root),
        "protocol": args.protocol,
        "dataset_id": int(args.dataset_id),
        "dataset_name": args.dataset_name,
        "channels": channels,
        "num_training": num_training,
        "imagesTr": len(list((dataset_root / "imagesTr").glob("*.png"))),
        "labelsTr": len(list((dataset_root / "labelsTr").glob("*.png"))),
        "imagesTs": len(list((dataset_root / "imagesTs").glob("*.png"))),
        "heldout_labelsTs": len(list((dataset_root / "heldout_labelsTs").glob("*.png"))),
        "case_mapping_rows": len(mapping_rows),
        "excluded_cases": len(excluded),
    }
    write_yaml(dataset_root / "conversion_summary.yaml", summary)
    write_report(dataset_root / "conversion_report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    body = [
        f"- dataset root: `{summary['dataset_root']}`",
        f"- protocol: `{summary['protocol']}`",
        f"- channels: `{summary['channels']}`",
        f"- num training: `{summary['num_training']}`",
        f"- imagesTr files: `{summary['imagesTr']}`",
        f"- labelsTr files: `{summary['labelsTr']}`",
        f"- imagesTs files: `{summary['imagesTs']}`",
        f"- heldout_labelsTs files: `{summary['heldout_labelsTs']}`",
    ]
    write_markdown_report(path, "PR-11 nnU-Net v2 Dataset Conversion Report", [("Summary", body)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = create_dataset(args)
    print(f"[done] nnU-Net dataset written to {summary['dataset_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

