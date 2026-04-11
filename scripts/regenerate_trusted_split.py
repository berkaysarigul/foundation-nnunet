"""Regenerate the trusted processed-dataset split and refresh dataset_manifest.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_manifest import (
    DATASET_MANIFEST_FILENAME,
    build_dataset_manifest,
    repo_root_from_here,
)
from src.data.preprocess import _SPLIT_POLICY_NAME, create_splits


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(dataset_dir: str, seed: int) -> None:
    dataset_root = Path(dataset_dir)
    repo_root = repo_root_from_here()

    manifest_path = dataset_root / DATASET_MANIFEST_FILENAME
    splits_path = dataset_root / "splits.json"
    mask_variants_path = dataset_root / "mask_variants.json"
    images_dir = dataset_root / "images"
    original_masks_dir = dataset_root / "original_masks"

    manifest = _load_json(manifest_path)
    mask_variant_manifest = _load_json(mask_variants_path)

    image_ids = sorted(path.stem for path in images_dir.glob("*.png"))
    # Preserve the accepted image-level label definition based on original-mask foreground.
    from PIL import Image
    import numpy as np

    positive_id_set: set[str] = set()
    for image_id in image_ids:
        mask = np.array(Image.open(original_masks_dir / f"{image_id}.png").convert("L"))
        if mask.max() > 0:
            positive_id_set.add(image_id)

    splits = create_splits(image_ids, positive_id_set, seed=seed)
    with splits_path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)

    raw_dir = repo_root / manifest["raw_source"]["raw_dir"]
    annotation_csv_path = repo_root / manifest["raw_source"]["annotation_csv_path"]
    dicom_dir = repo_root / manifest["raw_source"]["dicom_dir"]
    code_paths = [
        repo_root / "src/data/preprocess.py",
        repo_root / "src/data/rle_contract.py",
        repo_root / "src/data/mask_variants.py",
        repo_root / "src/data/dicom_intensity.py",
        repo_root / "src/data/dataset_manifest.py",
        repo_root / "configs/config.yaml",
    ]
    refreshed_manifest = build_dataset_manifest(
        dataset_version=manifest["dataset_version"],
        dataset_root=dataset_root,
        raw_dir=raw_dir,
        annotation_csv_path=annotation_csv_path,
        dicom_dir=dicom_dir,
        image_size=int(manifest["generation"]["image_size"]),
        seed=seed,
        resolved_rle_mode=manifest["generation"]["resolved_rle_mode"],
        split_policy=_SPLIT_POLICY_NAME,
        skipped_images=int(manifest["generation"]["skipped_images"]),
        splits=splits,
        mask_variant_manifest=mask_variant_manifest,
        code_paths=code_paths,
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(refreshed_manifest, handle, indent=2)

    print(f"dataset_dir={dataset_root}")
    print(f"split_policy={refreshed_manifest['generation']['split_policy']}")
    print(f"split_fingerprint={refreshed_manifest['fingerprints']['splits']}")
    print(f"dataset_fingerprint={refreshed_manifest['dataset_fingerprint']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate trusted processed-dataset split")
    parser.add_argument(
        "--dataset_dir",
        default="data/processed/pneumothorax_trusted_v1",
        help="Trusted processed dataset root",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    args = parser.parse_args()
    main(args.dataset_dir, args.seed)
