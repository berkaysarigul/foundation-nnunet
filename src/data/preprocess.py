"""
preprocess.py - DICOM to PNG conversion and RLE mask decoding for SIIM-ACR.

Produces:
    {output_dir}/images/{ImageId}.png   - 512x512 grayscale PNG
    {output_dir}/original_masks/{ImageId}.png - 512x512 binary PNG (0 or 255)
    {output_dir}/dilated_masks/{ImageId}.png  - 512x512 binary PNG (0 or 255)
    {output_dir}/mask_variants.json     - processed mask-variant contract
    {output_dir}/splits.json            - train/val/test split (70/15/15, seed=42)
    {output_dir}/dataset_manifest.json  - trusted dataset version/fingerprint manifest

Usage:
    python -m src.data.preprocess \
        --raw_dir data/raw/SIIM-ACR \
        --output_dir data/processed/pneumothorax_trusted_v1 \
        --dataset_version pneumothorax_trusted_v1 \
        --img_size 512 \
        --rle_mode auto \
        --seed 42
"""

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.dataset_manifest import (
    DATASET_MANIFEST_FILENAME,
    DEFAULT_DATASET_ROOT,
    DEFAULT_DATASET_VERSION,
    build_dataset_manifest,
    repo_root_from_here,
)
from src.data.dicom_intensity import prepare_dicom_pixels_for_png, read_dicom_dataset
from src.data.mask_variants import build_mask_variant_manifest
from src.data.rle_contract import decode_runs, resolve_rle_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------

def rle2mask(
    rle_string: str,
    height: int = 1024,
    width: int = 1024,
    rle_mode: str = "cumulative_gap_pairs",
) -> np.ndarray:
    """Decode a single RLE string to a binary uint8 mask (0 or 255).

    The local ``train-rle.csv`` bundle resolves to ``cumulative_gap_pairs``:
    the first value is the absolute 1-based start and later values are
    cumulative zero-count gaps after the previous run. ``absolute_pairs`` is
    still supported explicitly for compatibility with standard start/length RLE.
    """
    flat = np.zeros(height * width, dtype=np.uint8)
    for start, length in decode_runs(rle_string, rle_mode=rle_mode):
        end = start + length
        if start < 0 or end > flat.size:
            raise ValueError(
                f"Decoded run [{start}, {end}) is out of bounds for mask size {flat.size}"
            )
        flat[start:end] = 255

    return flat.reshape((height, width), order="F")


def merge_rle_rows(
    rle_list: list[str],
    height: int = 1024,
    width: int = 1024,
    rle_mode: str = "cumulative_gap_pairs",
) -> np.ndarray:
    """Merge multiple RLE annotations for the same image (OR operation)."""
    merged = np.zeros((height, width), dtype=np.uint8)
    for rle in rle_list:
        mask = rle2mask(rle, height, width, rle_mode=rle_mode)
        merged = np.maximum(merged, mask)
    return merged


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_image(
    dicom_path: Path,
    output_image_path: Path,
    img_size: int,
) -> None:
    """Read a DICOM file, apply the accepted intensity policy, resize, and save as PNG."""
    ds = read_dicom_dataset(dicom_path, stop_before_pixels=False)
    pixels, _ = prepare_dicom_pixels_for_png(ds)

    img = Image.fromarray(pixels).convert("L")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img.save(output_image_path)


_DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def save_mask(mask: np.ndarray, output_mask_path: Path, img_size: int) -> None:
    """Resize a binary mask with nearest-neighbor interpolation and save it."""
    mask_img = Image.fromarray(mask).convert("L")
    mask_img = mask_img.resize((img_size, img_size), Image.NEAREST)
    mask_img.save(output_mask_path)


def process_mask(
    rle_list: list[str],
    output_original_mask_path: Path,
    output_dilated_mask_path: Path,
    img_size: int,
    orig_height: int = 1024,
    orig_width: int = 1024,
    rle_mode: str = "cumulative_gap_pairs",
) -> bool:
    """Merge RLE annotations and save both original and dilated mask variants.

    The original decoded mask and the dilated training-friendly mask are saved
    separately so the pipeline can train with one variant and report on the
    official variant without overwriting provenance.

    Returns True if the image has at least one pneumothorax region.
    """
    original_mask = merge_rle_rows(rle_list, height=orig_height, width=orig_width, rle_mode=rle_mode)
    has_pneumothorax = original_mask.max() > 0
    dilated_mask = original_mask.copy()

    # Dilate positive masks at original resolution before resize.
    if has_pneumothorax:
        dilated_mask = cv2.dilate(dilated_mask, _DILATION_KERNEL, iterations=1)

    save_mask(original_mask, output_original_mask_path, img_size)
    save_mask(dilated_mask, output_dilated_mask_path, img_size)

    return has_pneumothorax


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

_TEST_SIZE = 0.15
_VAL_SIZE_FROM_TRAIN_VAL = 0.17647058823529413
_SPLIT_POLICY_NAME = "stratified_two_stage_70_15_15_image_level_binary_label"


def create_splits(
    image_ids: list[str],
    positive_ids: set[str],
    seed: int,
) -> dict[str, list[str]]:
    """Return the deterministic publication-facing stratified split policy.

    Policy tracked in D-023:
    - stage 1: stratified train_val / test split with test_size=0.15
    - stage 2: stratified train / val split with val_size=0.17647058823529413
    - image-level binary labels are derived from original-mask foreground presence
    - final split IDs are sorted for stable manifests and diffs
    """
    sorted_ids = sorted(image_ids)
    labels = [1 if image_id in positive_ids else 0 for image_id in sorted_ids]

    train_val, test = train_test_split(
        sorted_ids,
        test_size=_TEST_SIZE,
        random_state=seed,
        stratify=labels,
    )

    train_val_labels = [1 if image_id in positive_ids else 0 for image_id in train_val]
    train, val = train_test_split(
        train_val,
        test_size=_VAL_SIZE_FROM_TRAIN_VAL,
        random_state=seed,
        stratify=train_val_labels,
    )
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    raw_dir: str,
    output_dir: str,
    img_size: int,
    seed: int,
    rle_mode: str,
    dataset_version: str,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    original_masks_dir = out_path / "original_masks"
    dilated_masks_dir = out_path / "dilated_masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    original_masks_dir.mkdir(parents=True, exist_ok=True)
    dilated_masks_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Read annotations CSV
    # ------------------------------------------------------------------
    csv_path = raw_path / "train-rle.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [column.strip() for column in df.columns]
    df["ImageId"] = df["ImageId"].str.strip()
    logger.info("Loaded CSV: %d rows, %d unique ImageIds", len(df), df["ImageId"].nunique())

    resolved_rle_mode, rle_evidence = resolve_rle_mode(
        df["EncodedPixels"].tolist(),
        requested_mode=rle_mode,
    )
    logger.info(
        "Resolved RLE mode: %s (positive_rows=%d, absolute_pairs=%d, cumulative_gap_pairs=%d)",
        resolved_rle_mode,
        rle_evidence.positive_rows,
        rle_evidence.valid_absolute_pairs,
        rle_evidence.valid_cumulative_gap_pairs,
    )

    # Group RLE strings by ImageId (a single image may span multiple rows).
    grouped = df.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()

    # ------------------------------------------------------------------
    # 2. Locate DICOM files
    # ------------------------------------------------------------------
    dicom_dir = raw_path / "dicom-images-train"
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    dicom_files: dict[str, Path] = {}
    for dcm in dicom_dir.rglob("*.dcm"):
        dicom_files[dcm.stem] = dcm

    logger.info("Found %d DICOM files", len(dicom_files))

    # ------------------------------------------------------------------
    # 3. Process each image
    # ------------------------------------------------------------------
    image_ids: list[str] = []
    positive_ids: set[str] = set()
    positive_count = 0
    skipped = 0

    for image_id, rle_list in tqdm(grouped.items(), desc="Processing"):
        dcm_path = dicom_files.get(image_id)
        if dcm_path is None:
            logger.warning("DICOM not found for ImageId: %s - skipping", image_id)
            skipped += 1
            continue

        out_image = images_dir / f"{image_id}.png"
        out_original_mask = original_masks_dir / f"{image_id}.png"
        out_dilated_mask = dilated_masks_dir / f"{image_id}.png"

        try:
            process_image(dcm_path, out_image, img_size)
            has_px = process_mask(
                rle_list,
                out_original_mask,
                out_dilated_mask,
                img_size,
                rle_mode=resolved_rle_mode,
            )
        except Exception as exc:
            logger.warning("Failed to process %s: %s - skipping", image_id, exc)
            skipped += 1
            continue

        image_ids.append(image_id)
        if has_px:
            positive_count += 1
            positive_ids.add(image_id)

    negative_count = len(image_ids) - positive_count
    logger.info(
        "Processed %d images (%d positive, %d negative). Skipped: %d",
        len(image_ids),
        positive_count,
        negative_count,
        skipped,
    )

    # ------------------------------------------------------------------
    # 4. Create and save splits
    # ------------------------------------------------------------------
    splits = create_splits(image_ids, positive_ids, seed)
    splits_path = out_path / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)

    logger.info(
        "Split - train: %d | val: %d | test: %d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    logger.info("splits.json saved to %s", splits_path)

    mask_variants_path = out_path / "mask_variants.json"
    mask_variant_manifest = build_mask_variant_manifest()
    with open(mask_variants_path, "w", encoding="utf-8") as handle:
        json.dump(mask_variant_manifest, handle, indent=2)
    logger.info("mask_variants.json saved to %s", mask_variants_path)

    # ------------------------------------------------------------------
    # 5. Verification + dataset manifest
    # ------------------------------------------------------------------
    n_images = len(list(images_dir.glob("*.png")))
    n_original_masks = len(list(original_masks_dir.glob("*.png")))
    n_dilated_masks = len(list(dilated_masks_dir.glob("*.png")))
    logger.info(
        "Verification - images: %d | original_masks: %d | dilated_masks: %d",
        n_images,
        n_original_masks,
        n_dilated_masks,
    )
    if n_images != n_original_masks or n_images != n_dilated_masks:
        logger.warning("Image/mask variant count mismatch. Check processing logs above.")

    repo_root = repo_root_from_here()
    code_paths = [
        repo_root / "src/data/preprocess.py",
        repo_root / "src/data/rle_contract.py",
        repo_root / "src/data/mask_variants.py",
        repo_root / "src/data/dicom_intensity.py",
        repo_root / "src/data/dataset_manifest.py",
        repo_root / "configs/config.yaml",
    ]
    manifest = build_dataset_manifest(
        dataset_version=dataset_version,
        dataset_root=out_path,
        raw_dir=raw_path,
        annotation_csv_path=csv_path,
        dicom_dir=dicom_dir,
        image_size=img_size,
        seed=seed,
        resolved_rle_mode=resolved_rle_mode,
        split_policy=_SPLIT_POLICY_NAME,
        skipped_images=skipped,
        splits=splits,
        mask_variant_manifest=mask_variant_manifest,
        code_paths=code_paths,
    )
    dataset_manifest_path = out_path / DATASET_MANIFEST_FILENAME
    with open(dataset_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info(
        "%s saved to %s (dataset_fingerprint=%s)",
        DATASET_MANIFEST_FILENAME,
        dataset_manifest_path,
        manifest["dataset_fingerprint"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SIIM-ACR DICOM data")
    parser.add_argument("--raw_dir", default="data/raw/SIIM-ACR", help="Raw data directory")
    parser.add_argument(
        "--output_dir", default=str(DEFAULT_DATASET_ROOT), help="Output directory"
    )
    parser.add_argument(
        "--dataset_version",
        default=DEFAULT_DATASET_VERSION,
        help="Dataset version label stored in dataset_manifest.json",
    )
    parser.add_argument("--img_size", type=int, default=512, help="Output image size")
    parser.add_argument(
        "--rle_mode",
        choices=["auto", "absolute_pairs", "cumulative_gap_pairs"],
        default="auto",
        help="RLE decoding mode. 'auto' validates the corpus and resolves one compatible mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = parser.parse_args()

    main(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        seed=args.seed,
        rle_mode=args.rle_mode,
        dataset_version=args.dataset_version,
    )
