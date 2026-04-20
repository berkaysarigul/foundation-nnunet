"""Helpers for publication-grade repeated split studies on the trusted dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from src.data.dataset_manifest import compute_split_fingerprint
from src.data.preprocess import create_splits


def load_processed_dataset_binary_labels(dataset_root: str | Path) -> tuple[list[str], set[str]]:
    """Load sorted image IDs plus original-mask-derived positive IDs."""
    dataset_path = Path(dataset_root)
    images_dir = dataset_path / "images"
    original_masks_dir = dataset_path / "original_masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not original_masks_dir.exists():
        raise FileNotFoundError(f"original_masks directory not found: {original_masks_dir}")

    image_ids = sorted(path.stem for path in images_dir.glob("*.png"))
    if not image_ids:
        raise ValueError(f"No processed images found under {images_dir}.")

    positive_ids: set[str] = set()
    for image_id in image_ids:
        mask_path = original_masks_dir / f"{image_id}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image_id={image_id!r}: {mask_path}")
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if mask.max() > 0:
            positive_ids.add(image_id)

    return image_ids, positive_ids


def build_repeated_split_instances(
    image_ids: list[str],
    positive_ids: set[str],
    *,
    split_seeds: Iterable[int],
) -> list[dict[str, object]]:
    """Build canonical repeated split instances from the trusted single-split policy."""
    normalized_seeds = [int(seed) for seed in split_seeds]
    if not normalized_seeds:
        raise ValueError("split_seeds must contain at least one seed.")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("split_seeds must be unique.")

    split_instances: list[dict[str, object]] = []
    seen_fingerprints: dict[str, int] = {}
    for index, split_seed in enumerate(normalized_seeds, start=1):
        splits = create_splits(image_ids, positive_ids, seed=split_seed)
        split_fingerprint = compute_split_fingerprint(splits)
        previous_seed = seen_fingerprints.get(split_fingerprint)
        if previous_seed is not None:
            raise ValueError(
                "Repeated split studies require distinct split instances; "
                f"seed {split_seed} reproduced the same split fingerprint as seed {previous_seed}."
            )
        seen_fingerprints[split_fingerprint] = split_seed
        split_instances.append(
            {
                "split_instance_id": f"split_{index:03d}",
                "split_seed": split_seed,
                "train_ids": splits["train"],
                "val_ids": splits["val"],
                "test_ids": splits["test"],
            }
        )

    return split_instances
