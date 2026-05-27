"""Dataset for Foundation X prior-guided refiner training and evaluation."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


INPUT_MODES = {"image_prior", "image_only", "prior_only"}
NORMALIZATION_MODES = {"zero_one", "image_zscore"}
REQUIRED_COLUMNS = {
    "case_id",
    "image_path",
    "label_path",
    "probability_map_path",
}


def input_channels_for_mode(input_mode: str) -> int:
    if input_mode == "image_prior":
        return 2
    if input_mode in {"image_only", "prior_only"}:
        return 1
    raise ValueError(
        f"Unsupported input_mode {input_mode!r}. Expected one of {sorted(INPUT_MODES)}."
    )


def _resolve_manifest_path(raw_path: str, repo_root: Path) -> Path:
    normalized = str(raw_path).strip().replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_grayscale_float(path: Path, *, target_size: int, interpolation: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), interpolation)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _load_binary_mask(path: Path, *, target_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), Image.NEAREST)
        arr = np.asarray(image)
    return (arr > 0).astype(np.float32)


def _normalize_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zero_one":
        return image
    if mode == "image_zscore":
        mean = float(image.mean())
        std = float(image.std())
        if std < 1e-6:
            std = 1.0
        return ((image - mean) / std).astype(np.float32)
    raise ValueError(
        f"Unsupported normalization {mode!r}. Expected one of {sorted(NORMALIZATION_MODES)}."
    )


class FoundationXPriorRefinerDataset(Dataset):
    """Load Dataset101 image/mask pairs with Foundation X probability priors."""

    def __init__(
        self,
        manifest_csv: str | Path,
        *,
        input_mode: str | None = None,
        mode: str | None = None,
        target_size: int = 512,
        augment: bool = False,
        normalization: str = "zero_one",
        repo_root: str | Path | None = None,
    ) -> None:
        selected_mode = input_mode if input_mode is not None else mode
        if selected_mode is None:
            selected_mode = "image_prior"
        if mode is not None and input_mode is not None and mode != input_mode:
            raise ValueError(f"Conflicting mode={mode!r} and input_mode={input_mode!r}.")
        if selected_mode not in INPUT_MODES:
            raise ValueError(
                f"Unsupported input_mode {selected_mode!r}. Expected one of {sorted(INPUT_MODES)}."
            )
        if normalization not in NORMALIZATION_MODES:
            raise ValueError(
                f"Unsupported normalization {normalization!r}. "
                f"Expected one of {sorted(NORMALIZATION_MODES)}."
            )
        if int(target_size) <= 0:
            raise ValueError("target_size must be a positive integer.")

        self.manifest_csv = Path(manifest_csv)
        self.input_mode = selected_mode
        self.target_size = int(target_size)
        self.augment = bool(augment)
        self.normalization = normalization
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()

        with self.manifest_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - columns)
            if missing:
                raise ValueError(
                    f"Manifest {self.manifest_csv} is missing required columns: {missing}"
                )
            self.rows = [dict(row) for row in reader]

        if not self.rows:
            raise ValueError(f"Manifest {self.manifest_csv} contains no rows.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        row = self.rows[index]
        image_path = _resolve_manifest_path(row["image_path"], self.repo_root)
        label_path = _resolve_manifest_path(row["label_path"], self.repo_root)
        probability_map_path = _resolve_manifest_path(row["probability_map_path"], self.repo_root)

        image = _load_grayscale_float(
            image_path,
            target_size=self.target_size,
            interpolation=Image.BILINEAR,
        )
        prior = _load_grayscale_float(
            probability_map_path,
            target_size=self.target_size,
            interpolation=Image.BILINEAR,
        )
        mask = _load_binary_mask(label_path, target_size=self.target_size)

        if self.augment and random.random() < 0.5:
            image = np.fliplr(image).copy()
            prior = np.fliplr(prior).copy()
            mask = np.fliplr(mask).copy()

        image = _normalize_image(image, self.normalization)

        if self.input_mode == "image_prior":
            x_arr = np.stack([image, prior], axis=0)
        elif self.input_mode == "image_only":
            x_arr = image[None, ...]
        else:
            x_arr = prior[None, ...]

        y_arr = mask[None, ...]
        metadata = {
            "case_id": str(row["case_id"]),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "probability_map_path": str(probability_map_path),
        }
        return (
            torch.from_numpy(x_arr.astype(np.float32, copy=False)),
            torch.from_numpy(y_arr.astype(np.float32, copy=False)),
            metadata,
        )
