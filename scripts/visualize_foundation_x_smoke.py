"""
Foundation X smoke test visual diagnostics.

Generates per-stage PCA-RGB, mean heatmap, and overlay images for each case.
Called by smoke_foundation_x.py or run standalone with saved .npy features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _pca_rgb(feat: np.ndarray, max_pixels: int = 4096, seed: int = 42) -> np.ndarray:
    """
    feat: (C, H, W) float32 numpy array.
    Returns: (H, W, 3) float32 in [0, 1].
    Uses top-3 right singular vectors computed on a random pixel subsample.
    """
    C, H, W = feat.shape
    flat = feat.reshape(C, H * W).T  # (H*W, C)
    flat = flat - flat.mean(axis=0, keepdims=True)

    n = flat.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(n, max_pixels), replace=False)
    sub = flat[idx]

    _, _, Vt = np.linalg.svd(sub, full_matrices=False)
    n_components = min(3, Vt.shape[0], C)
    V3 = Vt[:n_components].T  # (C, n_components)

    proj = flat @ V3  # (H*W, n_components)
    if n_components < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - n_components)))

    proj = proj.reshape(H, W, 3)
    lo = np.percentile(proj, 1, axis=(0, 1), keepdims=True)
    hi = np.percentile(proj, 99, axis=(0, 1), keepdims=True)
    proj = np.clip((proj - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return proj.astype(np.float32)


def _mean_heatmap(feat: np.ndarray) -> np.ndarray:
    """
    feat: (C, H, W) float32 numpy array.
    Returns: (H, W) float32 in [0, 1] (viridis colormapping applied at save time).
    """
    mean_map = feat.mean(axis=0)  # (H, W)
    lo = np.percentile(mean_map, 1)
    hi = np.percentile(mean_map, 99)
    return np.clip((mean_map - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def _upscale_to(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upscale (H, W) or (H, W, C) array to target size with INTER_NEAREST."""
    if arr.shape[0] == target_h and arr.shape[1] == target_w:
        return arr
    return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _save_rgb(arr: np.ndarray, path: Path) -> None:
    """arr: (H, W, 3) float32 in [0,1]."""
    img = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(str(path))


def _save_heatmap(arr: np.ndarray, path: Path) -> None:
    """arr: (H, W) float32 in [0,1]. Saved with viridis colormap as RGB PNG."""
    colored = (cm.viridis(arr)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(str(path))


def _feat_to_numpy(feat: Union["torch.Tensor", np.ndarray]) -> np.ndarray:
    """Convert a feature (C,H,W) or (1,C,H,W) tensor/array to (C,H,W) float32 numpy."""
    if hasattr(feat, "numpy"):
        arr = feat.float().numpy()
    else:
        arr = np.asarray(feat, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def render_case(
    case_id: str,
    image_uint8: np.ndarray,
    feats: list,
    output_dir: Path,
) -> dict:
    """
    Generate per-stage PCA-RGB, mean heatmap, and stage-0 overlay for one case.

    Args:
        case_id:      e.g. 'siim_009589'
        image_uint8:  (H, W) uint8 grayscale input image
        feats:        List of 4 items, each (C, H, W) Tensor or numpy array
        output_dir:   Parent of 'visuals/' directory

    Returns:
        Dict mapping stage index to saved file paths.
    """
    case_dir = output_dir / "visuals" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    H_in, W_in = image_uint8.shape

    input_path = case_dir / "input.png"
    Image.fromarray(image_uint8).save(str(input_path))

    saved: dict = {"input": str(input_path), "stages": {}}

    for stage_idx, feat in enumerate(feats):
        feat_np = _feat_to_numpy(feat)

        pca = _pca_rgb(feat_np)
        pca_up = _upscale_to(pca, H_in, W_in)
        pca_path = case_dir / f"stage{stage_idx}_pca_rgb.png"
        _save_rgb(pca_up, pca_path)

        heat = _mean_heatmap(feat_np)
        heat_up = _upscale_to(heat, H_in, W_in)
        heat_path = case_dir / f"stage{stage_idx}_mean_heatmap.png"
        _save_heatmap(heat_up, heat_path)

        stage_saved: dict = {
            "pca_rgb": str(pca_path),
            "mean_heatmap": str(heat_path),
        }

        if stage_idx == 0:
            input_rgb = np.stack([image_uint8 / 255.0] * 3, axis=-1).astype(np.float32)
            overlay = np.clip(0.55 * input_rgb + 0.45 * pca_up, 0.0, 1.0)
            overlay_path = case_dir / f"stage{stage_idx}_overlay.png"
            _save_rgb(overlay, overlay_path)
            stage_saved["overlay"] = str(overlay_path)

        saved["stages"][stage_idx] = stage_saved

    return saved


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render Foundation X smoke visuals from saved .npy feature files.",
    )
    p.add_argument("--case_id", required=True, help="Case ID (e.g. siim_009589)")
    p.add_argument("--image_path", required=True, type=Path, help="Path to the input grayscale PNG image")
    p.add_argument(
        "--features_dir",
        required=True,
        type=Path,
        help="Directory containing <case_id>_stage{0..3}.npy saved by --save_raw_features",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output parent directory; a visuals/<case_id>/ subdirectory will be created",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    image_uint8 = np.array(Image.open(args.image_path).convert("L"), dtype=np.uint8)
    feats = [np.load(str(args.features_dir / f"{args.case_id}_stage{i}.npy")) for i in range(4)]
    render_case(args.case_id, image_uint8, feats, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
