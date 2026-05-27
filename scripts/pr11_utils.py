"""Shared helpers for PR-11 SIIM/PTX data preparation scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
    import yaml as _yaml  # type: ignore

    _HAS_YAML = True
except ImportError:  # pragma: no cover - yaml is in requirements.txt
    _HAS_YAML = False


REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_BINARY_VALUES = {0, 1, 255}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected true/false value, got {value!r}.")


def repo_relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_existing_path(raw: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return repo_root / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    payload = to_jsonable(payload)
    if _HAS_YAML:
        with path.open("w", encoding="utf-8") as handle:
            _yaml.safe_dump(payload, handle, sort_keys=False)
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_for_csv(row.get(key, "")) for key in fieldnames})


def scalar_for_csv(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def mask_stats(path: Path) -> dict[str, Any]:
    arr = load_grayscale(path)
    values = sorted(int(v) for v in np.unique(arr).tolist())
    foreground = int((arr > 0).sum())
    return {
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "unique_values": values,
        "is_binary": set(values).issubset(MASK_BINARY_VALUES),
        "foreground_pixels": foreground,
        "is_positive": foreground > 0,
    }


def binarize_mask(path: Path) -> np.ndarray:
    return (load_grayscale(path) > 0).astype(np.uint8)


def save_binary_mask(arr: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray((arr > 0).astype(np.uint8)).save(path)


def save_uint8(arr: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def resize_uint8(path: Path, size: tuple[int, int], interpolation: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != size:
            image = image.resize(size, interpolation)
        return np.asarray(image, dtype=np.uint8)


def probability_to_uint8(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob.astype(np.float32), 0.0, 1.0).reshape(prob.shape) * 255.0


def size_distribution(paths: Iterable[Path], max_items: int = 0) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idx, path in enumerate(paths):
        if max_items and idx >= max_items:
            break
        try:
            width, height = image_size(path)
        except Exception:
            key = "unreadable"
        else:
            key = f"{width}x{height}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def infer_siim_filename_parts(filename: str) -> dict[str, str]:
    match = re.match(r"^(?P<index>\d+)_(?P<split>train|test)_(?P<target>[01])_\.png$", filename)
    if not match:
        return {"index": "", "split": "", "target": ""}
    return match.groupdict()


def sanitize_identifier(value: str, prefix: str = "") -> str:
    stem = Path(value).stem
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    if prefix:
        normalized = f"{prefix}_{normalized}"
    return normalized


def percentile_summary(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def make_overlay(image_path: Path, mask_path: Path, out_path: Path, *, pred_path: Path | None = None) -> None:
    with Image.open(image_path) as image:
        image = image.convert("L")
        base = np.asarray(image, dtype=np.uint8)
    gt = binarize_mask(mask_path)
    if gt.shape != base.shape:
        gt = np.asarray(
            Image.fromarray(gt).resize((base.shape[1], base.shape[0]), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    rgb[gt > 0, 1] = 255
    rgb[gt > 0, 0] *= 0.35
    if pred_path is not None and pred_path.exists():
        pred = binarize_mask(pred_path)
        if pred.shape != base.shape:
            pred = np.asarray(
                Image.fromarray(pred).resize((base.shape[1], base.shape[0]), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        rgb[pred > 0, 0] = 255
        rgb[pred > 0, 1] *= 0.35
        rgb[(gt > 0) & (pred > 0)] = np.array([255, 220, 0], dtype=np.float32)
    ensure_dir(out_path.parent)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out_path)


def write_markdown_report(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> None:
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", ""])
        lines.extend(body)
        lines.append("")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def normalized_thumbnail_hash(path: Path, size: int = 64) -> str:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32)
    arr = arr - float(arr.mean())
    std = float(arr.std())
    if std > 1e-6:
        arr = arr / std
    quantized = np.clip((arr + 4.0) / 8.0 * 255.0, 0, 255).astype(np.uint8)
    return hashlib.sha256(quantized.tobytes()).hexdigest()

