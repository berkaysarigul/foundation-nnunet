"""PTX-498 filename discovery helpers for PR-11B."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


PTX_ROLE_SUFFIXES = {
    "image_png": ".img.png",
    "mask_png": ".mask.png",
    "merge_png": ".merge.png",
    "image_nii": ".img.nii.gz",
    "mask_nii": ".mask.nii.gz",
}

PTX_ROLE_ORDINALS = {
    "image_png": "1",
    "mask_png": "2",
    "merge_png": "3",
    "image_nii": "4",
    "mask_nii": "5",
}

EXPECTED_SITE_NAMES = {"SiteA", "SiteB", "SiteC"}


def ptx_role_for_name(name: str) -> str | None:
    for role, suffix in PTX_ROLE_SUFFIXES.items():
        if name.endswith(suffix):
            return role
    return None


def raw_case_stem(name: str, role: str) -> str:
    suffix = PTX_ROLE_SUFFIXES[role]
    if not name.endswith(suffix):
        raise ValueError(f"{name!r} does not end with PTX suffix {suffix!r}")
    return name[: -len(suffix)]


def normalize_case_stem(name: str, role: str) -> str:
    """Return the case key used for pairing PTX modality files.

    PTX-498 files are commonly named like ``1.1.img.png`` and
    ``1.2.mask.png``. Removing only ``.img.png`` leaves role ordinals in the
    stem, so strip the expected role ordinal when present. Files without the
    ordinal, such as ``case001.img.png``, are preserved.
    """

    stem = raw_case_stem(name, role)
    ordinal = PTX_ROLE_ORDINALS.get(role)
    if ordinal and stem.endswith(f".{ordinal}") and len(stem) > len(ordinal) + 1:
        return stem[: -(len(ordinal) + 1)]
    return stem


def safe_token(value: str, *, fallback: str = "unknown") -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return normalized or fallback


def ptx_case_id(site: str, case_stem: str) -> str:
    return f"ptx498_{safe_token(site)}_{safe_token(case_stem, fallback='case')}"


def natural_sort_key(value: str) -> list[object]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def discover_site_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    site_dirs: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_file() and ptx_role_for_name(path.name) is not None:
            site_dirs.add(path.parent)
    if not site_dirs and any(
        path.is_file() and ptx_role_for_name(path.name) is not None for path in root.iterdir()
    ):
        site_dirs.add(root)
    return sorted(site_dirs, key=lambda p: natural_sort_key(p.relative_to(root).as_posix()))


def collect_site_files(site_dir: Path) -> dict[str, dict[str, list[Path]]]:
    by_role: dict[str, dict[str, list[Path]]] = {role: {} for role in PTX_ROLE_SUFFIXES}
    for path in sorted((p for p in site_dir.iterdir() if p.is_file()), key=lambda p: natural_sort_key(p.name)):
        role = ptx_role_for_name(path.name)
        if role is None:
            continue
        stem = normalize_case_stem(path.name, role)
        by_role[role].setdefault(stem, []).append(path)
    return by_role


def first_path(grouped: dict[str, dict[str, list[Path]]], role: str, stem: str) -> Path | None:
    paths = grouped.get(role, {}).get(stem, [])
    return paths[0] if paths else None


def duplicate_role_rows(
    grouped: dict[str, dict[str, list[Path]]],
    *,
    site: str,
    relpath,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for role, by_stem in grouped.items():
        for stem, paths in by_stem.items():
            if len(paths) <= 1:
                continue
            rows.append(
                {
                    "site": site,
                    "case_stem": stem,
                    "role": role,
                    "paths": ";".join(relpath(path) for path in paths),
                }
            )
    return rows


def detect_v2_fix_evidence(root: Path, paths: Iterable[Path]) -> dict[str, object]:
    checked = [root, *paths]
    evidence = []
    for path in checked:
        text = str(path).lower()
        if "v2" in text and "fix" in text:
            evidence.append(str(path))
    return {
        "detectable": bool(evidence),
        "evidence": evidence[:20],
    }
