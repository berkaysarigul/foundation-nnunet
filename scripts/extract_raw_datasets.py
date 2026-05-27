"""Extract PR-11 raw SIIM-ACR and PTX-498 ZIP datasets idempotently."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pr11_utils import parse_bool, sha256_file, write_markdown_report, write_yaml


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    zip_path: Path
    out_dir: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PR-11 raw dataset ZIP files.")
    parser.add_argument("--siim-zip", type=Path, required=True)
    parser.add_argument("--ptx-zip", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--force", type=parse_bool, default=False)
    return parser.parse_args(argv)


def _entry_parts(name: str) -> list[str]:
    path = PurePosixPath(name.replace("\\", "/"))
    return [part for part in path.parts if part not in {"", "."}]


def _inspect_zip(zip_path: Path) -> dict[str, Any]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        entries = zf.infolist()
        file_entries = [e for e in entries if not e.is_dir()]
        top_roots = sorted({parts[0] for e in entries if (parts := _entry_parts(e.filename))})
        return {
            "zip_path": str(zip_path),
            "zip_sha256": sha256_file(zip_path),
            "entry_count": len(entries),
            "file_count": len(file_entries),
            "top_level_roots": top_roots,
            "first_entries": [e.filename for e in entries[:50]],
        }


def _strip_root_for_dataset(dataset_name: str, top_roots: list[str]) -> str | None:
    if dataset_name == "SIIM-ACR" and len(top_roots) == 1:
        return top_roots[0]
    return None


def _destination_for(entry_name: str, out_dir: Path, stripped_root: str | None) -> Path | None:
    parts = _entry_parts(entry_name)
    if not parts:
        return None
    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe ZIP path contains '..': {entry_name}")
    if PurePosixPath(entry_name).is_absolute():
        raise ValueError(f"Unsafe absolute ZIP path: {entry_name}")
    if stripped_root is not None:
        if parts[0] != stripped_root:
            raise ValueError(f"Expected root {stripped_root!r}, got {parts[0]!r} in {entry_name}")
        parts = parts[1:]
    if not parts:
        return None
    return out_dir.joinpath(*parts)


def _safe_remove_dir(path: Path, out_root: Path) -> None:
    resolved = path.resolve()
    root = out_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to remove path outside out-root: {path}") from exc
    if path.exists():
        shutil.rmtree(path)


def _previous_matches(summary: dict[str, Any], spec: DatasetSpec, inspect: dict[str, Any]) -> bool:
    datasets = summary.get("datasets", {})
    prior = datasets.get(spec.name, {}) if isinstance(datasets, dict) else {}
    return (
        prior.get("zip_sha256") == inspect["zip_sha256"]
        and int(prior.get("entry_count", -1)) == int(inspect["entry_count"])
        and Path(prior.get("output_dir", "")).resolve() == spec.out_dir.resolve()
    )


def _load_previous_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        return None


def _validate_destinations(zip_path: Path, out_dir: Path, stripped_root: str | None) -> dict[str, Any]:
    destinations: dict[Path, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.infolist():
            if entry.is_dir():
                continue
            dest = _destination_for(entry.filename, out_dir, stripped_root)
            if dest is None:
                continue
            resolved = dest.resolve()
            try:
                resolved.relative_to(out_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"Unsafe ZIP destination escapes output dir: {entry.filename}") from exc
            if dest in destinations:
                raise ValueError(
                    f"ZIP has duplicate destination {dest}: {destinations[dest]} and {entry.filename}"
                )
            destinations[dest] = entry.filename
    return {"destination_count": len(destinations)}


def _extract_zip(zip_path: Path, out_dir: Path, stripped_root: str | None) -> int:
    count = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.infolist():
            if entry.is_dir():
                continue
            dest = _destination_for(entry.filename, out_dir, stripped_root)
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry, "r") as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def process_dataset(
    spec: DatasetSpec,
    *,
    out_root: Path,
    force: bool,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    inspection = _inspect_zip(spec.zip_path)
    stripped_root = _strip_root_for_dataset(spec.name, inspection["top_level_roots"])
    inspection["stripped_root"] = stripped_root or ""
    inspection["output_dir"] = str(spec.out_dir)

    if spec.out_dir.exists():
        if previous and _previous_matches(previous, spec, inspection):
            inspection["status"] = "already_extracted"
            inspection["files_extracted"] = 0
            return inspection
        if not force:
            raise FileExistsError(
                f"{spec.out_dir} already exists and does not match extraction summary. "
                "Pass --force true to replace it."
            )
        _safe_remove_dir(spec.out_dir, out_root)

    inspection.update(_validate_destinations(spec.zip_path, spec.out_dir, stripped_root))
    inspection["files_extracted"] = _extract_zip(spec.zip_path, spec.out_dir, stripped_root)
    inspection["status"] = "extracted"
    return inspection


def write_report(path: Path, payload: dict[str, Any]) -> None:
    body: list[str] = []
    for name, info in payload["datasets"].items():
        body.extend(
            [
                f"### {name}",
                "",
                f"- ZIP: `{info['zip_path']}`",
                f"- output: `{info['output_dir']}`",
                f"- status: `{info['status']}`",
                f"- entries: `{info['entry_count']}`",
                f"- files extracted this run: `{info['files_extracted']}`",
                f"- top-level roots: `{', '.join(info['top_level_roots'])}`",
                f"- stripped root: `{info.get('stripped_root') or ''}`",
                "",
            ]
        )
    write_markdown_report(path, "PR-11 Raw Dataset Extraction Report", [("Datasets", body)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "extraction_summary.yaml"
    report_path = out_root / "extraction_report.md"
    previous = _load_previous_summary(summary_path)

    specs = [
        DatasetSpec("SIIM-ACR", args.siim_zip, out_root / "SIIM-ACR"),
        DatasetSpec("PTX-498", args.ptx_zip, out_root / "PTX-498"),
    ]
    payload = {
        "schema_version": 1,
        "out_root": str(out_root),
        "force": bool(args.force),
        "datasets": {},
    }
    for spec in specs:
        payload["datasets"][spec.name] = process_dataset(
            spec,
            out_root=out_root,
            force=bool(args.force),
            previous=previous,
        )

    write_yaml(summary_path, payload)
    write_report(report_path, payload)
    print(f"[done] extraction report written to {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

