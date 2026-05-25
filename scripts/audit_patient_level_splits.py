"""Read-only patient/study/series leakage audit for a processed split against raw DICOM metadata."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dicom_intensity import read_dicom_dataset

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only patient/study/series leakage audit for processed split vs raw DICOM metadata.",
    )
    parser.add_argument("--raw_dir", required=True, type=Path, help="Raw SIIM root containing dicom-images-train")
    parser.add_argument("--processed_dir", required=True, type=Path, help="Processed trusted dataset root containing splits.json")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory where audit artifacts are written")
    parser.add_argument("--strict", action="store_true", help="Return non-zero for audit incompleteness or metadata-quality warnings")
    parser.add_argument("--verbose", action="store_true", help="Print progress while reading DICOM metadata")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: scan at most N sorted DICOM files and mark audit INCOMPLETE")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DICOM root resolution
# ---------------------------------------------------------------------------


def resolve_dicom_root(raw_dir: Path) -> tuple[Path, bool]:
    """Return (dicom_root, used_fallback). Prefer raw_dir/dicom-images-train."""
    canonical = raw_dir / "dicom-images-train"
    if canonical.is_dir():
        return canonical, False
    fallback = raw_dir / "pneumothorax" / "dicom-images-train"
    if fallback.is_dir():
        return fallback, True
    sys.exit(2)


# ---------------------------------------------------------------------------
# Split loading / validation
# ---------------------------------------------------------------------------


def load_splits(processed_dir: Path) -> dict[str, list[str]]:
    path = processed_dir / "splits.json"
    if not path.is_file():
        print(f"ERROR: splits.json not found at {path}", file=sys.stderr)
        sys.exit(2)

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        print("ERROR: splits.json must be a JSON object", file=sys.stderr)
        sys.exit(2)

    expected_keys = {"train", "val", "test"}
    actual_keys = set(data.keys())
    if actual_keys != expected_keys:
        print(f"ERROR: splits.json keys must be exactly {sorted(expected_keys)}, got {sorted(actual_keys)}", file=sys.stderr)
        sys.exit(2)

    for key in expected_keys:
        if not isinstance(data[key], list):
            print(f"ERROR: splits.{key} must be a list of strings", file=sys.stderr)
            sys.exit(2)
        for item in data[key]:
            if not isinstance(item, str):
                print(f"ERROR: splits.{key} contains non-string value: {item!r}", file=sys.stderr)
                sys.exit(2)

    return data


# ---------------------------------------------------------------------------
# Split lookup / duplicate detection
# ---------------------------------------------------------------------------


def build_processed_split_lookup(splits: dict[str, list[str]]) -> tuple[dict[str, str], int, int]:
    """Return (lookup, within_dup_count, across_dup_count)."""
    lookup: dict[str, str] = {}
    seen: dict[str, str] = {}  # image_id -> first split
    within_dups = 0
    across_dups = 0

    for split_name, ids in splits.items():
        id_set = set()
        for img_id in ids:
            if img_id in id_set:
                within_dups += 1
            id_set.add(img_id)

            if img_id in seen:
                if seen[img_id] != split_name:
                    across_dups += 1
            else:
                seen[img_id] = split_name

            lookup[img_id] = split_name

    return lookup, within_dups, across_dups


# ---------------------------------------------------------------------------
# Processed dataset inspection
# ---------------------------------------------------------------------------


def inspect_processed_dataset(processed_dir: Path, splits: dict[str, list[str]]) -> dict:
    result: dict = {}

    images_dir = processed_dir / "images"
    images = sorted(images_dir.glob("*.png")) if images_dir.is_dir() else []
    result["image_png_count"] = len(images)

    original_masks_dir = processed_dir / "original_masks"
    result["original_mask_png_count"] = len(list(original_masks_dir.glob("*.png"))) if original_masks_dir.is_dir() else 0

    dilated_masks_dir = processed_dir / "dilated_masks"
    result["dilated_mask_png_count"] = len(list(dilated_masks_dir.glob("*.png"))) if dilated_masks_dir.is_dir() else 0

    image_stems = {p.stem for p in images}
    all_split_ids = set()
    for ids in splits.values():
        all_split_ids.update(ids)
    result["split_union_matches_images"] = (image_stems == all_split_ids)

    manifest_path = processed_dir / "dataset_manifest.json"
    result["manifest_found"] = manifest_path.is_file()
    result["manifest_malformed"] = False
    result["expected_image_count"] = None
    result["expected_count_source"] = None

    if result["manifest_found"]:
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            counts = manifest.get("counts", {})
            imgs = counts.get("images")
            if isinstance(imgs, int):
                result["expected_image_count"] = imgs
                result["expected_count_source"] = "dataset_manifest.counts.images"
            else:
                result["manifest_malformed"] = True
        except (json.JSONDecodeError, OSError):
            result["manifest_malformed"] = True

    if result["expected_image_count"] is None:
        result["expected_image_count"] = result["image_png_count"]
        if result["manifest_found"] and result["manifest_malformed"]:
            result["expected_count_source"] = "fallback:manifest_malformed"
        else:
            result["expected_count_source"] = "fallback:images_dir_png_count"

    result["split_total_matches_expected"] = (len(all_split_ids) == result["expected_image_count"])
    _exts = (".png", ".jpg", ".jpeg", ".dcm", ".tif", ".tiff", ".nii", ".nii.gz")
    result["ids_have_file_extensions"] = any(
        sid.lower().endswith(_exts) for sid in all_split_ids
    )

    return result


# ---------------------------------------------------------------------------
# DICOM file discovery
# ---------------------------------------------------------------------------


def find_dicom_files(dicom_root: Path, limit: int = 0) -> list[Path]:
    files = sorted(dicom_root.rglob("*.dcm"))
    if limit and limit > 0:
        files = files[:limit]
    return files


# ---------------------------------------------------------------------------
# DICOM metadata reading
# ---------------------------------------------------------------------------


_TAGS = ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")


def read_dicom_metadata(dicom_path: Path) -> dict:
    """Extract the four tag values; errors are surfaced as a dict with 'error'."""
    try:
        ds = read_dicom_dataset(dicom_path, stop_before_pixels=True)
    except Exception as exc:
        return {
            "error": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    return {
        tag: str(getattr(ds, tag, "")).strip()
        for tag in _TAGS
    }


def collect_dicom_metadata(
    dicom_files: list[Path],
    raw_dir: Path,
    dicom_root: Path,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Return (metadata_rows, read_error_rows)."""
    metadata_rows: list[dict] = []
    read_error_rows: list[dict] = []

    total = len(dicom_files)
    for idx, dicom_path in enumerate(dicom_files, 1):
        if verbose and idx % 2000 == 0:
            print(f"  ... read {idx}/{total} DICOM headers ...")

        meta = read_dicom_metadata(dicom_path)
        path_stem = dicom_path.stem

        if meta.get("error"):
            read_error_rows.append({
                "dicom_path": str(dicom_path),
                "error_type": meta["error_type"],
                "error_message": meta["error_message"],
            })
            row = {
                "processed_image_id": "",
                "split": "",
                "in_processed_split": "false",
                "match_method": "none",
                "image_id_from_path_stem": path_stem,
                "sop_instance_uid": "",
                "patient_id": "",
                "study_instance_uid": "",
                "series_instance_uid": "",
                "raw_relative_path": _posix_rel(dicom_path, raw_dir),
                "dicom_root_relative_path": _posix_rel(dicom_path, dicom_root),
                "path_stem_equals_sop_instance_uid": "",
                "patient_id_missing": "true",
                "study_instance_uid_missing": "true",
                "series_instance_uid_missing": "true",
                "sop_instance_uid_missing": "true",
                "dicom_read_error": "true",
                "dicom_read_error_type": meta["error_type"],
                "dicom_read_error_message": meta["error_message"],
            }
        else:
            patient_id = meta.get("PatientID", "")
            study_uid = meta.get("StudyInstanceUID", "")
            series_uid = meta.get("SeriesInstanceUID", "")
            sop_uid = meta.get("SOPInstanceUID", "")

            patient_missing = patient_id == ""
            study_missing = study_uid == ""
            series_missing = series_uid == ""
            sop_missing = sop_uid == ""

            path_eq_sop = ""
            if not sop_missing and path_stem:
                path_eq_sop = "true" if path_stem == sop_uid else "false"

            row = {
                "processed_image_id": "",
                "split": "",
                "in_processed_split": "false",
                "match_method": "none",
                "image_id_from_path_stem": path_stem,
                "sop_instance_uid": sop_uid,
                "patient_id": patient_id,
                "study_instance_uid": study_uid,
                "series_instance_uid": series_uid,
                "raw_relative_path": _posix_rel(dicom_path, raw_dir),
                "dicom_root_relative_path": _posix_rel(dicom_path, dicom_root),
                "path_stem_equals_sop_instance_uid": path_eq_sop,
                "patient_id_missing": "true" if patient_missing else "false",
                "study_instance_uid_missing": "true" if study_missing else "false",
                "series_instance_uid_missing": "true" if series_missing else "false",
                "sop_instance_uid_missing": "true" if sop_missing else "false",
                "dicom_read_error": "false",
                "dicom_read_error_type": "",
                "dicom_read_error_message": "",
            }

        row["_dicom_path"] = dicom_path  # internal use only; not written to CSV
        metadata_rows.append(row)

    if verbose:
        print(f"  ... finished reading {total} DICOM headers")

    return metadata_rows, read_error_rows


# ---------------------------------------------------------------------------
# Metadata indexes (for fast matching)
# ---------------------------------------------------------------------------


def build_metadata_indexes(metadata_rows: list[dict]) -> tuple[dict[str, list[int]], dict[str, list[int]], set[str]]:
    """Return (stem_index, sop_index, duplicate_sops).

    stem_index: path_stem -> list of row indices
    sop_index: non-empty SOPInstanceUID -> list of row indices
    duplicate_sops: SOPInstanceUIDs that appear more than once
    """
    stem_index: dict[str, list[int]] = defaultdict(list)
    sop_index: dict[str, list[int]] = defaultdict(list)

    for idx, row in enumerate(metadata_rows):
        stem = row["image_id_from_path_stem"]
        if stem:
            stem_index[stem].append(idx)

        sop = row["sop_instance_uid"]
        if sop:
            sop_index[sop].append(idx)

    duplicate_sops = {sop for sop, indices in sop_index.items() if len(indices) > 1}

    return dict(stem_index), dict(sop_index), duplicate_sops


# ---------------------------------------------------------------------------
# Join metadata to splits
# ---------------------------------------------------------------------------


def join_metadata_to_splits(
    metadata_rows: list[dict],
    split_lookup: dict[str, str],
    duplicate_sops: set[str],
) -> None:
    """Mutate metadata_rows in-place: assign processed_image_id, split, in_processed_split, match_method.

    Strategy:
    1. Match by path_stem directly.
    2. If path_stem not matched, fallback to exact unique SOPInstanceUID.
    """
    matched_splits: dict[int, str] = {}  # row_index -> split_name

    # --- Pass 1: path_stem match ---
    for idx, row in enumerate(metadata_rows):
        stem = row["image_id_from_path_stem"]
        if stem in split_lookup:
            row["processed_image_id"] = stem
            row["split"] = split_lookup[stem]
            row["in_processed_split"] = "true"
            row["match_method"] = "path_stem"
            matched_splits[idx] = split_lookup[stem]

    # --- Pass 2: SOP fallback ---
    for idx, row in enumerate(metadata_rows):
        if idx in matched_splits:
            continue
        sop = row["sop_instance_uid"]
        if not sop:
            continue
        if sop in duplicate_sops:
            # ambiguous — will be handled by detect_missing_cases
            continue
        if sop in split_lookup:
            row["processed_image_id"] = sop
            row["split"] = split_lookup[sop]
            row["in_processed_split"] = "true"
            row["match_method"] = "sop_instance_uid"
            matched_splits[idx] = split_lookup[sop]
            continue

    # --- Mark processed IDs that were NOT found in any row ---
    matched_processed_ids = {row["processed_image_id"] for row in metadata_rows if row["in_processed_split"] == "true"}


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------


def detect_cross_split_leakage(rows: list[dict], key_name: str) -> list[dict]:
    """Detect keys appearing in multiple splits among processed rows."""
    # key_name is csv column name; map to row key
    if key_name == "PatientID":
        row_key = "patient_id"
    elif key_name == "StudyInstanceUID":
        row_key = "study_instance_uid"
    elif key_name == "SeriesInstanceUID":
        row_key = "series_instance_uid"
    else:
        return []

    processed = [r for r in rows if r["in_processed_split"] == "true"]
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for row in processed:
        val = row[row_key]
        if not val:
            continue
        split = row["split"]
        grouped[val][split].append(row)

    cases: list[dict] = []
    for val, split_map in grouped.items():
        splits_set = sorted(split_map.keys())
        if len(splits_set) <= 1:
            continue

        all_rows_for_key = []
        for rows_in_split in split_map.values():
            all_rows_for_key.extend(rows_in_split)

        case = _build_leakage_case(
            case_type="cross_split_leakage",
            key_name=key_name,
            key_value=val,
            splits=splits_set,
            matched_rows=all_rows_for_key,
        )
        cases.append(case)

    return cases


def detect_sop_duplicates(rows: list[dict]) -> list[dict]:
    """Detect duplicate SOPInstanceUID among processed rows."""
    processed = [r for r in rows if r["in_processed_split"] == "true"]
    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in processed:
        sop = row["sop_instance_uid"]
        if not sop:
            continue
        grouped[sop].append(row)

    cases: list[dict] = []
    for sop, dup_rows in grouped.items():
        if len(dup_rows) <= 1:
            continue
        splits_set = sorted({r["split"] for r in dup_rows})
        case = _build_leakage_case(
            case_type="duplicate_sop_instance_uid",
            key_name="SOPInstanceUID",
            key_value=sop,
            splits=splits_set,
            matched_rows=dup_rows,
        )
        cases.append(case)

    return cases


def _pipe_sorted(values: set) -> str:
    return "|".join(sorted(values))


def _build_leakage_case(
    case_type: str,
    key_name: str,
    key_value: str,
    splits: list[str],
    matched_rows: list[dict],
) -> dict:
    image_ids = sorted({r["processed_image_id"] for r in matched_rows})
    patient_ids = {r["patient_id"] for r in matched_rows if r["patient_id"]}
    study_uids = {r["study_instance_uid"] for r in matched_rows if r["study_instance_uid"]}
    series_uids = {r["series_instance_uid"] for r in matched_rows if r["series_instance_uid"]}
    sop_uids = {r["sop_instance_uid"] for r in matched_rows if r["sop_instance_uid"]}
    raw_paths = {r["raw_relative_path"] for r in matched_rows if r["raw_relative_path"]}

    train_count = sum(1 for r in matched_rows if r["split"] == "train")
    val_count = sum(1 for r in matched_rows if r["split"] == "val")
    test_count = sum(1 for r in matched_rows if r["split"] == "test")

    return {
        "case_type": case_type,
        "key_name": key_name,
        "key_value": key_value,
        "splits": "|".join(splits),
        "split_count": len(splits),
        "image_count": len(matched_rows),
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "processed_image_ids": "|".join(image_ids),
        "patient_ids": _pipe_sorted(patient_ids),
        "study_instance_uids": _pipe_sorted(study_uids),
        "series_instance_uids": _pipe_sorted(series_uids),
        "sop_instance_uids": _pipe_sorted(sop_uids),
        "raw_relative_paths": _pipe_sorted(raw_paths),
    }


# ---------------------------------------------------------------------------
# Missing metadata cases
# ---------------------------------------------------------------------------


def detect_missing_cases(
    split_lookup: dict[str, str],
    metadata_rows: list[dict],
    read_error_rows: list[dict],
    duplicate_sops: set[str],
) -> list[dict]:
    """Emit missing cases for processed IDs with problems."""
    cases: list[dict] = []

    matched_processed_ids = {r["processed_image_id"] for r in metadata_rows if r["in_processed_split"] == "true"}
    all_processed_ids = set(split_lookup.keys())

    # 1. processed_id_missing_raw_metadata
    missing_ids = all_processed_ids - matched_processed_ids
    for pid in sorted(missing_ids):
        cases.append({
            "case_type": "processed_id_missing_raw_metadata",
            "processed_image_id": pid,
            "split": split_lookup.get(pid, ""),
            "tag_name": "",
            "image_id_from_path_stem": "",
            "sop_instance_uid": "",
            "raw_relative_path": "",
            "dicom_read_error_type": "",
            "message": f"Processed image ID {pid} ({split_lookup.get(pid, '?')}) has no matching raw DICOM metadata",
        })

    # 2. dicom_read_error (for rows that affect a processed ID via path stem)
    error_stems = {}
    for err in read_error_rows:
        p = Path(err["dicom_path"])
        error_stems[p.stem] = err

    for pid in sorted(missing_ids & set(error_stems.keys())):
        err = error_stems[pid]
        cases.append({
            "case_type": "dicom_read_error",
            "processed_image_id": pid,
            "split": split_lookup.get(pid, ""),
            "tag_name": "",
            "image_id_from_path_stem": pid,
            "sop_instance_uid": "",
            "raw_relative_path": "",
            "dicom_read_error_type": err.get("error_type", ""),
            "message": f"DICOM read error for stem {pid}: {err.get('error_type', '?')}: {err.get('error_message', '?')}",
        })

    # 3. missing_required_tag / empty_required_tag (processed rows)
    for row in metadata_rows:
        if row["in_processed_split"] != "true":
            continue

        for tag, col, missing_col in [
            ("PatientID", "patient_id", "patient_id_missing"),
            ("StudyInstanceUID", "study_instance_uid", "study_instance_uid_missing"),
            ("SeriesInstanceUID", "series_instance_uid", "series_instance_uid_missing"),
            ("SOPInstanceUID", "sop_instance_uid", "sop_instance_uid_missing"),
        ]:
            val = row[col]
            if row[missing_col] == "true":
                if not val:
                    cases.append({
                        "case_type": "empty_required_tag",
                        "processed_image_id": row["processed_image_id"],
                        "split": row["split"],
                        "tag_name": tag,
                        "image_id_from_path_stem": row["image_id_from_path_stem"],
                        "sop_instance_uid": row["sop_instance_uid"],
                        "raw_relative_path": row["raw_relative_path"],
                        "dicom_read_error_type": "",
                        "message": f"{tag} is empty for processed image {row['processed_image_id']} at {row['raw_relative_path']}",
                    })

    # 4. ambiguous_sop_match
    for sop in sorted(duplicate_sops):
        sop_rows = [r for r in metadata_rows if r["sop_instance_uid"] == sop and r["in_processed_split"] != "true"]
        num_processed = len([r for r in metadata_rows if r["sop_instance_uid"] == sop and r["in_processed_split"] == "true"])
        if not sop_rows or sop in split_lookup:
            continue
        for row in sop_rows:
            cases.append({
                "case_type": "ambiguous_sop_match",
                "processed_image_id": sop,
                "split": split_lookup.get(sop, ""),
                "tag_name": "",
                "image_id_from_path_stem": row["image_id_from_path_stem"],
                "sop_instance_uid": sop,
                "raw_relative_path": row["raw_relative_path"],
                "dicom_read_error_type": "",
                "message": f"Ambiguous SOP match: {sop} appears {len(sop_rows)} times among unprocessed rows; {num_processed} processed rows also share this SOP",
            })

    return cases


# ---------------------------------------------------------------------------
# Summary building
# ---------------------------------------------------------------------------


def _posix_rel(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = path
    return rel.as_posix()


def build_summary(
    args: argparse.Namespace,
    splits: dict[str, list[str]],
    split_lookup: dict[str, str],
    within_dups: int,
    across_dups: int,
    proc_info: dict,
    dicom_root: Path,
    used_fallback: bool,
    metadata_rows: list[dict],
    read_error_rows: list[dict],
    leakage_cases: list[dict],
    missing_cases: list[dict],
) -> dict:
    limited = bool(args.limit and args.limit > 0)

    # Split file
    split_total = sum(len(v) for v in splits.values())
    all_ids = list(split_lookup.keys())

    processed_rows = [r for r in metadata_rows if r["in_processed_split"] == "true"]

    # Raw DICOM
    raw_scanned = len(metadata_rows)
    raw_success = sum(1 for r in metadata_rows if r["dicom_read_error"] == "false")
    raw_error = raw_scanned - raw_success
    unique_stems = len({r["image_id_from_path_stem"] for r in metadata_rows if r["image_id_from_path_stem"]})
    raw_in_split = sum(1 for r in metadata_rows if r["in_processed_split"] == "true")
    raw_not_in_split = raw_success - raw_in_split

    matched_ids = {r["processed_image_id"] for r in processed_rows}
    processed_missing = len(set(all_ids) - matched_ids)

    # Path stem / SOP mismatch
    mismatch_count = 0
    for r in processed_rows:
        stem = r["image_id_from_path_stem"]
        sop = r["sop_instance_uid"]
        if stem and sop:
            if r["path_stem_equals_sop_instance_uid"] == "false":
                mismatch_count += 1

    # Primary match method
    method_counts = Counter(r["match_method"] for r in processed_rows)
    primary_match_method = method_counts.most_common(1)[0][0] if method_counts else "none"

    # Metadata completeness (processed rows only)
    def _completeness(row_key: str, missing_key: str) -> dict:
        total_processed = len(processed_rows)
        missing = sum(1 for r in processed_rows if r[missing_key] == "true")
        nonempty = total_processed - missing
        empty = sum(1 for r in processed_rows if r[missing_key] == "true" and r[row_key] == "")
        return {
            "nonempty_count": nonempty,
            "missing_count": missing,
            "empty_count": empty,
        }

    # Leakage
    def _leakage_stats(key_name: str, case_type: str = "cross_split_leakage") -> dict:
        matching = [c for c in leakage_cases if c["case_type"] == case_type and c["key_name"] == key_name]
        return {
            "cross_split_key_count" if case_type == "cross_split_leakage" else "duplicate_key_count": len(matching),
            "affected_image_count": sum(c["image_count"] for c in matching),
            "pass": len(matching) == 0,
        }

    def _sop_leakage_stats() -> dict:
        matching = [c for c in leakage_cases if c["case_type"] == "duplicate_sop_instance_uid"]
        return {
            "duplicate_key_count": len(matching),
            "affected_image_count": sum(c["image_count"] for c in matching),
            "pass": len(matching) == 0,
        }

    has_dicom_read_errors = len(read_error_rows) > 0
    has_fallback = used_fallback

    # Has any processed row got a missing/empty required tag?
    has_missing_tags = any(
        c["case_type"] in ("missing_required_tag", "empty_required_tag")
        for c in missing_cases
    )
    has_missing_metadata = len([c for c in missing_cases if c["case_type"] == "processed_id_missing_raw_metadata"]) > 0

    manifest_invalid = not proc_info["manifest_found"] or proc_info["manifest_malformed"]

    # Pass criteria
    pass_criteria = {
        "processed_image_total_matches_expected": proc_info["split_total_matches_expected"],
        "no_patient_id_cross_split_leakage": _leakage_stats("PatientID")["pass"],
        "no_study_instance_uid_cross_split_leakage": _leakage_stats("StudyInstanceUID")["pass"],
        "no_series_instance_uid_cross_split_leakage": _leakage_stats("SeriesInstanceUID")["pass"],
        "no_duplicated_sop_instance_uid": _sop_leakage_stats()["pass"],
        "no_processed_image_ids_missing_raw_metadata": processed_missing == 0,
    }

    all_pass = all(pass_criteria.values())

    # Strict-only failures
    strict_fail = False
    if limited:
        strict_fail = True
    if has_dicom_read_errors:
        strict_fail = True
    if has_fallback:
        strict_fail = True
    if has_missing_tags:
        strict_fail = True
    if manifest_invalid:
        strict_fail = True
    if mismatch_count > 0:
        strict_fail = True

    overall_status = "PASS"
    if limited:
        overall_status = "INCOMPLETE"
    elif not all_pass:
        overall_status = "FAIL"
    elif strict_fail and args.strict:
        overall_status = "FAIL"

    # Warnings
    warnings: list[str] = []
    if raw_not_in_split > 0:
        warnings.append(f"{raw_not_in_split} raw DICOM files not present in processed split")
    if not _HAS_YAML:
        warnings.append("PyYAML unavailable; using JSON-compatible YAML fallback")

    # Failures
    failures: list[str] = []
    if not all_pass:
        if not pass_criteria["processed_image_total_matches_expected"]:
            failures.append("processed_image_total_matches_expected")
        if not pass_criteria["no_patient_id_cross_split_leakage"]:
            failures.append("no_patient_id_cross_split_leakage")
        if not pass_criteria["no_study_instance_uid_cross_split_leakage"]:
            failures.append("no_study_instance_uid_cross_split_leakage")
        if not pass_criteria["no_series_instance_uid_cross_split_leakage"]:
            failures.append("no_series_instance_uid_cross_split_leakage")
        if not pass_criteria["no_duplicated_sop_instance_uid"]:
            failures.append("no_duplicated_sop_instance_uid")
        if not pass_criteria["no_processed_image_ids_missing_raw_metadata"]:
            failures.append("no_processed_image_ids_missing_raw_metadata")
    if strict_fail and args.strict:
        if limited:
            failures.append("--limit was used")
        if has_dicom_read_errors:
            failures.append("DICOM read errors exist")
        if has_fallback:
            failures.append("fallback DICOM root used")
        if has_missing_tags:
            failures.append("missing or empty required tags")
        if manifest_invalid:
            failures.append("missing or malformed dataset_manifest.json")
        if mismatch_count > 0:
            failures.append("path_stem/SOPInstanceUID mismatches detected")

    # Compute exit code
    exit_code = 0
    if overall_status == "FAIL":
        exit_code = 1
    elif limited:
        exit_code = 0 if not args.strict else 1

    _repo_root = REPO_ROOT

    def _repo_posix(p: Path) -> str:
        try:
            return p.relative_to(_repo_root).as_posix()
        except ValueError:
            return p.as_posix()

    summary = {
        "schema_version": 1,
        "audit_name": "patient_level_split_leakage_audit",
        "status": overall_status,
        "strict": args.strict,
        "limited_run": limited,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inputs": {
            "raw_dir": _repo_posix(args.raw_dir.resolve()),
            "selected_dicom_root": _repo_posix(dicom_root),
            "processed_dir": _repo_posix(args.processed_dir.resolve()),
            "splits_path": _repo_posix((args.processed_dir / "splits.json").resolve()),
            "images_dir": _repo_posix((args.processed_dir / "images").resolve()),
            "dataset_manifest_path": _repo_posix((args.processed_dir / "dataset_manifest.json").resolve()),
            "output_dir": _repo_posix(args.output_dir.resolve()),
        },
        "split_file": {
            "keys": ["train", "val", "test"],
            "counts": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
                "total": split_total,
            },
            "unique_id_count": len(all_ids),
            "duplicate_ids_within_split_count": within_dups,
            "duplicate_ids_across_splits_count": across_dups,
            "ids_have_file_extensions": proc_info["ids_have_file_extensions"],
        },
        "processed_dataset": {
            "expected_image_count": proc_info["expected_image_count"],
            "expected_count_source": proc_info["expected_count_source"],
            "image_png_count": proc_info["image_png_count"],
            "original_mask_png_count": proc_info["original_mask_png_count"],
            "dilated_mask_png_count": proc_info["dilated_mask_png_count"],
            "split_total_matches_expected": proc_info["split_total_matches_expected"],
            "split_union_matches_images": proc_info["split_union_matches_images"],
        },
        "raw_dicom": {
            "raw_dicom_files_scanned": raw_scanned,
            "raw_dicom_read_success_count": raw_success,
            "raw_dicom_read_error_count": raw_error,
            "raw_unique_path_stem_count": unique_stems,
            "raw_ids_in_processed_split_count": raw_in_split,
            "raw_ids_not_in_processed_split_count": raw_not_in_split,
            "processed_ids_missing_raw_metadata_count": processed_missing,
            "path_stem_sop_mismatch_count": mismatch_count,
            "primary_match_method": primary_match_method,
        },
        "metadata_completeness": {
            "PatientID": _completeness("patient_id", "patient_id_missing"),
            "StudyInstanceUID": _completeness("study_instance_uid", "study_instance_uid_missing"),
            "SeriesInstanceUID": _completeness("series_instance_uid", "series_instance_uid_missing"),
            "SOPInstanceUID": _completeness("sop_instance_uid", "sop_instance_uid_missing"),
        },
        "leakage": {
            "PatientID": _leakage_stats("PatientID"),
            "StudyInstanceUID": _leakage_stats("StudyInstanceUID"),
            "SeriesInstanceUID": _leakage_stats("SeriesInstanceUID"),
            "SOPInstanceUID": _sop_leakage_stats(),
        },
        "pass_criteria": pass_criteria,
        "outputs": {
            "dicom_split_metadata_csv": _repo_posix((args.output_dir.resolve() / "dicom_split_metadata.csv")),
            "leakage_cases_csv": _repo_posix((args.output_dir.resolve() / "leakage_cases.csv")),
            "missing_metadata_cases_csv": _repo_posix((args.output_dir.resolve() / "missing_metadata_cases.csv")),
            "summary_yaml": _repo_posix((args.output_dir.resolve() / "patient_split_audit_summary.yaml")),
        },
        "warnings": warnings,
        "failures": failures,
        "exit_code": exit_code,
    }

    return summary


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


_METADATA_CSV_FIELDS = [
    "processed_image_id",
    "split",
    "in_processed_split",
    "match_method",
    "image_id_from_path_stem",
    "sop_instance_uid",
    "patient_id",
    "study_instance_uid",
    "series_instance_uid",
    "raw_relative_path",
    "dicom_root_relative_path",
    "path_stem_equals_sop_instance_uid",
    "patient_id_missing",
    "study_instance_uid_missing",
    "series_instance_uid_missing",
    "sop_instance_uid_missing",
    "dicom_read_error",
    "dicom_read_error_type",
    "dicom_read_error_message",
]

_LEAKAGE_CSV_FIELDS = [
    "case_type",
    "key_name",
    "key_value",
    "splits",
    "split_count",
    "image_count",
    "train_count",
    "val_count",
    "test_count",
    "processed_image_ids",
    "patient_ids",
    "study_instance_uids",
    "series_instance_uids",
    "sop_instance_uids",
    "raw_relative_paths",
]

_MISSING_CSV_FIELDS = [
    "case_type",
    "processed_image_id",
    "split",
    "tag_name",
    "image_id_from_path_stem",
    "sop_instance_uid",
    "raw_relative_path",
    "dicom_read_error_type",
    "message",
]


def write_csv_outputs(
    output_dir: Path,
    metadata_rows: list[dict],
    leakage_cases: list[dict],
    missing_cases: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # dicom_split_metadata.csv
    with (output_dir / "dicom_split_metadata.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_METADATA_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow(row)

    # leakage_cases.csv
    with (output_dir / "leakage_cases.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LEAKAGE_CSV_FIELDS)
        writer.writeheader()
        for case in leakage_cases:
            writer.writerow(case)

    # missing_metadata_cases.csv
    with (output_dir / "missing_metadata_cases.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MISSING_CSV_FIELDS)
        writer.writeheader()
        for case in missing_cases:
            writer.writerow(case)


# ---------------------------------------------------------------------------
# Summary YAML output
# ---------------------------------------------------------------------------


def _json_compatible_yaml(data: dict, indent: int = 0) -> str:
    """Minimal JSON-compatible YAML writer. Only handles the summary structure."""
    lines: list[str] = []
    prefix = "  " * indent

    def _fmt(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            if any(ch in value for ch in ('"', "'", ":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", "-", "<", ">", "=", "!", "%", "@", "`")):
                return json.dumps(value)
            if value and (value[0] in "0123456789" or value.lower() in ("true", "false", "null", "yes", "no", "on", "off")):
                return json.dumps(value)
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return "null"
        return json.dumps(value)

    for key, value in data.items():
        if value is None:
            lines.append(f"{prefix}{key}: null")
        elif isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_json_compatible_yaml(value, indent + 1))
        elif isinstance(value, list):
            if not value:
                lines.append(f"{prefix}{key}: []")
            else:
                lines.append(f"{prefix}{key}:")
                for item in value:
                    lines.append(f"{prefix}  - {_fmt(item)}")
        else:
            lines.append(f"{prefix}{key}: {_fmt(value)}")

    return "\n".join(lines)


def write_summary_yaml(output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = output_dir / "patient_split_audit_summary.yaml"
    if _HAS_YAML:
        with yaml_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(summary, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    else:
        with yaml_path.open("w", encoding="utf-8") as fh:
            fh.write(_json_compatible_yaml(summary))
            fh.write("\n")


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def print_terminal_summary(summary: dict) -> None:
    s = summary
    print(f"patient_split_audit={s['status']}")
    print(f"processed_split_total={s['split_file']['counts']['total']}")
    print(f"raw_dicom_scanned={s['raw_dicom']['raw_dicom_files_scanned']}")
    print(f"matched_processed_metadata={s['raw_dicom']['raw_ids_in_processed_split_count']}")
    print(f"raw_not_in_split={s['raw_dicom']['raw_ids_not_in_processed_split_count']}")
    print(f"processed_missing_raw_metadata={s['raw_dicom']['processed_ids_missing_raw_metadata_count']}")
    print(f"patient_leakage_keys={s['leakage']['PatientID']['cross_split_key_count']}")
    print(f"study_leakage_keys={s['leakage']['StudyInstanceUID']['cross_split_key_count']}")
    print(f"series_leakage_keys={s['leakage']['SeriesInstanceUID']['cross_split_key_count']}")
    print(f"duplicate_sop_keys={s['leakage']['SOPInstanceUID']['duplicate_key_count']}")
    print(f"output_dir={s['inputs']['output_dir']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve()

    # Resolve DICOM root
    dicom_root, used_fallback = resolve_dicom_root(raw_dir)

    # Load splits
    splits = load_splits(processed_dir)
    split_lookup, within_dups, across_dups = build_processed_split_lookup(splits)

    # Inspect processed dataset
    proc_info = inspect_processed_dataset(processed_dir, splits)

    # Find DICOM files
    dicom_files = find_dicom_files(dicom_root, limit=args.limit)

    # Collect DICOM metadata
    if args.verbose:
        print(f"Scanning {len(dicom_files)} DICOM files under {dicom_root} ...")

    metadata_rows, read_error_rows = collect_dicom_metadata(
        dicom_files, raw_dir, dicom_root, verbose=args.verbose
    )

    # Build indexes and join metadata to splits
    stem_index, sop_index, duplicate_sops = build_metadata_indexes(metadata_rows)
    join_metadata_to_splits(metadata_rows, split_lookup, duplicate_sops)

    # Detect leakage
    leakage_cases: list[dict] = []
    for key_name in ("PatientID", "StudyInstanceUID", "SeriesInstanceUID"):
        leakage_cases.extend(detect_cross_split_leakage(metadata_rows, key_name))
    leakage_cases.extend(detect_sop_duplicates(metadata_rows))

    # Detect missing cases (including ambiguous SOP matches)
    missing_cases = detect_missing_cases(split_lookup, metadata_rows, read_error_rows, duplicate_sops)

    # Build summary
    summary = build_summary(
        args=args,
        splits=splits,
        split_lookup=split_lookup,
        within_dups=within_dups,
        across_dups=across_dups,
        proc_info=proc_info,
        dicom_root=dicom_root,
        used_fallback=used_fallback,
        metadata_rows=metadata_rows,
        read_error_rows=read_error_rows,
        leakage_cases=leakage_cases,
        missing_cases=missing_cases,
    )

    # Write outputs
    write_csv_outputs(output_dir, metadata_rows, leakage_cases, missing_cases)
    write_summary_yaml(output_dir, summary)

    # Print terminal summary
    print_terminal_summary(summary)

    # If FAIL or INCOMPLETE, print failure reasons to stderr
    if summary["status"] != "PASS":
        for failure in summary["failures"]:
            print(f"FAIL: {failure}", file=sys.stderr)

    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
