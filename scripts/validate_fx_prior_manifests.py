"""
PR-10B gate 1: validate Foundation X prior manifests for train and heldout test.

This script is the precondition for any PR-10B refiner training. It must pass
locally (with dryrun manifests) and on Colab (with the full 9073/1602 manifests)
before the refiner dataset loader, model, or trainer is invoked.

It is diagnostic only. It does not modify the Foundation X export pipeline or
any Dataset101 file. Outputs are written under the user-supplied --out dir.

Checks (see PR-10B plan section 2 for the authoritative list):

Schema / structural
  - CSV is readable and has the PR-10A 20-column header.
  - case_id is unique within each manifest.
  - Provenance triple is constant: state_key=teacher_model,
    preprocess_variant=official_siim_224, head_key=head_5.
  - split column is the expected single value per manifest.

File existence (every row)
  - image_path resolves on disk under --dataset-root or repo root.
  - label_path resolves on disk.
  - probability_map_path resolves on disk.

Leakage
  - Train manifest: no row references imagesTs/ or heldout_labelsTs/.
  - Test manifest: no row references imagesTr/ (unless --allow-imagesTr-in-test).
  - case_id sets of train and test are disjoint.

Probability map sanity (sampled, default 32 rows per manifest)
  - PNG opens, mode L, dtype uint8.
  - Pixel range fits [0, 255]; degenerate (min == max) flagged as warning.
  - Records H x W per sample.

Image / label / prob alignment (sampled)
  - image and label share H x W. Mismatch is a hard fail.
  - prob H x W may differ from image (e.g. exported at 224); recorded but only
    a warning, unless --strict.

Mask binarity (sampled)
  - Label PNG values are a subset of {0, 1} or {0, 255}.

CLI:

    python scripts/validate_fx_prior_manifests.py \\
        --train-manifest <path/to/train_manifest.csv> \\
        --test-manifest  <path/to/test_manifest.csv>  \\
        --dataset-root   nnUNet_raw/Dataset101_Pneumothorax \\
        --out            artifacts/refiner/dataset_audit/<run_tag>/ \\
        [--strict] \\
        [--expected-train-rows 9073] \\
        [--expected-test-rows 1602] \\
        [--sample-size 32] \\
        [--allow-imagesTr-in-test] \\
        [--repo-root <path>]

Exit code is 0 if all hard checks pass, 1 otherwise. Warnings do not affect the
exit code unless --strict is set.

Outputs in --out:
  - summary.yaml         machine-readable verdict + per-check details
  - report.md            human-readable summary
  - manifest_audit.csv   per-row audit (existence checks + sampled file stats)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
    import yaml as _yaml  # type: ignore

    _HAS_YAML = True
except ImportError:  # pragma: no cover - yaml is in requirements.txt
    _HAS_YAML = False


# --- constants ---------------------------------------------------------------

MANIFEST_COLUMNS = [
    "case_id",
    "split",
    "image_path",
    "label_path",
    "probability_map_path",
    "binary_mask_path",
    "state_key",
    "preprocess_variant",
    "head_key",
    "threshold",
    "gt_is_positive",
    "pred_is_positive",
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "gt_foreground_pixels",
    "pred_foreground_pixels",
]

EXPECTED_PROVENANCE = {
    "state_key": "teacher_model",
    "preprocess_variant": "official_siim_224",
    "head_key": "head_5",
}

TRAIN_ALLOWED_SPLITS = {"train"}
TEST_ALLOWED_SPLITS = {"test", "eval", "heldout", "val"}

LEAKAGE_TOKENS_TRAIN = ("imagesTs", "heldout_labelsTs")
LEAKAGE_TOKEN_TEST_TRAIN_IMAGES = "imagesTr"


# --- data classes ------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str = "error"  # "error" or "warning"
    detail: str = ""
    offenders: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "severity": self.severity,
            "detail": self.detail,
            "offenders_sample": self.offenders[:10],
            "offenders_count": len(self.offenders),
        }


@dataclass
class ManifestAudit:
    kind: str  # "train" or "test"
    manifest_path: Path
    row_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    duplicate_case_ids: list[str] = field(default_factory=list)
    sampled_image_sizes: dict[str, int] = field(default_factory=dict)  # "HxW" -> count
    sampled_label_sizes: dict[str, int] = field(default_factory=dict)
    sampled_prob_sizes: dict[str, int] = field(default_factory=dict)
    sampled_prob_min: int | None = None
    sampled_prob_max: int | None = None
    degenerate_prob_samples: list[str] = field(default_factory=list)
    label_unique_values_seen: list[int] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    case_ids: set[str] = field(default_factory=set)
    per_row: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "manifest_path": str(self.manifest_path),
            "row_count": self.row_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "duplicate_case_ids": self.duplicate_case_ids[:20],
            "sampled_image_sizes": self.sampled_image_sizes,
            "sampled_label_sizes": self.sampled_label_sizes,
            "sampled_prob_sizes": self.sampled_prob_sizes,
            "sampled_prob_min": self.sampled_prob_min,
            "sampled_prob_max": self.sampled_prob_max,
            "degenerate_prob_samples": self.degenerate_prob_samples[:20],
            "label_unique_values_seen": sorted(set(self.label_unique_values_seen)),
            "checks": [c.to_dict() for c in self.checks],
        }


# --- helpers -----------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Foundation X prior manifests (train + heldout test) for PR-10B.",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset101 root, e.g. nnUNet_raw/Dataset101_Pneumothorax",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory; summary.yaml, report.md, manifest_audit.csv written here.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings (e.g. prob/image size mismatch, degenerate prob maps) as errors.",
    )
    parser.add_argument(
        "--expected-train-rows",
        type=int,
        default=None,
        help="If set, fail when train row count does not match exactly.",
    )
    parser.add_argument(
        "--expected-test-rows",
        type=int,
        default=None,
        help="If set, fail when test row count does not match exactly.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=32,
        help="Number of rows per manifest to fully load (image + label + prob) for stats.",
    )
    parser.add_argument(
        "--allow-imagesTr-in-test",
        action="store_true",
        help="Permit test manifest rows whose image_path points under imagesTr (default: forbidden).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root for resolving relative manifest paths. Defaults to parent of scripts/.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="RNG seed for the per-manifest sampling of full reads.",
    )
    return parser.parse_args(argv)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _norm_path_str(raw: str) -> str:
    """Normalize Windows-style backslashes to POSIX for joining/parsing."""
    return raw.replace("\\", "/").strip()


def _resolve_manifest_path(raw: str, repo_root: Path) -> Path:
    """Resolve a manifest-recorded path against repo_root if relative."""
    s = _norm_path_str(raw)
    if not s:
        return Path("")
    p = Path(s)
    if p.is_absolute():
        return p
    return (repo_root / s).resolve(strict=False)


def _path_contains_segment(raw: str, token: str) -> bool:
    """Return True if `token` appears as a path segment in `raw` (Windows or POSIX)."""
    s = _norm_path_str(raw)
    segments = PurePosixPath(s).parts
    return token in segments


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = [row for row in reader if any((v or "").strip() for v in row.values())]
    return header, rows


def _parse_bool(value: str) -> bool | None:
    s = (value or "").strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    return None


def _sanitize_for_yaml(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_sanitize_for_yaml(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _sanitize_for_yaml(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_yaml(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# --- per-row file existence audit -------------------------------------------


def _audit_existence(
    manifest_rows: list[dict[str, str]],
    repo_root: Path,
    kind: str,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    per_row: list[dict[str, Any]] = []
    missing_images: list[str] = []
    missing_labels: list[str] = []
    missing_probs: list[str] = []
    for row in manifest_rows:
        case_id = (row.get("case_id") or "").strip()
        image_raw = row.get("image_path", "") or ""
        label_raw = row.get("label_path", "") or ""
        prob_raw = row.get("probability_map_path", "") or ""
        image_p = _resolve_manifest_path(image_raw, repo_root)
        label_p = _resolve_manifest_path(label_raw, repo_root)
        prob_p = _resolve_manifest_path(prob_raw, repo_root)

        img_exists = image_p.exists() if str(image_p) else False
        lbl_exists = label_p.exists() if str(label_p) else False
        prb_exists = prob_p.exists() if str(prob_p) else False

        if not img_exists:
            missing_images.append(case_id)
        if not lbl_exists:
            missing_labels.append(case_id)
        if not prb_exists:
            missing_probs.append(case_id)

        per_row.append(
            {
                "manifest_kind": kind,
                "case_id": case_id,
                "image_path": image_raw,
                "label_path": label_raw,
                "probability_map_path": prob_raw,
                "image_exists": img_exists,
                "label_exists": lbl_exists,
                "prob_exists": prb_exists,
                "image_size_hxw": "",
                "label_size_hxw": "",
                "prob_size_hxw": "",
                "prob_min": "",
                "prob_max": "",
                "label_unique_values": "",
                "alignment_ok": "",
                "sampled": False,
            }
        )
    return per_row, missing_images, missing_labels, missing_probs


# --- sampled deep audit ------------------------------------------------------


def _audit_sampled(
    manifest_rows: list[dict[str, str]],
    per_row: list[dict[str, Any]],
    repo_root: Path,
    sample_size: int,
    seed: int,
    audit: ManifestAudit,
) -> tuple[list[str], list[str]]:
    """Pick `sample_size` rows where all three files exist; load and record stats."""
    candidates = [
        idx for idx, r in enumerate(per_row)
        if r["image_exists"] and r["label_exists"] and r["prob_exists"]
    ]
    if not candidates:
        return [], []
    rng = random.Random(seed)
    pick = rng.sample(candidates, min(sample_size, len(candidates)))
    nonbinary_labels: list[str] = []
    alignment_mismatches: list[str] = []

    prob_min_global: int | None = None
    prob_max_global: int | None = None

    for idx in pick:
        row = per_row[idx]
        case_id = row["case_id"]
        try:
            image = Image.open(_resolve_manifest_path(row["image_path"], repo_root)).convert("L")
            label = Image.open(_resolve_manifest_path(row["label_path"], repo_root)).convert("L")
            prob = Image.open(_resolve_manifest_path(row["probability_map_path"], repo_root)).convert("L")
        except Exception as exc:  # noqa: BLE001
            row["sampled"] = True
            row["alignment_ok"] = False
            row["prob_min"] = "ERR"
            row["prob_max"] = f"open_failed: {exc!r}"
            continue

        img_arr = np.asarray(image)
        lbl_arr = np.asarray(label)
        prb_arr = np.asarray(prob)

        img_hxw = f"{img_arr.shape[0]}x{img_arr.shape[1]}"
        lbl_hxw = f"{lbl_arr.shape[0]}x{lbl_arr.shape[1]}"
        prb_hxw = f"{prb_arr.shape[0]}x{prb_arr.shape[1]}"

        audit.sampled_image_sizes[img_hxw] = audit.sampled_image_sizes.get(img_hxw, 0) + 1
        audit.sampled_label_sizes[lbl_hxw] = audit.sampled_label_sizes.get(lbl_hxw, 0) + 1
        audit.sampled_prob_sizes[prb_hxw] = audit.sampled_prob_sizes.get(prb_hxw, 0) + 1

        unique_vals = sorted(int(v) for v in np.unique(lbl_arr).tolist())
        for v in unique_vals:
            audit.label_unique_values_seen.append(v)
        is_binary = set(unique_vals).issubset({0, 1, 255})
        if not is_binary:
            nonbinary_labels.append(case_id)

        prob_min = int(prb_arr.min()) if prb_arr.size else 0
        prob_max = int(prb_arr.max()) if prb_arr.size else 0
        prob_min_global = prob_min if prob_min_global is None else min(prob_min_global, prob_min)
        prob_max_global = prob_max if prob_max_global is None else max(prob_max_global, prob_max)
        if prob_min == prob_max:
            audit.degenerate_prob_samples.append(case_id)

        image_label_align = img_arr.shape == lbl_arr.shape
        if not image_label_align:
            alignment_mismatches.append(case_id)

        row["sampled"] = True
        row["image_size_hxw"] = img_hxw
        row["label_size_hxw"] = lbl_hxw
        row["prob_size_hxw"] = prb_hxw
        row["prob_min"] = prob_min
        row["prob_max"] = prob_max
        row["label_unique_values"] = ",".join(str(v) for v in unique_vals)
        row["alignment_ok"] = image_label_align

    audit.sampled_prob_min = prob_min_global
    audit.sampled_prob_max = prob_max_global
    return nonbinary_labels, alignment_mismatches


# --- per-manifest validation -------------------------------------------------


def _check_header(header: list[str]) -> CheckResult:
    missing = [c for c in MANIFEST_COLUMNS if c not in header]
    extras = [c for c in header if c not in MANIFEST_COLUMNS]
    passed = not missing
    detail = (
        f"missing={missing}; extras={extras}"
        if (missing or extras)
        else "schema matches PR-10A 20-column contract"
    )
    return CheckResult(
        name="schema_header",
        passed=passed,
        severity="error",
        detail=detail,
        offenders=missing,
    )


def _check_row_count(rows: list[dict[str, str]], expected: int | None, kind: str) -> CheckResult:
    actual = len(rows)
    if expected is None:
        return CheckResult(
            name=f"{kind}_row_count",
            passed=True,
            severity="warning",
            detail=f"row count not validated; got {actual} (no --expected-{kind}-rows supplied)",
        )
    passed = actual == expected
    return CheckResult(
        name=f"{kind}_row_count",
        passed=passed,
        severity="error",
        detail=f"expected {expected}, got {actual}",
    )


def _check_unique_case_ids(rows: list[dict[str, str]], kind: str) -> tuple[CheckResult, list[str]]:
    seen: dict[str, int] = {}
    for row in rows:
        cid = (row.get("case_id") or "").strip()
        seen[cid] = seen.get(cid, 0) + 1
    dupes = sorted(cid for cid, n in seen.items() if n > 1)
    return (
        CheckResult(
            name=f"{kind}_case_id_unique",
            passed=not dupes,
            severity="error",
            detail=f"{len(dupes)} duplicate case_id(s)",
            offenders=dupes,
        ),
        dupes,
    )


def _check_split_values(rows: list[dict[str, str]], allowed: set[str], kind: str) -> CheckResult:
    bad: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sv = (row.get("split") or "").strip()
        seen.add(sv)
        if sv not in allowed:
            bad.append((row.get("case_id") or "").strip())
    return CheckResult(
        name=f"{kind}_split_value",
        passed=not bad,
        severity="error",
        detail=f"observed splits={sorted(seen)}; allowed={sorted(allowed)}",
        offenders=bad,
    )


def _check_provenance(rows: list[dict[str, str]], kind: str) -> CheckResult:
    bad: list[str] = []
    for row in rows:
        for col, want in EXPECTED_PROVENANCE.items():
            if (row.get(col) or "").strip() != want:
                bad.append((row.get("case_id") or "").strip())
                break
    return CheckResult(
        name=f"{kind}_provenance",
        passed=not bad,
        severity="error",
        detail=f"expected {EXPECTED_PROVENANCE}",
        offenders=bad,
    )


def _check_leakage_train(rows: list[dict[str, str]]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for token in LEAKAGE_TOKENS_TRAIN:
        offenders: list[str] = []
        for row in rows:
            cid = (row.get("case_id") or "").strip()
            for col in ("image_path", "label_path", "probability_map_path", "binary_mask_path"):
                if _path_contains_segment(row.get(col, "") or "", token):
                    offenders.append(cid)
                    break
        results.append(
            CheckResult(
                name=f"train_leakage_{token}",
                passed=not offenders,
                severity="error",
                detail=f"no path may include the '{token}/' segment in train manifest",
                offenders=offenders,
            )
        )
    return results


def _check_leakage_test(rows: list[dict[str, str]], allow_imagestr: bool) -> CheckResult:
    if allow_imagestr:
        return CheckResult(
            name="test_leakage_imagesTr",
            passed=True,
            severity="warning",
            detail="--allow-imagesTr-in-test was set; check skipped",
        )
    offenders: list[str] = []
    for row in rows:
        cid = (row.get("case_id") or "").strip()
        if _path_contains_segment(row.get("image_path", "") or "", LEAKAGE_TOKEN_TEST_TRAIN_IMAGES):
            offenders.append(cid)
    return CheckResult(
        name="test_leakage_imagesTr",
        passed=not offenders,
        severity="error",
        detail="no image_path may include the 'imagesTr/' segment in test manifest",
        offenders=offenders,
    )


def _check_file_existence(
    missing_images: list[str],
    missing_labels: list[str],
    missing_probs: list[str],
    kind: str,
) -> list[CheckResult]:
    return [
        CheckResult(
            name=f"{kind}_image_paths_exist",
            passed=not missing_images,
            severity="error",
            detail=f"{len(missing_images)} missing image_path",
            offenders=missing_images,
        ),
        CheckResult(
            name=f"{kind}_label_paths_exist",
            passed=not missing_labels,
            severity="error",
            detail=f"{len(missing_labels)} missing label_path",
            offenders=missing_labels,
        ),
        CheckResult(
            name=f"{kind}_probability_map_paths_exist",
            passed=not missing_probs,
            severity="error",
            detail=f"{len(missing_probs)} missing probability_map_path",
            offenders=missing_probs,
        ),
    ]


def _check_label_binarity(nonbinary: list[str], kind: str) -> CheckResult:
    return CheckResult(
        name=f"{kind}_label_binarity_sampled",
        passed=not nonbinary,
        severity="error",
        detail="sampled label pixel values must be a subset of {0, 1, 255}",
        offenders=nonbinary,
    )


def _check_alignment_image_label(mismatched: list[str], kind: str) -> CheckResult:
    return CheckResult(
        name=f"{kind}_image_label_alignment_sampled",
        passed=not mismatched,
        severity="error",
        detail="sampled image and label must share H x W",
        offenders=mismatched,
    )


def _check_prob_range(audit: ManifestAudit, kind: str) -> CheckResult:
    if audit.sampled_prob_min is None or audit.sampled_prob_max is None:
        return CheckResult(
            name=f"{kind}_prob_range_sampled",
            passed=True,
            severity="warning",
            detail="no samples loaded (no rows had all three files present)",
        )
    in_range = 0 <= audit.sampled_prob_min and audit.sampled_prob_max <= 255
    return CheckResult(
        name=f"{kind}_prob_range_sampled",
        passed=in_range,
        severity="error",
        detail=f"sampled prob uint8 range observed: [{audit.sampled_prob_min}, {audit.sampled_prob_max}]",
    )


def _check_prob_degeneracy(audit: ManifestAudit, kind: str) -> CheckResult:
    return CheckResult(
        name=f"{kind}_prob_nondegenerate_sampled",
        passed=not audit.degenerate_prob_samples,
        severity="warning",
        detail="sampled prob maps with min == max are usually all-zero placeholders",
        offenders=audit.degenerate_prob_samples,
    )


def _check_prob_size_uniform(audit: ManifestAudit, kind: str) -> CheckResult:
    sizes = list(audit.sampled_prob_sizes.keys())
    return CheckResult(
        name=f"{kind}_prob_size_uniform_sampled",
        passed=len(sizes) <= 1,
        severity="warning",
        detail=f"sampled prob H x W values: {audit.sampled_prob_sizes}",
    )


# --- orchestration -----------------------------------------------------------


def audit_manifest(
    manifest_path: Path,
    kind: str,
    repo_root: Path,
    *,
    expected_rows: int | None,
    allowed_splits: set[str],
    sample_size: int,
    sample_seed: int,
    allow_imagestr_in_test: bool = False,
) -> ManifestAudit:
    header, rows = _read_manifest(manifest_path)
    audit = ManifestAudit(kind=kind, manifest_path=manifest_path, row_count=len(rows))

    header_check = _check_header(header)
    audit.checks.append(header_check)

    audit.checks.append(_check_row_count(rows, expected_rows, kind))

    uniq_check, dupes = _check_unique_case_ids(rows, kind)
    audit.duplicate_case_ids = dupes
    audit.checks.append(uniq_check)

    audit.checks.append(_check_split_values(rows, allowed_splits, kind))
    audit.checks.append(_check_provenance(rows, kind))

    if kind == "train":
        audit.checks.extend(_check_leakage_train(rows))
    else:
        audit.checks.append(_check_leakage_test(rows, allow_imagestr_in_test))

    pos = 0
    neg = 0
    for row in rows:
        b = _parse_bool(row.get("gt_is_positive", ""))
        if b is True:
            pos += 1
        elif b is False:
            neg += 1
    audit.positive_count = pos
    audit.negative_count = neg

    audit.case_ids = {(r.get("case_id") or "").strip() for r in rows}

    per_row, missing_images, missing_labels, missing_probs = _audit_existence(rows, repo_root, kind)
    audit.per_row = per_row
    audit.checks.extend(_check_file_existence(missing_images, missing_labels, missing_probs, kind))

    nonbinary, alignment_mismatches = _audit_sampled(rows, per_row, repo_root, sample_size, sample_seed, audit)
    audit.checks.append(_check_label_binarity(nonbinary, kind))
    audit.checks.append(_check_alignment_image_label(alignment_mismatches, kind))
    audit.checks.append(_check_prob_range(audit, kind))
    audit.checks.append(_check_prob_degeneracy(audit, kind))
    audit.checks.append(_check_prob_size_uniform(audit, kind))

    return audit


def _cross_manifest_disjoint(train: ManifestAudit, test: ManifestAudit) -> CheckResult:
    overlap = sorted(train.case_ids & test.case_ids)
    return CheckResult(
        name="case_id_disjoint_train_test",
        passed=not overlap,
        severity="error",
        detail=f"{len(overlap)} case_id(s) appear in both train and test manifests",
        offenders=overlap,
    )


def _verdict(audits: list[ManifestAudit], strict: bool) -> tuple[str, list[str]]:
    failed: list[str] = []
    for audit in audits:
        for c in audit.checks:
            if c.passed:
                continue
            if c.severity == "error" or (strict and c.severity == "warning"):
                failed.append(f"{audit.kind}:{c.name}")
    return ("PASS" if not failed else "FAIL"), failed


# --- output writers ----------------------------------------------------------


def _write_summary(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = _sanitize_for_yaml(payload)
    if _HAS_YAML:
        (out_dir / "summary.yaml").write_text(
            _yaml.safe_dump(serializable, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    (out_dir / "summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _write_audit_csv(out_dir: Path, audits: list[ManifestAudit]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "manifest_kind",
        "case_id",
        "image_path",
        "label_path",
        "probability_map_path",
        "image_exists",
        "label_exists",
        "prob_exists",
        "sampled",
        "image_size_hxw",
        "label_size_hxw",
        "prob_size_hxw",
        "prob_min",
        "prob_max",
        "label_unique_values",
        "alignment_ok",
    ]
    with (out_dir / "manifest_audit.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for audit in audits:
            for row in audit.per_row:
                writer.writerow({k: row.get(k, "") for k in columns})


def _format_check(c: CheckResult) -> str:
    status = "PASS" if c.passed else ("WARN" if c.severity == "warning" else "FAIL")
    line = f"  - [{status}] {c.name}: {c.detail}"
    if not c.passed and c.offenders:
        line += f"\n      first offenders: {c.offenders[:5]}"
    return line


def _write_report(out_dir: Path, audits: list[ManifestAudit], cross: CheckResult,
                  verdict: str, failed_checks: list[str], strict: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# PR-10B manifest validation report",
        "",
        f"- Generated at: {_dt.datetime.now(_dt.timezone.utc).isoformat()}",
        f"- Strict mode: {strict}",
        f"- Overall verdict: **{verdict}**",
    ]
    if failed_checks:
        lines.append(f"- Failed checks ({len(failed_checks)}): {failed_checks[:10]}"
                     + (" ..." if len(failed_checks) > 10 else ""))
    lines.append("")

    for audit in audits:
        lines += [
            f"## {audit.kind} manifest",
            f"- Path: `{audit.manifest_path}`",
            f"- Row count: {audit.row_count}",
            f"- Positives / negatives: {audit.positive_count} / {audit.negative_count}",
            f"- Sampled image sizes: {audit.sampled_image_sizes}",
            f"- Sampled label sizes: {audit.sampled_label_sizes}",
            f"- Sampled prob sizes: {audit.sampled_prob_sizes}",
            f"- Sampled prob range: [{audit.sampled_prob_min}, {audit.sampled_prob_max}]",
            f"- Sampled label unique values seen: "
            f"{sorted(set(audit.label_unique_values_seen))}",
            "",
            "### Checks",
        ]
        for c in audit.checks:
            lines.append(_format_check(c))
        lines.append("")

    lines += [
        "## Cross-manifest",
        _format_check(cross),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# --- main --------------------------------------------------------------------


def run(
    train_manifest: Path,
    test_manifest: Path,
    dataset_root: Path,
    out_dir: Path,
    *,
    strict: bool = False,
    expected_train_rows: int | None = None,
    expected_test_rows: int | None = None,
    sample_size: int = 32,
    allow_imagestr_in_test: bool = False,
    repo_root: Path | None = None,
    sample_seed: int = 42,
) -> dict[str, Any]:
    repo_root = (repo_root or _default_repo_root()).resolve()

    train_audit = audit_manifest(
        manifest_path=train_manifest,
        kind="train",
        repo_root=repo_root,
        expected_rows=expected_train_rows,
        allowed_splits=TRAIN_ALLOWED_SPLITS,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    test_audit = audit_manifest(
        manifest_path=test_manifest,
        kind="test",
        repo_root=repo_root,
        expected_rows=expected_test_rows,
        allowed_splits=TEST_ALLOWED_SPLITS,
        sample_size=sample_size,
        sample_seed=sample_seed,
        allow_imagestr_in_test=allow_imagestr_in_test,
    )
    cross = _cross_manifest_disjoint(train_audit, test_audit)

    audits = [train_audit, test_audit]
    # attach cross check to verdict computation
    class _Wrap:
        kind = "cross"
        checks = [cross]
    verdict, failed = _verdict(audits + [_Wrap()], strict=strict)  # type: ignore[list-item]

    payload: dict[str, Any] = {
        "verdict": verdict,
        "strict_mode": strict,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "repo_root": str(repo_root),
        "expected_train_rows": expected_train_rows,
        "expected_test_rows": expected_test_rows,
        "sample_size": sample_size,
        "allow_imagestr_in_test": allow_imagestr_in_test,
        "train": train_audit.to_dict(),
        "test": test_audit.to_dict(),
        "cross": cross.to_dict(),
        "failed_checks": failed,
    }

    _write_summary(out_dir, payload)
    _write_audit_csv(out_dir, audits)
    _write_report(out_dir, audits, cross, verdict, failed, strict)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run(
            train_manifest=args.train_manifest,
            test_manifest=args.test_manifest,
            dataset_root=args.dataset_root,
            out_dir=args.out,
            strict=args.strict,
            expected_train_rows=args.expected_train_rows,
            expected_test_rows=args.expected_test_rows,
            sample_size=args.sample_size,
            allow_imagestr_in_test=args.allow_imagesTr_in_test,
            repo_root=args.repo_root,
            sample_seed=args.sample_seed,
        )
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    verdict = payload["verdict"]
    print(f"[{verdict}] PR-10B manifest validation written to {args.out}")
    if payload["failed_checks"]:
        print(f"failed_checks: {payload['failed_checks'][:10]}"
              + (" ..." if len(payload['failed_checks']) > 10 else ""))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
