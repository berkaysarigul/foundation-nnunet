"""Export trusted pneumothorax dataset to nnU-Net v2 raw PNG format."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


_ACCEPTED_MASK_VALUES = ({0, 255}, {0, 1})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trusted pneumothorax dataset to nnU-Net v2 raw PNG format.",
    )
    parser.add_argument("--dataset_id", type=int, required=True, help="Integer dataset ID (PR-2: 101)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset suffix (PR-2: Pneumothorax)")
    parser.add_argument("--source_dir", type=Path, required=True, help="Processed trusted dataset root")
    parser.add_argument("--nnunet_raw", type=Path, default=None, help="nnU-Net raw root (falls back to $env:nnUNet_raw)")
    parser.add_argument("--mask_variant", type=str, required=True, help="Mask subdirectory name (PR-2: original_masks)")
    parser.add_argument("--export_heldout_labels", action="store_true", help="Export source test labels into heldout_labelsTs")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output dataset folder")
    parser.add_argument("--no_reader_writer_override", action="store_true", help="Omit overwrite_image_reader_writer from dataset.json")
    parser.add_argument("--allow_non_original_masks", action="store_true", help="Permit mask variants other than original_masks")
    parser.add_argument("--dry_run", action="store_true", help="Validate and print planned counts without writing files")
    parser.add_argument("--limit", type=int, default=0, help="Debug only; allowed only with --dry_run")
    return parser.parse_args()


def resolve_nnunet_raw_root(args: argparse.Namespace) -> Path:
    if args.nnunet_raw:
        raw = Path(args.nnunet_raw)
    else:
        env_val = os.environ.get("nnUNet_raw", "")
        if not env_val:
            print("ERROR: --nnunet_raw not provided and $env:nnUNet_raw is not set.", file=sys.stderr)
            sys.exit(2)
        raw = Path(env_val)
    if not raw or str(raw).strip() == "":
        print("ERROR: nnU-Net raw root is empty.", file=sys.stderr)
        sys.exit(2)
    return raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _posix_rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# Source validation helpers
# ---------------------------------------------------------------------------


def load_splits(source_dir: Path) -> dict[str, list[str]]:
    path = source_dir / "splits.json"
    if not path.is_file():
        print(f"ERROR: splits.json not found at {path}", file=sys.stderr)
        sys.exit(2)

    data = _load_json(path)
    expected = {"train", "val", "test"}
    actual = set(data.keys())
    if actual != expected:
        print(f"ERROR: splits.json keys must be {sorted(expected)}, got {sorted(actual)}", file=sys.stderr)
        sys.exit(2)

    for key in expected:
        if not isinstance(data[key], list) or not all(isinstance(v, str) for v in data[key]):
            print(f"ERROR: splits.{key} must be a list of strings", file=sys.stderr)
            sys.exit(2)

    return data


def load_source_manifests(source_dir: Path) -> tuple[dict, dict]:
    manifest_path = source_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: dataset_manifest.json not found at {manifest_path}", file=sys.stderr)
        sys.exit(2)
    manifest = _load_json(manifest_path)

    variants_path = source_dir / "mask_variants.json"
    if not variants_path.is_file():
        print(f"ERROR: mask_variants.json not found at {variants_path}", file=sys.stderr)
        sys.exit(2)
    variants = _load_json(variants_path)

    return manifest, variants


def load_pr1_summary() -> dict | None:
    pr1_path = REPO_ROOT / "artifacts" / "diagnostics" / "patient_split_audit" / "patient_split_audit_summary.yaml"
    if not pr1_path.is_file():
        return None

    if _HAS_YAML:
        with pr1_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    else:
        try:
            with pr1_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data
        except (json.JSONDecodeError, OSError):
            return None


def validate_source_dataset(
    source_dir: Path,
    splits: dict[str, list[str]],
    mask_variant: str,
    manifest: dict,
    variants: dict,
) -> None:
    images_dir = source_dir / "images"
    masks_dir = source_dir / mask_variant

    if not images_dir.is_dir():
        print(f"ERROR: images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(2)
    if not masks_dir.is_dir():
        print(f"ERROR: mask variant directory not found: {masks_dir}", file=sys.stderr)
        sys.exit(2)

    all_split_ids = set()
    for ids in splits.values():
        all_split_ids.update(ids)

    image_stems = {p.stem for p in images_dir.glob("*.png")}
    mask_stems = {p.stem for p in masks_dir.glob("*.png")}

    if image_stems != mask_stems:
        sym_diff = image_stems.symmetric_difference(mask_stems)
        print(f"ERROR: image and mask stems do not match. Symmetric difference: {len(sym_diff)} stems.", file=sys.stderr)
        sys.exit(2)

    if image_stems != all_split_ids:
        sym_diff = image_stems.symmetric_difference(all_split_ids)
        print(f"ERROR: split IDs and image stems do not match. Symmetric difference: {len(sym_diff)} stems.", file=sys.stderr)
        sys.exit(2)

    total = len(all_split_ids)
    print(f"Source validation: {total} images, {len(image_stems)} image stems, {len(mask_stems)} mask stems, masks from {mask_variant}")

    dupes_within = 0
    dupes_across = 0
    seen: dict[str, str] = {}
    for split_name, ids in splits.items():
        id_set = set()
        for img_id in ids:
            if img_id in id_set:
                dupes_within += 1
            id_set.add(img_id)
            if img_id in seen and seen[img_id] != split_name:
                dupes_across += 1
            seen[img_id] = split_name

    if dupes_within > 0 or dupes_across > 0:
        print(f"ERROR: duplicate IDs detected: {dupes_within} within-split, {dupes_across} across-split", file=sys.stderr)
        sys.exit(1)


def validate_mask_values(mask_path: Path) -> set[int]:
    """Return unique mask values. Raises ValueError on invalid pixel values."""
    img = Image.open(mask_path)
    arr = np.array(img)
    unique = set(np.unique(arr).tolist())
    if not (unique.issubset({0, 255}) or unique.issubset({0, 1})):
        raise ValueError(
            f"Mask values {unique} not a subset of {{0,255}} or {{0,1}}"
        )
    return unique


def validate_image_mask_pair(image_path: Path, mask_path: Path) -> None:
    with Image.open(image_path) as im:
        iw, ih = im.size
    with Image.open(mask_path) as mm:
        mw, mh = mm.size
    if (iw, ih) != (mw, mh):
        raise ValueError(
            f"Dimension mismatch: image {iw}x{ih} vs mask {mw}x{mh}"
        )


# ---------------------------------------------------------------------------
# Case mapping
# ---------------------------------------------------------------------------


def build_case_mapping(
    splits: dict[str, list[str]],
    source_dir: Path,
    dataset_root: Path,
    mask_variant: str,
    export_heldout_labels: bool,
) -> list[dict]:
    rows: list[dict] = []
    counter = 0

    for split_name in ["train", "val", "test"]:
        for original_id in splits[split_name]:
            counter += 1
            case_id = f"siim_{counter:06d}"

            if split_name in ("train", "val"):
                nnunet_split = "imagesTr"
                label_out = f"labelsTr/{case_id}.png"
                heldout_out = ""
            else:
                nnunet_split = "imagesTs"
                label_out = ""
                heldout_out = f"heldout_labelsTs/{case_id}.png" if export_heldout_labels else ""

            row = {
                "case_id": case_id,
                "original_image_id": original_id,
                "source_split": split_name,
                "nnunet_split": nnunet_split,
                "image_source_path": f"images/{original_id}.png",
                "mask_source_path": f"{mask_variant}/{original_id}.png",
                "image_output_path": f"{nnunet_split}/{case_id}_0000.png",
                "label_output_path": label_out,
                "heldout_label_output_path": heldout_out,
            }
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Output preparation
# ---------------------------------------------------------------------------


def build_dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def prepare_output_dirs(dataset_root: Path, overwrite: bool, dry_run: bool, export_heldout_labels: bool) -> None:
    if dry_run:
        return

    if dataset_root.exists():
        if not overwrite:
            print(f"ERROR: Output dataset folder already exists: {dataset_root}. Use --overwrite to replace.", file=sys.stderr)
            sys.exit(2)
        shutil.rmtree(dataset_root)

    (dataset_root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labelsTr").mkdir(parents=True, exist_ok=True)
    (dataset_root / "imagesTs").mkdir(parents=True, exist_ok=True)
    if export_heldout_labels:
        (dataset_root / "heldout_labelsTs").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Export operations
# ---------------------------------------------------------------------------


def copy_image_channel(image_source_path: Path, image_output_path: Path) -> None:
    shutil.copy2(image_source_path, image_output_path)


def convert_and_write_label(mask_source_path: Path, label_output_path: Path) -> None:
    img = Image.open(mask_source_path)
    arr = np.array(img)
    unique = set(np.unique(arr).tolist())

    if not (unique.issubset({0, 255}) or unique.issubset({0, 1})):
        raise ValueError(f"Mask {mask_source_path} has invalid values {unique}")

    if 255 in unique:
        arr = (arr > 0).astype(np.uint8)
    elif 1 in unique and 0 in unique:
        arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    Image.fromarray(arr, mode="L").save(label_output_path)


# ---------------------------------------------------------------------------
# dataset.json
# ---------------------------------------------------------------------------


def write_dataset_json(
    dataset_root: Path,
    dataset_id: int,
    dataset_name: str,
    num_training: int,
    no_reader_writer_override: bool,
) -> None:
    content: dict = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "pneumothorax": 1},
        "numTraining": num_training,
        "file_ending": ".png",
        "name": f"Dataset{dataset_id:03d}_{dataset_name}",
        "description": (
            "SIIM-ACR pneumothorax trusted v1 export. "
            "Source train+val are imagesTr/labelsTr; source test is imagesTs. "
            "Labels use original_masks converted from 0/255 to 0/1 uint8. "
            "heldout_labelsTs, if present, is for external evaluation only "
            "and is not official nnU-Net labelsTs."
        ),
    }

    if not no_reader_writer_override:
        content["overwrite_image_reader_writer"] = "NaturalImage2DIO"

    with (dataset_root / "dataset.json").open("w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# case_mapping.csv
# ---------------------------------------------------------------------------


_MAPPING_CSV_FIELDS = [
    "case_id",
    "original_image_id",
    "source_split",
    "nnunet_split",
    "image_source_path",
    "mask_source_path",
    "image_output_path",
    "label_output_path",
    "heldout_label_output_path",
]


def write_case_mapping_csv(dataset_root: Path, mapping_rows: list[dict]) -> None:
    with (dataset_root / "case_mapping.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MAPPING_CSV_FIELDS)
        writer.writeheader()
        for row in mapping_rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# splits_final_source.json
# ---------------------------------------------------------------------------


def write_splits_final_source(dataset_root: Path, mapping_rows: list[dict]) -> None:
    train_cases = [r["case_id"] for r in mapping_rows if r["source_split"] == "train"]
    val_cases = [r["case_id"] for r in mapping_rows if r["source_split"] == "val"]

    fold = {
        "train": train_cases,
        "val": val_cases,
    }

    with (dataset_root / "splits_final_source.json").open("w", encoding="utf-8") as fh:
        json.dump([fold], fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Post-export validation
# ---------------------------------------------------------------------------


def validate_export(
    dataset_root: Path,
    mapping_rows: list[dict],
    export_heldout_labels: bool,
) -> None:
    images_tr = sorted((dataset_root / "imagesTr").glob("*_0000.png"))
    labels_tr = sorted((dataset_root / "labelsTr").glob("*.png"))
    images_ts = sorted((dataset_root / "imagesTs").glob("*_0000.png"))

    if len(images_tr) != len(labels_tr):
        print(f"ERROR: imagesTr count {len(images_tr)} != labelsTr count {len(labels_tr)}", file=sys.stderr)
        sys.exit(1)

    if (dataset_root / "labelsTs").exists():
        print("ERROR: labelsTs/ directory must not exist in this export", file=sys.stderr)
        sys.exit(1)

    # Verify each labelsTr file contains only 0/1
    for lp in labels_tr:
        arr = np.array(Image.open(lp))
        unique = set(np.unique(arr).tolist())
        if not unique.issubset({0, 1}):
            print(f"ERROR: Output label {lp.name} contains values {unique} (expected subset of {{0,1}})", file=sys.stderr)
            sys.exit(1)

    heldout_labels = sorted((dataset_root / "heldout_labelsTs").glob("*.png")) if (dataset_root / "heldout_labelsTs").exists() else []

    if export_heldout_labels and len(heldout_labels) != len(images_ts):
        print(f"ERROR: heldout_labelsTs count {len(heldout_labels)} != imagesTs count {len(images_ts)}", file=sys.stderr)
        sys.exit(1)

    if not export_heldout_labels and heldout_labels:
        print("ERROR: heldout_labelsTs/ exists but --export_heldout_labels was not passed", file=sys.stderr)
        sys.exit(1)

    # Verify every imagesTr entry has matching labelsTr
    img_tr_stems = {p.name.replace("_0000.png", "") for p in images_tr}
    lbl_tr_stems = {p.stem for p in labels_tr}
    if img_tr_stems != lbl_tr_stems:
        diff = img_tr_stems.symmetric_difference(lbl_tr_stems)
        print(f"ERROR: imagesTr/labelsTr case ID mismatch: {len(diff)} cases differ", file=sys.stderr)
        sys.exit(1)

    # Verify imagesTs
    img_ts_stems = {p.name.replace("_0000.png", "") for p in images_ts}
    if export_heldout_labels:
        hld_stems = {p.stem for p in heldout_labels}
        if img_ts_stems != hld_stems:
            diff = img_ts_stems.symmetric_difference(hld_stems)
            print(f"ERROR: imagesTs/heldout_labelsTs case ID mismatch: {len(diff)} cases differ", file=sys.stderr)
            sys.exit(1)

    # Validate case_mapping.csv consistency
    mapping_cases = {r["case_id"] for r in mapping_rows}
    all_exported = img_tr_stems | img_ts_stems
    if mapping_cases != all_exported:
        diff = mapping_cases.symmetric_difference(all_exported)
        print(f"ERROR: case_mapping.csv and exported files mismatch: {len(diff)} cases", file=sys.stderr)
        sys.exit(1)

    # Verify dataset.json exists
    if not (dataset_root / "dataset.json").is_file():
        print("ERROR: dataset.json not found in output", file=sys.stderr)
        sys.exit(1)

    # Verify splits_final_source.json exists
    if not (dataset_root / "splits_final_source.json").is_file():
        print("ERROR: splits_final_source.json not found in output", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(
    status: str,
    dataset_name: str,
    dataset_id: int,
    source_total: int,
    images_tr_count: int,
    labels_tr_count: int,
    images_ts_count: int,
    heldout_labels_ts_count: int,
    fold0_train: int,
    fold0_val: int,
    mask_variant: str,
    output_root: Path,
    dry_run: bool,
) -> None:
    print(f"nnunet_export={status}")
    print(f"dataset=Dataset{dataset_id:03d}_{dataset_name}")
    print(f"source_total={source_total}")
    print(f"imagesTr={images_tr_count}")
    print(f"labelsTr={labels_tr_count}")
    print(f"imagesTs={images_ts_count}")
    print(f"heldout_labelsTs={heldout_labels_ts_count}")
    print(f"fold0_train={fold0_train}")
    print(f"fold0_val={fold0_val}")
    print(f"mask_variant={mask_variant}")
    if dry_run:
        print(f"output_root={output_root} (dry_run)")
    else:
        print(f"output_root={output_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # --- CLI validation ---
    if args.limit and not args.dry_run:
        print("ERROR: --limit is allowed only with --dry_run", file=sys.stderr)
        sys.exit(2)

    if args.mask_variant != "original_masks" and not args.allow_non_original_masks:
        print(
            f"ERROR: mask_variant '{args.mask_variant}' is not 'original_masks'. "
            "Pass --allow_non_original_masks to override.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Resolve paths ---
    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        print(f"ERROR: source_dir does not exist: {source_dir}", file=sys.stderr)
        sys.exit(2)

    nnunet_raw = resolve_nnunet_raw_root(args).resolve()
    dataset_folder = build_dataset_folder_name(args.dataset_id, args.dataset_name)
    dataset_root = nnunet_raw / dataset_folder

    # --- Warnings ---
    warnings: list[str] = []

    # Check nnunetv2 availability
    try:
        import nnunetv2  # noqa: F401
    except ImportError:
        warnings.append("nnunetv2 is not installed (not required for export, but needed for PR-3 preprocessing)")

    # Check NaturalImage2DIO
    if not args.no_reader_writer_override:
        try:
            from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO  # noqa: F401
        except ImportError:
            warnings.append(
                "NaturalImage2DIO cannot be import-checked (nnunetv2 may not be installed). "
                "If PR-3 preprocessing fails, rerun export with --overwrite --no_reader_writer_override."
            )

    if args.no_reader_writer_override:
        warnings.append("--no_reader_writer_override: overwrite_image_reader_writer omitted from dataset.json")

    # --- Load source data ---
    splits = load_splits(source_dir)
    manifest, mask_variants = load_source_manifests(source_dir)

    train_count = len(splits["train"])
    val_count = len(splits["val"])
    test_count = len(splits["test"])
    source_total = train_count + val_count + test_count
    images_tr_expected = train_count + val_count
    images_ts_expected = test_count

    # --- PR-1 summary check ---
    pr1 = load_pr1_summary()
    if pr1 is not None:
        pr1_status = pr1.get("status", "UNKNOWN")
        if pr1_status != "PASS":
            print(f"ERROR: PR-1 audit summary reports status={pr1_status}, expected PASS", file=sys.stderr)
            sys.exit(1)
        raw_warnings = pr1.get("warnings", [])
        for w in raw_warnings:
            if "37 raw DICOM" in str(w):
                warnings.append(f"PR-1 audit: {w}")
    else:
        warnings.append("PR-1 audit summary not found at artifacts/diagnostics/patient_split_audit/")

    # --- Source validation ---
    validate_source_dataset(source_dir, splits, args.mask_variant, manifest, mask_variants)

    # --- Build case mapping ---
    mapping_rows = build_case_mapping(
        splits, source_dir, dataset_root, args.mask_variant, args.export_heldout_labels
    )

    fold0_train = train_count
    fold0_val = val_count
    heldout_count = test_count if args.export_heldout_labels else 0

    # Print warnings
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    # --- Dry run: validate all image/mask pairs ---
    if args.dry_run:
        print("DRY RUN: validating source dataset (no files written) ...")
        rows_to_check = mapping_rows[: args.limit] if args.limit else mapping_rows

        for row in rows_to_check:
            img_path = source_dir / row["image_source_path"]
            msk_path = source_dir / row["mask_source_path"]

            if not img_path.is_file():
                print(f"ERROR: Missing image: {img_path}", file=sys.stderr)
                sys.exit(1)
            if not msk_path.is_file():
                print(f"ERROR: Missing mask: {msk_path}", file=sys.stderr)
                sys.exit(1)

            validate_image_mask_pair(img_path, msk_path)
            try:
                validate_mask_values(msk_path)
            except ValueError as e:
                print(f"ERROR: Invalid mask values in {msk_path}: {e}", file=sys.stderr)
                sys.exit(1)

        status = "PASS"
        if args.limit:
            status = "LIMITED"
        print_summary(
            status=status,
            dataset_name=args.dataset_name,
            dataset_id=args.dataset_id,
            source_total=source_total,
            images_tr_count=images_tr_expected,
            labels_tr_count=images_tr_expected,
            images_ts_count=images_ts_expected,
            heldout_labels_ts_count=heldout_count,
            fold0_train=fold0_train,
            fold0_val=fold0_val,
            mask_variant=args.mask_variant,
            output_root=dataset_root,
            dry_run=True,
        )
        return 0

    # --- Real export ---
    prepare_output_dirs(dataset_root, args.overwrite, dry_run=False, export_heldout_labels=args.export_heldout_labels)

    print(f"Exporting {source_total} cases to {dataset_root} ...")

    for idx, row in enumerate(mapping_rows, 1):
        if idx % 2000 == 0:
            print(f"  ... {idx}/{source_total} ...")

        img_src = source_dir / row["image_source_path"]
        msk_src = source_dir / row["mask_source_path"]

        # Validate on the fly
        validate_image_mask_pair(img_src, msk_src)

        # Copy image
        img_dst = dataset_root / row["image_output_path"]
        copy_image_channel(img_src, img_dst)

        # Convert and write label (train/val only)
        label_out = row["label_output_path"]
        if label_out:
            lbl_dst = dataset_root / label_out
            convert_and_write_label(msk_src, lbl_dst)

        # Heldout labels
        heldout_out = row["heldout_label_output_path"]
        if heldout_out:
            hld_dst = dataset_root / heldout_out
            convert_and_write_label(msk_src, hld_dst)

    # --- Write metadata files ---
    write_dataset_json(
        dataset_root,
        args.dataset_id,
        args.dataset_name,
        images_tr_expected,
        args.no_reader_writer_override,
    )
    write_case_mapping_csv(dataset_root, mapping_rows)
    write_splits_final_source(dataset_root, mapping_rows)

    # --- Post-export validation ---
    validate_export(dataset_root, mapping_rows, args.export_heldout_labels)

    # --- Final summary ---
    status = "FAIL"
    actual_images_tr = len(list((dataset_root / "imagesTr").glob("*_0000.png")))
    actual_labels_tr = len(list((dataset_root / "labelsTr").glob("*.png")))
    actual_images_ts = len(list((dataset_root / "imagesTs").glob("*_0000.png")))
    actual_heldout = len(list((dataset_root / "heldout_labelsTs").glob("*.png"))) if (dataset_root / "heldout_labelsTs").exists() else 0

    if (
        actual_images_tr == images_tr_expected
        and actual_labels_tr == images_tr_expected
        and actual_images_ts == images_ts_expected
        and (not args.export_heldout_labels or actual_heldout == images_ts_expected)
    ):
        status = "PASS"

    print_summary(
        status=status,
        dataset_name=args.dataset_name,
        dataset_id=args.dataset_id,
        source_total=source_total,
        images_tr_count=actual_images_tr,
        labels_tr_count=actual_labels_tr,
        images_ts_count=actual_images_ts,
        heldout_labels_ts_count=actual_heldout,
        fold0_train=fold0_train,
        fold0_val=fold0_val,
        mask_variant=args.mask_variant,
        output_root=dataset_root,
        dry_run=False,
    )

    if status != "PASS":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
