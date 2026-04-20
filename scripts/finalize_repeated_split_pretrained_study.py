"""Finalize a repeated-split study from authoritative per-split run inventories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.run_artifacts import (  # noqa: E402
    build_paired_delta_records,
    build_split_level_records_from_authoritative_runs,
    prepare_repeated_split_study_artifacts,
    write_final_repeated_split_summary,
    write_paired_delta_csv,
    write_split_level_csv,
)


SPLIT_MANIFEST_FILENAME = "split_manifest.yaml"
RUN_INVENTORY_FILENAME_SUFFIX = "_run_inventory.yaml"


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in {path}, got {type(payload).__name__}.")
    return payload


def validate_split_manifest_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.name != SPLIT_MANIFEST_FILENAME:
        raise ValueError(
            f"study_manifest_path must end with {SPLIT_MANIFEST_FILENAME!r}; got {path.name!r}."
        )
    if path.parent.name != "metadata":
        raise ValueError("study_manifest_path must live inside a metadata directory.")
    return path.resolve()


def validate_run_inventory_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.name.endswith(RUN_INVENTORY_FILENAME_SUFFIX):
        raise ValueError(
            "run_inventory_path must end with "
            f"{RUN_INVENTORY_FILENAME_SUFFIX!r}; got {path.name!r}."
        )
    if path.parent.name != "metadata":
        raise ValueError("run_inventory_path must live inside a metadata directory.")
    return path.resolve()


def resolve_study_artifacts_from_manifest_path(study_manifest_path: str | Path):
    validated_manifest_path = validate_split_manifest_path(study_manifest_path)
    study_dir = validated_manifest_path.parent.parent
    return prepare_repeated_split_study_artifacts(
        study_dir.name,
        study_root=study_dir.parent,
    )


def build_model_runs_from_inventory_payloads(
    inventory_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not inventory_payloads:
        raise ValueError("inventory_payloads must not be empty.")

    model_runs: list[dict[str, Any]] = []
    for inventory_payload in inventory_payloads:
        model_name = str(inventory_payload["model_name"]).strip()
        if not model_name:
            raise ValueError("run inventory model_name must not be empty.")

        entries = inventory_payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("run inventory must contain a non-empty entries list.")

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("run inventory entries must be mappings.")
            model_runs.append(
                {
                    "split_instance_id": str(entry["split_instance_id"]),
                    "model_name": model_name,
                    "run_dir": str(Path(entry["run_dir"]).resolve()),
                }
            )

    return model_runs


def parse_comparison_spec(raw_spec: str) -> dict[str, str]:
    parts = [part.strip() for part in raw_spec.split(":")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "comparison specs must use "
            "'comparison_name:reference_model:candidate_model'."
        )

    comparison_name, reference_model, candidate_model = parts
    return {
        "comparison_name": comparison_name,
        "reference_model": reference_model,
        "candidate_model": candidate_model,
    }


def finalize_repeated_split_study(
    *,
    run_inventory_paths: list[str | Path],
    comparison_specs: list[str] | None = None,
    primary_metric: str = "test_dice_pos_mean",
    ci_level: float = 95.0,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    if not run_inventory_paths:
        raise ValueError("run_inventory_paths must contain at least one inventory path.")

    inventory_payloads: list[dict[str, Any]] = []
    manifest_paths: set[Path] = set()
    study_ids: set[str] = set()
    for raw_path in run_inventory_paths:
        inventory_path = validate_run_inventory_path(raw_path)
        inventory_payload = load_yaml_mapping(inventory_path)
        inventory_payloads.append(inventory_payload)
        manifest_paths.add(validate_split_manifest_path(inventory_payload["study_manifest_path"]))
        study_ids.add(str(inventory_payload["study_id"]))

    if len(manifest_paths) != 1:
        raise ValueError("All run inventories must reference the same study_manifest_path.")
    if len(study_ids) != 1:
        raise ValueError("All run inventories must reference the same study_id.")

    manifest_path = manifest_paths.pop()
    split_manifest = load_yaml_mapping(manifest_path)
    if str(split_manifest["study_id"]) != next(iter(study_ids)):
        raise ValueError("Run inventory study_id does not match split manifest study_id.")

    study_artifacts = resolve_study_artifacts_from_manifest_path(manifest_path)
    model_runs = build_model_runs_from_inventory_payloads(inventory_payloads)
    split_level_records = build_split_level_records_from_authoritative_runs(
        split_manifest=split_manifest,
        model_runs=model_runs,
        repo_root=REPO_ROOT,
    )
    split_level_df = write_split_level_csv(
        study_artifacts.split_level_table_path,
        split_level_records,
    )

    comparison_specs = comparison_specs or []
    paired_delta_paths: list[str] = []
    all_paired_delta_records: list[dict[str, Any]] = []
    for raw_spec in comparison_specs:
        comparison_spec = parse_comparison_spec(raw_spec)
        paired_delta_records = build_paired_delta_records(
            split_level_records=split_level_records,
            comparison_name=comparison_spec["comparison_name"],
            reference_model=comparison_spec["reference_model"],
            candidate_model=comparison_spec["candidate_model"],
            metric_name=primary_metric,
        )
        paired_delta_path = study_artifacts.paired_delta_table_path(
            comparison_spec["comparison_name"]
        )
        write_paired_delta_csv(paired_delta_path, paired_delta_records)
        paired_delta_paths.append(str(paired_delta_path))
        all_paired_delta_records.extend(paired_delta_records)

    summary_payload = write_final_repeated_split_summary(
        study_artifacts.final_summary_path,
        split_manifest=split_manifest,
        split_level_records=split_level_records,
        paired_delta_records=all_paired_delta_records,
        primary_metric=primary_metric,
        ci_level=ci_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "study_id": study_artifacts.study_id,
        "split_manifest_path": str(manifest_path),
        "split_level_table_path": str(study_artifacts.split_level_table_path),
        "paired_delta_table_paths": paired_delta_paths,
        "final_summary_path": str(study_artifacts.final_summary_path),
        "split_level_row_count": int(len(split_level_df)),
        "comparison_count": int(len(comparison_specs)),
        "summary_model_count": int(len(summary_payload["model_summaries"])),
        "summary_comparison_count": int(len(summary_payload["paired_comparisons"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a repeated-split study from authoritative per-split run inventories "
            "into split-level tables, optional paired-delta tables, and final summary output."
        )
    )
    parser.add_argument(
        "--run_inventory",
        action="append",
        required=True,
        help=(
            "Path to a *_run_inventory.yaml file produced by a repeated-split study runner. "
            "Pass once per model inventory."
        ),
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help=(
            "Optional comparison spec formatted as "
            "'comparison_name:reference_model:candidate_model'."
        ),
    )
    parser.add_argument(
        "--primary_metric",
        default="test_dice_pos_mean",
        help="Split-level metric column used for paired comparisons and final summary.",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=95.0,
        help="Two-sided percentile bootstrap confidence level.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for final summary generation.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=42,
        help="Seed for bootstrap reproducibility.",
    )
    args = parser.parse_args()

    payload = finalize_repeated_split_study(
        run_inventory_paths=args.run_inventory,
        comparison_specs=args.comparison,
        primary_metric=args.primary_metric,
        ci_level=args.ci_level,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(payload["final_summary_path"])


if __name__ == "__main__":
    main()
