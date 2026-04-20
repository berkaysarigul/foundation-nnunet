"""Helpers for authoritative training run artifacts and provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import numpy as np

from src.data.dataset_manifest import compute_split_fingerprint


DEFAULT_CODE_FINGERPRINT_PATTERNS = (
    "configs/**/*.yaml",
    "src/**/*.py",
    "requirements.txt",
)

HISTORY_CSV_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_dice_mean",
    "val_dice_pos_mean",
    "val_iou_mean",
)

LEGACY_HISTORY_COLUMN_ALIASES = {
    "val_dice": "val_dice_mean",
    "val_dice_pos": "val_dice_pos_mean",
    "val_iou": "val_iou_mean",
}

EVALUATION_CSV_COLUMNS = (
    "image_id",
    "split",
    "model_type",
    "checkpoint_path",
    "eval_mask_variant",
    "selection_metric",
    "selected_threshold",
    "selected_postprocess",
    "positive",
    "dice",
    "iou",
    "precision",
    "recall",
    "f1",
)

SPLIT_LEVEL_CSV_COLUMNS = (
    "study_id",
    "split_instance_id",
    "split_seed",
    "model_name",
    "model_type",
    "run_id",
    "run_dir",
    "dataset_fingerprint",
    "base_split_fingerprint",
    "study_split_fingerprint",
    "run_split_fingerprint",
    "selection_metric",
    "selected_threshold",
    "selected_postprocess",
    "selection_state_path",
    "checkpoint_path",
    "train_mask_variant",
    "eval_mask_variant",
    "test_dice_mean",
    "test_dice_pos_mean",
    "test_iou_mean",
    "test_iou_pos_mean",
)

PAIRED_DELTA_CSV_COLUMNS = (
    "comparison_name",
    "metric_name",
    "reference_model",
    "candidate_model",
    "split_instance_id",
    "split_seed",
    "dataset_fingerprint",
    "base_split_fingerprint",
    "study_split_fingerprint",
    "reference_run_id",
    "candidate_run_id",
    "reference_run_dir",
    "candidate_run_dir",
    "reference_value",
    "candidate_value",
    "delta",
)


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    metadata_dir: Path
    metrics_dir: Path
    checkpoints_dir: Path
    selection_dir: Path
    reports_dir: Path
    qualitative_dir: Path
    qualitative_validation_dir: Path
    qualitative_test_dir: Path

    @property
    def run_metadata_path(self) -> Path:
        return self.metadata_dir / "run_metadata.yaml"

    @property
    def config_snapshot_path(self) -> Path:
        return self.metadata_dir / "config_snapshot.yaml"

    @property
    def history_path(self) -> Path:
        return self.metrics_dir / "history.csv"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / "best_checkpoint.pth"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / "last_checkpoint.pth"

    @property
    def best_checkpoint_metadata_path(self) -> Path:
        return self.checkpoints_dir / "best_checkpoint_metadata.yaml"

    @property
    def selection_state_path(self) -> Path:
        return self.selection_dir / "selection_state.yaml"

    @property
    def test_metrics_path(self) -> Path:
        return self.reports_dir / "test_metrics.csv"

    @property
    def test_summary_path(self) -> Path:
        return self.reports_dir / "test_summary.yaml"

    @property
    def qualitative_validation_manifest_path(self) -> Path:
        return self.qualitative_validation_dir / "manifest.yaml"

    @property
    def qualitative_test_manifest_path(self) -> Path:
        return self.qualitative_test_dir / "manifest.yaml"


@dataclass(frozen=True)
class RepeatedSplitStudyArtifacts:
    study_id: str
    study_dir: Path
    metadata_dir: Path
    aggregations_dir: Path
    comparisons_dir: Path
    summary_dir: Path

    @property
    def split_manifest_path(self) -> Path:
        return self.metadata_dir / "split_manifest.yaml"

    @property
    def split_level_table_path(self) -> Path:
        return self.aggregations_dir / "split_level_metrics.csv"

    @property
    def final_summary_path(self) -> Path:
        return self.summary_dir / "final_summary.yaml"

    def paired_delta_table_path(self, comparison_name: str) -> Path:
        safe_name = comparison_name.strip().replace(" ", "_")
        if not safe_name:
            raise ValueError("comparison_name must not be empty.")
        return self.comparisons_dir / f"{safe_name}_paired_deltas.csv"


def canonicalize_workspace_path(path_like: str | Path, repo_root: Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def _canonicalize_split_ids(split_name: str, image_ids: list[str]) -> list[str]:
    if not image_ids:
        raise ValueError(f"{split_name} ids must not be empty.")

    normalized_ids = [str(image_id) for image_id in image_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError(f"{split_name} ids must be unique within a split instance.")
    return sorted(normalized_ids)


def build_repeated_split_manifest(
    *,
    study_id: str,
    dataset_root: str | Path,
    repo_root: Path,
    split_instances: list[dict[str, Any]],
    split_policy: str = "repeated_stratified_train_val_test",
    selection_metric: str = "val_dice_pos_mean",
    primary_test_metric: str = "test_positive_only_dice_mean",
) -> dict[str, Any]:
    if not study_id.strip():
        raise ValueError("study_id must not be empty.")
    if not split_instances:
        raise ValueError("split_instances must contain at least one split instance.")

    dataset_manifest = load_dataset_manifest(dataset_root, repo_root=repo_root)
    normalized_instances: list[dict[str, Any]] = []
    seen_instance_ids: set[str] = set()

    for raw_instance in split_instances:
        split_instance_id = str(raw_instance["split_instance_id"]).strip()
        if not split_instance_id:
            raise ValueError("split_instance_id must not be empty.")
        if split_instance_id in seen_instance_ids:
            raise ValueError(f"Duplicate split_instance_id: {split_instance_id!r}")
        seen_instance_ids.add(split_instance_id)

        train_ids = _canonicalize_split_ids("train", list(raw_instance["train_ids"]))
        val_ids = _canonicalize_split_ids("val", list(raw_instance["val_ids"]))
        test_ids = _canonicalize_split_ids("test", list(raw_instance["test_ids"]))

        overlaps = {
            "train/val": sorted(set(train_ids) & set(val_ids)),
            "train/test": sorted(set(train_ids) & set(test_ids)),
            "val/test": sorted(set(val_ids) & set(test_ids)),
        }
        overlapping_groups = {name: ids for name, ids in overlaps.items() if ids}
        if overlapping_groups:
            raise ValueError(
                "Repeated split instances must keep train/val/test disjoint; "
                f"{split_instance_id!r} has overlaps: {overlapping_groups}"
            )

        split_map = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }
        normalized_instances.append(
            {
                "split_instance_id": split_instance_id,
                "split_seed": int(raw_instance["split_seed"]),
                "counts": {
                    "train": len(train_ids),
                    "val": len(val_ids),
                    "test": len(test_ids),
                },
                "split_fingerprint": compute_split_fingerprint(split_map),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            }
        )

    normalized_instances.sort(key=lambda item: item["split_instance_id"])
    return {
        "study_id": study_id,
        "schema_version": 1,
        "dataset_root": canonicalize_workspace_path(dataset_root, repo_root),
        "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
        "base_split_fingerprint": dataset_manifest["fingerprints"]["splits"],
        "split_policy": split_policy,
        "selection_metric": selection_metric,
        "primary_test_metric": primary_test_metric,
        "statistical_unit": "split_instance",
        "paired_comparison_unit": "shared_split_instance",
        "split_count": len(normalized_instances),
        "split_instances": normalized_instances,
    }


def _load_yaml_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in {path}, got {type(payload).__name__}.")
    return payload


def _split_instance_lookup(split_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    instances = split_manifest.get("split_instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("split_manifest must contain a non-empty split_instances list.")

    lookup: dict[str, dict[str, Any]] = {}
    for instance in instances:
        if not isinstance(instance, dict):
            raise ValueError("split_manifest split_instances entries must be mappings.")
        split_instance_id = str(instance["split_instance_id"])
        if split_instance_id in lookup:
            raise ValueError(f"split_manifest contains duplicate split_instance_id {split_instance_id!r}.")
        lookup[split_instance_id] = instance
    return lookup


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(model_type: str, *, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{model_type}"


def prepare_run_artifacts(
    model_type: str,
    *,
    run_dir: str | Path | None,
    run_root: Path,
    now: datetime | None = None,
) -> RunArtifacts:
    resolved_run_dir = Path(run_dir) if run_dir is not None else run_root / make_run_id(model_type, now=now)
    if not resolved_run_dir.is_absolute():
        resolved_run_dir = (run_root.parent.parent / resolved_run_dir).resolve()

    run_id = resolved_run_dir.name
    metadata_dir = resolved_run_dir / "metadata"
    metrics_dir = resolved_run_dir / "metrics"
    checkpoints_dir = resolved_run_dir / "checkpoints"
    selection_dir = resolved_run_dir / "selection"
    reports_dir = resolved_run_dir / "reports"
    qualitative_dir = resolved_run_dir / "qualitative"
    qualitative_validation_dir = qualitative_dir / "validation_samples"
    qualitative_test_dir = qualitative_dir / "test_samples"

    for directory in (
        metadata_dir,
        metrics_dir,
        checkpoints_dir,
        selection_dir,
        reports_dir,
        qualitative_validation_dir,
        qualitative_test_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return RunArtifacts(
        run_id=run_id,
        run_dir=resolved_run_dir,
        metadata_dir=metadata_dir,
        metrics_dir=metrics_dir,
        checkpoints_dir=checkpoints_dir,
        selection_dir=selection_dir,
        reports_dir=reports_dir,
        qualitative_dir=qualitative_dir,
        qualitative_validation_dir=qualitative_validation_dir,
        qualitative_test_dir=qualitative_test_dir,
    )


def prepare_repeated_split_study_artifacts(
    study_id: str,
    *,
    study_root: Path,
) -> RepeatedSplitStudyArtifacts:
    if not study_id.strip():
        raise ValueError("study_id must not be empty.")

    resolved_study_dir = study_root / study_id
    metadata_dir = resolved_study_dir / "metadata"
    aggregations_dir = resolved_study_dir / "aggregations"
    comparisons_dir = resolved_study_dir / "comparisons"
    summary_dir = resolved_study_dir / "summary"

    for directory in (metadata_dir, aggregations_dir, comparisons_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return RepeatedSplitStudyArtifacts(
        study_id=study_id,
        study_dir=resolved_study_dir,
        metadata_dir=metadata_dir,
        aggregations_dir=aggregations_dir,
        comparisons_dir=comparisons_dir,
        summary_dir=summary_dir,
    )


def compute_config_hash(cfg: dict[str, Any]) -> str:
    payload = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_code_fingerprint_scope(
    repo_root: Path,
    *,
    patterns: tuple[str, ...] = DEFAULT_CODE_FINGERPRINT_PATTERNS,
) -> list[Path]:
    scope: set[Path] = set()
    for pattern in patterns:
        scope.update(path for path in repo_root.glob(pattern) if path.is_file())
    return sorted(scope)


def compute_code_fingerprint(scope_paths: list[Path], *, repo_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(scope_paths):
        relative_path = path.resolve().relative_to(repo_root.resolve()).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_git_revision(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    revision = result.stdout.strip()
    return revision or None


def resolve_code_provenance(repo_root: Path) -> dict[str, Any]:
    scope_paths = resolve_code_fingerprint_scope(repo_root)
    return {
        "code_revision": resolve_git_revision(repo_root),
        "code_fingerprint": compute_code_fingerprint(scope_paths, repo_root=repo_root),
        "code_fingerprint_scope": [
            path.resolve().relative_to(repo_root.resolve()).as_posix()
            for path in scope_paths
        ],
    }


def load_dataset_manifest(dataset_root: str | Path, *, repo_root: Path) -> dict[str, Any]:
    dataset_manifest_path = Path(canonicalize_workspace_path(dataset_root, repo_root)) / "dataset_manifest.json"
    with dataset_manifest_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_initial_checkpoint_reference(cfg: dict[str, Any], *, repo_root: Path) -> str:
    model_type = cfg["model"]["type"]
    if model_type == "pretrained_resnet34_unet":
        return "torchvision://resnet34_imagenet1k_v1"
    if model_type == "hybrid":
        return canonicalize_workspace_path(cfg["foundation_x"]["checkpoint_path"], repo_root)
    return "random_init"


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_config_snapshot(path: Path, cfg: dict[str, Any]) -> None:
    write_yaml(path, cfg)


def canonicalize_history(history: dict[str, list[float]]) -> dict[str, list[float]]:
    canonical_history: dict[str, list[float]] = {}
    extra_history: dict[str, list[float]] = {}

    for raw_key, raw_values in history.items():
        values = list(raw_values)
        canonical_key = LEGACY_HISTORY_COLUMN_ALIASES.get(raw_key, raw_key)
        if canonical_key in HISTORY_CSV_COLUMNS:
            if canonical_key in canonical_history:
                raise ValueError(
                    f"History contains duplicate data for canonical column {canonical_key!r}."
                )
            canonical_history[canonical_key] = values
        else:
            extra_history[raw_key] = values

    missing_columns = [
        column for column in HISTORY_CSV_COLUMNS if column != "epoch" and column not in canonical_history
    ]
    if missing_columns:
        raise ValueError(
            "History is missing required canonical columns: "
            f"{missing_columns}"
        )

    all_series = list(canonical_history.values()) + list(extra_history.values())
    series_lengths = {len(values) for values in all_series}
    if len(series_lengths) > 1:
        raise ValueError("History columns must all have the same length.")

    row_count = series_lengths.pop() if series_lengths else 0
    expected_epoch = list(range(1, row_count + 1))
    epoch_values = canonical_history.get("epoch")
    if epoch_values is None:
        canonical_history["epoch"] = expected_epoch
    elif list(epoch_values) != expected_epoch:
        raise ValueError(
            "History epoch column must be contiguous and 1-indexed to match authoritative history.csv."
        )

    ordered_history = {
        column: list(canonical_history[column])
        for column in HISTORY_CSV_COLUMNS
    }
    for column in sorted(extra_history):
        ordered_history[column] = list(extra_history[column])
    return ordered_history


def write_history_csv(path: Path, history: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(canonicalize_history(history)).to_csv(path, index=False)


def build_evaluation_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=EVALUATION_CSV_COLUMNS)

    df = pd.DataFrame(records)
    missing_columns = [
        column for column in EVALUATION_CSV_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Evaluation records are missing required canonical columns: "
            f"{missing_columns}"
        )

    extra_columns = sorted(
        column for column in df.columns if column not in EVALUATION_CSV_COLUMNS
    )
    return df.loc[:, list(EVALUATION_CSV_COLUMNS) + extra_columns]


def write_evaluation_csv(path: Path, records: list[dict[str, Any]]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_evaluation_dataframe(records)
    df.to_csv(path, index=False)
    return df


def build_split_level_records_from_authoritative_runs(
    *,
    split_manifest: dict[str, Any],
    model_runs: list[dict[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    if not model_runs:
        raise ValueError("model_runs must contain at least one authoritative run reference.")

    instance_lookup = _split_instance_lookup(split_manifest)
    study_id = str(split_manifest["study_id"])
    dataset_fingerprint = str(split_manifest["dataset_fingerprint"])
    base_split_fingerprint = str(split_manifest["base_split_fingerprint"])
    study_dataset_root = canonicalize_workspace_path(str(split_manifest["dataset_root"]), repo_root)
    required_selection_metric = str(split_manifest["selection_metric"])

    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for model_run in model_runs:
        split_instance_id = str(model_run["split_instance_id"])
        model_name = str(model_run["model_name"])
        if split_instance_id not in instance_lookup:
            raise ValueError(
                f"model_runs references unknown split_instance_id {split_instance_id!r}."
            )
        dedupe_key = (split_instance_id, model_name)
        if dedupe_key in seen_keys:
            raise ValueError(
                "Each model may contribute at most one authoritative run per split instance; "
                f"duplicate key={dedupe_key!r}."
            )
        seen_keys.add(dedupe_key)

        run_dir = Path(model_run["run_dir"])
        if not run_dir.is_absolute():
            run_dir = (repo_root / run_dir).resolve()
        metadata = _load_yaml_payload(run_dir / "metadata" / "run_metadata.yaml")
        summary = _load_yaml_payload(run_dir / "reports" / "test_summary.yaml")

        if summary.get("split") != "test":
            raise ValueError(f"{run_dir} test_summary.yaml must describe split='test'.")
        if canonicalize_workspace_path(str(summary["dataset_root"]), repo_root) != study_dataset_root:
            raise ValueError(
                f"{run_dir} dataset_root does not match split study dataset_root."
            )
        if str(summary["selection_metric"]) != required_selection_metric:
            raise ValueError(
                f"{run_dir} selection_metric={summary['selection_metric']!r} does not match "
                f"study selection_metric={required_selection_metric!r}."
            )

        split_instance = instance_lookup[split_instance_id]
        records.append(
            {
                "study_id": study_id,
                "split_instance_id": split_instance_id,
                "split_seed": int(split_instance["split_seed"]),
                "model_name": model_name,
                "model_type": str(summary["model_type"]),
                "run_id": str(metadata["run_id"]),
                "run_dir": str(run_dir),
                "dataset_fingerprint": dataset_fingerprint,
                "base_split_fingerprint": base_split_fingerprint,
                "study_split_fingerprint": str(split_instance["split_fingerprint"]),
                "run_split_fingerprint": str(metadata["split_fingerprint"]),
                "selection_metric": str(summary["selection_metric"]),
                "selected_threshold": float(summary["selected_threshold"]),
                "selected_postprocess": str(summary["selected_postprocess"]),
                "selection_state_path": str(summary["selection_state_path"]),
                "checkpoint_path": str(summary["checkpoint_path"]),
                "train_mask_variant": str(summary["train_mask_variant"]),
                "eval_mask_variant": str(summary["eval_mask_variant"]),
                "test_dice_mean": float(summary["subsets"]["all"]["dice"]["mean"]),
                "test_dice_pos_mean": float(summary["subsets"]["positive"]["dice"]["mean"]),
                "test_iou_mean": float(summary["subsets"]["all"]["iou"]["mean"]),
                "test_iou_pos_mean": float(summary["subsets"]["positive"]["iou"]["mean"]),
            }
        )

    return sorted(records, key=lambda row: (row["split_instance_id"], row["model_name"]))


def build_split_level_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=SPLIT_LEVEL_CSV_COLUMNS)

    df = pd.DataFrame(records)
    missing_columns = [
        column for column in SPLIT_LEVEL_CSV_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Split-level records are missing required canonical columns: "
            f"{missing_columns}"
        )

    extra_columns = sorted(
        column for column in df.columns if column not in SPLIT_LEVEL_CSV_COLUMNS
    )
    return df.loc[:, list(SPLIT_LEVEL_CSV_COLUMNS) + extra_columns]


def write_split_level_csv(path: Path, records: list[dict[str, Any]]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_split_level_dataframe(records)
    df.to_csv(path, index=False)
    return df


def build_paired_delta_records(
    *,
    split_level_records: list[dict[str, Any]],
    comparison_name: str,
    reference_model: str,
    candidate_model: str,
    metric_name: str = "test_dice_pos_mean",
) -> list[dict[str, Any]]:
    if not comparison_name.strip():
        raise ValueError("comparison_name must not be empty.")
    if not split_level_records:
        raise ValueError("split_level_records must not be empty.")

    split_level_df = build_split_level_dataframe(split_level_records)
    if metric_name not in split_level_df.columns:
        raise ValueError(f"Unknown metric_name {metric_name!r} for paired delta records.")

    ref_df = split_level_df[split_level_df["model_name"] == reference_model]
    cand_df = split_level_df[split_level_df["model_name"] == candidate_model]
    shared_ids = sorted(set(ref_df["split_instance_id"]) & set(cand_df["split_instance_id"]))
    if not shared_ids:
        raise ValueError("Paired delta construction requires at least one shared split instance.")

    records: list[dict[str, Any]] = []
    for split_instance_id in shared_ids:
        ref_rows = ref_df[ref_df["split_instance_id"] == split_instance_id]
        cand_rows = cand_df[cand_df["split_instance_id"] == split_instance_id]
        if len(ref_rows) != 1 or len(cand_rows) != 1:
            raise ValueError(
                "Paired delta construction requires exactly one row per model per shared split instance."
            )

        ref_row = ref_rows.iloc[0]
        cand_row = cand_rows.iloc[0]
        records.append(
            {
                "comparison_name": comparison_name,
                "metric_name": metric_name,
                "reference_model": reference_model,
                "candidate_model": candidate_model,
                "split_instance_id": split_instance_id,
                "split_seed": int(ref_row["split_seed"]),
                "dataset_fingerprint": ref_row["dataset_fingerprint"],
                "base_split_fingerprint": ref_row["base_split_fingerprint"],
                "study_split_fingerprint": ref_row["study_split_fingerprint"],
                "reference_run_id": ref_row["run_id"],
                "candidate_run_id": cand_row["run_id"],
                "reference_run_dir": ref_row["run_dir"],
                "candidate_run_dir": cand_row["run_dir"],
                "reference_value": float(ref_row[metric_name]),
                "candidate_value": float(cand_row[metric_name]),
                "delta": float(cand_row[metric_name]) - float(ref_row[metric_name]),
            }
        )

    return records


def build_paired_delta_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=PAIRED_DELTA_CSV_COLUMNS)

    df = pd.DataFrame(records)
    missing_columns = [
        column for column in PAIRED_DELTA_CSV_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Paired-delta records are missing required canonical columns: "
            f"{missing_columns}"
        )

    extra_columns = sorted(
        column for column in df.columns if column not in PAIRED_DELTA_CSV_COLUMNS
    )
    return df.loc[:, list(PAIRED_DELTA_CSV_COLUMNS) + extra_columns]


def write_paired_delta_csv(path: Path, records: list[dict[str, Any]]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_paired_delta_dataframe(records)
    df.to_csv(path, index=False)
    return df


def _bootstrap_percentile_ci(
    values: list[float],
    *,
    ci_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float]:
    if not values:
        raise ValueError("Bootstrap CI requires at least one value.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    if not 0 < ci_level < 100:
        raise ValueError("ci_level must be between 0 and 100.")

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Bootstrap CI values must all be finite.")

    rng = np.random.default_rng(bootstrap_seed)
    sampled_indices = rng.integers(0, len(array), size=(bootstrap_samples, len(array)))
    sampled_means = array[sampled_indices].mean(axis=1)
    alpha = (100.0 - ci_level) / 2.0
    return {
        "mean": float(array.mean()),
        "ci_lower": float(np.percentile(sampled_means, alpha)),
        "ci_upper": float(np.percentile(sampled_means, 100.0 - alpha)),
    }


def build_final_repeated_split_summary_payload(
    *,
    split_manifest: dict[str, Any],
    split_level_records: list[dict[str, Any]],
    paired_delta_records: list[dict[str, Any]],
    primary_metric: str = "test_dice_pos_mean",
    ci_level: float = 95.0,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    split_level_df = build_split_level_dataframe(split_level_records)
    if primary_metric not in split_level_df.columns:
        raise ValueError(f"Unknown primary_metric {primary_metric!r} for final repeated-split summary.")

    study_id = str(split_manifest["study_id"])
    dataset_fingerprint = str(split_manifest["dataset_fingerprint"])
    base_split_fingerprint = str(split_manifest["base_split_fingerprint"])
    split_policy = str(split_manifest["split_policy"])
    selection_metric = str(split_manifest["selection_metric"])
    manifest_instance_ids = {
        str(instance["split_instance_id"])
        for instance in split_manifest["split_instances"]
    }

    if set(split_level_df["study_id"]) != {study_id}:
        raise ValueError("split_level_records must all match split_manifest study_id.")
    if set(split_level_df["dataset_fingerprint"]) != {dataset_fingerprint}:
        raise ValueError("split_level_records must all match split_manifest dataset_fingerprint.")
    if set(split_level_df["base_split_fingerprint"]) != {base_split_fingerprint}:
        raise ValueError("split_level_records must all match split_manifest base_split_fingerprint.")
    if set(split_level_df["selection_metric"]) != {selection_metric}:
        raise ValueError("split_level_records must all match split_manifest selection_metric.")

    model_summaries: list[dict[str, Any]] = []
    for model_name in sorted(split_level_df["model_name"].unique().tolist()):
        model_df = split_level_df[split_level_df["model_name"] == model_name]
        split_instance_ids = sorted(model_df["split_instance_id"].astype(str).tolist())
        if len(split_instance_ids) != len(set(split_instance_ids)):
            raise ValueError(
                f"split_level_records contain duplicate split instances for model_name={model_name!r}."
            )
        if not set(split_instance_ids).issubset(manifest_instance_ids):
            raise ValueError(
                f"split_level_records reference split instances outside the split manifest for model_name={model_name!r}."
            )

        ci_payload = _bootstrap_percentile_ci(
            model_df[primary_metric].astype(float).tolist(),
            ci_level=ci_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        model_summaries.append(
            {
                "model_name": model_name,
                "model_type": str(model_df.iloc[0]["model_type"]),
                "metric_name": primary_metric,
                "mean": ci_payload["mean"],
                "ci_lower": ci_payload["ci_lower"],
                "ci_upper": ci_payload["ci_upper"],
                "contributing_split_count": int(len(split_instance_ids)),
                "contributing_split_instance_ids": split_instance_ids,
            }
        )

    paired_delta_df = build_paired_delta_dataframe(paired_delta_records)
    paired_comparisons: list[dict[str, Any]] = []
    if not paired_delta_df.empty:
        grouped = paired_delta_df.groupby("comparison_name", sort=True)
        for comparison_name, comparison_df in grouped:
            metric_names = sorted(comparison_df["metric_name"].astype(str).unique().tolist())
            if len(metric_names) != 1:
                raise ValueError(
                    f"Paired comparison {comparison_name!r} must contain exactly one metric_name."
                )
            split_instance_ids = sorted(comparison_df["split_instance_id"].astype(str).tolist())
            if len(split_instance_ids) != len(set(split_instance_ids)):
                raise ValueError(
                    f"Paired comparison {comparison_name!r} contains duplicate split instances."
                )
            if not set(split_instance_ids).issubset(manifest_instance_ids):
                raise ValueError(
                    f"Paired comparison {comparison_name!r} references split instances outside the split manifest."
                )

            ci_payload = _bootstrap_percentile_ci(
                comparison_df["delta"].astype(float).tolist(),
                ci_level=ci_level,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            paired_comparisons.append(
                {
                    "comparison_name": str(comparison_name),
                    "metric_name": metric_names[0],
                    "reference_model": str(comparison_df.iloc[0]["reference_model"]),
                    "candidate_model": str(comparison_df.iloc[0]["candidate_model"]),
                    "mean_delta": ci_payload["mean"],
                    "ci_lower": ci_payload["ci_lower"],
                    "ci_upper": ci_payload["ci_upper"],
                    "contributing_split_count": int(len(split_instance_ids)),
                    "contributing_split_instance_ids": split_instance_ids,
                }
            )

    return {
        "study_id": study_id,
        "schema_version": 1,
        "dataset_fingerprint": dataset_fingerprint,
        "base_split_fingerprint": base_split_fingerprint,
        "split_policy": split_policy,
        "selection_metric": selection_metric,
        "primary_metric": primary_metric,
        "ci_level": float(ci_level),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "model_summaries": model_summaries,
        "paired_comparisons": paired_comparisons,
    }


def write_final_repeated_split_summary(
    path: Path,
    *,
    split_manifest: dict[str, Any],
    split_level_records: list[dict[str, Any]],
    paired_delta_records: list[dict[str, Any]],
    primary_metric: str = "test_dice_pos_mean",
    ci_level: float = 95.0,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    payload = build_final_repeated_split_summary_payload(
        split_manifest=split_manifest,
        split_level_records=split_level_records,
        paired_delta_records=paired_delta_records,
        primary_metric=primary_metric,
        ci_level=ci_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    write_yaml(path, payload)
    return payload


def build_run_metadata(
    *,
    cfg: dict[str, Any],
    config_path: str | Path,
    repo_root: Path,
    run_id: str,
    resume_checkpoint_path: str | Path | None,
    started_at: str | None = None,
) -> dict[str, Any]:
    dataset_manifest = load_dataset_manifest(cfg["data"]["processed_dir"], repo_root=repo_root)
    code_provenance = resolve_code_provenance(repo_root)

    return {
        "run_id": run_id,
        "started_at": started_at or utc_timestamp(),
        "model_type": cfg["model"]["type"],
        "config_path": canonicalize_workspace_path(config_path, repo_root),
        "config_hash": compute_config_hash(cfg),
        "code_revision": code_provenance["code_revision"],
        "code_fingerprint": code_provenance["code_fingerprint"],
        "code_fingerprint_scope": code_provenance["code_fingerprint_scope"],
        "dataset_root": canonicalize_workspace_path(cfg["data"]["processed_dir"], repo_root),
        "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
        "split_fingerprint": dataset_manifest["fingerprints"]["splits"],
        "train_mask_variant": cfg["data"].get("train_mask_variant", "dilated_masks"),
        "eval_mask_variant": cfg["data"].get("eval_mask_variant", "original_masks"),
        "initial_checkpoint_path": resolve_initial_checkpoint_reference(cfg, repo_root=repo_root),
        "resume_checkpoint_path": (
            canonicalize_workspace_path(resume_checkpoint_path, repo_root)
            if resume_checkpoint_path is not None
            else None
        ),
        "input_size": int(cfg["data"]["input_size"]),
        "seed": int(cfg["seed"]),
        "selection_metric": cfg["selection"]["metric"],
        "selected_threshold": None,
        "selected_postprocess": cfg["selection"]["postprocess"],
    }


def build_best_checkpoint_metadata(
    *,
    checkpoint_path: str | Path,
    cfg: dict[str, Any],
    repo_root: Path,
    epoch: int,
    best_metric_value: float,
    training_components: dict[str, str],
) -> dict[str, Any]:
    return {
        "checkpoint_path": canonicalize_workspace_path(checkpoint_path, repo_root),
        "model_type": cfg["model"]["type"],
        "epoch": int(epoch),
        "selection_metric": cfg["selection"]["metric"],
        "best_metric_value": float(best_metric_value),
        "train_mask_variant": cfg["data"].get("train_mask_variant", "dilated_masks"),
        "eval_mask_variant": cfg["data"].get("eval_mask_variant", "original_masks"),
        "input_size": int(cfg["data"]["input_size"]),
        "training_components": training_components,
    }
