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


def canonicalize_workspace_path(path_like: str | Path, repo_root: Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


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
