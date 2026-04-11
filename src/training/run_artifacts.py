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


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    metadata_dir: Path
    metrics_dir: Path
    checkpoints_dir: Path
    selection_dir: Path
    qualitative_dir: Path
    qualitative_validation_dir: Path

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
    qualitative_dir = resolved_run_dir / "qualitative"
    qualitative_validation_dir = qualitative_dir / "validation_samples"

    for directory in (
        metadata_dir,
        metrics_dir,
        checkpoints_dir,
        selection_dir,
        qualitative_validation_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return RunArtifacts(
        run_id=run_id,
        run_dir=resolved_run_dir,
        metadata_dir=metadata_dir,
        metrics_dir=metrics_dir,
        checkpoints_dir=checkpoints_dir,
        selection_dir=selection_dir,
        qualitative_dir=qualitative_dir,
        qualitative_validation_dir=qualitative_validation_dir,
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


def write_history_csv(path: Path, history: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


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
