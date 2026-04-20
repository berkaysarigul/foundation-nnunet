"""Run the authoritative pretrained baseline across a repeated-split study manifest."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.run_artifacts import prepare_repeated_split_study_artifacts, write_yaml  # noqa: E402


SPLIT_MANIFEST_FILENAME = "split_manifest.yaml"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "pretrained_resnet34_authoritative.yaml"
SPLIT_OVERRIDES_DIRNAME = "split_overrides"
CONFIG_OVERRIDES_DIRNAME = "config_overrides"
RUN_INVENTORY_FILENAME = "pretrained_resnet34_run_inventory.yaml"


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


def resolve_study_artifacts_from_manifest_path(
    study_manifest_path: str | Path,
):
    validated_manifest_path = validate_split_manifest_path(study_manifest_path)
    study_dir = validated_manifest_path.parent.parent
    return prepare_repeated_split_study_artifacts(
        study_dir.name,
        study_root=study_dir.parent,
    )


def build_split_override_payload(split_instance: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "train": list(split_instance["train_ids"]),
        "val": list(split_instance["val_ids"]),
        "test": list(split_instance["test_ids"]),
    }


def build_cfg_with_split_override(
    base_cfg: dict[str, Any],
    *,
    split_override_path: str | Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("data", {})
    cfg["data"]["splits_path"] = str(Path(split_override_path).resolve())
    return cfg


def build_split_run_dir(
    *,
    study_id: str,
    split_instance_id: str,
    model_name: str = "pretrained_resnet34_unet",
) -> Path:
    return (REPO_ROOT / "artifacts" / "runs" / f"{study_id}__{split_instance_id}__{model_name}").resolve()


def build_runner_command(
    *,
    config_path: Path,
    run_dir: Path,
    stage: str,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_authoritative_pretrained_baseline.py"),
        "--config",
        str(config_path),
        "--run_dir",
        str(run_dir),
        "--stage",
        stage,
    ]


def build_run_inventory_payload(
    *,
    study_id: str,
    study_manifest_path: Path,
    base_config_path: Path,
    stage: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "study_id": study_id,
        "schema_version": 1,
        "runner": str((REPO_ROOT / "scripts" / "run_repeated_split_pretrained_study.py").resolve()),
        "study_manifest_path": str(study_manifest_path.resolve()),
        "base_config_path": str(base_config_path.resolve()),
        "stage": stage,
        "model_name": "pretrained_resnet34_unet",
        "entries": entries,
    }


def run_repeated_split_study(
    *,
    study_manifest_path: str | Path,
    base_config_path: str | Path = DEFAULT_BASE_CONFIG,
    stage: str = "all",
) -> Path:
    manifest_path = validate_split_manifest_path(study_manifest_path)
    study_artifacts = resolve_study_artifacts_from_manifest_path(manifest_path)
    split_manifest = load_yaml_mapping(manifest_path)
    study_id = str(split_manifest["study_id"])
    base_config = load_yaml_mapping(base_config_path)
    base_config_path = Path(base_config_path).resolve()

    split_overrides_dir = study_artifacts.metadata_dir / SPLIT_OVERRIDES_DIRNAME
    config_overrides_dir = study_artifacts.metadata_dir / CONFIG_OVERRIDES_DIRNAME
    split_overrides_dir.mkdir(parents=True, exist_ok=True)
    config_overrides_dir.mkdir(parents=True, exist_ok=True)

    inventory_entries: list[dict[str, Any]] = []
    inventory_path = study_artifacts.metadata_dir / RUN_INVENTORY_FILENAME

    for split_instance in split_manifest["split_instances"]:
        split_instance_id = str(split_instance["split_instance_id"])
        split_seed = int(split_instance["split_seed"])

        split_override_path = split_overrides_dir / f"{split_instance_id}.json"
        with split_override_path.open("w", encoding="utf-8") as handle:
            json.dump(build_split_override_payload(split_instance), handle, indent=2)

        cfg_with_override = build_cfg_with_split_override(
            base_config,
            split_override_path=split_override_path,
        )
        config_override_path = config_overrides_dir / f"{split_instance_id}.yaml"
        write_yaml(config_override_path, cfg_with_override)

        run_dir = build_split_run_dir(
            study_id=study_id,
            split_instance_id=split_instance_id,
        )
        command = build_runner_command(
            config_path=config_override_path,
            run_dir=run_dir,
            stage=stage,
        )
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
        )

        inventory_entries.append(
            {
                "split_instance_id": split_instance_id,
                "split_seed": split_seed,
                "split_override_path": str(split_override_path.resolve()),
                "config_override_path": str(config_override_path.resolve()),
                "run_dir": str(run_dir),
                "stage": stage,
            }
        )
        write_yaml(
            inventory_path,
            build_run_inventory_payload(
                study_id=study_id,
                study_manifest_path=manifest_path,
                base_config_path=base_config_path,
                stage=stage,
                entries=inventory_entries,
            ),
        )

    return inventory_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the authoritative pretrained baseline across every split instance "
            "listed in a repeated-split study manifest."
        )
    )
    parser.add_argument(
        "--study_manifest",
        required=True,
        help="Path to artifacts/repeated_splits/<study_id>/metadata/split_manifest.yaml",
    )
    parser.add_argument(
        "--base_config",
        default=str(DEFAULT_BASE_CONFIG),
        help="Protocol-locked authoritative pretrained baseline config.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "train", "select", "test", "select_test"),
        default="all",
        help="Stage forwarded to scripts/run_authoritative_pretrained_baseline.py for each split instance.",
    )
    args = parser.parse_args()

    inventory_path = run_repeated_split_study(
        study_manifest_path=args.study_manifest,
        base_config_path=args.base_config,
        stage=args.stage,
    )
    print(inventory_path)


if __name__ == "__main__":
    main()
