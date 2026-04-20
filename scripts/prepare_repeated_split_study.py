"""Create a canonical repeated-split study package for the trusted dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.repeated_splits import (  # noqa: E402
    build_repeated_split_instances,
    load_processed_dataset_binary_labels,
)
from src.training.run_artifacts import (  # noqa: E402
    build_repeated_split_manifest,
    prepare_repeated_split_study_artifacts,
    write_yaml,
)


DEFAULT_STUDY_ROOT = REPO_ROOT / "artifacts" / "repeated_splits"


def parse_split_seeds(raw_value: str) -> list[int]:
    if not raw_value.strip():
        raise ValueError("split_seeds must not be empty.")

    split_seeds: list[int] = []
    for token in raw_value.split(","):
        normalized = token.strip()
        if not normalized:
            raise ValueError("split_seeds must not contain empty entries.")
        split_seeds.append(int(normalized))
    return split_seeds


def prepare_repeated_split_study_manifest(
    *,
    study_id: str,
    dataset_dir: str | Path,
    split_seeds: list[int],
    study_root: str | Path | None = None,
) -> Path:
    dataset_root = Path(dataset_dir)
    resolved_study_root = Path(study_root) if study_root is not None else DEFAULT_STUDY_ROOT
    if not resolved_study_root.is_absolute():
        resolved_study_root = (REPO_ROOT / resolved_study_root).resolve()

    study_artifacts = prepare_repeated_split_study_artifacts(
        study_id,
        study_root=resolved_study_root,
    )
    image_ids, positive_ids = load_processed_dataset_binary_labels(dataset_root)
    split_instances = build_repeated_split_instances(
        image_ids,
        positive_ids,
        split_seeds=split_seeds,
    )
    split_manifest = build_repeated_split_manifest(
        study_id=study_id,
        dataset_root=dataset_root,
        repo_root=REPO_ROOT,
        split_instances=split_instances,
    )
    write_yaml(study_artifacts.split_manifest_path, split_manifest)
    return study_artifacts.split_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create metadata/split_manifest.yaml for a repeated-split study over the trusted "
            "processed dataset."
        )
    )
    parser.add_argument(
        "--study_id",
        required=True,
        help="Stable study identifier used under artifacts/repeated_splits/<study_id>/.",
    )
    parser.add_argument(
        "--dataset_dir",
        default="data/processed/pneumothorax_trusted_v1",
        help="Trusted processed dataset root.",
    )
    parser.add_argument(
        "--split_seeds",
        required=True,
        help="Comma-separated split seeds, for example: 42,43,44,45,46",
    )
    parser.add_argument(
        "--study_root",
        default=str(DEFAULT_STUDY_ROOT),
        help="Root directory for repeated-split study packages.",
    )
    args = parser.parse_args()

    split_manifest_path = prepare_repeated_split_study_manifest(
        study_id=args.study_id,
        dataset_dir=args.dataset_dir,
        split_seeds=parse_split_seeds(args.split_seeds),
        study_root=args.study_root,
    )
    print(split_manifest_path)


if __name__ == "__main__":
    main()
