"""Run the fixed single-split hybrid sanity check through train/select/test."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.evaluate import evaluate, select_threshold_and_save
from src.training.run_artifacts import prepare_run_artifacts
from src.training.trainer import train


EXPECTED_PROTOCOL = {
    ("model", "type"): "hybrid",
    ("foundation_x", "checkpoint_path"): "checkpoints/foundation_x.pth",
    ("foundation_x", "frozen"): False,
    ("data", "processed_dir"): "data/processed/pneumothorax_trusted_v1",
    ("data", "input_size"): 512,
    ("data", "train_mask_variant"): "dilated_masks",
    ("data", "eval_mask_variant"): "original_masks",
    ("training", "batch_size"): 4,
    ("training", "epochs"): 150,
    ("training", "learning_rate"): 0.0001,
    ("training", "optimizer"): "AdamW",
    ("training", "weight_decay"): 0.01,
    ("training", "scheduler"): "ReduceLROnPlateau",
    ("training", "early_stopping_patience"): 30,
    ("loss", "type"): "dice_focal",
    ("selection", "metric"): "val_dice_pos_mean",
    ("selection", "postprocess"): "none",
    ("branch_input_views", "dataset_emitted_view", "space"): "grayscale",
    ("branch_input_views", "cnn_branch", "transform"): "identity",
    ("branch_input_views", "foundation_x_branch", "rgb_adapter"): "repeat_grayscale_to_rgb",
    ("branch_input_views", "foundation_x_branch", "normalization", "type"): "imagenet_mean_std",
    ("seed",): 42,
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_nested_value(cfg: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = cfg
    for key in path:
        value = value[key]
    return value


def validate_hybrid_sanity_protocol(cfg: dict[str, Any]) -> None:
    mismatches = []
    for path, expected in EXPECTED_PROTOCOL.items():
        actual = get_nested_value(cfg, path)
        if actual != expected:
            mismatches.append((".".join(path), expected, actual))

    normalization = cfg["branch_input_views"]["foundation_x_branch"]["normalization"]
    expected_mean = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    if normalization["mean"] != expected_mean:
        mismatches.append(
            ("branch_input_views.foundation_x_branch.normalization.mean", expected_mean, normalization["mean"])
        )
    if normalization["std"] != expected_std:
        mismatches.append(
            ("branch_input_views.foundation_x_branch.normalization.std", expected_std, normalization["std"])
        )

    if mismatches:
        mismatch_lines = ", ".join(
            f"{field} expected {expected!r} but got {actual!r}"
            for field, expected, actual in mismatches
        )
        raise ValueError(
            "Hybrid single-split sanity runner refuses off-protocol config: "
            f"{mismatch_lines}"
        )


def prepare_hybrid_run_dir(
    *,
    config_path: str | Path,
    run_dir: str | Path | None,
) -> tuple[dict[str, Any], object]:
    cfg = load_config(config_path)
    validate_hybrid_sanity_protocol(cfg)
    repo_root = Path(__file__).resolve().parents[1]
    run_artifacts = prepare_run_artifacts(
        cfg["model"]["type"],
        run_dir=run_dir,
        run_root=repo_root / "artifacts" / "runs",
    )
    return cfg, run_artifacts


def require_existing_best_checkpoint(best_checkpoint_path: Path, stage: str) -> None:
    if not best_checkpoint_path.exists():
        raise FileNotFoundError(
            f"{stage} stage requires an existing best checkpoint at {best_checkpoint_path}. "
            "Run the train stage first or point --run_dir at the hybrid sanity run directory."
        )


def run_stage(
    *,
    config_path: str | Path,
    run_dir: str | Path | None,
    stage: str,
) -> Path:
    if stage != "all" and run_dir is None:
        raise ValueError(
            f"stage={stage!r} requires --run_dir so the hybrid sanity artifacts can be reused."
        )

    cfg, run_artifacts = prepare_hybrid_run_dir(
        config_path=config_path,
        run_dir=run_dir,
    )
    resolved_run_dir = run_artifacts.run_dir
    model_type = cfg["model"]["type"]

    if stage in {"all", "train"}:
        train(
            cfg,
            config_path=config_path,
            run_dir=resolved_run_dir,
        )

    if stage in {"all", "select", "select_test"}:
        select_stage_name = "select_test" if stage == "select_test" else "select"
        require_existing_best_checkpoint(
            run_artifacts.best_checkpoint_path,
            stage=select_stage_name,
        )
        select_threshold_and_save(
            cfg,
            checkpoint_path=str(run_artifacts.best_checkpoint_path),
            model_type=model_type,
            selection_state_path=run_artifacts.selection_state_path,
        )

    if stage in {"all", "test", "select_test"}:
        test_stage_name = "select_test" if stage == "select_test" else "test"
        require_existing_best_checkpoint(
            run_artifacts.best_checkpoint_path,
            stage=test_stage_name,
        )
        evaluate(
            cfg,
            checkpoint_path=str(run_artifacts.best_checkpoint_path),
            model_type=model_type,
            selection_state_path=run_artifacts.selection_state_path,
        )

    return resolved_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed single-split hybrid sanity protocol."
    )
    parser.add_argument(
        "--config",
        default="configs/hybrid_single_split_sanity.yaml",
        help="Protocol-locked single-split hybrid sanity config.",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help=(
            "Hybrid sanity run directory. Omit for stage=all to create a new run under "
            "artifacts/runs/, or reuse an existing run_dir to resume."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("all", "train", "select", "test", "select_test"),
        default="all",
        help=(
            "Which part of the hybrid sanity pipeline to execute. "
            "'select_test' reuses an existing best checkpoint and runs "
            "validation threshold selection followed by test evaluation."
        ),
    )
    args = parser.parse_args()

    run_dir = run_stage(
        config_path=args.config,
        run_dir=args.run_dir,
        stage=args.stage,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
