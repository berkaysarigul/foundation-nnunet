"""Mask-variant contract for the processed SIIM dataset."""

from __future__ import annotations

from pathlib import Path

MASK_VARIANTS = ("original_masks", "dilated_masks")
DEFAULT_TRAIN_MASK_VARIANT = "dilated_masks"
DEFAULT_EVAL_MASK_VARIANT = "original_masks"


def validate_mask_variant(mask_variant: str) -> str:
    """Validate and return a supported processed-mask variant name."""
    if mask_variant not in MASK_VARIANTS:
        raise ValueError(
            f"Unknown mask variant '{mask_variant}'. Expected one of {list(MASK_VARIANTS)}."
        )
    return mask_variant


def resolve_mask_variant(mask_variant: str | None = None, *, purpose: str | None = None) -> str:
    """Resolve a mask variant explicitly or from the intended usage purpose."""
    if mask_variant is not None:
        return validate_mask_variant(mask_variant)

    if purpose == "train":
        return DEFAULT_TRAIN_MASK_VARIANT
    if purpose in {"val", "test", "eval"}:
        return DEFAULT_EVAL_MASK_VARIANT

    raise ValueError("mask_variant is required when purpose is not one of train/val/test/eval")


def resolve_mask_dir(
    data_dir: str | Path,
    mask_variant: str | None = None,
    *,
    purpose: str | None = None,
) -> Path:
    """Return the processed-mask directory for the requested variant."""
    variant = resolve_mask_variant(mask_variant, purpose=purpose)
    return Path(data_dir) / variant


def build_mask_variant_manifest() -> dict:
    """Return the canonical processed-dataset mask-variant manifest."""
    return {
        "available_variants": list(MASK_VARIANTS),
        "default_train_mask_variant": DEFAULT_TRAIN_MASK_VARIANT,
        "default_eval_mask_variant": DEFAULT_EVAL_MASK_VARIANT,
        "final_reporting_mask_variant": "original_masks",
        "dilation_policy": {
            "type": "separate_mask_variant",
            "dilated_variant_name": "dilated_masks",
            "original_variant_name": "original_masks",
            "kernel_shape": "ellipse",
            "kernel_size": [15, 15],
            "applied_before_resize": True,
        },
        "scientific_implications": {
            "original_masks": (
                "Use for official SIIM-style validation and final reporting because this "
                "variant preserves the undecorated decoded annotations."
            ),
            "dilated_masks": (
                "May be used as a training target for sparse-mask optimization, but runs "
                "must declare this explicitly and final claims must still report "
                "original-mask performance."
            ),
        },
    }
