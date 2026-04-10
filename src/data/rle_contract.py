"""Accepted RLE parsing contract for the local SIIM annotation bundle.

This module is intentionally stdlib-only so the contract can be validated
without requiring the full training/preprocessing dependency stack.
"""

from __future__ import annotations

from dataclasses import dataclass

NEGATIVE_RLE_TOKENS = {"", "-1"}
SUPPORTED_RLE_MODES = ("absolute_pairs", "cumulative_gap_pairs")


@dataclass(frozen=True)
class RLEModeEvidence:
    positive_rows: int
    negative_rows: int
    valid_absolute_pairs: int
    valid_cumulative_gap_pairs: int


def normalize_rle_string(rle_string: object) -> str:
    """Return a stripped string representation of an annotation cell."""
    if rle_string is None:
        return ""

    text = str(rle_string).strip()
    return "" if text.lower() == "nan" else text


def parse_rle_pairs(rle_string: object) -> tuple[list[int], list[int]]:
    """Parse alternating value/length tokens from a non-empty RLE string."""
    text = normalize_rle_string(rle_string)
    if text in NEGATIVE_RLE_TOKENS:
        return [], []

    tokens = text.split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"RLE token count must be even, got {len(tokens)} tokens")

    values = [int(token) for token in tokens[0::2]]
    lengths = [int(token) for token in tokens[1::2]]

    if any(value < 0 for value in values):
        raise ValueError("RLE values must be non-negative")
    if any(length <= 0 for length in lengths):
        raise ValueError("RLE lengths must be positive")

    return values, lengths


def decode_runs(
    rle_string: object,
    *,
    rle_mode: str = "cumulative_gap_pairs",
) -> list[tuple[int, int]]:
    """Decode an RLE string into 0-based flat start/length runs.

    Supported modes:
    - ``absolute_pairs``: standard start/length pairs with 1-based starts.
    - ``cumulative_gap_pairs``: the local ``train-rle.csv`` contract where:
      - the first value is the absolute 1-based start of the first run
      - each subsequent value is the zero-count gap after the previous run
    """
    if rle_mode not in SUPPORTED_RLE_MODES:
        raise ValueError(f"Unsupported rle_mode '{rle_mode}'")

    values, lengths = parse_rle_pairs(rle_string)
    if not values:
        return []

    if rle_mode == "absolute_pairs":
        starts = [value - 1 for value in values]
    else:
        starts = [values[0] - 1]
        cursor = starts[0] + lengths[0]
        for gap, length in zip(values[1:], lengths[1:]):
            start = cursor + gap
            starts.append(start)
            cursor = start + length

    return list(zip(starts, lengths))


def decode_flat_mask(
    rle_string: object,
    *,
    mask_size: int,
    rle_mode: str = "cumulative_gap_pairs",
    foreground_value: int = 1,
) -> list[int]:
    """Decode an RLE string into a flat binary mask stored as a Python list."""
    if mask_size <= 0:
        raise ValueError("mask_size must be positive")

    flat_mask = [0] * mask_size
    for start, length in decode_runs(rle_string, rle_mode=rle_mode):
        end = start + length
        if start < 0 or end > mask_size:
            raise ValueError(
                f"Decoded run [{start}, {end}) is out of bounds for mask size {mask_size}"
            )
        for index in range(start, end):
            flat_mask[index] = foreground_value
    return flat_mask


def flat_mask_to_grid_fortran(flat_mask: list[int], *, height: int, width: int) -> list[list[int]]:
    """Reshape a flat mask into a 2D grid using Fortran/column-major order."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if len(flat_mask) != height * width:
        raise ValueError(
            f"Flat mask length {len(flat_mask)} does not match height*width={height * width}"
        )

    return [[flat_mask[row + (column * height)] for column in range(width)] for row in range(height)]


def decode_grid_mask(
    rle_string: object,
    *,
    height: int,
    width: int,
    rle_mode: str = "cumulative_gap_pairs",
    foreground_value: int = 1,
) -> list[list[int]]:
    """Decode an RLE string into a 2D grid using the accepted Fortran layout."""
    flat_mask = decode_flat_mask(
        rle_string,
        mask_size=height * width,
        rle_mode=rle_mode,
        foreground_value=foreground_value,
    )
    return flat_mask_to_grid_fortran(flat_mask, height=height, width=width)


def runs_are_strictly_non_overlapping(runs: list[tuple[int, int]]) -> bool:
    """Return True when decoded flat runs are strictly increasing."""
    previous_end = -1
    for start, length in runs:
        if start < 0 or length <= 0:
            return False
        if start <= previous_end:
            return False
        previous_end = start + length - 1
    return True


def row_is_valid_for_mode(rle_string: object, *, rle_mode: str) -> bool:
    """Return True when a row decodes cleanly and non-overlapping in the mode."""
    runs = decode_runs(rle_string, rle_mode=rle_mode)
    if not runs:
        return normalize_rle_string(rle_string) in NEGATIVE_RLE_TOKENS
    return runs_are_strictly_non_overlapping(runs)


def inspect_rle_mode_evidence(rle_strings: list[object]) -> RLEModeEvidence:
    """Summarize how many positive rows are internally consistent per mode."""
    positive_rows = 0
    negative_rows = 0
    valid_counts = {mode: 0 for mode in SUPPORTED_RLE_MODES}

    for rle_string in rle_strings:
        text = normalize_rle_string(rle_string)
        if text in NEGATIVE_RLE_TOKENS:
            negative_rows += 1
            continue

        positive_rows += 1
        for mode in SUPPORTED_RLE_MODES:
            if row_is_valid_for_mode(text, rle_mode=mode):
                valid_counts[mode] += 1

    return RLEModeEvidence(
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        valid_absolute_pairs=valid_counts["absolute_pairs"],
        valid_cumulative_gap_pairs=valid_counts["cumulative_gap_pairs"],
    )


def resolve_rle_mode(
    rle_strings: list[object],
    *,
    requested_mode: str = "auto",
) -> tuple[str, RLEModeEvidence]:
    """Resolve the accepted RLE mode from corpus-level evidence."""
    if requested_mode not in ("auto", *SUPPORTED_RLE_MODES):
        raise ValueError(f"Unsupported requested_mode '{requested_mode}'")

    evidence = inspect_rle_mode_evidence(rle_strings)
    if evidence.positive_rows == 0:
        if requested_mode == "auto":
            raise ValueError("Cannot auto-resolve RLE mode without positive rows")
        return requested_mode, evidence

    mode_hits = {
        "absolute_pairs": evidence.valid_absolute_pairs,
        "cumulative_gap_pairs": evidence.valid_cumulative_gap_pairs,
    }

    if requested_mode == "auto":
        matching_modes = [
            mode for mode, hit_count in mode_hits.items() if hit_count == evidence.positive_rows
        ]
        if len(matching_modes) != 1:
            raise ValueError(
                "RLE mode auto-detection is ambiguous: "
                f"positive_rows={evidence.positive_rows}, "
                f"absolute_pairs={evidence.valid_absolute_pairs}, "
                f"cumulative_gap_pairs={evidence.valid_cumulative_gap_pairs}"
            )
        return matching_modes[0], evidence

    if mode_hits[requested_mode] != evidence.positive_rows:
        raise ValueError(
            f"Requested rle_mode '{requested_mode}' is incompatible with the corpus: "
            f"positive_rows={evidence.positive_rows}, "
            f"absolute_pairs={evidence.valid_absolute_pairs}, "
            f"cumulative_gap_pairs={evidence.valid_cumulative_gap_pairs}"
        )

    return requested_mode, evidence
