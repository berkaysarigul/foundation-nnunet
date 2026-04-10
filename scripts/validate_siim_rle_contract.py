"""Small audit helper for validating the local SIIM RLE contract."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.rle_contract import NEGATIVE_RLE_TOKENS, decode_runs, resolve_rle_mode


def _load_rles(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        encoded_key = next(key for key in reader.fieldnames if key.strip() == "EncodedPixels")
        return [row[encoded_key] for row in reader]


def _iter_curated_examples(csv_path: Path, limit: int) -> list[tuple[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        encoded_key = next(key for key in reader.fieldnames if key.strip() == "EncodedPixels")
        examples: list[tuple[str, str]] = []
        for row in reader:
            rle = row[encoded_key].strip()
            if rle in NEGATIVE_RLE_TOKENS:
                continue
            examples.append((row["ImageId"].strip(), rle))
            if len(examples) >= limit:
                break
        return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local SIIM RLE contract")
    parser.add_argument(
        "--csv_path",
        default="data/raw/SIIM-ACR/train-rle.csv",
        help="Path to the authoritative local SIIM annotation CSV",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="How many curated positive examples to print",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rle_strings = _load_rles(csv_path)
    resolved_mode, evidence = resolve_rle_mode(rle_strings, requested_mode="auto")

    print(f"csv_path={csv_path}")
    print(f"resolved_mode={resolved_mode}")
    print(f"positive_rows={evidence.positive_rows}")
    print(f"negative_rows={evidence.negative_rows}")
    print(f"valid_absolute_pairs={evidence.valid_absolute_pairs}")
    print(f"valid_cumulative_gap_pairs={evidence.valid_cumulative_gap_pairs}")

    for image_id, rle in _iter_curated_examples(csv_path, args.examples):
        runs = decode_runs(rle, rle_mode=resolved_mode)
        preview = [(start + 1, length) for start, length in runs[:8]]
        print(f"example_image_id={image_id}")
        print(f"raw_tokens={rle.split()[:16]}")
        print(f"decoded_start_length_preview={preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
