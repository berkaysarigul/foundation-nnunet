"""Golden decode tests for the accepted local SIIM RLE contract."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.rle_contract import (
    NEGATIVE_RLE_TOKENS,
    decode_grid_mask,
    decode_runs,
    resolve_rle_mode,
)


FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "siim_rle_golden_cases.json"
CSV_PATH = REPO_ROOT / "data" / "raw" / "SIIM-ACR" / "train-rle.csv"


def _load_fixture() -> dict:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        encoded_key = next(key for key in reader.fieldnames if key.strip() == "EncodedPixels")
        image_key = next(key for key in reader.fieldnames if key.strip() == "ImageId")
        rows = []
        for row in reader:
            rows.append(
                {
                    "ImageId": row[image_key].strip(),
                    "EncodedPixels": row[encoded_key].strip(),
                }
            )
        return rows


class TestRLEContractGoldenCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()
        cls.csv_rows = _load_csv_rows()
        cls.all_rles = [row["EncodedPixels"] for row in cls.csv_rows]

    def test_local_corpus_resolves_unambiguously_to_cumulative_gap_pairs(self) -> None:
        resolved_mode, evidence = resolve_rle_mode(self.all_rles, requested_mode="auto")
        self.assertEqual(resolved_mode, "cumulative_gap_pairs")
        self.assertEqual(evidence.positive_rows, 3286)
        self.assertEqual(evidence.negative_rows, 8296)
        self.assertEqual(evidence.valid_absolute_pairs, 0)
        self.assertEqual(evidence.valid_cumulative_gap_pairs, evidence.positive_rows)

    def test_incompatible_absolute_pairs_mode_fails_fast_on_local_corpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible with the corpus"):
            resolve_rle_mode(self.all_rles, requested_mode="absolute_pairs")

    def test_synthetic_golden_cases_decode_exactly(self) -> None:
        for case in self.fixture["synthetic_cases"]:
            with self.subTest(case=case["name"]):
                runs_1based = [
                    [start + 1, length]
                    for start, length in decode_runs(case["rle"], rle_mode="cumulative_gap_pairs")
                ]
                grid = decode_grid_mask(
                    case["rle"],
                    height=case["height"],
                    width=case["width"],
                    rle_mode="cumulative_gap_pairs",
                )
                self.assertEqual(runs_1based, case["expected_runs_1based"])
                self.assertEqual(grid, case["expected_grid"])

    def test_negative_tokens_remain_empty(self) -> None:
        for token in NEGATIVE_RLE_TOKENS:
            with self.subTest(token=token):
                grid = decode_grid_mask(token, height=2, width=2, rle_mode="cumulative_gap_pairs")
                self.assertEqual(grid, [[0, 0], [0, 0]])

    def test_curated_local_csv_cases_match_expected_previews(self) -> None:
        grouped: dict[str, list[str]] = {}
        for row in self.csv_rows:
            grouped.setdefault(row["ImageId"], []).append(row["EncodedPixels"])

        for case in self.fixture["local_csv_cases"]:
            with self.subTest(case=case["name"]):
                rows = grouped[case["image_id"]]
                if "expected_group_row_count" in case:
                    self.assertEqual(len(rows), case["expected_group_row_count"])

                rle = rows[case["row_index_within_image"]]
                self.assertEqual(rle.split()[:16], case["expected_token_prefix"])

                preview = [
                    [start + 1, length]
                    for start, length in decode_runs(
                        rle,
                        rle_mode="cumulative_gap_pairs",
                    )[:8]
                ]
                self.assertEqual(preview, case["expected_run_preview_1based"])


if __name__ == "__main__":
    unittest.main()
