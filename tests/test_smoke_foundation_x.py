"""
Tests for scripts/smoke_foundation_x.py

All tests use a stubbed FoundationXBackbone — the real 2.62 GiB checkpoint is
NOT required. Tests decorated @pytest.mark.slow touch the real checkpoint and
are skipped by default.
"""

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Stub timm if not installed so backbone.py can be imported without it
if "timm" not in sys.modules:
    sys.modules["timm"] = types.SimpleNamespace(
        create_model=lambda *a, **kw: None,
    )

import smoke_foundation_x as sfx


# ─── Dummy backbone ───────────────────────────────────────────────────────────


class DummySmokeBackbone(torch.nn.Module):
    """Returns random tensors with correct shapes; never touches a checkpoint file."""

    def __init__(self, checkpoint_path: str, frozen: bool = True, img_size: int = 256):
        super().__init__()
        self.frozen = frozen
        self.img_size = img_size
        self.backbone = torch.nn.Identity()  # provides .state_dict() → {}

    def to(self, device):  # noqa: D102
        return self

    def eval(self):  # noqa: D102
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]
        return [
            torch.randn(B, 128, H // 4,  W // 4),
            torch.randn(B, 256, H // 8,  W // 8),
            torch.randn(B, 512, H // 16, W // 16),
            torch.randn(B, 1024, H // 32, W // 32),
        ]


class DummySmokeBackboneNaN(DummySmokeBackbone):
    """Like DummySmokeBackbone but injects NaN into stage 2."""

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = super().forward(x)
        feats[2] = torch.full_like(feats[2], float("nan"))
        return feats


# ─── Shared mock checkpoint metadata ─────────────────────────────────────────

_MOCK_CKPT_META = {
    "checkpoint_path": "checkpoints/foundation_x.pth",
    "checkpoint_sha256": "deadbeefcafe0000",
    "checkpoint_size_bytes": 42,
    "top_level_keys": ["model"],
    "state_dict_len": 1,
    "backbone_prefix": "backbone.0.",
    "prefix_match_count": 1,
    "sample_prefix_keys": ["backbone.0.patch_embed.proj.weight"],
    "_remapped_keys": {"patch_embed.proj.weight"},
}


# ─── Test fixtures ────────────────────────────────────────────────────────────


def _make_png(path: Path, size: int = 256, positive_label: bool | None = None) -> None:
    """Write a minimal grayscale PNG.  If positive_label is set, write binary 0/1 mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if positive_label is None:
        arr = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    else:
        arr = np.zeros((size, size), dtype=np.uint8)
        if positive_label:
            arr[4:12, 4:12] = 1
    Image.fromarray(arr, mode="L").save(str(path))


def _build_test_dir(tmpdir: Path, num_priority: int = 7, num_extra: int = 10) -> tuple[Path, Path, Path]:
    """
    Create images/, labels/, and a fake checkpoint file in tmpdir.
    Returns (images_dir, labels_dir, checkpoint_path).
    """
    images_dir = tmpdir / "images"
    labels_dir = tmpdir / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    priority_ids = sfx.PRIORITY_FN_IDS + sfx.PRIORITY_FP_IDS
    for cid in priority_ids:
        _make_png(images_dir / f"{cid}_0000.png")
        _make_png(labels_dir / f"{cid}.png", positive_label=(cid in sfx.PRIORITY_FN_IDS))

    for i in range(num_extra):
        cid = f"siim_{900000 + i:06d}"
        _make_png(images_dir / f"{cid}_0000.png")
        _make_png(labels_dir / f"{cid}.png", positive_label=(i % 2 == 0))

    ckpt = tmpdir / "checkpoint.pth"
    ckpt.write_bytes(b"\x00" * 16)  # fake non-empty file
    return images_dir, labels_dir, ckpt


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestCliParsing(unittest.TestCase):
    def test_default_values_match_plan(self):
        args = sfx.parse_args([])
        self.assertEqual(args.img_size, 512)
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.num_cases, 12)
        self.assertFalse(args.strict)
        self.assertFalse(args.no_visuals)
        self.assertFalse(args.save_raw_features)
        self.assertFalse(args.dry_run)
        self.assertIsNone(args.case_list)

    def test_all_flags_accepted(self):
        args = sfx.parse_args([
            "--input_dir", "/tmp/images",
            "--labels_dir", "/tmp/labels",
            "--checkpoint", "/tmp/ckpt.pth",
            "--output_dir", "/tmp/out",
            "--img_size", "256",
            "--device", "cpu",
            "--num_cases", "6",
            "--strict",
            "--no_visuals",
            "--save_raw_features",
            "--dry_run",
        ])
        self.assertEqual(args.img_size, 256)
        self.assertTrue(args.strict)
        self.assertTrue(args.dry_run)

    def test_invalid_img_size_rejected(self):
        with self.assertRaises(SystemExit):
            sfx.parse_args(["--img_size", "128"])


class TestCaseSelection(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.images_dir, self.labels_dir, self.ckpt = _build_test_dir(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def test_selects_exactly_num_cases(self):
        cases = sfx.select_cases(self.images_dir, self.labels_dir, 12, None)
        self.assertEqual(len(cases), 12)

    def test_priority_ids_appear_first(self):
        cases = sfx.select_cases(self.images_dir, self.labels_dir, 12, None)
        ids = [c["case_id"] for c in cases]
        for priority_id in sfx.PRIORITY_FN_IDS + sfx.PRIORITY_FP_IDS:
            self.assertIn(priority_id, ids)

    def test_priority_fn_source_label(self):
        cases = sfx.select_cases(self.images_dir, self.labels_dir, 12, None)
        fn_cases = [c for c in cases if c["source"] == "priority_fn"]
        self.assertEqual(len(fn_cases), len(sfx.PRIORITY_FN_IDS))

    def test_fp_source_label(self):
        cases = sfx.select_cases(self.images_dir, self.labels_dir, 12, None)
        fp_cases = [c for c in cases if c["source"] == "priority_fp"]
        self.assertEqual(len(fp_cases), len(sfx.PRIORITY_FP_IDS))

    def test_deterministic_across_calls(self):
        c1 = sfx.select_cases(self.images_dir, self.labels_dir, 12, None)
        c2 = sfx.select_cases(self.images_dir, self.labels_dir, 12, None)
        self.assertEqual([c["case_id"] for c in c1], [c["case_id"] for c in c2])

    def test_case_list_override(self):
        case_file = self.tmpdir / "list.txt"
        # Use only 2 priority IDs explicitly
        ids = sfx.PRIORITY_FN_IDS[:2]
        case_file.write_text("\n".join(ids), encoding="utf-8")
        cases = sfx.select_cases(self.images_dir, self.labels_dir, 10, case_file)
        self.assertEqual(len(cases), 2)
        self.assertEqual([c["case_id"] for c in cases], ids)

    def test_missing_priority_id_is_skipped(self):
        # Remove one priority FN image
        (self.images_dir / f"{sfx.PRIORITY_FN_IDS[0]}_0000.png").unlink()
        cases = sfx.select_cases(self.images_dir, self.labels_dir, 12, None)
        ids = [c["case_id"] for c in cases]
        self.assertNotIn(sfx.PRIORITY_FN_IDS[0], ids)

    def test_no_duplicate_ids(self):
        cases = sfx.select_cases(self.images_dir, self.labels_dir, 12, None)
        ids = [c["case_id"] for c in cases]
        self.assertEqual(len(ids), len(set(ids)))


class TestFullStubRun(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.images_dir, self.labels_dir, self.ckpt = _build_test_dir(self.tmpdir)
        self.output_dir = self.tmpdir / "smoke_out"

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def _run(self, extra_args=None) -> dict:
        argv = [
            "--input_dir", str(self.images_dir),
            "--labels_dir", str(self.labels_dir),
            "--checkpoint", str(self.ckpt),
            "--output_dir", str(self.output_dir),
            "--img_size", "256",
            "--device", "cpu",
            "--num_cases", "12",
            "--no_visuals",
        ] + (extra_args or [])
        mock_meta = dict(_MOCK_CKPT_META)  # fresh copy
        with patch("smoke_foundation_x.FoundationXBackbone", DummySmokeBackbone), \
             patch("smoke_foundation_x._load_checkpoint_metadata", return_value=mock_meta):
            sfx.main(argv)
        with (self.output_dir / "foundation_x_smoke_summary.json").open(encoding="utf-8") as fh:
            return json.load(fh)

    def test_summary_json_written(self):
        summary = self._run()
        self.assertTrue((self.output_dir / "foundation_x_smoke_summary.json").exists())

    def test_schema_version_is_1(self):
        summary = self._run()
        self.assertEqual(summary["schema_version"], 1)

    def test_status_is_pass_with_clean_backbone(self):
        summary = self._run()
        self.assertEqual(summary["status"], "PASS")

    def test_exit_code_0_on_pass(self):
        summary = self._run()
        self.assertEqual(summary["exit_code"], 0)

    def test_selection_log_csv_written_with_correct_row_count(self):
        self._run()
        path = self.output_dir / "selection_log.csv"
        self.assertTrue(path.exists())
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 12)

    def test_per_case_stats_csv_has_48_rows(self):
        self._run()
        path = self.output_dir / "per_case_stats.csv"
        self.assertTrue(path.exists())
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 12 * 4)

    def test_report_md_written_with_six_sections(self):
        self._run()
        report = (self.output_dir / "foundation_x_smoke_report.md").read_text(encoding="utf-8")
        for section in (
            "## Repository findings",
            "## Smoke test setup",
            "## Results",
            "## Failure cases",
            "## Decision",
            "## Next recommended PR",
        ):
            self.assertIn(section, report)

    def test_load_metadata_json_written(self):
        self._run()
        self.assertTrue((self.output_dir / "load_metadata.json").exists())

    def test_num_cases_processed_in_summary(self):
        summary = self._run()
        self.assertEqual(summary["inputs"]["num_cases_processed"], 12)


class TestNaNInjectionYieldsPartialPass(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.images_dir, self.labels_dir, self.ckpt = _build_test_dir(self.tmpdir)
        self.output_dir = self.tmpdir / "smoke_nan_out"

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def test_nan_backbone_yields_partial_pass_with_exit_0(self):
        argv = [
            "--input_dir", str(self.images_dir),
            "--labels_dir", str(self.labels_dir),
            "--checkpoint", str(self.ckpt),
            "--output_dir", str(self.output_dir),
            "--img_size", "256",
            "--device", "cpu",
            "--num_cases", "12",
            "--no_visuals",
        ]
        mock_meta = dict(_MOCK_CKPT_META)
        with patch("smoke_foundation_x.FoundationXBackbone", DummySmokeBackboneNaN), \
             patch("smoke_foundation_x._load_checkpoint_metadata", return_value=mock_meta):
            exit_code = sfx.main(argv)

        self.assertEqual(exit_code, 0)

        with (self.output_dir / "foundation_x_smoke_summary.json").open(encoding="utf-8") as fh:
            summary = json.load(fh)

        self.assertEqual(summary["status"], "PARTIAL_PASS")
        self.assertEqual(summary["exit_code"], 0)
        self.assertFalse(summary["pass_criteria"]["features_finite"])
        self.assertTrue(len(summary["failures"]) > 0)

    def test_partial_pass_failure_messages_name_stage(self):
        argv = [
            "--input_dir", str(self.images_dir),
            "--labels_dir", str(self.labels_dir),
            "--checkpoint", str(self.ckpt),
            "--output_dir", str(self.output_dir),
            "--img_size", "256",
            "--device", "cpu",
            "--num_cases", "12",
            "--no_visuals",
        ]
        mock_meta = dict(_MOCK_CKPT_META)
        with patch("smoke_foundation_x.FoundationXBackbone", DummySmokeBackboneNaN), \
             patch("smoke_foundation_x._load_checkpoint_metadata", return_value=mock_meta):
            sfx.main(argv)

        with (self.output_dir / "foundation_x_smoke_summary.json").open(encoding="utf-8") as fh:
            summary = json.load(fh)

        # Every failure message should reference "stage 2" (where NaN is injected)
        for f_str in summary["failures"]:
            self.assertIn("stage 2", f_str)


class TestBlockedOnMissingCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.images_dir = self.tmpdir / "images"
        self.images_dir.mkdir()
        self.labels_dir = self.tmpdir / "labels"
        self.labels_dir.mkdir()
        self.output_dir = self.tmpdir / "out"

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def test_exits_2_when_checkpoint_missing(self):
        argv = [
            "--input_dir", str(self.images_dir),
            "--labels_dir", str(self.labels_dir),
            "--checkpoint", str(self.tmpdir / "nonexistent.pth"),
            "--output_dir", str(self.output_dir),
            "--img_size", "256",
            "--device", "cpu",
            "--num_cases", "12",
            "--no_visuals",
        ]
        with self.assertRaises(SystemExit) as cm:
            sfx.main(argv)
        self.assertEqual(cm.exception.code, 2)

    def test_blocked_summary_json_written(self):
        argv = [
            "--input_dir", str(self.images_dir),
            "--labels_dir", str(self.labels_dir),
            "--checkpoint", str(self.tmpdir / "nonexistent.pth"),
            "--output_dir", str(self.output_dir),
            "--img_size", "256",
            "--device", "cpu",
            "--num_cases", "12",
            "--no_visuals",
        ]
        with self.assertRaises(SystemExit):
            sfx.main(argv)
        summary_path = self.output_dir / "foundation_x_smoke_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertEqual(summary["exit_code"], 2)


class TestD039Compliance(unittest.TestCase):
    """No 'hausdorff' key may appear anywhere in the JSON or YAML summary (D-039)."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.images_dir, self.labels_dir, self.ckpt = _build_test_dir(self.tmpdir)
        self.output_dir = self.tmpdir / "smoke_d039"

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def _run_and_read_json(self) -> str:
        argv = [
            "--input_dir", str(self.images_dir),
            "--labels_dir", str(self.labels_dir),
            "--checkpoint", str(self.ckpt),
            "--output_dir", str(self.output_dir),
            "--img_size", "256",
            "--device", "cpu",
            "--num_cases", "12",
            "--no_visuals",
        ]
        mock_meta = dict(_MOCK_CKPT_META)
        with patch("smoke_foundation_x.FoundationXBackbone", DummySmokeBackbone), \
             patch("smoke_foundation_x._load_checkpoint_metadata", return_value=mock_meta):
            sfx.main(argv)
        return (self.output_dir / "foundation_x_smoke_summary.json").read_text(encoding="utf-8")

    def test_no_hausdorff_in_json(self):
        raw = self._run_and_read_json()
        self.assertNotIn("hausdorff", raw.lower())

    def test_no_hausdorff_in_yaml(self):
        self._run_and_read_json()  # also produces YAML
        yaml_path = self.output_dir / "foundation_x_smoke_summary.yaml"
        if yaml_path.exists():
            raw = yaml_path.read_text(encoding="utf-8")
            self.assertNotIn("hausdorff", raw.lower())

    def test_no_hausdorff_in_report_md(self):
        self._run_and_read_json()
        raw = (self.output_dir / "foundation_x_smoke_report.md").read_text(encoding="utf-8")
        self.assertNotIn("hausdorff", raw.lower())


class TestSummarySchemaInvariants(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.images_dir, self.labels_dir, self.ckpt = _build_test_dir(self.tmpdir)
        self.output_dir = self.tmpdir / "smoke_schema"

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def _run(self) -> dict:
        argv = [
            "--input_dir", str(self.images_dir),
            "--labels_dir", str(self.labels_dir),
            "--checkpoint", str(self.ckpt),
            "--output_dir", str(self.output_dir),
            "--img_size", "256",
            "--device", "cpu",
            "--num_cases", "12",
            "--no_visuals",
        ]
        mock_meta = dict(_MOCK_CKPT_META)
        with patch("smoke_foundation_x.FoundationXBackbone", DummySmokeBackbone), \
             patch("smoke_foundation_x._load_checkpoint_metadata", return_value=mock_meta):
            sfx.main(argv)
        return json.loads(
            (self.output_dir / "foundation_x_smoke_summary.json").read_text(encoding="utf-8")
        )

    def test_schema_version_equals_1(self):
        self.assertEqual(self._run()["schema_version"], 1)

    def test_audit_name_is_foundation_x_smoke(self):
        self.assertEqual(self._run()["audit_name"], "foundation_x_smoke")

    def test_status_is_valid_string(self):
        summary = self._run()
        self.assertIn(summary["status"], {"PASS", "PARTIAL_PASS", "BLOCKED"})

    def test_limited_run_is_true(self):
        self.assertTrue(self._run()["limited_run"])

    def test_four_stages_in_per_stage(self):
        summary = self._run()
        per_stage = summary["inference"]["per_stage"]
        self.assertEqual(len(per_stage), 4)
        for i in range(4):
            self.assertIn(f"stage_{i}", per_stage)

    def test_pass_criteria_present(self):
        summary = self._run()
        pc = summary["pass_criteria"]
        for key in ("checkpoint_loads", "num_cases_ok", "shapes_match_contract",
                    "features_finite", "features_non_degenerate"):
            self.assertIn(key, pc)

    def test_exit_code_is_int_0_or_2(self):
        summary = self._run()
        self.assertIn(summary["exit_code"], (0, 2))

    def test_no_nan_in_json_values(self):
        self._run()
        raw = (self.output_dir / "foundation_x_smoke_summary.json").read_text(encoding="utf-8")
        # Python's json module with allow_nan=False would fail on NaN;
        # verify the written JSON can be parsed by a strict parser.
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)


if __name__ == "__main__":
    unittest.main()
