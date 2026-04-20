"""Regression tests for repeated-split authoritative pretrained study orchestration."""

from __future__ import annotations

import runpy
import tempfile
import unittest
from pathlib import Path

import yaml


def load_runner_module() -> dict:
    return runpy.run_path("scripts/run_repeated_split_pretrained_study.py")


class TestRepeatedSplitPretrainedStudyRunner(unittest.TestCase):
    def test_build_cfg_with_split_override_sets_data_splits_path(self) -> None:
        module = load_runner_module()
        cfg = {
            "model": {"type": "pretrained_resnet34_unet"},
            "data": {"processed_dir": "data/processed/pneumothorax_trusted_v1"},
        }
        override_path = Path("C:/tmp/split_001.json")

        updated = module["build_cfg_with_split_override"](
            cfg,
            split_override_path=override_path,
        )

        self.assertEqual(updated["data"]["processed_dir"], "data/processed/pneumothorax_trusted_v1")
        self.assertEqual(updated["data"]["splits_path"], str(override_path.resolve()))
        self.assertNotIn("splits_path", cfg["data"])

    def test_run_repeated_split_study_materializes_overrides_and_inventory(self) -> None:
        module = load_runner_module()
        globals_dict = module["run_repeated_split_study"].__globals__

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            study_dir = root / "artifacts" / "repeated_splits" / "study_alpha"
            metadata_dir = study_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            study_manifest_path = metadata_dir / "split_manifest.yaml"
            study_manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "study_id": "study_alpha",
                        "split_instances": [
                            {
                                "split_instance_id": "split_001",
                                "split_seed": 42,
                                "train_ids": ["img_001", "img_002"],
                                "val_ids": ["img_003"],
                                "test_ids": ["img_004"],
                            },
                            {
                                "split_instance_id": "split_002",
                                "split_seed": 43,
                                "train_ids": ["img_005", "img_006"],
                                "val_ids": ["img_007"],
                                "test_ids": ["img_008"],
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            base_config_path = root / "configs" / "pretrained_resnet34_authoritative.yaml"
            base_config_path.parent.mkdir(parents=True, exist_ok=True)
            base_config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"type": "pretrained_resnet34_unet"},
                        "data": {"processed_dir": "data/processed/pneumothorax_trusted_v1"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            call_log: list[tuple[list[str], Path]] = []

            original_prepare = globals_dict["prepare_repeated_split_study_artifacts"]
            original_subprocess_run = globals_dict["subprocess"].run

            def fake_prepare(study_id: str, *, study_root: Path):
                self.assertEqual(study_id, "study_alpha")
                self.assertEqual(study_root, study_dir.parent)
                return original_prepare(study_id, study_root=study_root)

            def fake_subprocess_run(command, cwd, check):
                call_log.append((list(command), Path(cwd)))
                return None

            try:
                globals_dict["prepare_repeated_split_study_artifacts"] = fake_prepare
                globals_dict["subprocess"].run = fake_subprocess_run

                inventory_path = module["run_repeated_split_study"](
                    study_manifest_path=study_manifest_path,
                    base_config_path=base_config_path,
                    stage="select_test",
                )
            finally:
                globals_dict["prepare_repeated_split_study_artifacts"] = original_prepare
                globals_dict["subprocess"].run = original_subprocess_run

            self.assertEqual(len(call_log), 2)
            first_command, first_cwd = call_log[0]
            self.assertEqual(first_cwd, module["REPO_ROOT"])
            self.assertIn("scripts/run_authoritative_pretrained_baseline.py", first_command[1].replace("\\", "/"))
            self.assertIn("--stage", first_command)
            self.assertIn("select_test", first_command)

            split_override_path = study_dir / "metadata" / "split_overrides" / "split_001.json"
            config_override_path = study_dir / "metadata" / "config_overrides" / "split_001.yaml"
            self.assertTrue(split_override_path.exists())
            self.assertTrue(config_override_path.exists())
            config_payload = yaml.safe_load(config_override_path.read_text(encoding="utf-8"))
            self.assertEqual(config_payload["data"]["splits_path"], str(split_override_path.resolve()))

            self.assertTrue(inventory_path.exists())
            inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
            self.assertEqual(inventory["study_id"], "study_alpha")
            self.assertEqual(inventory["stage"], "select_test")
            self.assertEqual(len(inventory["entries"]), 2)
            self.assertEqual(inventory["entries"][0]["split_instance_id"], "split_001")
            self.assertEqual(
                inventory["entries"][0]["split_override_path"],
                str(split_override_path.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
