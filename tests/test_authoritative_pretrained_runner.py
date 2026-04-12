"""Regression tests for the Colab-friendly authoritative pretrained baseline runner."""

from __future__ import annotations

import runpy
import tempfile
import unittest
from pathlib import Path


def load_runner_module() -> dict:
    return runpy.run_path("scripts/run_authoritative_pretrained_baseline.py")


class TestAuthoritativePretrainedRunner(unittest.TestCase):
    def test_protocol_validation_accepts_dedicated_config(self) -> None:
        module = load_runner_module()
        cfg = module["load_config"]("configs/pretrained_resnet34_authoritative.yaml")

        module["validate_authoritative_pretrained_protocol"](cfg)

    def test_protocol_validation_accepts_dedicated_roi_crop_config(self) -> None:
        module = load_runner_module()
        cfg = module["load_config"]("configs/pretrained_resnet34_roi_crop_authoritative.yaml")

        module["validate_authoritative_pretrained_protocol"](cfg)

    def test_protocol_validation_rejects_off_protocol_config(self) -> None:
        module = load_runner_module()
        cfg = module["load_config"]("configs/pretrained_resnet34_authoritative.yaml")
        cfg["training"]["optimizer"] = "Adam"

        with self.assertRaisesRegex(ValueError, "training.optimizer"):
            module["validate_authoritative_pretrained_protocol"](cfg)

    def test_non_all_stage_requires_run_dir(self) -> None:
        module = load_runner_module()

        with self.assertRaisesRegex(ValueError, "requires --run_dir"):
            module["run_stage"](
                config_path="configs/pretrained_resnet34_authoritative.yaml",
                run_dir=None,
                stage="select",
            )

    def test_all_stage_reuses_one_authoritative_run_dir_across_steps(self) -> None:
        module = load_runner_module()
        run_stage = module["run_stage"]
        globals_dict = run_stage.__globals__

        call_log: list[tuple[str, object, object]] = []

        def fake_train(cfg, *, config_path, run_dir):
            checkpoint_path = Path(run_dir) / "checkpoints" / "best_checkpoint.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")
            call_log.append(("train", Path(config_path), Path(run_dir)))
            return 0.42

        def fake_select(cfg, *, checkpoint_path, model_type, selection_state_path):
            selection_path = Path(selection_state_path)
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text("selected_threshold: 0.5\n", encoding="utf-8")
            call_log.append(
                ("select", Path(checkpoint_path), selection_path)
            )
            return {"selected_threshold": 0.5}

        def fake_evaluate(cfg, *, checkpoint_path, model_type, selection_state_path):
            call_log.append(
                ("test", Path(checkpoint_path), Path(selection_state_path))
            )
            return None

        original_train = globals_dict["train"]
        original_select = globals_dict["select_threshold_and_save"]
        original_evaluate = globals_dict["evaluate"]

        try:
            globals_dict["train"] = fake_train
            globals_dict["select_threshold_and_save"] = fake_select
            globals_dict["evaluate"] = fake_evaluate

            with tempfile.TemporaryDirectory() as tmp_dir:
                run_dir = Path(tmp_dir) / "artifacts" / "runs" / "run_001"
                resolved_run_dir = run_stage(
                    config_path="configs/pretrained_resnet34_authoritative.yaml",
                    run_dir=run_dir,
                    stage="all",
                )

                self.assertEqual(resolved_run_dir, run_dir.resolve())
                self.assertEqual(call_log[0], ("train", Path("configs/pretrained_resnet34_authoritative.yaml"), run_dir.resolve()))
                self.assertEqual(call_log[1][0], "select")
                self.assertEqual(call_log[2][0], "test")
                self.assertEqual(call_log[1][1], run_dir.resolve() / "checkpoints" / "best_checkpoint.pth")
                self.assertEqual(call_log[1][2], run_dir.resolve() / "selection" / "selection_state.yaml")
                self.assertEqual(call_log[2][1], run_dir.resolve() / "checkpoints" / "best_checkpoint.pth")
                self.assertEqual(call_log[2][2], run_dir.resolve() / "selection" / "selection_state.yaml")
        finally:
            globals_dict["train"] = original_train
            globals_dict["select_threshold_and_save"] = original_select
            globals_dict["evaluate"] = original_evaluate

    def test_select_test_stage_reuses_existing_best_checkpoint_without_training(self) -> None:
        module = load_runner_module()
        run_stage = module["run_stage"]
        globals_dict = run_stage.__globals__

        call_log: list[tuple[str, object, object]] = []

        def fake_train(cfg, *, config_path, run_dir):
            raise AssertionError("select_test stage must not invoke train()")

        def fake_select(cfg, *, checkpoint_path, model_type, selection_state_path):
            selection_path = Path(selection_state_path)
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text("selected_threshold: 0.5\n", encoding="utf-8")
            call_log.append(("select", Path(checkpoint_path), selection_path))
            return {"selected_threshold": 0.5}

        def fake_evaluate(cfg, *, checkpoint_path, model_type, selection_state_path):
            call_log.append(("test", Path(checkpoint_path), Path(selection_state_path)))
            return None

        original_train = globals_dict["train"]
        original_select = globals_dict["select_threshold_and_save"]
        original_evaluate = globals_dict["evaluate"]

        try:
            globals_dict["train"] = fake_train
            globals_dict["select_threshold_and_save"] = fake_select
            globals_dict["evaluate"] = fake_evaluate

            with tempfile.TemporaryDirectory() as tmp_dir:
                run_dir = Path(tmp_dir) / "artifacts" / "runs" / "run_001"
                checkpoint_path = run_dir / "checkpoints" / "best_checkpoint.pth"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_bytes(b"checkpoint")

                resolved_run_dir = run_stage(
                    config_path="configs/pretrained_resnet34_authoritative.yaml",
                    run_dir=run_dir,
                    stage="select_test",
                )

                self.assertEqual(resolved_run_dir, run_dir.resolve())
                self.assertEqual(call_log[0][0], "select")
                self.assertEqual(call_log[1][0], "test")
                self.assertEqual(call_log[0][1], checkpoint_path.resolve())
                self.assertEqual(call_log[0][2], run_dir.resolve() / "selection" / "selection_state.yaml")
                self.assertEqual(call_log[1][1], checkpoint_path.resolve())
                self.assertEqual(call_log[1][2], run_dir.resolve() / "selection" / "selection_state.yaml")
        finally:
            globals_dict["train"] = original_train
            globals_dict["select_threshold_and_save"] = original_select
            globals_dict["evaluate"] = original_evaluate


if __name__ == "__main__":
    unittest.main()
