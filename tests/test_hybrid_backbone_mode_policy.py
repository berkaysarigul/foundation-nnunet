import unittest

import torch

from src.training.trainer import apply_foundation_x_backbone_train_mode_policy


class DummyBackbone(torch.nn.Module):
    pass


class DummyFoundationX(torch.nn.Module):
    def __init__(self, frozen: bool):
        super().__init__()
        self.frozen = frozen
        self.backbone = DummyBackbone()


class DummyHybrid(torch.nn.Module):
    def __init__(self, frozen_backbone: bool):
        super().__init__()
        self.frozen_backbone = frozen_backbone
        self.foundation_x = DummyFoundationX(frozen_backbone)


class FoundationXBackboneModePolicyTests(unittest.TestCase):
    def test_frozen_backbone_is_forced_to_eval_during_train(self):
        model = DummyHybrid(frozen_backbone=True)

        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.foundation_x.training)
        self.assertTrue(model.foundation_x.backbone.training)

        apply_foundation_x_backbone_train_mode_policy(model)

        self.assertFalse(model.foundation_x.backbone.training)

    def test_unfrozen_backbone_is_not_forced_to_eval_during_train(self):
        model = DummyHybrid(frozen_backbone=False)

        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.foundation_x.training)
        self.assertTrue(model.foundation_x.backbone.training)

        apply_foundation_x_backbone_train_mode_policy(model)

        self.assertTrue(model.foundation_x.backbone.training)

    def test_models_without_foundation_x_are_ignored(self):
        model = torch.nn.Conv2d(1, 1, kernel_size=1)
        model.train()

        apply_foundation_x_backbone_train_mode_policy(model)

        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
