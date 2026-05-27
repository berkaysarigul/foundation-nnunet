"""Tests for the PR-10C refiner U-Net."""

from __future__ import annotations

import unittest

import torch

from src.models.refiner_unet import FoundationXPriorRefinerUNet


class TestFoundationXPriorRefinerUNet(unittest.TestCase):
    def test_forward_two_channel_input(self) -> None:
        model = FoundationXPriorRefinerUNet(in_channels=2, base_channels=4, norm="group")
        x = torch.randn(2, 2, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 64, 64))

    def test_forward_one_channel_input(self) -> None:
        model = FoundationXPriorRefinerUNet(in_channels=1, base_channels=4, norm="group")
        x = torch.randn(2, 1, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
