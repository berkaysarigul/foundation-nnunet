import unittest
import sys
import types

if "timm" not in sys.modules:
    sys.modules["timm"] = types.SimpleNamespace(create_model=lambda *args, **kwargs: None)

from src.models.hybrid import assert_corrected_hybrid_scale_contract


class TestHybridScaleContract(unittest.TestCase):
    def test_contract_accepts_256_input_shapes(self):
        assert_corrected_hybrid_scale_contract(
            fx0=(2, 128, 64, 64),
            fx1=(2, 256, 32, 32),
            fx2=(2, 512, 16, 16),
            fx3=(2, 1024, 8, 8),
            e3=(2, 256, 64, 64),
            e4=(2, 512, 32, 32),
            h16_context=(2, 1024, 16, 16),
            h32_context=(2, 1024, 8, 8),
        )

    def test_contract_accepts_512_input_shapes(self):
        assert_corrected_hybrid_scale_contract(
            fx0=(1, 128, 128, 128),
            fx1=(1, 256, 64, 64),
            fx2=(1, 512, 32, 32),
            fx3=(1, 1024, 16, 16),
            e3=(1, 256, 128, 128),
            e4=(1, 512, 64, 64),
            h16_context=(1, 1024, 32, 32),
            h32_context=(1, 1024, 16, 16),
        )

    def test_contract_rejects_fx0_to_wrong_encoder_scale(self):
        with self.assertRaisesRegex(AssertionError, r"fx\[0\].*e3"):
            assert_corrected_hybrid_scale_contract(
                fx0=(2, 128, 32, 32),
                fx1=(2, 256, 32, 32),
                fx2=(2, 512, 16, 16),
                fx3=(2, 1024, 8, 8),
                e3=(2, 256, 64, 64),
                e4=(2, 512, 32, 32),
                h16_context=(2, 1024, 16, 16),
                h32_context=(2, 1024, 8, 8),
            )

    def test_contract_rejects_non_adjacent_h32_to_h16_transition(self):
        with self.assertRaisesRegex(AssertionError, r"H/32 -> H/16"):
            assert_corrected_hybrid_scale_contract(
                fx0=(2, 128, 64, 64),
                fx1=(2, 256, 32, 32),
                fx2=(2, 512, 16, 16),
                fx3=(2, 1024, 7, 7),
                e3=(2, 256, 64, 64),
                e4=(2, 512, 32, 32),
                h16_context=(2, 1024, 16, 16),
                h32_context=(2, 1024, 7, 7),
            )


if __name__ == "__main__":
    unittest.main()
