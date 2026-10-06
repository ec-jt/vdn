"""CPU regressions for malformed adapters; no checkpoints or Hub access."""
import unittest

import torch

from src.inference.utils.lora import merge_lora_state


class LoraValidationTests(unittest.TestCase):
    def model(self, dtype=torch.float32):
        model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False, dtype=dtype))
        model[0].weight.data.zero_()
        return model

    def test_orphan_b_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pair"):
            merge_lora_state(self.model(), {"0.lora_B.default.weight": torch.ones(3, 2)})

    def test_orphan_a_is_rejected_before_any_merge(self):
        model = self.model()
        state = {"0.lora_A.default.weight": torch.ones(2, 4),
                 "0.lora_B.default.weight": torch.ones(3, 2),
                 "0.lora_A.turbo.weight": torch.ones(2, 4)}
        with self.assertRaisesRegex(ValueError, "pair"):
            merge_lora_state(model, state)
        self.assertEqual(torch.count_nonzero(model[0].weight), 0)

    def test_broadcastable_delta_is_rejected(self):
        for a_shape, b_shape in [((2, 4), (1, 2)), ((2, 1), (3, 2))]:
            with self.subTest(a=a_shape, b=b_shape):
                model = self.model()
                with self.assertRaisesRegex(ValueError, "shape"):
                    merge_lora_state(model, {"0.lora_A.weight": torch.ones(a_shape),
                                             "0.lora_B.weight": torch.ones(b_shape)})
                self.assertEqual(torch.count_nonzero(model[0].weight), 0)

    def test_bad_rank_or_dimension_is_rejected_before_any_merge(self):
        for a_shape, b_shape in [((2, 4), (3, 1)), ((4,), (3, 1)), ((0, 4), (3, 0))]:
            with self.subTest(a=a_shape, b=b_shape):
                model = self.model()
                state = {"0.lora_A.default.weight": torch.ones(2, 4),
                         "0.lora_B.default.weight": torch.ones(3, 2),
                         "0.lora_A.turbo.weight": torch.ones(a_shape),
                         "0.lora_B.turbo.weight": torch.ones(b_shape)}
                with self.assertRaisesRegex(ValueError, "shape"):
                    merge_lora_state(model, state)
                self.assertEqual(torch.count_nonzero(model[0].weight), 0)

    def test_valid_default_turbo_then_external_preserves_bf16_arithmetic(self):
        model = self.model(torch.bfloat16)
        generator = torch.Generator().manual_seed(75)
        pairs = [(torch.randn(2, 4, generator=generator),
                  torch.randn(3, 2, generator=generator)) for _ in range(3)]
        expected = model[0].weight.detach().clone()
        state = {}
        for adapter, (a, b) in zip(("default", "turbo"), pairs[:2]):
            state[f"0.lora_A.{adapter}.weight"] = a
            state[f"0.lora_B.{adapter}.weight"] = b
            expected.add_((b.float() @ a.float()).to(expected.dtype))
        self.assertEqual(merge_lora_state(model, state), 2)
        a, b = pairs[2]
        expected.add_(((b.float() @ a.float()) * 0.5).to(expected.dtype))
        self.assertEqual(merge_lora_state(
            model, {"0.lora_A.weight": a, "0.lora_B.weight": b}, 0.5), 1)
        torch.testing.assert_close(model[0].weight, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
