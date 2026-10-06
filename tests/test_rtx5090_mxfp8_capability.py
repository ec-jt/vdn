"""Focused synthetic tests for standalone capability helpers only."""

import unittest

import torch

from scripts.probe_rtx5090_mxfp8 import (
    dequantize_block32,
    mxfp8_mm,
    prepare_torch_mxfp8,
    quantize_block32,
    swizzle_scales,
    torch_mxfp8_mm,
)


class TestMXFP8Format(unittest.TestCase):
    def test_nonuniform_e8m0(self):
        x = torch.cat([torch.full((2, 32), v) for v in (1., 4., 16.)], dim=1)
        q, s = quantize_block32(x)
        self.assertEqual(s.tolist(), [[119, 121, 123]] * 2)
        torch.testing.assert_close(dequantize_block32(q, s), x)

    def test_zero_and_padding(self):
        x = torch.zeros(3, 33)
        q, s = quantize_block32(x)
        self.assertEqual(q.shape, (3, 64))
        self.assertTrue(bool((s == 127).all()))
        self.assertEqual(float(dequantize_block32(q, s).abs().sum()), 0.)

    def test_swizzle_index_mapping(self):
        s = (torch.arange(256 * 8).reshape(256, 8) % 255).to(torch.uint8)
        actual = swizzle_scales(s)
        expected = torch.empty_like(actual)
        for row in range(256):
            for col in range(8):
                idx = ((row // 128) * 2 + col // 4) * 512
                idx += (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
                expected[idx] = s[row, col]
        torch.testing.assert_close(actual, expected)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
                     "requires CUDA SM120")
class TestSM120Execution(unittest.TestCase):
    def test_triton_ragged_and_k_tail(self):
        torch.manual_seed(19)
        a, sa = quantize_block32(torch.randn(37, 160, device="cuda"))
        w, sw = quantize_block32(torch.randn(73, 160, device="cuda"))
        y, kernel = mxfp8_mm(a, w, sa, sw, return_kernel=True)
        ref = dequantize_block32(a, sa) @ dequantize_block32(w, sw).T
        torch.testing.assert_close(y.float(), ref, atol=0.15, rtol=0.004)
        self.assertIn(".target sm_120a", kernel.asm["ptx"])
        self.assertIn("block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0", kernel.asm["ptx"])

    def test_pytorch_prepared_ragged(self):
        torch.manual_seed(21)
        x = torch.randn(37, 160, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(73, 160, device="cuda", dtype=torch.bfloat16)
        original = x.clone()
        a, sa = prepare_torch_mxfp8(x)
        b, sb = prepare_torch_mxfp8(w)
        y = torch_mxfp8_mm(a, b, sa, sb, output_rows=37, output_cols=73)
        aq, ascale = quantize_block32(x)
        bq, bscale = quantize_block32(w)
        ref = dequantize_block32(aq, ascale) @ dequantize_block32(bq, bscale).T
        torch.testing.assert_close(y.float(), ref, atol=0.15, rtol=0.004)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        self.assertEqual(y.dtype, torch.bfloat16)

    def test_triton_validation(self):
        a, sa = quantize_block32(torch.zeros(32, 32, device="cuda"))
        with self.assertRaisesRegex(ValueError, "uint8 E8M0"):
            mxfp8_mm(a, a, sa.float(), sa)


if __name__ == "__main__":
    unittest.main()
