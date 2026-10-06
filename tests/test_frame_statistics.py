"""Regression coverage for bounded eager evaluation statistics."""

import unittest
from unittest.mock import patch

import torch

from src.models.linear_attention import scan


class FrameStatisticsTests(unittest.TestCase):
    def inputs(self, device="cpu", dtype=torch.float32):
        generator = torch.Generator(device=device).manual_seed(41)
        frames = 2 * scan.STATS_CHUNK_FRAMES + 3
        # Match the noncontiguous frame/head/token/channel production layout.
        key = torch.randn(frames, 7, 2, 4, generator=generator,
                          device=device, dtype=dtype).permute(0, 2, 1, 3)
        value = torch.randn(frames, 7, 2, 6, generator=generator,
                            device=device, dtype=dtype).permute(0, 2, 1, 3)
        beta = torch.rand(frames, 2, 7, generator=generator, device=device)
        return key, value, beta

    def check_eager(self, device, dtype):
        key, value, beta = self.inputs(device, dtype)
        for a_fp32 in (False, True):
            with torch.no_grad():
                expected = scan._frame_statistics_pass(key, value, beta, a_fp32)
                with patch.object(scan, "_frame_statistics_pass",
                                  wraps=scan._frame_statistics_pass) as calls:
                    actual = scan._frame_statistics(key, value, beta, a_fp32)
                self.assertEqual(len(calls.call_args_list), 3)
                for call in calls.call_args_list:
                    self.assertLessEqual(call.args[0].shape[0], scan.STATS_CHUNK_FRAMES)
                    self.assertFalse(call.kwargs["inference"])
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want)

    def test_eager_cpu_chunk_parity(self):
        self.check_eager("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_eager_cuda_bf16_chunk_parity(self):
        self.check_eager("cuda", torch.bfloat16)

    def test_training_retains_single_pass_and_gradients(self):
        key, value, beta = self.inputs()
        key.requires_grad_()
        value.requires_grad_()
        beta.requires_grad_()
        with torch.enable_grad(), patch.object(
            scan, "_frame_statistics_pass", wraps=scan._frame_statistics_pass
        ) as calls:
            a, b = scan._frame_statistics(key, value, beta, True)
            (a.sum() + b.sum()).backward()
        self.assertEqual(calls.call_count, 1)
        for tensor in (key, value, beta):
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
