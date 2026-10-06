"""Deterministic standard-Ulysses frame-mean regression (no weights, no workers).

The per-frame FP32 sum that seeds the far branch's alpha used to be an atomic
``index_add_``. On CUDA its accumulation order -- and therefore the rounding seed
handed to the denoising trajectory -- varied run to run; that nondeterminism was
measured to grow to O(0.27) relative L2 at the 124-frame endpoint between two
otherwise identical eight-GPU runs, while a single-GPU run is bitwise repeatable.
The replacement sums each overlapped frame as one contiguous slice.

This pins both properties:
  * correctness -- the result equals a plain per-frame sum of the same rows;
  * reproducibility -- repeated calls are bitwise identical.
"""
import os
import unittest
from types import SimpleNamespace

import torch

from src.inference.utils.ulysses_runtime import UlyssesRuntime
from src.models.sequence_layout import SequenceLayout


def _reference_sums(local_x, local_start, layout):
    """Plain per-frame FP32 sums of the video rows inside the local range."""
    out = torch.zeros(layout.num_frames, local_x.shape[-1], dtype=torch.float32,
                      device=local_x.device)
    v0, tpf = layout.video_start, layout.tokens_per_frame
    for frame in range(layout.num_frames):
        a = max(v0 + frame * tpf, local_start)
        b = min(v0 + (frame + 1) * tpf, local_start + local_x.shape[0])
        if b > a:
            out[frame] = local_x[a - local_start:b - local_start].float().sum(dim=0)
    return out


class FrameMeanDeterminismTests(unittest.TestCase):
    def _layout(self):
        tpf, frames, prefix = 6, 5, 4          # small text/audio prefix
        return SequenceLayout(seq_len=prefix + frames * tpf, video_start=prefix,
                              num_frames=frames, tokens_per_frame=tpf)

    def _check(self, device):
        layout = self._layout()
        seq, hidden = layout.seq_len, 8
        torch.manual_seed(0)
        x = torch.randn(seq, hidden, device=device, dtype=torch.bfloat16)
        ranges = [
            (0, seq),                                  # whole sequence
            (0, layout.video_start),                   # prefix only, no video
            (0, layout.video_start + 8),               # prefix plus a partial frame
            (layout.video_start + 1, seq - 3),         # video interior
            (seq - 7, seq),                            # suffix rows only
        ]
        for lo, hi in ranges:
            with self.subTest(device=device, rng=(lo, hi)):
                stub = SimpleNamespace(local_start=lo, local_end=hi)
                local = x[lo:hi]
                first = UlyssesRuntime._local_video_frame_sums(stub, local, layout)
                second = UlyssesRuntime._local_video_frame_sums(stub, local, layout)
                self.assertTrue(torch.equal(first, second),
                                "frame sums must be reproducible")
                torch.testing.assert_close(first, _reference_sums(local, lo, layout),
                                           rtol=0, atol=0)

    def test_cpu(self):
        self._check("cpu")

    @unittest.skipUnless(os.environ.get("RUN_ULYSSES_REFERENCE_CUDA") == "1",
                         "CUDA requires RUN_ULYSSES_REFERENCE_CUDA=1")
    def test_cuda(self):
        self._check("cuda")


if __name__ == "__main__":
    unittest.main()
