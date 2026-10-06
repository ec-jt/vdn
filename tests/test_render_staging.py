"""Small decoder/sampler residency checks only, never a production render."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.inference.render import decode_and_save, generate_latents
from src.inference.utils.metrics import PhaseMetrics


class TinyDecoder(torch.nn.Module):
    def __init__(self, name, events):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.name, self.events, self.other = name, events, None
        self.fail = False
        self.config = SimpleNamespace(latents_mean=[0], latents_std=[1], sampling_rate=24000)

    def decode(self, value, return_dict=False):
        self.events.append(self.name)
        assert self.weight.is_cuda and value.is_cuda
        assert not self.other.weight.is_cuda, "both decoders were resident"
        if self.fail:
            raise RuntimeError("injected decode failure")
        return (value * self.weight,)


class ReleaseSpy:
    def __init__(self, events):
        self.events = events

    def release(self):
        self.events.append("release")


class MetricsTests(unittest.TestCase):
    def test_persistent_failure_and_cold_labels(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "metrics.jsonl")
            metrics = PhaseMetrics(path, "cpu")
            with metrics.phase("cpu_assembly"):
                pass
            with metrics.phase("denoise"):
                for i in range(2):
                    with metrics.denoise_step(i):
                        pass
            with self.assertRaisesRegex(RuntimeError, "decode"):
                with metrics.phase("video_decode"):
                    raise RuntimeError("decode")
            events = [json.loads(line) for line in Path(path).read_text().splitlines()]
            ends = [e for e in events if e["event"] == "phase_end"]
            self.assertEqual(ends[-1]["status"], "error")
            steps = [e for e in ends if e["phase"] == "denoise_step"]
            self.assertIn("first_cold_call", steps[0]["execution_label"])
            self.assertIn("later_call", steps[1]["execution_label"])
            self.assertGreater(ends[0]["cpu_peak_rss_bytes"], 0)
            self.assertTrue(all(e["seconds"] >= 0 for e in ends))


class StageBSamplerTests(unittest.TestCase):
    def test_fifty_conditional_forwards_and_repeatable_finite_latents(self):
        class TinyDenoiser(torch.nn.Module):
            config = SimpleNamespace(patch_size=(1, 1, 1), in_channels=4)
            transformer_blocks = ()
            calls = 0

            def forward(self, hidden_states, audio_hidden_states, **kwargs):
                self.calls += 1
                return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states)

        model = TinyDenoiser()
        with tempfile.TemporaryDirectory() as root, patch(
                "src.inference.render.LATENT_H", 4), patch("src.inference.render.LATENT_W", 4):
            results = []
            for run in range(2):
                metrics = PhaseMetrics(str(Path(root) / f"run-{run}.jsonl"), "cpu")
                results.append(generate_latents(
                    model, torch.zeros(2, 8, dtype=torch.bfloat16),
                    torch.ones(2, dtype=torch.long), 124, 50, 42, "cpu",
                    video_shift=12.0, audio_shift=3.0, metrics=metrics))
                self.assertEqual(model.calls, (run + 1) * 50)
                self.assertEqual(metrics.denoise_steps, 50)
                self.assertTrue(all(torch.isfinite(t).all() for t in results[-1]))
            for first, second in zip(*results):
                torch.testing.assert_close(first, second, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GPUStagingTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.video = TinyDecoder("video", self.events)
        self.audio = TinyDecoder("audio", self.events)
        self.video.other, self.audio.other = lambda: None, lambda: None
        # Avoid registering the peer as a recursive child module.
        object.__setattr__(self.video, "other", self.audio)
        object.__setattr__(self.audio, "other", self.video)
        self.release = ReleaseSpy(self.events)
        self.latents = torch.zeros(1, 3, 2, 4, 4, device="cuda")
        self.audio_latents = torch.zeros(2, 1, 16, device="cuda")

    def encode(self, frames, *, output_path, audio, **kwargs):
        self.assertEqual(frames.device.type, "cpu")
        self.assertEqual(audio.device.type, "cpu")
        self.assertEqual(frames.dtype, torch.uint8)
        self.assertEqual(tuple(frames.shape), (2, 4, 4, 3))
        self.assertEqual(tuple(audio.shape), (2, 16))
        self.assertFalse(self.video.weight.is_cuda)
        self.assertFalse(self.audio.weight.is_cuda)
        self.events.append("encode")
        Path(output_path).write_bytes(b"test output, not a video")

    def test_staged_residency_repeat_and_atomic_output(self):
        with tempfile.TemporaryDirectory() as root, patch(
                "src.inference.render.encode_video", side_effect=self.encode):
            metrics = PhaseMetrics(str(Path(root) / "metrics.jsonl"), "cuda:0")
            for _ in range(2):
                decode_and_save(self.latents, self.audio_latents, self.video, self.audio,
                                str(Path(root) / "out.mp4"), "cuda:0", sequential=True,
                                offload=self.release, metrics=metrics)
            self.assertEqual(self.events, ["release", "video", "audio", "encode"] * 2)
            self.assertTrue((Path(root) / "out.mp4").exists())
            self.assertFalse((Path(root) / "out.partial.mp4").exists())

    def test_failure_evicts_and_retains_metrics_then_retry(self):
        for failing in (self.video, self.audio):
            self.events.clear()
            with self.subTest(decoder=failing.name), tempfile.TemporaryDirectory() as root, patch(
                    "src.inference.render.encode_video", side_effect=self.encode):
                path = str(Path(root) / "metrics.jsonl")
                metrics = PhaseMetrics(path, "cuda:0")
                failing.fail = True
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    decode_and_save(self.latents, self.audio_latents, self.video, self.audio,
                                    str(Path(root) / "out.mp4"), "cuda:0", sequential=True,
                                    offload=self.release, metrics=metrics)
                self.assertFalse(self.video.weight.is_cuda)
                self.assertFalse(self.audio.weight.is_cuda)
                self.assertNotIn("encode", self.events)
                persisted = [json.loads(s) for s in Path(path).read_text().splitlines()]
                self.assertEqual(persisted[-1]["status"], "error")
                self.assertIn("cuda_peak_reserved_bytes", persisted[-1])
                failing.fail = False
                decode_and_save(self.latents, self.audio_latents, self.video, self.audio,
                                str(Path(root) / "out.mp4"), "cuda:0", sequential=True,
                                offload=self.release, metrics=metrics)

    def test_nested_cuda_peaks_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            metrics = PhaseMetrics(str(Path(root) / "metrics.jsonl"), "cuda:0")
            with metrics.phase("outer"):
                with metrics.phase("inner"):
                    value = torch.zeros(1024 * 1024, device="cuda")
                del value
                with metrics.phase("later"):
                    pass
            ends = {e["phase"]: e for e in metrics.events if e["event"] == "phase_end"}
            self.assertGreaterEqual(ends["outer"]["cuda_peak_allocated_bytes"],
                                    ends["inner"]["cuda_peak_allocated_bytes"])

    def test_native_sampler_eight_steps_and_metrics(self):
        class TinyDenoiser(torch.nn.Module):
            config = SimpleNamespace(patch_size=(1, 1, 1), in_channels=4)
            transformer_blocks = ()

            def forward(self, hidden_states, audio_hidden_states, **kwargs):
                return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states)

        with tempfile.TemporaryDirectory() as root, patch(
                "src.inference.render.LATENT_H", 4), patch("src.inference.render.LATENT_W", 4):
            metrics = PhaseMetrics(str(Path(root) / "metrics.jsonl"), "cuda:0")
            seconds = []
            result = generate_latents(
                TinyDenoiser(), torch.zeros(2, 8, device="cuda", dtype=torch.bfloat16),
                torch.ones(2, dtype=torch.long), 124, 8, 42, "cuda:0",
                step_seconds=seconds, metrics=metrics)
            self.assertEqual(len(seconds), 8)
            self.assertEqual(metrics.denoise_steps, 8)
            self.assertTrue(all(torch.isfinite(t).all() for t in result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
