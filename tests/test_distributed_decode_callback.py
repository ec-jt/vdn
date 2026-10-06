"""CPU-only callback lifecycle and Ulysses wiring tests; no weights or media codecs."""

from contextlib import ExitStack, nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from src.inference import infer_ulysses, render
from src.inference.utils import distributed_decode_callback as callback


class CallbackTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.out = Path(root) / "clip.mp4"
        self.partial = self.out.with_name("clip.partial.mp4")
        self.cfg = NS(vae_source="local-vae", render=NS(out=str(self.out)))
        self.runtime = NS(rank=0, world_size=2, is_main=True, device=torch.device("cpu"))
        self.model = NS(weight_shards=NS(released=True))
        self.video = torch.zeros(1, 3, 2, 2, 2)
        self.audio = torch.zeros(2, 1, 4)
        self.events, self.video_refs, self.raw_refs, self.audio_refs = [], [], [], []
        self.gathers, self.broadcasts = [], []
        self.remote_errors = {}
        self.remote_status = {"error": None, "timings": {"total_seconds": 1.25}}
        self.fail = None

        def gather(values, value):
            self.gathers.append(value)
            if isinstance(value, dict):
                values[:] = [dict(value, rank=rank) for rank in range(2)]
            else:
                values[:] = [None, None]
                values[self.runtime.rank] = value
                values[1 - self.runtime.rank] = self.remote_errors.get(len(self.gathers))

        def broadcast(values, src):
            self.assertEqual(src, 0)
            if not self.runtime.is_main:
                values[0] = self.remote_status
            self.broadcasts.append(values[0])

        def load_video(source):
            self.assertEqual(source, "resolved-local-vae")
            self.events.append("video_load")
            vae = DecoderToken()
            self.video_refs.append(weakref.ref(vae))
            return vae

        def decode_video(vae, latents, **kwargs):
            self.assertIs(vae, self.video_refs[-1]())
            self.assertIs(latents, self.video)
            self.assertEqual(kwargs, dict(device=self.runtime.device, output_cpu=False,
                                          reset_peak_memory=True))
            self.assertFalse(torch.is_grad_enabled())
            self.events.extend(["video_decode", "video_evict"])
            if self.fail == "video_decode":
                raise RuntimeError("injected video_decode")
            if not self.runtime.is_main:
                return None, {"rank": self.runtime.rank}
            raw = torch.linspace(-4, 4, 24).reshape(1, 3, 2, 2, 2).half()
            self.raw_refs.append(weakref.ref(raw))
            return raw, {"rank": self.runtime.rank, "mock_video_metric": 17}

        real_postprocess = callback._postprocess_video

        def postprocess(raw, device):
            self.events.append("postprocess")
            self.assertIs(raw, self.raw_refs[-1]())
            self.assertIsNotNone(self.video_refs[-1]())
            if self.fail == "postprocess":
                raise RuntimeError("injected postprocess")
            return real_postprocess(raw, device)

        owner = self

        class AudioDecoder:
            config = NS(sampling_rate=24000)

            def eval(self):
                owner.events.append("audio_eval")
                return self

            def requires_grad_(self, enabled):
                owner.assertFalse(enabled)
                owner.events.append("audio_no_grad")
                return self

            def to(self, device):
                # Runtime's CPU device represents onload; literal "cpu" is eviction.
                stage = "audio_evict" if isinstance(device, str) else "audio_onload"
                owner.events.append(stage)
                if owner.fail == stage:
                    raise RuntimeError("injected " + stage)
                return self

        def load_audio(*args, **kwargs):
            self.events.append("audio_load")
            self.assertTrue(all(ref() is None for ref in self.video_refs + self.raw_refs))
            self.assertEqual(self.events[-2], "cache")
            if self.fail == "audio_load":
                raise RuntimeError("injected audio_load")
            vae = AudioDecoder()
            self.audio_refs.append(weakref.ref(vae))
            return vae

        def decode_audio(latents, vae, device):
            self.assertIs(latents, self.audio)
            self.assertIs(vae, self.audio_refs[-1]())
            self.assertEqual(device, self.runtime.device)
            self.events.append("audio_decode")
            if self.fail == "audio_decode":
                raise RuntimeError("injected audio_decode")
            if self.fail == "output_race":
                self.out.write_bytes(b"existing")
            return torch.zeros(2, 16)

        def encode(frames, *, fps, output_path, audio, audio_sample_rate):
            self.events.append("encode")
            self.assertTrue(all(ref() is None for ref in self.audio_refs))
            self.assertEqual(frames.device.type, "cpu")
            self.assertEqual(frames.dtype, torch.uint8)
            self.assertEqual(tuple(frames.shape), (2, 2, 2, 3))
            self.assertEqual(audio.device.type, "cpu")
            self.assertEqual(fps, render.FPS)
            self.assertEqual(audio_sample_rate, 24000)
            target = Path(output_path)
            self.assertEqual(target, self.out.with_name(self.out.stem + ".partial.mp4"))
            self.assertFalse(self.out.exists())
            target.write_bytes(b"complete mocked media")
            if self.fail == "writer":
                raise RuntimeError("injected writer")

        def cache():
            self.events.append("cache")
            if self.fail == "cache":
                raise RuntimeError("injected cache")

        self.resolve = self.stack.enter_context(patch.object(
            callback, "resolve_inference_weights", return_value="resolved-local-vae"))
        self.load_video = self.stack.enter_context(patch.object(
            callback, "load_video_decoder", side_effect=load_video))
        self.load_audio = self.stack.enter_context(patch.object(
            callback, "load_cpu_model", side_effect=load_audio))
        self.writer = self.stack.enter_context(patch.object(callback, "encode_video", side_effect=encode))
        # Plain functions avoid Mock.call_args retaining decoders and raw tensors.
        for name, value in (("distributed_decode", decode_video),
                            ("_postprocess_video", postprocess), ("_decode_audio", decode_audio)):
            self.stack.enter_context(patch.object(callback, name, new=value))
        self.stack.enter_context(patch.object(callback.dist, "all_gather_object", side_effect=gather))
        self.stack.enter_context(patch.object(callback.dist, "broadcast_object_list", side_effect=broadcast))
        self.stack.enter_context(patch.object(callback.gc, "collect", side_effect=lambda: self.events.append("gc")))
        self.empty_cache = self.stack.enter_context(patch.object(torch.cuda, "empty_cache", side_effect=cache))

    def call(self):
        return callback.distributed_decode_callback(
            self.cfg, self.model, self.runtime, self.video, self.audio)

    def assert_no_media(self):
        self.assertFalse(self.out.exists())
        self.assertFalse(self.partial.exists())

    def test_sequential_lifetimes_atomic_write_and_fresh_repeat(self):
        expected = ["video_load", "video_decode", "video_evict", "postprocess", "gc", "cache",
                    "audio_load", "audio_eval", "audio_no_grad", "audio_onload", "audio_decode",
                    "audio_evict", "gc", "cache", "encode"]
        real_replace = callback.os.replace
        with patch.object(callback.os, "replace", wraps=real_replace) as replace:
            for index in range(2):
                self.out = self.out.with_name(f"clip-{index}.mp4")
                self.cfg.render.out = str(self.out)
                result = self.call()
                self.assertEqual(self.out.read_bytes(), b"complete mocked media")
                self.assertFalse(self.out.with_name(f"clip-{index}.partial.mp4").exists())
                self.assertEqual(result["distributed_video"], {"rank": 0, "mock_video_metric": 17})
                self.assertEqual([m["rank"] for m in result["decoder_load_seconds_by_rank"]], [0, 1])
                for key in ("video_postprocess_seconds", "audio_load_seconds", "audio_decode_seconds",
                            "media_encode_seconds", "total_seconds"):
                    self.assertGreaterEqual(result[key], 0)
                self.assertTrue(all(ref() is None for ref in self.video_refs + self.raw_refs + self.audio_refs))
            self.assertEqual(replace.call_count, 2)
            for call in replace.call_args_list:
                source, target = map(Path, call.args)
                self.assertEqual(source.parent, target.parent)
                self.assertEqual(source.name, target.stem + ".partial.mp4")
        self.assertEqual(self.events, expected * 2)
        self.assertEqual(len(self.video_refs), 2)
        self.assertIsNot(self.video_refs[0], self.video_refs[1])
        self.assertEqual(len(self.audio_refs), 2)
        self.assertEqual(self.writer.call_count, 2)
        self.assertEqual(self.load_video.call_count, 2)
        self.resolve.assert_called_with("local-vae", local_files_only=True)
        self.load_audio.assert_called_with(callback.AutoencoderKLMiniMaxH3Audio, "resolved-local-vae",
                                           subfolder="audio_vae", torch_dtype=torch.float32,
                                           local_files_only=True)
        self.assertEqual(vars(self.model), {"weight_shards": self.model.weight_shards})

    def test_nonmain_loads_video_but_never_audio_or_writer(self):
        self.runtime.rank, self.runtime.is_main = 1, False
        result = self.call()
        self.load_video.assert_called_once()
        self.load_audio.assert_not_called()
        self.writer.assert_not_called()
        self.assertNotIn("postprocess", self.events)
        self.assertEqual(result["distributed_video"], {"rank": 1})
        self.assertEqual(result["total_seconds"], 1.25)
        self.assertIsNone(self.video_refs[0]())
        self.assert_no_media()

    def test_local_unreleased_and_missing_controller_guards(self):
        for model in (NS(), NS(weight_shards=None), NS(weight_shards=NS(released=False)),
                      NS(weight_shards=NS())):
            with self.subTest(model=model):
                self.model = model
                with self.assertRaisesRegex(RuntimeError, "decoder preflight failed: rank 0"):
                    self.call()
                self.resolve.assert_not_called()
                self.load_video.assert_not_called()
                self.load_audio.assert_not_called()
                self.writer.assert_not_called()
        self.assert_no_media()

    def test_existing_partial_is_preserved_before_loading(self):
        partial = self.out.with_name(self.out.stem + ".partial.mp4")
        partial.write_bytes(b"prior partial output")
        with self.assertRaisesRegex(RuntimeError, "refusing existing partial"):
            self.call()
        self.assertEqual(partial.read_bytes(), b"prior partial output")
        self.resolve.assert_not_called()
        self.load_video.assert_not_called()
        self.writer.assert_not_called()

    def test_remote_release_guard_stops_all_loading(self):
        for error in ("RuntimeError: denoiser not released", "AttributeError: missing released"):
            with self.subTest(error=error):
                self.gathers.clear()
                self.remote_errors[1] = error
                with self.assertRaisesRegex(RuntimeError, "decoder preflight failed: rank 1"):
                    self.call()
                self.resolve.assert_not_called()
                self.load_video.assert_not_called()
                self.writer.assert_not_called()

    def test_local_resolution_and_video_loading_fail_collectively(self):
        for name in ("resolve", "load_video"):
            with self.subTest(stage=name):
                self.gathers.clear()
                target = getattr(self, name)
                saved = target.side_effect
                target.side_effect = OSError("injected " + name)
                with self.assertRaisesRegex(RuntimeError, "video decoder loading failed: rank 0: OSError"):
                    self.call()
                target.side_effect = saved
                self.assertEqual(len(self.gathers), 2)
                self.assertNotIn("video_decode", self.events)
                self.assertEqual(self.events[-2:], ["gc", "cache"])
                self.load_audio.assert_not_called()
                self.writer.assert_not_called()

    def test_remote_load_error_frees_local_decoder_before_decode(self):
        self.remote_errors[2] = "OSError: remote decoder missing"
        with self.assertRaisesRegex(RuntimeError, "video decoder loading failed: rank 1"):
            self.call()
        self.assertIsNone(self.video_refs[0]())
        self.assertEqual(len(self.gathers), 2)
        self.assertNotIn("video_decode", self.events)
        self.assertEqual(self.events[-2:], ["gc", "cache"])
        self.load_audio.assert_not_called()

    def test_decode_error_cleans_without_recovery_collective(self):
        self.fail = "video_decode"
        with self.assertRaisesRegex(RuntimeError, "injected video_decode"):
            self.call()
        self.assertEqual(len(self.gathers), 3)
        self.assertEqual(self.broadcasts, [])
        self.assertEqual(self.events[-2:], ["gc", "cache"])
        self.assertIsNone(self.video_refs[0]())
        self.load_audio.assert_not_called()

    def test_postprocess_and_cache_errors_stop_audio_collectively(self):
        for stage in ("postprocess", "cache"):
            with self.subTest(stage=stage):
                self.fail = stage
                self.gathers.clear()
                with self.assertRaisesRegex(RuntimeError, "video postprocess/cleanup failed: rank 0"):
                    self.call()
                self.assertEqual(len(self.gathers), 4)
                self.assertTrue(all(ref() is None for ref in self.video_refs + self.raw_refs))
                self.assertEqual(self.events[-2:], ["gc", "cache"])
                self.load_audio.assert_not_called()
                self.writer.assert_not_called()
                self.assertEqual(self.broadcasts, [])
                self.assert_no_media()

    def test_remote_postprocess_cleanup_error_stops_audio(self):
        self.remote_errors[4] = "RuntimeError: remote cleanup failed"
        with self.assertRaisesRegex(RuntimeError, "video postprocess/cleanup failed: rank 1"):
            self.call()
        self.load_audio.assert_not_called()
        self.writer.assert_not_called()
        self.assertTrue(all(ref() is None for ref in self.video_refs + self.raw_refs))

    def test_audio_load_onload_decode_and_eviction_errors_broadcast_and_cleanup(self):
        for stage in ("audio_load", "audio_onload", "audio_decode", "audio_evict"):
            with self.subTest(stage=stage):
                self.fail = stage
                self.events.clear()
                with self.assertRaisesRegex(RuntimeError, "rank0 audio/write failed: RuntimeError: injected " + stage):
                    self.call()
                self.assertIn(stage, self.broadcasts[-1]["error"])
                self.assertEqual(self.events[-2:], ["gc", "cache"])
                if stage != "audio_load":
                    self.assertIn("audio_evict", self.events)
                if stage in ("audio_load", "audio_onload"):
                    self.assertNotIn("audio_decode", self.events)
                self.assertTrue(all(ref() is None for ref in self.audio_refs))
                self.writer.assert_not_called()
                self.assert_no_media()

    def test_writer_failure_removes_partial_and_broadcasts(self):
        self.fail = "writer"
        with self.assertRaisesRegex(RuntimeError, "rank0 audio/write failed: RuntimeError: injected writer"):
            self.call()
        self.writer.assert_called_once()
        self.assertIn("injected writer", self.broadcasts[-1]["error"])
        self.assert_no_media()

    def test_atomic_replace_failure_removes_partial(self):
        with patch.object(callback.os, "replace", side_effect=OSError("injected replace")):
            with self.assertRaisesRegex(RuntimeError, "rank0 audio/write failed: OSError: injected replace"):
                self.call()
        self.assert_no_media()

    def test_nonmain_receives_rank_zero_audio_or_writer_failure(self):
        self.runtime.rank, self.runtime.is_main = 1, False
        self.remote_status = {"error": "OSError: rank zero writer failed"}
        with self.assertRaisesRegex(RuntimeError, "rank0 audio/write failed: OSError: rank zero writer failed"):
            self.call()
        self.load_audio.assert_not_called()
        self.writer.assert_not_called()
        self.assertIsNone(self.video_refs[-1]())

    def test_existing_output_is_preserved_before_loading(self):
        self.out.write_bytes(b"existing")
        with self.assertRaisesRegex(RuntimeError, "decoder preflight failed: rank 0: FileExistsError"):
            self.call()
        self.assertEqual(self.out.read_bytes(), b"existing")
        self.resolve.assert_not_called()
        self.load_video.assert_not_called()
        self.load_audio.assert_not_called()
        self.writer.assert_not_called()

    def test_writer_rechecks_output_created_after_preflight(self):
        self.fail = "output_race"
        with self.assertRaisesRegex(RuntimeError, "rank0 audio/write failed: FileExistsError"):
            self.call()
        self.assertEqual(self.out.read_bytes(), b"existing")
        self.assertFalse(self.partial.exists())
        self.writer.assert_not_called()

    def test_default_vae_source_is_resolved_locally(self):
        self.cfg.vae_source = None
        self.call()
        self.resolve.assert_called_once_with(callback.H3_BASE, local_files_only=True)


class DecoderToken:
    """Weak-referenceable fake video decoder, with no tensor or device allocation."""


class PostprocessTests(unittest.TestCase):
    def test_matches_real_render_arithmetic_for_fp16_and_fp32(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                raw = torch.linspace(-5, 5, 72, dtype=dtype).reshape(1, 3, 2, 3, 4)
                vae = NS(config=NS(latents_mean=[0, 0, 0], latents_std=[1, 1, 1]),
                         decode=Mock(return_value=(raw,)))
                with patch.object(render.torch, "autocast", return_value=nullcontext()):
                    expected = render._decode_video(torch.zeros_like(raw), vae, "cpu")
                actual = callback._postprocess_video(raw, "cpu")
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.uint8)
                self.assertEqual(actual.device.type, "cpu")
                self.assertEqual(tuple(actual.shape), (2, 3, 4, 3))
                self.assertEqual((actual.min().item(), actual.max().item()), (0, 255))


class UlyssesIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.cfg = NS(memory=NS(policy="resident_weight_shards"), parallel=NS(softmax_ranks=0),
                      render=NS(out=str(Path(root) / "out.mp4"), save_latents=False,
                                prompt_file="unused.pt", num_frames=2, num_steps=1, seed=0,
                                video_shift=12.0, audio_shift=3.0, warmup_steps=0,
                                denoise_only=False, record=True))
        self.runtime = NS(device=torch.device("cpu"), rank=0, world_size=2, is_main=True,
                          standard_exchange="alltoall",
                          branch_parallel=False, backend="mock", splits=(1, 1), heads_per_rank=1,
                          softmax_ranks=0, softmax_head_splits=(), linear_head_splits=(),
                          profile_enabled=False, barrier=Mock(), reset_profile=Mock(),
                          profile_milliseconds=Mock(return_value={}))
        self.controller = NS(released=False, summary=Mock(return_value={}))
        self.model = NS(weight_shards=self.controller, is_hybrid=True, transformer=object(),
                        startup_metrics={}, vae=object(), audio_vae=object())
        self.video, self.audio = torch.zeros(1), torch.zeros(1)
        self.events = []

        def release(model, runtime):
            self.assertIs(model, self.model)
            self.assertIs(runtime, self.runtime)
            self.controller.released = True
            self.events.append("release")

        def sample(*args, **kwargs):
            self.assertFalse(self.controller.released)
            self.events.append("sample")
            return self.video, self.audio

        dependencies = dict(build_inference_model=Mock(return_value=self.model),
                            install_ulysses=Mock(), load_prompt=Mock(return_value=(None, None, None)),
                            generate_latents=Mock(side_effect=sample),
                            release_sharded_denoiser=Mock(side_effect=release),
                            decode_and_save=Mock(), render_record=Mock(return_value={}), write_json=Mock())
        self.mocks = dependencies
        for name, value in dependencies.items():
            self.stack.enter_context(patch.object(infer_ulysses, name, value))
        for name in ("synchronize", "empty_cache", "reset_peak_memory_stats",
                     "max_memory_allocated", "max_memory_reserved"):
            self.stack.enter_context(patch.object(torch.cuda, name, return_value=0))
        self.stack.enter_context(patch.object(infer_ulysses.dist, "broadcast_object_list"))
        self.stack.enter_context(patch.object(infer_ulysses.dist, "all_reduce"))
        self.stack.enter_context(patch.object(infer_ulysses.dist, "all_gather_object",
                                             side_effect=lambda values, value: values.__setitem__(slice(None), [value] * 2)))
        # _run disables gradients globally; restore the caller's setting on cleanup.
        enabled = torch.is_grad_enabled()
        self.addCleanup(torch.set_grad_enabled, enabled)

    def invoke(self, decoder=None):
        infer_ulysses._run(self.cfg, self.runtime, 0.0, decoder=decoder)

    def decoder(self, result):
        def call(cfg, model, runtime, video, audio):
            self.assertTrue(self.controller.released)
            self.assertEqual(self.events, ["sample", "release"])
            for actual, expected in zip((cfg, model, runtime, video, audio),
                                        (self.cfg, self.model, self.runtime, self.video, self.audio)):
                self.assertIs(actual, expected)
            self.events.append("callback")
            return result
        return Mock(side_effect=call)

    def test_default_policy_callback_runs_after_release_and_records_metrics(self):
        result = {"distributed_video": {"rank": 0}, "total_seconds": 2.0}
        decoder = self.decoder(result)
        with patch.object(callback, "distributed_decode_callback", decoder):
            self.invoke()
        decoder.assert_called_once()
        self.mocks["build_inference_model"].assert_called_once_with(
            self.cfg, self.runtime.device, load_decoders=False, log=True, runtime=self.runtime)
        record = self.mocks["write_json"].call_args.args[0]
        self.assertIs(record["timings"]["decoder_metrics"], result)
        self.mocks["decode_and_save"].assert_not_called()

    def test_explicit_callback_none_result_is_accepted(self):
        decoder = self.decoder(None)
        self.invoke(decoder)
        decoder.assert_called_once()
        record = self.mocks["write_json"].call_args.args[0]
        self.assertNotIn("decoder_metrics", record["timings"])
        self.mocks["decode_and_save"].assert_not_called()

    def test_nonmain_also_calls_decoder_after_release(self):
        self.runtime.rank, self.runtime.is_main = 1, False
        decoder = self.decoder({"distributed_video": {"rank": 1}})
        self.invoke(decoder)
        decoder.assert_called_once()
        self.mocks["write_json"].assert_not_called()
        self.mocks["decode_and_save"].assert_not_called()

    def test_denoise_only_releases_but_skips_callback_selection(self):
        self.cfg.render.denoise_only = True
        decoder = Mock()
        with patch.object(infer_ulysses, "_select_decoder") as select:
            self.invoke(decoder)
        select.assert_not_called()
        decoder.assert_not_called()
        self.assertEqual(self.events, ["sample", "release"])
        self.mocks["decode_and_save"].assert_not_called()

    def test_legacy_decode_remains_rank_zero_only_without_release(self):
        self.cfg.memory.policy = "legacy"
        for main in (True, False):
            with self.subTest(is_main=main):
                self.runtime.is_main, self.runtime.rank = main, int(not main)
                self.invoke()
        self.mocks["release_sharded_denoiser"].assert_not_called()
        self.mocks["decode_and_save"].assert_called_once_with(
            self.video, self.audio, self.model.vae, self.model.audio_vae,
            self.cfg.render.out, "cpu", sequential=False, offload=self.controller)
        self.assertEqual([c.kwargs["load_decoders"] for c in
                          self.mocks["build_inference_model"].call_args_list], [True, False])

    def test_selector_default_explicit_override_and_legacy(self):
        explicit = Mock()
        self.assertIs(infer_ulysses._select_decoder(self.cfg), callback.distributed_decode_callback)
        self.assertIs(infer_ulysses._select_decoder(self.cfg, explicit), explicit)
        self.cfg.memory.policy = "legacy"
        self.assertIsNone(infer_ulysses._select_decoder(self.cfg))
        self.assertIs(infer_ulysses._select_decoder(self.cfg, explicit), explicit)

    def test_nsight_capture_windows_all_ranks_and_warmup_excluded(self):
        from src.inference.utils import nsight

        self.cfg.render.warmup_steps = 4
        self.cfg.render.num_steps = 8
        events = []
        fake_cuda = NS(
            profiler=NS(start=lambda: events.append("start"),
                        stop=lambda: events.append("stop")),
            synchronize=lambda device: events.append("sync"),
        )

        def sample(*args, **kwargs):
            capture = kwargs["nsight_capture"]
            if capture is None:
                events.append("warmup")
            else:
                for step in range(8):
                    with capture.step(step, last=step == 7):
                        events.append(f"step{step}")
            return self.video, self.audio

        self.mocks["generate_latents"].side_effect = sample
        self.runtime.barrier.side_effect = lambda: events.append("barrier")
        with patch.object(nsight, "CAPTURE_ENABLED", True), \
                patch.object(nsight, "NVTX_ENABLED", False), \
                patch.object(nsight, "_cuda", return_value=fake_cuda):
            for rank in range(8):
                with self.subTest(rank=rank):
                    events.clear()
                    self.runtime.rank, self.runtime.is_main = rank, rank == 0
                    self.controller.released = False
                    self.invoke(Mock(side_effect=lambda *args: events.append("decode")))
                    self.assertEqual(events, [
                        "barrier", "warmup", "step0", "step1", "start", "step2",
                        "step3", "sync", "stop", "step4", "step5", "step6", "step7",
                        "start", "decode", "barrier", "sync", "stop",
                    ])


if __name__ == "__main__":
    unittest.main()
