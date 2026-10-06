"""CPU-only tests of pinned H3 geometry, arithmetic, loading, and Gloo transport."""

from datetime import timedelta
import json
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import AutoencoderKLMiniMaxH3
from safetensors import safe_open
from safetensors.torch import save_file

from src.inference.utils.distributed_decode import (
    distributed_decode, load_video_decoder, plan_decode, stitch_clip,
)


class FakeDecoder(nn.Module):
    """Noncausal, tile-local context makes incorrect splitting observable."""

    def __init__(self, ratio=2, half=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.875))
        self.ratio = ratio
        self.half = half
        self.fail = False
        self.calls = []

    def forward(self, z):
        if self.fail:
            raise RuntimeError("deliberate decoder failure")
        self.calls.append(tuple(z.shape))
        # Depend on every frame/pixel in this exact clip/tile, not independent frames.
        value = z * self.weight + z.mean(dim=(2, 3, 4), keepdim=True) * 0.137
        value = value.repeat_interleave(4, 2).repeat_interleave(self.ratio, 3).repeat_interleave(self.ratio, 4)
        t = torch.arange(value.shape[2], dtype=torch.float32, device=value.device).view(1, 1, -1, 1, 1)
        value = value + t * 0.013
        return value.half() if self.half else value


class FakeVAE(nn.Module):
    # Invoke the actual installed functions, not copied reference implementations.
    _split_tiles = AutoencoderKLMiniMaxH3._split_tiles
    _blend = AutoencoderKLMiniMaxH3._blend
    _stitch_tiles = AutoencoderKLMiniMaxH3._stitch_tiles
    _decode_clip = AutoencoderKLMiniMaxH3._decode_clip
    _decode = AutoencoderKLMiniMaxH3._decode

    def __init__(self, ratio=2, half=False):
        super().__init__()
        self.config = SimpleNamespace(
            latent_channels=2, out_channels=2, clip_length=17, token_drop=3,
            latents_mean=(0.75, -0.25), latents_std=(1.5, 0.625),
        )
        self.spatial_compression_ratio = ratio
        self.temporal_compression_ratio = 4
        self.tokens_chunk_size = 5
        self.token_overlap = 2
        self.frame_pre_padding = 3
        self.frame_overlap = 5
        self.use_tiling = True
        self.use_slicing = False
        self.tile_sample_min_height = 8 * ratio
        self.tile_sample_min_width = 8 * ratio
        self.tile_sample_min_overlap_height = 2 * ratio
        self.tile_sample_min_overlap_width = 2 * ratio
        self.post_quant_conv = nn.Conv3d(2, 2, 1)
        with torch.no_grad():
            self.post_quant_conv.weight.copy_(torch.tensor([0.9, 0.2, -0.1, 1.2]).view(2, 2, 1, 1, 1))
            self.post_quant_conv.bias.copy_(torch.tensor([0.1, -0.2]))
        self.decoder = FakeDecoder(ratio, half)
        self.eval()


def reference(vae, z, normalize=True):
    with torch.no_grad():
        if normalize:
            mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
            z = z * std + mean
        return vae._decode(z)


def tiny_vae():
    return AutoencoderKLMiniMaxH3(
        latent_channels=2, in_channels=2, out_channels=2,
        block_out_channels=(4,), layers_per_block=1,
        spatial_downsample_factors=(2,), temporal_downsample_factors=(4,),
        norm_num_groups=1, decoder_num_layers=1,
        decoder_num_attention_heads=1, decoder_attention_head_dim=8,
        decoder_num_register_tokens=1, decoder_ffn_mult=1,
        latents_mean=(0.75, -0.25), latents_std=(1.5, 0.625),
    ).eval()


def write_checkpoint(root, vae, indexed):
    folder = Path(root) / "vae"
    folder.mkdir()
    vae.save_config(folder)
    state = vae.state_dict()
    if indexed:
        names = sorted(state)
        weight_map = {}
        for i in range(2):
            filename = f"part-{i}.safetensors"
            shard = {key: state[key].contiguous() for key in names[i::2]}
            save_file(shard, folder / filename)
            weight_map.update({key: filename for key in shard})
        (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file(state, folder / "diffusion_pytorch_model.safetensors")
    return folder


def gloo_worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world, timeout=timedelta(seconds=30))
    try:
        vae = FakeVAE()
        # Repeated calls, rank-zero only result, exact reference.
        z = torch.arange(2 * 13 * 11 * 17, dtype=torch.float32).reshape(1, 2, 13, 11, 17) / 1000
        expected = reference(vae, z)
        for _ in range(2):
            output, metrics = distributed_decode(vae, z, device="cpu")
            assert metrics["rank_job_counts"] == [6, 6, 6], metrics
            assert metrics["ranks"][0]["gathered_tiles_peak"] == 6
            if rank == 0:
                assert torch.equal(output, expected)
            else:
                assert output is None
            assert next(vae.decoder.parameters()).device.type == "cpu"
        uneven = z[:, :, :8, :11, :11]
        output, metrics = distributed_decode(vae, uneven, device="cpu")
        assert metrics["rank_job_counts"] == [3, 3, 2]
        if rank == 0:
            assert torch.equal(output, reference(vae, uneven))
        # One job across three ranks: two ranks have no work, still converge.
        small = z[:, :, :7, :2, :2]
        output, metrics = distributed_decode(vae, small, device="cpu")
        assert metrics["rank_job_counts"] == [1, 0, 0]
        if rank == 0:
            assert torch.equal(output, reference(vae, small))
        # Group rank zero is global rank one, not the default-group root.
        subgroup = dist.new_group([1, 2], backend="gloo", timeout=timedelta(seconds=30))
        if rank in (1, 2):
            output, metrics = distributed_decode(vae, z, device="cpu", group=subgroup)
            assert metrics["rank"] == rank - 1
            assert metrics["rank_job_counts"] == [9, 9]
            assert (output is not None) == (rank == 1)
            if rank == 1:
                assert torch.equal(output, expected)
        dist.barrier()
        # A preflight error on only one rank must not strand the other ranks.
        try:
            distributed_decode(vae, z.half() if rank == 1 else z, device="cpu")
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("collective validation did not fail")
        # Valid but different plans/options must also fail before P2P.
        try:
            distributed_decode(vae, z, device="cpu", normalize_latents=rank != 1)
        except ValueError:
            pass
        else:
            raise AssertionError("plan signature mismatch did not fail")
        # Decoder failure propagates at the clip boundary and remains repeat safe.
        vae.decoder.fail = rank == 2
        try:
            distributed_decode(vae, z, device="cpu")
        except RuntimeError:
            pass
        else:
            raise AssertionError("remote decoder failure did not propagate")
        vae.decoder.fail = False
        output, _ = distributed_decode(vae, small, device="cpu")
        if rank == 0:
            assert torch.equal(output, reference(vae, small))
    finally:
        dist.destroy_process_group()


class DistributedDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_production_job_counts_and_stable_ids(self):
        vae = FakeVAE(ratio=16)
        vae.tile_sample_min_height = vae.tile_sample_min_width = 256
        vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 64
        for tokens, frames, total in ((37, 124, 196), (102, 345, 560)):
            plan = plan_decode(vae, (1, 2, tokens, 48, 84))
            self.assertEqual((plan.tiles_per_clip, len(plan.jobs), plan.output_frames), (28, total, frames))
            self.assertEqual([job.id for job in plan.jobs], list(range(total)))
            self.assertEqual([job.start for job in plan.clip_jobs(1)], [5] * 28)
            counts = [sum(job.id % 8 == rank for job in plan.jobs) for rank in range(8)]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_random_completion_order_unequal_overlap_and_dtype(self):
        vae = FakeVAE()
        # 34 pixels covered by three 16-pixel tiles: unequal 8/6 overlaps.
        plan = plan_decode(vae, (1, 2, 7, 17, 17))
        self.assertEqual(plan.height_overlaps, (8, 6))
        self.assertEqual(plan.width_overlaps, (8, 6))
        for dtype in (torch.float16, torch.float32):
            generator = torch.Generator().manual_seed(731)
            tiles = [torch.randn(1, 2, 28, 16, 16, generator=generator).to(dtype) for _ in plan.jobs]
            original = [tile.clone() for tile in tiles]
            expected = vae._stitch_tiles([tiles[:3], tiles[3:6], tiles[6:]], [8, 6], [8, 6])
            for seed in range(8):
                completed = list(enumerate(tiles))
                random.Random(seed).shuffle(completed)
                result = stitch_clip(vae, plan, 0, completed)
                self.assertTrue(torch.equal(result, expected))
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape[-2:], (34, 34))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(tiles, original)))

    def test_temporal_ramps_padding_and_trimming_exact_reference(self):
        for half in (False, True):
            vae = FakeVAE(half=half)
            for frames in (*range(3, 19), 37, 102):
                with self.subTest(half=half, frames=frames):
                    z = torch.arange(frames, dtype=torch.float32).view(1, 1, -1, 1, 1).expand(1, 2, frames, 2, 3)
                    original = z.clone()
                    expected = reference(vae, z)
                    actual, metrics = distributed_decode(vae, z, device="cpu")
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(z, original))
                    self.assertEqual(actual.shape[2], metrics["output_frames"])
                    self.assertTrue(all(shape[2] == 7 for shape in vae.decoder.calls))
                    self.assertEqual(metrics["rank_job_counts"], [metrics["clips"]])

    def test_spatiotemporal_repeat_batch_and_output_options(self):
        torch.manual_seed(2026)
        vae = FakeVAE(half=True)
        z = torch.randn(2, 2, 13, 11, 17)
        for normalize in (True, False):
            expected = reference(vae, z, normalize)
            for output_cpu in (True, False):
                for managed in (True, False):
                    actual, metrics = distributed_decode(
                        vae, z, device="cpu", normalize_latents=normalize,
                        output_cpu=output_cpu, manage_lifecycle=managed,
                    )
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.dtype, torch.float16)
                    self.assertEqual(metrics["ranks"][0]["gathered_tiles_peak"], 6)
                    self.assertEqual(metrics["total_jobs"], 18)
                    self.assertEqual(metrics["ranks"][0]["peak_allocated_bytes"], 0)
                    for key, value in metrics["ranks"][0].items():
                        if key.endswith("_seconds"):
                            self.assertGreaterEqual(value, 0)
        self.assertFalse(hasattr(vae, "_distributed_decode_state"))

    def test_installed_tiny_real_decoder_exact_reference(self):
        torch.manual_seed(11)
        vae = tiny_vae()
        vae.enable_tiling(8, 8, 2, 2)
        z = torch.randn(1, 2, 8, 5, 7)
        actual, _ = distributed_decode(vae, z, device="cpu")
        self.assertTrue(torch.equal(actual, reference(vae, z)))

    def test_validation(self):
        z = torch.zeros(1, 2, 7, 3, 3)
        vae = FakeVAE()
        for bad in (z.half(), z.long(), z[0], z[:, :, :2], z[:, :1], z[:, :, :0], None):
            with self.subTest(bad=None if bad is None else (bad.shape, bad.dtype)):
                with self.assertRaises(ValueError):
                    distributed_decode(vae, bad, device="cpu")
        for name, value in (("use_tiling", False), ("frame_overlap", 4),
                            ("tile_sample_min_height", 3), ("tile_sample_min_overlap_width", 16)):
            with patch.object(vae, name, value), self.assertRaises(ValueError):
                distributed_decode(vae, z, device="cpu")
        with patch.object(vae, "use_slicing", True), self.assertRaises(ValueError):
            distributed_decode(vae, z.expand(2, -1, -1, -1, -1), device="cpu")
        vae.train()
        with self.assertRaises(ValueError):
            distributed_decode(vae, z, device="cpu")
        vae.eval().half()
        with self.assertRaises(ValueError):
            distributed_decode(vae, z, device="cpu")
        with self.assertRaises(ValueError):
            distributed_decode(FakeVAE(), z, device="cpu", group=object())

    def test_stitch_rejects_missing_duplicate_wrong_ids_and_dtypes(self):
        vae = FakeVAE()
        plan = plan_decode(vae, (1, 2, 7, 11, 11))
        completed = [(job.id, torch.zeros(1, 2, 28, 16, 16)) for job in plan.jobs]
        for bad in (completed[:-1], completed + completed[:1], [(999, completed[0][1])] + completed[1:],
                    [(0, completed[0][1].half())] + completed[1:]):
            with self.assertRaises(ValueError):
                stitch_clip(vae, plan, 0, bad)
        with self.assertRaises(ValueError):
            stitch_clip(vae, plan, 99, completed)

    def test_failure_evicts_and_next_call_is_clean(self):
        vae = FakeVAE()
        z = torch.zeros(1, 2, 7, 3, 3)
        vae.decoder.fail = True
        with patch.object(vae.decoder, "to", wraps=vae.decoder.to) as move:
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                distributed_decode(vae, z, device="cpu")
            self.assertEqual(move.call_args_list[-1].args, ("cpu",))
        vae.decoder.fail = False
        actual, _ = distributed_decode(vae, z, device="cpu")
        self.assertTrue(torch.equal(actual, reference(vae, z)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_backing_restore_repeat_and_failure(self):
        vae = FakeVAE(half=True)
        z = torch.randn(1, 2, 12, 11, 11)
        original = {name: (p.data_ptr(), p.detach().clone()) for name, p in vae.named_parameters()}
        first = None
        for fail in (False, True, False):
            vae.decoder.fail = fail
            if fail:
                with self.assertRaisesRegex(RuntimeError, "deliberate"):
                    distributed_decode(vae, z, device="cuda:0")
            else:
                actual, _ = distributed_decode(vae, z, device="cuda:0")
                if first is None:
                    first = actual
                else:
                    self.assertTrue(torch.equal(first, actual))
            for name, parameter in vae.named_parameters():
                self.assertEqual(parameter.device.type, "cpu")
                self.assertEqual(parameter.data_ptr(), original[name][0])
                self.assertTrue(torch.equal(parameter, original[name][1]))

    def test_loader_local_cpu_decoder_only_and_rope(self):
        for indexed in (False, True):
            with self.subTest(indexed=indexed), tempfile.TemporaryDirectory() as root:
                vae = tiny_vae()
                write_checkpoint(root, vae, indexed)
                read_keys = []

                class TrackingFile:
                    def __init__(self, *args, **kwargs):
                        self.file = safe_open(*args, **kwargs)

                    def __enter__(self):
                        self.file.__enter__()
                        return self

                    def __exit__(self, *args):
                        return self.file.__exit__(*args)

                    def keys(self):
                        return self.file.keys()

                    def get_slice(self, key):
                        return self.file.get_slice(key)

                    def get_tensor(self, key):
                        read_keys.append(key)
                        return self.file.get_tensor(key)

                with patch("safetensors.safe_open", TrackingFile):
                    loaded = load_video_decoder(root)
                self.assertTrue(read_keys)
                self.assertTrue(all(key.startswith(("decoder.", "post_quant_conv.")) for key in read_keys))
                self.assertIsNone(loaded.encoder)
                self.assertIsNone(loaded.quant_conv)
                self.assertTrue(loaded.use_tiling)
                self.assertFalse(loaded.training)
                for key, tensor in loaded.state_dict().items():
                    self.assertTrue(torch.equal(tensor, vae.state_dict()[key]))
                    self.assertEqual((tensor.device.type, tensor.dtype), ("cpu", torch.float32))
                self.assertTrue(torch.equal(loaded.decoder.rope.inv_freq, vae.decoder.rope.inv_freq))
                self.assertFalse(any(t.is_meta for t in (*loaded.parameters(), *loaded.buffers())))
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                z = torch.randn(1, 2, 7, 2, 3)
                actual, _ = distributed_decode(loaded, z, device="cpu")
                self.assertTrue(torch.equal(actual, reference(vae, z)))

    def test_loader_rejects_missing_local_and_bad_decoder_keys_shapes(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                load_video_decoder(root)
            vae = tiny_vae()
            folder = write_checkpoint(root, vae, False)
            original = vae.state_dict()
            key = "post_quant_conv.weight"
            for kind in ("missing", "shape", "extra"):
                state = dict(original)
                if kind == "missing":
                    del state[key]
                elif kind == "shape":
                    state[key] = torch.zeros(1)
                else:
                    state["decoder.unexpected"] = torch.zeros(1)
                save_file(state, folder / "diffusion_pytorch_model.safetensors")
                with self.assertRaises(ValueError):
                    load_video_decoder(root)

    def test_gloo_three_ranks_subgroup_and_collective_failures(self):
        with tempfile.TemporaryDirectory() as root:
            context = mp.start_processes(gloo_worker, args=(3, str(Path(root) / "rendezvous")),
                                         nprocs=3, join=False, start_method="spawn")
            deadline = time.monotonic() + 90
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("CPU distributed decode test timed out")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
