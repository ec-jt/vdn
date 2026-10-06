# Temporal orchestration adapted from Diffusers autoencoder_kl_minimax_h3.py.
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Modifications: independent job planning, distributed transport and lifecycle.

"""Exact, clip-windowed distribution of the pinned MiniMax H3 video decode.

This is not temporal sequence parallelism: a job is one *existing* spatial tile
of one seven-latent, noncausal clip. No attention/cache state crosses jobs. The
installed VAE owns spatial split/blend/stitch arithmetic. Temporal orchestration
mirrors its ``_decode`` (stride 5, raw 28, main 17, tail 5).

All ranks must call with identical latents/configuration and the same process
group. Rank zero means group-relative rank zero. Only small control/metric
tensors are all-gathered; decoded tiles use point-to-point sends to that rank.
Gather storage is bounded by one clip, regardless of total video duration.
Concurrent calls on the same VAE/group are not supported. Distributed transport
failures require the caller's usual process-group abort/timeout handling.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
import torch.distributed as dist

from src.inference.utils.nsight import nvtx_annotate, nvtx_range


@dataclass(frozen=True)
class DecodeJob:
    """Stable row-major global ID, independent of rank count/completion order."""

    id: int
    clip: int
    row: int
    column: int
    start: int
    y: int
    x: int
    height: int
    width: int


@dataclass(frozen=True)
class DecodePlan:
    jobs: tuple[DecodeJob, ...]
    rows: int
    columns: int
    clips: int
    pad_tokens: int
    trim_frames: int
    output_frames: int
    height_overlaps: tuple[int, ...]
    width_overlaps: tuple[int, ...]

    @property
    def tiles_per_clip(self):
        return self.rows * self.columns

    def clip_jobs(self, clip):
        start = clip * self.tiles_per_clip
        return self.jobs[start : start + self.tiles_per_clip]


def load_video_decoder(base_model_path):
    """Load local ``base_model_path/vae`` as an FP32, CPU, decode-only H3 VAE.

    No Hub lookup or pickle fallback. Build on meta, omit encoder/quant_conv
    payloads, strictly check decoder keys/shapes, and reconstruct the nonpersistent
    RoPE buffer on CPU. Supports the installed unsharded and indexed safetensors
    layouts. The returned VAE supports decode but deliberately cannot encode.
    Loading is outside ``distributed_decode`` timing; callers should time it
    separately. Each rank loads its own CPU decoder before the collective call.
    """
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed
    from safetensors import safe_open

    folder = Path(base_model_path) / "vae"
    if not folder.is_dir():
        raise FileNotFoundError(f"Local VAE directory not found: {folder}")
    config = json.loads((folder / "config.json").read_text())
    if config.get("_class_name", "AutoencoderKLMiniMaxH3") != "AutoencoderKLMiniMaxH3":
        raise ValueError("Expected an AutoencoderKLMiniMaxH3 checkpoint")
    if config.get("quantization_config") is not None:
        raise ValueError("Quantized decoder checkpoints are not supported")
    with torch.device("meta"):
        vae = AutoencoderKLMiniMaxH3.from_config(config)
    vae.encoder = None
    vae.quant_conv = None
    cfg = vae.config
    with torch.device("cpu"):
        vae.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(
            int(cfg.decoder_attention_head_dim * cfg.decoder_rope_dim_ratio),
            theta=cfg.decoder_rope_theta,
        )
    expected = vae.state_dict()
    index = folder / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(folder / "diffusion_pytorch_model.safetensors", framework="pt", device="cpu") as f:
            weight_map = {key: "diffusion_pytorch_model.safetensors" for key in f.keys()}
    selected = {k: v for k, v in weight_map.items() if k.startswith(("decoder.", "post_quant_conv."))}
    if selected.keys() != expected.keys():
        raise ValueError(
            f"Decoder checkpoint keys mismatch: missing={sorted(expected.keys() - selected.keys())}, "
            f"unexpected={sorted(selected.keys() - expected.keys())}"
        )
    state = {}
    for filename in sorted(set(selected.values())):
        path = (folder / filename).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError(f"Checkpoint shard escapes the local VAE directory: {filename}")
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in sorted(k for k, shard in selected.items() if shard == filename):
                if key not in f.keys() or tuple(f.get_slice(key).get_shape()) != tuple(expected[key].shape):
                    raise ValueError(f"Decoder checkpoint shape/key mismatch: {key}")
                tensor = f.get_tensor(key)
                if not tensor.is_floating_point():
                    raise ValueError(f"Non-floating decoder weight: {key}")
                state[key] = tensor.to(dtype=torch.float32)
    vae.load_state_dict(state, strict=True, assign=True)
    if any(t.is_meta for t in (*vae.parameters(), *vae.buffers())):
        raise ValueError("Decoder has unmaterialized meta tensors")
    vae.requires_grad_(False).eval()
    vae.enable_tiling()
    return vae


def plan_decode(vae, shape) -> DecodePlan:
    """Plan the installed tiling exactly, without touching tensors or devices."""
    if len(shape) != 5 or any(not isinstance(n, int) or n <= 0 for n in shape):
        raise ValueError("latents must have nonempty (B, C, T, H, W) dimensions")
    if shape[1] != vae.config.latent_channels:
        raise ValueError("latent channel count does not match the VAE")
    if shape[2] < 3:
        raise ValueError("Pinned video decode needs at least three latent frames")
    if vae.use_slicing and shape[0] > 1:
        raise ValueError("Batch slicing is unsupported; call separately for each video")
    geometry = (
        vae.config.clip_length, vae.config.token_drop, vae.tokens_chunk_size,
        vae.token_overlap, vae.frame_pre_padding, vae.frame_overlap, vae.temporal_compression_ratio,
    )
    if geometry != (17, 3, 5, 2, 3, 5, 4):
        raise ValueError(f"Unsupported temporal geometry: {geometry}")
    if not vae.use_tiling:
        raise ValueError("Installed H3 spatial tiling must remain enabled")
    ratio = vae.spatial_compression_ratio
    if not isinstance(ratio, int) or ratio <= 0:
        raise ValueError("Invalid spatial compression ratio")
    axes = []
    for length, tile, overlap in (
        (shape[3] * ratio, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height),
        (shape[4] * ratio, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width),
    ):
        if not (0 < overlap < tile) or tile % ratio or overlap % ratio:
            raise ValueError("Tile sizes/overlaps must be positive, latent-aligned, with overlap < tile")
        axes.append(vae._split_tiles(length, tile, overlap))
    (ys, hs, yo), (xs, ws, xo) = axes
    pad = (-(shape[2] + 3)) % 5
    clips = (shape[2] + 3 + pad) // 5 - 1
    trim = sum(1 if (shape[2] + k) % 5 == 0 else 4 for k in range(pad))
    jobs = []
    for clip in range(clips):
        for row, (y, h) in enumerate(zip(ys, hs)):
            for column, (x, w) in enumerate(zip(xs, ws)):
                jobs.append(DecodeJob(len(jobs), clip, row, column, clip * 5,
                                      y // ratio, x // ratio, h // ratio, w // ratio))
    return DecodePlan(tuple(jobs), len(ys), len(xs), clips, pad, trim,
                      clips * 17 + 5 - trim, tuple(yo), tuple(xo))


@nvtx_annotate("decode.stitch.spatial")
def stitch_clip(vae, plan: DecodePlan, clip: int, completed) -> torch.Tensor:
    """Restore spatial order from ``(job_id, tensor)`` completions, then stitch.

    Do not blend against already-blended neighbors. The installed method uses
    original tiles, vertical THEN horizontal blending, then crops and concatenates.
    No cast or device transfer is permitted before that arithmetic.
    """
    if not 0 <= clip < plan.clips:
        raise ValueError("Invalid clip index")
    jobs = plan.clip_jobs(clip)
    expected = {job.id for job in jobs}
    tiles = {}
    for job_id, tensor in completed:
        if job_id not in expected or job_id in tiles:
            raise ValueError(f"Unexpected or duplicate tile ID: {job_id}")
        tiles[job_id] = tensor
    if tiles.keys() != expected:
        raise ValueError("Missing completed tiles")
    ordered = [tiles[job.id] for job in jobs]
    if any(t.dtype != ordered[0].dtype or t.device != ordered[0].device for t in ordered):
        raise ValueError("Tile dtype/device mismatch")
    rows = [ordered[i : i + plan.columns] for i in range(0, len(ordered), plan.columns)]
    return vae._stitch_tiles(rows, list(plan.height_overlaps), list(plan.width_overlaps))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _raise_collectively(error, device, group, world):
    """Converge predictable local failures before entering tile transfers."""
    if world > 1:
        flag = torch.tensor([int(error is not None)], dtype=torch.int64, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        failed = bool(flag.item())
    else:
        failed = error is not None
    if failed:
        if error is not None:
            raise error
        raise RuntimeError("Distributed video decode failed on another rank")


@torch.no_grad()
@nvtx_annotate("decode.distributed_video")
def distributed_decode(
    vae, latents, *, device, group=None, normalize_latents=True,
    output_cpu=True, manage_lifecycle=True, reset_peak_memory=False,
):
    """Return ``(decoded_or_None, metrics)`` on all participating ranks.

    Input defaults to the normalized FP32 latents returned by generate_latents.
    Set normalize_latents=False for already denormalized VAE inputs. Output is
    raw ImageNet-normalized RGB, B,C,T,H,W, exactly like vae.decode, NOT uint8.
    CUDA compute uses FP16 autocast over FP32 weights; stitching stays on that
    CUDA device in decoder output dtype. CPU mode (for testing) has no autocast.
    output_cpu=True copies finished temporal chunks after GPU blending, bounding
    GPU output storage as well as gather storage. Pixel postprocessing is caller
    owned; use output_cpu=False if it must run on CUDA as in render._decode_video.

    manage_lifecycle=True onloads ONLY decoder/post_quant_conv and restores both
    to CPU in finally, including failures. CPU-backed immutable tensors are retained
    during CUDA execution to avoid copying unchanged weights back over PCIe.
    False requires these already on device
    and leaves placement unchanged. No encoder, denoiser release, audio, file IO,
    barrier outside this group, or empty_cache is managed here. Times are synced
    wall seconds; comm includes waiting/control collectives. Allocator peaks are
    since the caller's last reset unless reset_peak_memory=True (which mutates the
    process CUDA peak counters). RSS is the process-lifetime high-water mark.
    Metrics include every rank, with rank-local timings and byte peaks.
    """
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/Gloo and CUDA/NCCL execution are supported")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    active = dist.is_available() and dist.is_initialized()
    if group is not None and not active:
        raise ValueError("A process group requires initialized torch.distributed")
    world = dist.get_world_size(group) if active else 1
    rank = dist.get_rank(group) if active else 0
    if rank < 0:
        raise ValueError("This process is not a member of the decode group")
    if world > 1:
        backend = str(dist.get_backend(group))
        if (device.type == "cuda" and backend != "nccl") or (device.type == "cpu" and backend != "gloo"):
            raise ValueError("Use NCCL for CUDA and Gloo for CPU decode")
    root = dist.get_global_rank(group, 0) if active and group is not None else 0
    error = None
    try:
        if not isinstance(latents, torch.Tensor) or not latents.is_floating_point() or latents.is_meta:
            raise ValueError("latents must be a materialized floating tensor")
        plan = plan_decode(vae, tuple(latents.shape))
        if latents.dtype != torch.float32:
            raise ValueError("Video latents must be FP32, matching the pinned render path")
        modules = (vae.post_quant_conv, vae.decoder)
        for module in modules:
            if module.training:
                raise ValueError("Decoder modules must be in eval mode")
            for tensor in (*module.parameters(), *module.buffers()):
                if tensor.is_meta or (tensor.is_floating_point() and tensor.dtype != torch.float32):
                    raise ValueError("Decoder weights/buffers must be materialized FP32")
                if not manage_lifecycle and tensor.device != device:
                    raise ValueError("Unmanaged decoder modules must already be on the requested device")
        if normalize_latents:
            if len(vae.config.latents_mean) != latents.shape[1] or len(vae.config.latents_std) != latents.shape[1]:
                raise ValueError("Invalid latent normalization channel count")
    except Exception as exc:
        error = exc
    _raise_collectively(error, device, group, world)
    if world > 1:
        # Metadata only, never video tensors. Catch different plans/options before P2P.
        signature = repr((plan, tuple(latents.shape), normalize_latents, output_cpu,
                          vae.config.out_channels, vae.config.latents_mean, vae.config.latents_std))
        digest = hashlib.sha256(signature.encode()).digest()
        check = torch.tensor(list(digest), dtype=torch.uint8, device=device)
        checks = [torch.empty_like(check) for _ in range(world)]
        dist.all_gather(checks, check, group=group)
        if any(not torch.equal(check, other) for other in checks):
            raise ValueError("Decode plans/options differ across ranks")

    _sync(device)
    started = time.perf_counter()
    if device.type == "cuda" and reset_peak_memory:
        torch.cuda.reset_peak_memory_stats(device)
    stats = {name: 0.0 for name in ("onload_seconds", "prepare_seconds", "decode_seconds",
                                   "comm_seconds", "stitch_seconds", "output_cpu_seconds", "evict_seconds")}
    stats.update(rank=rank, jobs=0, gathered_tiles_peak=0)
    output = None
    overlap = None
    completed = []
    local = {}
    z = None
    # Immutable CPU backing avoids copying roughly 9 GiB of unchanged weights
    # back over PCIe on every rank. Keep this call-local so repeat calls are clean.
    backing = []
    if manage_lifecycle and device.type == "cuda" and all(t.device.type == "cpu" for module in modules
                                for t in (*module.parameters(), *module.buffers())):
        for module in modules:
            for child in module.modules():
                for name, parameter in child._parameters.items():
                    if parameter is not None:
                        backing.append((child, name, True, parameter.detach()))
                for name, buffer in child._buffers.items():
                    if buffer is not None:
                        backing.append((child, name, False, buffer))
    try:
        t = time.perf_counter()
        error = None
        try:
            if manage_lifecycle:
                for module in modules:
                    module.to(device)
            _sync(device)
        except Exception as exc:
            error = exc
        stats["onload_seconds"] = time.perf_counter() - t
        _raise_collectively(error, device, group, world)
        t = time.perf_counter()
        error = None
        try:
            z = latents.to(device)
            if normalize_latents:
                mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
                std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
                z = z * std + mean
            if plan.pad_tokens:
                z = torch.cat([z, z[:, :, -1:].repeat(1, 1, plan.pad_tokens, 1, 1)], dim=2)
            _sync(device)
        except Exception as exc:
            error = exc
        stats["prepare_seconds"] = time.perf_counter() - t
        _raise_collectively(error, device, group, world)
        position = 0
        for clip in range(plan.clips):
            jobs = plan.clip_jobs(clip)
            t = time.perf_counter()
            error = None
            try:
                worker_range = ("decode.worker" if any(job.id % world == rank for job in jobs)
                                else "decode.idle_worker.no_jobs")
                with nvtx_range(worker_range), torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext():
                    for job in jobs:
                        if job.id % world != rank:
                            continue
                        with nvtx_range(f"decode.job.{job.id}.clip.{clip}"):
                            tile = z[:, :, job.start : job.start + 7,
                                     job.y : job.y + job.height, job.x : job.x + job.width]
                            decoded = vae.decoder(vae.post_quant_conv(tile)).contiguous()
                            expected_shape = (z.shape[0], vae.config.out_channels, 28,
                                              job.height * vae.spatial_compression_ratio,
                                              job.width * vae.spatial_compression_ratio)
                            if tuple(decoded.shape) != expected_shape:
                                raise ValueError("Decoder tile has unexpected shape")
                            if world > 1 and decoded.dtype != (torch.float16 if device.type == "cuda" else torch.float32):
                                raise ValueError("Unexpected distributed decoder output dtype")
                            local[job.id] = decoded
                            stats["jobs"] += 1
                            del decoded, tile
                _sync(device)
            except Exception as exc:
                error = exc
            stats["decode_seconds"] += time.perf_counter() - t
            t = time.perf_counter()
            _raise_collectively(error, device, group, world)
            operations = []
            send_tensors = []
            for job in jobs:
                owner = job.id % world
                if rank == 0:
                    if owner == 0:
                        tensor = local.pop(job.id)
                    else:
                        tensor = torch.empty(
                            (z.shape[0], vae.config.out_channels, 28,
                             job.height * vae.spatial_compression_ratio,
                             job.width * vae.spatial_compression_ratio),
                            device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32,
                        )
                        peer = dist.get_global_rank(group, owner) if group is not None else owner
                        operations.append(dist.P2POp(dist.irecv, tensor, peer, group))
                    completed.append((job.id, tensor))
                    del tensor
                elif rank == owner:
                    tensor = local.pop(job.id)
                    send_tensors.append(tensor)
                    operations.append(dist.P2POp(dist.isend, tensor, root, group))
                    del tensor
            if operations:
                with nvtx_range("decode.tile_exchange"):
                    for work in dist.batch_isend_irecv(operations):
                        work.wait()
            _sync(device)
            operations.clear()
            send_tensors.clear()
            stats["comm_seconds"] += time.perf_counter() - t
            error = None
            try:
                if rank == 0:
                    stats["gathered_tiles_peak"] = max(stats["gathered_tiles_peak"], len(completed))
                    t = time.perf_counter()
                    with nvtx_range(f"decode.stitch.clip.{clip}"):
                        decoded_clip = stitch_clip(vae, plan, clip, completed)
                        completed.clear()
                        main = decoded_clip[:, :, 3:20]
                        if overlap is not None:
                            main = vae._blend(overlap, main, 5, dim=-3)
                        # Clone the tail to avoid retaining the full 28-frame clip.
                        overlap = decoded_clip[:, :, 23:28].clone()
                    _sync(device)
                    stats["stitch_seconds"] += time.perf_counter() - t
                    t = time.perf_counter()
                    with nvtx_range(f"decode.output.clip.{clip}"):
                        if output is None:
                            output = torch.empty(
                                (main.shape[0], main.shape[1], plan.output_frames, *main.shape[-2:]),
                                dtype=main.dtype, device="cpu" if output_cpu else device,
                            )
                        pieces = (main, overlap) if clip == plan.clips - 1 else (main,)
                        for piece in pieces:
                            count = min(piece.shape[2], max(plan.output_frames - position, 0))
                            if count:
                                output[:, :, position : position + count].copy_(piece[:, :, :count])
                            position += count
                    _sync(device)
                    stats["output_cpu_seconds" if output_cpu else "stitch_seconds"] += time.perf_counter() - t
                    del pieces, piece, main, decoded_clip
            except Exception as exc:
                error = exc
            t = time.perf_counter()
            with nvtx_range("decode.stitch_completion" if rank == 0 else "decode.idle_worker.stitch"):
                _raise_collectively(error, device, group, world)
            stats["comm_seconds"] += time.perf_counter() - t
    finally:
        completed.clear()
        local.clear()
        z = overlap = None
        t = time.perf_counter()
        if backing:
            for child, name, is_parameter, tensor in backing:
                if is_parameter:
                    child._parameters[name].data = tensor
                else:
                    child._buffers[name] = tensor
            backing.clear()
        elif manage_lifecycle:
            for module in modules:
                module.to("cpu")
        _sync(device)
        stats["evict_seconds"] = time.perf_counter() - t
    stats["total_seconds"] = time.perf_counter() - started
    stats["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    stats["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    stats["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    names = tuple(stats)
    if world > 1:
        values = torch.tensor([stats[name] for name in names], dtype=torch.float64, device=device)
        gathered = [torch.empty_like(values) for _ in range(world)]
        dist.all_gather(gathered, values, group=group)
        ranks = [dict(zip(names, item.cpu().tolist())) for item in gathered]
        for item in ranks:
            for name in ("rank", "jobs", "gathered_tiles_peak", "peak_allocated_bytes",
                         "peak_reserved_bytes", "peak_host_rss_bytes"):
                item[name] = int(item[name])
    else:
        ranks = [stats]
    metrics = {
        "rank": rank, "world_size": world, "total_jobs": len(plan.jobs), "clips": plan.clips,
        "tiles_per_clip": plan.tiles_per_clip, "rank_job_counts": [item["jobs"] for item in ranks],
        "pad_tokens": plan.pad_tokens, "trim_frames": plan.trim_frames,
        "output_frames": plan.output_frames, "ranks": ranks,
        "peak_scope": "call" if reset_peak_memory else "since_last_allocator_reset",
        "output_cpu": output_cpu, "managed_lifecycle": manage_lifecycle,
        "wall_seconds": time.perf_counter() - started,
    }
    return output, metrics
