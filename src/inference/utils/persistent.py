"""Bounded sequential paper requests with immutable model-resource ownership.

No server, background request execution, model mutation API or request retries.
The caller must use the same eight ranks in the same order for every operation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from src.inference.utils.paper_runtime import PaperRuntime
from src.inference.utils.nsight import nvtx_range


def model_signature(cfg):
    """All non-request options are immutable, including adapters and precision."""
    value = OmegaConf.to_container(cfg, resolve=True)
    value.pop("render")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prompt_digest(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("persistent requests require an existing local prompt cache")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_request(cfg, signature):
    if model_signature(cfg) != signature:
        raise ValueError("model/LoRA/compute signature changed")
    if (cfg.render.num_frames != 124 or cfg.render.num_steps != 8
            or cfg.render.warmup_steps != 0):
        raise ValueError("persistent protocol requires 124 frames, 8 NFE, no warmup")
    if type(cfg.render.seed) is not int or not 0 <= cfg.render.seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    digest = prompt_digest(cfg.render.prompt_file)
    from src.inference.utils.prompt_cache import load_prompt
    embeds, tags, conditions = load_prompt(cfg.render.prompt_file, "cpu")
    if embeds.ndim != 2 or embeds.shape[1] != 5120 or tags.shape != embeds.shape[:1]:
        raise ValueError("invalid cached prompt shape")
    if not torch.isfinite(embeds).all():
        raise ValueError("nonfinite cached prompt")
    if conditions and any(not torch.isfinite(t).all() for t in conditions[1]):
        raise ValueError("nonfinite cached condition")
    return digest


def consensus(runtime, label, error=None, signature=None):
    """Recoverable admission votes use Gloo, before any request NCCL payload."""
    group = runtime.paper_runtime.group
    values = [None] * runtime.world_size
    dist.all_gather_object(values, (error, signature), group=group)
    errors = [f"rank {i}: {v[0]}" for i, v in enumerate(values) if v[0]]
    if errors or any(v[1] != values[0][1] for v in values):
        raise ValueError(f"{label}: " + ("; ".join(errors) or "rank request signatures differ"))


class PersistentPaperRuntime(PaperRuntime):
    """Model owner: wrappers/group/geometry/source weights survive requests."""
    persistent = True

    def __init__(self, cfg, runtime, *, control_group=None):
        def option(name):
            value = os.environ.get(name, "0")
            if value not in ("0", "1"):
                raise ValueError(name + " must be 0 or 1")
            return value == "1"
        self.fused_producers = option("VDN_TP_FUSED_PRODUCERS")
        self.fast_finite = option("VDN_TP_FAST_FINITE")
        self.cache_adaln = option("VDN_TP_CACHE_ADALN")
        if (self.fused_producers or self.fast_finite or self.cache_adaln) and not getattr(runtime, "permanent_tensor_parallel", False):
            raise ValueError("TP optimizations require permanent tensor parallelism")
        super().__init__(cfg, runtime, control_group=control_group)
        self.signature = model_signature(cfg)
        self.sources = None
        self.source_bytes = 0
        self.geometry_cache = {}
        self.poisoned = False
        self.request_count = 0
        self.state = "idle"
        from src.inference.utils.paper_adaln import AdaLNTableCache
        self.table_cache = AdaLNTableCache(self.options["paper_adaln_table_mib"] << 20)
        self.source_identity = None

    def retain_adaln_sources(self, transformer):
        # Deep-copy ONLY the required modules before distribution rebinds their
        # parameter slots to meta. Never retain the merged transformer itself.
        self.sources = SimpleNamespace(
            time_proj=copy.deepcopy(transformer.time_proj),
            time_embedder=copy.deepcopy(transformer.time_embedder),
            transformer_blocks=[SimpleNamespace(adaln_proj=copy.deepcopy(b.adaln_proj))
                                for b in transformer.transformer_blocks],
            norm_out=copy.deepcopy(transformer.norm_out))
        modules = [self.sources.time_proj, self.sources.time_embedder, self.sources.norm_out,
                   *(b.adaln_proj for b in self.sources.transformer_blocks)]
        self.source_bytes = sum(t.numel() * t.element_size() for m in modules
                                for t in (*m.parameters(), *m.buffers()))
        if any(t.device.type != "cpu" for m in modules for t in (*m.parameters(), *m.buffers())):
            raise ValueError("AdaLN retained sources must be CPU-only")
        self.source_identity = self._source_identity()

    def _source_identity(self):
        if self.sources is None:
            return None
        modules = [self.sources.time_proj, self.sources.time_embedder, self.sources.norm_out,
                   *(b.adaln_proj for b in self.sources.transformer_blocks)]
        return tuple((id(m), m.training, tuple((name, id(t), t._version, str(t.dtype), str(t.device), tuple(t.shape))
            for name, t in (*m.named_parameters(), *m.named_buffers()))) for m in modules)

    def cached_adaln_key(self, cfg):
        if model_signature(cfg) != self.signature or self._source_identity() != self.source_identity:
            self.poisoned = True
            raise RuntimeError("immutable model/AdaLN source signature changed")
        # Tables use only sorted time values and immutable source modules, never
        # prompt embedding values or noise. Geometry/roles are conservative keys.
        schedule = tuple(t.numpy().tobytes() for t in self.schedule._timesteps)
        return (self.signature, schedule, self.request_key, str(self.runtime.device), self.shifts)

    def prepare_cached_adaln(self, cfg):
        with self.timed("adaln_cache_lookup"):
            with self.guard.local_phase("adaln_cache_lookup"):
                key = self.cached_adaln_key(cfg)
                self.adaln = self.table_cache.lookup(key)
            # A hit must agree on every rank before peers skip table transport.
            hit = self.adaln is not None
            if self.runtime.world_size > 1:
                hits = [None] * self.runtime.world_size
                dist.all_gather_object(hits, hit, group=self.group)
                if len(set(hits)) != 1:
                    self.poisoned = True
                    raise RuntimeError("ranks disagree on AdaLN cache ownership")
        self.call("adaln_cache_hit" if hit else "adaln_cache_miss")
        if not hit:
            with self.guard.local_phase("persistent_adaln_precompute"):
                if self.runtime.is_main:
                    self.prepare_adaln(self.sources)
            self.transport_adaln()
            with self.guard.local_phase("adaln_cache_store"):
                self.adaln = self.table_cache.store(key, self.adaln)
        from dataclasses import asdict
        self.adaln_stats = asdict(self.adaln.stats)
        self.adaln_stats.update(cache_hit=hit, cache_bytes=self.table_cache.bytes,
                                cache_entries=1, cache_evictions=self.table_cache.evictions)

    def prepare_request(self, cfg, transformer):
        if self.poisoned or self.state not in ("idle", "finished", "rejected"):
            raise RuntimeError("model owner is poisoned, busy or closed")
        self.request = self.new_request(cfg)
        self.request_count += 1
        self.plans.update(self.geometry_cache)
        from src.inference.utils.ulysses import prebuild_prompt_masks
        with self.guard.local_phase("persistent_request_planning"):
            with self.timed("prompt_layout_and_masks"):
                prebuild_prompt_masks(cfg, transformer, self.runtime.device, paper=self)
        # Keep only this geometry's plans. No prompt values or tables are cached.
        from src.models.hybrid_transform import iter_hybrids
        keys = {self.plan_key(attn, self.request_key[0]) for attn in iter_hybrids(transformer)}
        self.plans = {k: v for k, v in self.plans.items() if k in keys}
        self.geometry_cache = dict(self.plans)
        if self.cache_adaln:
            self.prepare_cached_adaln(cfg)
            return
        with self.guard.local_phase("persistent_adaln_precompute"):
            if self.runtime.is_main:
                self.prepare_adaln(self.sources)
        self.transport_adaln()

    def drain_request(self, controller, *, failed=False):
        """Finite/request failures close tables; transport failures poison owner.

        No recovery collective is attempted here. A failed batch exits even if
        resources drained cleanly; callers must not infer safe retry from this.
        """
        try:
            controller.finish_request()
            if self.runtime.device.type == "cuda":
                torch.cuda.synchronize(self.runtime.device)
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.close_tables()
            self.schedule = None
            if failed:
                self.state = "rejected"

    def close(self):
        try:
            super().close()
        finally:
            self.sources = None
            self.geometry_cache.clear()
            self.table_cache.clear()


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    with partial.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    # link is no-clobber, unlike replace. Only remove this call's own partial.
    os.link(partial, path)
    partial.unlink()


class ResidentDecoders:
    """Explicit resident policy. FP32 decode-only video; rank-zero audio owner."""
    policy = "resident"

    def __init__(self):
        self.video = self.audio = None
        self.loads = 0

    def decode(self, cfg, runtime, video, audio):
        from diffusers import AutoencoderKLMiniMaxH3Audio
        from src.inference.utils.cpu_loading import load_cpu_model, resolve_inference_weights
        from src.inference.utils.distributed_decode import load_video_decoder, distributed_decode
        from src.inference.utils.distributed_decode_callback import _postprocess_video, _check_status
        from src.inference.render import _decode_audio, encode_video, FPS
        started = time.perf_counter()
        loads = {}
        error = None
        try:
            source = resolve_inference_weights(cfg.vae_source, local_files_only=True)
            t = time.perf_counter()
            if self.video is None:
                with nvtx_range("persistent.decoder.video_load_once"):
                    self.video = load_video_decoder(source)
                loads["video_load_seconds"] = time.perf_counter() - t
                t = time.perf_counter()
                with nvtx_range("persistent.decoder.video_onload_once"):
                    self.video.decoder.to(runtime.device)
                    self.video.post_quant_conv.to(runtime.device)
                torch.cuda.synchronize(runtime.device)
                loads["video_onload_seconds"] = time.perf_counter() - t
                self.loads += 1
            else:
                loads.update(video_load_seconds=0.0, video_onload_seconds=0.0)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _check_status(runtime, "resident decoder load", error)
        raw, metrics = distributed_decode(self.video, video, device=runtime.device,
            output_cpu=False, manage_lifecycle=False, reset_peak_memory=False)
        frames = None
        error = None
        post_seconds = 0.0
        try:
            if runtime.is_main:
                t = time.perf_counter()
                if not torch.isfinite(raw).all():
                    raise FloatingPointError("nonfinite raw video")
                frames = _postprocess_video(raw, runtime.device)
                post_seconds = time.perf_counter() - t
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            raw = None
        _check_status(runtime, "resident video postprocess", error)
        result = [None]
        if runtime.is_main:
            try:
                t = time.perf_counter()
                if self.audio is None:
                    with nvtx_range("persistent.decoder.audio_load_once"):
                        self.audio = load_cpu_model(AutoencoderKLMiniMaxH3Audio, source,
                            subfolder="audio_vae", torch_dtype=torch.float32, local_files_only=True)
                        self.audio.eval().requires_grad_(False)
                    loads["audio_load_seconds"] = time.perf_counter() - t
                    t = time.perf_counter()
                    self.audio.to(runtime.device)
                    torch.cuda.synchronize(runtime.device)
                    loads["audio_onload_seconds"] = time.perf_counter() - t
                else:
                    loads.update(audio_load_seconds=0.0, audio_onload_seconds=0.0)
                t = time.perf_counter()
                decoded_audio = _decode_audio(audio, self.audio, runtime.device)
                if not torch.isfinite(torch.as_tensor(decoded_audio)).all():
                    raise FloatingPointError("nonfinite decoded audio")
                loads["audio_decode_seconds"] = time.perf_counter() - t
                t = time.perf_counter()
                out = Path(cfg.render.out)
                partial = out.with_name(out.stem + ".partial" + out.suffix)
                if out.exists() or partial.exists():
                    raise FileExistsError(out)
                with nvtx_range("decode.output.media_mux"):
                    encode_video(frames, fps=FPS, output_path=str(partial), audio=decoded_audio,
                                 audio_sample_rate=self.audio.config.sampling_rate)
                    os.link(partial, out)
                    partial.unlink()
                loads["media_encode_seconds"] = time.perf_counter() - t
                result[0] = dict(error=None, root=loads, video_postprocess_seconds=post_seconds)
            except Exception as exc:
                result[0] = dict(error=f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(result, src=0, group=runtime.paper_runtime.group)
        if result[0]["error"]:
            raise RuntimeError(result[0]["error"])
        return dict(distributed_video=metrics, local_loads=loads, **result[0],
                    total_seconds=time.perf_counter() - started, policy=self.policy)

    def close(self):
        self.video = self.audio = None
