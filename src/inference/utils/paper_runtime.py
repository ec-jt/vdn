"""Explicit, one-request paper stack ownership and coordinated boundary checks.

No checkpoint IO, global enablement, global request cache or decoder changes.
The CPU control group is deliberately separate from both NCCL communicators:
finite votes never violate the bounded weight-prefetch attention fence.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import timedelta
import time
import types
import json
import zlib

import torch
import torch.distributed as dist

from src.config.inference import paper_enabled
from src.inference.utils.nsight import nvtx_range
from src.inference.utils.paper_adaln import AdaLNSchedule, precompute_adaln


class CoordinatedFiniteGuard:
    """Accumulate device predicates, synchronize only at explicit boundaries.

    All ranks must enter boundaries in the same order. CUDA/NCCL fatal errors
    still require the launcher's bounded process teardown, never peer recovery.
    """
    def __init__(self, device, *, group=None, world_size=1, fast=False):
        self.device = torch.device(device)
        self.group, self.world_size = group, world_size
        if world_size > 1 and group is None:
            raise ValueError("distributed finite guard requires a CPU control group")
        self.pending = torch.ones((), device=self.device, dtype=torch.bool)
        self.failed = False
        self.boundaries = 0
        self.fast = fast and self.device.type == "cuda"
        self.parts = []

    def predicate(self, parts):
        """Own a producer's finite predicate and join its producing stream."""
        if self.fast:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.parts.append((parts, event))
        else:
            self.pending.logical_and_(parts.all())

    def observe(self, *tensors):
        for tensor in tensors:
            if tensor is not None:
                if self.fast:
                    from src.models.ops.mxfp8_kernels import finite_parts
                    with nvtx_range("paper.finite.kernel.float"):
                        self.predicate(finite_parts(tensor))
                else:
                    self.pending.logical_and_(torch.isfinite(tensor).all())

    def encoded(self, data, scales):
        if self.fast:
            from src.models.ops.mxfp8_kernels import finite_parts
            with nvtx_range("paper.finite.kernel.encoded"):
                self.predicate(finite_parts(data, 1))
                self.predicate(finite_parts(scales, 2))
            return
        self.pending.logical_and_(~((data & 127) == 127).any())
        self.pending.logical_and_(~(scales == 255).any())

    def check(self, label, *tensors, error=None):
        if self.failed:
            raise RuntimeError("paper finite guard is terminal")
        try:
            self.observe(*tensors)
        except Exception as exc:
            error = error or f"finite predicate failed: {exc}"
        with nvtx_range("paper.finite." + label):
            try:
                if self.parts:
                    stream = torch.cuda.current_stream(self.device)
                    for part, event in self.parts:
                        stream.wait_event(event)
                        part.record_stream(stream)
                    with nvtx_range("paper.finite.kernel.combine"):
                        self.pending.logical_and_(torch.cat([p for p, _ in self.parts]).all())
                    self.parts.clear()
                with nvtx_range("paper.finite.host_read"):
                    vote = self.pending.to(device="cpu", dtype=torch.int32)
            except Exception as exc:
                # A poisoned CUDA context cannot copy a predicate, but the CPU
                # control group can still tell peers not to enter NCCL payloads.
                vote = torch.zeros((), dtype=torch.int32)
                error = error or f"CUDA predicate read failed: {exc}"
            if error is not None:
                vote.zero_()
            if self.world_size > 1:
                # Also detect a local exception unwinding to an outer phase
                # while peers reached an inner boundary. Both must fail now,
                # never let a mismatched vote masquerade as collective readiness.
                tag = zlib.crc32(label.encode("utf-8"))
                ballot = torch.tensor([vote.item(), tag, -tag], dtype=torch.int64)
                with nvtx_range("paper.finite.gloo_vote"):
                    dist.all_reduce(ballot, op=dist.ReduceOp.MIN, group=self.group)
                vote = ballot[0]
                if ballot[1].item() != -ballot[2].item():
                    vote.zero_()
                    error = "ranks reached different guarded phases"
            self.boundaries += 1
            if not vote.item():
                self.failed = True
                raise FloatingPointError(f"paper all-rank failure before {label}: {error or 'nonfinite or peer failure'}")
            self.pending.fill_(True)

    @contextmanager
    def local_phase(self, label):
        """A communication-free region ends with one common failure vote."""
        error = None
        try:
            yield
        except Exception as exc:
            if self.failed:
                raise
            error = f"{type(exc).__name__}: {exc}"
        self.check(label, error=error)


def convert_paper_projections(model, cfg):
    from src.models.ops.mxfp8_linear import convert_linear_to_mxfp8
    enabled = paper_enabled(cfg) and cfg.compute.paper_mxfp8
    if enabled and len(model.transformer_blocks) != 50:
        raise ValueError("paper MXFP8 requires all 50 blocks, seven roles each")
    records = convert_linear_to_mxfp8(model, enabled=enabled)
    if enabled and len(records) != 350:
        raise ValueError("paper MXFP8 conversion must produce exactly 350 roles")
    return records


@dataclass
class PaperRequestState:
    """Owned by one request only. Never reset or recycled after table closure."""
    guard: object
    steps: int
    seed: int
    frames: int
    shifts: tuple
    calls: Counter = field(default_factory=Counter)
    preparation: Counter = field(default_factory=Counter)
    adaln: object = None
    schedule: object = None
    plans: dict = field(default_factory=dict)
    rotary: object = None
    request_key: object = None
    state: str = "prepared"
    step_index: int = 0
    adaln_stats: object = None
    charged_preparation_seconds: object = None


def _request_property(name):
    return property(lambda self: getattr(self.request, name),
                    lambda self, value: setattr(self.request, name, value))


class PaperRuntime:
    """Single-use request, created before assembly and closed on every exit."""
    def __init__(self, cfg, runtime, *, control_group=None):
        from src.config.inference import PaperComputeConfig
        self.options = {name: getattr(cfg.compute, name) for name in asdict(PaperComputeConfig())}
        self.runtime = runtime
        self.group = control_group
        self.request = self.new_request(cfg)
        self._restore = []
        self.conversion = []
        self.installed = False

    # Compatibility surface for installed consumers. The wrapper owner is stable,
    # but every mutable request field belongs to a separate single-use object.
    guard = _request_property("guard")
    calls = _request_property("calls")
    preparation = _request_property("preparation")
    adaln = _request_property("adaln")
    schedule = _request_property("schedule")
    plans = _request_property("plans")
    rotary = _request_property("rotary")
    request_key = _request_property("request_key")
    state = _request_property("state")
    step_index = _request_property("step_index")
    steps = _request_property("steps")
    seed = _request_property("seed")
    frames = _request_property("frames")
    shifts = _request_property("shifts")
    adaln_stats = _request_property("adaln_stats")
    charged_preparation_seconds = _request_property("charged_preparation_seconds")

    def new_request(self, cfg):
        return PaperRequestState(
            CoordinatedFiniteGuard(self.runtime.device, group=self.group,
                                   world_size=self.runtime.world_size,
                                   fast=getattr(self, "fast_finite", False)),
            cfg.render.num_steps, cfg.render.seed, cfg.render.num_frames,
            (cfg.render.video_shift, cfg.render.audio_shift))

    @classmethod
    def create(cls, cfg, runtime):
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=cfg.parallel.timeout_seconds))
        try:
            return cls(cfg, runtime, control_group=group)
        except BaseException:
            dist.destroy_process_group(group)
            raise

    @contextmanager
    def timed(self, name):
        started = time.perf_counter()
        try:
            with nvtx_range("paper.prepare." + name):
                yield
        finally:
            if self.runtime.device.type == "cuda":
                torch.cuda.synchronize(self.runtime.device)
            self.preparation[name] += time.perf_counter() - started

    def call(self, component):
        self.calls[component] += 1

    def prepare_geometry(self, cfg, model, layout, condition_rows):
        from diffusers import MiniMaxH3Scheduler
        from src.models.hybrid_transform import iter_hybrids
        from src.models.softmax_attention.paper_window import prepare_paper_window_plan
        with self.timed("schedule_and_window_plan"):
            video, audio = (MiniMaxH3Scheduler(shift=s) for s in self.shifts)
            for scheduler in (video, audio):
                scheduler.set_timesteps(self.steps + 1, device="cpu")
            roles = ["video_text", "audio"]
            if condition_rows:
                roles.append("condition_video")
            self.schedule = AdaLNSchedule.from_timesteps(video.timesteps, audio.timesteps, active_roles=roles)
            if len(self.schedule) != self.steps:
                raise ValueError("paper request schedule must contain every requested NFE")
            self.request_key = (layout, condition_rows)
            for attn in iter_hybrids(model):
                if self.options["paper_window"]:
                    key = self.plan_key(attn, layout)
                    if key not in self.plans:
                        self.plans[key] = prepare_paper_window_plan(
                            layout, attn._bounds(layout), attn.anchor_frames,
                            device=self.runtime.device,
                            max_gather_rows=self.options["paper_max_gather_rows"],
                            max_query_rows=self.options["paper_max_query_rows"])
                        self.call("window_plan")

    def preparation_charge(self):
        """Slowest complete rank setup, charged once to every rank's denoise."""
        value = torch.tensor(sum(self.preparation.values()), dtype=torch.float64)
        if self.runtime.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self.group)
        self.charged_preparation_seconds = value.item()
        return self.charged_preparation_seconds

    def plan_key(self, attn, layout):
        return (layout, tuple(tuple(b) for b in attn._bounds(layout)), attn.anchor_frames,
                self.runtime.device, attn.num_heads // self.runtime.world_size, attn.head_dim,
                self.options["paper_max_gather_rows"], self.options["paper_max_query_rows"])

    def window_plan(self, attn):
        if self.state != "active":
            raise RuntimeError("paper attention requires an active request")
        key = self.plan_key(attn, attn.layout)
        if key not in self.plans:
            raise ValueError("paper window request/head/device/geometry changed")
        return self.plans[key]

    def prepare_adaln(self, model):
        if not self.options["paper_adaln"]:
            return
        with self.timed("adaln_precompute"):
            self.adaln = precompute_adaln(
                self.schedule, time_proj=model.time_proj, time_embedder=model.time_embedder,
                block_projections=(b.adaln_proj for b in model.transformer_blocks),
                norm_out=model.norm_out, compute_device=self.runtime.device, table_device="cpu",
                max_table_bytes=self.options["paper_adaln_table_mib"] << 20)
            self.adaln_stats = asdict(self.adaln.stats)
            self.calls["adaln_precompute_projection"] += self.adaln.stats.projection_calls

    def transport_adaln(self):
        if not self.options["paper_adaln"]:
            return
        with self.timed("adaln_cpu_transport_and_device_tables"):
            payload = [self.adaln]
            if self.runtime.world_size > 1:
                # Trusted local ranks only. No GPU model/weights in the payload.
                dist.broadcast_object_list(payload, src=0, group=self.group)
            self.adaln = payload[0]
            with self.guard.local_phase("adaln_transport"):
                self.adaln.to(self.runtime.device)
                self.adaln_stats = asdict(self.adaln.stats)
                for table in (*self.adaln._embeddings, *self.adaln._output,
                              *(t for block in self.adaln._blocks for t in block)):
                    self.guard.observe(table)
                self.call("adaln_table_transport")

    def install(self, model):
        from src.models.hybrid_transform import iter_hybrids
        from src.models.ops.mxfp8_linear import MXFP8Linear
        from src.models.ops.fused_block import fast_ff_forward
        if self.installed:
            raise RuntimeError("paper runtime cannot install nested wrappers")
        self.installed = True
        for attn in iter_hybrids(model):
            attn._paper_runtime = self
            attn.linear_attention.paper_vda_enabled = self.options["paper_vda"]
            attn.inference_vda_overlap = self.options["paper_vda_overlap"]
            if attn.inference_mode or attn.linear_attention.inference_vda_kernels:
                raise ValueError("paper runtime cannot inherit global/legacy inference kernels")
        for index, block in enumerate(model.transformer_blocks):
            if self.options["paper_fused_block"]:
                self._replace(block.ff, "forward", types.MethodType(fast_ff_forward, block.ff))
            for name, module in block.named_modules():
                if not isinstance(module, MXFP8Linear):
                    continue
                # Quantized QKV bypasses forward(), so instrument the actual GEMM
                # seam. Never cache the sharder-rebound compressed tensor slots.
                original = module.forward_quantized
                role = next(r["role"] for r in self.conversion
                            if r["module"] == f"transformer_blocks.{index}.{name}")
                def checked(this, data, scales, _original=original, _role=role, **kwargs):
                    self.guard.encoded(data, scales)
                    with nvtx_range("paper.mxfp8." + _role):
                        result = _original(data, scales, **kwargs)
                    self.guard.observe(result)
                    self.call("mxfp8." + _role)
                    return result
                self._replace(module, "forward_quantized", types.MethodType(checked, module))
                forward = module.forward
                def observed(this, x, _forward=forward):
                    self.guard.observe(x)
                    return _forward(x)
                self._replace(module, "forward", types.MethodType(observed, module))
                self._restore.append((module, "check_finite", module.check_finite, True))
                module.check_finite = False
        model._paper_runtime = self
        # Load the extension outside the side stream. Charge it, do not hide it.
        if self.options["paper_vda"]:
            with self.timed("vda_factor_extension"):
                from src.models.linear_attention.paper_factor import _load_extension
                _load_extension(torch.cuda.get_device_capability(self.runtime.device))

    def _replace(self, module, name, value):
        self._restore.append((module, name, getattr(module, name), name in module.__dict__))
        setattr(module, name, value)

    def begin(self, layout, condition_rows, video_times, audio_times, *, frames, steps, seed, shifts):
        with self.guard.local_phase("request_begin"):
            if self.state != "prepared":
                raise RuntimeError("paper runtime permits exactly one preplanned request after sharding")
            if ((layout, condition_rows) != self.request_key or frames != self.frames
                    or steps != self.steps or seed != self.seed or shifts != self.shifts):
                raise ValueError("paper request changed after sharding")
            roles = ["video_text", "audio"] + (["condition_video"] if condition_rows else [])
            actual = AdaLNSchedule.from_timesteps(video_times, audio_times, active_roles=roles)
            if len(actual) != len(self.schedule) or any(
                    not torch.equal(actual.timestep(i), self.schedule.timestep(i)) for i in range(len(actual))):
                raise ValueError("paper request schedule changed after sharding")
            self.state = "active"

    def step(self, timestep):
        view = None
        with self.guard.local_phase("step_begin"):
            if self.state != "active" or self.step_index >= self.steps:
                raise RuntimeError("paper forward is outside its planned request")
            if not torch.equal(timestep.detach().cpu(), self.schedule.timestep(self.step_index)):
                raise ValueError("actual timestep differs from planned request")
            if self.adaln is not None:
                view = self.adaln.step(self.step_index, timestep)
                self.call("adaln_step")
            self.step_index += 1
        return view

    def finish(self):
        self.guard.check("request_finish", error=None if self.step_index == self.steps else "incomplete NFE schedule")
        self.state = "finished"
        self.close_tables()

    def close_tables(self):
        if self.adaln is not None:
            self.adaln.close()
            self.adaln = None
        self.plans.clear()
        self.rotary = None

    def inventory(self):
        return dict(rank=self.runtime.rank, selected=dict(self.options), called=dict(self.calls),
                    finite_boundaries=self.guard.boundaries, finite_failed=self.guard.failed,
                    preparation_seconds=dict(self.preparation),
                    preparation_added_to_denoise_seconds=self.charged_preparation_seconds,
                    adaln=self.adaln_stats, converted_roles=self.conversion,
                    request_state=self.state, completed_forwards=self.step_index)

    def close(self):
        if self.state == "closed":
            return
        # Every rank emits its actual completed-call inventory, including failure
        # paths where gathering JSON over an unhealthy communicator is unsafe.
        print("paper_stack_inventory=" + json.dumps(self.inventory(), sort_keys=True), flush=True)
        self.close_tables()
        for module, name, original, owned in reversed(self._restore):
            if owned:
                setattr(module, name, original)
            else:
                delattr(module, name)
        self._restore.clear()
        self.state = "closed"
        if self.group is not None and dist.is_initialized():
            dist.destroy_process_group(self.group)
            self.group = None
