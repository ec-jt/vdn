"""Bounded, request-owned MiniMax-H3 AdaLN tables, with no model mutation.

Prepare after all adapter merges and before ResidentWeightShards.distribute().
All request steps are evaluated while each projection is staged exactly once.
This module deliberately does not install hooks, distribute tables, or hide
preparation in a first-forward cache. The caller owns those integration seams.
"""

from __future__ import annotations

import copy
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _timestep_copy(value: Tensor) -> Tensor:
    if value.ndim != 1 or value.numel() == 0 or value.dtype != torch.float32:
        raise ValueError("timesteps must be nonempty one-dimensional FP32 tensors")
    value = value.detach().to("cpu", copy=True)
    if not torch.isfinite(value).all() or not ((value >= 0) & (value <= 1)).all():
        raise ValueError("H3 timesteps must be finite and in [0, 1]")
    if value.numel() > 1 and not (value[1:] > value[:-1]).all():
        raise ValueError("each timestep table must be sorted and unique")
    return value


class AdaLNSchedule:
    """Snapshot of the *actual* sorted unique H3 time tables, not step ordinals.

    Times are 1 - sigma, not sigma and not training times scaled by 1000.
    Construct from the prepared pipeline's tables for arbitrary layouts. Use
    ``from_timesteps`` for the local render schedule and its four row roles.
    No token-sized table is retained per step.
    """

    def __init__(self, timesteps: Iterable[Tensor]):
        self._timesteps = tuple(_timestep_copy(t) for t in timesteps)
        if not self._timesteps:
            raise ValueError("a request must contain at least one forward")

    @classmethod
    def from_timesteps(
        cls,
        video_timesteps: Tensor,
        audio_timesteps: Tensor,
        *,
        active_roles: Sequence[str] = ("video_text", "audio"),
        condition_time: float = 0.999,
    ) -> AdaLNSchedule:
        """Mirror render's paired schedules without constructing token rows.

        Active roles must reflect rows actually present in the packed layout:
        video_text = generated video OR text (including vision text tokens),
        audio = generated audio, condition_video = keyframes/image references,
        condition_audio = clean reference audio (installed pipeline only).
        Pairing deliberately uses zip, matching render's shorter-grid behavior.
        Schedulers must already be set up; this method never advances them.
        """
        roles = tuple(active_roles)
        allowed = {"video_text", "audio", "condition_video", "condition_audio"}
        if not roles or len(set(roles)) != len(roles) or not set(roles) <= allowed:
            raise ValueError("active_roles must be distinct supported row roles")
        if not 0 <= condition_time <= 1:
            raise ValueError("condition_time must be in [0, 1]")
        grids = []
        for grid in (video_timesteps, audio_timesteps):
            if grid.ndim != 1 or grid.dtype != torch.float32:
                raise ValueError("scheduler timesteps must be one-dimensional FP32")
            grid = grid.detach().cpu()
            if not torch.isfinite(grid).all() or not ((grid >= 0) & (grid <= 1)).all():
                raise ValueError("scheduler timesteps must be finite and in [0, 1]")
            grids.append(grid)
        tables = []
        for video_t, audio_t in zip(*grids):
            values = {
                "video_text": float(video_t),
                "audio": float(audio_t),
                "condition_video": max(float(video_t), condition_time),
                "condition_audio": 1.0,
            }
            tables.append(torch.unique(torch.tensor(
                [values[role] for role in roles], dtype=torch.float32), sorted=True))
        return cls(tables)

    def __len__(self) -> int:
        return len(self._timesteps)

    def timestep(self, step: int, *, device: torch.device | str = "cpu") -> Tensor:
        """Return an independent copy; mutation cannot change the plan."""
        _check_index(step, len(self))
        return self._timesteps[step].to(device=device, copy=True)


@dataclass(frozen=True)
class AdaLNStats:
    steps: int
    blocks: int
    table_bytes: int
    peak_staged_projection_bytes: int
    projection_calls: int


def _check_index(index: int, length: int) -> None:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < length:
        raise IndexError(f"index {index!r} outside [0, {length})")


def _nbytes(value: Tensor) -> int:
    return value.numel() * value.element_size()


def _linear(module: nn.Module, *, output: bool) -> tuple[nn.Linear, int]:
    linear = getattr(module, "linear", None)
    if type(linear) is not nn.Linear:
        raise TypeError("AdaLN requires a plain, fully merged nn.Linear projection")
    if output:
        hidden_size = linear.out_features // 2
        factor = 2
    else:
        hidden_size = getattr(module, "hidden_size", 0)
        factor = 18
    if hidden_size <= 0 or linear.out_features != factor * hidden_size:
        raise ValueError("unexpected MiniMax-H3 AdaLN projection shape")
    if linear.bias is None:
        raise ValueError("MiniMax-H3 AdaLN projections must have a bias")
    for value in (linear.weight, linear.bias):
        if value.is_meta:
            raise ValueError("AdaLN weights are meta; prepare before shard distribution")
        if value.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError("AdaLN projection must retain its merged floating-point dtype")
    if linear.weight.dtype != linear.bias.dtype:
        raise ValueError("projection weight and bias dtypes must match")
    return linear, hidden_size


@contextmanager
def _stage_linear(linear: nn.Linear, device: torch.device) -> Iterator[tuple[Tensor, Tensor]]:
    """Only this projection has device staging; original Parameters never move."""
    weight = bias = None
    try:
        weight = linear.weight.detach().to(device)
        bias = linear.bias.detach().to(device)
        yield weight, bias
    finally:
        # No empty_cache/synchronize/global allocator or CUDA policy mutation.
        # Default-stream consumers finish before this storage can be reused.
        del weight, bias


def _embed_schedule(
    schedule: AdaLNSchedule, time_proj: nn.Module, time_embedder: nn.Module,
    device: torch.device,
) -> tuple[Tensor, ...]:
    # These are shared, small root modules, never the transformer or its blocks.
    for module in (time_proj, time_embedder):
        if module.training:
            raise ValueError("timestep modules must be in evaluation mode")
        if any(t.is_meta for t in (*module.parameters(), *module.buffers())):
            raise ValueError("timestep modules must be materialized")
    dtype = next(time_embedder.parameters()).dtype
    if dtype != torch.float32:
        raise ValueError("the merged H3 timestep MLP must remain FP32")
    proj = copy.deepcopy(time_proj).to(device)
    embedder = copy.deepcopy(time_embedder).to(device)
    return tuple(embedder(proj(t.to(device)).to(dtype)).detach() for t in schedule._timesteps)


def _project_schedule(
    linear: nn.Linear, embeddings: tuple[Tensor, ...],
    compute_device: torch.device, table_device: torch.device,
) -> tuple[Tensor, ...]:
    # Keep each original [K, E] GEMM shape. Concatenating steps changes algorithm
    # selection/rounding and is not equivalent to the patched baseline.
    result = []
    weight = bias = None
    try:
        with _stage_linear(linear, compute_device) as (weight, bias):
            for temb in embeddings:
                result.append(F.linear(F.silu(temb.float()).to(weight.dtype), weight, bias)
                              .detach().to(table_device))
        return tuple(result)
    finally:
        weight = bias = None
        result.clear()


class AdaLNStep:
    """Checked per-forward view. All returned tensors are read-only by contract."""

    def __init__(self, request: RequestAdaLN, index: int):
        self._request = request
        self._index = index

    @property
    def embedding(self) -> Tensor:
        self._request._check_open()
        return self._request._embeddings[self._index]

    def block(self, block_index: int) -> tuple[Tensor, ...]:
        """Six [3*K, H] tables in shift/scale/gate attention, then MLP order."""
        request = self._request
        request._check_open()
        _check_index(block_index, len(request._blocks))
        projected = request._blocks[block_index][self._index]
        hidden_size = projected.shape[-1] // 18
        return projected.view(-1, 6 * hidden_size).chunk(6, dim=-1)

    def output(self) -> tuple[Tensor, Tensor]:
        """Output-norm shift then scale, each [K, H], without modality rows."""
        self._request._check_open()
        if self._request._output is None:
            raise RuntimeError("output AdaLN was not requested")
        return self._request._output[self._index].chunk(2, dim=-1)

    def apply_output(self, hidden: Tensor, norm: nn.Module, timestep_indices: Tensor) -> Tensor:
        """Use norm_out.norm here, not norm_out; keep hidden-dependent RMSNorm."""
        shift, scale = self.output()
        hidden = norm(hidden)
        return hidden * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices)


class RequestAdaLN:
    """Fresh per prepare call, containing only small detached tables and metadata.

    Use as a context manager (or close in finally) around exactly one request.
    No model, projection, hook, scheduler, or cross-request cache is retained.
    Returned tensor views must not be mutated or retained beyond the request.
    """

    def __init__(self, schedule: AdaLNSchedule, embeddings: tuple[Tensor, ...],
                 blocks: tuple[tuple[Tensor, ...], ...], output: tuple[Tensor, ...] | None,
                 stats: AdaLNStats):
        self._schedule = schedule
        self._embeddings = embeddings
        self._blocks = blocks
        self._output = output
        self.stats = stats
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("request AdaLN tables have been closed")

    def step(self, index: int, timestep: Tensor) -> AdaLNStep:
        """Check actual sorted unique times once, immediately before the block loop.

        This detects a changed schedule even with identical step count/shape.
        The small device-to-host check can synchronize and belongs in timing.
        Token modality/index layout remains the caller's original calculation.
        """
        self._check_open()
        _check_index(index, len(self._schedule))
        if not torch.equal(_timestep_copy(timestep), self._schedule._timesteps[index]):
            raise ValueError("actual timestep table differs from this request's schedule")
        return AdaLNStep(self, index)

    def to(self, device: torch.device | str) -> RequestAdaLN:
        """Move only small request tables once, e.g. after CPU-table transport.

        This is an explicit pre-loop operation, not a lookup-time weight copy.
        Do not retain tensor views across this operation. The table-byte budget
        applies to final storage; source and destination coexist during a move.
        A failed move closes the request rather than leaving mixed devices.
        """
        self._check_open()
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU/CUDA table devices are supported")
        try:
            self._embeddings = tuple(t.to(device) for t in self._embeddings)
            self._blocks = tuple(tuple(t.to(device) for t in block) for block in self._blocks)
            if self._output is not None:
                self._output = tuple(t.to(device) for t in self._output)
        except BaseException as exc:
            self.close()
            traceback.clear_frames(exc.__traceback__)
            raise
        return self

    def close(self) -> None:
        """Idempotently release owned tables; also invalidate outstanding step views."""
        self._embeddings = ()
        self._blocks = ()
        self._output = None
        self._schedule = None
        self._closed = True

    def __enter__(self) -> RequestAdaLN:
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class CachedAdaLNView(RequestAdaLN):
    """Request-local checked state; table storage belongs to a bounded owner."""
    def __init__(self, owner):
        source = owner.table
        super().__init__(source._schedule, source._embeddings, source._blocks,
                         source._output, source.stats)
        self._owner = owner
        owner.references += 1

    def _check_open(self):
        super()._check_open()
        self._owner.table._check_open()

    def to(self, device):
        raise RuntimeError("cached tables cannot be moved through a request view")

    def close(self):
        if not self._closed:
            self._owner.references -= 1
        super().close()


class AdaLNTableCache:
    """One-entry, byte-budgeted cache. Active views prohibit replacement."""
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.table = self.key = None
        self.references = self.hits = self.misses = self.evictions = 0
        self.versions = ()

    @property
    def bytes(self):
        return 0 if self.table is None else self.table.stats.table_bytes

    def _versions(self):
        if self.table is None:
            return ()
        table = self.table
        return tuple((id(t), t._version) for t in (*table._embeddings, *(table._output or ()),
                     *(t for block in table._blocks for t in block)))

    def lookup(self, key):
        if self.references:
            raise RuntimeError("cached AdaLN still has an active request view")
        if self.table is not None and self.versions != self._versions():
            raise RuntimeError("immutable cached AdaLN table was mutated")
        if self.table is not None and self.key == key:
            self.hits += 1
            return CachedAdaLNView(self)
        self.misses += 1
        if self.table is not None:
            self.evictions += 1
            self.clear()
        return None

    def store(self, key, table):
        if self.table is not None or self.references:
            raise RuntimeError("cache must be empty before storing tables")
        if table.stats.table_bytes > self.max_bytes:
            raise MemoryError("AdaLN cache byte budget exceeded")
        self.table, self.key = table, key
        self.versions = self._versions()
        return CachedAdaLNView(self)

    def clear(self):
        if self.references:
            raise RuntimeError("cannot evict active cached AdaLN views")
        if self.table is not None:
            self.table.close()
        self.table = self.key = None
        self.versions = ()


@torch.no_grad()
def precompute_adaln(
    schedule: AdaLNSchedule,
    *,
    time_proj: nn.Module,
    time_embedder: nn.Module,
    block_projections: Iterable[nn.Module],
    norm_out: nn.Module | None = None,
    compute_device: torch.device | str = "cpu",
    table_device: torch.device | str = "cpu",
    max_table_bytes: int = 256 * 1024 * 1024,
) -> RequestAdaLN:
    """Compute the complete request before any transformer block loop.

    Pass ``(b.adaln_proj for b in transformer.transformer_blocks)`` after merging
    adapters, before sharding makes those weights meta. CPU source weights stay
    untouched; only one block projection is staged on compute_device at a time.
    Output projection is staged separately. Root embedding clones are released
    before any block projection is staged. Only final tables go to table_device.

    Uses the installed patch's FP32 SiLU -> weight-dtype cast -> biased linear,
    with autocast explicitly disabled. It does not alter TF32/backend policy.
    For same-device parity choose the baseline's device and precision policy;
    CPU vs GPU evaluation is not promised bitwise equivalent. No weights are
    copied at denoise steps. The caller must transport tables to peer ranks.

    max_table_bytes bounds owned timestep, embedding and projection tables,
    not the source model, one staged projection, or temporary GEMM workspace.
    Exceeding it fails, never falls back to per-step weight copies.
    """
    compute_device, table_device = torch.device(compute_device), torch.device(table_device)
    if compute_device.type not in ("cpu", "cuda") or table_device.type not in ("cpu", "cuda"):
        raise ValueError("only CPU/CUDA compute and table devices are supported")
    if not isinstance(max_table_bytes, int) or max_table_bytes <= 0:
        raise ValueError("max_table_bytes must be positive")
    # Independent snapshots make reusing a plan across requests safe; outputs
    # are always recomputed, including after weights/adapters change.
    snapshot = AdaLNSchedule(schedule._timesteps)
    used = sum(_nbytes(t) for t in snapshot._timesteps)

    def reserve(size: int) -> None:
        nonlocal used
        if used + size > max_table_bytes:
            raise MemoryError(f"AdaLN table budget exceeded: {used + size} > {max_table_bytes}")
        used += size

    reserve(0)
    embedding_output = getattr(time_embedder, "linear_2", None)
    if type(embedding_output) is not nn.Linear:
        raise TypeError("expected the merged H3 timestep MLP with plain linear_2")
    embedding_width = embedding_output.out_features
    reserve(sum(t.numel() for t in snapshot._timesteps) * embedding_width * 4)
    embeddings = stored_embeddings = ()
    blocks = []
    output = None
    peak_bytes = calls = 0
    try:
        with torch.autocast(device_type=compute_device.type, enabled=False):
            embeddings = _embed_schedule(snapshot, time_proj, time_embedder, compute_device)
            if any(t.shape != (times.numel(), embedding_width) or t.dtype != torch.float32
                   for t, times in zip(embeddings, snapshot._timesteps)):
                raise ValueError("timestep embedding must return FP32 [K, E]")

            def project(module: nn.Module, *, output: bool) -> tuple[Tensor, ...]:
                nonlocal peak_bytes, calls
                linear, _ = _linear(module, output=output)
                if any(t.shape[-1] != linear.in_features for t in embeddings):
                    raise ValueError("timestep embedding width does not match projection")
                reserve(sum(t.shape[0] for t in embeddings) * linear.out_features
                        * linear.weight.element_size())
                peak_bytes = max(peak_bytes, _nbytes(linear.weight) + _nbytes(linear.bias))
                result = _project_schedule(linear, embeddings, compute_device, table_device)
                calls += len(embeddings)
                return result

            for module in block_projections:
                blocks.append(project(module, output=False))
            if not blocks:
                raise ValueError("at least one block projection is required")
            output = project(norm_out, output=True) if norm_out is not None else None
            stored_embeddings = tuple(t.to(table_device) for t in embeddings)
        return RequestAdaLN(snapshot, stored_embeddings, tuple(blocks), output,
                            AdaLNStats(len(snapshot), len(blocks), used, peak_bytes, calls))
    except BaseException as exc:
        # Request runners can retain exceptions for reporting. Do not let their
        # tracebacks retain partial tables, embedding clones or staged weights.
        embeddings = stored_embeddings = ()
        blocks.clear()
        output = None
        traceback.clear_frames(exc.__traceback__)
        raise
