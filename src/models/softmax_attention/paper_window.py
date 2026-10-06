# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded packed visible-key attention for the paper stack.

Semantic adaptation of SGLang PR #37903, merge
ff1ce11348673eea9b51165bd6b2da7238c958b4, hybrid_window_attn_h3.py.
See licenses/SGLang-LICENSE.txt and licenses/SGLang-NOTICE.txt. New local
implementation: public Torch native varlen on SM120, explicit reusable plans,
hard gather/query budgets, and no global mask or dependency on FA4.
Existing decomposed/flex/reference dispatch remains untouched.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate
import math

import torch

from src.models.sequence_layout import SequenceLayout


def _union(ranges):
    result = []
    for a, b in sorted(ranges):
        if a >= b:
            continue
        if result and result[-1][1] >= a:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return tuple(result)


def _count(ranges):
    return sum(b - a for a, b in ranges)


def _split(ranges, budget):
    current, size = [], 0
    for a, b in ranges:
        while a < b:
            end = min(b, a + budget - size)
            current.append((a, end))
            size += end - a
            a = end
            if size == budget:
                yield tuple(current)
                current, size = [], 0
    if current:
        yield tuple(current)


@dataclass(frozen=True)
class VisibleGroup:
    query_ranges: tuple
    key_ranges: tuple


@dataclass(frozen=True)
class PackedPass:
    """Disjoint queries; key rows can repeat BETWEEN varlen segments only."""
    groups: tuple[VisibleGroup, ...]
    query_rows: torch.Tensor
    key_rows: torch.Tensor | None
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int
    dense: bool


@dataclass(frozen=True)
class PaperWindowPlan:
    layout: SequenceLayout
    bounds: tuple
    anchor_frames: str
    max_gather_rows: int
    max_query_rows: int
    device: torch.device
    passes: tuple[PackedPass, ...]

    @property
    def peak_gather_rows(self):
        return max((p.key_rows.numel() for p in self.passes if not p.dense), default=0)


def prepare_paper_window_plan(
    layout: SequenceLayout, bounds, anchor_frames="none", *, device="cpu",
    max_gather_rows=65_536, max_query_rows=8_192,
) -> PaperWindowPlan:
    """Prepare once per request/device, outside the block loop and CUDA graphs.

    layout.seq_len counts USED rows: any extra input rows are padding, never
    keys. Text/audio are all non-video used rows, irrespective of text_start.
    Bounds are inclusive frame indices. Each visible set is a sorted UNION,
    so overlapping anchors/windows and F=1 never duplicate softmax mass.

    Both budgets are hard limits per pass. A non-dense visible set larger
    than max_gather_rows raises before allocating indices. Increase the budget;
    we never split a softmax over keys or approximate a merge. Full-cover groups
    use the original K/V directly and need no gather budget.
    """
    if not isinstance(layout, SequenceLayout):
        raise TypeError("layout must be SequenceLayout")
    s, vs, f, tpf = layout.seq_len, layout.video_start, layout.num_frames, layout.tokens_per_frame
    if any(type(x) is not int for x in (s, vs, f, tpf)) or s <= 0 or f < 0 or tpf <= 0 or vs < 0 or vs + f * tpf > s:
        raise ValueError("invalid used-row/video geometry")
    if anchor_frames not in ("none", "rows", "columns", "both"):
        raise ValueError("anchor_frames must be none, rows, columns, or both")
    if any(type(x) is not int or x <= 0 for x in (max_gather_rows, max_query_rows)):
        raise ValueError("row budgets must be positive integers")
    bounds = tuple(tuple(b) for b in bounds)
    if len(bounds) != f or any(len(b) != 2 or any(type(x) is not int for x in b) or b[0] > b[1] for b in bounds):
        raise ValueError("one ordered integer bound pair required per video frame")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    ve = vs + f * tpf
    globals_ = _union(((0, vs), (ve, s)))
    anchors = {0, f - 1} if f else set()
    row_anchors = anchors if anchor_frames in ("rows", "both") else set()
    col_anchors = anchors if anchor_frames in ("columns", "both") else set()
    dense_ranges, groups = list(globals_), []
    for frame in range(f):
        qr = (vs + frame * tpf, vs + (frame + 1) * tpf)
        lo, hi = bounds[frame]
        kr = _union(list(globals_) + [(vs + max(0, lo) * tpf, vs + min(f, hi + 1) * tpf)]
                    + [(vs + a * tpf, vs + (a + 1) * tpf) for a in col_anchors])
        if frame in row_anchors or kr == ((0, s),):
            dense_ranges.append(qr)
        else:
            if not kr:
                raise ValueError(f"video frame {frame} has no visible keys")
            if _count(kr) > max_gather_rows:
                raise ValueError(f"visible set requires {_count(kr)} gather rows, budget={max_gather_rows}; increase max_gather_rows")
            if groups and groups[-1].key_ranges == kr and groups[-1].query_ranges[-1][1] == qr[0]:
                groups[-1] = VisibleGroup(_union(groups[-1].query_ranges + (qr,)), kr)
            else:
                groups.append(VisibleGroup((qr,), kr))

    def indices(ranges):
        # Build on CPU then transfer once, avoiding one GPU allocation per range.
        return torch.cat([torch.arange(a, b, dtype=torch.long) for a, b in ranges]).to(device)

    def make_pass(gs, dense=False):
        qlens, klens = [_count(g.query_ranges) for g in gs], [_count(g.key_ranges) for g in gs]
        def cumulative(lengths):
            return torch.tensor([0] + list(accumulate(lengths)), dtype=torch.int32, device=device)
        return PackedPass(tuple(gs), indices(tuple(r for g in gs for r in g.query_ranges)),
                          None if dense else indices(tuple(r for g in gs for r in g.key_ranges)),
                          cumulative(qlens), cumulative(klens), max(qlens), max(klens), dense)

    passes = [make_pass([VisibleGroup(qr, ((0, s),))], True)
              for qr in _split(_union(dense_ranges), max_query_rows)]
    current, nq, nk = [], 0, 0
    for group in groups:
        for qr in _split(group.query_ranges, max_query_rows):
            small = VisibleGroup(qr, group.key_ranges)
            qlen, klen = _count(qr), _count(small.key_ranges)
            if current and (nq + qlen > max_query_rows or nk + klen > max_gather_rows):
                passes.append(make_pass(current))
                current, nq, nk = [], 0, 0
            current.append(small)
            nq, nk = nq + qlen, nk + klen
    if current:
        passes.append(make_pass(current))
    if sum(p.query_rows.numel() for p in passes) != s:
        raise AssertionError("internal plan coverage mismatch")
    return PaperWindowPlan(layout, bounds, anchor_frames, max_gather_rows, max_query_rows, device, tuple(passes))


@contextmanager
def _stage(name, enabled):
    if enabled:
        torch.cuda.nvtx.range_push("paper_attention/" + name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def paper_window_attention(query, key, value, plan: PaperWindowPlan, scale=None, *,
                           enabled=False, return_lse=False, nvtx=True):
    """[used+padding,H,D] -> same layout, fresh output; padding is zero.

    Explicit enabled=True AND no_grad/inference_mode are required. Disabled
    calls raise: caller must retain its existing path. Native public Torch
    varlen only, no fallback, global dense mask, FA4 import or arch override.
    Tested native SM120 BF16/FP16, H=7,D=128. H arbitrary positive, D=64/128/256.
    Return (output,lse) when requested; LSE is FP32 [rows,H], natural log,
    negative infinity on padding. No key splits, hence no LSE merge is needed.
    num_splits=1 fixes native split-KV policy independently of pass grouping.
    Inputs may be QKV/head slices with unit innermost stride. No input mutation,
    output gating, collectives or head reordering occur in this function.
    """
    if not enabled:
        raise RuntimeError("paper window attention is disabled; retain the existing caller path or explicitly enable")
    if torch.is_grad_enabled():
        raise RuntimeError("paper window attention is inference-only; use no_grad/inference_mode")
    if not isinstance(plan, PaperWindowPlan):
        raise TypeError("plan must be prepared with prepare_paper_window_plan")
    if any(x.ndim != 3 or x.shape != query.shape or x.dtype != query.dtype or x.device != query.device for x in (query, key, value)):
        raise ValueError("Q/K/V must have identical [rows,H,D], dtype and device")
    if query.shape[0] < plan.layout.seq_len or query.shape[1] <= 0 or query.shape[2] not in (64, 128, 256):
        raise ValueError("invalid Q/K/V geometry or insufficient used rows")
    if query.dtype not in (torch.bfloat16, torch.float16) or any(x.stride(-1) != 1 for x in (query, key, value)):
        raise ValueError("native varlen requires BF16/FP16 and unit innermost stride")
    if not query.is_cuda or query.device != plan.device:
        raise ValueError("Q/K/V and prepared plan must be on the same CUDA device")
    if torch.compiler.is_compiling():
        raise RuntimeError("paper window attention requires eager inference")
    if torch.cuda.get_device_capability(query.device) != (12, 0):
        raise RuntimeError("paper window attention requires native SM120")
    scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    from torch.nn.attention.varlen import AuxRequest, varlen_attn

    with torch.cuda.device(query.device), _stage("allocate", nvtx):
        out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        lse = torch.empty(query.shape[:2], dtype=torch.float32, device=query.device) if return_lse else None
        out[plan.layout.seq_len:].zero_()
        if lse is not None:
            lse[plan.layout.seq_len:].fill_(-float("inf"))
    for p in plan.passes:
        with _stage("dense_gather" if p.dense else "window_gather", nvtx):
            q = query.index_select(0, p.query_rows)
            k = key[:plan.layout.seq_len] if p.dense else key.index_select(0, p.key_rows)
            v = value[:plan.layout.seq_len] if p.dense else value.index_select(0, p.key_rows)
        with _stage("dense_varlen" if p.dense else "window_varlen", nvtx):
            result = varlen_attn(q, k, v, p.cu_q, p.cu_k, p.max_q, p.max_k,
                                 scale=scale, num_splits=1,
                                 return_aux=AuxRequest(lse=True) if return_lse else None)
        with _stage("scatter", nvtx):
            if return_lse:
                result, local_lse = result
                lse.index_copy_(0, p.query_rows, local_lse.transpose(0, 1))
            out.index_copy_(0, p.query_rows, result)
        # Free pass buffers before allocating the next pass, not after RHS gather.
        del q, k, v, result
        if return_lse:
            del local_lse
    return (out, lse) if return_lse else out
