# SPDX-License-Identifier: Apache-2.0
# Modified/adapted 2026-10-05 from SGLang PR #37903, pinned
# ff1ce11348673eea9b51165bd6b2da7238c958b4. See licenses/SGLang-NOTICE.txt.
"""Explicit inference-only VDA paper stack, with no reference fallback.

Preparation/gather/epilogue are the pinned SGLang kernels. Joint statistics,
parallel chunk composition and the single-launch bidirectional compact scan are
new adaptations: upstream uses two GEMMs and Python-loop boundary scanning.
All inverse, alpha, composition and recurrence storage/arithmetic stay FP32.
"""
from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl

from .paper_factor import paper_delta_factors
from .paper_sglang import (vdn_silu_l2norm, vdn_temporal_conv_act,
                           vdn_linear_epilogue, _gather_state_kernel)


def _cuda(t, name, dtype=None):
    if not t.is_cuda or (dtype is not None and t.dtype != dtype):
        raise ValueError(f"{name} must be CUDA {dtype or 'floating point'}")
    if t.requires_grad:
        raise ValueError(f"{name}: paper stack is inference only")


def paper_prepare(tokens, *, frames, spatial, l2norm, conv=None, proj=None,
                  frame_size=None):
    """Raw [F*S,H,D] -> contiguous [F,H,S,D], including temporal conv if set.

    Spatial depthwise convolution remains cuDNN, as in the released author path.
    Temporal conv, SiLU, L2 and the layout store are one kernel per projection.
    Prompt callers omit conv. Arithmetic rounds once at the BF16/FP32 store.
    """
    _cuda(tokens, "tokens")
    if tokens.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("paper_prepare supports BF16/FP32")
    if tokens.ndim != 3 or tokens.shape[0] != frames * spatial or min(frames, spatial) < 1:
        raise ValueError("tokens must be [frames*spatial, heads, dim] with positive sizes")
    if conv is not None and proj in conv.projs:
        if frame_size is None or frame_size[0] * frame_size[1] != spatial:
            raise ValueError("convolution requires the exact spatial frame_size")
        x, w = conv.spatial(proj, tokens, frames, frame_size)
        return vdn_temporal_conv_act(x, w, *tokens.shape[-2:], l2norm, frame_major=True)
    return vdn_silu_l2norm(tokens, l2norm, per_frame=spatial)


@triton.jit
def _joint_stats(K, V, Beta, A, B, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 BS: tl.constexpr, BD: tl.constexpr, BF16: tl.constexpr):
    fh = tl.program_id(0)
    r = tl.program_id(1) * BD + tl.arange(0, BD)
    c = tl.program_id(2) * BD + tl.arange(0, BD)
    ss = tl.arange(0, BS)
    aa = tl.zeros((BD, BD), tl.float32)
    aa_t = tl.zeros((BD, BD), tl.float32)
    bb = tl.zeros((BD, BD), tl.float32)
    for start in range(tl.cdiv(S, BS)):
        s = start * BS + ss
        b = tl.load(Beta + fh * S + s, s < S, 0).to(tl.float32)
        kl = tl.load(K + (fh * S + s[None, :]) * D + r[:, None],
                     (s[None, :] < S) & (r[:, None] < D), 0).to(tl.float32)
        kr = tl.load(K + (fh * S + s[:, None]) * D + c[None, :],
                     (s[:, None] < S) & (c[None, :] < D), 0).to(tl.float32)
        vl = tl.load(V + (fh * S + s[None, :]) * D + r[:, None],
                     (s[None, :] < S) & (r[:, None] < D), 0).to(tl.float32)
        aa += tl.dot(kl * b[None, :], kr, input_precision="tf32x3")
        aa_t += tl.dot(kl, kr * b[:, None], input_precision="tf32x3")
        if BF16:
            # Preserve the reference B operand and result rounding, not FP32 B.
            vb = (vl * b.to(tl.bfloat16).to(tl.float32)[None, :]).to(tl.bfloat16)
            bb += tl.dot(vb, kr.to(tl.bfloat16))
        else:
            bb += tl.dot(vl * b[None, :], kr, input_precision="tf32x3")
    off = fh * D * D + r[:, None] * D + c[None, :]
    mask = (r[:, None] < D) & (c[None, :] < D)
    tl.store(A + off, (aa + aa_t) * 0.5, mask)
    if BF16:
        bb = bb.to(tl.bfloat16).to(tl.float32)
    tl.store(B + off, bb, mask)


def paper_statistics(key, value, beta):
    """Joint one-pass tiled A/B reduction, no full-size prologue operands.

    Contiguous [F,H,S,D] BF16 or FP32 K/V and [F,H,S] beta. A is symmetrized
    FP32; B preserves BF16 scaled-V and final BF16 GEMM rounding when K/V are
    BF16. Both returned banks are FP32. Inputs are read per matrix tile, not
    literally once globally. Shared K/beta tiles feed both reductions.
    """
    _cuda(key, "key")
    _cuda(value, "value")
    _cuda(beta, "beta")
    if (key.ndim != 4 or value.shape != key.shape or beta.shape != key.shape[:-1]
            or value.dtype != key.dtype or key.dtype not in (torch.bfloat16, torch.float32)
            or beta.dtype not in (torch.bfloat16, torch.float32)
            or any(t.device != key.device or not t.is_contiguous() for t in (value, beta, key))):
        raise ValueError("joint statistics require contiguous compatible F,H,S,D inputs")
    f, h, s, d = key.shape
    if min(f, h, s) < 1 or d not in (16, 32, 64, 128):
        raise ValueError("positive F,H,S and D in 16,32,64,128 required")
    a = torch.empty((f, h, d, d), device=key.device, dtype=torch.float32)
    b = torch.empty_like(a)
    _joint_stats[(f * h, triton.cdiv(d, 32), triton.cdiv(d, 32))](
        key, value, beta, a, b, s, h, d, 32, 32, key.dtype == torch.bfloat16,
        num_warps=4, enable_fp_fusion=False)
    return a, b


@dataclass(frozen=True)
class BoundaryPlan:
    frames: int
    cuts: tuple
    starts: torch.Tensor
    ends: torch.Tensor
    before: torch.Tensor
    after: torch.Tensor
    has_before: torch.Tensor
    has_after: torch.Tensor
    bridge_before: torch.Tensor
    bridge_after: torch.Tensor

    @property
    def chunks(self):
        return len(self.cuts) - 1


@lru_cache(maxsize=64)
def _boundary_plan(bounds, frames, chunk_size, device):
    if frames < 1 or len(bounds) != frames or chunk_size < 1:
        raise ValueError("positive frames/chunk_size and one bound per frame required")
    cuts = {0, frames}
    for f, (lo, hi) in enumerate(bounds):
        if not lo <= f <= hi:
            raise ValueError("each inclusive window must contain its query frame")
        cuts.update((max(0, lo), min(frames, hi + 1)))
    required = sorted(cuts)
    # Split long affine compositions without assuming a window grid or offset.
    for lo, hi in zip(required[:-1], required[1:]):
        cuts.update(range(lo + chunk_size, hi, chunk_size))
    cuts = tuple(sorted(cuts))
    pos = {f: i for i, f in enumerate(cuts)}
    def tensor(items):
        return torch.tensor(items, dtype=torch.int32, device=device)
    before = [max(0, pos[max(0, lo)] - 1) for lo, _ in bounds]
    after = [min(len(cuts) - 2, pos[min(frames, hi + 1)]) for _, hi in bounds]
    return BoundaryPlan(frames, cuts, tensor(cuts[:-1]), tensor(cuts[1:]),
                        tensor(before), tensor(after), tensor([lo > 0 for lo, _ in bounds]),
                        tensor([hi + 1 < frames for _, hi in bounds]),
                        tensor([max(0, lo) for lo, _ in bounds]),
                        tensor([min(frames, hi + 1) for _, hi in bounds]))


def paper_boundary_plan(bounds, frames, device, chunk_size=8):
    """Cached compact boundary plan, exact for irregular and anchor-rebased windows."""
    return _boundary_plan(tuple(tuple(b) for b in bounds), frames, chunk_size, str(device))


@triton.jit
def _compose(T, I, Starts, Ends, CT, CI, H: tl.constexpr, C: tl.constexpr,
             D: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    hc = tl.program_id(1)
    direction = tl.program_id(2)
    h, c = hc % H, hc // H
    col = tl.arange(0, D)
    lo, hi = tl.load(Starts + c), tl.load(Ends + c)
    m = (row[:, None] == col[None, :]).to(tl.float32)
    b = tl.zeros((R, D), tl.float32)
    for step in range(hi - lo):
        f = tl.where(direction == 0, lo + step, hi - 1 - step)
        t = tl.load(T + ((f * H + h) * D + col[:, None]) * D + col[None, :])
        inj = tl.load(I + ((f * H + h) * D + row[:, None]) * D + col[None, :])
        m = tl.dot(m, t, input_precision="ieee")
        b = tl.dot(b, t, input_precision="ieee") + inj
    off = (((direction * C + c) * H + h) * D + row[:, None]) * D + col[None, :]
    tl.store(CT + off, m)
    tl.store(CI + off, b)


@triton.jit
def _scan_both(CT, CI, Seed, Bank, H: tl.constexpr, C: tl.constexpr,
               D: tl.constexpr, R: tl.constexpr, HAS_SEED: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    h, direction = tl.program_id(1), tl.program_id(2)
    col = tl.arange(0, D)
    state = tl.zeros((R, D), tl.float32)
    if HAS_SEED:
        state = tl.load(Seed + (h * D + row[:, None]) * D + col[None, :])
    for step in range(C):
        c = tl.where(direction == 0, step, C - 1 - step)
        base = ((direction * C + c) * H + h) * D * D
        trans = tl.load(CT + base + col[:, None] * D + col[None, :])
        inj = tl.load(CI + base + row[:, None] * D + col[None, :])
        state = tl.dot(state, trans, input_precision="ieee") + inj
        tl.store(Bank + base + row[:, None] * D + col[None, :], state)


def paper_boundary_scan(transitions, injections, plan, text_state=None):
    """Two launches: parallel affine composition, then BOTH scans in one launch.

    Return [2,C,H,D,D] FP32 compact states, not full-frame prefix/suffix banks.
    Arbitrary windows may require C=F; normal chunked windows have C << F.
    The directional seed is supplied already scaled (half prompt injection).
    """
    _cuda(transitions, "transitions", torch.float32)
    _cuda(injections, "injections", torch.float32)
    if (transitions.ndim != 4 or transitions.shape != injections.shape
            or transitions.device != injections.device
            or not transitions.is_contiguous() or not injections.is_contiguous()):
        raise ValueError("matching contiguous FP32 [F,H,D,D] factors required")
    f, h, d, dk = transitions.shape
    if f != plan.frames or d != dk or d not in (16, 32, 64, 128) or h < 1:
        raise ValueError("factor geometry incompatible with boundary plan")
    if plan.starts.device != transitions.device:
        raise ValueError("plan and factors must share a device")
    if text_state is not None:
        _cuda(text_state, "text_state", torch.float32)
        if text_state.shape != (h, d, d) or text_state.device != transitions.device:
            raise ValueError("text_state must be head-local H,D,D")
        text_state = text_state.contiguous()
    shape = (2, plan.chunks, h, d, d)
    ct = transitions.new_empty(shape)
    ci = transitions.new_empty(shape)
    bank = transitions.new_empty(shape)
    _compose[(d // 16, plan.chunks * h, 2)](
        transitions, injections, plan.starts, plan.ends, ct, ci, h, plan.chunks,
        d, 16, num_warps=4, num_stages=1, enable_fp_fusion=False)
    _scan_both[(d // 16, h, 2)](
        ct, ci, transitions if text_state is None else text_state, bank, h,
        plan.chunks, d, 16, text_state is not None, num_warps=4, num_stages=1,
        enable_fp_fusion=False)
    return bank


def paper_gather(bank, alpha, plan, text_state=None, out_dtype=torch.bfloat16,
                 bridge=True):
    """Author gather/decay kernel with compact bank indices instead of frame indices."""
    _cuda(bank, "bank", torch.float32)
    _cuda(alpha, "alpha", torch.float32)
    if bank.ndim != 5 or bank.shape[:2] != (2, plan.chunks):
        raise ValueError("bank must be [2,chunks,H,D,D]")
    _, _, h, d, dk = bank.shape
    if (d != dk or alpha.shape != (plan.frames, h, d) or not bank.is_contiguous()
            or alpha.device != bank.device or plan.starts.device != bank.device
            or out_dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("gather geometry/device/dtype mismatch")
    if text_state is not None:
        _cuda(text_state, "text_state", torch.float32)
        if text_state.shape != (h, d, d) or text_state.device != bank.device:
            raise ValueError("text_state geometry/device mismatch")
        text_state = text_state.contiguous()
    logp = torch.cat((torch.zeros_like(alpha[:1]), alpha.clamp_min(1e-12).log().cumsum(0)))
    out = torch.empty((plan.frames, h, d, d), device=bank.device, dtype=out_dtype)
    _gather_state_kernel[(plan.frames, h, triton.cdiv(d, 16))](
        bank[0], bank[1], logp, bank if text_state is None else text_state,
        plan.before, plan.after, plan.has_before, plan.has_after,
        plan.bridge_before, plan.bridge_after, out, h, d,
        text_state is not None, bridge, 16, d, num_warps=4, enable_fp_fusion=False)
    return out


def paper_readout(branch, xv, num_frames, tokens_per_frame, bounds, qkv_raw,
                  frame_size=None, text_x=None, text_qkv_raw=None, heads=None,
                  beta=None, gate=None, frame_mean=None, text_beta=None):
    """Branch body. The existing forward owns anchor pruning and zero scattering."""
    from .branch import _HeadSliceSepConv

    conv = branch.short_conv
    if conv is not None and heads is not None:
        conv = _HeadSliceSepConv(conv, heads, branch.head_dim)
    q, k, v = [paper_prepare(t, frames=num_frames, spatial=tokens_per_frame,
                             l2norm=p != "v", conv=conv, proj=p, frame_size=frame_size)
               for p, t in zip(("q", "k", "v"), qkv_raw)]
    h = q.shape[1]
    if beta is None:
        beta = torch.sigmoid(branch.beta_proj(xv))
    beta = beta.view(num_frames, tokens_per_frame, h).permute(0, 2, 1).contiguous()
    a, b = paper_statistics(k, v, beta)
    del k, v
    if frame_mean is None:
        frame_mean = xv.view(num_frames, tokens_per_frame, -1).mean(1, dtype=torch.float32)
    alpha = branch.alpha(frame_mean, heads=heads).float().contiguous()
    has_text = text_x is not None or text_beta is not None
    if has_text:
        if text_qkv_raw is None or text_qkv_raw[1].shape[0] < 1:
            raise ValueError("prompt inputs require nonempty text QKV")
        length = text_qkv_raw[1].shape[0]
        tk = paper_prepare(text_qkv_raw[1], frames=1, spatial=length, l2norm=True)
        tv = paper_prepare(text_qkv_raw[2], frames=1, spatial=length, l2norm=False)
        if text_beta is None:
            text_beta = torch.sigmoid(branch.beta_proj(text_x))
        tb = text_beta.view(1, length, h).permute(0, 2, 1).contiguous()
        ta, tb = paper_statistics(tk, tv, tb)
        # The virtual frame participates in the SAME unscaled factor batch.
        # It is removed before scanning, not inserted as an extra video step.
        a, b = torch.cat((ta, a)), torch.cat((tb, b))
        factor_alpha = torch.cat((torch.ones_like(alpha[:1]), alpha))
    else:
        factor_alpha = alpha
    transitions, injections = paper_delta_factors(a, b, factor_alpha)
    del a, b
    seed = None
    if has_text:
        seed = injections[0] * branch.TEXT_STATE_SCALE
        transitions, injections = transitions[1:], injections[1:]
    plan = paper_boundary_plan(bounds, num_frames, q.device, branch.paper_vda_chunk_size)
    bank = paper_boundary_scan(transitions, injections, plan, seed)
    del transitions, injections
    state = paper_gather(bank, alpha, plan, seed, q.dtype, branch.bridge == "alpha")
    del bank
    if gate is None:
        gate = branch.output_gate(xv)
    readout = torch.matmul(q, state.transpose(-1, -2))
    return vdn_linear_epilogue(readout, branch.norm.weight, gate, branch.norm.eps)
