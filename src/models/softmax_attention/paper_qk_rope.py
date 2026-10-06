"""Default-off, inference-only QK RMSNorm/RoPE for the paper stack.

Semantic adaptation of SGLang PR #37903, merge
ff1ce11348673eea9b51165bd6b2da7238c958b4, qknorm_rope.cuh and
qknorm_rope_jit.py (Apache-2.0; see licenses/SGLang-LICENSE.txt and
licenses/SGLang-NOTICE.txt). This is a new Triton implementation, not the
upstream CUDA kernel. It preserves the upstream out-of-place VDA contract,
but uses installed Diffusers MiniMax-H3's full-width [T, R] cosine/sine tables
and its activation-dtype rotary arithmetic, including for FP32 tables.

No environment variable enables this API. ``enabled=False`` runs the eager
reference, not a no-op. ``enabled=True`` requires SM120 and never falls back.
Neither importing this module nor CPU reference execution initializes CUDA.
"""

from contextlib import nullcontext
from functools import lru_cache
import math
from numbers import Real

import torch


_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _validate(q, k, q_weight, k_weight, eps, cos, sin, enabled, nvtx):
    if not isinstance(enabled, bool) or not isinstance(nvtx, bool):
        raise TypeError("enabled and nvtx must be bool")
    if torch.is_grad_enabled():
        raise RuntimeError("paper_qk_rmsnorm_rope is inference-only; use torch.no_grad()")
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise TypeError("eps must be a finite positive real scalar")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    for name, value in (("q", q), ("k", k), ("q_weight", q_weight), ("k_weight", k_weight)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.layout != torch.strided:
            raise ValueError(f"{name} must have strided layout")
    if q.ndim != 3 or k.shape != q.shape:
        raise ValueError("q and k must have identical [T, H, D] shapes")
    t, h, d = q.shape
    if h <= 0 or d not in (64, 128, 256):
        raise ValueError("H must be positive and D must be 64, 128, or 256")
    if q.dtype not in _DTYPES or k.dtype != q.dtype:
        raise TypeError("q and k must share a BF16, FP16, or FP32 dtype")
    if q.device.type not in ("cpu", "cuda") or k.device != q.device:
        raise ValueError("q and k must share a CPU or CUDA device")
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if weight.shape != (d,):
            raise ValueError(f"{name} must have shape [D], shared by every head")
        if weight.device != q.device:
            raise ValueError(f"{name} must be on the q/k device")
        if weight.dtype not in (q.dtype, torch.float32):
            raise TypeError(f"{name} must have the activation dtype or FP32")
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must both be supplied or both be None")
    if cos is not None:
        for name, table in (("cos", cos), ("sin", sin)):
            if not isinstance(table, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if table.layout != torch.strided or table.ndim != 2:
                raise ValueError(f"{name} must have strided shape [T, R]")
            if table.device != q.device:
                raise ValueError(f"{name} must be on the q/k device")
            if table.dtype not in _DTYPES:
                raise TypeError(f"{name} must be BF16, FP16, or FP32")
        if cos.shape != sin.shape or cos.shape[0] != t:
            raise ValueError("cos and sin must have identical [T, R] shapes")
        r = cos.shape[1]
        if r <= 0 or r > d or r % 2:
            raise ValueError("rotary width R must be positive, even, and at most D")


def _stage(name, tensor, nvtx):
    if nvtx and tensor.is_cuda:
        return torch.cuda.nvtx.range("paper_stack.qk_rope." + name)
    return nullcontext()


def _eager_norm(x, weight, eps):
    # nn.RMSNorm used by MiniMaxH3Attention multiplies the weight in FP32,
    # then casts the complete weighted result, not the unweighted norm.
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
            * weight.float()).to(x.dtype).contiguous()


def _eager_rope(x, cos, sin):
    r = cos.shape[-1]
    a, b = x[..., :r].chunk(2, dim=-1)
    rotated = torch.cat((-b, a), dim=-1)
    c, s = cos.to(x.dtype)[:, None, :], sin.to(x.dtype)[:, None, :]
    return torch.cat((x[..., :r] * c + rotated * s, x[..., r:]), dim=-1).contiguous()


@lru_cache(maxsize=1)
def _get_kernel():
    # Lazy import: the default-off path neither requires Triton nor selects a
    # CUDA driver. The launch uses the real current-device target, never sm100.
    import triton
    import triton.language as tl

    @triton.jit
    def fused(
        Q, K, QW, KW, C, S, QO, KO,
        QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
        KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
        QWS: tl.constexpr, KWS: tl.constexpr,
        CS0: tl.constexpr, CS1: tl.constexpr,
        SS0: tl.constexpr, SS1: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, EPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        token, head = row // H, row % H
        lane = tl.arange(0, D)
        if tl.program_id(1) == 0:
            x = tl.load(Q + token * QS0 + head * QS1 + lane * QS2).to(tl.float32)
            w = tl.load(QW + lane * QWS).to(tl.float32)
            out = QO + row * D + lane
        else:
            x = tl.load(K + token * KS0 + head * KS1 + lane * KS2).to(tl.float32)
            w = tl.load(KW + lane * KWS).to(tl.float32)
            out = KO + row * D + lane
        inv = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
        dtype = QO.dtype.element_ty
        # An intentional activation-dtype boundary BEFORE rotary arithmetic.
        norm = ((x * inv) * w).to(dtype).to(tl.float32)
        if R > 0:
            partner = tl.where(lane < R, tl.where(lane < R // 2, lane + R // 2, lane - R // 2), lane)
            other = tl.gather(norm, partner, axis=0)
            other = tl.where(lane < R // 2, -other, other)
            c = tl.load(C + token * CS0 + lane * CS1, lane < R, other=0).to(dtype).to(tl.float32)
            s = tl.load(S + token * SS0 + lane * SS1, lane < R, other=0).to(dtype).to(tl.float32)
            # Eager _apply_rotary_emb rounds each product as well as the sum.
            lhs = (norm * c).to(dtype).to(tl.float32)
            rhs = (other * s).to(dtype).to(tl.float32)
            norm = tl.where(lane < R, lhs + rhs, norm)
        tl.store(out, norm.to(dtype))

    return fused


def paper_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    *,
    enabled: bool = False,
    nvtx: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return new contiguous normalized/rotated Q and K; never mutate inputs.

    q/k: identical [T,H,D] strided views, including fused-QKV/head slices and
        nonunit innermost strides. D is 64/128/256; H=7,D=128 is supported.
    q_weight/k_weight: [D], shared across heads, activation dtype or FP32.
    cos/sin: [T,R] full rotary width, even 0<R<=D, broadcast across heads.
        Values are cast to the activation dtype, including FP32 tables.
        Both None selects norm-only. No position lookup or compact cache.
    eps: one positive finite epsilon shared by Q/K; differing norm epsilons
        must be reconciled by the caller, not silently replaced.

    BF16/FP16/FP32 activations are supported. FP32 RMS reduction, rescaling,
    and weight multiplication precede activation rounding. Half-split rotary
    multiplies and sum each round to the activation dtype. FP32 reductions
    may differ in last bits across backends; bitwise CUDA parity is not promised.

    Requires an explicit no_grad/inference_mode scope even for detached inputs.
    Disabled: eager CPU/CUDA implementation with identical semantic boundaries.
    Enabled: native SM120 Triton only, no eager fallback or architecture spoofing.
    CUDA NVTX stages are eager_norm/eager_rope or allocate/fused_norm_rope
    (fused_norm for norm-only). No dense attention or distributed work is done.
    """
    _validate(q, k, q_weight, k_weight, eps, cos, sin, enabled, nvtx)
    if not enabled:
        with _stage("eager_norm", q, nvtx):
            qo, ko = _eager_norm(q, q_weight, eps), _eager_norm(k, k_weight, eps)
        if cos is not None:
            with _stage("eager_rope", q, nvtx):
                qo, ko = _eager_rope(qo, cos, sin), _eager_rope(ko, cos, sin)
        return qo, ko

    if not q.is_cuda:
        raise ValueError("enabled=True requires CUDA on native SM120")
    if torch.compiler.is_compiling():
        raise RuntimeError("enabled=True requires eager inference, not compiler tracing")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("enabled=True requires native SM120; no architecture override")
    with _stage("allocate", q, nvtx):
        qo = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        ko = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if q.shape[0] == 0:
        return qo, ko
    r = 0 if cos is None else cos.shape[1]
    cs = (0, 0) if cos is None else cos.stride()
    ss = (0, 0) if sin is None else sin.stride()
    # Dummy pointers are unused in the constexpr norm-only specialization.
    c, s = (q, q) if cos is None else (cos, sin)
    with torch.cuda.device(q.device), _stage("fused_norm_rope" if r else "fused_norm", q, nvtx):
        _get_kernel()[(q.shape[0] * q.shape[1], 2)](
            q, k, q_weight, k_weight, c, s, qo, ko,
            *q.stride(), *k.stride(), q_weight.stride(0), k_weight.stride(0),
            *cs, *ss, q.shape[1], q.shape[2], r, float(eps),
            num_warps=4, enable_fp_fusion=False,
        )
    return qo, ko
