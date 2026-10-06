"""Inference-only fused epilogue: RMSNorm + output gate in one pass."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_LINEAR_EPILOGUE_CACHE = {}


def _linear_epilogue_body(readout, weight, gate, eps):
    """RMSNorm + the output gate, as one expression. See `linear_epilogue`."""
    ms = torch.linalg.vector_norm(
        readout, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / readout.shape[-1]
    normed = readout * torch.rsqrt(ms + eps).to(readout.dtype) * weight.to(readout.dtype)
    return (normed * gate).reshape(readout.shape[0], -1)


def _linear_epilogue_fhsd_body(readout, weight, gate, eps):
    """`_linear_epilogue_body` for a readout still in [F, H, S, d], with the transpose back
    to token order folded into the store."""
    ms = torch.linalg.vector_norm(
        readout, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / readout.shape[-1]
    normed = readout * torch.rsqrt(ms + eps).to(readout.dtype) * weight.to(readout.dtype)
    frames, heads, per_frame, dim = normed.shape
    rows = frames * per_frame
    return (normed.permute(0, 2, 1, 3).reshape(rows, heads * dim)
            * gate.reshape(rows, heads * dim))


def linear_epilogue(readout, weight, gate, eps, inference=False, fhsd=False):
    """The branch's last two steps, fused when the caller says it is inferring.

    Eager this is a noticeable share of the linear branch for arithmetic that is pure
    bandwidth: the readout is [F*S, H, d] = ~1.4 GiB at H3 scale, and the norm, the
    gate product and the multiply each walk all of it and write all of it back. One
    kernel reads it once.

    `fhsd=True` takes the readout as [F, H, S, d] and folds the transpose back to token
    order into the store: with q kept in [F,H,S,d] the readout is a plain batched
    matmul instead of an einsum that permutes q in and the result out, and the epilogue
    was writing a full copy anyway. The two layouts are BITWISE equal -- both spellings
    do the same multiplies on the same values, only the traversal order differs.

    NOT BITWISE against the eager spelling -- inductor keeps the reciprocal and the two
    multiplies in fp32 and rounds once at the store. Same class of difference as
    `_qk_prep`, and in the same direction (fewer roundings, not more), which is why
    the training body keeps the eager one: a teacher whose job is to be numerically
    comparable to a student must not quietly get the better arithmetic.
    """
    body = _linear_epilogue_fhsd_body if fhsd else _linear_epilogue_body

    if not inference:
        out = body(readout, weight, gate, eps)
    else:
        key = "fhsd" if fhsd else "fn"
        if key not in _LINEAR_EPILOGUE_CACHE:
            _LINEAR_EPILOGUE_CACHE[key] = torch.compile(body, dynamic=False)
        out = _LINEAR_EPILOGUE_CACHE[key](readout, weight, gate, eps)

    return out


@triton.jit
def _vda_bf16_epilogue_kernel(X, W, G, Y, N: tl.constexpr, H: tl.constexpr,
                              XS0: tl.constexpr, XS1: tl.constexpr,
                              GS0: tl.constexpr, GS1: tl.constexpr,
                              EPS: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    col = tl.arange(0, D)
    x = tl.load(X + (row // H)[:, None] * XS0 + (row % H)[:, None] * XS1
                + col[None, :], row[:, None] < N, 0).to(tl.float32)
    # Preserve the eager BF16 rounding boundaries instead of rounding once at store.
    norm = tl.sqrt(tl.sum(x * x, 1))
    ms = norm * norm / D
    inv = tl.rsqrt(ms + EPS).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + col).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(G + (row // H)[:, None] * GS0 + (row % H)[:, None] * GS1
                   + col[None, :], row[:, None] < N, 0).to(tl.float32)
    y = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    y = (y * w[None, :]).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row[:, None] * D + col[None, :], y * gate, row[:, None] < N)


def vda_bf16_epilogue(readout, weight, gate, eps):
    """Experimental CUDA BF16 RMSNorm/gate fusion, retaining eager cast boundaries.

    Token/head strides may be noncontiguous (including packed Ulysses gates).
    FP32 reduction order can still differ from ATen; this is not a bitwise promise.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("vda_bf16_epilogue requires torch.no_grad()")
    if (readout.ndim != 3 or readout.shape[-1] != 128
            or gate.shape != readout.shape or weight.shape != (128,)):
        raise ValueError("expected readout/gate [tokens, heads, 128] and weight [128]")
    if (readout.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16
            or not readout.is_cuda or gate.device != readout.device
            or weight.device != readout.device or readout.stride(-1) != 1
            or gate.stride(-1) != 1 or not weight.is_contiguous()):
        raise ValueError("expected CUDA BF16 readout/gate with contiguous last dimension")
    tokens, heads, dim = readout.shape
    out = torch.empty((tokens, heads * dim), device=readout.device, dtype=readout.dtype)
    _vda_bf16_epilogue_kernel[(triton.cdiv(tokens * heads, 4),)](
        readout, weight, gate, out, tokens * heads, heads,
        *readout.stride()[:2], *gate.stride()[:2], eps, dim,
        num_warps=4, enable_fp_fusion=False)
    return out

