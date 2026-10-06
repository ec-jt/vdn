# SPDX-License-Identifier: Apache-2.0
"""SGLang-derived fused MXFP8 quantizer; see mxfp8_quant.py attribution.

Local changes: uint8 payload stores, masked input with fully written 128-padded
output, and zero-filled padded scale groups. Exponent rule/swizzle are upstream.
"""
import torch
import triton
import triton.language as tl

from .mxfp8_quant import padded_shape, scale_numel


@triton.jit
def _mx_e8m0_from_amax(amax):
    bits = amax.to(tl.int32, bitcast=True)
    e0 = ((bits >> 23) & 0xFF) - 135
    thr = (((e0 + 135) << 23) | 0x600000).to(tl.float32, bitcast=True)
    e = tl.maximum(e0 + (amax > thr).to(tl.int32), -127)
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    return e + 127, inv


@triton.jit
def _mx_scale_offsets(r, c, n_col_blocks):
    tile = (r // 128) * n_col_blocks + c // 4
    return tile * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4


@triton.jit
def _quantize(X, Q, S, R: tl.constexpr, K: tl.constexpr, RP: tl.constexpr,
              KP: tl.constexpr, STRIDE: tl.constexpr,
              BR: tl.constexpr = 32, G: tl.constexpr = 8):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    g = tl.program_id(1) * G + tl.arange(0, G)
    c = tl.program_id(1) * (G * 32) + tl.arange(0, G * 32)
    x = tl.load(X + r[:, None].to(tl.int64) * STRIDE + c[None, :],
                (r[:, None] < R) & (c[None, :] < K), other=0).to(tl.float32)
    blocks = tl.reshape(x, [BR, G, 32])
    scale, inv = _mx_e8m0_from_amax(tl.max(tl.abs(blocks), axis=2))
    q = tl.reshape(blocks * inv[:, :, None], [BR, G * 32])
    tl.store(Q + r[:, None].to(tl.int64) * KP + c[None, :],
             q.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
             (r[:, None] < RP) & (c[None, :] < KP))
    tl.store(S + _mx_scale_offsets(r[:, None], g[None, :], KP // 128),
             scale.to(tl.uint8), (r[:, None] < RP) & (g[None, :] < KP // 32))


def launch_quantize(x):
    r, k = x.shape
    rp, kp = padded_shape(r, k)
    q = torch.empty((rp, kp), dtype=torch.uint8, device=x.device)
    s = torch.empty(scale_numel(r, k), dtype=torch.uint8, device=x.device)
    if r:
        with torch.cuda.device(x.device):
            _quantize[(triton.cdiv(rp, 32), triton.cdiv(kp // 32, 8))](
                x, q, s, r, k, rp, kp, x.stride(0), num_warps=4)
    return q, s


@triton.jit
def _swiglu_mx(X, Q, S, FINITE, R: tl.constexpr, K: tl.constexpr,
               RP: tl.constexpr, KP: tl.constexpr, STRIDE: tl.constexpr,
               BR: tl.constexpr = 16, G: tl.constexpr = 8):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.program_id(1) * (G * 32) + tl.arange(0, G * 32)
    mask = (r[:, None] < R) & (c[None, :] < K)
    a = tl.load(X + r[:, None].to(tl.int64) * STRIDE + c[None, :], mask, 0).to(tl.float32)
    g = tl.load(X + r[:, None].to(tl.int64) * STRIDE + c[None, :] + K, mask, 0).to(tl.float32)
    # Match the compiled producer's BF16 store boundary, without storing it.
    v = (a * (g * tl.sigmoid(g))).to(tl.bfloat16).to(tl.float32)
    good = (tl.abs(a) < float('inf')) & (tl.abs(g) < float('inf')) & (tl.abs(v) < float('inf'))
    tl.store(FINITE + tl.program_id(0) * tl.num_programs(1) + tl.program_id(1),
             tl.min(tl.min(good.to(tl.int32), 1), 0))
    blocks = tl.reshape(v, [BR, G, 32])
    scale, inv = _mx_e8m0_from_amax(tl.max(tl.abs(blocks), 2))
    q = tl.reshape(blocks * inv[:, :, None], [BR, G * 32])
    tl.store(Q + r[:, None].to(tl.int64) * KP + c[None, :],
             q.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
             (r[:, None] < RP) & (c[None, :] < KP))
    group = tl.program_id(1) * G + tl.arange(0, G)
    tl.store(S + _mx_scale_offsets(r[:, None], group[None, :], KP // 128), scale.to(tl.uint8),
             (r[:, None] < RP) & (group[None, :] < KP // 32))


@triton.jit
def _prenorm_mx(X, W, SCALE, SHIFT, INDEX, Y, Q, S, FINITE,
                R: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                XS: tl.constexpr, SS: tl.constexpr, TS: tl.constexpr,
                EPS: tl.constexpr, WRITE_BF16: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + row.to(tl.int64) * XS + c, (row < R) & (c < K), 0).to(tl.float32)
    w = tl.load(W + c, c < K, 0).to(tl.float32)
    idx = tl.load(INDEX + row, row < R, 0)
    scale = tl.load(SCALE + idx * SS + c, (row < R) & (c < K), 0).to(tl.float32)
    shift = tl.load(SHIFT + idx * TS + c, (row < R) & (c < K), 0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) / K + EPS)
    value = ((x * inv_rms * w) * (1.0 + scale) + shift).to(tl.bfloat16)
    value = tl.where((row < R) & (c < K), value, 0)
    good = (tl.abs(x) < float('inf')) & (tl.abs(value.to(tl.float32)) < float('inf'))
    tl.store(FINITE + row, tl.min(good.to(tl.int32), 0))
    if WRITE_BF16:
        tl.store(Y + row.to(tl.int64) * K + c, value, (row < R) & (c < K))
    blocks = tl.reshape(value.to(tl.float32), [B // 32, 32])
    scale_bits, inv = _mx_e8m0_from_amax(tl.max(tl.abs(blocks), 1))
    q = tl.reshape(blocks * inv[:, None], [B])
    tl.store(Q + row.to(tl.int64) * KP + c, q.to(tl.float8e4nv).to(tl.uint8, bitcast=True), c < KP)
    group = tl.arange(0, B // 32)
    tl.store(S + _mx_scale_offsets(row, group, KP // 128), scale_bits.to(tl.uint8), group < KP // 32)


def launch_swiglu_quantize(x):
    r, twice_k = x.shape
    k = twice_k // 2
    rp, kp = padded_shape(r, k)
    q = torch.empty((rp, kp), dtype=torch.uint8, device=x.device)
    s = torch.empty(scale_numel(r, k), dtype=torch.uint8, device=x.device)
    grid = (triton.cdiv(rp, 16), triton.cdiv(kp, 256))
    finite = torch.empty(grid[0] * grid[1], dtype=torch.bool, device=x.device)
    if r:
        with torch.cuda.device(x.device):
            _swiglu_mx[grid](x, q, s, finite, r, k, rp, kp, x.stride(0), num_warps=4, enable_fp_fusion=False)
    return q, s, finite


def launch_prenorm_quantize(x, weight, eps, scale, shift, indices, *, write_bf16):
    r, k = x.shape
    rp, kp = padded_shape(r, k)
    q = torch.empty((rp, kp), dtype=torch.uint8, device=x.device)
    s = torch.empty(scale_numel(r, k), dtype=torch.uint8, device=x.device)
    y = torch.empty_like(x, memory_format=torch.contiguous_format) if write_bf16 else None
    finite = torch.empty(rp, dtype=torch.bool, device=x.device)
    if r:
        with torch.cuda.device(x.device):
            _prenorm_mx[(rp,)](x, weight, scale, shift, indices, y if y is not None else q,
                q, s, finite, r, k, kp, x.stride(0), scale.stride(0), shift.stride(0),
                eps, write_bf16, triton.next_power_of_2(kp), num_warps=8, enable_fp_fusion=False)
    return y, q, s, finite


@triton.jit
def _finite_parts(X, P, N: tl.constexpr, MODE: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0)
    if MODE == 1:
        good = (x & 127) != 127
    elif MODE == 2:
        good = x != 255
    else:
        good = tl.abs(x.to(tl.float32)) < float('inf')
    tl.store(P + tl.program_id(0), tl.min(good.to(tl.int32), 0))


def finite_parts(x, mode=0):
    # Noncontiguous tensors are uncommon small tables, never reinterpret strides.
    x = x.contiguous()
    parts = torch.empty(triton.cdiv(x.numel(), 4096), dtype=torch.bool, device=x.device)
    if x.numel():
        with torch.cuda.device(x.device):
            _finite_parts[(parts.numel(),)](x, parts, x.numel(), mode, 4096, num_warps=4)
    return parts
