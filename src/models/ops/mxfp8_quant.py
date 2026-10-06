# SPDX-License-Identifier: Apache-2.0
"""Block32 E4M3/E8M0 preparation, with byte storage for exact NCCL transport.

Scale bits and swizzle adapted from SGLang ff1ce11348673eea9b51165bd6b2da7238c958b4,
python/sglang/kernels/ops/diffusion/quantization/mxfp8_swizzled_triton.py.
Copyright SGLang contributors. See licenses/SGLang-LICENSE.txt and the local
artifacts/rtx5090/paper-stack-20261005/mxfp8/PROVENANCE.md for changes and hashes.
No SGLang runtime dependency. CPU preparation never initializes CUDA.
"""
from __future__ import annotations

import torch


def padded_shape(rows: int, k: int) -> tuple[int, int]:
    if rows < 0 or k <= 0:
        raise ValueError("rows must be nonnegative and K positive")
    return ((rows + 127) // 128 * 128, (k + 127) // 128 * 128)


def scale_numel(rows: int, k: int) -> int:
    rp, kp = padded_shape(rows, k)
    return rp * (kp // 32)


def require_finite(x: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"nonfinite MXFP8 {name}")


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Natural padded [rows,K/32] bytes to flat SWIZZLE_32_4_4 bytes."""
    r, c = scales.shape
    if scales.dtype != torch.uint8 or r % 128 or c % 4:
        raise ValueError("expected uint8 scales padded to 128 rows and 4 columns")
    return (scales.reshape(r // 128, 128, c // 4, 4).permute(0, 2, 1, 3)
            .reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous())


def quantize_cpu(x: torch.Tensor, *, chunk_rows: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one effective merged CPU matrix, bounded FP32 temporary rows.

    Allocates only the final padded byte payload/scales plus O(chunk_rows*K)
    temporaries. Does not keep a BF16 shadow or mutate the source. Meta creates
    identical buffer shapes without reading data. Padding represents exact zero.
    """
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("expected a floating 2D weight")
    if x.device.type not in ("cpu", "meta"):
        raise ValueError("weights must be converted on CPU before distribution")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    r, k = x.shape
    rp, kp = padded_shape(r, k)
    data = torch.zeros((rp, kp), dtype=torch.uint8, device=x.device)
    scales = torch.zeros((rp, kp // 32), dtype=torch.uint8, device=x.device)
    if x.is_meta:
        return data, scales.reshape(-1)
    for start in range(0, r, chunk_rows):
        rows = min(chunk_rows, r - start)
        values = torch.zeros((rows, kp), dtype=torch.float32)
        values[:, :k].copy_(x[start:start + rows].detach())
        require_finite(values, "weight")
        blocks = values.reshape(rows, -1, 32)
        amax = blocks.abs().amax(-1)
        bits = amax.view(torch.int32)
        e0 = ((bits >> 23) & 255) - 135
        threshold = (((e0 + 135) << 23) | 0x600000).view(torch.float32)
        exponent = (e0 + (amax > threshold).to(torch.int32)).clamp_min(-127)
        inv = ((127 - exponent) << 23).view(torch.float32)
        data[start:start + rows].copy_((blocks * inv[..., None]).reshape(rows, kp)
                                     .to(torch.float8_e4m3fn).view(torch.uint8))
        scales[start:start + rows].copy_((exponent + 127).to(torch.uint8))
    return data, swizzle_scales(scales)


def quantize_activation(x: torch.Tensor, *, check_finite: bool = True):
    """Fused pad + block32 quantization + swizzle, with no eager FP32 matrix.

    Returns padded contiguous uint8 data and flat uint8 scales. Finite checking
    is fail-closed by default; disabling requires a caller-owned global guard.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise ValueError("expected CUDA BF16 [rows,K] activation")
    if x.stride(-1) != 1:
        raise ValueError("activation K dimension must be contiguous")
    if check_finite:
        require_finite(x, "activation")
    from .mxfp8_kernels import launch_quantize
    return launch_quantize(x)
