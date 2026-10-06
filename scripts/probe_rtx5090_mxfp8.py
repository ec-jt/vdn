"""Standalone synthetic capability probe, not a production precision policy."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def quantize_block32(x):
    """E4M3 data with one decoding E8M0 scale per 32 contiguous K values.

    Returns row-major data and uint8 E8M0 storage, padding only K to 32.
    Ceil(log2(amax/448)) avoids saturation; zero blocks use exponent zero.
    This eager reference quantizer is intentionally not performance optimized.
    """
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("expected a floating-point matrix")
    rows, k = x.shape
    padded = torch.zeros((rows, triton.cdiv(k, 32) * 32), device=x.device)
    padded[:, :k] = x.float()
    blocks = padded.reshape(rows, -1, 32)
    amax = blocks.abs().amax(-1)
    exponent = torch.where(amax > 0, torch.ceil(torch.log2(amax / 448)), 0)
    exponent = exponent.clamp(-127, 127)
    scale = torch.exp2(exponent)
    q = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(rows, -1), (exponent + 127).to(torch.uint8)


def swizzle_scales(s):
    """cuBLAS 128-row by 4-scale tiles, matching installed torchao to_blocked."""
    r, c = s.shape
    rp, cp = triton.cdiv(r, 128) * 128, triton.cdiv(c, 4) * 4
    padded = torch.full((rp, cp), 127, dtype=torch.uint8, device=s.device)
    padded[:r, :c] = s
    blocks = padded.view(rp // 128, 128, cp // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def dequantize_block32(q, s):
    return (q.float().reshape(q.shape[0], -1, 32)
            * torch.exp2(s.float() - 127)[..., None]).reshape(q.shape)


def prepare_torch_mxfp8(x):
    """Pad rows/K to 128 and quantize. Preserve x; return E4M3/E8M0 pair.

    Weight preparation is one-time; activation preparation is per-call and must
    be included in the parent's operator timing. No performance claim is made.
    """
    rows, k = x.shape
    padded = torch.zeros((triton.cdiv(rows, 128) * 128, triton.cdiv(k, 128) * 128),
                         device=x.device, dtype=x.dtype)
    padded[:rows, :k] = x
    q, scales = quantize_block32(padded)
    return q, swizzle_scales(scales).view(torch.float8_e8m0fnu)


def torch_mxfp8_mm(a, w, sa, sw, *, output_rows=None, output_cols=None):
    """Native F.scaled_mm candidate over prepared A and row-major W.

    Return cropped BF16 A @ W.T. Optional contiguous output copy and bias are
    the caller's responsibility and must be counted in operator measurements.
    """
    y = F.scaled_mm(a, w.T, sa, F.ScalingType.BlockWise1x32,
                   sw, F.ScalingType.BlockWise1x32,
                   F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.SWIZZLE_32_4_4,
                   output_dtype=torch.bfloat16)
    return y[:output_rows, :output_cols]


@triton.jit
def _dot_scaled_kernel(A, W, SA, SW, C, M: tl.constexpr, N: tl.constexpr,
                       K: tl.constexpr, BM: tl.constexpr = 32,
                       BN: tl.constexpr = 64, BK: tl.constexpr = 128):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    si = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + ki
        ss = start * (BK // 32) + si
        a = tl.load(A + mi[:, None] * K + kk[None, :],
                    (mi[:, None] < M) & (kk[None, :] < K), 0)
        b = tl.load(W + ni[None, :] * K + kk[:, None],
                    (ni[None, :] < N) & (kk[:, None] < K), 0)
        sa = tl.load(SA + mi[:, None] * (K // 32) + ss[None, :],
                     (mi[:, None] < M) & (ss[None, :] < K // 32), 127)
        sb = tl.load(SW + ni[:, None] * (K // 32) + ss[None, :],
                     (ni[:, None] < N) & (ss[None, :] < K // 32), 127)
        acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    tl.store(C + mi[:, None] * N + ni[None, :], acc,
             (mi[:, None] < M) & (ni[None, :] < N))


def mxfp8_mm(a, w, sa, sw, *, return_kernel=False):
    """Prequantized A[M,K] @ W[N,K].T, natural uint8 scales [rows,K/32].

    Supports ragged M/N and K multiples of 32. Returns BF16; no bias.
    Intended only as a candidate for a separate real-input accuracy/speed gate.
    """
    if not (a.is_cuda and w.is_cuda and sa.is_cuda and sw.is_cuda):
        raise ValueError("CUDA tensors required")
    m, k = a.shape
    n, kw = w.shape
    if kw != k or k % 32 or sa.shape != (m, k // 32) or sw.shape != (n, k // 32):
        raise ValueError("incompatible data/scale shapes")
    if a.dtype != torch.float8_e4m3fn or w.dtype != a.dtype:
        raise ValueError("E4M3 data required")
    if sa.dtype != torch.uint8 or sw.dtype != torch.uint8:
        raise ValueError("uint8 E8M0 storage required")
    if any(not t.is_contiguous() for t in (a, w, sa, sw)):
        raise ValueError("row-major contiguous tensors required")
    if len({t.device for t in (a, w, sa, sw)}) != 1:
        raise ValueError("all tensors must share a device")
    c = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    kernel = _dot_scaled_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
        a.view(torch.uint8), w.view(torch.uint8), sa, sw, c, m, n, k, num_warps=4)
    return (c, kernel) if return_kernel else c


def tensor_info(t):
    raw = t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(exist_ok=False, parents=True)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    results = dict(versions=dict(torch=torch.__version__, cuda=torch.version.cuda,
                   triton=triton.__version__, torchao=importlib.metadata.version("torchao")),
                   gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                   visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), attempts={})
    x = torch.randn(128, 256, device="cuda")
    w = torch.randn(128, 256, device="cuda")
    # Deliberately nonuniform scales along both rows and contraction blocks.
    x *= torch.exp2((torch.arange(256, device="cuda") // 32 % 5 - 2).float())
    w *= torch.exp2((torch.arange(128, device="cuda")[:, None] % 5 - 2).float())
    a, sa = quantize_block32(x)
    b, sb = quantize_block32(w)
    sab = swizzle_scales(sa).view(torch.float8_e8m0fnu)
    sbb = swizzle_scales(sb).view(torch.float8_e8m0fnu)
    ref = dequantize_block32(a, sa) @ dequantize_block32(b, sb).T
    results["inputs"] = {n: tensor_info(t) for n, t in
                         dict(a=a,b_transposed=b.T,sa=sa,sb=sb,sa_swizzled=sab,sb_swizzled=sbb).items()}
    results["scale_bytes"] = {"a": sa.unique().tolist(), "b": sb.unique().tolist()}
    recipe = F.ScalingType.BlockWise1x32
    swizzle = F.SwizzleType.SWIZZLE_32_4_4
    calls = {
        "torch_legacy_swizzled": lambda: torch._scaled_mm(a, b.T, sab, sbb, out_dtype=torch.bfloat16),
        "torch_functional_swizzled": lambda: F.scaled_mm(a, b.T, sab, recipe, sbb, recipe,
                                                       swizzle, swizzle),
        "torch_v2_swizzled": lambda: torch._scaled_mm_v2(a, b.T, [sab], [recipe.value],
                              [swizzle.value], [sbb], [recipe.value], [swizzle.value], None,
                              torch.bfloat16, [], False),
        "torch_functional_unswizzled": lambda: F.scaled_mm(a, b.T,
            sa.view(torch.float8_e8m0fnu), recipe, sb.view(torch.float8_e8m0fnu), recipe,
            F.SwizzleType.NO_SWIZZLE, F.SwizzleType.NO_SWIZZLE),
        "triton_dot_scaled": lambda: mxfp8_mm(a, b, sa, sb),
    }
    for name, call in calls.items():
        record = results["attempts"][name] = {}
        try:
            y = call()
            torch.cuda.synchronize()
            record.update(status="success", finite=bool(y.isfinite().all()),
                          max_abs_vs_dequantized=float((y.float()-ref).abs().max()),
                          relative_l2_vs_dequantized=float((y.float()-ref).norm()/ref.norm()),
                          output_dtype=str(y.dtype))
            samples=[]
            for _ in range(5):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(); call(); end.record(); end.synchronize()
                samples.append(start.elapsed_time(end))
            record["unprofiled_ms_samples"]=samples
            record["unprofiled_ms_median"]=statistics.median(samples)
        except Exception:
            record.update(status="error", exception=traceback.format_exc())
            (out/f"{name}.exception.txt").write_text(record["exception"])
        # Profile each actual invocation, including dispatch failures.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function(name):
                try:
                    call(); torch.cuda.synchronize()
                except Exception:
                    pass
        prof.export_chrome_trace(str(out/f"{name}.trace.json"))
        print(name, json.dumps(record), flush=True)
    if results["attempts"]["triton_dot_scaled"]["status"] == "success":
        _, kernel = mxfp8_mm(a, b, sa, sb, return_kernel=True)
        for kind in ("ptx", "ttir", "ttgir", "llir"):
            (out/f"dot_scaled.{kind}").write_text(kernel.asm[kind])
        (out/"dot_scaled.cubin").write_bytes(kernel.asm["cubin"])
        results["triton_metadata"] = str(kernel.metadata)
        results["native_block_scale_ptx"] = "mma.sync.aligned" in kernel.asm["ptx"] and "block_scale" in kernel.asm["ptx"]
    results["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    results["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    torch.cuda.synchronize()
    (out/"results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
