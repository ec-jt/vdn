"""Opt-in replay of seven immutable captured effective real operator matrices.

No capture/full-model execution. Report historical gates without asserting them.
Invocation requires VDN_TEST_MXFP8_REAL=1 and a new MXFP8_REPORT_DIR.
"""
import gc
import json
import os
from pathlib import Path
import statistics
import time

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.benchmark_rtx5090_real_mxfp8 import TARGETS, error_metrics, tensor_info
from src.models.ops.mxfp8_linear import MXFP8Linear
from src.models.ops.mxfp8_quant import quantize_activation, quantize_cpu


CAPTURE = Path("artifacts/rtx5090/paper-fp8-20261005/real-operators-20261005T1811Z/capture")


def timed(call):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    value = call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end), value


@pytest.mark.skipif(os.getenv("VDN_TEST_MXFP8_REAL") != "1", reason="explicit real GPU replay opt-in")
@pytest.mark.parametrize("role", list(TARGETS))
@torch.no_grad()
def test_captured_real_fused_quant_native_gemm(role):
    report = Path(os.environ["MXFP8_REPORT_DIR"])
    report.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    master = torch.load(CAPTURE / f"{role}.pt", map_location="cpu", weights_only=True)
    manifest = json.loads((CAPTURE / "manifest.json").read_text())
    assert {k: None if t is None else tensor_info(t) for k, t in master.items()} == manifest[role]["tensors"]
    weight, xcpu = master["weight"], master["input"]
    xcpu = xcpu.reshape(-1, xcpu.shape[-1])
    n, k = weight.shape
    with torch.device("meta"):
        linear = nn.Linear(k, n, bias=master["bias"] is not None, dtype=weight.dtype)
    linear.weight = nn.Parameter(weight, requires_grad=False)
    if master["bias"] is not None:
        linear.bias = nn.Parameter(master["bias"], requires_grad=False)
    before = time.perf_counter()
    packed = MXFP8Linear(linear, chunk_rows=128)
    cpu_prep_seconds = time.perf_counter() - before
    compressed = tensor_info(packed.weight_data), tensor_info(packed.weight_scale)
    compressed_bytes = packed.weight_data.numel() + packed.weight_scale.numel()
    packed.cuda()
    x, w = xcpu.cuda(), weight.cuda()
    reference = F.linear(x, w, None if master["bias"] is None else master["bias"].cuda())
    assert torch.equal(reference.cpu(), master["actual_output"].reshape_as(reference).cpu())
    # Exact CPU vs fused GPU quantization on both captured activation and weight.
    aq, ass = quantize_activation(x)
    cq, cs = quantize_cpu(xcpu)
    assert torch.equal(aq.cpu(), cq) and torch.equal(ass.cpu(), cs)
    wq, ws = quantize_activation(w)
    assert torch.equal(wq, packed.weight_data) and torch.equal(ws, packed.weight_scale)
    del cq, cs, wq, ws
    y = packed(x)
    metrics = error_metrics(reference, y)
    assert metrics["finite"] and y.shape == reference.shape
    calls = {"bf16": lambda: F.linear(x, w), "checked_full": lambda: packed(x)}
    samples = {name: [] for name in ("bf16", "checked_full", "global_guard_full", "fused_quant", "gemm")}
    for _ in range(5):
        for call in calls.values():
            timed(call)
    for i in range(20):
        for name in (list(calls) if i % 2 == 0 else list(calls)[::-1]):
            ms, _ = timed(calls[name]); samples[name].append(ms)
    # Explicit parent-owned guard mode, not silently selected by production.
    packed.check_finite = False
    faster_calls = {"global_guard_full": lambda: packed(x),
                    "fused_quant": lambda: quantize_activation(x, check_finite=False),
                    "gemm": lambda: packed.forward_quantized(aq, ass, rows=x.shape[0])}
    for call in faster_calls.values():
        for _ in range(5):
            timed(call)
    for _ in range(20):
        for name, call in faster_calls.items():
            ms, _ = timed(call); samples[name].append(ms)
    # Actual candidate trace to establish native dispatch and fused preparation.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        packed(x)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(report / f"{role}.trace.json"))
    record = dict(role=role, shape_mnk=[x.shape[0], n, k], metrics=metrics,
                  cpu_weight_prep_seconds=cpu_prep_seconds, compressed_weight_bytes=compressed_bytes,
                  bf16_weight_bytes=weight.numel() * weight.element_size(), compressed_hashes=compressed,
                  timings={name: dict(median_ms=statistics.median(s), samples_ms=s) for name, s in samples.items()},
                  gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  historical_gates_preserved=True, quality_promoted=False,
                  scope="rank0/block0/evaluation0 immutable real capture; no full-model execution")
    with (report / f"{role}.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    assert {k: None if t is None else tensor_info(t) for k, t in master.items()} == manifest[role]["tensors"]
    print(json.dumps({"role": role, "metrics": metrics, "medians_ms": {k: statistics.median(v) for k, v in samples.items()}}), flush=True)
    del packed, x, w, reference, y, master, weight, linear, calls, faster_calls, aq, ass
    gc.collect()
    torch.cuda.empty_cache()
