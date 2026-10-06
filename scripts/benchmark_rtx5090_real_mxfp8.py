"""Opt-in real-operand capture and isolated MXFP8 benchmark, never integration.

Capture installs hooks only in this process and stops after block zero on all
ranks. Benchmark uses exactly the previously selected functional scaled_mm path.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from scripts.probe_rtx5090_mxfp8 import prepare_torch_mxfp8, torch_mxfp8_mm, tensor_info


# Dimensions checked against the checkpoint headers, not inferred from ratios.
TARGETS = {
    "q": ("attn.orig.to_q", (7168, 5376)),
    "k": ("attn.orig.to_k", (7168, 5376)),
    "v": ("attn.orig.to_v", (7168, 5376)),
    "softmax_out": ("attn.orig.to_out.0", (5376, 7168)),
    "linear_out": ("attn.to_out_linear", (5376, 7168)),
    "ff_up": ("ff.net.0.proj", (28672, 5376)),
    "ff_down": ("ff.net.2", (5376, 14336)),
}
FORMAT = "mxfp8-e4m3fn-e8m0-block32-swizzle32_4_4-v1"


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_master(tensor):
    """Independent floating CPU storage, never a shard/parameter alias."""
    if tensor.is_meta or not tensor.is_floating_point():
        raise ValueError("materialized floating tensor required")
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def pack_prepared(q, scale):
    """Byte envelope avoids CPU E8M0 arithmetic/pickle implementation assumptions."""
    if q.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu:
        raise ValueError("expected E4M3/E8M0")
    return {"format": FORMAT, "q": q.detach().view(torch.uint8).cpu().clone(),
            "scale": scale.detach().view(torch.uint8).cpu().clone(),
            "q_dtype": str(q.dtype), "scale_dtype": str(scale.dtype)}


def unpack_prepared(record):
    if record["format"] != FORMAT or record["q"].dtype != torch.uint8 or record["scale"].dtype != torch.uint8:
        raise ValueError("invalid MXFP8 serialization format")
    if record["q_dtype"] != "torch.float8_e4m3fn" or record["scale_dtype"] != "torch.float8_e8m0fnu":
        raise ValueError("invalid MXFP8 dtype metadata")
    return record["q"].view(torch.float8_e4m3fn), record["scale"].view(torch.float8_e8m0fnu)


def error_metrics(reference, candidate):
    """Fixed protocol gates. Zero reference accepts only exactly zero error."""
    if reference.shape != candidate.shape:
        raise ValueError("output shape mismatch")
    finite = bool(reference.isfinite().all() and candidate.isfinite().all())
    if not finite:
        return {"finite": False, "pass": False, "output_dtype": str(candidate.dtype)}
    r, c = reference.double(), candidate.double()
    delta = c - r
    rms = float(r.square().mean().sqrt())
    rmse = float(delta.square().mean().sqrt())
    maximum = float(delta.abs().max())
    relative = rmse / rms if rms else (0.0 if maximum == 0 else None)
    normalized_max = maximum / rms if rms else (0.0 if maximum == 0 else None)
    passed = (maximum == 0) if not rms else relative <= .05 and normalized_max <= .5
    return dict(finite=True, reference_rms=rms, rmse=rmse, max_abs_error=maximum,
                relative_l2=relative, max_error_over_reference_rms=normalized_max,
                zero_norm_absolute_check="max_abs_error == 0", output_dtype=str(candidate.dtype),
                reference_abs_max=float(r.abs().max()), candidate_abs_max=float(c.abs().max()),
                **{"pass": bool(passed and candidate.dtype == torch.bfloat16)})


def speed_gate(cases):
    baseline = sum(c["timing"]["bf16"]["median_ms"] for c in cases)
    candidate = sum(c["timing"]["candidate"]["median_ms"] for c in cases)
    individuals = all(c["timing"]["candidate"]["median_ms"] <= 1.05 * c["timing"]["bf16"]["median_ms"] for c in cases)
    return {"bf16_sum_median_ms": baseline, "candidate_sum_median_ms": candidate,
            "reduction_fraction": 1 - candidate / baseline,
            "aggregate_pass": candidate <= .95 * baseline, "individual_pass": individuals,
            "pass": candidate <= .95 * baseline and individuals}


class CaptureComplete(Exception):
    """Coordinated normal stop after the first block, caught after shard cleanup."""


def capture(out):
    import torch.distributed as dist
    import src.inference.infer_ulysses as entry

    original_install, original_run = entry.install_ulysses, entry.run
    saved, metadata, handles = {}, {}, []
    capture_dir = out / "capture"
    rank = int(os.environ["RANK"])
    if rank == 0:
        capture_dir.mkdir(exist_ok=False)

    def install(transformer, runtime, **kwargs):
        result = original_install(transformer, runtime, **kwargs)
        block = transformer.transformer_blocks[0]
        if runtime.is_main:
            for role, (suffix, expected) in TARGETS.items():
                module = block.get_submodule(suffix)

                def pre(module, args, role=role, suffix=suffix, expected=expected):
                    x = args[0]
                    if role in saved:
                        raise RuntimeError("duplicate capture")
                    if module.weight.is_meta or tuple(module.weight.shape) != expected:
                        raise RuntimeError("placeholder or unexpected effective weight")
                    if x.dtype != torch.bfloat16 or module.weight.dtype != torch.bfloat16:
                        raise RuntimeError("capture is not the BF16 control")
                    if x.shape[-1] != expected[1]:
                        raise RuntimeError("capture input width mismatch")
                    saved[role] = {"input": cpu_master(x), "weight": cpu_master(module.weight),
                                   "bias": None if module.bias is None else cpu_master(module.bias)}
                    metadata[role] = dict(module="transformer_blocks.0." + suffix,
                        input_original_shape=list(x.shape), input_original_stride=list(x.stride()),
                        rank=runtime.rank, local_start=runtime.local_start, local_end=runtime.local_end,
                        call_index=0, block=0, denoise_evaluation=0,
                        input_scope="local video rows only" if role == "linear_out" else "all local packed rows",
                        effective_weight="post-checkpoint remapping, branch load and LoRA fold; full gathered BF16 weight")

                def post(module, args, output, role=role):
                    saved[role]["actual_output"] = cpu_master(output)

                handles.extend([module.register_forward_pre_hook(pre), module.register_forward_hook(post)])

            def first_call(module, args, kwargs):
                timestep = kwargs.get("timestep", args[3] if len(args) > 3 else None)
                if timestep is not None:
                    metadata["timestep"] = timestep.detach().cpu().tolist()

            handles.append(transformer.register_forward_pre_hook(first_call, with_kwargs=True))

        def stop(module, args, output):
            raise CaptureComplete("block zero complete on this rank")

        handles.append(block.register_forward_hook(stop))
        return result

    def run(cfg, runtime, started, **kwargs):
        try:
            original_run(cfg, runtime, started, **kwargs)
        except CaptureComplete:
            # Original run has released shards and drained prefetch before here.
            torch.cuda.synchronize()
            status = None
            if runtime.is_main:
                try:
                    if set(saved) != set(TARGETS) or any("actual_output" not in v for v in saved.values()):
                        raise RuntimeError("incomplete capture")
                    for role, tensors in saved.items():
                        path = capture_dir / f"{role}.pt"
                        with path.open("xb") as stream:
                            torch.save(tensors, stream)
                        metadata[role]["tensors"] = {k: None if v is None else tensor_info(v) for k, v in tensors.items()}
                        metadata[role]["file_sha256"] = file_hash(path)
                    write_json(capture_dir / "manifest.json", metadata)
                except Exception as error:
                    status = repr(error)
            statuses = [status]
            dist.broadcast_object_list(statuses, src=0)
            write_json(capture_dir / f"rank-{runtime.rank}-stop.json", dict(
                rank=runtime.rank, status=statuses[0] or "captured_and_cleanly_stopped",
                weight_shards_released=runtime.weight_shards.released,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.perf_counter()-started))
            if statuses[0]:
                raise RuntimeError(statuses[0])
            print(f"rank {runtime.rank}: graceful block-zero capture stop", flush=True)
        else:
            raise RuntimeError("capture failed to stop inference")

    entry.install_ulysses, entry.run = install, run
    try:
        entry.main()
    finally:
        for handle in handles:
            handle.remove()
        entry.install_ulysses, entry.run = original_install, original_run


def timed(call):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    value = call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end), value


def summarize(samples):
    return dict(samples_ms=samples, median_ms=statistics.median(samples), min_ms=min(samples),
                max_ms=max(samples), mean_ms=statistics.mean(samples), stdev_ms=statistics.stdev(samples))


@torch.no_grad()
def benchmark(out, warmup=5, repeats=20):
    bench = out / "benchmark"
    bench.mkdir(exist_ok=False)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    versions = {name: importlib.metadata.version(name) for name in ("torch", "triton", "torchao", "diffusers", "safetensors")}
    write_json(bench / "environment.json", dict(versions=versions, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        python=sys.version, warmup=warmup, repeats=repeats,
        method="CUDA events; alternating BF16/candidate order; synchronized each call; no graphs; unprofiled",
        weight_policy="one-time prepared GPU weights; CPU floating master retained unchanged",
        per_call="activation pad/quantize/swizzle + functional scaled_mm + crop/contiguous + bias"))
    manifest = json.loads((out / "capture/manifest.json").read_text())
    records = []
    for role in TARGETS:
        master = torch.load(out / "capture" / f"{role}.pt", map_location="cpu", weights_only=True)
        before = {k: None if v is None else tensor_info(v) for k, v in master.items()}
        if before != manifest[role]["tensors"]:
            raise RuntimeError("capture tensor hash mismatch")
        x = master["input"].reshape(-1, master["input"].shape[-1]).cuda()
        w = master["weight"].cuda()
        bias = None if master["bias"] is None else master["bias"].cuda()
        m, n = x.shape[0], w.shape[0]
        weight_ms, (qw, sw) = timed(lambda: prepare_torch_mxfp8(w))
        qa, sa = prepare_torch_mxfp8(x)
        prepared = pack_prepared(qw, sw)
        with (bench / f"{role}-prepared-weight.pt").open("xb") as stream:
            torch.save(prepared, stream)
        del prepared

        def bf16():
            return F.linear(x, w, bias)

        def prequantized():
            return torch_mxfp8_mm(qa, qw, sa, sw, output_rows=m, output_cols=n)

        def candidate():
            a, scale = prepare_torch_mxfp8(x)
            y = torch_mxfp8_mm(a, qw, scale, sw, output_rows=m, output_cols=n).contiguous()
            return y if bias is None else y + bias

        reference, result = bf16(), candidate()
        actual = master["actual_output"].reshape(m, n).cuda()
        replay = error_metrics(actual, reference)
        numerical = error_metrics(reference, result)
        del actual, reference, result
        calls = {"bf16": bf16, "candidate": candidate,
                 "activation_prep": lambda: prepare_torch_mxfp8(x), "prequantized_gemm": prequantized}
        for _ in range(warmup):
            for call in calls.values():
                call()
        torch.cuda.synchronize()
        samples = {key: [] for key in calls}
        order = []
        for i in range(repeats):
            keys = ["bf16", "candidate"] if i % 2 == 0 else ["candidate", "bf16"]
            order.append(keys.copy())
            for key in keys + ["activation_prep", "prequantized_gemm"]:
                elapsed, value = timed(calls[key])
                samples[key].append(elapsed)
                del value
        memory = {}
        for key in ("bf16", "candidate"):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated()
            base_reserved = torch.cuda.memory_reserved()
            value = calls[key]()
            torch.cuda.synchronize()
            memory[key] = dict(base_allocated_bytes=base_alloc, base_reserved_bytes=base_reserved,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-base_alloc)
            del value
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                with torch.profiler.record_function(f"{role}/{key}"):
                    value = calls[key]()
                    torch.cuda.synchronize()
                del value
            prof.export_chrome_trace(str(bench / f"{role}-{key}.trace.json"))
        after = {k: None if v is None else tensor_info(v) for k, v in master.items()}
        if before != after:
            raise RuntimeError("CPU master mutated")
        record = dict(role=role, shape_mnk=[m, n, x.shape[1]], input=manifest[role],
            numerical=numerical, bf16_replay_vs_actual=replay, cpu_master_unchanged=before == after,
            prepared_weight={"q": tensor_info(qw), "scale": tensor_info(sw)},
            prepared_activation={"q": tensor_info(qa), "scale": tensor_info(sa)},
            weight_prep_ms=weight_ms, timing={k: summarize(v) for k,v in samples.items()},
            interleaved_order=order, memory=memory)
        write_json(bench / f"{role}.json", record)
        records.append(record)
        print(role, json.dumps({"shape": record["shape_mnk"], "numerical": numerical,
              "bf16_ms": record["timing"]["bf16"]["median_ms"],
              "candidate_ms": record["timing"]["candidate"]["median_ms"]}), flush=True)
        del master, x, w, bias, qa, sa, qw, sw
        gc.collect()
        torch.cuda.empty_cache()
    speed = speed_gate(records)
    numeric_pass = all(r["numerical"]["pass"] for r in records)
    write_json(bench / "summary.json", dict(cases=records, speed=speed, numerical_pass=numeric_pass,
        decision="ADVANCE_OPERATOR_ONLY" if speed["pass"] and numeric_pass else "STOP",
        integration_performed=False, full_run_performed=False))
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "benchmark"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args, remaining = parser.parse_known_args()
    if args.mode == "capture":
        sys.argv = [sys.argv[0]] + remaining
        capture(args.output)
    else:
        if remaining or args.warmup < 1 or args.repeats < 2:
            parser.error("invalid benchmark arguments")
        benchmark(args.output, args.warmup, args.repeats)


if __name__ == "__main__":
    main()
