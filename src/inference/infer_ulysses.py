"""Dedicated multi-GPU, inference-only Ulysses entrypoint.

Launch with ``torchrun --nproc_per_node=8 src/inference/infer_ulysses.py
--config configs/inference/{8,50}nfe_tuned_fp8_ulysses_{h200,b200}.yaml ...``. The ordinary ``src/inference/infer.py``
path is intentionally untouched. The rank layout, optimisation ladder, profiling and
warm-up are the ``parallel.*`` / ``render.warmup_steps`` config fields.
VDN_NVTX=1 enables host ranges; VDN_NSYS_CAPTURE=1 captures measured steps 2/3
and decode on all ranks, independently of the event profiler.

t2va, i2va, fl2va and the ref2va-like request all run here: the conditioning rows a
keyframe or reference cache adds sit outside the layout's video span, so they shard,
gather and attend as ordinary global rows and no collective changes shape.
"""

from __future__ import annotations

import os
import sys
import time
import resource

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist

from src.config import load_config
from src.config.inference import (
    InferenceConfig,
    validate_ablation,
    validate_kernels,
    validate_parallel,
)
from src.inference.utils.assemble import (
    build_inference_model,
    latents_path,
    render_record,
    write_json,
    release_sharded_denoiser,
)
from src.inference.render import decode_and_save, generate_latents
from src.inference.utils.prompt_cache import conditioning_mode, load_prompt
from src.inference.utils.nsight import NsightCapture, nvtx_range
from src.inference.utils.ulysses import init_ulysses, install_ulysses


def main():
    process_started = time.perf_counter()
    cfg = load_config(
        InferenceConfig,
        extra_validators=[validate_ablation, validate_kernels, validate_parallel],
    )
    if not cfg.checkpoint:
        raise ValueError("the dedicated Ulysses path requires a hybrid checkpoint")
    if cfg.behavior.teacher_mode:
        raise ValueError("teacher_mode is not supported by inference-only Ulysses")

    runtime = init_ulysses(profile_enabled=cfg.parallel.profile,
                           timeout_seconds=cfg.parallel.timeout_seconds)
    runtime.standard_exchange = cfg.parallel.standard_exchange
    try:
        run(cfg, runtime, process_started)
    finally:
        # No barrier in an error handler: a failed peer may no longer enter one.
        # NCCL timeout and the bounded torchrun launcher terminate failed peers.
        if dist.is_initialized():
            dist.destroy_process_group()


def _select_decoder(cfg, decoder=None):
    """Explicit callbacks win; only resident shards opt into distributed decode."""
    if decoder is not None:
        return decoder
    if cfg.memory.policy == "resident_weight_shards":
        from src.inference.utils.distributed_decode_callback import distributed_decode_callback
        return distributed_decode_callback
    return None


def run(cfg, runtime, process_started, *, decoder=None):
    """All-rank entrypoint. Optional decoder(cfg, model, runtime, video, audio)
    runs on ALL ranks, only after the sharded denoiser's collective release.
    Explicit integrations override the policy default. Resident weight shards use
    distributed video then rank-zero audio/write; legacy decoding is unchanged.
    A returned dictionary is recorded as decoder_metrics; None remains supported.
    """
    capture = NsightCapture(runtime.device)
    try:
        return _run(cfg, runtime, process_started, decoder=decoder, nsight_capture=capture)
    finally:
        try:
            capture.stop()
        finally:
            # Keep weight cleanup even if stopping a failed CUDA capture raises.
            # No recovery collective: failed peers may no longer participate.
            controller = getattr(runtime, "weight_shards", None)
            try:
                if controller is not None and not controller.released:
                    controller.release()
            finally:
                paper = getattr(runtime, "paper_runtime", None)
                if paper is not None:
                    paper.close()


def _run(cfg, runtime, process_started, *, decoder=None, nsight_capture=None):
    capture = nsight_capture if nsight_capture is not None else NsightCapture(runtime.device)
    device = runtime.device
    torch.set_grad_enabled(False)
    sharded = cfg.memory.policy == "resident_weight_shards"
    if runtime.is_main:
        os.makedirs(os.path.dirname(cfg.render.out) or ".", exist_ok=True)
    # Fail before expensive assembly, without overwriting any previous outputs.
    status = [None]
    if runtime.is_main:
        targets = [cfg.render.out, cfg.render.out + ".inference.json"]
        if cfg.render.save_latents:
            targets.append(latents_path(cfg))
        status[0] = next((p for p in targets if os.path.exists(p)), None)
    dist.broadcast_object_list(status, src=0)
    if status[0]:
        raise FileExistsError(f"refusing existing output: {status[0]}")

    model = build_inference_model(
        cfg, device, load_decoders=runtime.is_main and not sharded,
        log=runtime.is_main, runtime=runtime
    )
    runtime.weight_shards = model.weight_shards
    paper = getattr(runtime, "paper_runtime", None)
    if paper is not None:
        with paper.guard.local_phase("paper_install"):
            paper.install(model.transformer)
    if not model.is_hybrid:
        raise ValueError("Ulysses path currently requires a HybridAttention checkpoint")

    install_ulysses(model.transformer, runtime, softmax_ranks=cfg.parallel.softmax_ranks)
    from src.inference.utils.trace_blocks import install_block_trace
    install_block_trace(model.transformer, rank=runtime.rank,
                        trace_rows=int(os.environ.get("VDN_TRACE_ROWS", "4814")))
    if runtime.is_main and runtime.branch_parallel:
        print(
            f"branch-parallel Ulysses: {runtime.softmax_ranks} softmax ranks + "
            f"{runtime.world_size - runtime.softmax_ranks} linear ranks; "
            f"QKV projected once on sequence owners",
            flush=True,
        )
    elif runtime.is_main:
        print(
            f"standard Ulysses: {runtime.world_size} ranks, every rank owns heads of "
            "both branches",
            flush=True,
        )

    prompt_embeds, text_token_tags, conditions = load_prompt(cfg.render.prompt_file, str(device))
    if conditions and runtime.is_main:
        print(f"{conditioning_mode(conditions[0])}: keyframes anchored {conditions[0]}", flush=True)
    runtime.barrier()
    torch.cuda.synchronize(device)
    model_setup_seconds = time.perf_counter() - process_started

    def sample(num_steps, step_seconds=None, *, measured=False):
        return generate_latents(
            model.transformer,
            prompt_embeds,
            text_token_tags,
            cfg.render.num_frames,
            num_steps,
            cfg.render.seed,
            device,
            video_shift=cfg.render.video_shift,
            audio_shift=cfg.render.audio_shift,
            runtime=runtime,
            step_seconds=step_seconds,
            conditions=conditions,
            nsight_capture=capture if measured else None,
        )

    warmup_steps = cfg.render.warmup_steps
    if warmup_steps:
        if runtime.is_main:
            print(f"warming up {warmup_steps} NFE in the measured process", flush=True)
        sample(warmup_steps)

    runtime.reset_profile()
    startup_peak = dict(allocated_bytes=torch.cuda.max_memory_allocated(device),
                        reserved_bytes=torch.cuda.max_memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)

    step_seconds: list[float] = []
    preparation_seconds = paper.preparation_charge() if paper is not None else 0.0
    denoise_started = time.perf_counter()
    latents, audio_latents = sample(cfg.render.num_steps, step_seconds, measured=True)
    denoise_loop_seconds = time.perf_counter() - denoise_started
    # Startup hoisting must not make complete schedule/projection/transport free.
    denoise_seconds = denoise_loop_seconds + preparation_seconds
    finite = torch.tensor(int(torch.isfinite(latents).all() and torch.isfinite(audio_latents).all()),
                          device=device, dtype=torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError("nonfinite final latents on one or more ranks")

    timings = {
        "denoise_seconds": denoise_seconds,
        "seconds_per_step": denoise_seconds / cfg.render.num_steps,
        "step_seconds": step_seconds,
        "model_setup_seconds": model_setup_seconds,
        "denoise_loop_seconds": denoise_loop_seconds,
        "paper_preparation_added_seconds": preparation_seconds,
        "step1_including_compiler_warm_seconds": step_seconds[0] if step_seconds else None,
    }
    local_profile = {
        name: milliseconds / cfg.render.num_steps
        for name, milliseconds in runtime.profile_milliseconds().items()
    }
    profile_by_rank = [None] * runtime.world_size
    dist.all_gather_object(profile_by_rank, local_profile)
    timings["parallel_profile_ms_per_nfe_by_rank"] = profile_by_rank
    local_metrics = dict(rank=runtime.rank, denoise_seconds=denoise_seconds,
                         step_seconds=step_seconds, startup=model.startup_metrics,
                         startup_cuda_peak=startup_peak,
                         cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                         cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                         cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          weights=model.weight_shards.summary() if sharded else None,
                          paper_stack=paper.inventory() if paper is not None else None,
                         finite=True)
    rank_metrics = [None] * runtime.world_size
    dist.all_gather_object(rank_metrics, local_metrics)
    timings["ranks"] = rank_metrics

    if runtime.is_main and cfg.render.save_latents:
        lpath = latents_path(cfg)
        os.makedirs(os.path.dirname(lpath) or ".", exist_ok=True)
        torch.save({"video": latents.cpu(), "audio": audio_latents.cpu()}, lpath)

    decode_started = time.perf_counter()
    if sharded:
        release_sharded_denoiser(model, runtime)
    if not cfg.render.denoise_only:
        with capture.decode():
            decoder = _select_decoder(cfg, decoder)
            if decoder is not None:
                if not sharded:
                    raise ValueError("decoder integration requires resident_weight_shards")
                decoder_metrics = decoder(cfg, model, runtime, latents, audio_latents)
                if isinstance(decoder_metrics, dict):
                    timings["decoder_metrics"] = decoder_metrics
            else:
                status = [None]
                with nvtx_range("decode.rank0" if runtime.is_main else "decode.idle_worker"):
                    if runtime.is_main:
                        try:
                            decode_and_save(
                                latents, audio_latents, model.vae, model.audio_vae, cfg.render.out, str(device),
                                sequential=sharded, offload=model.weight_shards,
                            )
                            torch.cuda.synchronize(device)
                        except Exception as exc:
                            status[0] = f"{type(exc).__name__}: {exc}"
                    dist.broadcast_object_list(status, src=0)
                if status[0]:
                    raise RuntimeError(f"rank0 decode failed: {status[0]}")
            # Existing completion barrier stays inside every rank's capture.
            with nvtx_range("decode.collective_completion"):
                runtime.barrier()
    else:
        runtime.barrier()
    timings["decode_and_encode_seconds"] = time.perf_counter() - decode_started
    timings["end_to_end_seconds"] = time.perf_counter() - process_started

    if runtime.is_main:
        record = render_record(cfg, model)
        record["parallel"] = {
            "kind": "ulysses_branch_parallel" if runtime.branch_parallel else "ulysses",
            "standard_exchange": runtime.standard_exchange,
            "world_size": runtime.world_size,
            "backend": runtime.backend,
            "sequence_splits": list(runtime.splits),
            "heads_per_rank": runtime.heads_per_rank,
            "softmax_ranks": runtime.softmax_ranks,
            "softmax_head_splits": list(runtime.softmax_head_splits),
            "linear_head_splits": list(runtime.linear_head_splits),
            "warmup_steps": warmup_steps,
        }
        record["timings"] = timings
        record["denoise_only"] = cfg.render.denoise_only
        record["numerical_note"] = (
            "Untuned head-sharded eager linear path. Collective FP32 frame-sum reduction "
            "and shape-dependent GEMMs can differ from single-rank rounding; no exact parity claim."
        )

        if cfg.render.record:
            json_path = cfg.render.out + ".inference.json"
            write_json(record, json_path)
            print(f"wrote {json_path}", flush=True)

        print(
            "Ulysses timing: "
            f"setup {timings['model_setup_seconds']:.2f}s, "
            f"denoise {timings['denoise_seconds']:.2f}s "
            f"({timings['seconds_per_step']:.2f}s/step; per NFE "
            + " ".join(f"{x:.2f}" for x in timings["step_seconds"]) + "), "
            f"decode+encode {timings['decode_and_encode_seconds']:.2f}s, "
            f"E2E {timings['end_to_end_seconds']:.2f}s",
            flush=True,
        )

        if runtime.profile_enabled:
            for rank, profile in enumerate(profile_by_rank):
                fields = ", ".join(
                    f"{name}={milliseconds:.1f}ms"
                    for name, milliseconds in sorted(profile.items())
                )
                print(f"profile rank {rank}: {fields}", flush=True)

if __name__ == "__main__":
    main()
