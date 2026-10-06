"""Opt-in bounded batch CLI. One assembly and eight workers for all requests."""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist

from src.config import load_config
from src.config.inference import InferenceConfig, validate_ablation, validate_kernels, validate_parallel
from src.inference.render import generate_latents
from src.inference.utils.assemble import build_inference_model, latents_path
from src.inference.utils.nsight import NsightCapture, nvtx_range
from src.inference.utils.persistent import (
    PersistentPaperRuntime, ResidentDecoders, atomic_json, consensus, validate_request,
)
from src.inference.utils.prompt_cache import load_prompt
from src.inference.utils.ulysses import init_ulysses, install_ulysses


def run(cfg, runtime, count, started):
    runtime.paper_factory = PersistentPaperRuntime
    tensor_parallel = os.environ.get("VDN_PERMANENT_TP", "0")
    if tensor_parallel not in ("0", "1"):
        raise ValueError("VDN_PERMANENT_TP must be 0 or 1")
    if tensor_parallel == "1":
        from src.inference.utils.tensor_parallel import PermanentTensorShards
        runtime.weight_controller_factory = PermanentTensorShards
        runtime.permanent_tensor_parallel = True
    decoders = ResidentDecoders()
    capture = NsightCapture(runtime.device)
    model = None
    records = []
    root = Path(cfg.render.out).parent
    try:
        with nvtx_range("persistent.model_setup_once"):
            model = build_inference_model(cfg, runtime.device, load_decoders=False,
                                          log=runtime.is_main, runtime=runtime)
            paper = runtime.paper_runtime
            with paper.guard.local_phase("paper_install"):
                paper.install(model.transformer)
            install_ulysses(model.transformer, runtime, softmax_ranks=0)
            if tensor_parallel == "1":
                from src.inference.utils.tensor_parallel import install_tp_block
                for block in model.transformer.transformer_blocks:
                    install_tp_block(block, runtime, paper)
        runtime.barrier()
        setup_seconds = time.perf_counter() - started
        model_setup = dict(model.startup_metrics)
        setup_inventory = paper.inventory()
        manifest_path = os.environ.get("VDN_PERSISTENT_MANIFEST")
        error = None
        manifest = None
        try:
            manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else [{}] * count
            if not isinstance(manifest, list) or len(manifest) != count:
                raise ValueError("request manifest must match bounded request count")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        consensus(runtime, "batch manifest", error, json.dumps(manifest, sort_keys=True))
        for index, entry in enumerate(manifest):
            request_started = time.perf_counter()
            request_started_unix = time.time()
            torch.cuda.reset_peak_memory_stats(runtime.device)
            runtime.reset_profile()
            request_cfg = copy.deepcopy(cfg)
            error = None
            digest = None
            try:
                if not isinstance(entry, dict) or set(entry) - {"prompt_file", "seed"}:
                    raise ValueError("manifest permits only cached prompt_file and seed")
                for key, value in entry.items():
                    setattr(request_cfg.render, key, value)
                digest = validate_request(request_cfg, paper.signature)
                outdir = root / f"request-{index + 1:02d}"
                if runtime.is_main:
                    outdir.mkdir(mode=0o700)
                request_cfg.render.out = str(outdir / "render.mp4")
                request_cfg.render.latents_root = str(outdir / "latents")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            consensus(runtime, "request admission", error, (index, digest, request_cfg.render.seed,
                       request_cfg.render.video_shift, request_cfg.render.audio_shift))
            success = False
            video = audio = prompt = tags = conditions = None
            # Request two alone is profiled. A third short capture covers its
            # cleanup -> request-three preparation boundary, not another denoise.
            measured = index == 1
            try:
                with nvtx_range(f"persistent.request.{index + 1}.prepare"):
                    paper.prepare_request(request_cfg, model.transformer)
                    with paper.guard.local_phase("persistent_prompt_load"):
                        prompt, tags, conditions = load_prompt(request_cfg.render.prompt_file, str(runtime.device))
                    if index == 2:
                        capture.stop()
                    runtime.barrier()
                    torch.cuda.synchronize(runtime.device)
                prep_seconds = time.perf_counter() - request_started
                step_seconds = []
                t = time.perf_counter()
                with nvtx_range(f"persistent.request.{index + 1}.denoise"):
                    video, audio = generate_latents(model.transformer, prompt, tags,
                        request_cfg.render.num_frames, request_cfg.render.num_steps,
                        request_cfg.render.seed, runtime.device,
                        video_shift=request_cfg.render.video_shift, audio_shift=request_cfg.render.audio_shift,
                        runtime=runtime, step_seconds=step_seconds, conditions=conditions,
                        nsight_capture=capture if measured else None)
                loop_seconds = time.perf_counter() - t
                paper.guard.check("persistent_final_latents", video, audio)
                paper.drain_request(model.weight_shards)
                denoise_peak = dict(allocated=torch.cuda.max_memory_allocated(runtime.device),
                                    reserved=torch.cuda.max_memory_reserved(runtime.device))
                t = time.perf_counter()
                error = None
                if runtime.is_main:
                    try:
                        path = Path(latents_path(request_cfg))
                        path.parent.mkdir(mode=0o700)
                        partial = path.with_name(path.name + ".partial")
                        with partial.open("xb") as handle:
                            torch.save(dict(video=video.cpu(), audio=audio.cpu()), handle)
                        os.link(partial, path)
                        partial.unlink()
                    except Exception as exc:
                        error = str(exc)
                consensus(runtime, "latent write", error)
                latent_write_seconds = time.perf_counter() - t
                with capture.decode() if measured else nvtx_range(f"persistent.request.{index + 1}.decode"):
                    decoder_metrics = decoders.decode(request_cfg, runtime, video, audio)
                    runtime.barrier()
                success = True
            finally:
                paper.drain_request(model.weight_shards, failed=not success)
                video = audio = prompt = tags = conditions = None
            # Include output, cleanup and all-rank completion in actual latency.
            torch.cuda.synchronize(runtime.device)
            runtime.barrier()
            wall_seconds = time.perf_counter() - request_started
            local = dict(rank=runtime.rank, pid=os.getpid(), request=index + 1,
                started_unix=request_started_unix, finished_unix=time.time(),
                request_seconds=wall_seconds, preparation_seconds=prep_seconds,
                denoise_loop_seconds=loop_seconds, step_seconds=step_seconds,
                latent_write_seconds=latent_write_seconds, denoise_peak=denoise_peak,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(runtime.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(runtime.device),
                resident_allocated_bytes=torch.cuda.memory_allocated(runtime.device),
                weights=model.weight_shards.summary(), paper_stack=paper.inventory(),
                decoder=decoder_metrics, decoder_load_count=decoders.loads,
                process_group_identity=id(dist.group.WORLD), weight_group_identity=id(model.weight_shards.weight_group),
                source_adaln_bytes=paper.source_bytes, model_setup_count=1, distribution_count=1)
            ranks = [None] * runtime.world_size
            dist.all_gather_object(ranks, local, group=paper.group)
            record = dict(request=index + 1, prompt_sha256=digest, seed=request_cfg.render.seed,
                          model_signature=paper.signature, ranks=ranks,
                          media=request_cfg.render.out, latents=latents_path(request_cfg))
            records.append(record)
            error = None
            if runtime.is_main:
                try:
                    atomic_json(outdir / "metrics.json", record)
                    print(f"persistent request {index + 1}: prep={prep_seconds:.6f}s loop={loop_seconds:.6f}s wall={wall_seconds:.6f}s", flush=True)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            consensus(runtime, "request metrics write", error)
            if measured and count >= 3:
                capture.start()
                with nvtx_range("persistent.request.2.finished.same_workers_boundary"):
                    runtime.barrier()
        result = dict(model_setup_seconds=setup_seconds, setup=model_setup,
                      setup_inventory=setup_inventory, requests=records,
                      process_seconds=time.perf_counter() - started,
                      numerical_policy="deterministic low-precision drift accepted for performance; no quality equivalence",
                       decoder_policy=decoders.policy)
        result["parallel_policy"] = "permanent_tp8" if tensor_parallel == "1" else "gathered_ulysses8"
        error = None
        if runtime.is_main:
            try:
                atomic_json(root / "batch.json", result)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        consensus(runtime, "batch metrics write", error)
    finally:
        capture.stop()
        decoders.close()
        controller = getattr(runtime, "weight_shards", None)
        try:
            if controller is not None and not controller.released:
                controller.release()
        finally:
            paper = getattr(runtime, "paper_runtime", None)
            if paper is not None:
                paper.close()
        model = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    started = time.perf_counter()
    cfg = load_config(InferenceConfig, extra_validators=[validate_ablation, validate_kernels, validate_parallel])
    count = int(os.environ.get("VDN_PERSISTENT_REQUESTS", "3"))
    if not 1 <= count <= 8:
        raise ValueError("bounded persistent batch requires 1..8 requests")
    torch.set_grad_enabled(False)
    runtime = init_ulysses(profile_enabled=False, timeout_seconds=cfg.parallel.timeout_seconds)
    runtime.standard_exchange = cfg.parallel.standard_exchange
    try:
        run(cfg, runtime, count, started)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
