"""All-rank, release-gated video decode followed by rank-zero audio and media IO."""

from __future__ import annotations

import gc
import os
import time

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLMiniMaxH3Audio

from src.inference.render import FPS, PIXEL_MEAN, PIXEL_STD, _decode_audio, encode_video
from src.inference.utils.cpu_loading import load_cpu_model, resolve_inference_weights
from src.inference.utils.distributed_decode import distributed_decode, load_video_decoder
from src.inference.utils.nsight import nvtx_annotate, nvtx_range
from src.paths import H3_BASE


def _check_status(runtime, stage, error):
    errors = [None] * runtime.world_size
    dist.all_gather_object(errors, error)
    if any(errors):
        details = "; ".join(f"rank {rank}: {value}" for rank, value in enumerate(errors) if value)
        raise RuntimeError(f"{stage} failed: {details}")


def _error(exc):
    # Do not retain tracebacks (and their GPU intermediates) across stage boundaries.
    return f"{type(exc).__name__}: {exc}"


@nvtx_annotate("decode.output.postprocess")
def _postprocess_video(raw, device):
    """Keep render._decode_video's FP32 arithmetic on the raw output's device."""
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (raw.float() * pixel_std + pixel_mean).clamp(0, 1)
    return (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu()


def _audio_and_write(cfg, source, runtime, audio_latents, frames):
    timings = {}
    audio_vae = None
    started = time.perf_counter()
    try:
        audio_vae = load_cpu_model(
            AutoencoderKLMiniMaxH3Audio, source, subfolder="audio_vae",
            torch_dtype=torch.float32, local_files_only=True,
        )
        audio_vae.eval().requires_grad_(False)
        timings["audio_load_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        audio_vae.to(runtime.device)
        audio = _decode_audio(audio_latents, audio_vae, runtime.device)
        sampling_rate = audio_vae.config.sampling_rate
    finally:
        try:
            if audio_vae is not None:
                audio_vae.to("cpu")
        finally:
            audio_vae = None
            gc.collect()
            torch.cuda.empty_cache()
    timings["audio_decode_seconds"] = time.perf_counter() - started

    out_path = os.fspath(cfg.render.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    stem, ext = os.path.splitext(out_path)
    partial_path = f"{stem}.partial{ext}"
    started = time.perf_counter()
    # Never remove or overwrite a partial artifact from an earlier invocation.
    if os.path.exists(out_path) or os.path.exists(partial_path):
        raise FileExistsError(f"refusing existing output or partial: {out_path}")
    try:
        with nvtx_range("decode.output.media_mux"):
            encode_video(frames, fps=FPS, output_path=partial_path,
                         audio=audio, audio_sample_rate=sampling_rate)
            os.replace(partial_path, out_path)
    finally:
        if os.path.exists(partial_path):
            os.unlink(partial_path)
    timings["media_encode_seconds"] = time.perf_counter() - started
    return timings


@torch.no_grad()
@nvtx_annotate("decode.distributed_callback")
def distributed_decode_callback(cfg, model, runtime, video, audio):
    """Decode/write once after collective denoiser release; call on EVERY rank.

    Implements ``infer_ulysses.run(..., decoder=callback)``. The runtime uses the
    default process group, with rank zero as the sole audio decoder and writer.
    Inputs are the sampler's normalized video/audio latents. No denoiser is loaded
    or released here: ``model.weight_shards.released`` must already be true on all
    ranks. Decoder sources resolve locally only, from cfg.vae_source or H3_BASE.

    Every rank loads an FP32 CPU decode-only video VAE. distributed_decode manages
    its onload/eviction and returns raw GPU BCTHW on rank zero. Pixel arithmetic
    matches render._decode_video before uint8 CPU transfer. All video references
    and allocator cache are freed before rank zero loads/onloads the audio VAE.
    Audio is evicted in finally before the one atomic sibling-partial media write.
    No decoders are retained on model, so subsequent calls can use new outputs.

    Returns a dictionary on every rank: decoder_load_seconds_by_rank (local-only
    resolution/loading), distributed_video (the decoder's full metrics), plus
    video_postprocess_seconds, audio_load_seconds, audio_decode_seconds (including
    onload/eviction), media_encode_seconds and total_seconds. Rank-zero stage
    timings are broadcast; distributed_video retains its rank-local rank field.
    Guard/load/postprocess/cleanup errors converge before later stages; audio/IO
    errors are broadcast. Distributed decode owns its internal collective errors.
    Transport/process failures still require the launcher's timeout/abort handling;
    no recovery barrier is attempted after a failed distributed decode.
    """
    started = time.perf_counter()
    error = None
    try:
        controller = getattr(model, "weight_shards", None)
        if controller is None or not controller.released:
            raise RuntimeError("decoder loading requires completely released denoiser")
        if runtime.is_main and os.path.exists(cfg.render.out):
            raise FileExistsError(f"refusing existing output: {cfg.render.out}")
        stem, ext = os.path.splitext(os.fspath(cfg.render.out))
        if runtime.is_main and os.path.exists(f"{stem}.partial{ext}"):
            raise FileExistsError(f"refusing existing partial output: {stem}.partial{ext}")
    except Exception as exc:
        error = _error(exc)
    _check_status(runtime, "decoder preflight", error)

    vae = raw = None
    load_started = time.perf_counter()
    error = None
    try:
        source = resolve_inference_weights(cfg.vae_source or H3_BASE, local_files_only=True)
        vae = load_video_decoder(source)
    except Exception as exc:
        error = _error(exc)
    load_seconds = time.perf_counter() - load_started
    try:
        _check_status(runtime, "video decoder loading", error)
        loads = [None] * runtime.world_size
        dist.all_gather_object(loads, dict(rank=runtime.rank, seconds=load_seconds))
        raw, video_metrics = distributed_decode(
            vae, video, device=runtime.device, output_cpu=False, reset_peak_memory=True,
        )
    except Exception:
        # The managed decoder evicts on errors. Do not add a collective here:
        # transport failures cannot safely enter another status exchange.
        vae = raw = None
        gc.collect()
        torch.cuda.empty_cache()
        raise

    error = None
    frames = None
    postprocess_seconds = 0.0
    try:
        if runtime.is_main:
            postprocess_started = time.perf_counter()
            frames = _postprocess_video(raw, runtime.device)
            postprocess_seconds = time.perf_counter() - postprocess_started
    except Exception as exc:
        error = _error(exc)
    finally:
        raw = vae = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            error = error or _error(exc)
    # Also ensures no rank has live video decoder/output storage when audio starts.
    _check_status(runtime, "video postprocess/cleanup", error)

    status = [None]
    with nvtx_range("decode.audio_output" if runtime.is_main else "decode.idle_worker.audio_output"):
        if runtime.is_main:
            try:
                timings = _audio_and_write(cfg, source, runtime, audio, frames)
                timings["video_postprocess_seconds"] = postprocess_seconds
                timings["total_seconds"] = time.perf_counter() - started
                status[0] = dict(timings=timings, error=None)
            except Exception as exc:
                status[0] = dict(error=_error(exc))
        dist.broadcast_object_list(status, src=0)
    if status[0]["error"]:
        raise RuntimeError(f"rank0 audio/write failed: {status[0]['error']}")
    return dict(decoder_load_seconds_by_rank=loads, distributed_video=video_metrics,
                **status[0]["timings"])
