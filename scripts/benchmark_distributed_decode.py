"""Offline decoder-only parity/timing benchmark. Launch with eight torchrun ranks.

No denoiser, audio, or media writer is loaded. Existing reports are never replaced.
--probe crops the saved latent to one seven-token, 512x512-pixel clip.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.distributed as dist

from src.inference.utils.distributed_decode import distributed_decode, load_video_decoder
from src.inference.render import PIXEL_MEAN, PIXEL_STD


def sync(device):
    torch.cuda.synchronize(device)


def pixels(raw, device):
    # Same eager CUDA arithmetic as render._decode_video, without re-decoding.
    mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (raw.float() * std + mean).clamp(0, 1)
    return (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu()


def compare(a, b, byte_a, byte_b):
    if a.shape != b.shape or a.dtype != b.dtype or byte_a.shape != byte_b.shape:
        raise ValueError("Reference/distributed output shape or dtype mismatch")
    maximum = total = squared = different = count = 0
    for x, y in zip(a.unbind(2), b.unbind(2)):
        delta = (x.float() - y.float()).abs()
        maximum = max(maximum, delta.max().item())
        total += delta.double().sum().item()
        squared += delta.double().square().sum().item()
        different += torch.count_nonzero(delta).item()
        count += delta.numel()
    bd = (byte_a.to(torch.int16) - byte_b.to(torch.int16)).abs()
    return dict(raw_shape=list(a.shape), raw_dtype=str(a.dtype),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                float_max_abs=maximum, float_mean_abs=total/count,
                float_rmse=(squared/count)**0.5, float_differing_values=different,
                float_values=count, byte_max_abs=bd.max().item(),
                byte_mean_abs=bd.double().mean().item(),
                byte_differing_values=torch.count_nonzero(bd).item(), byte_values=bd.numel(),
                reference_rgb_sha256=hashlib.sha256(byte_a.numpy().tobytes()).hexdigest(),
                distributed_rgb_sha256=hashlib.sha256(byte_b.numpy().tobytes()).hexdigest())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="../models/OpenVDN/vdn-minimax-h3/h3-base")
    parser.add_argument("--latents", default="artifacts/latents/8nfe_rtx5090_124_final.mp4.latents.pt")
    parser.add_argument("--report", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--callback-out", help="Also exercise video/audio/single-writer callback with saved latents")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if Path(args.report).exists():
        raise FileExistsError(args.report)
    if args.callback_out and Path(args.callback_out).exists():
        raise FileExistsError(args.callback_out)
    started = time.perf_counter()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=device)
    try:
        video = torch.load(args.latents, map_location="cpu", weights_only=True)["video"]
        if args.probe:
            video = video[:, :, :7, :32, :32].contiguous()
        video = video.to(device)
        report = dict(arguments=vars(args), world_size=dist.get_world_size(),
                      input_shape=list(video.shape), torch_version=torch.__version__,
                      gpu=torch.cuda.get_device_name(device),
                      scope="Video only. No denoiser/audio/writer. Checkpoint load separate. Cold then repeat; disk caches retained.")
        report["source_sha256"] = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in ("src/inference/utils/distributed_decode.py", __file__,
                         "diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py")
        }
        reference = reference_bytes = None
        status = [None]
        if rank == 0:
            try:
                from diffusers import AutoencoderKLMiniMaxH3
                t = time.perf_counter()
                vae = AutoencoderKLMiniMaxH3.from_pretrained(
                    args.base, subfolder="vae", torch_dtype=torch.float32,
                    local_files_only=True).eval().requires_grad_(False)
                vae.enable_tiling()
                report["reference_load_seconds"] = time.perf_counter() - t
                torch.cuda.reset_peak_memory_stats(device)
                sync(device)
                t = time.perf_counter()
                vae.to(device)
                sync(device)
                onload = time.perf_counter() - t
                td = time.perf_counter()
                mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
                std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
                with torch.autocast("cuda", dtype=torch.float16):
                    raw = vae.decode(video * std + mean, return_dict=False)[0]
                sync(device)
                decode_seconds = time.perf_counter() - td
                tp = time.perf_counter()
                reference_bytes = pixels(raw, device)
                sync(device)
                pixel_seconds = time.perf_counter() - tp
                tc = time.perf_counter()
                reference = raw.cpu()
                del raw
                sync(device)
                float_copy = time.perf_counter() - tc
                te = time.perf_counter()
                vae.to("cpu")
                torch.cuda.empty_cache()
                sync(device)
                report["reference"] = dict(onload_seconds=onload, decode_seconds=decode_seconds,
                    postprocess_output_cpu_seconds=pixel_seconds, float_copy_validation_seconds=float_copy,
                    evict_seconds=time.perf_counter()-te,
                    comparable_video_seconds=time.perf_counter()-t-float_copy,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                del vae
                gc.collect()
                torch.cuda.empty_cache()
                print("REFERENCE " + json.dumps(report["reference"]), flush=True)
            except Exception as exc:
                status[0] = repr(exc)
        dist.broadcast_object_list(status, src=0)
        if status[0]:
            raise RuntimeError(status[0])
        t = time.perf_counter()
        vae = load_video_decoder(args.base)
        load_seconds = time.perf_counter() - t
        loads = [None] * dist.get_world_size()
        dist.all_gather_object(loads, load_seconds)
        report["decoder_only_load_seconds_by_rank"] = loads
        report["decoder_parameters"] = sum(p.numel() for p in vae.parameters())
        report["runs"] = []
        for repeat in range(args.repeats):
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
            sync(device)
            t = time.perf_counter()
            raw, metrics = distributed_decode(vae, video, device=device, output_cpu=False,
                                               reset_peak_memory=True)
            if rank == 0:
                tp = time.perf_counter()
                rgb = pixels(raw, device)
                sync(device)
                metrics["postprocess_output_cpu_seconds"] = time.perf_counter() - tp
                torch.cuda.empty_cache()
                metrics["comparable_video_seconds"] = time.perf_counter() - t
                metrics["including_postprocess_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                metrics["including_postprocess_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                result = raw.cpu()
                del raw
                metrics["parity"] = compare(reference, result, reference_bytes, rgb)
                del result, rgb
                report["runs"].append(metrics)
                print("DISTRIBUTED " + json.dumps(metrics), flush=True)
            dist.barrier()
        del vae
        gc.collect()
        torch.cuda.empty_cache()
        if args.callback_out:
            from src.inference.utils.distributed_decode_callback import distributed_decode_callback
            audio = torch.load(args.latents, map_location="cpu", weights_only=True)["audio"].to(device)
            cfg = SimpleNamespace(vae_source=args.base, render=SimpleNamespace(out=args.callback_out))
            runtime = SimpleNamespace(rank=rank, world_size=dist.get_world_size(), is_main=rank == 0, device=device)
            # Decoder-only harness: there is no denoiser to release. This does not
            # claim to exercise actual denoiser assembly/release.
            model = SimpleNamespace(weight_shards=SimpleNamespace(released=True))
            callback_metrics = distributed_decode_callback(cfg, model, runtime, video, audio)
            report["callback"] = callback_metrics
            report["callback_note"] = "Saved latents; simulated already-released guard, no denoiser was loaded."
        if rank == 0:
            report["process_seconds"] = time.perf_counter() - started
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.report, "x") as f:
                json.dump(report, f, indent=2)
                f.write("\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
