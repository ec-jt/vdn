"""The render shared by every inference entrypoint: generate_latents (the packed layout and
the denoising loop) and decode_and_save with its atomic .partial.mp4 write.
"""
import os
import time
from contextlib import nullcontext
from functools import wraps
from types import SimpleNamespace

import torch

from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep, MiniMaxH3Ref2VAPrepareLayoutStep, patchify_video_latents)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS as AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG as AUDIO_TAG,
    MINIMAX_H3_FPS as FPS, MINIMAX_H3_VIDEO_TAG as VIDEO_TAG, align_num_frames,
    audio_latent_num_frames, video_latent_num_frames)
from diffusers.utils.export_utils import encode_video

from src.inference.utils.prompt_cache import is_reference_request
from src.inference.utils.nsight import nvtx_annotate, nvtx_range
from src.models.sequence_layout import layout_from_indices
from src.models.hybrid_transform import iter_hybrids, set_layout

# The production canvas, 768x1344, through the 16x spatial VAE.
LATENT_H, LATENT_W = 48, 84
# Mirrors of MiniMaxH3ModularPipeline.pixel_mean / pixel_std / keyframe_noise_aug: those
# are properties of an assembled pipeline, which the inlined blocks below never build.
# 0.999 is the t the fl2va keyframes are held at, just short of clean.
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
KEYFRAME_NOISE_AUG = 0.999


def _paper_request_cleanup(fn):
    @wraps(fn)
    def wrapped(transformer, *args, **kwargs):
        paper = getattr(transformer, "_paper_runtime", None)
        try:
            return fn(transformer, *args, **kwargs)
        except BaseException:
            if paper is not None:
                paper.state = "failed"
            raise
        finally:
            if paper is not None:
                paper.close_tables()
    return wrapped


@torch.no_grad()
@_paper_request_cleanup
def generate_latents(transformer, prompt_embeds, text_token_tags, num_frames, num_steps, seed, device,
                     video_shift=12.0, audio_shift=3.0, runtime=None, step_seconds=None,
                     conditions=None, metrics=None, nsight_capture=None):
    """The sampler. `runtime` (Ulysses) adds a barrier before and after the loop so every
    rank enters and leaves together; `step_seconds`, if a list, receives the
    device-synchronised wall time of every NFE (for the timing log and the record).

    `conditions` = (keyframe_anchors, condition_latents) from load_prompt turns the
    request into i2va / fl2va: the layout reserves one frame of conditioning rows per
    keyframe between the text and the audio (`[text | keyframes | audio | video]`), the
    anchors are noised to t = 0.999 and pinned there, and only the generated rows are
    ever stepped -- the diffusers fl2va blocks, inlined. On a hybrid model the
    conditioning rows fall outside the layout's video span, i.e. they are attended like
    text and audio: dense in both directions by the window softmax and absent from the
    linear scan.

    Anchors that are all "ref" (encode_keyframes.py --refs) are image references instead:
    the diffusers ref2va layout, `[text | one block per reference | audio | video]`, each
    block on its own geometry and its own rotary slot ahead of the generated rows. The
    noise, the pinning and the hybrid treatment are the keyframes', and the transformer
    is whatever was loaded, the t2va/fl2va one, not MiniMax-H3's transformer_ref."""
    num_frames = align_num_frames(num_frames, 17, 5)
    num_latent_frames = video_latent_num_frames(num_frames, 17, 5)
    num_audio_latents = audio_latent_num_frames(num_frames)
    anchors, condition_latents = conditions if conditions else ((), [])
    patch = tuple(transformer.config.patch_size)                    # (1, 2, 2)
    channels = transformer.config.in_channels                       # 24
    frame_h, frame_w = LATENT_H // patch[1], LATENT_W // patch[2]

    if is_reference_request(anchors):
        # The layout step reads a reference's modality only (`kind`, `has_audio`); the
        # geometry comes from the latents.
        references = [SimpleNamespace(kind="image", has_audio=False) for _ in anchors]
        layout = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            text_token_tags, references, condition_latents, [], num_latent_frames, LATENT_H,
            LATENT_W, num_audio_latents, patch, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG,
        )
    else:
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags, num_latent_frames, LATENT_H, LATENT_W, num_audio_latents,
            patch, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG, keyframe_anchors=anchors,
        )
    position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, _ = layout
    position_ids, token_tags = position_ids.to(device), token_tags.to(device)
    video_indices, audio_indices, text_indices = (
        video_indices.to(device), audio_indices.to(device), text_indices.to(device),
    )

    # A hybrid-converted transformer needs the packed layout before every forward; the
    # layout is the same at every denoising step, so set it once per generation.
    if next(iter_hybrids(transformer), None) is not None:
        # frame_size and text_indices unconditionally: carrying them is free, and only
        # their consumers are gated (short_conv / text_state).
        set_layout(transformer, layout_from_indices(
            video_indices[num_condition_rows:], num_latent_frames, frame_h * frame_w,
            seq_len=position_ids.shape[0], frame_size=(frame_h, frame_w),
            text_indices=text_indices,
        ))

    # The scheduler counts sigma grid points, the terminal 0 included: one more than the
    # num_steps model evaluations.
    scheduler = MiniMaxH3Scheduler(shift=video_shift)
    audio_scheduler = MiniMaxH3Scheduler(shift=audio_shift)
    scheduler.set_timesteps(num_steps + 1, device=device)
    audio_scheduler.set_timesteps(num_steps + 1, device=device)
    paper = getattr(transformer, "_paper_runtime", None)
    if paper is not None:
        paper.begin(next(iter_hybrids(transformer)).layout, num_condition_rows,
                    scheduler.timesteps, audio_scheduler.timesteps,
                    frames=num_frames, steps=num_steps, seed=seed,
                    shifts=(video_shift, audio_shift))

    generator = torch.Generator(device).manual_seed(seed)
    # The conditioning noise is drawn first, one draw per keyframe or reference, before
    # the generated rows' noise (MiniMaxH3PrepareConditionLatentsStep's order).
    condition_rows = []
    for condition in condition_latents:
        noise = torch.randn(condition.shape, generator=generator, device=device, dtype=torch.float32)
        noised = scheduler.scale_noise(condition, KEYFRAME_NOISE_AUG, noise)
        condition_rows.append(patchify_video_latents(noised, patch))
    latents = torch.randn((1, channels, num_latent_frames, LATENT_H, LATENT_W),
                          generator=generator, device=device, dtype=torch.float32)
    video_rows = patchify_video_latents(latents, patch)
    if condition_rows:
        video_rows = torch.cat(condition_rows + [video_rows])
        if video_rows.shape[0] != video_indices.numel():
            raise ValueError(f"{video_rows.shape[0]} video rows (with conditioning) != "
                             f"{video_indices.numel()} layout rows")
    audio_rows = torch.randn((num_audio_latents * AUDIO_CHANNELS, 32),
                             generator=generator, device=device, dtype=torch.float32)

    if runtime is not None:
        runtime.barrier()
        torch.cuda.synchronize(device)

    seq_len = position_ids.shape[0]
    for step_index, (t, audio_t) in enumerate(zip(scheduler.timesteps, audio_scheduler.timesteps)):
        step_started = time.perf_counter()
        step_range = (nsight_capture.step(step_index, last=step_index == num_steps - 1)
                      if nsight_capture is not None else nvtx_range(f"denoise.step.{step_index}"))
        with step_range, metrics.denoise_step(step_index) if metrics else nullcontext():
            row_timesteps = torch.full((seq_len,), float(t), dtype=torch.float32, device=device)
            if num_condition_rows:
                row_timesteps[video_indices[:num_condition_rows]] = max(float(t), KEYFRAME_NOISE_AUG)
            row_timesteps[audio_indices] = float(audio_t)
            timestep, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
            noise_pred, audio_noise_pred = transformer(
                hidden_states=video_rows[None],
                audio_hidden_states=audio_rows[None],
                encoder_hidden_states=prompt_embeds[None],
                timestep=timestep,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                return_dict=False,
            )
            # only the generated rows step; the anchors ride through unchanged
            video_rows[num_condition_rows:] = scheduler.step(
                noise_pred[0, num_condition_rows:].float(), t, video_rows[num_condition_rows:],
                return_dict=False)[0]
            audio_rows = audio_scheduler.step(audio_noise_pred[0].float(), audio_t, audio_rows, return_dict=False)[0]
            if paper is not None:
                paper.guard.check("scheduler_output", video_rows, audio_rows)

        if step_seconds is not None:
            torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - step_started)

    if paper is not None:
        paper.finish()
    if runtime is not None:
        torch.cuda.synchronize(device)
        runtime.barrier()

    # Unpatchify (the AfterDenoise step's reshape) and unpack the channel-major audio rows.
    video_rows = video_rows[num_condition_rows:]
    rows = video_rows.reshape(-1, num_latent_frames, frame_h, frame_w, channels, *patch)
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = rows.reshape(-1, channels, num_latent_frames, LATENT_H, LATENT_W).contiguous()
    audio_latents = audio_rows.reshape(AUDIO_CHANNELS, num_audio_latents, 32).permute(0, 2, 1).contiguous()
    return latents, audio_latents


@torch.no_grad()
@nvtx_annotate("decode.video")
def _decode_video(latents, vae, device):
    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        video = vae.decode(latents * latents_std + latents_mean, return_dict=False)[0]
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    return (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu()


@torch.no_grad()
@nvtx_annotate("decode.audio")
def _decode_audio(audio_latents, audio_vae, device):
    audio_latents_mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    audio_latents_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
    audio = audio_vae.decode(audio_latents * audio_latents_std + audio_latents_mean, return_dict=False)[0]
    return audio.float().permute(1, 0, 2)[0].cpu()  # (2, num_samples)


@torch.no_grad()
def decode_and_save(latents, audio_latents, vae, audio_vae, out_path: str, device: str,
                    *, sequential=False, offload=None, metrics=None):
    """Opt-in staging releases denoiser weights, then decodes one VAE at a time.

    CPU outputs and evicted VAEs survive repeated renders. Each decoder is evicted
    even on failure. Legacy callers retain their existing simultaneous residency.
    Phase times include onload, decode, result transfer, and eviction.
    """
    phase = metrics.phase if metrics else (lambda *args: nullcontext())
    if sequential:
        if offload is None:
            raise ValueError("sequential decode requires a denoiser offload controller")
        with phase("release_denoiser"):
            offload.release()
            # Also makes repeated calls safe if a caller moved either decoder.
            vae.to("cpu")
            audio_vae.to("cpu")
            torch.cuda.empty_cache()
        with phase("video_decode"):
            try:
                vae.to(device)
                frames = _decode_video(latents, vae, device)
            finally:
                vae.to("cpu")
                torch.cuda.empty_cache()
        with phase("audio_decode"):
            try:
                audio_vae.to(device)
                audio = _decode_audio(audio_latents, audio_vae, device)
            finally:
                audio_vae.to("cpu")
                torch.cuda.empty_cache()
    else:
        vae.to(device)
        audio_vae.to(device)
        with phase("video_decode"):
            frames = _decode_video(latents, vae, device)
        with phase("audio_decode"):
            audio = _decode_audio(audio_latents, audio_vae, device)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Encode to a sibling temp name and rename into place, so a render that is killed
    # mid-encode never leaves a truncated clip under the final name (anything that
    # treats "the mp4 exists" as "done" would otherwise pick it up). os.replace is
    # atomic within a directory, so the final name only ever appears complete.
    # The suffix goes BEFORE the extension: PyAV infers the container format from the
    # output filename, and "clip.mp4.partial" makes av.open raise "Could not determine
    # output format". Same directory either way, which is what os.replace needs.
    stem, ext = os.path.splitext(out_path)
    partial_path = f"{stem}.partial{ext}"
    with nvtx_range("decode.output.media_mux"), phase("media_encode"):
        encode_video(frames, fps=FPS, output_path=partial_path,
                     audio=audio, audio_sample_rate=audio_vae.config.sampling_rate)
        os.replace(partial_path, out_path)
    print(f"wrote {out_path}", flush=True)
