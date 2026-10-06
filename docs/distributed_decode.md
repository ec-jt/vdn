# Eight-GPU pinned H3 video decoder

The independent [decoder module](../src/inference/utils/distributed_decode.py)
distributes the installed H3 VAE's spatial tiles within its existing temporal
clips. It does not split the video into independent temporal sequences or keep a
causal cache. No FreeVideo implementation was copied. The source includes Apache
attribution for temporal orchestration adapted from Diffusers.

## Preserved semantics

- FP32 decoder weights, FP32 input latents, CUDA FP16 video autocast.
- Seven latent frames per noncausal clip, latent stride five, 28 raw output frames.
- Original frame slices: main frames 3:20, tail frames 23:28. The previous tail
  blends into the next main with the installed five-frame ramp.
- Original global latent padding, output trimming, normalization and dtype.
- Original spatial tile geometry and vertical **then** horizontal blending against
  unmodified neighbor tiles, followed by original crop and concatenation.
- At 768 x 1344: 28 spatial jobs per clip, 196 jobs for 124 frames, 560 for 345.

Stable global row-major job IDs determine cyclic rank ownership. Rank zero
receives at most one clip's raw tiles, invokes the installed spatial stitch helper,
and restores temporal order. Batched point-to-point transfers avoid replicating
decoded video onto every GPU. No decoded-video all-gather is used. Small control
and metric collectives still synchronize clip phases; compute/communication are
not pipelined across clips. The default CPU-output API bounds GPU output storage;
the production callback uses GPU output for the unchanged CUDA pixel arithmetic.
Thus callback output storage grows with video length even though gathering does not.

## APIs and lifecycle

[`load_video_decoder()`](../src/inference/utils/distributed_decode.py:77) constructs
the VAE on meta and reads only local decoder/post-quant safetensors. Encoder and
quant-conv payloads are omitted; the nonpersistent decoder rotary buffer is
reconstructed on CPU. It returns an evaluation-only CPU decoder with tiling enabled.
No Hub fallback or weight mutation is performed. The loaded decoder has
2,423,531,096 parameters, about 9.028 GiB in FP32 per rank.

[`distributed_decode()`](../src/inference/utils/distributed_decode.py:231) returns
a pair: raw B,C,T,H,W output on group rank zero (no output on other ranks), and a
metrics dictionary on every rank. All ranks supply identical normalized FP32
latents and options. It supports CPU/Gloo for tests and CUDA/NCCL for production.
The group argument supports a subgroup with its own root. Configuration mismatch
and predictable local errors converge before transport; peer death or transport
failure still relies on process-group and launcher timeouts.

Managed execution onloads only decoder/post-quant modules. For CPU-backed CUDA
calls, immutable CPU storage survives the call and is restored in the finally
block, avoiding a redundant approximately 9 GiB GPU-to-CPU weight copy per rank.
Unmanaged execution leaves placement to the caller. No persistent output or cache
is retained. Concurrent calls on the same model/group and training are unsupported.

The [all-rank callback](../src/inference/utils/distributed_decode_callback.py)
is the default only for the existing resident-weight-shards policy selected by
[`_select_decoder()`](../src/inference/infer_ulysses.py:67). The user-described
gpu-resident-shard policy corresponds to this existing configuration name; no
new policy alias was added. Legacy decoding and explicit callbacks are preserved.

[`run()`](../src/inference/infer_ulysses.py:77) invokes the callback with configuration,
model, runtime, video latents, and audio latents only after collective denoiser
release. A dictionary return is recorded under decoder metrics; callbacks returning
no value remain supported. The default callback checks release, loads video on all
ranks, performs distributed video decode, postprocesses rank zero's pixels, and
clears every video decoder before rank zero alone loads and decodes audio. Audio
is evicted before the single atomic media writer. Existing final or partial outputs
are rejected rather than overwritten. No decoder is retained on the model.

## Reproduce decoder-only measurements

Use idle GPUs and fresh report/log/output names. The commands below use existing
local models and latent artifacts; they do not run a denoiser.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4

# Bounded 22-frame, 512x512 crop, nine jobs exercising all eight ranks.
timeout --signal=TERM --kill-after=20s 300s .venv/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/benchmark_distributed_decode.py --probe \
  --report artifacts/rtx5090/decoder-probe-new.json

# Full saved 124-frame latents, ordinary reference plus two distributed calls.
timeout --signal=TERM --kill-after=20s 420s .venv/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/benchmark_distributed_decode.py \
  --report artifacts/rtx5090/decoder-124-new.json

# Optional real audio/writer callback test after the comparison, still no denoiser.
# Add: --callback-out results/decoder_8gpu_124_new.mp4

# For 345 frames, additionally select:
# --latents artifacts/latents/8nfe_rtx5090_345.mp4.latents.pt

# Run separately after GPU benchmarks exit.
CUDA_VISIBLE_DEVICES=0 RUN_ULYSSES_REFERENCE_CUDA=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

The [benchmark](../scripts/benchmark_distributed_decode.py) loads the ordinary full
installed single-GPU VAE as reference, then separately loads decoder-only VAEs on
all ranks. It compares every raw float and every pre-codec RGB byte, reports finite
status, maximum/mean error, RMSE, differing-value counts and RGB hashes. Repeated
distributed calls use the same decoder objects. Checkpoint loading, validation
copies and comparison are outside the comparable video timing. Onload, decoding,
transport/stitching, CUDA pixel conversion, RGB CPU transfer, eviction and final
allocator cache clearing are inside final benchmark video timings. Startup/imports,
audio and media encoding are outside those timings. Optional callback timings are
reported separately. Disk caches are retained; first calls are not globally cold.

Results, per-rank timings and caveats are in the
[measurement report](../artifacts/rtx5090/DISTRIBUTED_DECODER.md).
