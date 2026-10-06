# Bounded persistent eight-GPU paper runtime

The opt-in persistent CLI keeps the same eight workers, default process group,
weight-prefetch group, compressed model shards, compiled kernels, geometry plans,
and decoder weights across sequential requests. It does not implement an HTTP
service or tensor parallelism. The legacy single-request entrypoint is unchanged.

## Reproduction

Run from the repository root with the existing environment and local weights.
Use a new label for every invocation. The launcher refuses existing labels,
nonidle GPUs, insufficient disk space, partial stacks, and batches outside 1..8.

```bash
.venv/bin/python scripts/benchmark_rtx5090_persistent.py UNIQUE_LABEL plain --requests 3
.venv/bin/python scripts/benchmark_rtx5090_persistent.py UNIQUE_PROFILE profile \
  --requests 3 --storage-root /mnt/nvme0/vdn-paper-20261006T114350Z-9edad0ad
```

The [launcher](../scripts/benchmark_rtx5090_persistent.py) reuses the existing
fixed [paper preset](../configs/inference/8nfe_rtx5090_paper_stack.yaml), offline
environment, fresh path reservation, source snapshots, GPU preflight, disk
watchdog, and bounded process supervisor. No installation or download occurs.
Plain mode retains the existing local compiler caches. External storage mode
uses fresh private caches under the new run directory. Profile results are
diagnostic, not unprofiled latency samples.

An optional local JSON manifest has exactly one object per request. Only cached
prompt paths and seeds may differ; model, LoRA, precision, and compute options
are immutable. Existing caches have no embedded model signature, so the batch
owner binds all admitted caches to its one model/configuration signature rather
than claiming cache provenance it cannot verify.

```json
[
  {"prompt_file": "prompts/example_0.pt", "seed": 42},
  {"prompt_file": "prompts/example_1.pt", "seed": 17},
  {"prompt_file": "prompts/example_0.pt", "seed": 42}
]
```

Pass the local manifest with `--request-manifest`. It never invokes an encoder.
The measured production workload remains 124 frames, eight NFE, prompt0, seed42.
Changed prompt/noise/schedule equivalence is tested on a tiny transformer, not a
full production render of every possible conditioning mode.

## Resource and request ownership

- [PersistentPaperRuntime](../src/inference/utils/persistent.py:68) owns the
  installed wrappers, control group, bounded current-geometry cache, and retained
  source modules. Rank zero retains only timestep, block AdaLN and output AdaLN
  modules on CPU, not a complete merged transformer. Measured source storage is
  24.347 GiB on rank zero and zero on peers.
- [PaperRequestState](../src/inference/utils/paper_runtime.py:109) owns each fresh
  finite guard, schedule, tables, plans dictionary, counters, rotary view, and
  step index. Closing tables invalidates old step views. The legacy single-use
  begin guard remains in force within each request object.
- [The batch runner](../src/inference/infer_persistent.py) owns fresh prompt,
  conditions, noise/scheduler execution, latents and output paths per request.
  It drops GPU outputs before the next request. No dummy NFE is run.
- [finish_request()](../src/inference/utils/weight_sharding.py:573) drains pending
  prefetch work without destroying shards or the weight group. Terminal release
  occurs only at batch shutdown. A transport drain failure poisons the owner.
- Predictable admission errors converge through the CPU control group before
  request NCCL payloads. Finite failures close request tables and drain owned
  work. This CLI aborts a failed batch; it does not promise automatic retry after
  CUDA, NCCL, process, or transport failure. The existing supervisor owns fatal
  teardown. Tests separately show clean admission rejection and subsequent group
  reuse, and request-local failure cleanup with resident shards intact.

Each request recomputes all 408 rank-zero AdaLN projections. The CPU sources are
staged one projection at a time. Thus preparation still has legitimate source
weight H2D traffic. Warm denoise has no per-step weight H2D, and warm decode has
no repeated decoder weight onload. Geometry reuse depends on the complete
layout, bounds, anchors, device, heads and budget key, never only tensor shape.

## Explicit decoder policy

The first implementation supports one explicit policy: resident FP32 decoder
weights. Every rank retains the decode-only video VAE; rank zero also retains
the audio VAE. Encoder modules are not loaded for video. The existing video
decode function is called with caller-owned lifecycle. Clip length seven,
stride five, 28 spatial tiles, spatial/temporal blend order, normalization,
FP16 decode autocast over FP32 weights, and GPU pixel postprocessing are unchanged.
CPU video/audio output precedes the atomic, rank-zero media write.

For the measured three-request run, maximum physical memory sampled every 200 ms
was **31,143 MiB on rank zero** and **26,635 MiB on each peer**. Against the
reported usable capacity of 32,607 MiB, rank zero had 1,464 MiB (1.43 GiB) of
sampled headroom. Peak allocator
usage was 22.219 GiB allocated and 27.357 GiB reserved. This fits the fixed tested
workload but has narrow rank-zero physical headroom. Sampling is not a proof of
the instantaneous physical maximum. Larger shapes, concurrent requests, and
unbounded serving are not admitted or qualified. No hidden fallback changes
precision or evicts a denoiser rank.

## Measured scope

The completed plain batch used one model setup of 155.557 seconds. Complete
request walls, using the maximum rank, were 101.092, 94.531 and 94.313 seconds.
Preparation was 3.199, 3.108 and 2.912 seconds. Denoise loops were 88.425, 85.484
and 85.377 seconds. Decode plus postprocess/encode was 9.433, 5.903 and 5.987
seconds. The first request includes decoder loading and onload. The full launcher
wall was 462.418 seconds for three requests, including one-time setup and teardown.

The retained fresh single-request deterministic launchers took 272.447 and
271.978 seconds each. The warm persistent complete-request mean of 94.422 seconds
is 65.31% below their mean whole-process wall, but these scopes differ: persistence
removes repeated setup. The denoise loop itself improves only modestly, from the
retained 88.602-90.016 seconds to 85.377-85.484 seconds. This is not an eightfold
denoise speedup or a quality-equivalence claim.

All three requests have byte-identical video/audio latents and complete decoded
streams to the retained deterministic candidate. All are finite and decode to
124 frames, 1344x768 at 24 fps, with stereo 32-kHz audio. Against the retained
BF16 control, latent relative L2 remains 0.533661 video and 0.343765 audio.
The user accepts this deterministic low-precision drift for performance work.
The previous 0.001 gate does not block this deliverable, but no perceptual or
quality-equivalence qualification is claimed.

Detailed timings, complete hashes, sampled peaks and per-rank identities are in
the [comparison](../artifacts/rtx5090/paper-stack-20261005/persistent-20261006/comparison.json).
The [raw batch](../artifacts/rtx5090/paper-stack-20261005/integration/persistent-three-20261006T2006Z/batch.json)
contains all per-request rank metrics. Each request has its own media, latent
file and metrics. No earlier artifact was removed or overwritten.

## Profile evidence

The [all-rank profile summary](../artifacts/rtx5090/paper-stack-20261005/persistent-20261006/nsight-summary.json)
contains 24 reports and SQLite exports: eight ranks times request-two steps2/3,
request-two decode, and the request-two-to-three boundary. PID identity is stable
across all three captures. The boundary contains both the completed-request
marker and the next preparation marker on every rank. No model-setup, MXFP8
conversion, or decoder-load range appears in any warm capture.

The [before/after trace inventory](../artifacts/rtx5090/paper-stack-20261005/persistent-20261006/trace-before-after.json)
reuses the retained historical paper-stack profile. Two denoise steps still
upload exactly eight bytes per rank. Decode H2D falls from 10,299,433,927 bytes
to 776 bytes on rank zero, and from 9,694,125,280 bytes to 496 bytes on each peer.
These remaining copies are small metadata, not decoder weights. Request-three
preparation legitimately uploads 26,295,907,388 bytes on rank zero, including
CPU AdaLN source staging, and 153,827,840 bytes per peer for tables and prompt.
No claim of zero whole-request H2D traffic is made.

The external run is recorded in its
[storage receipt](../artifacts/rtx5090/paper-stack-20261005/integration/persistent-profile-20261006T2014Z/storage-pointer.json).
Reports and exports occupy about 296 MiB, below the 64 GiB budget. The separate
[profile comparison](../artifacts/rtx5090/paper-stack-20261005/persistent-20261006/profile-comparison.json)
verifies all three profiled outputs against the retained deterministic candidate.
Profiled timings are not used for the performance result.

After the full runs, the entrypoint gained coordinated error votes around manifest
parsing and rank-zero metrics writes. These are control-only hardening changes;
the model, sampler and decoder arithmetic are unchanged. The measured source
snapshots precede these final votes. The real two-rank failure-vote regression
passed afterward. No additional full render was run for this control-only change.
# Permanent tensor/head parallel follow-up

The default gathered-weight Ulysses policy is unchanged. The separate
[permanent TP policy](tensor_parallel_rtx5090.md) adds an explicit launcher option
for seven resident heads per rank and row-parallel residual reductions. Its
measurements and numerical drift must not be conflated with the historical
persistent results below.
