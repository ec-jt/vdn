# RTX 5090 denoise and decode speedup plan

Goal: make the eight-GPU denoiser and video decode substantially faster, guided by
`nsys` and the existing phase metrics. Scope is denoise and decode only. No prompt
encoding, no training, no quality benchmarking in this effort.

Reference target: the paper reports 6.70 s DiT denoising on eight B200 for 345
frames at 1344x768, eight evaluations. That is a systems result on a mature
SGLang stack. It is a direction, not a promised number on PCIe 5090s.

## Historical measured state (345 frames, eight NFE)

Per-rank, from `artifacts/rtx5090/8gpu-345_final_deterministic-summary.json`:

| Region | Seconds per NFE | Share of 36.61 s |
| --- | ---: | ---: |
| Attention input redistribution | 18.57 | 50.7% |
| Attention output redistribution | 6.26 | 17.1% |
| Block weight gather | 4.77 | 13.0% |
| Attention compute, both branches | 3.69 | 10.1% |
| Everything else | 3.31 | 9.1% |

These are historical measurements, not authorization for another 345-frame model
run. Current optimization model runs are **124 frames only**, eight NFE, seed 42,
prompt 0, BF16, Flex, FP8 off. Decoder-only parity may use saved 345-frame latents.

The completed [all-rank baseline](../artifacts/rtx5090/baseline-nsys-20261005T0728Z/HANDOFF.md)
measured 131.332139 s denoise, first step 17.8876 s and steady steps about 16.19 s,
plus 4.221623 s video and 0.180026 s pixel processing. Setup is excluded.
Steady captured step 3 correlated kernel medians are input transport 6.7791 s,
output transport 2.2698 s and weight gather 4.6944 s per NFE. Weight gather is
about 29% of the current steady step, not negligible. Host VDA spans include
asynchronous waits and are not compute attribution. Profiled total time is not
ground truth (the baseline capture inflated denoise by about 7%).

## Established transport facts

From `artifacts/rtx5090/8gpu-TRANSPORT-AUDIT.md` and `8gpu-probe-final.json`:

- All-gather, all-P2P (`NCCL_P2P_LEVEL=SYS`): historically 0.094 s for 1.377 GB
  output, about 12.8 GB/s logical remote payload per rank, not physical wire
  throughput. Fresh baseline is 0.092796 s and 12.982 logical GB/s/rank.
- All-gather, defaults: 1.16 s, about 1.04 GB/s. Poor.
- All-to-all round trip, all-P2P: 3.76 s. Pathological.
- All-to-all round trip, defaults: 1.05 s.
- Shared-memory copy variant looked fast in probes but stalled production.

Implication: the fast primitive exists (all-gather under P2P). Production fell back
to broadcast routing because P2P all-to-all is unusable. Broadcast delivers the full
payload to every rank, far more bytes than needed. Useful attention receive rate
and all-gather remote payload rate have different numerators and cannot be compared
as equivalent wire bandwidth. Fresh pair probes show about 26 GB/s one-way and
51 GB/s bidirectional summed; pairs sharing an uplink aggregate about 26 GB/s
one-way, not 52. These are delivered payload rates, not link-byte counters.

The hardware is Gen4 x16 per endpoint with shared upstream links in pairs, no NVLink.
Capability is not throughput, so every candidate must be measured at production shape.

## Workstreams

### 1. Transport baseline and primitive selection

Baseline complete: bounded search did not find nccl-tests; existing PyTorch/NCCL
probes measured pair patterns, block gather and attention broadcast. No installation
is authorized. Production geometry is 38,510 rows, splits [4814]*7+[4812], 56
heads, seven/rank, BF16 widths 770 input and 256 output. Benchmark bounded candidate
routes at this exact geometry, with full coordinate identity, warm repetitions,
max-rank medians and memory. Do not retry pathological SYS all-to-all or the
production-hanging SHM CUDA memcpy variant blindly.

Exit criteria: a ranked table of primitives with correctness verified, and a
decision on whether sequence-to-heads can use all-gather plus local selection
instead of all-to-all, subject to memory fit.

### 2. Profiling instrumentation

Nsight Systems 2025.5.2 is already installed and verified with CUDA 13 / SM120.
No installation or environment change is needed or authorized.

Existing opt-in NVTX ranges show named regions: denoise step, per block, attention
input exchange, attention compute, attention output exchange, weight gather,
release, video decode job, audio decode, media mux. Convert or complement the
existing `profile_start` / `profile_end` events to emit NVTX ranges.

Use the existing per-worker wrapper and descendant-aware supervisor for all eight
ranks. Capture steps 2/3 and decode of a 124-frame render, excluding model load.
Keep normal instrumentation a no-op and introduce no normal per-region sync.
Validate all-rank CUDA/NVTX coverage and use innermost application range/kernel
correlation, not inclusive host waits. No 345-frame model comparison in this pass.

### 3. Attention exchange replacement

This is the highest-value change. Candidate order, each measured before keeping:

1. Replace the input exchange with an all-gather based reshuffle if memory fits.
2. Measure bounded chunked all-gather if useful for speed or memory.
3. Reduce payload: send only what each destination consumes where possible.
4. Overlap exchange with compute, following the paper's head-sharded side-stream idea,
   only after the exchange itself is fast.

Guard with full-coordinate identity tests for both independent directions,
nondivisible splits, head ordering, dtype, striding, repeated-call lifetime and
errors/timeouts. A production candidate must bitwise match both corrected video
and audio latents. Run two optimized plain124 renders and an interleaved fresh
broadcast plain124, then all-rank Nsight; report cold step separately from steady
steps, source/cache conditions and all-rank phase peaks. Keep broadcast selectable.

### 4. Weight gather reduction

Block gather is about 4.69 s per NFE in the fresh trace. The
[CPU review](../artifacts/rtx5090/optimization-review-20261005T0805Z.md) defers prefetch:
same-communicator ordering, event/buffer lifetime, release and shared-uplink
contention need dedicated evidence. Separate streams alone do not establish safe
overlap or speedup. The subsequent
[bounded follow-up](../artifacts/rtx5090/prefetch-20261005T0912Z/FINAL.md) tested and
retained one disabled-by-default next-block candidate: dedicated group/stream,
strict dtype ordering, readiness and consumer lifetime fences, forward/error
flush, and no weight/attention traffic overlap. Two plain124 runs average
115.720364 s versus fresh allgather-only121.050597 s (4.4033% reduction).
Actual step3 weight/compute overlap is0.744061 s/rank median; transport itself
does not become faster. Reserved peak increases1.283203 GiB. This completes the
bounded iteration; no further prefetch or transport tuning is authorized here.

### 5. Decode check

Decode is a smaller target, not a demonstrated practical floor. There are 196 jobs
balanced 25/24; rank-zero stitch is about 0.087 s in the trace. Other workers'
NCCL waits include rank-zero audio/media work, not solely payload transport.
Onload staging could be studied later, but benefit is unproven. Preserve clip 7,
stride 5, tile count 28 and exact blend/arithmetic order. Run existing full-geometry
decoder-only reference parity at both saved geometries, with exact float and RGB
checks; no new denoiser run at 345 frames.

## Constraints and safeguards

- Preserve the existing test suite and all prior artifacts. Do not overwrite outputs.
- Do not modify or re-download weights. Do not reinstall the model environment.
- No commits, installs, downloads, CUDA/driver/topology changes. Serialize GPU work.
- Keep the denoiser correctness status explicit. It is currently experimental with
  material single-GPU drift (video relative L2 about 0.36, audio about 0.083).
  This remains unresolved. Do not change precision, masks, scheduler or arithmetic.
  Exact transport parity against the corrected distributed baseline is not a
  claim of single-GPU equivalence or model quality.
- Run the original 130 regression tests plus all additions with CUDA enabled and
  no skips; inspect decoder parity metrics rather than trusting exit status alone.
- Measure every change with `nsys` plus the phase metrics, and keep a before/after table.

## Reporting

Maintain one running table: change, region affected, measured seconds before and
after, memory delta, correctness status. Update the operational guide when a change
is kept.

## Completed bounded124 paper-candidate evaluation

The [measured handoff](../artifacts/rtx5090/paper-bf16-20261005/model-HANDOFF.md)
closes this evaluation without promoting either candidate. Two interleaved
fused runs average116.272521 s denoise versus115.740921 s controls; combined
denoise/decode-encode is0.493084% slower. Video/audio final-latent relative L2
0.353122/0.129312 exceeds the predeclared0.001 gate. Drift repeats exactly in
both plain runs and the profiled run. Overlap is exact but its single result
is within noise, with586 MiB more sampled device memory. Full media arrays,
208 tests plus482 subtests with zero skips, and16 exported all-rank traces are
recorded. No model source changes, default promotion, combined candidate or
new345-frame run occurred. Retain baseline allgather/next-block prefetch.
