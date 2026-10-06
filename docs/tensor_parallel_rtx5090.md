# Permanent eight-way tensor/head parallelism

This is an explicit, default-off persistent inference policy for the tested
124-frame, eight-NFE workload on eight RTX 5090 GPUs. It uses the complete paper
stack and local weights. It does not change the encoder, resolution, NFE count,
decoder arithmetic, or checkpoint files. The existing gathered-weight Ulysses
path remains the default.

### Optional producer/cache candidate

The persistent launcher also accepts three independent, default-off options:
`--fused-producers`, `--fast-finite`, and `--cache-adaln`. All require
`--tensor-parallel`. The producer option changes fixed-tree prenorm arithmetic;
deterministic trajectory drift is expected. The finite option preserves all
coordinated failure boundaries. The cache retains one immutable table set with
fresh request views and an explicit byte budget.

One combined three-request batch passed complete finite/media repeatability
checks, with warm mean 50.113 seconds. This is preliminary, not a promoted
default: fresh-control, second-process and all-rank profile qualification were
blocked by unrelated GPU users. See the [candidate handoff](../artifacts/rtx5090/paper-stack-20261005/producer-20261006/HANDOFF.md)
for implementation contracts, measurements, exact commands and remaining work.

## Run

From the repository root, use a fresh label:

```bash
.venv/bin/python scripts/benchmark_rtx5090_persistent.py UNIQUE_TP_LABEL plain \
  --tensor-parallel --requests 3 --timeout-seconds 1500
```

Omit `--tensor-parallel` to run the gathered-weight Ulysses control. The launcher
records the explicit selection in its environment manifest. Direct persistent
workers accept `VDN_PERMANENT_TP=1`; the ordinary single-request worker does not
select TP. The resident decoder policy is explicit, with no hidden OOM fallback.

For all-rank Nsight profiling, use the existing private external storage root:

```bash
.venv/bin/python scripts/benchmark_rtx5090_persistent.py UNIQUE_TP_PROFILE profile \
  --tensor-parallel --requests 3 --timeout-seconds 1500 \
  --storage-root /mnt/nvme0/vdn-paper-20261006T114350Z-9edad0ad
```

The profile captures warm request-two denoise steps 2/3, its complete distributed
decode, and the request-two to request-three boundary. Profiled time is diagnostic,
not an unprofiled speed sample. Setup and request preparation remain separate.

## Ownership and arithmetic

[Permanent ownership](../src/inference/utils/tensor_parallel.py) retains complete
residual rows on every rank and seven of the 56 attention heads per rank.
Q/K/V output width is 896. Each MLP owns 1,792 values and the matching 1,792 gates.
The two attention output projections and MLP down projection own their K slices.
Each block has two residual-width all-reduces: merged attention output and MLP.
There are no attention row/head exchanges, no frame-mean all-reduce, and no block
weight gathers on this route.

Rank zero alone performs the existing strict CPU assembly and adapter merge.
Startup distributes compact tensors through bounded staging buffers. Each final
GPU field owns independent storage. Encoded MXFP8 scales are unswizzled, sliced
by block32 groups, and repacked into canonical 128-row/four-column tiles. Encoded
scales are never sliced as if they were a natural matrix. No floating weight
shadow or replicated gathered backing is retained.

Head-dependent beta, output gates, alpha parameters and depthwise convolutions
are sliced consistently. Shared bottleneck projections and head-width norms are
replicated. Prompt states, full raw Q/K, rotary tables, global modality rows and
anchor semantics remain intact. The frame mean is computed from the full local
residual, without an eightfold distributed sum. Bias is applied once after SUM.

The existing rank-zero CPU AdaLN source owner retains 24.347 GiB of source
weights. Fresh request tables are replicated on GPU; large per-block AdaLN
projection weights are not placed on GPU. Replicated root modules total
1,723,246,144 bytes. All 50 compact blocks total 2,906,843,000 bytes per rank.
Total denoiser weights are 4.312107 GiB per rank, before tables, activations,
decoders and runtime allocations. Runtime inventories check exact storage sizes.

## Numerical policy and limits

The user accepts deterministic low-precision and reduction drift. TP changes
GEMM K reduction boundaries, rounds local BF16 partials, and merges the softmax
and VDA partial outputs before their shared SUM. FP32 algebra is unchanged, but
this is not byte-equivalent to the gathered-weight path. The old 0.001 trajectory
gate is not a TP acceptance gate. Nonfinite output, missing prompt/mask/head
semantics, memory corruption and unexplained nonrepeatability remain failures.

The deterministic paper pointwise compiler policy is preserved. Distributed
finite votes occur before NCCL payload collectives. The VDA side stream records
input and permanent-weight lifetimes and joins before the output reduction.
Request failure poisons the TP controller; the batch exits rather than retrying
on possibly unhealthy resources. Teardown drains streams and dematerializes
owned weights. Selected aliases or distinct storage views are rejected rather
than silently untied. There is no distributed training API.

## Measured evidence

The [real-block comparison](../artifacts/rtx5090/paper-stack-20261005/tp-20261006/block-comparison.json)
uses captured effective block-zero weights and full real packed inputs: 38,510
rows, width 5,376, 56 heads. It includes actual baseline resident weight gathers
and attention exchanges. After two warmups, ten synchronized maximum-rank repeats
have median 220.337 ms for gathered Ulysses and 123.092 ms for TP, 44.13% lower.
A single-block baseline cannot hide next-block weight gathering under the
preceding block's tail; full-request measurements are the integration decision.

The [first three-request verification](../artifacts/rtx5090/paper-stack-20261005/tp-20261006/full1-comparison.json)
records full-request times 69.083, 60.611 and 60.600 seconds, with denoise loops
56.544, 51.437 and 51.343 seconds. All three latent and complete decoded streams
repeat byte-exactly. Against the retained gathered paper path, video/audio latent
relative L2 is 0.427086/0.295893. This is measured drift, not perceptual equivalence.
The maximum sampled physical usage is 27,741 MiB on rank zero, versus 32,607 MiB
capacity. Sampling every 200 ms can miss instantaneous peaks. Larger workloads
and concurrent requests are not qualified.

Two fresh unprofiled three-request TP batches and one fresh three-request control
batch completed. [Timing comparison](../artifacts/rtx5090/paper-stack-20261005/tp-20261006/timing-comparison.json):

| Warm mean | Gathered Ulysses | Permanent TP | Reduction |
| --- | ---: | ---: | ---: |
| Complete request | 94.304573 s | 60.475643 s | 35.87% |
| Denoise loop | 85.278639 s | 51.356691 s | 39.78% |
| Decode/postprocess/encode | 6.018292 s | 6.008984 s | 0.15% |

Control means use requests 2/3. TP means use requests 2/3 from both fresh batches.
First requests, setup and complete process walls remain in the reports, not
discarded from those measurements. All six plain TP requests and all three
profiled requests have byte-identical latents and complete decoded streams.
The fresh control remains byte-identical to the retained gathered paper path.
Against the older BF16 control, TP video/audio relative L2 is 0.533814/0.385825.
No perceptual-quality assessment was performed.

All-rank [trace inventory](../artifacts/rtx5090/paper-stack-20261005/tp-20261006/profile-tp-trace.json)
contains 24 production captures and their SQLite references. Across two captured
denoise steps, each rank executes exactly 100 attention SUMs and 100 MLP SUMs,
each with a 414,059,520-byte BF16 payload. There are no block weight gathers or
attention all-gathers. Median NCCL device interval union is 5.913256 seconds;
non-NCCL compute union is 5.900331 seconds. Unions are not additive critical paths.
H2D is exactly two four-byte copies per rank during denoise. Warm decode has no
weight reloads; next-request preparation legitimately stages CPU AdaLN weights.

The [task handoff](../artifacts/rtx5090/paper-stack-20261005/tp-20261006/HANDOFF.md)
records commands, exact memory evidence, test limits and all external artifact
paths. Do not infer a universal GPU speedup or a nine-second request target from
these measurements.
