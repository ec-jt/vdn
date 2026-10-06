#!/usr/bin/env bash
# Fixed 124-frame benchmark with optional route, prefetch and all-rank Nsight capture.
set -euo pipefail
set -C
cd "$(dirname "$0")/.."
if [[ $# -lt 2 || $# -gt 5 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,60}$ || ! "$2" =~ ^(plain|nsys)$ || ! "${3:-broadcast}" =~ ^(broadcast|allgather)$ || ! "${4:-disabled}" =~ ^(disabled|next_block)$ || ! "${5:-baseline}" =~ ^(baseline|vda_fused|vda_overlap|vda_both)$ ]]; then
    echo 'Usage: bash scripts/benchmark_rtx5090_nsight.sh UNIQUE_LABEL {plain|nsys} [broadcast|allgather] [disabled|next_block] [baseline|vda_fused|vda_overlap|vda_both]' >&2
    exit 2
fi
profile="${5:-baseline}"
vda_fused=false
vda_overlap=false
case "$profile" in
    vda_fused) vda_fused=true ;;
    vda_overlap) vda_overlap=true ;;
    vda_both) vda_fused=true; vda_overlap=true ;;
esac
prefetch=false
[[ "${4:-disabled}" != next_block ]] || prefetch=true
apps=$(timeout -k 2s 10s nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "${apps//[[:space:]]/}" ]] || { echo 'GPUs busy' >&2; exit 1; }
out="artifacts/rtx5090/$1"
media="results/$1.mp4"
[[ ! -e "$out" && ! -e "$media" && ! -e "$media.inference.json" && ! -e "artifacts/latents/$1.mp4.latents.pt" ]] || exit 1
mkdir "$out"
printf '%s\n' "${3:-broadcast}" > "$out/route.txt"
printf '%s\n' "${4:-disabled}" > "$out/prefetch.txt"
printf '%s\n' "$profile" > "$out/selectedprofile.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=SYS NCCL_SHM_USE_CUDA_MEMCPY=0
export TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/rtx5090/8gpu-kernel-cache/inductor"
export TRITON_CACHE_DIR="$PWD/artifacts/rtx5090/8gpu-kernel-cache/triton"
unset FA_CLC
export VDN_NVTX=0 VDN_NSYS_CAPTURE=0
worker=(src/inference/infer_ulysses.py)
if [[ "$2" == nsys ]]; then
    export VDN_NVTX=1 VDN_NSYS_CAPTURE=1 VDN_PROFILE_DIR="$PWD/$out"
    worker=(--no-python bash scripts/nsys_rtx5090_rank.sh src/inference/infer_ulysses.py)
fi
(
    while true; do
        date -u +%FT%TZ
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits
        sleep 1
    done
) > "$out/activity.log" 2>&1 &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
.venv/bin/python - <<'PY' > "$out/source-sha256.json"
import hashlib,json,pathlib
print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ['src','scripts','configs'] for p in pathlib.Path(root).rglob('*') if p.suffix in ['.py','.sh','.yaml']},indent=2))
PY
status=0
/usr/bin/time -v .venv/bin/python scripts/supervise_rtx5090.py --seconds 900 -- \
    .venv/bin/python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 "${worker[@]}" \
    --config configs/inference/8nfe_rtx5090_8gpu.yaml \
    memory.shard_distribution=broadcast "parallel.standard_exchange=${3:-broadcast}" parallel.profile=false \
    "memory.weight_prefetch=$prefetch" \
    render.num_frames=124 render.num_steps=8 render.seed=42 render.prompt_file=prompts/example_0.pt \
    precision.dtype=bfloat16 precision.fp8.enabled=false kernels.softmax_backend=flex \
    kernels.inference_kernels=false "kernels.vda_fused=$vda_fused" "kernels.vda_overlap=$vda_overlap" \
    "render.out=$media" render.save_latents=true > "$out/launch.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out/exit.txt"
tail -n 40 "$out/launch.log"
exit "$status"
