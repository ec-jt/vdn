#!/usr/bin/env bash
# Existing local environment and weights only. Never install or fetch assets.
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset FA_CLC
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/artifacts/rtx5090/8gpu-kernel-cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/artifacts/rtx5090/8gpu-kernel-cache/triton}"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "Refusing to launch while GPU compute processes exist" >&2
    exit 1
fi
exec timeout --signal=TERM --kill-after=30s 1800s \
    .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    src/inference/infer_ulysses.py --config configs/inference/8nfe_rtx5090_8gpu.yaml "$@"
