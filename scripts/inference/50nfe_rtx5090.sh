#!/usr/bin/env bash
# Local Stage-B cached-prompt 50 NFE, BF16 synchronous whole-block CPU offload.
# Use a fresh render.out for each measurement; do not overwrite baseline artifacts.
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset FA_CLC
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/artifacts/rtx5090/kernel-cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/artifacts/rtx5090/kernel-cache/triton}"

exec .venv/bin/python src/inference/infer.py \
    --config configs/inference/50nfe_rtx5090.yaml "$@"
