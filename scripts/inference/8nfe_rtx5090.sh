#!/usr/bin/env bash
# Native cached-prompt 8 NFE, CPU assembly and synchronous whole-block offload.
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
    --config configs/inference/8nfe_rtx5090.yaml "$@"
