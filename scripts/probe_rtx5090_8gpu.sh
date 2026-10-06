#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "Probe requires idle GPUs" >&2; exit 1
fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
exec timeout --signal=TERM --kill-after=20s 420s .venv/bin/python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 scripts/probe_rtx5090_8gpu.py "$@"
