#!/usr/bin/env bash
# Measured on the local eight-5090 Gen4 topology. Reference launcher is unchanged.
# All-pair P2P accelerates gathers; bounded broadcasts avoid slow P2P all-to-all.
set -euo pipefail
cd "$(dirname "$0")/../.."
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
# SHM-copy experiments stalled production despite passing isolated probes.
export NCCL_SHM_USE_CUDA_MEMCPY="${NCCL_SHM_USE_CUDA_MEMCPY:-0}"
printf 'Transport: NCCL_P2P_LEVEL=%s NCCL_SHM_USE_CUDA_MEMCPY=%s\n' "$NCCL_P2P_LEVEL" "$NCCL_SHM_USE_CUDA_MEMCPY"
exec bash scripts/inference/8nfe_rtx5090_8gpu.sh memory.shard_distribution=broadcast parallel.standard_exchange=broadcast "$@"
