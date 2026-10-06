#!/usr/bin/env bash
# Parent-only, after production: bash -c 'bash scripts/probe_rtx5090_transport.sh --after-production LABEL'
set -euo pipefail
set -C
if [[ $# != 2 || "$1" != --after-production || ! "$2" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
    echo 'Usage: bash scripts/probe_rtx5090_transport.sh --after-production LABEL' >&2
    exit 2
fi
if [[ ! -x .venv/bin/python || ! -f scripts/probe_rtx5090_transport.py ]]; then
    echo 'Run from the repository root using the existing .venv' >&2
    exit 2
fi
# CPU-only fail-closed process guard, before any CUDA import or output creation.
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B - <<'PY'
import os
import runpy
probe = runpy.run_path('scripts/probe_rtx5090_transport.py')
pids = probe['production_pids']()
if pids:
    raise SystemExit(f'Refusing launch: active inference PIDs {pids}')
probe['transport_configs'](os.environ)
PY
# Read-only query. A query failure must not be interpreted as idle.
apps=$(timeout --signal=TERM --kill-after=2s 10s nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "${apps//[[:space:]]/}" ]]; then
    echo 'Refusing launch: GPU compute processes are still present' >&2
    exit 1
fi
# Prevent NCCL itself from writing to any pre-existing externally chosen file.
if [[ -n "${NCCL_TOPO_DUMP_FILE:-}" || -n "${NCCL_GRAPH_DUMP_FILE:-}" ]]; then
    echo 'Unset NCCL_TOPO_DUMP_FILE and NCCL_GRAPH_DUMP_FILE; use the exclusive run log' >&2
    exit 2
fi
unset NCCL_DEBUG_FILE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONDONTWRITEBYTECODE=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,P2P,SHM,NET,TUNING
out="artifacts/rtx5090/transport-$(date -u +%Y%m%dT%H%M%S)-${2}-$$"
mkdir -m 700 "$out"
printf 'Exclusive transport artifacts: %s\n' "$out"
timeout --signal=TERM --kill-after=2s 10s nvidia-smi --query-gpu=index,pci.bus_id,memory.used,utilization.gpu --format=csv > "$out/gpu-before.csv"
# Linux subreaper contains torchrun's detached worker sessions as well as launcher.
# TERM at 180s, KILL remaining descendants after at most 10s grace.
status=0
.venv/bin/python -B scripts/probe_rtx5090_transport.py --supervise -- \
    .venv/bin/python -B -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
    scripts/probe_rtx5090_transport.py --run --output-dir "$out" > "$out/launch.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out/exit-code.txt"
printf 'Transport exit=%s; log=%s/launch.log\n' "$status" "$out"
exit "$status"
