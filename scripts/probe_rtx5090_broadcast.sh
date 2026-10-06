#!/usr/bin/env bash
# Separate supervised routing or toy probe; never launches production weights.
set -euo pipefail
set -C
mode=${1:?Usage: bash scripts/probe_rtx5090_broadcast.sh routing|toy [alltoall|broadcast]}
exchange=${2:-broadcast}
[[ "$mode" == routing || "$mode" == toy ]]
[[ "$exchange" == alltoall || "$exchange" == broadcast ]]
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B - <<'PY'
import os, runpy
p = runpy.run_path('scripts/probe_rtx5090_transport.py')
assert not p['production_pids'](), 'Production is active'
p['transport_configs'](os.environ)
PY
apps=$(timeout --signal=TERM --kill-after=2s 10s nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "${apps//[[:space:]]/}" ]] || { echo 'GPUs not idle'; exit 1; }
[[ -z "${NCCL_TOPO_DUMP_FILE:-}${NCCL_GRAPH_DUMP_FILE:-}" ]]
unset NCCL_DEBUG_FILE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1
out="artifacts/rtx5090/broadcast-$(date -u +%Y%m%dT%H%M%S)-${mode}-${exchange}-$$"
mkdir -m 700 "$out"
echo "$out"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$out/gpu-before.csv"
if [[ "$mode" == routing ]]; then
    command=(scripts/probe_rtx5090_broadcast.py --output-dir "$out" --exchange "$exchange")
else
    command=(scripts/probe_rtx5090_8gpu.py --report "$out/toy.json" --toy-only --distribution broadcast --exchange "$exchange")
fi
status=0
.venv/bin/python -B scripts/probe_rtx5090_transport.py --supervise -- \
    .venv/bin/python -B -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node=8 --max-restarts=0 "${command[@]}" > "$out/launch.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out/exit-code.txt"
nvidia-smi --query-compute-apps=pid --format=csv,noheader > "$out/compute-after.csv"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$out/gpu-after.csv"
echo "Exit $status: $out"
exit "$status"
