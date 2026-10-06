#!/usr/bin/env bash
# Fixed bounded experiment. Each underlying wrapper owns its supervised workers.
set -euo pipefail
set -C
cd "$(dirname "$0")/.."
out=artifacts/rtx5090/paper-bf16-20261005
git status --porcelain=v1 > "$out/model-initial-git-status.txt"
git diff --binary > "$out/model-initial-working.diff"
nvidia-smi > "$out/model-initial-gpu.txt"
for spec in control1:baseline:plain fused1:vda_fused:plain control2:baseline:plain fused2:vda_fused:plain overlap1:vda_overlap:plain fused-nsys:vda_fused:nsys; do
    IFS=: read -r name profile mode <<< "$spec"
    label="paper124-${name}-20261005T1237Z"
    printf '%s START %s\n' "$(date -u +%FT%TZ)" "$label"
    printf 'bash scripts/benchmark_rtx5090_nsight.sh %s %s allgather next_block %s\n' "$label" "$mode" "$profile"
    status=0
    bash scripts/benchmark_rtx5090_nsight.sh "$label" "$mode" allgather next_block "$profile" || status=$?
    printf '%s EXIT %s %s\n' "$(date -u +%FT%TZ)" "$label" "$status"
    [[ "$status" == 0 ]] || exit "$status"
    [[ "$(cat "artifacts/rtx5090/$label/exit.txt")" == 0 ]] || exit 1
done
printf '%s ALL_RUNS_COMPLETE\n' "$(date -u +%FT%TZ)"
