#!/usr/bin/env bash
# One renderer at a time; unique label protects logs, media, records and latents.
set -euo pipefail
set -C
cd "$(dirname "$0")/.."
if [[ $# != 2 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ || ! "$2" =~ ^(124|345)$ ]]; then
    echo 'Usage: bash scripts/benchmark_rtx5090_production.sh LABEL {124|345}' >&2
    exit 2
fi
label="$1"
frames="$2"
stem="artifacts/rtx5090/8gpu-${label}"
out="results/8nfe_rtx5090_8gpu_${label}.mp4"
for path in "$stem.log" "$stem-activity.log" "$stem-exit.txt" "$out" "$out.inference.json" "artifacts/latents/$(basename "$out").latents.pt"; do
    if [[ -e "$path" ]]; then echo "Refusing existing artifact: $path" >&2; exit 1; fi
done
apps=$(timeout 10s nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "${apps//[[:space:]]/}" ]]; then echo 'GPUs are busy' >&2; exit 1; fi
mkdir -p artifacts/rtx5090 results
(
    while true; do
        date -u +%FT%TZ
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits
        ps -eo rss,args | awk '/[s]rc\/inference\/infer_ulysses.py/ {sum+=$1} END {print "aggregate_inference_rss_kib",sum+0}'
        sleep 1
    done
) > "$stem-activity.log" 2>&1 &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
status=0
# Exclusive log descriptor rather than tee (which otherwise truncates old logs).
/usr/bin/time -v bash scripts/inference/8nfe_rtx5090_8gpu_transport.sh \
    "render.num_frames=$frames" "render.out=$out" render.save_latents=true \
    > "$stem.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$stem-exit.txt"
tail -n 36 "$stem.log"
exit "$status"
