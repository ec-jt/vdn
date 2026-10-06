#!/usr/bin/env bash
# Separate, opt-in encoding phase. Never invoked by the native render launcher.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PROCESSOR_ROOT:?Set PROCESSOR_ROOT to an existing local processor directory}"
: "${TEXT_ENCODER_ROOT:?Set TEXT_ENCODER_ROOT to an existing local encoder directory}"
: "${ENCODER_GPU:?Set ENCODER_GPU to one explicitly scheduled physical GPU index}"
[[ -d "$PROCESSOR_ROOT" && -d "$TEXT_ENCODER_ROOT" ]] || {
  echo 'Both encoder roots must be existing local directories; no downloads are allowed.' >&2
  exit 2
}
[[ "$ENCODER_GPU" =~ ^[0-9]+$ ]] || {
  echo 'ENCODER_GPU must select exactly one physical GPU index.' >&2
  exit 2
}
export CUDA_VISIBLE_DEVICES="$ENCODER_GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
exec "$ROOT/.venv/bin/python" "$ROOT/src/inference/encode_prompt.py" "$@" \
  --local_only --cpu_offload --no_overwrite --device cuda:0 \
  --processor_root "$PROCESSOR_ROOT" --text_encoder_root "$TEXT_ENCODER_ROOT"
