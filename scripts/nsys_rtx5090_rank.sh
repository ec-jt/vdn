#!/usr/bin/env bash
# Invoked by torchrun --no-python, so every worker gets its own CUDA tracer.
set -euo pipefail
: "${VDN_PROFILE_DIR:?Must specify a fresh output directory}"
: "${LOCAL_RANK:?Must be launched by torchrun}"
exec /usr/local/cuda/bin/nsys profile --sample=none --cpuctxsw=none \
    --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=repeat \
    --output="$VDN_PROFILE_DIR/rank-$LOCAL_RANK" \
    .venv/bin/python "$@"
