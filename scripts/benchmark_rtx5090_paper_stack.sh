#!/usr/bin/env bash
# Offline research launcher. No arbitrary renderer dotlist arguments accepted.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python scripts/benchmark_rtx5090_paper_stack.py "$@"
