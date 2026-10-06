#!/usr/bin/env bash
# Run from the repository root. No checkpoint access or inference is performed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/rtx5090
exec > >(tee artifacts/rtx5090/install.log) 2>&1
run() {
  printf '%q ' "$@" | tee -a artifacts/rtx5090/commands.log
  printf '\n' | tee -a artifacts/rtx5090/commands.log
  "$@"
}
export UV_NO_CACHE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
if [[ ! -x .venv/bin/python ]]; then
  run uv venv --no-cache --python 3.12 .venv
fi
run .venv/bin/python - <<'PY'
from pathlib import Path
import sys
import tomllib
assert sys.version_info[:2] == (3, 12), sys.version
deps = tomllib.loads(Path('pyproject.toml').read_text())['project']['dependencies']
deps = [dep for dep in deps if not dep.startswith('flash-attn-4')]
Path('artifacts/rtx5090/requirements.in').write_text(
    '# Project pins preserved; FA4 omitted for sm120; FLA not needed.\n'
    + '\n'.join(deps) + '\ntorch==2.13.0+cu130\n'
)
PY
requirements=artifacts/rtx5090/requirements.in
if [[ -f artifacts/rtx5090/requirements.lock.txt ]]; then
  requirements=artifacts/rtx5090/requirements.lock.txt
fi
run uv pip install --no-cache --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
  -r "$requirements" -c artifacts/rtx5090/requirements.in
# Avoid fetching every historical source blob. The project script still checks
# out its own pinned base and applies its own patches without modification.
if [[ ! -d diffusers/.git ]]; then
  run git clone --quiet --filter=blob:none --no-checkout \
    https://github.com/huggingface/diffusers.git diffusers
fi
export PIP='uv pip --no-cache'
export UV_PYTHON="${PWD}/.venv/bin/python"
export UV_CONSTRAINT="${PWD}/${requirements}"
# Supply an identity only to this process for git am, not to global git config.
export GIT_COMMITTER_NAME='RTX5090 Environment Setup'
export GIT_COMMITTER_EMAIL='rtx5090-setup@localhost'
run bash scripts/setup_diffusers.sh
run uv pip check --python .venv/bin/python
printf '%s\n' 'uv pip freeze --python .venv/bin/python > artifacts/rtx5090/freeze.txt' >> artifacts/rtx5090/commands.log
uv pip freeze --python .venv/bin/python > artifacts/rtx5090/freeze.txt
printf '\nINSTALL COMPLETE\n'
