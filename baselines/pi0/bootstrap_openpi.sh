#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=${1:-/home/ubuntu/openpi}

if ! command -v uv >/dev/null 2>&1; then
  UV_BOOTSTRAP="$HOME/.local/share/uv-bootstrap"
  python3 -m venv "$UV_BOOTSTRAP"
  "$UV_BOOTSTRAP/bin/pip" install --upgrade pip uv
  export PATH="$UV_BOOTSTRAP/bin:$PATH"
fi

if [[ ! -d "$OPENPI_ROOT/.git" ]]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI_ROOT"
else
  git -C "$OPENPI_ROOT" pull --ff-only
  git -C "$OPENPI_ROOT" submodule update --init --recursive
fi

cd "$OPENPI_ROOT"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo "OpenPI is ready at $OPENPI_ROOT"
