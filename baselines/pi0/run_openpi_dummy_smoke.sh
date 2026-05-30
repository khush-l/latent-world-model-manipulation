#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=${1:-/home/ubuntu/openpi}
CONFIG_NAME=${2:-pi05_droid}
UV_BIN=${UV:-${HOME}/.local/share/uv-bootstrap/bin/uv}

cd "$OPENPI_ROOT"
"$UV_BIN" run python /home/ubuntu/cs231n-project/baselines/pi0/smoke_openpi_dummy.py \
  --config-name "$CONFIG_NAME"
