#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=${1:-/home/ubuntu/openpi}
CONFIG_NAME=${2:-pi05_droid}

cd "$OPENPI_ROOT"
uv run python /home/ubuntu/cs231n-project/baselines/pi0/smoke_openpi_dummy.py \
  --config-name "$CONFIG_NAME"
