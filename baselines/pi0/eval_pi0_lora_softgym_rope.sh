#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
OPENPI_CONFIG=${OPENPI_CONFIG:-pi05_softgym_rope_lora}
OPENPI_CHECKPOINT=${OPENPI_CHECKPOINT:-}
OPENPI_EVAL_OUT=${OPENPI_EVAL_OUT:-data/evals/pi0_rope_lora}
OPENPI_NUM_EPISODES=${OPENPI_NUM_EPISODES:-10}
OPENPI_HORIZON=${OPENPI_HORIZON:-75}

if [[ -z "$OPENPI_CHECKPOINT" ]]; then
  cat >&2 <<USAGE
Missing OPENPI_CHECKPOINT. Example:
  OPENPI_CHECKPOINT=/home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/20000 \
  baselines/pi0/eval_pi0_lora_softgym_rope.sh
USAGE
  exit 2
fi

cd "$REPO_ROOT"
OPENPI_ROOT="$OPENPI_ROOT" \
OPENPI_CONFIG="$OPENPI_CONFIG" \
OPENPI_CHECKPOINT="$OPENPI_CHECKPOINT" \
OPENPI_EVAL_OUT="$OPENPI_EVAL_OUT" \
OPENPI_NUM_EPISODES="$OPENPI_NUM_EPISODES" \
OPENPI_HORIZON="$OPENPI_HORIZON" \
baselines/pi0/eval_openpi_softgym_smoke.sh
