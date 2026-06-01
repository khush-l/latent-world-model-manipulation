#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
CONFIG_NAME=${OPENPI_CONFIG:-pi05_softgym_rope_lora}
EXP_NAME=${OPENPI_EXP_NAME:-softgym_rope_lora_g5}
TRAIN_SESSION=${TRAIN_SESSION:-pi0_lora}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$OPENPI_ROOT/checkpoints/$CONFIG_NAME/$EXP_NAME"}
OPENPI_EVAL_OUT=${OPENPI_EVAL_OUT:-data/evals/pi0_rope_lora_g5}
OPENPI_NUM_EPISODES=${OPENPI_NUM_EPISODES:-10}
OPENPI_HORIZON=${OPENPI_HORIZON:-75}
LOG=${LOG:-"$REPO_ROOT/baselines/logs/pi0_lora_eval_after_train.log"}
POLL_SECONDS=${POLL_SECONDS:-60}

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] Waiting for training session '$TRAIN_SESSION' to finish"
while tmux has-session -t "=$TRAIN_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

echo "[$(date -Is)] Training session finished. Looking for checkpoints in $CHECKPOINT_ROOT"
if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
  echo "Checkpoint root not found: $CHECKPOINT_ROOT" >&2
  exit 1
fi

latest=$(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | awk '/^[0-9]+$/ {print}' | sort -n | tail -1)
if [[ -z "$latest" ]]; then
  echo "No numeric checkpoint dirs found under $CHECKPOINT_ROOT" >&2
  exit 1
fi
checkpoint="$CHECKPOINT_ROOT/$latest"
echo "[$(date -Is)] Evaluating checkpoint: $checkpoint"

cd "$REPO_ROOT"
OPENPI_ROOT="$OPENPI_ROOT" \
OPENPI_CONFIG="$CONFIG_NAME" \
OPENPI_CHECKPOINT="$checkpoint" \
OPENPI_EVAL_OUT="$OPENPI_EVAL_OUT" \
OPENPI_NUM_EPISODES="$OPENPI_NUM_EPISODES" \
OPENPI_HORIZON="$OPENPI_HORIZON" \
baselines/pi0/eval_pi0_lora_softgym_rope.sh

echo "[$(date -Is)] Eval complete: $REPO_ROOT/simulation/$OPENPI_EVAL_OUT"
