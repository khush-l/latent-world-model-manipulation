#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
OPENPI_CONFIG=${OPENPI_CONFIG:-pi05_softgym_rope_lora}
EXP_NAME=${OPENPI_EXP_NAME:-softgym_rope_cem_final_lora_40k_b4_ckpt5k}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$OPENPI_ROOT/checkpoints/$OPENPI_CONFIG/$EXP_NAME"}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"5000 10000 15000"}
OPENPI_NUM_EPISODES=${OPENPI_NUM_EPISODES:-50}
OPENPI_HORIZON=${OPENPI_HORIZON:-75}
BASE_PORT=${BASE_PORT:-8865}
LOG_DIR=${LOG_DIR:-"$REPO_ROOT/baselines/logs"}

mkdir -p "$LOG_DIR"

echo "[$(date -Is)] Sequential checkpoint eval starting"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Steps: $CHECKPOINT_STEPS"
echo "Episodes per checkpoint: $OPENPI_NUM_EPISODES"
echo "Horizon: $OPENPI_HORIZON"

i=0
for step in $CHECKPOINT_STEPS; do
  checkpoint="$CHECKPOINT_ROOT/$step"
  if [[ ! -d "$checkpoint" ]]; then
    echo "[$(date -Is)] Missing checkpoint, skipping: $checkpoint" >&2
    continue
  fi

  port=$((BASE_PORT + i))
  out="data/evals/pi0_cem_final_ckpt${step}_episodes${OPENPI_NUM_EPISODES}"
  server_log="$LOG_DIR/pi0_eval_ckpt${step}_server.log"

  echo "[$(date -Is)] Evaluating checkpoint $step on port $port"
  echo "Output: $REPO_ROOT/simulation/$out"

  cd "$REPO_ROOT"
  CUDA_VISIBLE_DEVICES="" \
  JAX_PLATFORMS=cpu \
  OPENPI_ROOT="$OPENPI_ROOT" \
  OPENPI_CONFIG="$OPENPI_CONFIG" \
  OPENPI_CHECKPOINT="$checkpoint" \
  OPENPI_POLICY_PORT="$port" \
  OPENPI_EVAL_OUT="$out" \
  OPENPI_NUM_EPISODES="$OPENPI_NUM_EPISODES" \
  OPENPI_HORIZON="$OPENPI_HORIZON" \
  OPENPI_SERVER_LOG="$server_log" \
  baselines/pi0/eval_pi0_lora_softgym_rope.sh

  echo "[$(date -Is)] Finished checkpoint $step"
  i=$((i + 1))
done

echo "[$(date -Is)] Sequential checkpoint eval complete"
