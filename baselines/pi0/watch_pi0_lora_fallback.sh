#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

PRIMARY_SESSION=${PRIMARY_SESSION:-pi0_cem_final_lora_40k_b4_ckpt5k}
PRIMARY_LOG=${PRIMARY_LOG:-"$REPO_ROOT/baselines/logs/pi0_cem_final_lora_40k_b4_ckpt5k_train.log"}

FALLBACK_SESSION=${FALLBACK_SESSION:-pi0_cem_final_lora_40k_b2_ckpt5k}
FALLBACK_LOG=${FALLBACK_LOG:-"$REPO_ROOT/baselines/logs/pi0_cem_final_lora_40k_b2_ckpt5k_train.log"}
FALLBACK_EXP_NAME=${FALLBACK_EXP_NAME:-softgym_rope_cem_final_lora_40k_b2_ckpt5k}

OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
OPENPI_DATASET_REPO_ID=${OPENPI_DATASET_REPO_ID:-khush/softgym-rope-cem-openpi-final}
OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS:-40000}
OPENPI_SAVE_INTERVAL=${OPENPI_SAVE_INTERVAL:-5000}
OPENPI_KEEP_PERIOD=${OPENPI_KEEP_PERIOD:-5000}
OPENPI_NUM_WORKERS=${OPENPI_NUM_WORKERS:-1}
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-60}

CRASH_PATTERN=${CRASH_PATTERN:-"RESOURCE_EXHAUSTED|out of memory|OutOfMemory|OOM|CUDA_ERROR_OUT_OF_MEMORY|No space left on device|ENOSPC|Resource exhausted"}

mkdir -p "$REPO_ROOT/baselines/logs"

echo "Watching primary tmux session: $PRIMARY_SESSION"
echo "Primary log: $PRIMARY_LOG"
echo "Fallback session: $FALLBACK_SESSION"
echo "Fallback log: $FALLBACK_LOG"

while true; do
  if tmux has-session -t "=$PRIMARY_SESSION" 2>/dev/null; then
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  echo "$(date -Is) primary session is no longer running"

  if [[ ! -f "$PRIMARY_LOG" ]]; then
    echo "Primary log is missing, not starting fallback: $PRIMARY_LOG" >&2
    exit 1
  fi

  if ! tail -n 400 "$PRIMARY_LOG" | grep -Eiq "$CRASH_PATTERN"; then
    echo "Primary stopped, but recent log does not look like OOM/resource/no-space failure."
    echo "Leaving fallback stopped so the cause can be inspected."
    exit 0
  fi

  echo "Detected crash pattern in recent primary log. Starting batch-size-2 fallback."

  if tmux has-session -t "=$FALLBACK_SESSION" 2>/dev/null; then
    echo "Fallback session already exists: $FALLBACK_SESSION"
    exit 0
  fi

  cd "$REPO_ROOT"
  OPENPI_ROOT="$OPENPI_ROOT" \
  OPENPI_DATASET_REPO_ID="$OPENPI_DATASET_REPO_ID" \
  OPENPI_EXP_NAME="$FALLBACK_EXP_NAME" \
  OPENPI_NUM_TRAIN_STEPS="$OPENPI_NUM_TRAIN_STEPS" \
  OPENPI_SAVE_INTERVAL="$OPENPI_SAVE_INTERVAL" \
  OPENPI_KEEP_PERIOD="$OPENPI_KEEP_PERIOD" \
  OPENPI_BATCH_SIZE=2 \
  OPENPI_NUM_WORKERS="$OPENPI_NUM_WORKERS" \
  SESSION="$FALLBACK_SESSION" \
  LOG="$FALLBACK_LOG" \
  "$REPO_ROOT/baselines/pi0/start_pi0_lora_training_tmux.sh"

  echo "Fallback started."
  exit 0
done
