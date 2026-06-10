#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-"$REPO_ROOT/training/.venv/bin/python"}
TRAIN=${TRAIN:-"$REPO_ROOT/training/.venv/bin/lerobot-train"}

DATASET_NAME=${DATASET_NAME:-rope_cem_final}
INPUT_H5=${INPUT_H5:-"$REPO_ROOT/training/data/cem_ropeflatten_final.h5"}
DATASET_ROOT=${DATASET_ROOT:-/home/ubuntu/lerobot_datasets/softgym_ropeflatten_cem_final}
DATASET_REPO=${DATASET_REPO:-khush/softgym-ropeflatten-cem-final}

ACT_RUN_NAME=${ACT_RUN_NAME:-act_rope_cem_final_25k}
ACT_OUT=${ACT_OUT:-"$REPO_ROOT/outputs/train/$ACT_RUN_NAME"}
ACT_STEPS=${ACT_STEPS:-25000}
ACT_BATCH_SIZE=${ACT_BATCH_SIZE:-16}
ACT_SAVE_FREQ=${ACT_SAVE_FREQ:-5000}
NUM_WORKERS=${NUM_WORKERS:-2}
CHUNK_SIZE=${CHUNK_SIZE:-25}

LOG_DIR=${LOG_DIR:-"$REPO_ROOT/baselines/logs"}
ACT_LOG=${ACT_LOG:-"$LOG_DIR/train_${ACT_RUN_NAME}.log"}

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

if [[ ! -x "$TRAIN" ]]; then
  echo "Missing lerobot-train at $TRAIN" >&2
  exit 2
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  if [[ ! -f "$INPUT_H5" ]]; then
    echo "Missing dataset root $DATASET_ROOT and source HDF5 $INPUT_H5" >&2
    exit 2
  fi
  echo "[$(date -Is)] Converting $INPUT_H5 to LeRobot dataset $DATASET_ROOT"
  "$PY" baselines/scripts/convert_lerobot.py \
    --input "$INPUT_H5" \
    --output-root "$DATASET_ROOT" \
    --repo-id "$DATASET_REPO" \
    --instruction "straighten the rope" \
    2>&1 | tee "$LOG_DIR/convert_${DATASET_NAME}_lerobot.log"
else
  echo "[$(date -Is)] Reusing LeRobot dataset at $DATASET_ROOT"
fi

echo "[$(date -Is)] Training ACT run=$ACT_RUN_NAME dataset=$DATASET_REPO"
"$TRAIN" \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.n_action_steps="$CHUNK_SIZE" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$ACT_OUT" \
  --job_name="$ACT_RUN_NAME" \
  --steps="$ACT_STEPS" \
  --batch_size="$ACT_BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --save_freq="$ACT_SAVE_FREQ" \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$ACT_LOG"

echo "[$(date -Is)] ACT training finished."
echo "ACT_CHECKPOINT=$ACT_OUT/checkpoints/$(printf '%06d' "$ACT_STEPS")/pretrained_model"
