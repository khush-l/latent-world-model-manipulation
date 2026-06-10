#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-"$REPO_ROOT/training/.venv/bin/python"}
TRAIN=${TRAIN:-"$REPO_ROOT/training/.venv/bin/lerobot-train"}

DATASET_NAME=${DATASET_NAME:-rope_cem_final}
INPUT_H5=${INPUT_H5:-"$REPO_ROOT/training/data/cem_ropeflatten_final.h5"}
DATASET_ROOT=${DATASET_ROOT:-/home/ubuntu/lerobot_datasets/softgym_ropeflatten_cem_final}
DATASET_REPO=${DATASET_REPO:-khush/softgym-ropeflatten-cem-final}

SMOLVLA_STEPS=${SMOLVLA_STEPS:-50000}
SMOLVLA_BATCH_SIZE=${SMOLVLA_BATCH_SIZE:-2}
SMOLVLA_SAVE_FREQ=${SMOLVLA_SAVE_FREQ:-5000}
NUM_WORKERS=${NUM_WORKERS:-0}

SMOLVLA_RUN_NAME=${SMOLVLA_RUN_NAME:-smolvla_${DATASET_NAME}_stronger}
SMOLVLA_OUT=${SMOLVLA_OUT:-"$REPO_ROOT/outputs/train/$SMOLVLA_RUN_NAME"}
LOG_DIR=${LOG_DIR:-"$REPO_ROOT/baselines/logs"}

mkdir -p "$LOG_DIR"

if [[ ! -x "$TRAIN" ]]; then
  echo "Missing lerobot-train at $TRAIN" >&2
  exit 2
fi
if [[ ! -f "$INPUT_H5" ]]; then
  echo "Missing input HDF5: $INPUT_H5" >&2
  exit 2
fi

cd "$REPO_ROOT"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[$(date -Is)] Converting $INPUT_H5 to LeRobot dataset $DATASET_ROOT"
  "$PY" baselines/scripts/convert_lerobot.py \
    --input "$INPUT_H5" \
    --output-root "$DATASET_ROOT" \
    --repo-id "$DATASET_REPO" \
    --instruction "straighten the rope"
fi

echo "[$(date -Is)] Training SmolVLA run=$SMOLVLA_RUN_NAME dataset=$DATASET_REPO"
"$TRAIN" \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$SMOLVLA_OUT" \
  --job_name="$SMOLVLA_RUN_NAME" \
  --steps="$SMOLVLA_STEPS" \
  --batch_size="$SMOLVLA_BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --save_freq="$SMOLVLA_SAVE_FREQ" \
  --log_freq=50 \
  --wandb.enable=false
