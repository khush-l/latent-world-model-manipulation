#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-"$REPO_ROOT/training/.venv/bin/python"}
TRAIN=${TRAIN:-"$REPO_ROOT/training/.venv/bin/lerobot-train"}

DATASET_NAME=${DATASET_NAME:-rope_geometric_5k_v3}
DATASET_ROOT=${DATASET_ROOT:-/home/ubuntu/lerobot_datasets/softgym_ropeflatten_geometric_5k_v3}
DATASET_REPO=${DATASET_REPO:-khush/softgym-ropeflatten-geometric-5k-v3}
INPUT_H5=${INPUT_H5:-"$REPO_ROOT/training/data/ropeflatten_geometric_5k_v3.h5"}

ACT_STEPS=${ACT_STEPS:-20000}
SMOLVLA_STEPS=${SMOLVLA_STEPS:-20000}
ACT_BATCH_SIZE=${ACT_BATCH_SIZE:-16}
SMOLVLA_BATCH_SIZE=${SMOLVLA_BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-2}

ACT_RUN_NAME=${ACT_RUN_NAME:-act_${DATASET_NAME}_h100}
SMOLVLA_RUN_NAME=${SMOLVLA_RUN_NAME:-smolvla_${DATASET_NAME}_h100}
ACT_OUT=${ACT_OUT:-"$REPO_ROOT/outputs/train/$ACT_RUN_NAME"}
SMOLVLA_OUT=${SMOLVLA_OUT:-"$REPO_ROOT/outputs/train/$SMOLVLA_RUN_NAME"}
LOG_DIR=${LOG_DIR:-"$REPO_ROOT/baselines/logs"}
PLOT_DIR=${PLOT_DIR:-"$REPO_ROOT/baselines/results/h100_direct_action_retrain"}

mkdir -p "$LOG_DIR" "$PLOT_DIR"

usage() {
  cat <<'USAGE'
Optionally retrain ACT and SmolVLA on H100.

Defaults target the geometric rope dataset used in the main π0 comparison:
  DATASET_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_geometric_5k_v3
  DATASET_REPO=khush/softgym-ropeflatten-geometric-5k-v3
  ACT_STEPS=20000
  SMOLVLA_STEPS=20000

After this finishes, use these checkpoints with run_h100_direct_action_evals.sh:
  ACT_CHECKPOINT=outputs/train/<act_run>/checkpoints/<step>/pretrained_model
  SMOLVLA_CHECKPOINT=outputs/train/<smolvla_run>/checkpoints/<step>/pretrained_model
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "$TRAIN" ]]; then
  echo "Missing lerobot-train at $TRAIN. Set TRAIN=/path/to/lerobot-train or install training env." >&2
  exit 2
fi

cd "$REPO_ROOT"

if [[ ! -d "$DATASET_ROOT" ]]; then
  if [[ ! -f "$INPUT_H5" ]]; then
    echo "Missing dataset root $DATASET_ROOT and missing source HDF5 $INPUT_H5" >&2
    exit 2
  fi
  echo "[$(date -Is)] Converting $INPUT_H5 to LeRobot dataset $DATASET_ROOT"
  "$PY" baselines/softgym_hdf5_to_lerobot.py \
    --input "$INPUT_H5" \
    --output-root "$DATASET_ROOT" \
    --repo-id "$DATASET_REPO" \
    --instruction "straighten the rope" \
    2>&1 | tee "$LOG_DIR/convert_${DATASET_NAME}_lerobot.log"
fi

plot_train_log() {
  local log=$1
  local run_name=$2
  "$PY" baselines/plot_lerobot_metrics.py "$log" \
    --run-name "$run_name" \
    --output "$PLOT_DIR/${run_name}_charts.png" \
    2>&1 | tee "$LOG_DIR/plot_${run_name}.log"
}

echo "[$(date -Is)] Training ACT run=$ACT_RUN_NAME dataset=$DATASET_REPO"
ACT_LOG="$LOG_DIR/train_${ACT_RUN_NAME}.log"
"$TRAIN" \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.chunk_size=25 \
  --policy.n_action_steps=25 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$ACT_OUT" \
  --job_name="$ACT_RUN_NAME" \
  --steps="$ACT_STEPS" \
  --batch_size="$ACT_BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --save_freq="$ACT_STEPS" \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$ACT_LOG"
plot_train_log "$ACT_LOG" "$ACT_RUN_NAME"

echo "[$(date -Is)] Training SmolVLA run=$SMOLVLA_RUN_NAME dataset=$DATASET_REPO"
SMOLVLA_LOG="$LOG_DIR/train_${SMOLVLA_RUN_NAME}.log"
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
  --num_workers=0 \
  --save_freq="$SMOLVLA_STEPS" \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$SMOLVLA_LOG"
plot_train_log "$SMOLVLA_LOG" "$SMOLVLA_RUN_NAME"

echo
echo "Retrain complete. Suggested eval exports:"
echo "export ACT_CHECKPOINT=$ACT_OUT/checkpoints/$(printf '%06d' "$ACT_STEPS")/pretrained_model"
echo "export SMOLVLA_CHECKPOINT=$SMOLVLA_OUT/checkpoints/$(printf '%06d' "$SMOLVLA_STEPS")/pretrained_model"
