#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/cs231n-project

PY=/home/ubuntu/cs231n-project/training/.venv/bin/python
TRAIN=/home/ubuntu/cs231n-project/training/.venv/bin/lerobot-train
EVAL=/home/ubuntu/cs231n-project/simulation/docker/eval-lerobot-policy.sh
LOG_DIR=/home/ubuntu/cs231n-project/baselines/logs
mkdir -p "$LOG_DIR"

ROPE_H5=/home/ubuntu/cs231n-project/training/data/rope_full_mixed_v1.h5
ROPE_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_full_mixed_v1
ROPE_REPO=khush/softgym-ropeflatten-full-mixed-v1
RUN_NAME=act_rope_full_mixed_v1_50k
OUT_DIR=outputs/train/act_softgym_rope_full_mixed_v1_50k

STEPS=${STEPS:-50000}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-2}
CHUNK_SIZE=${CHUNK_SIZE:-25}
RUN_EVAL=${RUN_EVAL:-0}
EVAL_EPISODES=${EVAL_EPISODES:-10}

echo "[$(date -Is)] Waiting for full mixed rope HDF5..."
while [[ ! -s "$ROPE_H5" ]]; do
  ls -lh "$ROPE_H5" 2>/dev/null || true
  sleep 120
done

echo "[$(date -Is)] Found $ROPE_H5"
ls -lh "$ROPE_H5"

echo "[$(date -Is)] Validating HDF5 mapping..."
"$PY" baselines/softgym_hdf5_to_lerobot.py \
  --input "$ROPE_H5" \
  --output-root "$ROPE_ROOT" \
  --repo-id "$ROPE_REPO" \
  --instruction "straighten the rope" \
  --dry-run \
  2>&1 | tee "$LOG_DIR/convert_rope_full_mixed_v1_dry_run.log"

if [[ ! -d "$ROPE_ROOT" ]]; then
  echo "[$(date -Is)] Converting full mixed rope HDF5 to LeRobot..."
  "$PY" baselines/softgym_hdf5_to_lerobot.py \
    --input "$ROPE_H5" \
    --output-root "$ROPE_ROOT" \
    --repo-id "$ROPE_REPO" \
    --instruction "straighten the rope" \
    2>&1 | tee "$LOG_DIR/convert_rope_full_mixed_v1_lerobot.log"
else
  echo "[$(date -Is)] LeRobot dataset already exists at $ROPE_ROOT"
fi

echo "[$(date -Is)] Training ACT RopeFlatten full mixed baseline..."
"$TRAIN" \
  --dataset.repo_id="$ROPE_REPO" \
  --dataset.root="$ROPE_ROOT" \
  --policy.type=act \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.n_action_steps="$CHUNK_SIZE" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT_DIR" \
  --job_name="$RUN_NAME" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --save_freq=10000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_${RUN_NAME}.log"

echo "[$(date -Is)] Plotting training metrics..."
"$PY" baselines/plot_lerobot_metrics.py "$LOG_DIR/train_${RUN_NAME}.log" \
  --run-name "$RUN_NAME" \
  --output "baselines/plots/${RUN_NAME}_charts.png" \
  2>&1 | tee "$LOG_DIR/plot_${RUN_NAME}.log"

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "[$(date -Is)] Evaluating ACT RopeFlatten full mixed baseline..."
  SOFTGYM_SOFTWARE_GL=0 "$EVAL" \
    --env-name RopeFlatten \
    --policy-checkpoint "$OUT_DIR/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model" \
    --policy-name "$RUN_NAME" \
    --output-dir "data/evals/$RUN_NAME" \
    --num-episodes "$EVAL_EPISODES" \
    --horizon 75 \
    --img-size 128 \
    --success-threshold 0.8 \
    --save-every-video 1 \
    2>&1 | tee "$LOG_DIR/eval_${RUN_NAME}.log"
fi

echo "[$(date -Is)] ACT RopeFlatten full mixed baseline queue finished."
