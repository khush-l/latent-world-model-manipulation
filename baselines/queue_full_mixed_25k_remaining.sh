#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/cs231n-project

PY=/home/ubuntu/cs231n-project/training/.venv/bin/python
TRAIN=/home/ubuntu/cs231n-project/training/.venv/bin/lerobot-train
EVAL=/home/ubuntu/cs231n-project/simulation/docker/eval-lerobot-policy.sh
LOG_DIR=/home/ubuntu/cs231n-project/baselines/logs
PLOT_DIR=/home/ubuntu/cs231n-project/baselines/plots/new
mkdir -p "$LOG_DIR" "$PLOT_DIR"

EVAL_EPISODES=${EVAL_EPISODES:-10}
ROPE_HORIZON=${ROPE_HORIZON:-75}
IMG_SIZE=${IMG_SIZE:-128}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}

FULL_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_full_mixed_v1
FULL_REPO=khush/softgym-ropeflatten-full-mixed-v1

plot_train_log() {
  local log="$1"
  local run_name="$2"
  "$PY" baselines/plot_lerobot_metrics.py "$log" \
    --run-name "$run_name" \
    --output "$PLOT_DIR/${run_name}_charts.png" \
    2>&1 | tee "$LOG_DIR/plot_${run_name}.log"
}

run_eval() {
  local run_name="$1"
  local ckpt="$2"
  echo "[$(date -Is)] Evaluating $run_name checkpoint=$ckpt"
  SOFTGYM_SOFTWARE_GL=0 "$EVAL" \
    --env-name RopeFlatten \
    --policy-checkpoint "$ckpt" \
    --policy-name "$run_name" \
    --output-dir "data/evals/$run_name" \
    --num-episodes "$EVAL_EPISODES" \
    --horizon "$ROPE_HORIZON" \
    --img-size "$IMG_SIZE" \
    --success-threshold "$SUCCESS_THRESHOLD" \
    --save-every-video 1 \
    2>&1 | tee "$LOG_DIR/eval_${run_name}.log"
}

train_act() {
  local run_name=act_rope_full_mixed_v1_25k
  local out_dir=outputs/train/act_softgym_rope_full_mixed_v1_25k
  local steps=25000
  local log="$LOG_DIR/train_${run_name}.log"

  echo "[$(date -Is)] Training ACT $run_name..."
  "$TRAIN" \
    --dataset.repo_id="$FULL_REPO" \
    --dataset.root="$FULL_ROOT" \
    --policy.type=act \
    --policy.chunk_size=25 \
    --policy.n_action_steps=25 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="$out_dir" \
    --job_name="$run_name" \
    --steps="$steps" \
    --batch_size=16 \
    --num_workers=2 \
    --save_freq="$steps" \
    --log_freq=50 \
    --wandb.enable=false \
    2>&1 | tee "$log"
  plot_train_log "$log" "$run_name"
  run_eval "$run_name" "$out_dir/checkpoints/$(printf '%06d' "$steps")/pretrained_model"
}

train_smolvla() {
  local run_name=smolvla_rope_full_mixed_v1_25k
  local out_dir=outputs/train/smolvla_softgym_rope_full_mixed_v1_25k
  local steps=25000
  local log="$LOG_DIR/train_${run_name}.log"

  echo "[$(date -Is)] Training SmolVLA $run_name..."
  "$TRAIN" \
    --dataset.repo_id="$FULL_REPO" \
    --dataset.root="$FULL_ROOT" \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="$out_dir" \
    --job_name="$run_name" \
    --steps="$steps" \
    --batch_size=2 \
    --num_workers=0 \
    --save_freq="$steps" \
    --log_freq=50 \
    --wandb.enable=false \
    2>&1 | tee "$log"
  plot_train_log "$log" "$run_name"
  run_eval "$run_name" "$out_dir/checkpoints/$(printf '%06d' "$steps")/pretrained_model"
}

if [[ ! -d "$FULL_ROOT" ]]; then
  echo "Missing LeRobot dataset root: $FULL_ROOT" >&2
  exit 1
fi

train_act
train_smolvla

echo "[$(date -Is)] Aggregating full-mixed 25k eval results..."
"$PY" baselines/plot_eval_results.py \
  --eval-root simulation/data/evals \
  --output-dir baselines/plots/new \
  --prefix eval_rope_full_mixed_25k \
  --include-policy act_rope_full_mixed_v1_25k \
  --include-policy smolvla_rope_full_mixed_v1_25k \
  2>&1 | tee "$LOG_DIR/plot_eval_rope_full_mixed_25k.log"

echo "[$(date -Is)] Full-mixed 25k remaining queue finished."
