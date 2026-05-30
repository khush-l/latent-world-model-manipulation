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

convert_dataset() {
  local h5="$1"
  local root="$2"
  local repo="$3"
  local name="$4"

  if [[ ! -d "$root" ]]; then
    echo "[$(date -Is)] Converting $name to LeRobot..."
    "$PY" baselines/softgym_hdf5_to_lerobot.py \
      --input "$h5" \
      --output-root "$root" \
      --repo-id "$repo" \
      --instruction "straighten the rope" \
      2>&1 | tee "$LOG_DIR/convert_${name}_lerobot.log"
  else
    echo "[$(date -Is)] LeRobot dataset already exists for $name at $root"
  fi
}

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

aggregate_selected_evals() {
  local prefix="$1"
  shift
  local args=(
    --eval-root simulation/data/evals
    --output-dir baselines/plots/new
    --prefix "$prefix"
  )
  for policy in "$@"; do
    args+=(--include-policy "$policy")
  done
  echo "[$(date -Is)] Aggregating eval results prefix=$prefix policies=$@"
  "$PY" baselines/plot_eval_results.py "${args[@]}" \
    2>&1 | tee "$LOG_DIR/plot_${prefix}.log"
}

train_act() {
  local repo="$1"
  local root="$2"
  local run_name="$3"
  local out_dir="$4"
  local steps="$5"
  local log="$LOG_DIR/train_${run_name}.log"

  echo "[$(date -Is)] Training ACT $run_name..."
  "$TRAIN" \
    --dataset.repo_id="$repo" \
    --dataset.root="$root" \
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
  local repo="$1"
  local root="$2"
  local run_name="$3"
  local out_dir="$4"
  local steps="$5"
  local log="$LOG_DIR/train_${run_name}.log"

  echo "[$(date -Is)] Training SmolVLA $run_name..."
  "$TRAIN" \
    --dataset.repo_id="$repo" \
    --dataset.root="$root" \
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

FULL_H5=/home/ubuntu/cs231n-project/training/data/rope_full_mixed_v1.h5
FULL_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_full_mixed_v1
FULL_REPO=khush/softgym-ropeflatten-full-mixed-v1

GEOM_H5=/home/ubuntu/cs231n-project/training/data/ropeflatten_geometric_5k_v3.h5
GEOM_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_geometric_5k_v3
GEOM_REPO=khush/softgym-ropeflatten-geometric-5k-v3

convert_dataset "$GEOM_H5" "$GEOM_ROOT" "$GEOM_REPO" "rope_geometric_5k_v3"
convert_dataset "$FULL_H5" "$FULL_ROOT" "$FULL_REPO" "rope_full_mixed_v1"

train_act "$GEOM_REPO" "$GEOM_ROOT" "act_rope_geometric_5k_v3_20k" \
  "outputs/train/act_softgym_rope_geometric_5k_v3_20k" 20000
train_smolvla "$GEOM_REPO" "$GEOM_ROOT" "smolvla_rope_geometric_5k_v3_20k" \
  "outputs/train/smolvla_softgym_rope_geometric_5k_v3_20k" 20000

aggregate_selected_evals "eval_rope_geometric_5k_v3" \
  "act_rope_geometric_5k_v3_20k" \
  "smolvla_rope_geometric_5k_v3_20k"

train_act "$FULL_REPO" "$FULL_ROOT" "act_rope_full_mixed_v1_25k" \
  "outputs/train/act_softgym_rope_full_mixed_v1_25k" 25000
train_smolvla "$FULL_REPO" "$FULL_ROOT" "smolvla_rope_full_mixed_v1_25k" \
  "outputs/train/smolvla_softgym_rope_full_mixed_v1_25k" 25000

echo "[$(date -Is)] Aggregating eval results..."
"$PY" baselines/plot_eval_results.py \
  --eval-root simulation/data/evals \
  --output-dir baselines/plots/new \
  --prefix eval_rope_full_and_geometric \
  2>&1 | tee "$LOG_DIR/plot_eval_rope_full_and_geometric.log"

echo "[$(date -Is)] Rope full/geometric baseline queue finished."
