#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/cs231n-project

EVAL=./simulation/docker/eval-lerobot-policy.sh
PY=./training/.venv/bin/python
LOG_DIR=/home/ubuntu/cs231n-project/baselines/logs
PLOT_DIR=/home/ubuntu/cs231n-project/baselines/plots
mkdir -p "$LOG_DIR" "$PLOT_DIR"

EVAL_EPISODES=${EVAL_EPISODES:-10}
ROPE_HORIZON=${ROPE_HORIZON:-75}
CLOTH_HORIZON=${CLOTH_HORIZON:-150}
IMG_SIZE=${IMG_SIZE:-128}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}

wait_for_session_end() {
  local session="$1"
  echo "[$(date -Is)] Waiting for tmux session $session to finish before eval..."
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 120
  done
}

latest_ckpt() {
  local run_dir="$1"
  local ckpt
  ckpt=$(find "$run_dir/checkpoints" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sort | tail -1 || true)
  if [[ -z "$ckpt" ]]; then
    return 1
  fi
  echo "$ckpt/pretrained_model"
}

plot_train_log() {
  local log="$1"
  local run_name="$2"
  if [[ -f "$log" ]]; then
    "$PY" baselines/plot_lerobot_metrics.py "$log" \
      --run-name "$run_name" \
      --output "$PLOT_DIR/${run_name}_charts.png" \
      2>&1 | tee "$LOG_DIR/plot_${run_name}.log"
  else
    echo "[$(date -Is)] SKIP train plot $run_name: missing $log"
  fi
}

run_eval() {
  local env_name="$1"
  local policy_name="$2"
  local run_dir="$3"
  local horizon="$4"
  local ckpt
  ckpt=$(latest_ckpt "$run_dir") || {
    echo "[$(date -Is)] SKIP $policy_name: no checkpoint under $run_dir"
    return 0
  }
  echo "[$(date -Is)] Evaluating $policy_name checkpoint=$ckpt"
  SOFTGYM_SOFTWARE_GL=0 "$EVAL" \
    --env-name "$env_name" \
    --policy-checkpoint "$ckpt" \
    --policy-name "$policy_name" \
    --output-dir "data/evals/$policy_name" \
    --num-episodes "$EVAL_EPISODES" \
    --horizon "$horizon" \
    --img-size "$IMG_SIZE" \
    --success-threshold "$SUCCESS_THRESHOLD" \
    --save-every-video 1 \
    2>&1 | tee "$LOG_DIR/eval_${policy_name}.log"
}

wait_for_session_end train_success_filtered_baselines

echo "[$(date -Is)] Plotting training metrics..."
plot_train_log "$LOG_DIR/train_act_rope_random_5k75_5k.log" "act_rope_random_5k75_5k"
plot_train_log "$LOG_DIR/train_act_cloth_random_5k75_5k.log" "act_cloth_random_5k75_5k"
plot_train_log "$LOG_DIR/train_act_rope_success_filtered_v2_8k.log" "act_rope_success_filtered_v2_8k"
plot_train_log "$LOG_DIR/train_act_cloth_success_filtered_v2_8k.log" "act_cloth_success_filtered_v2_8k"
plot_train_log "$LOG_DIR/train_smolvla_rope_success_filtered_v2_20k.log" "smolvla_rope_success_filtered_v2_20k"
plot_train_log "$LOG_DIR/train_smolvla_cloth_success_filtered_v2_20k.log" "smolvla_cloth_success_filtered_v2_20k"

echo "[$(date -Is)] Running SoftGym evals..."
run_eval RopeFlatten act_rope_random_5k75_5k outputs/train/act_softgym_rope_random_5k75_5k "$ROPE_HORIZON"
run_eval ClothFlatten act_cloth_random_5k75_5k outputs/train/act_softgym_cloth_random_5k75_5k "$CLOTH_HORIZON"
run_eval RopeFlatten act_rope_success_filtered_v2_8k outputs/train/act_softgym_rope_success_filtered_v2_8k "$ROPE_HORIZON"
run_eval ClothFlatten act_cloth_success_filtered_v2_8k outputs/train/act_softgym_cloth_success_filtered_v2_8k "$CLOTH_HORIZON"
run_eval RopeFlatten smolvla_rope_success_filtered_v2_20k outputs/train/smolvla_softgym_rope_success_filtered_v2_20k "$ROPE_HORIZON"
run_eval ClothFlatten smolvla_cloth_success_filtered_v2_20k outputs/train/smolvla_softgym_cloth_success_filtered_v2_20k "$CLOTH_HORIZON"

echo "[$(date -Is)] Aggregating eval results..."
"$PY" baselines/plot_eval_results.py \
  --eval-root simulation/data/evals \
  --output-dir baselines/plots \
  --prefix eval_success_filtered \
  2>&1 | tee "$LOG_DIR/plot_eval_success_filtered.log"

echo "[$(date -Is)] Success-filtered evaluation queue finished."
