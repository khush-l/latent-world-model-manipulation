#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/cs231n-project

EVAL=./simulation/docker/eval-lerobot-policy.sh
LOG_DIR=/home/ubuntu/cs231n-project/baselines/logs
mkdir -p "$LOG_DIR"

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

run_eval() {
  local env_name="$1"
  local policy_name="$2"
  local run_dir="$3"
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
    --num-episodes 10 \
    --horizon 75 \
    --img-size 128 \
    --success-threshold 0.8 \
    2>&1 | tee "$LOG_DIR/eval_${policy_name}.log"
}

wait_for_session_end extend_baselines_to_10k

run_eval RopeFlatten act_rope_3k outputs/train/act_softgym_rope_scripted_vla_150
run_eval ClothFlatten act_cloth_3k outputs/train/act_softgym_cloth_scripted_vla_150
run_eval RopeFlatten act_rope_tuned_20k outputs/train/act_softgym_rope_scripted_vla_150_tuned_20k
run_eval ClothFlatten act_cloth_tuned_20k outputs/train/act_softgym_cloth_scripted_vla_150_tuned_20k
run_eval RopeFlatten smolvla_rope_final outputs/train/smolvla_softgym_rope_scripted_vla_150
run_eval ClothFlatten smolvla_cloth_final outputs/train/smolvla_softgym_cloth_scripted_vla_150

echo "[$(date -Is)] Evaluation queue finished."
