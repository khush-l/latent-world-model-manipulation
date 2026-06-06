#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

TRAIN_SESSION=${TRAIN_SESSION:-pi0_cem_final_lora_40k_b4_ckpt5k}
WATCHER_SESSION=${WATCHER_SESSION:-pi0_cem_final_fallback_watch}
OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
OPENPI_CONFIG=${OPENPI_CONFIG:-pi05_softgym_rope_lora}
PI0_EXP_NAME=${PI0_EXP_NAME:-softgym_rope_cem_final_lora_40k_b4_ckpt5k}
PI0_CHECKPOINT_ROOT=${PI0_CHECKPOINT_ROOT:-"$OPENPI_ROOT/checkpoints/$OPENPI_CONFIG/$PI0_EXP_NAME"}

WAIT_STEP=${WAIT_STEP:-20000}
PI0_STEPS=${PI0_STEPS:-"10000 15000 20000"}
EVAL_EPISODES=${EVAL_EPISODES:-50}
HORIZON=${HORIZON:-75}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}
IMG_SIZE=${IMG_SIZE:-128}
LOG_DIR=${LOG_DIR:-"$REPO_ROOT/baselines/logs"}

ACT_CHECKPOINT=${ACT_CHECKPOINT:-"$REPO_ROOT/outputs/train/act_rope_cem_final_25k/checkpoints/015000/pretrained_model"}
SMOLVLA_CHECKPOINT=${SMOLVLA_CHECKPOINT:-"$REPO_ROOT/outputs/train/smolvla_rope_cem_final_stronger/checkpoints/045000/pretrained_model"}
DIRECT_EVAL=${DIRECT_EVAL:-"$REPO_ROOT/simulation/docker/eval-lerobot-policy.sh"}

mkdir -p "$LOG_DIR"

checkpoint_ready() {
  local step_dir="$PI0_CHECKPOINT_ROOT/$1"
  [[ -d "$step_dir/assets" && -d "$step_dir/params" && -d "$step_dir/train_state" ]]
}

wait_for_checkpoint() {
  local step=$1
  echo "[$(date -Is)] Waiting for pi0 checkpoint $step under $PI0_CHECKPOINT_ROOT"
  while ! checkpoint_ready "$step"; do
    sleep 60
  done
  echo "[$(date -Is)] pi0 checkpoint $step is ready"
}

stop_training_for_eval() {
  if tmux has-session -t "=$TRAIN_SESSION" 2>/dev/null; then
    echo "[$(date -Is)] Stopping training session $TRAIN_SESSION for GPU eval"
    tmux send-keys -t "$TRAIN_SESSION" C-c || true
    for _ in $(seq 1 60); do
      if ! tmux has-session -t "=$TRAIN_SESSION" 2>/dev/null; then
        break
      fi
      sleep 5
    done
  fi

  if tmux has-session -t "=$WATCHER_SESSION" 2>/dev/null; then
    echo "[$(date -Is)] Stopping fallback watcher $WATCHER_SESSION during intentional eval pause"
    tmux kill-session -t "$WATCHER_SESSION" || true
  fi

  echo "[$(date -Is)] Waiting for GPU memory to clear"
  for _ in $(seq 1 60); do
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
      break
    fi
    sleep 5
  done
  nvidia-smi || true
}

run_pi0_eval() {
  local step=$1
  local checkpoint="$PI0_CHECKPOINT_ROOT/$step"
  local out="data/evals/pi0_cem_final_ckpt${step}_episodes${EVAL_EPISODES}_gpu"
  local server_log="$LOG_DIR/pi0_eval_ckpt${step}_gpu_server.log"

  echo "[$(date -Is)] Evaluating pi0 checkpoint $step on GPU"
  cd "$REPO_ROOT"
  OPENPI_ROOT="$OPENPI_ROOT" \
  OPENPI_CONFIG="$OPENPI_CONFIG" \
  OPENPI_CHECKPOINT="$checkpoint" \
  OPENPI_POLICY_PORT=8865 \
  OPENPI_EVAL_OUT="$out" \
  OPENPI_NUM_EPISODES="$EVAL_EPISODES" \
  OPENPI_HORIZON="$HORIZON" \
  OPENPI_SERVER_LOG="$server_log" \
  baselines/pi0/eval_pi0_lora_softgym_rope.sh
}

run_lerobot_eval() {
  local name=$1
  local checkpoint=$2
  local port=$3
  local out="data/evals/rope_cem_final_gpu_after_pi0_20k/$name"
  local server_log="$LOG_DIR/${name}_gpu_policy_server.log"

  if [[ ! -d "$checkpoint" ]]; then
    echo "[$(date -Is)] Missing checkpoint for $name: $checkpoint" >&2
    exit 1
  fi

  echo "[$(date -Is)] Evaluating $name on GPU"
  cd "$REPO_ROOT"
  LEROBOT_POLICY_DEVICE=cuda \
  LEROBOT_POLICY_PORT="$port" \
  LEROBOT_POLICY_SERVER_LOG="$server_log" \
  "$DIRECT_EVAL" \
    --env-name RopeFlatten \
    --policy-checkpoint "$checkpoint" \
    --policy-name "$name" \
    --output-dir "$out" \
    --num-episodes "$EVAL_EPISODES" \
    --horizon "$HORIZON" \
    --img-size "$IMG_SIZE" \
    --success-threshold "$SUCCESS_THRESHOLD" \
    --save-every-video 0
}

echo "[$(date -Is)] GPU eval queue started"
echo "Will pause training after checkpoint $WAIT_STEP, then evaluate pi0 steps: $PI0_STEPS"
echo "Then evaluate ACT and SmolVLA with $EVAL_EPISODES episodes each"

wait_for_checkpoint "$WAIT_STEP"
stop_training_for_eval

for step in $PI0_STEPS; do
  wait_for_checkpoint "$step"
  run_pi0_eval "$step"
done

run_lerobot_eval act_rope_cem_final_015000_50eps_gpu "$ACT_CHECKPOINT" 8875
run_lerobot_eval smolvla_rope_cem_final_045000_50eps_gpu "$SMOLVLA_CHECKPOINT" 8875

echo "[$(date -Is)] GPU eval queue complete"
find "$REPO_ROOT/simulation/data/evals" -path '*episodes50_gpu/summary.json' -o -path '*rope_cem_final_gpu_after_pi0_20k*/summary.json' -print
