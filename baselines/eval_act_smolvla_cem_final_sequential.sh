#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVAL=${EVAL:-"$REPO_ROOT/simulation/docker/eval-lerobot-policy.sh"}

WAIT_SESSION=${WAIT_SESSION:-pi0_eval_ckpts_seq_50eps}
EVAL_EPISODES=${EVAL_EPISODES:-50}
HORIZON=${HORIZON:-75}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}
IMG_SIZE=${IMG_SIZE:-128}
DEVICE=${LEROBOT_POLICY_DEVICE:-cpu}
OUT_ROOT=${OUT_ROOT:-data/evals/rope_cem_final_act_smolvla_50eps}
LOG_DIR=${LOG_DIR:-"$REPO_ROOT/baselines/logs"}

ACT_CHECKPOINT=${ACT_CHECKPOINT:-"$REPO_ROOT/outputs/train/act_rope_cem_final_25k/checkpoints/015000/pretrained_model"}
SMOLVLA_CHECKPOINT=${SMOLVLA_CHECKPOINT:-"$REPO_ROOT/outputs/train/smolvla_rope_cem_final_stronger/checkpoints/045000/pretrained_model"}

mkdir -p "$LOG_DIR" "$REPO_ROOT/simulation/$OUT_ROOT"

echo "[$(date -Is)] ACT/SmolVLA sequential eval queued"
echo "Waiting for tmux session: $WAIT_SESSION"
echo "Episodes: $EVAL_EPISODES"
echo "Horizon: $HORIZON"
echo "Device: $DEVICE"
echo "Output root: $REPO_ROOT/simulation/$OUT_ROOT"

while tmux has-session -t "=$WAIT_SESSION" 2>/dev/null; do
  sleep 60
done

run_eval() {
  local name=$1
  local checkpoint=$2
  local port=$3
  local server_log="$LOG_DIR/${name}_policy_server.log"

  if [[ ! -d "$checkpoint" ]]; then
    echo "[$(date -Is)] Missing checkpoint for $name: $checkpoint" >&2
    exit 1
  fi

  echo "[$(date -Is)] Evaluating $name"
  echo "Checkpoint: $checkpoint"
  echo "Server log: $server_log"

  cd "$REPO_ROOT"
  LEROBOT_POLICY_DEVICE="$DEVICE" \
  LEROBOT_POLICY_PORT="$port" \
  LEROBOT_POLICY_SERVER_LOG="$server_log" \
  "$EVAL" \
    --env-name RopeFlatten \
    --policy-checkpoint "$checkpoint" \
    --policy-name "$name" \
    --output-dir "$OUT_ROOT/$name" \
    --num-episodes "$EVAL_EPISODES" \
    --horizon "$HORIZON" \
    --img-size "$IMG_SIZE" \
    --success-threshold "$SUCCESS_THRESHOLD" \
    --save-every-video 0

  echo "[$(date -Is)] Finished $name"
}

run_eval act_rope_cem_final_015000_50eps "$ACT_CHECKPOINT" 8875
run_eval smolvla_rope_cem_final_045000_50eps "$SMOLVLA_CHECKPOINT" 8875

echo "[$(date -Is)] ACT/SmolVLA sequential eval complete"
find "$REPO_ROOT/simulation/$OUT_ROOT" -maxdepth 2 -name summary.json -print
