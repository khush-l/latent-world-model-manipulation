#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_NAME=${ENV_NAME:-RopeFlatten}
NUM_EPISODES=${NUM_EPISODES:-10}
HORIZON=${HORIZON:-75}
IMG_SIZE=${IMG_SIZE:-128}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}
OUT_ROOT=${OUT_ROOT:-data/evals/h100_direct_action}
DEVICE=${LEROBOT_POLICY_DEVICE:-cuda}

ACT_CHECKPOINT=${ACT_CHECKPOINT:-}
SMOLVLA_CHECKPOINT=${SMOLVLA_CHECKPOINT:-}
PI0_CHECKPOINT=${PI0_CHECKPOINT:-}
PI0_CONFIG=${PI0_CONFIG:-pi05_softgym_rope_lora}
PI0_TIMEOUT_S=${OPENPI_TIMEOUT_S:-180}

usage() {
  cat <<'USAGE'
Run fair same-hardware direct-action SoftGym evals.

Required environment variables:
  ACT_CHECKPOINT=/path/to/act/pretrained_model
  SMOLVLA_CHECKPOINT=/path/to/smolvla/pretrained_model
  PI0_CHECKPOINT=/path/to/openpi/checkpoint_step_dir

Optional:
  OUT_ROOT=data/evals/h100_direct_action
  NUM_EPISODES=10
  HORIZON=75
  IMG_SIZE=128
  SUCCESS_THRESHOLD=0.8
  LEROBOT_POLICY_DEVICE=cuda
  PI0_CONFIG=pi05_softgym_rope_lora
  OPENPI_TIMEOUT_S=180
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

missing=0
for var in ACT_CHECKPOINT SMOLVLA_CHECKPOINT PI0_CHECKPOINT; do
  if [[ -z "${!var}" ]]; then
    echo "Missing $var" >&2
    missing=1
  fi
done
if [[ "$missing" == "1" ]]; then
  usage >&2
  exit 2
fi

run_lerobot_eval() {
  local name=$1
  local checkpoint=$2
  local port=$3
  echo
  echo "== Evaluating $name =="
  LEROBOT_POLICY_DEVICE="$DEVICE" \
  LEROBOT_POLICY_PORT="$port" \
  "$REPO_ROOT/simulation/docker/eval-lerobot-policy.sh" \
    --env-name "$ENV_NAME" \
    --policy-checkpoint "$checkpoint" \
    --output-dir "$OUT_ROOT/$name" \
    --policy-name "$name" \
    --num-episodes "$NUM_EPISODES" \
    --horizon "$HORIZON" \
    --img-size "$IMG_SIZE" \
    --success-threshold "$SUCCESS_THRESHOLD" \
    --save-every-video 1
}

cd "$REPO_ROOT"
mkdir -p "simulation/$OUT_ROOT" baselines/logs

run_lerobot_eval act_rope_h100 "$ACT_CHECKPOINT" 8765
run_lerobot_eval smolvla_rope_h100 "$SMOLVLA_CHECKPOINT" 8765

echo
echo "== Evaluating pi0_rope_lora_h100 =="
OPENPI_CONFIG="$PI0_CONFIG" \
OPENPI_CHECKPOINT="$PI0_CHECKPOINT" \
OPENPI_EVAL_OUT="$OUT_ROOT/pi0_rope_lora_h100" \
OPENPI_NUM_EPISODES="$NUM_EPISODES" \
OPENPI_HORIZON="$HORIZON" \
OPENPI_POLICY_PORT=8766 \
OPENPI_TIMEOUT_S="$PI0_TIMEOUT_S" \
"$REPO_ROOT/baselines/pi0/eval_pi0_lora_softgym_rope.sh"

echo
echo "Done. Outputs under: $REPO_ROOT/simulation/$OUT_ROOT"
find "$REPO_ROOT/simulation/$OUT_ROOT" -maxdepth 2 -name summary.json -print
