#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PY="$REPO_ROOT/training/.venv/bin/python"
SERVER="$REPO_ROOT/baselines/policy_server.py"
PORT=${LEROBOT_POLICY_PORT:-8765}
HOST_FOR_CONTAINER=${LEROBOT_POLICY_HOST_FOR_CONTAINER:-172.17.0.1}
SERVER_LOG=${LEROBOT_POLICY_SERVER_LOG:-"$REPO_ROOT/baselines/logs/policy_server_eval.log"}
DEVICE=${LEROBOT_POLICY_DEVICE:-cuda}

usage() {
  cat <<'USAGE'
Usage:
  simulation/docker/eval-lerobot-policy.sh --env-name RopeFlatten --policy-checkpoint CKPT --output-dir OUT [eval args...]

Required:
  --env-name              SoftGym env, e.g. RopeFlatten or ClothFlatten
  --policy-checkpoint     LeRobot pretrained_model checkpoint directory
  --output-dir            Output directory for scores.csv, summary.json, videos/

Common:
  --policy-name           Label in score files
  --num-episodes          Default 10
  --horizon               Default 75
  --img-size              Default 128
  --success-threshold     Default 0.8

Environment:
  LEROBOT_POLICY_PORT                 Host policy server port (default 8765)
  LEROBOT_POLICY_HOST_FOR_CONTAINER   Host address visible from Docker (default 172.17.0.1)
  LEROBOT_POLICY_DEVICE               Policy server device (default cuda)
USAGE
}

quote_command() {
  local quoted=()
  local arg
  for arg in "$@"; do
    quoted+=("$(printf "%q" "$arg")")
  done
  printf "%s " "${quoted[@]}"
}

env_name=""
checkpoint=""
output_dir=""
policy_name="lerobot_policy"
pass_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      env_name="$2"; pass_args+=("$1" "$2"); shift 2 ;;
    --policy-checkpoint)
      checkpoint="$2"; shift 2 ;;
    --output-dir)
      output_dir="$2"; pass_args+=("$1" "$2"); shift 2 ;;
    --policy-name)
      policy_name="$2"; pass_args+=("$1" "$2"); shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      pass_args+=("$1"); shift ;;
  esac
done

if [[ -z "$env_name" || -z "$checkpoint" || -z "$output_dir" ]]; then
  usage >&2
  exit 2
fi

case "$env_name" in
  RopeFlatten) task="straighten the rope" ;;
  ClothFlatten) task="flatten the cloth" ;;
  *) task="$env_name" ;;
esac

mkdir -p "$(dirname "$SERVER_LOG")"

"$PY" "$SERVER" \
  --checkpoint "$checkpoint" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --task "$task" \
  --device "$DEVICE" \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Started policy server pid=$server_pid log=$SERVER_LOG"
for _ in $(seq 1 120); do
  if "$PY" - <<PY >/dev/null 2>&1
import json
from urllib import request
request.urlopen("http://127.0.0.1:${PORT}/health", timeout=1).read()
PY
  then
    break
  fi
  sleep 1
done

"$PY" - <<PY
from urllib import request
print(request.urlopen("http://127.0.0.1:${PORT}/health", timeout=5).read().decode("utf-8"))
PY

cmd=(
  python
  utils/eval_lerobot_policy.py
  "${pass_args[@]}"
  --policy-url "http://${HOST_FOR_CONTAINER}:${PORT}"
  --checkpoint "$checkpoint"
)

cd "$REPO_ROOT/simulation"
./docker/softgym-local.sh run "$(quote_command "${cmd[@]}")"
