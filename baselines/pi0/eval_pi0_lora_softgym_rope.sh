#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
OPENPI_CONFIG=${OPENPI_CONFIG:-pi05_softgym_rope_lora}
OPENPI_CHECKPOINT=${OPENPI_CHECKPOINT:-}
OPENPI_EVAL_OUT=${OPENPI_EVAL_OUT:-data/evals/pi0_rope_lora}
OPENPI_NUM_EPISODES=${OPENPI_NUM_EPISODES:-10}
OPENPI_HORIZON=${OPENPI_HORIZON:-75}
OPENPI_PORT=${OPENPI_POLICY_PORT:-8766}
HOST_FOR_CONTAINER=${OPENPI_POLICY_HOST_FOR_CONTAINER:-172.17.0.1}
OPENPI_TIMEOUT_S=${OPENPI_TIMEOUT_S:-180}
UV_BIN=${UV:-${HOME}/.local/share/uv-bootstrap/bin/uv}
LOG=${OPENPI_SERVER_LOG:-"$REPO_ROOT/baselines/logs/pi0_softgym_server.log"}

if [[ -z "$OPENPI_CHECKPOINT" ]]; then
  cat >&2 <<USAGE
Missing OPENPI_CHECKPOINT. Example:
  OPENPI_CHECKPOINT=/home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/run_name/20000 \
  baselines/pi0/eval_pi0_lora_softgym_rope.sh
USAGE
  exit 2
fi

mkdir -p "$(dirname "$LOG")"

cd "$OPENPI_ROOT"
"$UV_BIN" run python "$REPO_ROOT/baselines/pi0/openpi_softgym_server.py"   --config-name "$OPENPI_CONFIG"   --checkpoint "$OPENPI_CHECKPOINT"   --host 0.0.0.0   --port "$OPENPI_PORT"   --task "straighten the rope"   >"$LOG" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if python3 -c "from urllib import request; request.urlopen('http://127.0.0.1:${OPENPI_PORT}/health', timeout=1).read()" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

python3 -c "from urllib import request; print(request.urlopen('http://127.0.0.1:${OPENPI_PORT}/health', timeout=5).read().decode('utf-8'))"

cd "$REPO_ROOT/simulation"
./docker/softgym-local.sh run "python utils/eval_lerobot_policy.py   --env-name RopeFlatten   --policy-url http://${HOST_FOR_CONTAINER}:${OPENPI_PORT}   --output-dir ${OPENPI_EVAL_OUT}   --policy-name pi0_rope_lora   --checkpoint ${OPENPI_CHECKPOINT}   --num-episodes ${OPENPI_NUM_EPISODES}   --horizon ${OPENPI_HORIZON}   --timeout-s ${OPENPI_TIMEOUT_S}   --img-size 128   --success-threshold 0.8   --save-every-video 0"
