#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
OPENPI_CONFIG=${OPENPI_CONFIG:-pi05_droid}
OPENPI_CHECKPOINT=${OPENPI_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_droid}
PORT=${OPENPI_POLICY_PORT:-8766}
HOST_FOR_CONTAINER=${OPENPI_POLICY_HOST_FOR_CONTAINER:-172.17.0.1}
OUT=${OPENPI_EVAL_OUT:-data/evals/pi0_rope_smoke}
LOG=${OPENPI_SERVER_LOG:-"$REPO_ROOT/baselines/logs/pi0_softgym_server.log"}

mkdir -p "$(dirname "$LOG")"

cd "$OPENPI_ROOT"
uv run python "$REPO_ROOT/baselines/pi0/openpi_softgym_server.py" \
  --config-name "$OPENPI_CONFIG" \
  --checkpoint "$OPENPI_CHECKPOINT" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --task "straighten the rope" \
  >"$LOG" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Started OpenPI server pid=$server_pid log=$LOG"
for _ in $(seq 1 240); do
  if python3 - <<PY >/dev/null 2>&1
from urllib import request
request.urlopen("http://127.0.0.1:${PORT}/health", timeout=1).read()
PY
  then
    break
  fi
  sleep 1
done

python3 - <<PY
from urllib import request
print(request.urlopen("http://127.0.0.1:${PORT}/health", timeout=5).read().decode("utf-8"))
PY

cd "$REPO_ROOT/simulation"
./docker/softgym-local.sh run "python utils/eval_lerobot_policy.py \
  --env-name RopeFlatten \
  --policy-url http://${HOST_FOR_CONTAINER}:${PORT} \
  --output-dir ${OUT} \
  --policy-name pi0_rope_smoke \
  --checkpoint ${OPENPI_CHECKPOINT} \
  --num-episodes ${OPENPI_NUM_EPISODES:-3} \
  --horizon ${OPENPI_HORIZON:-75} \
  --img-size 128 \
  --success-threshold 0.8 \
  --save-every-video 1"
