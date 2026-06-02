#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

DATA=${DATA:-simulation/data/ropeconfiguration_route_u_1000_collectonly}
TOTAL_EPISODES=${TOTAL_EPISODES:-1000}
SHARDS=${SHARDS:-10}
MAX_PAR=${MAX_PAR:-2}
IMG=${IMG:-128}
HORIZON=${HORIZON:-300}
SCRIPT_NOISE_SCALE=${SCRIPT_NOISE_SCALE:-0.0}
MIN_FINAL_NORMALIZED_PERFORMANCE=${MIN_FINAL_NORMALIZED_PERFORMANCE:-0.60}
STOP_ON_NORMALIZED_PERFORMANCE=${STOP_ON_NORMALIZED_PERFORMANCE:-$MIN_FINAL_NORMALIZED_PERFORMANCE}
MAX_ATTEMPTS_PER_EPISODE=${MAX_ATTEMPTS_PER_EPISODE:-60}
MIN_RECORDED_HORIZON_FOR_ACCEPTANCE=${MIN_RECORDED_HORIZON_FOR_ACCEPTANCE:-20}
GOAL_CHARACTER=${GOAL_CHARACTER:-U}
MIN_VARIATIONS_PER_SHARD=${MIN_VARIATIONS_PER_SHARD:-150}

SHARD_SIZE=$(( (TOTAL_EPISODES + SHARDS - 1) / SHARDS ))
mkdir -p "$DATA/logs"

echo "=== Route-U collect-only ==="
echo "data=$DATA total=$TOTAL_EPISODES shards=$SHARDS shard_size=$SHARD_SIZE max_par=$MAX_PAR"
echo "gate=$MIN_FINAL_NORMALIZED_PERFORMANCE stop=$STOP_ON_NORMALIZED_PERFORMANCE min_horizon=$MIN_RECORDED_HORIZON_FOR_ACCEPTANCE"

collect_one() {
  local i=$1
  local start=$(( i * SHARD_SIZE ))
  local remaining=$(( TOTAL_EPISODES - start ))
  local count=$SHARD_SIZE
  if (( remaining <= 0 )); then
    return 0
  fi
  if (( remaining < SHARD_SIZE )); then
    count=$remaining
  fi

  local seed=$(( 3000000 + i * 100000 ))
  local outdir="${DATA#simulation/}/shard$i"
  local variations=$count
  if (( variations < MIN_VARIATIONS_PER_SHARD )); then
    variations=$MIN_VARIATIONS_PER_SHARD
  fi
  local log="$DATA/logs/collect_shard${i}.log"

  echo "[collect shard $i] count=$count seed=$seed variations=$variations -> $outdir"
  SOFTGYM_SOFTWARE_GL=1 CUDA_VISIBLE_DEVICES=0 bash simulation/docker/collect-trajectories.sh \
    --env-name RopeConfiguration \
    --policy route_u \
    --num-episodes "$count" \
    --horizon "$HORIZON" \
    --num-variations "$variations" \
    --seed "$seed" \
    --img-size "$IMG" \
    --script-noise-scale "$SCRIPT_NOISE_SCALE" \
    --min-final-normalized-performance "$MIN_FINAL_NORMALIZED_PERFORMANCE" \
    --stop-on-normalized-performance "$STOP_ON_NORMALIZED_PERFORMANCE" \
    --max-attempts-per-episode "$MAX_ATTEMPTS_PER_EPISODE" \
    --min-recorded-horizon-for-acceptance "$MIN_RECORDED_HORIZON_FOR_ACCEPTANCE" \
    --goal-character "$GOAL_CHARACTER" \
    --output-dir "$outdir" > "$log" 2>&1
}

running=0
failed=0
for (( i=0; i<SHARDS; i++ )); do
  collect_one "$i" &
  running=$(( running + 1 ))
  if (( running >= MAX_PAR )); then
    if ! wait -n; then failed=1; fi
    running=$(( running - 1 ))
  fi
done
if ! wait; then failed=1; fi

if (( failed != 0 )); then
  echo "ERROR: collection failed; check $DATA/logs" >&2
  exit 1
fi

echo "=== collection complete ==="
