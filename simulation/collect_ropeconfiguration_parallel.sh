#!/usr/bin/env bash
# Collect RopeConfiguration goal-conditioned data in v3 NPZ/HDF5 format.
#
# Defaults produce 5000 total episodes split into shards. Run from the repo
# root in WSL:
#
#   PY="/mnt/c/Users/Ethan Lai/Desktop/cs231n-project/.venv/Scripts/python.exe" \
#   bash simulation/collect_ropeconfiguration_parallel.sh
#
# Tunables:
#   TOTAL_EPISODES  total episodes, default 5000
#   SHARDS          shard count, default 10
#   MAX_PAR         max concurrent collectors/converters, default 2
#   POLICY          route_u, configure, manipulate, or push, default configure
#   SCRIPT_NOISE_SCALE low-level action perturbation, default 0.0 for clean data
#   HORIZON         maximum episode length, default 300 for route_u, 200 for configure
#   MIN_FINAL_NORMALIZED_PERFORMANCE acceptance threshold, default 0.60 for route_u,
#                   0.60 for configure
#   STOP_ON_NORMALIZED_PERFORMANCE early-stop threshold, default same as MIN_FINAL...
#   MAX_ATTEMPTS_PER_EPISODE retries for accepted data, default 50
#   MIN_RECORDED_HORIZON_FOR_ACCEPTANCE reject accepted route_u clips shorter
#                   than this many actions, default 20 for route_u
#   GOALS           balanced goals for configure, default "S O M C U"; route_u is U-only
#   GOAL_CHARACTER  force every shard to one goal, overrides GOALS
#   MIN_VARIATIONS_PER_SHARD minimum cached configs for forced-goal shards, default 50
#   IMG             image size, default 128
#   DATA            output directory, default simulation/data/ropeconfiguration_<policy>_<total>
#   OUT_H5          merged output HDF5, default $DATA.h5
#   SMOKE=1         collect 4 episodes in 2 shards
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PY=${PY:-.venv/bin/python}
COLLECT=simulation/docker/collect-trajectories.sh

POLICY=${POLICY:-configure}
SCRIPT_NOISE_SCALE=${SCRIPT_NOISE_SCALE:-0.0}
TOTAL_EPISODES=${TOTAL_EPISODES:-5000}
SHARDS=${SHARDS:-10}
MAX_PAR=${MAX_PAR:-2}
IMG=${IMG:-128}
if [[ "$POLICY" == "route_u" ]]; then
  HORIZON=${HORIZON:-300}
elif [[ "$POLICY" == "configure" ]]; then
  HORIZON=${HORIZON:-200}
else
  HORIZON=${HORIZON:-}
fi
if [[ "$POLICY" == "route_u" ]]; then
  MIN_FINAL_NORMALIZED_PERFORMANCE=${MIN_FINAL_NORMALIZED_PERFORMANCE:-0.60}
elif [[ "$POLICY" == "configure" ]]; then
  MIN_FINAL_NORMALIZED_PERFORMANCE=${MIN_FINAL_NORMALIZED_PERFORMANCE:-0.60}
else
  MIN_FINAL_NORMALIZED_PERFORMANCE=${MIN_FINAL_NORMALIZED_PERFORMANCE:-}
fi
STOP_ON_NORMALIZED_PERFORMANCE=${STOP_ON_NORMALIZED_PERFORMANCE:-$MIN_FINAL_NORMALIZED_PERFORMANCE}
MAX_ATTEMPTS_PER_EPISODE=${MAX_ATTEMPTS_PER_EPISODE:-50}
if [[ "$POLICY" == "route_u" ]]; then
  MIN_RECORDED_HORIZON_FOR_ACCEPTANCE=${MIN_RECORDED_HORIZON_FOR_ACCEPTANCE:-20}
else
  MIN_RECORDED_HORIZON_FOR_ACCEPTANCE=${MIN_RECORDED_HORIZON_FOR_ACCEPTANCE:-0}
fi
GOALS=${GOALS:-"S O M C U"}
if [[ "$POLICY" == "route_u" ]]; then
  GOAL_CHARACTER=${GOAL_CHARACTER:-U}
else
  GOAL_CHARACTER=${GOAL_CHARACTER:-}
fi
if [[ "$POLICY" == "route_u" ]]; then
  MIN_VARIATIONS_PER_SHARD=${MIN_VARIATIONS_PER_SHARD:-150}
else
  MIN_VARIATIONS_PER_SHARD=${MIN_VARIATIONS_PER_SHARD:-50}
fi
DATA=${DATA:-simulation/data/ropeconfiguration_${POLICY}_${TOTAL_EPISODES}}
OUT_H5=${OUT_H5:-${DATA}.h5}
LOGS=$DATA/logs

if [[ "${SMOKE:-0}" == "1" ]]; then
  TOTAL_EPISODES=4
  SHARDS=2
  MAX_PAR=2
fi

SHARD_SIZE=$(( (TOTAL_EPISODES + SHARDS - 1) / SHARDS ))
mkdir -p "$LOGS" "$DATA" "$(dirname "$OUT_H5")"

echo "=== RopeConfiguration collection ==="
echo "policy=$POLICY total=$TOTAL_EPISODES shards=$SHARDS shard_size=$SHARD_SIZE max_par=$MAX_PAR img=$IMG horizon=${HORIZON:-env-default} script_noise_scale=$SCRIPT_NOISE_SCALE"
echo "min_final_normalized_performance=${MIN_FINAL_NORMALIZED_PERFORMANCE:-none} max_attempts=$MAX_ATTEMPTS_PER_EPISODE"
echo "min_recorded_horizon_for_acceptance=$MIN_RECORDED_HORIZON_FOR_ACCEPTANCE"
echo "stop_on_normalized_performance=${STOP_ON_NORMALIZED_PERFORMANCE:-none}"
echo "goals=${GOAL_CHARACTER:-$GOALS}"
echo "data=$DATA"
echo "out=$OUT_H5"

collect_shard() {
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
  local log="$LOGS/collect_shard${i}.log"
  local variations=$count
  local accept_args=()
  local horizon_args=()
  local goal_args=()
  if [[ -n "$HORIZON" ]]; then
    horizon_args+=(--horizon "$HORIZON")
  fi
  if [[ -n "$MIN_FINAL_NORMALIZED_PERFORMANCE" ]]; then
    accept_args+=(--min-final-normalized-performance "$MIN_FINAL_NORMALIZED_PERFORMANCE")
    accept_args+=(--max-attempts-per-episode "$MAX_ATTEMPTS_PER_EPISODE")
    accept_args+=(--min-recorded-horizon-for-acceptance "$MIN_RECORDED_HORIZON_FOR_ACCEPTANCE")
  fi
  if [[ -n "$STOP_ON_NORMALIZED_PERFORMANCE" ]]; then
    accept_args+=(--stop-on-normalized-performance "$STOP_ON_NORMALIZED_PERFORMANCE")
  fi
  if [[ -n "$GOAL_CHARACTER" ]]; then
    goal_args+=(--goal-character "$GOAL_CHARACTER")
  elif [[ "$POLICY" == "configure" && -n "$GOALS" ]]; then
    read -r -a goal_list <<< "$GOALS"
    if (( ${#goal_list[@]} > 0 )); then
      goal_args+=(--goal-character "${goal_list[$(( i % ${#goal_list[@]} ))]}")
    fi
  fi
  if (( ${#goal_args[@]} > 0 && variations < MIN_VARIATIONS_PER_SHARD )); then
    variations=$MIN_VARIATIONS_PER_SHARD
  fi
  echo "[collect shard $i] count=$count seed=$seed goal=${goal_args[*]:-random} -> $outdir"
  {
    SOFTGYM_SOFTWARE_GL=1 CUDA_VISIBLE_DEVICES=0 bash "$COLLECT" \
      --env-name RopeConfiguration \
      --policy "$POLICY" \
      --num-episodes "$count" \
      "${horizon_args[@]}" \
      --num-variations "$variations" \
      --seed "$seed" \
      --img-size "$IMG" \
      --script-noise-scale "$SCRIPT_NOISE_SCALE" \
      "${accept_args[@]}" \
      "${goal_args[@]}" \
      --output-dir "$outdir"
  } > "$log" 2>&1
}

running=0
failed=0
for (( i=0; i<SHARDS; i++ )); do
  collect_shard "$i" &
  running=$(( running + 1 ))
  if (( running >= MAX_PAR )); then
    if ! wait -n; then failed=1; fi
    running=$(( running - 1 ))
  fi
done
if ! wait; then failed=1; fi
if (( failed != 0 )); then
  echo "ERROR: at least one collection shard failed; check $LOGS" >&2
  exit 1
fi
echo "=== collection done; converting shards ==="

running=0
failed=0
shard_h5s=()
for (( i=0; i<SHARDS; i++ )); do
  sdir="$DATA/shard$i"
  sh5="$DATA/shard$i.h5"
  if [[ ! -d "$sdir" ]]; then
    continue
  fi
  shard_h5s+=("$sh5")
  {
    "$PY" simulation/utils/npz_to_hdf5.py --input-dir "$sdir" --output "$sh5"
  } > "$LOGS/convert_shard${i}.log" 2>&1 &
  running=$(( running + 1 ))
  if (( running >= MAX_PAR )); then
    if ! wait -n; then failed=1; fi
    running=$(( running - 1 ))
  fi
done
if ! wait; then failed=1; fi
if (( failed != 0 )); then
  echo "ERROR: at least one shard conversion failed; check $LOGS" >&2
  exit 1
fi
echo "=== shard conversion done; merging ==="

inputs=()
for h5 in "${shard_h5s[@]}"; do
  [[ -f "$h5" ]] && inputs+=(--input "$h5:$POLICY")
done
if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "ERROR: no shard HDF5 files produced; check $LOGS" >&2
  exit 1
fi
expected_shards=0
for (( i=0; i<SHARDS; i++ )); do
  if (( i * SHARD_SIZE < TOTAL_EPISODES )); then
    expected_shards=$(( expected_shards + 1 ))
  fi
done
actual_shards=$(( ${#inputs[@]} / 2 ))
if [[ $actual_shards -ne $expected_shards ]]; then
  echo "ERROR: expected $expected_shards shard HDF5 files, found $actual_shards; check $LOGS" >&2
  exit 1
fi

if ! "$PY" simulation/utils/merge_h5.py --output "$OUT_H5" "${inputs[@]}"; then
  echo "ERROR: merge failed; check $LOGS" >&2
  exit 1
fi

echo "=== DONE ==="
"$PY" - "$OUT_H5" <<'PYEOF'
import sys, h5py, numpy as np
f = h5py.File(sys.argv[1], "r")
print("rows:", f["pixels"].shape, "episodes:", len(set(f["episode_idx"][:].tolist())))
print("policy_id_map:", f.attrs.get("policy_id_map"))
px = f["pixels"]
print("pixel min/max/mean/std:", int(px[:].min()), int(px[:].max()), float(px[:].mean()), float(px[:].std()))
perf = f["info_normalized_performance"][:] if "info_normalized_performance" in f else np.array([])
fin = np.isfinite(perf)
if fin.any():
    print("normalized performance mean/final-ish:", float(perf[fin].mean()), float(perf[fin][-1]))
PYEOF
