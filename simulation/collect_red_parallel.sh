#!/usr/bin/env bash
# Re-collect the full RopeFlatten dataset with RED pickers, sharded + parallel.
#
# Reproduces the original 15k-episode recipe (5k each of random / geometric /
# manipulate, merged) but with SOFTGYM_PICKER_COLOR set so the pickers render
# red, and with each policy split into shards collected concurrently (separate
# Docker containers sharing the GPU) to cut wall-clock.
#
# Pipeline per shard:  collect (Docker/pyflex) -> npz dir
# Then (host, no GPU):  npz_to_hdf5 per shard -> merge shards per policy
#                       -> final merge of the 3 policies -> ONE dataset h5.
#
# Distinct --seed per shard => distinct rope variations (seed_everything +
# RandomState(seed+episode_idx) in collect_trajectories.py), so shards never
# duplicate data.
#
# Tunables (env vars):
#   PER_POLICY  episodes per policy           (default 5000)
#   SHARDS      shards per policy             (default 5  -> 1000 eps/shard)
#   MAX_PAR     max concurrent collect jobs   (default 4  -> GPU-bound)
#   IMG         image size                    (default 128)
#   COLOR       picker color "r,g,b"          (default 1,0,0 = red)
#   OUT_H5      final merged dataset path     (default simulation/data/rope/rope_full_dataset_red.h5)
#   SMOKE=1     tiny run (30 eps/policy) to validate the orchestration
#
# Usage:
#   bash simulation/collect_red_parallel.sh                 # full 15k
#   SMOKE=1 bash simulation/collect_red_parallel.sh         # quick test
#   PER_POLICY=2000 SHARDS=4 MAX_PAR=6 bash simulation/collect_red_parallel.sh
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PY=.venv/bin/python
COLLECT=simulation/docker/collect-trajectories.sh
PER_POLICY=${PER_POLICY:-5000}
SHARDS=${SHARDS:-5}
MAX_PAR=${MAX_PAR:-4}
IMG=${IMG:-128}
COLOR=${COLOR:-"1,0,0"}
[ "$COLOR" = "gray" ] && COLOR=""   # COLOR=gray -> no recolor (normal gray pickers)
OUT_H5=${OUT_H5:-simulation/data/rope/rope_full_dataset_red.h5}
if [[ "${SMOKE:-0}" == "1" ]]; then PER_POLICY=30; SHARDS=2; MAX_PAR=3; fi

POLICIES=(${COLLECT_POLICIES:-random geometric manipulate})  # override e.g. COLLECT_POLICIES="geometric manipulate"
SHARD_SIZE=$(( PER_POLICY / SHARDS ))
DATA=simulation/data/rope_red          # host path (== data/rope_red in container)
LOGS=$DATA/logs
mkdir -p "$LOGS"
export SOFTGYM_PICKER_COLOR="$COLOR"    # forwarded into the container by softgym-local.sh

echo "=== RED-picker collection ==="
echo "policies=${POLICIES[*]}  per_policy=$PER_POLICY  shards=$SHARDS  shard_size=$SHARD_SIZE"
echo "max_parallel=$MAX_PAR  color=$COLOR  out=$OUT_H5"

# ---- 1. collect all shards (GPU-parallel, capped at MAX_PAR) ----
running=0
for pi in "${!POLICIES[@]}"; do
  policy=${POLICIES[$pi]}
  extra=()
  [[ "$policy" == "geometric"  ]] && extra=(--script-noise-scale 0.02)
  [[ "$policy" == "manipulate" ]] && extra=(--script-num-waypoints 3)
  for (( i=0; i<SHARDS; i++ )); do
    seed=$(( pi * 1000000 + i * SHARD_SIZE ))     # globally distinct seed ranges
    outdir="data/rope_red/$policy/shard$i"        # container-relative
    log="$LOGS/${policy}_shard${i}.log"
    echo "  launch $policy shard$i  (eps=$SHARD_SIZE seed=$seed)"
    SOFTGYM_PICKER_COLOR="$COLOR" bash "$COLLECT" \
      --env-name RopeFlatten --policy "$policy" \
      --num-episodes "$SHARD_SIZE" --num-variations "$SHARD_SIZE" \
      --seed "$seed" --img-size "$IMG" --output-dir "$outdir" \
      "${extra[@]}" > "$log" 2>&1 &
    running=$(( running + 1 ))
    if (( running >= MAX_PAR )); then wait -n; running=$(( running - 1 )); fi
  done
done
wait
echo "=== all shards collected ==="

# ---- 2. consolidate npz -> h5 per shard (host, parallel) ----
running=0
shard_h5s_random=(); shard_h5s_geometric=(); shard_h5s_manipulate=()
for policy in "${POLICIES[@]}"; do
  for (( i=0; i<SHARDS; i++ )); do
    sdir="$DATA/$policy/shard$i"
    sh5="$DATA/$policy/shard$i.h5"
    "$PY" simulation/utils/npz_to_hdf5.py --input-dir "$sdir" --output "$sh5" \
      > "$LOGS/consolidate_${policy}_shard${i}.log" 2>&1 &
    running=$(( running + 1 ))
    if (( running >= MAX_PAR )); then wait -n; running=$(( running - 1 )); fi
    eval "shard_h5s_${policy}+=(\"$sh5\")"
  done
done
wait
echo "=== shards consolidated to h5 ==="

# ---- 3. merge shards -> one h5 per policy ----
policy_h5s=()
for policy in "${POLICIES[@]}"; do
  arr="shard_h5s_${policy}[@]"
  inputs=()
  for h in "${!arr}"; do inputs+=(--input "$h"); done
  ph5="$DATA/${policy}.h5"
  echo "  merge $policy shards -> $ph5"
  "$PY" simulation/utils/merge_h5.py --output "$ph5" "${inputs[@]}" \
    > "$LOGS/merge_${policy}.log" 2>&1
  policy_h5s+=("$ph5:$policy")
done

# ---- 4. final merge of the 3 policies (labels set policy_id) ----
final_inputs=()
for spec in "${policy_h5s[@]}"; do final_inputs+=(--input "$spec"); done
mkdir -p "$(dirname "$OUT_H5")"
echo "=== final merge -> $OUT_H5 ==="
"$PY" simulation/utils/merge_h5.py --output "$OUT_H5" "${final_inputs[@]}"

echo "=== DONE ==="
"$PY" - "$OUT_H5" <<'PYEOF'
import sys, h5py
f = h5py.File(sys.argv[1], "r")
print("rows:", f["pixels"].shape, " episodes:", len(set(f["episode_idx"][:].tolist())))
print("policy_id_map:", f.attrs.get("policy_id_map"))
PYEOF
