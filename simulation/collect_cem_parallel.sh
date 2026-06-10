#!/usr/bin/env bash
# Collect oracle-CEM expert RopeFlatten demonstrations, sharded + parallel.
#
# Pipeline per shard (independent, run concurrently up to MAX_PAR):
#   1. CEM plan  (softagent container): run_cem over EP_PER_SHARD episodes with a
#      DISTINCT --seed -> cem_traj.pkl  (actions in NormalizedEnv [-1,1] space).
#      Each CEM episode internally uses an 8-worker rollout Pool (8 CPU cores).
#   2. replay   (softgym container): replay_cem_traj.py replays each trajectory on
#      the RAW env (denormalizing actions, restoring start state) -> one v3 NPZ per
#      episode, global episode_idx offset = shard*EP_PER_SHARD.
#   3. consolidate: npz_to_hdf5 -> shard.h5.
# Then: merge_h5 of all shards -> OUT_H5.
#
# CONCURRENCY: each running CEM shard pins 8 cores (its rollout Pool) + 8 FleX envs
# (~8 GB RAM), and is PINNED to one GPU (round-robin over NUM_GPUS) so envs don't all
# pile on GPU 0. Set MAX_PAR so: MAX_PAR*8 <= vCPUs, MAX_PAR*8 FleX envs fit RAM
# (~1 GB/env), and MAX_PAR/NUM_GPUS shards-per-GPU don't choke a GPU. LESSON from a
# 1-GPU L4: ~16 FleX envs (2 shards) on ONE GPU already contends badly — so spread
# across GPUs and tune MAX_PAR with the sweep (docs/CEM_COLLECTION.md). 4xL40/126vCPU:
# start MAX_PAR=12 (3 shards/GPU = 24 envs/GPU), then sweep up/down. CEM is CPU-bound.
#
# Tunables (env vars):
#   TOTAL_EPISODES  total expert episodes        (default 2000)
#   SHARDS          number of shards             (default 28)
#   MAX_PAR         max concurrent CEM shards    (default 26)
#   MAX_ITERS       CEM iterations               (default 5)
#   TPD             timestep_per_decision        (default 2400; pop=TPD/iters/horizon)
#   IMG             image size                   (default 128)
#   ENV_NAME        SoftGym task                 (default RopeFlatten)
#   OUT_H5          merged dataset               (default simulation/data/<env>_cem/<env>_cem_dataset.h5)
#   SMOKE=1         tiny run (2 shards x 2 eps)  to validate the orchestration end-to-end
#
# Usage (on the big CPU box, after syncing the repo + building both images):
#   tmux new -s cem 'bash simulation/collect_cem_parallel.sh'
#   SMOKE=1 bash simulation/collect_cem_parallel.sh        # validate first!
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PY=${PY:-.venv/bin/python}
SOFTAGENT=simulation/docker/softagent-local.sh
SOFTGYM=simulation/docker/softgym-local.sh

ENV_NAME=${ENV_NAME:-RopeFlatten}
ENV_SLUG=$(echo "$ENV_NAME" | tr '[:upper:]' '[:lower:]')
TOTAL_EPISODES=${TOTAL_EPISODES:-2000}
SHARDS=${SHARDS:-200}              # small shards (10 eps) -> incremental, interrupt-safe
MAX_PAR=${MAX_PAR:-12}             # concurrent shards; MAX_PAR*8 <= vCPUs (126 -> <=15)
# Round-robin shards across GPUs. AUTO-DETECT from nvidia-smi (default) so a 1-GPU
# box never pins a shard to a nonexistent GPU. Override with NUM_GPUS=N if needed.
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)}
[ "${NUM_GPUS:-0}" -ge 1 ] 2>/dev/null || NUM_GPUS=1
MAX_ITERS=${MAX_ITERS:-3}          # slim CEM (validated: 0.998, ~2x faster than iters=5)
TPD=${TPD:-900}                    # population = TPD/MAX_ITERS/plan_horizon(15)
IMG=${IMG:-128}
OUT_H5=${OUT_H5:-simulation/data/${ENV_SLUG}_cem/${ENV_SLUG}_cem_dataset.h5}
# Per-box seed base: shard i uses seed = SEED_BASE + i*100000. To collect DISTINCT
# ropes across multiple machines (so merging adds diversity, not duplicates), give
# each box a different SEED_BASE spaced well beyond SHARDS*100000 (e.g. dev=1000,
# box2=50000000, box3=100000000).
SEED_BASE=${SEED_BASE:-1000}
if [[ "${SMOKE:-0}" == "1" ]]; then TOTAL_EPISODES=4; SHARDS=2; MAX_PAR=2; fi

EP_PER_SHARD=$(( (TOTAL_EPISODES + SHARDS - 1) / SHARDS ))
# Cross-container paths: softagent workdir = simulation/softagent/, softgym
# workdir = simulation/ (both mount simulation/). So CEM writes cem_traj.pkl under
# simulation/softagent/$CEM_BASE; the softgym replay container reads it as
# softagent/$CEM_BASE (relative to simulation/), and writes npz under $NPZ_BASE.
CEM_BASE=data/${ENV_SLUG}_cem_shards   # relative to softagent workdir (simulation/softagent/)
NPZ_BASE=data/${ENV_SLUG}_cem          # relative to softgym  workdir (simulation/)
LOGS=simulation/data/${ENV_SLUG}_cem/logs
mkdir -p "$LOGS" simulation/"$NPZ_BASE" simulation/softagent/"$CEM_BASE" "$(dirname "$OUT_H5")"

echo "=== CEM expert collection ($ENV_NAME) ==="
echo "total=$TOTAL_EPISODES shards=$SHARDS ep/shard=$EP_PER_SHARD max_par=$MAX_PAR num_gpus=$NUM_GPUS"
echo "CEM: max_iters=$MAX_ITERS tpd=$TPD (pop=$((TPD/MAX_ITERS/15)))  out=$OUT_H5"
echo "  -> ~$((MAX_PAR/NUM_GPUS>0?MAX_PAR/NUM_GPUS:1)) shards/GPU = ~$(((MAX_PAR/NUM_GPUS>0?MAX_PAR/NUM_GPUS:1)*8)) FleX envs/GPU"

# ---- 1. CEM planning + 2. replay->npz, per shard (capped at MAX_PAR) ----
shard_pipe() {  # $1 = shard index
  local i=$1
  local seed=$(( SEED_BASE + i * 100000 ))       # per-box distinct rope seeds
  local off=$(( i * EP_PER_SHARD ))              # global episode_idx offset
  local gpu=$(( i % NUM_GPUS ))                  # pin this shard to one GPU (round-robin)
  local cemdir="$CEM_BASE/shard$i"               # container-relative
  local npzdir="$NPZ_BASE/shard$i"
  local log="$LOGS/shard${i}.log"
  {
    echo "[shard $i] CEM plan: $EP_PER_SHARD eps seed=$seed gpu=$gpu -> $cemdir"
    CUDA_VISIBLE_DEVICES=$gpu bash "$SOFTAGENT" cem --env-name "$ENV_NAME" --test-episodes "$EP_PER_SHARD" \
      --num-variations "$EP_PER_SHARD" \
      --seed "$seed" --max-iters "$MAX_ITERS" --timestep-per-decision "$TPD" \
      --save-video False --exp-name "cem_shard$i" --log-dir "$cemdir" || return 1
    echo "[shard $i] replay -> npz (start-idx=$off, gpu=$gpu)"
    CUDA_VISIBLE_DEVICES=$gpu bash "$SOFTGYM" run \
      "python utils/replay_cem_traj.py --traj softagent/$cemdir/cem_traj.pkl \
       --output-dir $npzdir --start-idx $off --env-name $ENV_NAME --img-size $IMG" || return 1
    echo "[shard $i] consolidate -> shard.h5"
    "$PY" simulation/utils/npz_to_hdf5.py \
      --input-dir "simulation/$npzdir" --output "simulation/$npzdir.h5" || return 1
    echo "[shard $i] DONE"
  } > "$log" 2>&1
}

running=0
for (( i=0; i<SHARDS; i++ )); do
  shard_pipe "$i" &
  running=$(( running + 1 ))
  if (( running >= MAX_PAR )); then wait -n; running=$(( running - 1 )); fi
done
wait
echo "=== all shards done; merging ==="

# ---- 3. merge shard h5s -> one dataset ----
inputs=()
for (( i=0; i<SHARDS; i++ )); do
  h5="simulation/$NPZ_BASE/shard$i.h5"
  [ -f "$h5" ] && inputs+=(--input "$h5:cem")
done
[ ${#inputs[@]} -eq 0 ] && { echo "ERROR: no shard h5s produced — check $LOGS"; exit 1; }
"$PY" simulation/utils/merge_h5.py --output "$OUT_H5" "${inputs[@]}"

echo "=== DONE -> $OUT_H5 ==="
"$PY" - "$OUT_H5" <<'PYEOF'
import sys, h5py, numpy as np
f = h5py.File(sys.argv[1], "r")
perf = f["info_normalized_performance"][:]
fin = np.isfinite(perf)
print("rows:", f["pixels"].shape, "episodes:", len(set(f["episode_idx"][:].tolist())))
print("final-frame perf is per-row; mean over all frames: %.3f" % perf[fin].mean())
PYEOF
