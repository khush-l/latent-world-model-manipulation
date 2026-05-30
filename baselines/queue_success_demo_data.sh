#!/usr/bin/env bash
set -euo pipefail

# Collect candidate scripted SoftGym demos, filter by final normalized performance,
# then convert the successful subset to HDF5 and LeRobot for ACT/SmolVLA.
#
# Usage examples:
#   baselines/queue_success_demo_data.sh rope existing
#   baselines/queue_success_demo_data.sh cloth pilot
#   baselines/queue_success_demo_data.sh cloth full

TASK=${1:-rope}          # rope | cloth
MODE=${2:-pilot}         # existing | pilot | full
ROOT=/home/ubuntu/cs231n-project
cd "$ROOT"

case "$TASK" in
  rope)
    ENV_NAME=RopeFlatten
    POLICY=geometric
    HORIZON=75
    INSTRUCTION="straighten the rope"
    REPO_ID="khush/softgym-ropeflatten-success-demos"
    EXISTING_DIR="simulation/data/trajectories/khush_scripted_rope_vla_150"
    CAND_DIR="simulation/data/trajectories/khush_rope_success_candidates_v2"
    FILTERED_DIR="simulation/data/trajectories/khush_rope_success_filtered_v2"
    H5_PATH="training/data/khush_rope_success_filtered_v2.h5"
    LEROBOT_ROOT="/home/ubuntu/lerobot_datasets/softgym_ropeflatten_success_filtered_v2"
    THRESHOLD=0.80
    MIN_EPISODES=100
    TOP_K=300
    FULL_EPISODES=1000
    ;;
  cloth)
    echo "cloth success-demo collection is disabled after merging origin/main:" >&2
    echo "simulation/utils/collect_trajectories.py now supports RopeFlatten geometric/manipulate policies only." >&2
    echo "Add a ClothFlatten policy to simulation/utils/geometric_policy.py before rerunning cloth collection." >&2
    exit 2
    ENV_NAME=ClothFlatten
    POLICY=geometric
    HORIZON=150
    INSTRUCTION="flatten the cloth"
    REPO_ID="khush/softgym-clothflatten-success-demos"
    EXISTING_DIR="simulation/data/trajectories/khush_scripted_cloth_vla_150"
    CAND_DIR="simulation/data/trajectories/khush_cloth_success_candidates_v2"
    FILTERED_DIR="simulation/data/trajectories/khush_cloth_success_filtered_v2"
    H5_PATH="training/data/khush_cloth_success_filtered_v2.h5"
    LEROBOT_ROOT="/home/ubuntu/lerobot_datasets/softgym_clothflatten_success_filtered_v2"
    THRESHOLD=0.55
    MIN_EPISODES=100
    TOP_K=300
    FULL_EPISODES=2500
    ;;
  *)
    echo "TASK must be rope or cloth" >&2
    exit 2
    ;;
esac

case "$MODE" in
  existing)
    INPUT_DIR="$EXISTING_DIR"
    ;;
  pilot)
    INPUT_DIR="${CAND_DIR}_pilot"
    rm -rf "$INPUT_DIR"
    SOFTGYM_SOFTWARE_GL=0 ./simulation/docker/collect-trajectories.sh \
      --env-name "$ENV_NAME" \
      --policy "$POLICY" \
      --num-episodes 50 \
      --horizon "$HORIZON" \
      --img-size 128 \
      --num-variations 50 \
      --script-noise-scale 0.02 \
      --output-dir "${INPUT_DIR#simulation/}"
    FILTERED_DIR="${FILTERED_DIR}_pilot"
    H5_PATH="${H5_PATH%.h5}_pilot.h5"
    LEROBOT_ROOT="${LEROBOT_ROOT}_pilot"
    TOP_K=50
    MIN_EPISODES=25
    ;;
  full)
    INPUT_DIR="$CAND_DIR"
    mkdir -p "$INPUT_DIR"
    SOFTGYM_SOFTWARE_GL=0 ./simulation/docker/collect-trajectories.sh \
      --env-name "$ENV_NAME" \
      --policy "$POLICY" \
      --num-episodes "$FULL_EPISODES" \
      --horizon "$HORIZON" \
      --img-size 128 \
      --num-variations 1000 \
      --script-noise-scale 0.02 \
      --output-dir "${INPUT_DIR#simulation/}"
    ;;
  *)
    echo "MODE must be existing, pilot, or full" >&2
    exit 2
    ;;
esac

rm -rf "$FILTERED_DIR" "$LEROBOT_ROOT"
./training/.venv/bin/python simulation/utils/filter_successful_trajectories.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$FILTERED_DIR" \
  --metric info_normalized_performance \
  --threshold "$THRESHOLD" \
  --top-k "$TOP_K" \
  --min-episodes "$MIN_EPISODES"

./training/.venv/bin/python simulation/utils/npz_to_hdf5.py \
  --input-dir "$FILTERED_DIR" \
  --output "$H5_PATH"

./training/.venv/bin/python baselines/softgym_hdf5_to_lerobot.py \
  --input "$H5_PATH" \
  --output-root "$LEROBOT_ROOT" \
  --repo-id "$REPO_ID" \
  --instruction "$INSTRUCTION" \
  --fps 10

echo "done"
echo "filtered npz: $FILTERED_DIR"
echo "hdf5: $H5_PATH"
echo "lerobot: $LEROBOT_ROOT"
