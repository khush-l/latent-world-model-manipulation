#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/cs231n-project}
TRAINING_DATA=${TRAINING_DATA:-"$REPO_ROOT/training/data"}
LEROBOT_DATASETS=${LEROBOT_DATASETS:-/home/ubuntu/lerobot_datasets}

echo "== Expected dataset paths =="
for path in "$TRAINING_DATA" "$LEROBOT_DATASETS"; do
  echo
  echo "$path"
  if [[ -L "$path" ]]; then
    echo "  type: symlink"
    echo "  link: $(readlink "$path")"
    echo "  real: $(readlink -f "$path" || true)"
  elif [[ -e "$path" ]]; then
    echo "  type: local path"
    echo "  real: $(readlink -f "$path" || true)"
  else
    echo "  status: missing"
    echo "  real-if-created: $(readlink -f "$path" || true)"
  fi
done

echo
echo "== HDF5 candidates =="
if [[ -d "$TRAINING_DATA" ]]; then
  find "$TRAINING_DATA" -maxdepth 2 -type f \( -name "*.h5" -o -name "*.hdf5" \) -printf "  %p\n" | sort
else
  echo "  training data directory missing"
fi

echo
echo "== LeRobot dataset candidates =="
if [[ -d "$LEROBOT_DATASETS" ]]; then
  find "$LEROBOT_DATASETS" -maxdepth 2 -type d -printf "  %p\n" | sort | head -80
else
  echo "  LeRobot dataset directory missing"
fi

echo
echo "== Local SoftGym trajectories/evals =="
for path in "$REPO_ROOT/simulation/data/trajectories" "$REPO_ROOT/simulation/data/evals"; do
  echo
  echo "$path"
  if [[ -d "$path" ]]; then
    find "$path" -maxdepth 2 -type f \( -name "*.npz" -o -name "summary.json" -o -name "scores.csv" \) | wc -l | awk '{print "  files:", $1}'
    find "$path" -maxdepth 2 -type d | sort | head -40 | sed 's/^/  /'
  else
    echo "  missing"
  fi
done

