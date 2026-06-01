#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  baselines/pi0/rsync_pi0_data_from_old_instance.sh OLD_HOST [SSH_KEY]

Examples:
  baselines/pi0/rsync_pi0_data_from_old_instance.sh ubuntu@1.2.3.4
  baselines/pi0/rsync_pi0_data_from_old_instance.sh ubuntu@1.2.3.4 ~/.ssh/key.pem

Set DRY_RUN=1 to preview rsync operations.
USAGE
  exit 2
fi

OLD_HOST=$1
SSH_KEY=${2:-}
REPO_ROOT=${REPO_ROOT:-/home/ubuntu/cs231n-project}
TRAINING_DATA=${TRAINING_DATA:-"$REPO_ROOT/training/data"}
LEROBOT_DATASETS=${LEROBOT_DATASETS:-/home/ubuntu/lerobot_datasets}

RSYNC_SSH=(ssh)
if [[ -n "$SSH_KEY" ]]; then
  RSYNC_SSH=(ssh -i "$SSH_KEY" -o IdentitiesOnly=yes)
fi

RSYNC_FLAGS=(-avP)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RSYNC_FLAGS+=(--dry-run)
fi

mkdir -p "$REPO_ROOT/simulation/data/trajectories"
mkdir -p "$REPO_ROOT/simulation/data/evals"
mkdir -p "$TRAINING_DATA"
mkdir -p "$LEROBOT_DATASETS"

run_rsync() {
  local src=$1
  local dst=$2
  echo
  echo "== rsync $src -> $dst =="
  rsync "${RSYNC_FLAGS[@]}" -e "${RSYNC_SSH[*]}" "$src" "$dst"
}

run_rsync "$OLD_HOST:/home/ubuntu/cs231n-project/simulation/data/trajectories/" \
  "$REPO_ROOT/simulation/data/trajectories/"
run_rsync "$OLD_HOST:/home/ubuntu/cs231n-project/simulation/data/evals/" \
  "$REPO_ROOT/simulation/data/evals/"
run_rsync "$OLD_HOST:/home/ubuntu/cs231n-project/training/data/" \
  "$TRAINING_DATA/"
run_rsync "$OLD_HOST:/home/ubuntu/lerobot_datasets/" \
  "$LEROBOT_DATASETS/"

echo
echo "Data sync complete. Run:"
echo "  baselines/pi0/check_pi0_data_paths.sh"

