#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

usage() {
  cat <<'USAGE'
Usage:
  docker/collect-trajectories.sh [collector args...]

Examples:
  docker/collect-trajectories.sh --env-name ClothFlatten --num-episodes 10 --num-variations 10 --img-size 128
  docker/collect-trajectories.sh --env-name RopeFlatten --num-episodes 100 --save-state --save-depth

If no arguments are provided, this runs a small ClothFlatten collection:
  --env-name ClothFlatten --num-episodes 10 --num-variations 10 --img-size 128

Environment variables:
  SOFTGYM_LOCAL_IMAGE  Docker image tag used by softgym-local.sh
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

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

collector_args=("$@")
if [[ ${#collector_args[@]} -eq 0 ]]; then
  collector_args=(
    --env-name ClothFlatten
    --num-episodes 10
    --num-variations 10
    --img-size 128
  )
fi

if ! compgen -G "$SIM_ROOT/PyFlex/bindings/build/pyflex*.so" >/dev/null; then
  "$SCRIPT_DIR/softgym-local.sh" compile
fi

mkdir -p "$SIM_ROOT/data/trajectories"

cmd=(
  python
  utils/collect_trajectories.py
  "${collector_args[@]}"
)

"$SCRIPT_DIR/softgym-local.sh" run "$(quote_command "${cmd[@]}")"
