#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

usage() {
  cat <<'USAGE'
Usage:
  utils/view-trajectory.sh [viewer args...]

Examples:
  utils/view-trajectory.sh --input data/trajectories/ClothFlatten_000000.npz
  utils/view-trajectory.sh --stride 2 --max-frames 80

If --input is omitted, the viewer uses the most recently modified NPZ in
data/trajectories/. Output goes to data/viewer/<episode>/index.html.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$SIM_ROOT"
python utils/view_trajectory.py "$@"
