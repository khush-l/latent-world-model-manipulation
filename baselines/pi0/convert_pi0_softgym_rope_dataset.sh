#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
UV_BIN=${UV:-${HOME}/.local/share/uv-bootstrap/bin/uv}
INPUT_H5=${INPUT_H5:-"$REPO_ROOT/training/data/ropeflatten_geometric_5k_v3.h5"}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi}
REPO_ID=${OPENPI_DATASET_REPO_ID:-khush/softgym-rope-openpi}
INSTRUCTION=${OPENPI_TASK:-straighten the rope}
LOG=${LOG:-"$REPO_ROOT/baselines/logs/pi0_convert_softgym_rope.log"}

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] Starting SoftGym HDF5 -> OpenPI LeRobot conversion"
echo "input=$INPUT_H5"
echo "output=$OUTPUT_ROOT"
echo "repo_id=$REPO_ID"

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv not found at $UV_BIN. Run baselines/pi0/bootstrap_openpi.sh $OPENPI_ROOT first." >&2
  exit 1
fi
if [[ ! -f "$INPUT_H5" ]]; then
  echo "Missing input HDF5: $INPUT_H5" >&2
  exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  if [[ "${OVERWRITE:-0}" != "1" ]]; then
    echo "Output exists: $OUTPUT_ROOT" >&2
    echo "Set OVERWRITE=1 to replace it." >&2
    exit 1
  fi
  echo "Removing existing output because OVERWRITE=1: $OUTPUT_ROOT"
  rm -rf "$OUTPUT_ROOT"
fi

cd "$OPENPI_ROOT"
"$UV_BIN" run python "$REPO_ROOT/baselines/pi0/softgym_hdf5_to_openpi_lerobot.py" \
  --input "$INPUT_H5" \
  --output-root "$OUTPUT_ROOT" \
  --repo-id "$REPO_ID" \
  --instruction "$INSTRUCTION"

echo "[$(date -Is)] Conversion complete"
