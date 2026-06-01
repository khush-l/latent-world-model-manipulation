#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-gdrive:cs231n-pi0-handoff}
RAW_H5=${RAW_H5:-/home/ubuntu/cs231n-project/training/data/ropeflatten_geometric_5k_v3.h5}
LEROBOT_DATASET=${LEROBOT_DATASET:-/home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi}
PI0_CHECKPOINT=${PI0_CHECKPOINT:-/home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999}
INCLUDE_RAW_H5=${INCLUDE_RAW_H5:-0}
LOG=${LOG:-/home/ubuntu/cs231n-project/baselines/logs/pi0_drive_upload.log}

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] Uploading pi0 handoff artifacts"
echo "remote=$REMOTE"
echo "include_raw_h5=$INCLUDE_RAW_H5"
echo "raw_h5=$RAW_H5"
echo "lerobot_dataset=$LEROBOT_DATASET"
echo "pi0_checkpoint=$PI0_CHECKPOINT"

for path in "$LEROBOT_DATASET" "$PI0_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required artifact: $path" >&2
    exit 2
  fi
done
if [[ "$INCLUDE_RAW_H5" == "1" && ! -f "$RAW_H5" ]]; then
  echo "Missing raw HDF5: $RAW_H5" >&2
  exit 2
fi

manifest=$(mktemp)
cat >"$manifest" <<MANIFEST
# pi0 SoftGym Rope Handoff

Put this Drive folder at the top level for the teammate. It intentionally uses a flat layout.

Folders/files:
- softgym-rope-openpi/            Converted LeRobot/OpenPI dataset. Copy to /home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi
- pi0_lora_checkpoint_19999/      Trained OpenPI checkpoint. Copy to /home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999
- ropeflatten_geometric_5k_v3.h5  Optional raw source HDF5, only uploaded when INCLUDE_RAW_H5=1. Copy to /home/ubuntu/cs231n-project/training/data/ropeflatten_geometric_5k_v3.h5

Repo branch:
- khush-baseline-results

Main docs after pulling repo:
- baselines/H100_DIRECT_ACTION_EVAL_README.md
- baselines/pi0/README.md
MANIFEST

rclone rcat "$REMOTE/README_PI0_HANDOFF.txt" <"$manifest"
rm -f "$manifest"

# Flat, human-readable Drive layout. No deep openpi/lerobot nesting.
rclone copy "$LEROBOT_DATASET" "$REMOTE/softgym-rope-openpi" --progress
rclone copy "$PI0_CHECKPOINT" "$REMOTE/pi0_lora_checkpoint_19999" --progress
if [[ "$INCLUDE_RAW_H5" == "1" ]]; then
  rclone copy "$RAW_H5" "$REMOTE" --progress
fi

echo "[$(date -Is)] Upload complete"
echo "Remote layout:"
echo "$REMOTE/README_PI0_HANDOFF.txt"
echo "$REMOTE/softgym-rope-openpi/"
echo "$REMOTE/pi0_lora_checkpoint_19999/"
if [[ "$INCLUDE_RAW_H5" == "1" ]]; then
  echo "$REMOTE/ropeflatten_geometric_5k_v3.h5"
fi
