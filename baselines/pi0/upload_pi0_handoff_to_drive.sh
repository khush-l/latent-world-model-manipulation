#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-gdrive:cs231n-pi0-handoff}
RAW_H5=${RAW_H5:-/home/ubuntu/cs231n-project/training/data/ropeflatten_geometric_5k_v3.h5}
LEROBOT_DATASET=${LEROBOT_DATASET:-/home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi}
PI0_CHECKPOINT=${PI0_CHECKPOINT:-/home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999}
LOG=${LOG:-/home/ubuntu/cs231n-project/baselines/logs/pi0_drive_upload.log}

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] Uploading pi0 handoff artifacts"
echo "remote=$REMOTE"
echo "raw_h5=$RAW_H5"
echo "lerobot_dataset=$LEROBOT_DATASET"
echo "pi0_checkpoint=$PI0_CHECKPOINT"

for path in "$RAW_H5" "$LEROBOT_DATASET" "$PI0_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required artifact: $path" >&2
    exit 2
  fi
done

rclone copy "$RAW_H5" "$REMOTE/training-data" --progress
rclone copy "$LEROBOT_DATASET" "$REMOTE/lerobot_datasets/khush/softgym-rope-openpi" --progress
rclone copy "$PI0_CHECKPOINT" "$REMOTE/openpi_checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999" --progress

echo "[$(date -Is)] Upload complete"
echo "Remote layout:"
echo "$REMOTE/training-data/ropeflatten_geometric_5k_v3.h5"
echo "$REMOTE/lerobot_datasets/khush/softgym-rope-openpi/"
echo "$REMOTE/openpi_checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999/"
