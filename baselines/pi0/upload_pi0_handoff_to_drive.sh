#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-gdrive:cs231n-pi0-handoff}
RAW_H5=${RAW_H5:-/home/ubuntu/cs231n-project/training/data/ropeflatten_geometric_5k_v3.h5}
LEROBOT_DATASET=${LEROBOT_DATASET:-/home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi}
PI0_CHECKPOINT=${PI0_CHECKPOINT:-/home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999}
INCLUDE_RAW_H5=${INCLUDE_RAW_H5:-0}
ARCHIVE_DIR=${ARCHIVE_DIR:-/home/ubuntu/pi0_handoff_archives}
LOG=${LOG:-/home/ubuntu/cs231n-project/baselines/logs/pi0_drive_upload.log}

LEROBOT_ARCHIVE=$ARCHIVE_DIR/lerobot_dataset_softgym_rope_openpi.tar
CHECKPOINT_ARCHIVE=$ARCHIVE_DIR/pi0_model_checkpoint_19999.tar

mkdir -p "$(dirname "$LOG")" "$ARCHIVE_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] Uploading pi0 handoff artifacts as simple archives"
echo "remote=$REMOTE"
echo "archive_dir=$ARCHIVE_DIR"
echo "include_raw_h5=$INCLUDE_RAW_H5"
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

This folder intentionally has only a few top-level files.

Files:
- lerobot_dataset_softgym_rope_openpi.tar
  Extract to: /home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi

- pi0_model_checkpoint_19999.tar
  Extract to: /home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999

- ropeflatten_geometric_5k_v3.h5
  Optional raw source HDF5. Only uploaded when INCLUDE_RAW_H5=1.

Repo branch:
- khush-baseline-results

Main docs after pulling repo:
- baselines/H100_DIRECT_ACTION_EVAL_README.md
- baselines/pi0/README.md

Extract commands on H100:
mkdir -p /home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi
mkdir -p /home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999
tar -xf lerobot_dataset_softgym_rope_openpi.tar -C /home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi
tar -xf pi0_model_checkpoint_19999.tar -C /home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999
MANIFEST

if [[ ! -f "$LEROBOT_ARCHIVE" ]]; then
  echo "[$(date -Is)] Creating $LEROBOT_ARCHIVE"
  tar -cf "$LEROBOT_ARCHIVE" -C "$LEROBOT_DATASET" .
else
  echo "[$(date -Is)] Reusing existing $LEROBOT_ARCHIVE"
fi

if [[ ! -f "$CHECKPOINT_ARCHIVE" ]]; then
  echo "[$(date -Is)] Creating $CHECKPOINT_ARCHIVE"
  tar -cf "$CHECKPOINT_ARCHIVE" -C "$PI0_CHECKPOINT" .
else
  echo "[$(date -Is)] Reusing existing $CHECKPOINT_ARCHIVE"
fi

rclone rcat "$REMOTE/README_PI0_HANDOFF.txt" <"$manifest"
rm -f "$manifest"
rclone copy "$LEROBOT_ARCHIVE" "$REMOTE" --progress
rclone copy "$CHECKPOINT_ARCHIVE" "$REMOTE" --progress
if [[ "$INCLUDE_RAW_H5" == "1" ]]; then
  rclone copy "$RAW_H5" "$REMOTE" --progress
fi

echo "[$(date -Is)] Upload complete"
echo "Remote layout:"
echo "$REMOTE/README_PI0_HANDOFF.txt"
echo "$REMOTE/lerobot_dataset_softgym_rope_openpi.tar"
echo "$REMOTE/pi0_model_checkpoint_19999.tar"
if [[ "$INCLUDE_RAW_H5" == "1" ]]; then
  echo "$REMOTE/ropeflatten_geometric_5k_v3.h5"
fi
