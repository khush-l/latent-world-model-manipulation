#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/cs231n-project

PY=/home/ubuntu/cs231n-project/training/.venv/bin/python
TRAIN=/home/ubuntu/cs231n-project/training/.venv/bin/lerobot-train
LOG_DIR=/home/ubuntu/cs231n-project/baselines/logs
mkdir -p "$LOG_DIR"

ROPE_H5=/home/ubuntu/cs231n-project/training/data/khush_scripted_rope_vla_150.h5
CLOTH_H5=/home/ubuntu/cs231n-project/training/data/khush_scripted_cloth_vla_150.h5

ROPE_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_scripted_vla_150
CLOTH_ROOT=/home/ubuntu/lerobot_datasets/softgym_clothflatten_scripted_vla_150

echo "[$(date -Is)] Waiting for scripted HDF5 files..."
while [[ ! -s "$ROPE_H5" || ! -s "$CLOTH_H5" ]]; do
  ls -lh "$ROPE_H5" "$CLOTH_H5" 2>/dev/null || true
  sleep 120
done

echo "[$(date -Is)] Found both HDF5 files."
ls -lh "$ROPE_H5" "$CLOTH_H5"

echo "[$(date -Is)] Validating HDF5 inputs..."
"$PY" - <<'PY'
from pathlib import Path
import h5py
import numpy as np

for path in [
    Path("/home/ubuntu/cs231n-project/training/data/khush_scripted_rope_vla_150.h5"),
    Path("/home/ubuntu/cs231n-project/training/data/khush_scripted_cloth_vla_150.h5"),
]:
    with h5py.File(path, "r") as f:
        print(path.name, dict(f.attrs))
        print("  pixels", f["pixels"].shape, f["pixels"].dtype)
        print("  action", f["action"].shape, f["action"].dtype)
        print("  finite action rows", int(np.isfinite(f["action"][:]).all(axis=1).sum()))
PY

if [[ ! -d "$ROPE_ROOT" ]]; then
  echo "[$(date -Is)] Converting rope to LeRobot..."
  "$PY" baselines/softgym_hdf5_to_lerobot.py \
    --input "$ROPE_H5" \
    --output-root "$ROPE_ROOT" \
    --repo-id khush/softgym-ropeflatten-scripted-vla-150 \
    2>&1 | tee "$LOG_DIR/convert_rope_lerobot.log"
else
  echo "[$(date -Is)] Rope LeRobot dataset already exists at $ROPE_ROOT"
fi

if [[ ! -d "$CLOTH_ROOT" ]]; then
  echo "[$(date -Is)] Converting cloth to LeRobot..."
  "$PY" baselines/softgym_hdf5_to_lerobot.py \
    --input "$CLOTH_H5" \
    --output-root "$CLOTH_ROOT" \
    --repo-id khush/softgym-clothflatten-scripted-vla-150 \
    2>&1 | tee "$LOG_DIR/convert_cloth_lerobot.log"
else
  echo "[$(date -Is)] Cloth LeRobot dataset already exists at $CLOTH_ROOT"
fi

echo "[$(date -Is)] Training ACT RopeFlatten baseline..."
"$TRAIN" \
  --dataset.repo_id=khush/softgym-ropeflatten-scripted-vla-150 \
  --dataset.root="$ROPE_ROOT" \
  --policy.type=act \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_softgym_rope_scripted_vla_150 \
  --job_name=act_softgym_rope_scripted_vla_150 \
  --steps=3000 \
  --batch_size=16 \
  --num_workers=2 \
  --save_freq=1000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_act_rope_scripted_vla_150.log"

echo "[$(date -Is)] Training ACT ClothFlatten baseline..."
"$TRAIN" \
  --dataset.repo_id=khush/softgym-clothflatten-scripted-vla-150 \
  --dataset.root="$CLOTH_ROOT" \
  --policy.type=act \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_softgym_cloth_scripted_vla_150 \
  --job_name=act_softgym_cloth_scripted_vla_150 \
  --steps=3000 \
  --batch_size=16 \
  --num_workers=2 \
  --save_freq=1000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_act_cloth_scripted_vla_150.log"

echo "[$(date -Is)] ACT scripted baseline queue finished."
