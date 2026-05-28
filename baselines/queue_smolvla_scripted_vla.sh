#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/cs231n-project

TRAIN=/home/ubuntu/cs231n-project/training/.venv/bin/lerobot-train
LOG_DIR=/home/ubuntu/cs231n-project/baselines/logs
mkdir -p "$LOG_DIR"

ROPE_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_scripted_vla_150
CLOTH_ROOT=/home/ubuntu/lerobot_datasets/softgym_clothflatten_scripted_vla_150

echo "[$(date -Is)] Waiting for ACT queue to finish before starting SmolVLA..."
while tmux has-session -t train_act_scripted_vla 2>/dev/null; do
  sleep 120
done

echo "[$(date -Is)] ACT queue is done. Checking LeRobot datasets..."
test -d "$ROPE_ROOT"
test -d "$CLOTH_ROOT"

echo "[$(date -Is)] SmolVLA 1-step smoke train on RopeFlatten..."
"$TRAIN" \
  --dataset.repo_id=khush/softgym-ropeflatten-scripted-vla-150 \
  --dataset.root="$ROPE_ROOT" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/smolvla_softgym_smoke \
  --job_name=smolvla_softgym_smoke \
  --steps=1 \
  --batch_size=2 \
  --num_workers=0 \
  --save_freq=1 \
  --log_freq=1 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_smolvla_smoke.log"

echo "[$(date -Is)] Training SmolVLA RopeFlatten baseline..."
"$TRAIN" \
  --dataset.repo_id=khush/softgym-ropeflatten-scripted-vla-150 \
  --dataset.root="$ROPE_ROOT" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/smolvla_softgym_rope_scripted_vla_150 \
  --job_name=smolvla_softgym_rope_scripted_vla_150 \
  --steps=3000 \
  --batch_size=2 \
  --num_workers=0 \
  --save_freq=1000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_smolvla_rope_scripted_vla_150.log"

echo "[$(date -Is)] Training SmolVLA ClothFlatten baseline..."
"$TRAIN" \
  --dataset.repo_id=khush/softgym-clothflatten-scripted-vla-150 \
  --dataset.root="$CLOTH_ROOT" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/smolvla_softgym_cloth_scripted_vla_150 \
  --job_name=smolvla_softgym_cloth_scripted_vla_150 \
  --steps=3000 \
  --batch_size=2 \
  --num_workers=0 \
  --save_freq=1000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_smolvla_cloth_scripted_vla_150.log"

echo "[$(date -Is)] SmolVLA scripted baseline queue finished."
