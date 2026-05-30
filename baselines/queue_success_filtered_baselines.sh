#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/cs231n-project

TRAIN=/home/ubuntu/cs231n-project/training/.venv/bin/lerobot-train
LOG_DIR=/home/ubuntu/cs231n-project/baselines/logs
mkdir -p "$LOG_DIR"

ROPE_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_success_filtered_v2
CLOTH_ROOT=/home/ubuntu/lerobot_datasets/softgym_clothflatten_success_filtered_v2
ROPE_RANDOM_H5=/home/ubuntu/cs231n-project/training/data/khush_random_rope_5k75.h5
CLOTH_RANDOM_H5=/home/ubuntu/cs231n-project/training/data/khush_random_cloth_5k75.h5
ROPE_RANDOM_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_random_5k75
CLOTH_RANDOM_ROOT=/home/ubuntu/lerobot_datasets/softgym_clothflatten_random_5k75

ROPE_REPO=khush/softgym-ropeflatten-success-demos
CLOTH_REPO=khush/softgym-clothflatten-success-demos
ROPE_RANDOM_REPO=khush/softgym-ropeflatten-random-5k75
CLOTH_RANDOM_REPO=khush/softgym-clothflatten-random-5k75

echo "[$(date -Is)] Waiting for filtered success-demo LeRobot datasets..."
while [[ ! -d "$ROPE_ROOT" || ! -d "$CLOTH_ROOT" ]]; do
  ls -ld "$ROPE_ROOT" "$CLOTH_ROOT" 2>/dev/null || true
  sleep 120
done



echo "[$(date -Is)] Ensuring random-control LeRobot datasets exist..."
if [[ ! -d "$ROPE_RANDOM_ROOT" ]]; then
  ./training/.venv/bin/python baselines/softgym_hdf5_to_lerobot.py \
    --input "$ROPE_RANDOM_H5" \
    --output-root "$ROPE_RANDOM_ROOT" \
    --repo-id "$ROPE_RANDOM_REPO" \
    --instruction "straighten the rope" \
    2>&1 | tee "$LOG_DIR/convert_rope_random_lerobot.log"
fi
if [[ ! -d "$CLOTH_RANDOM_ROOT" ]]; then
  ./training/.venv/bin/python baselines/softgym_hdf5_to_lerobot.py \
    --input "$CLOTH_RANDOM_H5" \
    --output-root "$CLOTH_RANDOM_ROOT" \
    --repo-id "$CLOTH_RANDOM_REPO" \
    --instruction "flatten the cloth" \
    2>&1 | tee "$LOG_DIR/convert_cloth_random_lerobot.log"
fi

echo "[$(date -Is)] Training ACT random-control RopeFlatten baseline..."
"$TRAIN" \
  --dataset.repo_id="$ROPE_RANDOM_REPO" \
  --dataset.root="$ROPE_RANDOM_ROOT" \
  --policy.type=act \
  --policy.chunk_size=25 \
  --policy.n_action_steps=25 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_softgym_rope_random_5k75_5k \
  --job_name=act_softgym_rope_random_5k75_5k \
  --steps=5000 \
  --batch_size=16 \
  --num_workers=2 \
  --save_freq=5000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_act_rope_random_5k75_5k.log"

echo "[$(date -Is)] Training ACT random-control ClothFlatten baseline..."
"$TRAIN" \
  --dataset.repo_id="$CLOTH_RANDOM_REPO" \
  --dataset.root="$CLOTH_RANDOM_ROOT" \
  --policy.type=act \
  --policy.chunk_size=25 \
  --policy.n_action_steps=25 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_softgym_cloth_random_5k75_5k \
  --job_name=act_softgym_cloth_random_5k75_5k \
  --steps=5000 \
  --batch_size=16 \
  --num_workers=2 \
  --save_freq=5000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_act_cloth_random_5k75_5k.log"


echo "[$(date -Is)] Training ACT RopeFlatten success-filtered baseline..."
"$TRAIN" \
  --dataset.repo_id="$ROPE_REPO" \
  --dataset.root="$ROPE_ROOT" \
  --policy.type=act \
  --policy.chunk_size=25 \
  --policy.n_action_steps=25 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_softgym_rope_success_filtered_v2_8k \
  --job_name=act_softgym_rope_success_filtered_v2_8k \
  --steps=8000 \
  --batch_size=16 \
  --num_workers=2 \
  --save_freq=2000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_act_rope_success_filtered_v2_8k.log"

echo "[$(date -Is)] Training ACT ClothFlatten success-filtered baseline..."
"$TRAIN" \
  --dataset.repo_id="$CLOTH_REPO" \
  --dataset.root="$CLOTH_ROOT" \
  --policy.type=act \
  --policy.chunk_size=25 \
  --policy.n_action_steps=25 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_softgym_cloth_success_filtered_v2_8k \
  --job_name=act_softgym_cloth_success_filtered_v2_8k \
  --steps=8000 \
  --batch_size=16 \
  --num_workers=2 \
  --save_freq=2000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_act_cloth_success_filtered_v2_8k.log"

echo "[$(date -Is)] Training SmolVLA RopeFlatten success-filtered baseline..."
"$TRAIN" \
  --dataset.repo_id="$ROPE_REPO" \
  --dataset.root="$ROPE_ROOT" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/smolvla_softgym_rope_success_filtered_v2_20k \
  --job_name=smolvla_softgym_rope_success_filtered_v2_20k \
  --steps=20000 \
  --batch_size=2 \
  --num_workers=0 \
  --save_freq=5000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_smolvla_rope_success_filtered_v2_20k.log"

echo "[$(date -Is)] Training SmolVLA ClothFlatten success-filtered baseline..."
"$TRAIN" \
  --dataset.repo_id="$CLOTH_REPO" \
  --dataset.root="$CLOTH_ROOT" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/smolvla_softgym_cloth_success_filtered_v2_20k \
  --job_name=smolvla_softgym_cloth_success_filtered_v2_20k \
  --steps=20000 \
  --batch_size=2 \
  --num_workers=0 \
  --save_freq=5000 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_smolvla_cloth_success_filtered_v2_20k.log"

echo "[$(date -Is)] Success-filtered baseline training finished."
