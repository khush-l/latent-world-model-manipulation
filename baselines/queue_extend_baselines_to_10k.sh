#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/cs231n-project

TRAIN=/home/ubuntu/cs231n-project/training/.venv/bin/lerobot-train
LOG_DIR=/home/ubuntu/cs231n-project/baselines/logs
mkdir -p "$LOG_DIR"

ROPE_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_scripted_vla_150
CLOTH_ROOT=/home/ubuntu/lerobot_datasets/softgym_clothflatten_scripted_vla_150

wait_for_session_end() {
  local session="$1"
  echo "[$(date -Is)] Waiting for tmux session $session to finish..."
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 120
  done
}

wait_for_session_end train_smolvla_scripted_vla

echo "[$(date -Is)] Training tuned ACT RopeFlatten baseline from scratch..."
"$TRAIN" \
  --dataset.repo_id=khush/softgym-ropeflatten-scripted-vla-150 \
  --dataset.root="$ROPE_ROOT" \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --optimizer.type=adamw \
  --optimizer.lr=3e-5 \
  --optimizer.weight_decay=1e-4 \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=20000 \
  --scheduler.peak_lr=3e-5 \
  --scheduler.decay_lr=3e-6 \
  --output_dir=outputs/train/act_softgym_rope_scripted_vla_150_tuned_20k \
  --job_name=act_softgym_rope_scripted_vla_150_tuned_20k \
  --steps=20000 \
  --batch_size=32 \
  --num_workers=2 \
  --save_freq=2500 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_act_rope_scripted_vla_150_tuned_20k.log"

echo "[$(date -Is)] Training tuned ACT ClothFlatten baseline from scratch..."
"$TRAIN" \
  --dataset.repo_id=khush/softgym-clothflatten-scripted-vla-150 \
  --dataset.root="$CLOTH_ROOT" \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --optimizer.type=adamw \
  --optimizer.lr=3e-5 \
  --optimizer.weight_decay=1e-4 \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=20000 \
  --scheduler.peak_lr=3e-5 \
  --scheduler.decay_lr=3e-6 \
  --output_dir=outputs/train/act_softgym_cloth_scripted_vla_150_tuned_20k \
  --job_name=act_softgym_cloth_scripted_vla_150_tuned_20k \
  --steps=20000 \
  --batch_size=32 \
  --num_workers=2 \
  --save_freq=2500 \
  --log_freq=50 \
  --wandb.enable=false \
  2>&1 | tee "$LOG_DIR/train_act_cloth_scripted_vla_150_tuned_20k.log"

echo "[$(date -Is)] Extending SmolVLA RopeFlatten to 10k total steps..."
"$TRAIN" \
  --dataset.repo_id=khush/softgym-ropeflatten-scripted-vla-150 \
  --dataset.root="$ROPE_ROOT" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/smolvla_softgym_rope_scripted_vla_150 \
  --job_name=smolvla_softgym_rope_scripted_vla_150 \
  --steps=10000 \
  --batch_size=2 \
  --num_workers=0 \
  --save_freq=1000 \
  --log_freq=50 \
  --wandb.enable=false \
  --resume=true \
  2>&1 | tee "$LOG_DIR/train_smolvla_rope_scripted_vla_150_resume_10k.log"

echo "[$(date -Is)] Extending SmolVLA ClothFlatten to 10k total steps..."
"$TRAIN" \
  --dataset.repo_id=khush/softgym-clothflatten-scripted-vla-150 \
  --dataset.root="$CLOTH_ROOT" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/smolvla_softgym_cloth_scripted_vla_150 \
  --job_name=smolvla_softgym_cloth_scripted_vla_150 \
  --steps=10000 \
  --batch_size=2 \
  --num_workers=0 \
  --save_freq=1000 \
  --log_freq=50 \
  --wandb.enable=false \
  --resume=true \
  2>&1 | tee "$LOG_DIR/train_smolvla_cloth_scripted_vla_150_resume_10k.log"

echo "[$(date -Is)] 10k extension queue finished."
