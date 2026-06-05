# pi0 CEM Final Handoff

Prepared artifacts for the full CEM RopeFlatten dataset are on Drive:

```text
gdrive:cs231n-pi0-handoff/cem_final_openpi_ready
```

This folder contains:

- `softgym-rope-cem-openpi-final_lerobot_dataset.tar.gz`
- `softgym-rope-cem-openpi-final_norm_stats.tar.gz`
- `cem_ropeflatten_final.h5`
- `README_PI0_CEM_FINAL_HANDOFF.txt`

The artifacts were prepared on a `g4dn.4xlarge` / Tesla T4 instance. Dataset
conversion and OpenPI norm stats completed, but pi05 LoRA training OOMed even at
batch size 1 because the T4 exposes only about 15 GB of VRAM. Use `g5.2xlarge`
or a larger-VRAM GPU instance for training.

## Restore On New Instance

```bash
mkdir -p /home/ubuntu/cs231n-project/baselines/artifacts/pi0_cem_final_handoff
rclone copy gdrive:cs231n-pi0-handoff/cem_final_openpi_ready \
  /home/ubuntu/cs231n-project/baselines/artifacts/pi0_cem_final_handoff \
  --progress

mkdir -p /home/ubuntu/lerobot_datasets/khush
tar -C /home/ubuntu/lerobot_datasets/khush \
  -xzf /home/ubuntu/cs231n-project/baselines/artifacts/pi0_cem_final_handoff/softgym-rope-cem-openpi-final_lerobot_dataset.tar.gz

mkdir -p /home/ubuntu/openpi/assets/pi05_softgym_rope_lora/khush
tar -C /home/ubuntu/openpi/assets/pi05_softgym_rope_lora/khush \
  -xzf /home/ubuntu/cs231n-project/baselines/artifacts/pi0_cem_final_handoff/softgym-rope-cem-openpi-final_norm_stats.tar.gz

mkdir -p /home/ubuntu/cs231n-project/training/data
cp /home/ubuntu/cs231n-project/baselines/artifacts/pi0_cem_final_handoff/cem_ropeflatten_final.h5 \
  /home/ubuntu/cs231n-project/training/data/
```

## Train

```bash
cd /home/ubuntu/cs231n-project

OPENPI_ROOT=/home/ubuntu/openpi \
OPENPI_DATASET_REPO_ID=khush/softgym-rope-cem-openpi-final \
OPENPI_EXP_NAME=softgym_rope_cem_final_lora_40k_b1 \
OPENPI_NUM_TRAIN_STEPS=40000 \
OPENPI_SAVE_INTERVAL=10000 \
OPENPI_KEEP_PERIOD=10000 \
OPENPI_BATCH_SIZE=1 \
OPENPI_NUM_WORKERS=1 \
LOG=/home/ubuntu/cs231n-project/baselines/logs/pi0_cem_final_lora_40k_b1_train.log \
baselines/pi0/start_pi0_lora_training_tmux.sh
```
