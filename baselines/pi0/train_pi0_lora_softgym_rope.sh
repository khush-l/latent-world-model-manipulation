#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
UV_BIN=${UV:-${HOME}/.local/share/uv-bootstrap/bin/uv}
CONFIG_NAME=${OPENPI_CONFIG:-pi05_softgym_rope_lora}
EXP_NAME=${OPENPI_EXP_NAME:-softgym_rope_lora_h100}
DATASET_ROOT=${LEROBOT_HOME:-${HF_LEROBOT_HOME:-/home/ubuntu/lerobot_datasets}}
HF_HOME=${HF_HOME:-/home/ubuntu/.cache/huggingface}

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv not found at $UV_BIN. Run baselines/pi0/bootstrap_openpi.sh $OPENPI_ROOT first." >&2
  exit 1
fi

if ! grep -q "pi05_softgym_rope_lora" "$OPENPI_ROOT/src/openpi/training/config.py"; then
  "$REPO_ROOT/baselines/pi0/install_openpi_softgym_config.sh" "$OPENPI_ROOT"
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Missing LeRobot dataset root: $DATASET_ROOT" >&2
  echo "Recover /home/ubuntu/lerobot_datasets or convert an HDF5 with baselines/pi0/softgym_hdf5_to_openpi_lerobot.py." >&2
  exit 1
fi

export HF_LEROBOT_HOME="$DATASET_ROOT"
unset LEROBOT_HOME
export HF_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_DISABLED=${WANDB_DISABLED:-true}

cd "$OPENPI_ROOT"

if [[ ! -d "assets/$CONFIG_NAME" ]]; then
  echo "Computing OpenPI norm stats for $CONFIG_NAME..."
  "$UV_BIN" run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
fi

CHECKPOINT_ROOT="$OPENPI_ROOT/checkpoints/$CONFIG_NAME/$EXP_NAME"
if [[ -d "$CHECKPOINT_ROOT" ]] && find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -Eq '^[0-9]+$'; then
  TRAIN_MODE=--resume
  echo "Resuming OpenPI LoRA training config=$CONFIG_NAME exp=$EXP_NAME from $CHECKPOINT_ROOT"
else
  TRAIN_MODE=--overwrite
  echo "Starting fresh OpenPI LoRA training config=$CONFIG_NAME exp=$EXP_NAME"
fi
"$UV_BIN" run scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  "$TRAIN_MODE"

