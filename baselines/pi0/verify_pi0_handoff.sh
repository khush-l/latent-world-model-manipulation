#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
DATASET_ROOT=${HF_LEROBOT_HOME:-${LEROBOT_HOME:-/home/ubuntu/lerobot_datasets}}
RAW_H5=${RAW_H5:-"$REPO_ROOT/training/data/ropeflatten_geometric_5k_v3.h5"}
CONVERTED_DATASET=${CONVERTED_DATASET:-"$DATASET_ROOT/khush/softgym-rope-openpi"}
PI0_CHECKPOINT=${PI0_CHECKPOINT:-/home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999}

status=0

check_path() {
  local label=$1
  local path=$2
  if [[ -e "$path" ]]; then
    echo "[ok] $label: $path"
  else
    echo "[missing] $label: $path"
    status=1
  fi
}

check_executable() {
  local label=$1
  local path=$2
  if [[ -x "$path" ]]; then
    echo "[ok] $label: $path"
  else
    echo "[missing/not executable] $label: $path"
    status=1
  fi
}

echo "== Repo scripts =="
check_executable "OpenPI bootstrap" "$REPO_ROOT/baselines/pi0/bootstrap_openpi.sh"
check_executable "OpenPI config install" "$REPO_ROOT/baselines/pi0/install_openpi_softgym_config.sh"
check_executable "Dataset conversion" "$REPO_ROOT/baselines/pi0/convert_pi0_softgym_rope_dataset.sh"
check_executable "LoRA training" "$REPO_ROOT/baselines/pi0/train_pi0_lora_softgym_rope.sh"
check_executable "LoRA training tmux" "$REPO_ROOT/baselines/pi0/start_pi0_lora_training_tmux.sh"
check_executable "LoRA eval" "$REPO_ROOT/baselines/pi0/eval_pi0_lora_softgym_rope.sh"
check_executable "H100 direct eval runner" "$REPO_ROOT/baselines/run_h100_direct_action_evals.sh"
check_executable "H100 tmux eval runner" "$REPO_ROOT/baselines/start_h100_direct_action_evals_tmux.sh"

echo
echo "== Local data/checkpoints =="
check_path "Raw rope HDF5" "$RAW_H5"
check_path "Converted LeRobot/OpenPI dataset" "$CONVERTED_DATASET"
check_path "OpenPI root" "$OPENPI_ROOT"
check_path "pi0 LoRA checkpoint" "$PI0_CHECKPOINT"

echo
echo "== Python syntax =="
PYTHON_BIN=${PYTHON:-}
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$OPENPI_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$OPENPI_ROOT/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
"$PYTHON_BIN" -m py_compile \
  "$REPO_ROOT/baselines/pi0/openpi_softgym_server.py" \
  "$REPO_ROOT/baselines/pi0/softgym_hdf5_to_openpi_lerobot.py" \
  "$REPO_ROOT/baselines/plot_pi0_rope_lora_charts.py"
echo "[ok] Python scripts compile"

echo
if [[ "$status" == "0" ]]; then
  echo "Handoff check passed."
else
  echo "Handoff check found missing local inputs. That is expected on a fresh H100 until data/checkpoints are copied."
fi
exit "$status"
