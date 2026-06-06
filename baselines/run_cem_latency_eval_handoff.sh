#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-/home/ubuntu/openpi}
OPENPI_CONFIG=${OPENPI_CONFIG:-pi05_softgym_rope_lora}
PI0_EXP_NAME=${PI0_EXP_NAME:-softgym_rope_cem_final_lora_40k_b4_ckpt5k}
PI0_STEP=${PI0_STEP:-10000}
PI0_CHECKPOINT=${PI0_CHECKPOINT:-"$OPENPI_ROOT/checkpoints/$OPENPI_CONFIG/$PI0_EXP_NAME/$PI0_STEP"}
ACT_CHECKPOINT=${ACT_CHECKPOINT:-"$REPO_ROOT/outputs/train/act_rope_cem_final_25k/checkpoints/015000/pretrained_model"}
SMOLVLA_CHECKPOINT=${SMOLVLA_CHECKPOINT:-"$REPO_ROOT/outputs/train/smolvla_rope_cem_final_stronger/checkpoints/045000/pretrained_model"}
HANDOFF_DIR=${HANDOFF_DIR:-}
LATENCY_VENV=${LATENCY_VENV:-"$REPO_ROOT/.venv-latency-lerobot"}
OUT_ROOT=${OUT_ROOT:-"data/evals/cem_latency_handoff_$(date +%Y%m%d_%H%M%S)"}
EVAL_EPISODES=${EVAL_EPISODES:-50}
SEED=${SEED:-0}
CONFIG_START=${CONFIG_START:-0}
HORIZON=${HORIZON:-75}
IMG_SIZE=${IMG_SIZE:-128}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}
SAVE_EVERY_VIDEO=${SAVE_EVERY_VIDEO:-0}
DEVICE=${LEROBOT_POLICY_DEVICE:-cuda}
HOST_FOR_CONTAINER=${LEROBOT_POLICY_HOST_FOR_CONTAINER:-172.17.0.1}
UV_BIN=${UV:-${HOME}/.local/share/uv-bootstrap/bin/uv}
RUN_TARGETS=${1:-all}

usage() {
  cat <<USAGE
Usage:
  HANDOFF_DIR=/path/to/pi0_best_10k_latency_handoff $0 [all|pi0|act|smolvla|act,smolvla]

What this does:
  - restores handoff artifacts from HANDOFF_DIR when present
  - bootstraps OpenPI under OPENPI_ROOT for pi0 eval
  - creates a separate LeRobot policy-server venv at LATENCY_VENV
  - builds/compiles the SoftGym Docker image used for the simulator
  - runs the same fixed RopeFlatten config frame: config IDs $CONFIG_START..$((CONFIG_START + EVAL_EPISODES - 1)), seed=$SEED, horizon=$HORIZON

Host prerequisites that this script does not fully install:
  - NVIDIA driver and working nvidia-smi
  - Docker with NVIDIA Container Toolkit, so 'docker run --gpus all ...' works
  - git, curl, python3-venv, python3-pip; use --install-system to apt install these basics

Environment overrides:
  OPENPI_ROOT, PI0_CHECKPOINT, ACT_CHECKPOINT, SMOLVLA_CHECKPOINT, OUT_ROOT,
  EVAL_EPISODES, SEED, CONFIG_START, HORIZON, IMG_SIZE, SAVE_EVERY_VIDEO, LATENCY_VENV,
  LEROBOT_POLICY_DEVICE, LEROBOT_POLICY_HOST_FOR_CONTAINER
USAGE
}

log() { echo "[$(date -Is)] $*"; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
contains_target() {
  local target=$1
  [[ "$RUN_TARGETS" == "all" || ",${RUN_TARGETS}," == *",${target},"* ]]
}

if [[ "${RUN_TARGETS:-}" == "-h" || "${RUN_TARGETS:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${RUN_TARGETS:-}" == "--install-system" ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required for --install-system" >&2
    exit 1
  fi
  sudo apt update
  sudo apt install -y git curl python3 python3-venv python3-pip python3.12-dev docker.io
  usage
  exit 0
fi

restore_handoff_artifacts() {
  if [[ -z "$HANDOFF_DIR" || ! -d "$HANDOFF_DIR" ]]; then
    return
  fi
  log "Using handoff dir: $HANDOFF_DIR"
  if [[ -f "$HANDOFF_DIR/pi0_best_ckpt${PI0_STEP}_orbax.tar.gz" && ! -d "$PI0_CHECKPOINT/params" ]]; then
    log "Restoring pi0 checkpoint $PI0_STEP to $PI0_CHECKPOINT"
    mkdir -p "$(dirname "$PI0_CHECKPOINT")"
    tar -C "$(dirname "$PI0_CHECKPOINT")" -xzf "$HANDOFF_DIR/pi0_best_ckpt${PI0_STEP}_orbax.tar.gz"
  fi
  if [[ -f "$HANDOFF_DIR/pi0_norm_stats_softgym_rope_cem_final.tar.gz" ]]; then
    local stats_dir="$OPENPI_ROOT/assets/$OPENPI_CONFIG/khush"
    if [[ ! -d "$stats_dir/softgym-rope-cem-openpi-final" ]]; then
      log "Restoring pi0 norm stats to $stats_dir"
      mkdir -p "$stats_dir"
      tar -C "$stats_dir" -xzf "$HANDOFF_DIR/pi0_norm_stats_softgym_rope_cem_final.tar.gz"
    fi
  fi
}

setup_openpi() {
  if ! contains_target pi0; then
    return
  fi
  need_cmd git
  need_cmd python3
  if [[ ! -x "$UV_BIN" || ! -d "$OPENPI_ROOT/.git" ]]; then
    log "Bootstrapping OpenPI at $OPENPI_ROOT"
    "$REPO_ROOT/baselines/pi0/bootstrap_openpi.sh" "$OPENPI_ROOT"
  fi
  log "Installing repo OpenPI SoftGym config"
  "$REPO_ROOT/baselines/pi0/install_openpi_softgym_config.sh" "$OPENPI_ROOT"
}

setup_lerobot_venv() {
  if ! contains_target act && ! contains_target smolvla; then
    return
  fi
  need_cmd python3
  if [[ ! -x "$LATENCY_VENV/bin/python" ]]; then
    log "Creating LeRobot latency venv at $LATENCY_VENV"
    python3 -m venv "$LATENCY_VENV"
  fi
  log "Installing LeRobot latency dependencies"
  "$LATENCY_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$LATENCY_VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    'torch==2.6.0' 'torchvision==0.21.0'
  "$LATENCY_VENV/bin/python" -m pip install 'lerobot==0.4.4' 'huggingface-hub[cli,hf-transfer]==0.35.3' 'transformers==4.53.3' num2words
}

setup_softgym() {
  need_cmd docker
  log "Building SoftGym Docker image if needed"
  "$REPO_ROOT/simulation/docker/softgym-local.sh" build
  if ! compgen -G "$REPO_ROOT/simulation/PyFlex/bindings/build/pyflex*.so" >/dev/null; then
    log "Compiling PyFlex inside SoftGym Docker image"
    "$REPO_ROOT/simulation/docker/softgym-local.sh" compile
  fi
}

run_pi0_eval() {
  if [[ ! -d "$PI0_CHECKPOINT/params" ]]; then
    echo "Missing pi0 checkpoint: $PI0_CHECKPOINT" >&2
    exit 1
  fi
  local port=${OPENPI_POLICY_PORT:-8865}
  local server_log="$REPO_ROOT/baselines/logs/latency_pi0_ckpt${PI0_STEP}_server.log"
  local out="${OUT_ROOT}/pi0_ckpt${PI0_STEP}_episodes${EVAL_EPISODES}"
  mkdir -p "$(dirname "$server_log")"
  log "Starting pi0 server for checkpoint $PI0_CHECKPOINT"
  cd "$OPENPI_ROOT"
  "$UV_BIN" run python "$REPO_ROOT/baselines/pi0/openpi_softgym_server.py" \
    --config-name "$OPENPI_CONFIG" \
    --checkpoint "$PI0_CHECKPOINT" \
    --host 0.0.0.0 \
    --port "$port" \
    --task "straighten the rope" \
    >"$server_log" 2>&1 &
  local server_pid=$!
  trap 'kill "$server_pid" >/dev/null 2>&1 || true' RETURN
  for _ in $(seq 1 240); do
    if python3 - <<PY >/dev/null 2>&1
from urllib import request
request.urlopen('http://127.0.0.1:${port}/health', timeout=1).read()
PY
    then
      break
    fi
    sleep 1
  done
  python3 - <<PY
from urllib import request
print(request.urlopen('http://127.0.0.1:${port}/health', timeout=5).read().decode('utf-8'))
PY
  log "Running pi0 latency eval -> $out"
  cd "$REPO_ROOT/simulation"
  ./docker/softgym-local.sh run "python utils/eval_lerobot_policy.py \
    --env-name RopeFlatten \
    --policy-url http://${HOST_FOR_CONTAINER}:${port} \
    --output-dir ${out} \
    --policy-name pi0_ckpt${PI0_STEP}_latency \
    --checkpoint ${PI0_CHECKPOINT} \
    --num-episodes ${EVAL_EPISODES} \
    --seed ${SEED} \
    --fixed-config-ids \
    --config-start ${CONFIG_START} \
    --horizon ${HORIZON} \
    --timeout-s 180 \
    --img-size ${IMG_SIZE} \
    --success-threshold ${SUCCESS_THRESHOLD} \
    --save-every-video ${SAVE_EVERY_VIDEO}"
  kill "$server_pid" >/dev/null 2>&1 || true
  trap - RETURN
}

run_lerobot_eval() {
  local name=$1
  local checkpoint=$2
  local port=$3
  if [[ ! -d "$checkpoint" ]]; then
    echo "Missing checkpoint for $name: $checkpoint" >&2
    exit 1
  fi
  local out="${OUT_ROOT}/${name}_episodes${EVAL_EPISODES}"
  local server_log="$REPO_ROOT/baselines/logs/latency_${name}_server.log"
  log "Running $name latency eval -> $out"
  cd "$REPO_ROOT"
  LEROBOT_POLICY_DEVICE="$DEVICE" \
  LEROBOT_POLICY_PORT="$port" \
  LEROBOT_POLICY_SERVER_LOG="$server_log" \
  LEROBOT_POLICY_PYTHON="$LATENCY_VENV/bin/python" \
  "$REPO_ROOT/simulation/docker/eval-lerobot-policy.sh" \
    --env-name RopeFlatten \
    --policy-checkpoint "$checkpoint" \
    --policy-name "$name" \
    --output-dir "$out" \
    --num-episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --fixed-config-ids \
    --config-start "$CONFIG_START" \
    --horizon "$HORIZON" \
    --img-size "$IMG_SIZE" \
    --success-threshold "$SUCCESS_THRESHOLD" \
    --save-every-video "$SAVE_EVERY_VIDEO"
}

print_summaries() {
  log "Summaries under $OUT_ROOT"
  find "$OUT_ROOT" -name summary.json -print | sort | while read -r summary; do
    python3 - "$summary" <<'PY'
import json, sys
p = sys.argv[1]
with open(p) as f:
    d = json.load(f)
print(f"{p}: mean_norm={d.get('mean_final_normalized_performance'):.6f} success={float(d.get('success_rate')):.3f} return={d.get('mean_return'):.6f} infer_mean_ms={d.get('policy_inference_latency_ms_mean', float('nan')):.3f} roundtrip_mean_ms={d.get('policy_roundtrip_latency_ms_mean', float('nan')):.3f}")
PY
  done
}

need_cmd git
need_cmd curl
restore_handoff_artifacts
setup_openpi
setup_lerobot_venv
setup_softgym
mkdir -p "$OUT_ROOT"

if contains_target pi0; then
  run_pi0_eval
fi
if contains_target act; then
  run_lerobot_eval act_rope_cem_final_015000 "$ACT_CHECKPOINT" 8875
fi
if contains_target smolvla; then
  run_lerobot_eval smolvla_rope_cem_final_045000 "$SMOLVLA_CHECKPOINT" 8875
fi
print_summaries
log "Done"
