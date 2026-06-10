#!/usr/bin/env bash
#
# Overnight hyperparameter ablation sweep for the LeWM rope world-model.
#
# Runs a series of hyperparameter configurations SEQUENTIALLY on one GPU, ALL
# on the full mixed dataset, ~8 EPOCHS each. Per-run step counts are computed
# automatically from batch size and window length so every run sees the same
# number of epochs regardless of batch/history. Each logs to W&B under the
# shared tag `ablation_sweep_v1`.
#
# Runs are ordered by priority (most informative first) so that if you wake up
# before the sweep finishes, the important ablations are already done. The
# expensive batch_64 run (2x steps) is intentionally LAST.
#
# Usage (on the H100 box, from the repo root):
#   export WANDB_API_KEY=<your key>          # or `wandb login` beforehand
#   bash training/run_ablations.sh
#
# Override via env vars, e.g.:
#   EPOCHS=6 bash training/run_ablations.sh         # fewer epochs (faster)
#   START_AT=9 bash training/run_ablations.sh       # resume from run #9
#
set -u  # NOT -e: survive a single run failing.

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-8}"
IMG="${IMG:-128}"
PATCH="${PATCH:-16}"
LOG_EVERY="${LOG_EVERY:-25}"
PRECISION="${PRECISION:-bf16}"
PY="${PY:-.venv/bin/python}"
START_AT="${START_AT:-1}"               # skip runs before this index (resume)

WANDB_ENTITY="${WANDB_ENTITY:-231n-project}"
WANDB_PROJECT="${WANDB_PROJECT:-lewm-softgym}"
SWEEP_TAG="ablation_sweep_v1"

DATA_DIR="${DATA_DIR:-simulation/data/rope}"
DATA_FULL="$DATA_DIR/rope_full_dataset.h5"

RUNS_DIR="${RUNS_DIR:-training/runs}"
SUMMARY="$RUNS_DIR/${SWEEP_TAG}.summary"

mkdir -p "$RUNS_DIR"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if [[ ! -f "$DATA_FULL" ]]; then
  echo "MISSING DATA: $DATA_FULL (set DATA_DIR=...)"; exit 1
fi
if ! $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "WARNING: CUDA not available to $PY"
fi

# Number of windows in the dataset for a given (history+num_preds) window len.
# Computed once per distinct window length and cached.
declare -A WINDOWS_CACHE
count_windows() {
  local num_steps="$1"
  if [[ -n "${WINDOWS_CACHE[$num_steps]:-}" ]]; then
    echo "${WINDOWS_CACHE[$num_steps]}"; return
  fi
  local w
  w=$($PY - "$DATA_FULL" "$num_steps" <<'PYEOF'
import sys, h5py, numpy as np
path, num_steps = sys.argv[1], int(sys.argv[2])
with h5py.File(path, "r") as f:
    ep = f["episode_idx"][:]
_, counts = np.unique(ep, return_counts=True)
print(int((counts - num_steps + 1).clip(min=0).sum()))
PYEOF
)
  WINDOWS_CACHE[$num_steps]="$w"
  echo "$w"
}

echo "hyperparameter ablation sweep started: $(date)" | tee "$SUMMARY"
echo "data=$(basename "$DATA_FULL")  epochs/run=$EPOCHS  img=$IMG  precision=$PRECISION" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ---------------------------------------------------------------------------
# Run helper. Usage: run_one <name> <batch> <history> <num_preds> [extra flags...]
# Computes STEPS for EPOCHS epochs given batch + window length.
# ---------------------------------------------------------------------------
run_idx=0
run_one() {
  local name="$1"; local batch="$2"; local history="$3"; local num_preds="$4"; shift 4
  run_idx=$((run_idx + 1))
  if (( run_idx < START_AT )); then
    echo "[skip $run_idx] $name (START_AT=$START_AT)"; return
  fi

  local num_steps=$((history + num_preds))
  local windows; windows=$(count_windows "$num_steps")
  local steps=$(( EPOCHS * windows / batch ))

  local run_name="abl_${run_idx}_${name}"
  local logdir="$RUNS_DIR/$run_name"
  mkdir -p "$logdir"

  echo "==================================================================="
  echo "[$run_idx] $run_name  batch=$batch hist=$history preds=$num_preds steps=$steps ($EPOCHS ep)  $(date)"
  echo "  flags: $*"
  echo "==================================================================="

  local t0=$SECONDS
  "$PY" training/train.py \
    --data "$DATA_FULL" \
    --batch-size "$batch" --img-size "$IMG" --patch-size "$PATCH" \
    --history-size "$history" --num-preds "$num_preds" \
    --steps "$steps" --log-every "$LOG_EVERY" --save-every 0 \
    --gpu-cache --precision "$PRECISION" \
    --run-name "$run_name" \
    --wandb --wandb-entity "$WANDB_ENTITY" --wandb-project "$WANDB_PROJECT" \
    --wandb-tags "$SWEEP_TAG,$name" \
    "$@" \
    > "$logdir/train.log" 2>&1
  local rc=$?
  local dt=$((SECONDS - t0))

  if [[ $rc -eq 0 ]]; then
    local last; last=$(tail -n 40 "$logdir/train.log" | grep -E "^step " | tail -1)
    echo "[$run_idx] $run_name  OK  (${dt}s)  | $last" | tee -a "$SUMMARY"
  else
    echo "[$run_idx] $run_name  FAILED (rc=$rc, ${dt}s) — see $logdir/train.log" | tee -a "$SUMMARY"
    tail -n 12 "$logdir/train.log" | sed 's/^/    /'
  fi
}

# ---------------------------------------------------------------------------
# The sweep — ordered by priority (most informative first).
#            name                batch hist preds  extra flags
# ---------------------------------------------------------------------------
# Regularization
run_one baseline                 128   3    1
run_one no_sigreg                128   3    1    --no-sigreg
run_one sigreg_low               128   3    1    --sigreg-weight 0.01
run_one sigreg_high              128   3    1    --sigreg-weight 0.5
run_one dropout_off              128   3    1    --predictor-dropout 0.0

# Predictor capacity
run_one predictor_tiny           128   3    1    --predictor-dim-head 12
run_one predictor_shallow        128   3    1    --predictor-depth 3

# Optimization
run_one lr_high                  128   3    1    --lr 2e-4
run_one lr_low                   128   3    1    --lr 1e-5
run_one lr_cosine                128   3    1    --lr-schedule cosine --warmup-steps 2000
run_one weight_decay_off         128   3    1    --weight-decay 0.0
run_one weight_decay_high        128   3    1    --weight-decay 1e-2

# Prediction context / history
run_one history_5                128   5    1
run_one num_preds_2              128   3    2

# Batch size (batch_64 is 2x steps → LAST so it drops first if time runs short)
run_one batch_256                256   3    1
run_one batch_64                 64    3    1

echo "" | tee -a "$SUMMARY"
echo "ablation sweep finished: $(date)" | tee -a "$SUMMARY"
echo "All runs grouped in W&B under tag: $SWEEP_TAG" | tee -a "$SUMMARY"
echo "Local summary: $SUMMARY"



# #!/usr/bin/env bash
# # Sequentially train all final models on ONE GPU (single H100).
# #
# # 5 runs:
# #   1. mix19k_base       d192, proprio ON,  aux OFF   (baseline)
# #   2. mix19k_d384       d384  (capacity ablation)
# #   3. mix19k_aux        aux-state head ON (--aux-state-weight 0.5)
# #   4. mix19k_noproprio  proprio OFF
# #   5. mix20k_u          d192 baseline on the RopeFlatten+U union (multi-task)
# #
# # Dataloading: the final datasets are large (~71-76 GB of uint8 pixels), so
# # --gpu-cache would saturate an 80 GB H100 (and OOM on d384/the 20k set).
# # We cache pixels in HOST RAM (--cache-pixels, uint8, ~76 GB, COW-shared across
# # workers) when enough RAM is free, else fall back to plain streaming.
# #
# # Prereqs on the box: repo synced (with the aux-head edits in model.py/train.py),
# #   .venv with torch, wandb logged in (or set WANDB_MODE=offline), and both
# #   datasets present under simulation/data/rope/.
# #
# # Usage:  tmux new -s train 'bash training/run_all_ablations.sh'
# #   STEPS=60000 bash training/run_all_ablations.sh        # shorter runs
# #   WANDB_MODE=offline bash training/run_all_ablations.sh # no wandb login
# set -uo pipefail

# REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# cd "$REPO"
# PY=.venv/bin/python

# STEPS=${STEPS:-100000}
# SAVE_EVERY=${SAVE_EVERY:-20000}
# WMODE=${WANDB_MODE:-online}
# MIX=simulation/data/rope/rope_final_mix_19k.h5
# MIXU=simulation/data/rope/rope_final_mix_u_20k.h5

# [ -f "$MIX" ]  || { echo "MISSING $MIX";  exit 1; }
# [ -f "$MIXU" ] || { echo "MISSING $MIXU"; exit 1; }
# mkdir -p training/runs

# # --- choose dataloading by free RAM (cache-pixels needs ~80 GB) ---
# FREE_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')
# FREE_GB=${FREE_GB:-0}
# if [ "$FREE_GB" -ge 85 ]; then
#   CACHE="--cache-pixels --num-workers 8"
# else
#   CACHE="--num-workers 12"
# fi
# echo "=== free RAM ${FREE_GB} GB -> dataloading: $CACHE ==="
# echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) ==="
# echo "=== steps=$STEPS save_every=$SAVE_EVERY wandb=$WMODE ==="

# COMMON="--steps $STEPS --batch-size 128 --lr 5e-5 --lr-schedule constant --warmup-steps 0 \
#  --weight-decay 0.001 --grad-clip 1.0 --seed 3072 --img-size 128 --patch-size 16 \
#  --history-size 3 --num-preds 1 --frameskip 1 --predictor-depth 6 --predictor-heads 16 \
#  --predictor-dim-head 64 --predictor-dropout 0.1 --sigreg-weight 0.09 --precision bf16 \
#  $CACHE --save-every $SAVE_EVERY --log-every 50 \
#  --wandb --wandb-entity 231n-project --wandb-project lewm-softgym --wandb-mode $WMODE"

# run () {  # $1=run-name $2=data ; rest=extra flags
#   local name=$1 data=$2; shift 2
#   if [ -f "training/runs/$name/ckpt_final.pt" ]; then
#     echo "=== [$(date +%H:%M:%S)] SKIP $name (ckpt_final.pt exists) ==="; return 0
#   fi
#   echo "=== [$(date +%H:%M:%S)] START $name ==="
#   local t0=$(date +%s)
#   $PY training/train.py --data "$data" --run-name "$name" --wandb-name "$name" \
#       --wandb-tags rope,final,$name $COMMON "$@" \
#       > "training/runs/${name}_train.log" 2>&1
#   local rc=$?
#   echo "=== [$(date +%H:%M:%S)] END $name rc=$rc ($(( ($(date +%s)-t0)/60 )) min) ==="
#   [ $rc -ne 0 ] && echo "!!! $name FAILED — see training/runs/${name}_train.log (continuing)"
#   return 0
# }

# run mix19k_base       "$MIX"  --embed-dim 192 --use-proprio
# run mix19k_d384       "$MIX"  --embed-dim 384 --use-proprio
# run mix19k_aux        "$MIX"  --embed-dim 192 --use-proprio --aux-state-weight 0.5
# run mix19k_noproprio  "$MIX"  --embed-dim 192
# run mix20k_u          "$MIXU" --embed-dim 192 --use-proprio

# echo "=== ALL RUNS DONE $(date) ==="
# ls -1 training/runs/*/ckpt_final.pt 2>/dev/null