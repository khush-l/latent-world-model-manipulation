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
