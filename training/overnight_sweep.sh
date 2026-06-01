#!/usr/bin/env bash
# Overnight capacity sweep: train a few bigger-capacity variants to 250k,
# SEQUENTIALLY (back-to-back) so the GPU stays busy all night.
#
# Why sequential, not parallel: --gpu-cache loads the whole dataset (~37 GB for
# 760k frames) onto the GPU, so two runs won't co-fit on an 80 GB H100. Running
# them in series keeps each at full speed with no OOM risk. One run failing
# (e.g. OOM on the biggest latent) does NOT abort the rest.
#
# Edit the VARIANTS list to taste. Each entry: "run_name | extra train.py flags".
# Run on the H100 (needs the dataset + venv there), inside tmux:
#   tmux new -s sweep 'bash training/overnight_sweep.sh'
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

PY=.venv/bin/python
DATA=simulation/data/rope/rope_full_dataset_smoothgray.h5

# Flags shared by every run (the proven gray+smooth+proprio recipe at 250k).
COMMON=(
  --data "$DATA" --use-proprio
  --batch-size 128 --img-size 128 --patch-size 16
  --history-size 3 --num-preds 1
  --predictor-heads 16 --predictor-dim-head 64 --predictor-dropout 0.1
  --sigreg-weight 0.09 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0
  --steps 250000 --log-every 10 --save-every 25000
  --gpu-cache --precision bf16
  --wandb --wandb-entity 231n-project --wandb-project lewm-softgym
)

# "run_name | extra flags"  — capacity variants only (no arch rewrites).
VARIANTS=(
  "rope_sg_proprio_d384_250k       | --embed-dim 384 --predictor-depth 6"
  "rope_sg_proprio_d512_250k       | --embed-dim 512 --predictor-depth 6"
  "rope_sg_proprio_d384_deep_250k  | --embed-dim 384 --predictor-depth 12"
)

[ -f "$DATA" ] || { echo "ERROR: dataset not found at $DATA (transfer it first)"; exit 1; }
echo "=== overnight sweep: ${#VARIANTS[@]} runs x 250k, sequential ==="

for v in "${VARIANTS[@]}"; do
  name=$(echo "${v%%|*}" | xargs)
  extra=$(echo "${v#*|}" | xargs)
  log="/tmp/train_${name}.log"
  echo ""
  echo "######## [$(date '+%F %H:%M')] START ${name}  (${extra}) ########"
  "$PY" training/train.py "${COMMON[@]}" $extra \
    --run-name "$name" --wandb-name "$name" \
    --wandb-tags "rope,gray,smooth,proprio,sweep" 2>&1 | tee "$log"
  echo "######## [$(date '+%F %H:%M')] END ${name} (exit ${PIPESTATUS[0]}) ########"
done
echo "=== ALL SWEEP RUNS DONE [$(date '+%F %H:%M')] ==="
