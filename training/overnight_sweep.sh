#!/usr/bin/env bash
# Overnight MEGA-dataset capacity ablation: same data (random + smooth + noisy),
# only embed_dim / predictor-depth change, trained to 250k SEQUENTIALLY (back-to-
# back) so the GPU stays busy all night. d192 first = clean anchor vs v2's 75%
# (one variable changed: dataset); d384 + d384-deep test whether the bigger,
# random-inclusive data finally makes capacity pay off (it never did on smoothgray).
#
# NOTE: --cache-pixels (NOT --gpu-cache). The mega set is 1.9M frames ~= 93 GB of
# pixels, which does NOT fit on an 80 GB H100 VRAM. --cache-pixels holds the
# decompressed pixels in CPU RAM (~93 GB; check `free -g`) and removes per-step
# gzip, so throughput stays near gpu-cache speed. If the box is short on RAM, drop
# --cache-pixels (keep --num-workers) to stream from disk (slower but safe).
#
# Edit the VARIANTS list to taste. Each entry: "run_name | extra train.py flags".
# Run on the H100 (needs the dataset + venv there), inside tmux:
#   tmux new -s sweep 'bash training/overnight_sweep.sh'
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

PY=.venv/bin/python
DATA=simulation/data/rope/rope_mega_dataset.h5

# Flags shared by every run (the proven gray+proprio recipe at 250k, mega data).
COMMON=(
  --data "$DATA" --use-proprio
  --batch-size 128 --img-size 128 --patch-size 16
  --history-size 3 --num-preds 1
  --predictor-heads 16 --predictor-dim-head 64 --predictor-dropout 0.1
  --sigreg-weight 0.09 --lr 5e-5 --weight-decay 1e-3 --grad-clip 1.0
  --steps 250000 --log-every 10 --save-every 25000
  --cache-pixels --num-workers 16 --precision bf16
  --wandb --wandb-entity 231n-project --wandb-project lewm-softgym
)

# "run_name | extra flags"  — capacity ablation on the mega data.
# d192 = anchor (matches the v2 recipe that hit 75%); d384 = pure capacity test;
# d384-deep = +predictor depth (had the best holding on smoothgray).
VARIANTS=(
  "rope_mega_d192_250k       | --embed-dim 192 --predictor-depth 6"
  "rope_mega_d384_250k       | --embed-dim 384 --predictor-depth 6"
  "rope_mega_d384_deep_250k  | --embed-dim 384 --predictor-depth 12"
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
    --wandb-tags "rope,gray,mega,proprio,sweep" 2>&1 | tee "$log"
  echo "######## [$(date '+%F %H:%M')] END ${name} (exit ${PIPESTATUS[0]}) ########"
done
echo "=== ALL SWEEP RUNS DONE [$(date '+%F %H:%M')] ==="
