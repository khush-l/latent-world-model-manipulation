#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVAL=${EVAL:-"$REPO_ROOT/simulation/docker/eval-lerobot-policy.sh"}
PY=${PY:-"$REPO_ROOT/training/.venv/bin/python"}
LOG_DIR=${LOG_DIR:-"$REPO_ROOT/baselines/logs"}
RESULT_ROOT=${RESULT_ROOT:-"$REPO_ROOT/baselines/results/rope_cem_final_checkpoint_eval"}
PACKAGE_ROOT=${PACKAGE_ROOT:-"$REPO_ROOT/baselines/artifacts/rope_cem_final_checkpoints"}
DRIVE_DEST=${DRIVE_DEST:-"gdrive:baseline_checkpoints/rope_cem_final"}
WAIT_SESSION=${WAIT_SESSION:-rope_cem_final_act_after_smolvla}
EVAL_EPISODES=${EVAL_EPISODES:-8}
HORIZON=${HORIZON:-75}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}
UPLOAD_TO_DRIVE=${UPLOAD_TO_DRIVE:-1}

SMOLVLA_RUN_DIR=${SMOLVLA_RUN_DIR:-"$REPO_ROOT/outputs/train/smolvla_rope_cem_final_stronger"}
ACT_RUN_DIR=${ACT_RUN_DIR:-"$REPO_ROOT/outputs/train/act_rope_cem_final_25k"}
SMOLVLA_STEPS=(${SMOLVLA_STEPS:-30000 40000 45000 50000})
ACT_STEPS=(${ACT_STEPS:-5000 10000 15000 20000 25000})

mkdir -p "$LOG_DIR" "$RESULT_ROOT" "$PACKAGE_ROOT"
cd "$REPO_ROOT"

echo "[$(date -Is)] Waiting for tmux session $WAIT_SESSION to finish before eval/upload..."
while tmux has-session -t "=$WAIT_SESSION" 2>/dev/null; do
  sleep 300
done

echo "[$(date -Is)] Starting checkpoint evals. episodes=$EVAL_EPISODES horizon=$HORIZON threshold=$SUCCESS_THRESHOLD"

run_eval() {
  local family=$1
  local step=$2
  local run_dir=$3
  local label=$4
  local ckpt="$run_dir/checkpoints/$(printf '%06d' "$step")/pretrained_model"
  local out_dir="data/evals/rope_cem_final_checkpoint_eval/$label"
  local host_out="$REPO_ROOT/simulation/$out_dir"
  local log="$LOG_DIR/eval_${label}.log"

  if [[ ! -d "$ckpt" ]]; then
    echo "[$(date -Is)] SKIP $label missing checkpoint $ckpt"
    return 0
  fi

  echo "[$(date -Is)] Evaluating $label checkpoint=$ckpt"
  rm -rf "$host_out"
  SOFTGYM_SOFTWARE_GL=0 "$EVAL" \
    --env-name RopeFlatten \
    --policy-checkpoint "$ckpt" \
    --policy-name "$label" \
    --output-dir "$out_dir" \
    --num-episodes "$EVAL_EPISODES" \
    --horizon "$HORIZON" \
    --img-size 128 \
    --success-threshold "$SUCCESS_THRESHOLD" \
    --save-every-video 0 \
    2>&1 | tee "$log"

  if [[ -f "$host_out/summary.json" ]]; then
    cp "$host_out/summary.json" "$RESULT_ROOT/${label}_summary.json"
  fi
  if [[ -f "$host_out/scores.csv" ]]; then
    cp "$host_out/scores.csv" "$RESULT_ROOT/${label}_scores.csv"
  fi
}

for step in "${SMOLVLA_STEPS[@]}"; do
  run_eval smolvla "$step" "$SMOLVLA_RUN_DIR" "smolvla_rope_cem_final_$(printf '%06d' "$step")"
done

for step in "${ACT_STEPS[@]}"; do
  run_eval act "$step" "$ACT_RUN_DIR" "act_rope_cem_final_$(printf '%06d' "$step")"
done

"$PY" - <<PY
import csv, json, pathlib, re
root = pathlib.Path('$RESULT_ROOT')
rows = []
for p in sorted(root.glob('*_summary.json')):
    data = json.loads(p.read_text())
    label = p.name[:-len('_summary.json')]
    m = re.match(r'(smolvla|act)_rope_cem_final_(\d+)', label)
    family = m.group(1) if m else label.split('_', 1)[0]
    step = int(m.group(2)) if m else -1
    rows.append({
        'family': family,
        'step': step,
        'label': label,
        'checkpoint': (pathlib.Path('$SMOLVLA_RUN_DIR') if family == 'smolvla' else pathlib.Path('$ACT_RUN_DIR')) / 'checkpoints' / f'{step:06d}' / 'pretrained_model',
        'success_rate': float(data.get('success_rate', 0.0)),
        'mean_final_normalized_performance': float(data.get('mean_final_normalized_performance', data.get('mean_normalized_performance', 0.0))),
        'std_final_normalized_performance': float(data.get('std_final_normalized_performance', 0.0)),
        'n_episodes': int(data.get('n_episodes', 0)),
    })
rows.sort(key=lambda r: (r['family'], r['step']))
summary_csv = root / 'checkpoint_eval_summary.csv'
with summary_csv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['family','step','label','checkpoint','success_rate','mean_final_normalized_performance','std_final_normalized_performance','n_episodes'])
    w.writeheader(); w.writerows(rows)

best = {}
for family in sorted({r['family'] for r in rows}):
    candidates = [r for r in rows if r['family'] == family]
    if candidates:
        best[family] = max(candidates, key=lambda r: (r['success_rate'], r['mean_final_normalized_performance'], r['step']))
(root / 'best_checkpoints.json').write_text(json.dumps(best, indent=2, default=str))
print('Wrote', summary_csv)
print(json.dumps(best, indent=2, default=str))
PY

cp "$RESULT_ROOT/checkpoint_eval_summary.csv" "$PACKAGE_ROOT/"
cp "$RESULT_ROOT/best_checkpoints.json" "$PACKAGE_ROOT/"

package_best() {
  local family=$1
  local step
  step=$("$PY" - <<PY
import json
best=json.load(open('$RESULT_ROOT/best_checkpoints.json'))
print(best.get('$family', {}).get('step', ''))
PY
)
  if [[ -z "$step" ]]; then
    echo "[$(date -Is)] No best checkpoint for $family"
    return 0
  fi
  local run_dir="$SMOLVLA_RUN_DIR"
  [[ "$family" == "act" ]] && run_dir="$ACT_RUN_DIR"
  local step_dir
  step_dir=$(printf '%06d' "$step")
  local src="$run_dir/checkpoints/$step_dir/pretrained_model"
  local out="$PACKAGE_ROOT/${family}_rope_cem_final_${step_dir}_pretrained_model.tar.gz"
  echo "[$(date -Is)] Packaging $family best checkpoint step=$step_dir"
  tar -C "$run_dir/checkpoints/$step_dir" -czf "$out" pretrained_model
}

package_best smolvla
package_best act

if [[ "$UPLOAD_TO_DRIVE" == "1" ]]; then
  echo "[$(date -Is)] Uploading packages and eval summaries to $DRIVE_DEST"
  rclone mkdir "$DRIVE_DEST"
  rclone copy "$PACKAGE_ROOT" "$DRIVE_DEST" --progress
  rclone copy "$RESULT_ROOT" "$DRIVE_DEST/eval_results" --include '*summary.json' --include '*scores.csv' --include 'checkpoint_eval_summary.csv' --include 'best_checkpoints.json' --exclude '*' --progress
fi

echo "[$(date -Is)] Post-ACT eval/upload job complete."
echo "Results: $RESULT_ROOT/checkpoint_eval_summary.csv"
echo "Packages: $PACKAGE_ROOT"
echo "Drive: $DRIVE_DEST"
