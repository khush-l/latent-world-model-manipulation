#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SESSION=${SESSION:-rope_cem_final_smolvla_stronger}
LOG=${LOG:-"$REPO_ROOT/baselines/logs/rope_cem_final_smolvla_stronger.log"}

mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t =$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd '$REPO_ROOT' && LOG='$LOG' baselines/train_rope_cem_smolvla_stronger.sh 2>&1 | tee -a '$LOG'"
echo "Started SmolVLA CEM training in tmux session: $SESSION"
echo "Attach: tmux attach -t =$SESSION"
echo "Log: $LOG"
