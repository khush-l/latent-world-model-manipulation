#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SESSION=${SESSION:-h100_direct_action_evals}
LOG=${LOG:-"$REPO_ROOT/baselines/logs/h100_direct_action_evals.log"}

mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t =$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd '$REPO_ROOT' && baselines/run_h100_direct_action_evals.sh 2>&1 | tee -a '$LOG'"
echo "Started H100 direct-action evals in tmux session: $SESSION"
echo "Attach: tmux attach -t =$SESSION"
echo "Log: $LOG"
