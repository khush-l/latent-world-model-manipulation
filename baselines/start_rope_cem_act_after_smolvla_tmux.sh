#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SESSION=${SESSION:-rope_cem_final_act_after_smolvla}
WAIT_SESSION=${WAIT_SESSION:-rope_cem_final_smolvla_stronger}
LOG=${LOG:-"$REPO_ROOT/baselines/logs/rope_cem_final_act_after_smolvla_queue.log"}

mkdir -p "$(dirname "$LOG")"

if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t =$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd '$REPO_ROOT' && echo '[\$(date -Is)] Queue started; waiting for tmux session $WAIT_SESSION to finish.' | tee -a '$LOG'; while tmux has-session -t '=$WAIT_SESSION' 2>/dev/null; do sleep 300; done; echo '[\$(date -Is)] $WAIT_SESSION finished; starting ACT.' | tee -a '$LOG'; baselines/train_rope_cem_act_after_smolvla.sh 2>&1 | tee -a '$LOG'"

echo "Queued ACT training in tmux session: $SESSION"
echo "Waiting on tmux session: $WAIT_SESSION"
echo "Attach with: tmux attach -t =$SESSION"
echo "Log: $LOG"
