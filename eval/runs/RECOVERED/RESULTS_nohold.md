# Recovered results — proprio vs baseline, no hold-on-converge (k=5)

Reconstructed from `/tmp/cmp_nohold.log` after the original `cmp_*` run dirs
(summary_k5.json + GIFs) were accidentally deleted. Per-episode MPC/expert
numbers are from the log; `start` perfs are from the live analysis printed
during the session (true reset-state performance).

Setup: `--mode imitate --planner gradient --n-episodes 8 --seed 0 --goal-offset 5
--eval-budget 75 --history-size 3` (NO hold-on-converge), env_seed=0,
time_step-restore fix applied, proprio z-scored to training stats, raw actions.

Models: baseline = rope_full_mixed_v1/ckpt_final.pt; proprio = rope_proprio_v1/ckpt_final.pt.

| cfg | start(B/P) | B_max | BΔmax | P_max | PΔmax |
|----:|:----------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 0.307 / 0.307 | 0.791 | +0.484 | 0.572 | +0.266 |
| 1 | 0.506 / 0.506 | 0.445 | -0.061 | 0.843 | +0.337 |
| 2 | 0.967 / 0.967 | 0.979 | +0.012 | 0.979 | +0.012 |
| 3 | 0.291 / 0.331 | 0.456 | +0.165 | 0.358 | +0.026 |
| 4 | 0.435 / 0.459 | 0.440 | +0.005 | 0.858 | +0.399 |
| 5 | 0.800 / 0.800 | 0.996 | +0.197 | 0.838 | +0.038 |
| 6 | 0.427 / 0.423 | 0.533 | +0.105 | 0.465 | +0.042 |
| 7 | 0.431 / 0.431 | 0.567 | +0.136 | 0.632 | +0.201 |

**Aggregate (Δmax = peak − start):**

| | mean peak | Δmax | frac of expert |
|---|---|---|---|
| Baseline | 0.651 | +0.130 | 36.3% |
| Proprio  | 0.693 | +0.165 | 37.1% |

Matched-pair Δmax (proprio − baseline): mean **+0.035**, proprio better on 4/8,
worse on 4/8. Proprio edges baseline but within noise at n=8; two configs
(cfg1, cfg4) drive the proprio advantage.

**Key context:** this is the first valid imitate comparison — two prior bugs
(single-step rollout via unrestored time_step; cross-process rope
nondeterminism) invalidated all earlier evals, and hold-on-converge was
freezing the pickers after ~6 steps (now removed). See the memory notes
imitate-eval-bugs-fixed / proprio-fusion-no-improvement.

GIFs were NOT recoverable (untracked, deleted from disk). The fixed straight-rope
goal run (`--goal-offset 75`) only completed 1 episode before interruption —
needs a re-run.
