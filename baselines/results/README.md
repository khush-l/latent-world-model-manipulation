# Baseline Results

This folder keeps the small result files and figures from the baseline runs.
Raw rollout folders, videos, checkpoints, and logs were left local.

## Final Same-Machine Rope Run

These are the fixed 50-episode RopeFlatten results used for the final comparison.
All methods used the same cached SoftGym configs and a 75-step horizon.

| Method | Episodes | Mean final perf. | Latency note |
|---|---:|---:|---|
| LeWM+CEM | 50 | 0.848 | 821.6 ms CEM replan |
| ACT | 50 | 0.542 | 25.7 ms action chunk |
| SmolVLA | 50 | 0.444 | 282.6 ms action chunk |
| pi0 LoRA | 50 | 0.432 | 308.9 ms action chunk |

Figures and table live in `paper_comparison/`:

- `same_machine_accuracy.png`
- `same_machine_latency.png`
- `same_machine_comparison_table.csv`
- PDF copies of the two figures

## Earlier Baseline Runs

These runs were useful for debugging the direct-action baseline setup, even
though not all of them ended up in the paper.

| Run | Episodes | Mean final perf. | Notes |
|---|---:|---:|---|
| ACT, geometric rope demos | 10 | 0.404 | early LeRobot baseline |
| SmolVLA, geometric rope demos | 10 | 0.646 | strongest early direct-action result |
| ACT, full mixed rope data | 10 | 0.400 | mixed data did not help ACT here |
| SmolVLA, full mixed rope data | 10 | 0.436 | lower than geometric-only run |
| pi0 LoRA, geometric rope demos | 10 | 0.460 | first OpenPI/pi0 smoke-to-eval run |

Kept figures include the comparison plots under `rope_geometric_5k_v3_20k/`,
`rope_full_mixed_v1_25k/`, `pi0_rope_lora_g5/`, and `presentation/`.

## Checkpoint Sweep

The CEM-final checkpoint sweep helped pick which ACT/SmolVLA checkpoints to use
for the longer same-machine eval.

| Family | Best step in sweep | Episodes | Mean final perf. |
|---|---:|---:|---:|
| ACT | 10000 | 8 | 0.548 |
| SmolVLA | 45000 | 8 | 0.493 |

The full small sweep table is in `rope_cem_final_checkpoint_eval/`.

