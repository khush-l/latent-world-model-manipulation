# Baseline Results

This directory keeps the baseline outputs grouped by experiment so the current
results are easy to compare without digging through logs or temporary slide
drafts.

## Experiments

| Folder | Dataset / run | Contents |
|---|---|---|
| `rope_geometric_5k_v3_20k/` | ACT and SmolVLA trained on geometric rope demos, 20k steps | Per-episode scores, summary tables, rollout comparison plots, training charts |
| `rope_full_mixed_v1_25k/` | ACT and SmolVLA trained on full mixed rope data, 25k steps | Per-episode scores, summary tables, rollout comparison plots, training charts |
| `presentation/` | Final milestone-ready baseline table/figure | Publication-style PDF/PNG/SVG artifacts |

Older exploratory plots and logs are intentionally not part of this cleaned
results tree. Raw rollout logs live under `baselines/logs/` locally and are
ignored by git.
