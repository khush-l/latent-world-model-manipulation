# Baselines: ACT and SmolVLA

Goal: train direct-action imitation baselines on the same SoftGym data used by
the LeWM experiments, then compare success rate, training time, parameter count,
and data requirements.

## Data Policy

For a clean comparison, train every method on the same train split for a given
experiment. The final LeWM, ACT, and SmolVLA comparison should use the same
per-task mixed dataset:

| Policy type | Target episodes per task | Purpose |
|---|---:|---|
| Random / exploratory | 2k-3k | Broad local dynamics and failure cases |
| Scripted heuristic | 4k-5k | Goal-relevant contacts and realistic task progress |
| Full-state oracle expert | 3k-4k | High-value states and near-expert behavior |
| Noisy state / perturbation | 0k-1k | Recovery from off-distribution states |

That gives roughly 9k-13k episodes per task, depending on how many
perturbation rollouts we include. Use the same episode IDs/splits for all
methods whenever possible.

- **Debug baseline:** train ACT and SmolVLA on Khush's random datasets for
  RopeFlatten and ClothFlatten. This tests the LeRobot conversion and training
  loop before the full mixed data is ready.
- **Final comparison:** train LeWM, ACT, and SmolVLA on the same mixed-policy
  dataset for each task.
- **Useful ablation:** random-only vs mixed-data for all three methods. This
  makes the data-quality story explicit instead of accidentally comparing
  different supervision.

Do not mix extra teammate data into only one method unless the result is clearly
labeled as a different-data ablation.

## LeRobot Conversion

LeRobot v3 stores low-dimensional signals in Parquet, image observations as
videos/images, and metadata such as task labels, stats, and episode boundaries.
The fields we emit are:

| LeRobot key | Source | Notes |
|---|---|---|
| `observation.images.front` | SoftGym `pixels[t]` | RGB uint8 image |
| `observation.state` | concat(`proprio[t]`, `state[t]`) | 8D picker proprio + 15D privileged compact state = 23D |
| `action` | SoftGym `action[t]` | 8D two-picker action |
| `task` | CLI instruction | `"straighten the rope"` or `"flatten the cloth"` |

ACT uses images/state/actions and ignores language. SmolVLA uses the same data
plus `task`.

Dry-run mapping check:

```bash
cd /home/ubuntu/cs231n-project
./training/.venv/bin/python baselines/softgym_hdf5_to_lerobot.py \
  --input training/data/khush_random_rope_5k75.h5 \
  --output-root /home/ubuntu/lerobot_datasets/softgym_ropeflatten_random_5k75 \
  --repo-id khush/softgym-ropeflatten-random-5k75 \
  --dry-run
```

After installing LeRobot in a baseline environment:

```bash
python baselines/softgym_hdf5_to_lerobot.py \
  --input training/data/khush_random_rope_5k75.h5 \
  --output-root /home/ubuntu/lerobot_datasets/softgym_ropeflatten_random_5k75 \
  --repo-id khush/softgym-ropeflatten-random-5k75
```

For ClothFlatten:

```bash
python baselines/softgym_hdf5_to_lerobot.py \
  --input training/data/khush_random_cloth_5k75.h5 \
  --output-root /home/ubuntu/lerobot_datasets/softgym_clothflatten_random_5k75 \
  --repo-id khush/softgym-clothflatten-random-5k75
```

## Suggested Training Order

1. Convert RopeFlatten first because `khush_random_rope_5k75.h5` is already
   complete.
2. Train ACT on RopeFlatten. Use this to debug action dimensions and chunking.
3. Convert ClothFlatten after `khush_random_cloth_5k75.h5` finishes.
4. Train ACT on ClothFlatten.
5. Fine-tune SmolVLA on the same converted datasets.

Recommended first ACT chunk size: `50`. The SoftGym episodes here have horizon
`75`, so chunk size `50` gives long-horizon supervision without exceeding the
episode.

## Metrics To Report

- Task success / normalized performance over 50-100 held-out SoftGym episodes.
- Training wall-clock time and GPU type.
- Number of training episodes and policy mix.
- Parameter count.
- Evaluation latency per action or action chunk.

For fairness, evaluate LeWM+MPC, ACT, and SmolVLA on the same SoftGym seeds and
same task metrics.
