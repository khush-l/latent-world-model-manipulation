# Simulation Data Collection

This folder is intentionally isolated inside `simulation/`. Use it to collect
SoftGym/PyFlex trajectories, then train models from the saved files in a
separate ML environment.

## Usage

From `simulation/`, activate and prepare SoftGym first:

```bash
conda activate softgym
. ./prepare_1.0.sh
python data_collection/collect_trajectories.py --env-name ClothFlatten --num-episodes 10 --num-variations 10 --img-size 128
```

The script writes one compressed `.npz` per episode plus `manifest.jsonl`.
Each episode contains:

- `rgb`: uint8 frames with shape `[T + 1, H, W, 3]`
- `action_normalized`: actions in `[-1, 1]`
- `action_raw`: actions in the simulator's native action space
- `reward`, `done`
- `info_*`: scalar metrics exposed by the environment, such as `performance`
  and `normalized_performance`
- `metadata_json`: environment name, kwargs, action bounds, config summary,
  seed, horizon, and schema version

Optional flags:

```bash
python data_collection/collect_trajectories.py \
  --env-name RopeFlatten \
  --num-episodes 100 \
  --num-variations 100 \
  --img-size 128 \
  --save-state \
  --save-depth
```

`--save-state` stores particle positions, velocities, shape states, and phases
when the environment exposes them. This is useful for latent probing and
debugging, but it can make datasets much larger.

## Viewing A Trajectory

Generate a static HTML viewer for a collected episode:

```bash
python data_collection/view_trajectory.py --input data/trajectories/ClothFlatten_000000.npz
```

Or through the local wrapper:

```bash
data_collection/view-trajectory.sh --input data/trajectories/ClothFlatten_000000.npz
```

If `--input` is omitted, the viewer uses the most recently modified trajectory
in `data/trajectories/`. The output is written to `data/viewer/<episode>/` and
contains `index.html`, exported frame images, and `summary.json`.

Use `--stride` or `--max-frames` for large episodes:

```bash
python data_collection/view_trajectory.py --stride 2 --max-frames 80
```

## Notes

- The script uses the existing Gym-style SoftGym API: `obs, reward, done, info
  = env.step(action)`.
- One collected transition corresponds to one SoftGym step, not one raw physics
  tick. SoftGym internally repeats each action according to `action_repeat`.
- Defaults avoid generating or saving the original 1000 cached initial states.
  Use `--use-cached-states` and `--save-cached-states` when you deliberately
  want persistent initial-state caches.
