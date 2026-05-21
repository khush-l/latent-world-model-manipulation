# Simulation Data Collection

This folder is intentionally isolated inside `simulation/`. Use it to collect
SoftGym/PyFlex trajectories (NPZ), then convert and train from a separate ML
environment (HDF5).

See [`../../DATA_FORMAT.md`](../../DATA_FORMAT.md) for the authoritative schema.

## Collect (inside the SoftGym Docker)

From `simulation/`, activate and prepare SoftGym first:

```bash
conda activate softgym
. ./prepare_1.0.sh
python utils/collect_trajectories.py \
  --env-name ClothFlatten \
  --num-episodes 10 --num-variations 10 \
  --img-size 128
```

Writes one compressed `.npz` per episode plus `manifest.jsonl` to
`simulation/data/trajectories/` by default.

### Required arrays (v3 schema)

| Key | Shape | Notes |
|---|---|---|
| `pixels` | `(T+1, H, W, 3) uint8` | reset frame + post-step frames |
| `action` | `(T, 8) float32` | raw action; `[dx,dy,dz,grip] × 2 pickers` |
| `proprio` | `(T+1, 8) float32` | picker xyz + holding flag per picker |
| `state` | `(T+1, 15) float32` | privileged compact state |
| `reward`, `done` | `(T,)` | per-step |
| `info_*` | `(T,) float32` | scalars from env `info` dict (`performance`, `normalized_performance`, …) |
| `metadata_json` | string | env config and provenance |

### Optional arrays

```bash
python utils/collect_trajectories.py \
  --env-name RopeFlatten \
  --num-episodes 100 --num-variations 100 \
  --img-size 128 \
  --save-depth \
  --save-full-state
```

- `--save-depth` — adds `depth` (T+1, H, W) float32
- `--save-full-state` — adds `full_state_particle_pos`, `full_state_particle_vel`, `full_state_shape_pos`, `full_state_phase` for MPC `_set_state` at eval time

## Convert NPZ → HDF5 (outside the Docker)

Run this in a modern Python env with `h5py`:

```bash
python utils/npz_to_hdf5.py \
  --input-dir simulation/data/trajectories \
  --output simulation/data/clothflatten_random_v3.h5
```

Produces a single file with flat columnar root datasets (`/pixels`, `/action`,
`/proprio`, `/state`, `/reward`, `/done`, `/episode_idx`, `/step_idx`,
`/policy_id`, `/info_*`) plus `/episodes/<ep>/` metadata groups and `/stats/`
warm-cached normalizer stats. See `DATA_FORMAT.md` §4.

Add `--include-full-state` to copy variable-length particle arrays under
`/full_state/<ep>/`.

## View a trajectory

```bash
python utils/view_trajectory.py --input data/trajectories/ClothFlatten_000000.npz
# or
utils/view-trajectory.sh --input data/trajectories/ClothFlatten_000000.npz
```

If `--input` is omitted, the viewer uses the most recently modified NPZ in
`data/trajectories/`. The output is written to `data/viewer/<episode>/`
(`index.html`, `frames/`, `summary.json`).

Use `--stride` or `--max-frames` for large episodes:

```bash
python utils/view_trajectory.py --stride 2 --max-frames 80
```

## Notes

- One collected transition = one SoftGym step (which internally runs
  `action_repeat` physics ticks — currently 8). See `DATA_FORMAT.md` §2.
- Defaults avoid generating or saving the original 1000 cached initial states.
  Use `--use-cached-states` and `--save-cached-states` to manage caches.
- The collector enforces v3 length invariants and fails fast on mismatches.
