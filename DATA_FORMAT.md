# Data Format

Source of truth for trajectory data in this project. Two formats:

- **NPZ (per episode)** — what `simulation/utils/collect_trajectories.py` writes inside the SoftGym Docker (numpy-only, no h5py).
- **HDF5 (consolidated)** — what `simulation/utils/npz_to_hdf5.py` produces for LeWM training in the modern ML environment.

The HDF5 layout mirrors the columns LeWM consumes (`pixels`, `action`, `proprio`, `state`) so it loads with `stable_worldmodel.data.load_dataset`. The NPZ layout uses the same field names so conversion is a straight stack-along-episode operation. 

---

## 1. Time indexing convention

For an episode with `T` env steps (one step = `action_repeat` physics ticks):

- `T + 1` observations: `pixels[0]` is the reset frame; `pixels[t+1]` is the frame after applying `action[t]`.
- `T` actions, rewards, dones — defined for `t ∈ [0, T-1]`.
- Per-frame signals (`proprio`, `state`) are observation-aligned → length `T + 1`.
- Per-step signals (`action`, `reward`, `done`, `info_*`) are action-aligned → length `T`.

When flattening to HDF5 (one row per timestep), per-step columns are **right-padded with NaN/False** at the terminal frame so all columns share length `T + 1`. LeWM handles this via `torch.nan_to_num(batch["action"], 0.0)` (`references/le-wm-main/train.py:25`).

```
t:        0     1     2     ...    T-1    T
pixels:   o0    o1    o2    ...    o_{T-1}  o_T       (length T+1)
proprio:  p0    p1    p2    ...    p_{T-1}  p_T       (length T+1)
state:    s0    s1    s2    ...    s_{T-1}  s_T       (length T+1)
action:   a0    a1    a2    ...    a_{T-1}  NaN       (length T+1 after pad)
reward:   r0    r1    r2    ...    r_{T-1}  NaN       (length T+1 after pad)
done:     d0    d1    d2    ...    d_{T-1}  False     (length T+1 after pad)
```

Action `a_t` is the action applied **after** observing `o_t` to produce `o_{t+1}`.

---

## 2. Frameskip vs. action_repeat

Two compounding mechanisms — pin them once, here.

| Layer | Variable | Value | Meaning |
|---|---|---|---|
| SoftGym env | `action_repeat` | **8** | Physics ticks per `env.step()` call. One stored action = 8 ticks. |
| LeWM dataloader | `frameskip` | **1** | Number of stored rows aggregated into one model step. |

**One model step = 8 physics ticks ≈ one picker macro-motion.** Rationale:

- LeWM's PushT reference uses dense per-tick actions with `frameskip=5`. Our env's `action_repeat=8` already aggregates ticks at collection time, so `frameskip=1` keeps the same effective model timescale without inflating dataset size or the action encoder input dim.
- With `frameskip=1`, `action_encoder.input_dim = 1 * 8 = 8` (the 2-picker action). Matches what the picker controller sees.
- Going to `frameskip>1` later (coarser model steps) is a config flip — no recollection needed.

---

## 3. NPZ schema (per episode)

Filename: `{env_name}_{episode_idx:06d}.npz` in `simulation/data/<output-dir>/`. Compressed by default.

### Required arrays

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `pixels` | `(T+1, H, W, 3)` | `uint8` | HWC, RGB, no normalization. `H = W = img_size` (default 128). |
| `action` | `(T, 8)` | `float32` | **Raw** action values in the env's native range. Order: `[dx, dy, dz, grip] × 2 pickers`. Bounds in metadata. |
| `proprio` | `(T+1, 8)` | `float32` | `[picker_xyz × 2, picker_holding × 2]`. `picker_holding ∈ {0.0, 1.0}` indicates whether each picker currently holds a particle. |
| `state` | `(T+1, 15)` | `float32` | Compact privileged state (fixed-dim). See §5. |
| `reward` | `(T,)` | `float32` | Env reward at each step. |
| `done` | `(T,)` | `bool` | Terminal flag per step. |
| `metadata_json` | scalar | string | JSON blob — env config and provenance. See §6. |

### Optional arrays

| Key | Shape | Dtype | When |
|---|---|---|---|
| `depth` | `(T+1, H, W)` | `float32` | `--save-depth` |
| `full_state_particle_pos` | `(T+1, N_part, 4)` | `float32` | `--save-full-state`. Variable `N_part` per task/config. |
| `full_state_particle_vel` | `(T+1, N_part, 4)` | `float32` | `--save-full-state`. |
| `full_state_shape_pos` | `(T+1, N_picker, 14)` | `float32` | `--save-full-state`. |
| `full_state_phase` | `(T+1, N_part)` | `int32` | `--save-full-state`. |
| `info_<scalar_key>` | `(T,)` | `float32` | Always written for every scalar in env `info` dict (e.g. `info_performance`, `info_normalized_performance`). |

**`action_normalized` is removed in v3.** LeWM z-score normalizes actions from per-column stats; storing two action representations is redundant. Anyone needing normalized actions computes them at load time from `metadata_json.action_space_low` and `action_space_high`.

---

## 4. HDF5 schema (consolidated dataset)

Filename: `<dataset_name>.h5` (e.g. `clothflatten_mixed_v3.h5`). One file per dataset = one task × one policy mix.

### Root datasets (flat columnar; one row = one timestep)

For each row index `i ∈ [0, N_total)` where `N_total = Σ (T_ep + 1)` across all episodes:

| Dataset | Shape | Dtype | Chunking |
|---|---|---|---|
| `/pixels` | `(N_total, H, W, 3)` | `uint8` | `(1, H, W, 3)` |
| `/action` | `(N_total, 8)` | `float32` | `(256, 8)`; NaN at terminal frames |
| `/proprio` | `(N_total, 8)` | `float32` | `(256, 8)` |
| `/state` | `(N_total, 15)` | `float32` | `(256, 15)` |
| `/reward` | `(N_total,)` | `float32` | `(256,)`; NaN at terminal frames |
| `/done` | `(N_total,)` | `uint8` (bool) | `(256,)` |
| `/episode_idx` | `(N_total,)` | `int32` | `(256,)` |
| `/step_idx` | `(N_total,)` | `int32` | `(256,)`; `0..T` within each episode |
| `/policy_id` | `(N_total,)` | `int8` | `(256,)`; see §6 |

Per-info scalars are also promoted to root datasets `/info_<key>` with NaN padding at terminal rows.

### Attributes on `/`

- `schema_version` — `3`
- `env_name` — `"ClothFlatten"` etc.
- `num_picker` — `2`
- `action_dim` — `8`
- `proprio_dim` — `8`
- `state_dim` — `15`
- `action_repeat` — `8`
- `horizon` — `100`
- `img_size` — `128`
- `policy_id_map` — JSON string mapping `{0: "random", 1: "scripted", 2: "cem_expert", 3: "perturbation"}`
- `action_space_low`, `action_space_high` — `(8,) float32` for de-normalization

### Group `/episodes`

One subgroup per source episode, e.g. `/episodes/000042/` with attrs copied from that NPZ's `metadata_json` (`policy`, `seed`, `config_id`, `current_config`, `created_at`). Used for traceability and to look up the source NPZ.

### Group `/full_state` *(optional)*

When `--save-full-state` was set on collection and the converter is invoked with `--include-full-state`, particle/shape/phase arrays go in **per-episode** subgroups (`/full_state/000042/particle_pos`, etc) because `N_part` varies per config. Not part of the flat columnar table — only used for MPC `_set_state` / `_set_goal_state` at eval time.

### Stats (for LeWM normalizer warm-cache)

Top-level group `/stats` with `mean` and `std` arrays for each normalizable column (`action`, `proprio`, `state`). LeWM recomputes these at load time, but caching them lets us verify the dataloader's normalizer agrees with collection-time stats.

---

## 5. Compact `state` layout (15D)

Privileged, fixed-dim, computed at each `pixels` step:

| Index | Field | Computed from |
|---|---|---|
| `0..2` | `picker_0_xyz` | `pyflex.get_shape_states().reshape(-1, 14)[0, :3]` |
| `3..5` | `picker_1_xyz` | same, picker 1 |
| `6..8` | `particle_com_xyz` | `particle_pos[:, :3].mean(axis=0)` |
| `9..11` | `particle_bbox_min` | `particle_pos[:, :3].min(axis=0)` |
| `12..14` | `particle_bbox_max` | `particle_pos[:, :3].max(axis=0)` |

Picker grip state lives in `proprio` (not duplicated here). Per-task scalars (`covered_area` for ClothFlatten, `end_point_distance` for RopeFlatten) are already in `info_*`.

**Why fixed-dim:** LeWM applies a z-score normalizer to `state` (`references/le-wm-main/utils.py:25-32`). It needs `dataset.get_col_data("state")` to stack into a single 2D array across all rows. Variable-length particle arrays go in `/full_state/<ep>/` instead.

**Why 15D:** small enough to be cheap, large enough to capture per-step geometry that's invisible in a single frame (depth, occluded folds). Verified against ClothFlatten and RopeFlatten — both expose `pyflex.get_positions().reshape(-1, 4)`.

---

## 6. `metadata_json` contents (per NPZ)

JSON object with at least:

```json
{
  "schema_version": 3,
  "env_name": "ClothFlatten",
  "episode_idx": 7,
  "seed": 0,
  "policy": "random",
  "created_at": "2026-05-21T14:33:12-0700",
  "env_kwargs": { "...": "..." },
  "config_id": 4,
  "current_config": { "...": "..." },
  "action_space_low": [-0.01, -0.01, -0.01, 0.0, -0.01, -0.01, -0.01, 0.0],
  "action_space_high": [0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 1.0],
  "action_repeat": 8,
  "horizon": 100,
  "img_size": 128,
  "num_picker": 2,
  "action_dim": 8,
  "proprio_dim": 8,
  "state_dim": 15
}
```

`policy` is one of `random`, `scripted`, `cem_expert`, `perturbation`. The HDF5 converter uses this to populate `/policy_id` per the `policy_id_map` attr.

---

## 7. Manifest

`simulation/data/<output-dir>/manifest.jsonl` — one JSON line per saved episode, containing the same fields as `metadata_json` plus `"path": "<relative npz path>"`. Append-only. The converter reads this to discover episodes; the viewer falls back to the most recent entry when `--input` is omitted.

---

## 8. Validation rules

The collector and converter both enforce:

1. `pixels.shape[0] == proprio.shape[0] == state.shape[0] == action.shape[0] + 1`.
2. `pixels.dtype == uint8` and `pixels.shape[1] == pixels.shape[2] == img_size`.
3. `action.shape[1] == 4 * num_picker`.
4. `proprio.shape[1] == 4 * num_picker` (3 xyz + 1 grip per picker).
5. `state.shape[1] == 15`.
6. `reward.shape[0] == done.shape[0] == action.shape[0]`.
7. `metadata_json.schema_version == 3`.
8. Non-finite values are only allowed in `action`/`reward` at the terminal-padded row (HDF5 only, not NPZ).

Collector fails fast on violations.

---

## 9. Storage budget

Approximate per-episode disk usage at the default 128×128, horizon=100:

| Component | Size |
|---|---|
| `pixels` (101 × 128 × 128 × 3 uint8, compressed) | ~1.5 MB |
| `action`, `proprio`, `state`, `reward`, `done` | ~25 KB |
| `info_*` | ~1 KB |
| `metadata_json` | ~2 KB |
| `depth` (optional) | ~5 MB |
| `full_state_*` (optional, ~6k particles) | ~5 MB |

Defaults → ~1.5 MB/episode. Full pipeline (5k episodes) → ~7.5 GB. With `--save-full-state` add ~25 GB.

Schema Version 3
