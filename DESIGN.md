# Building a Latent World Model for Deformable Object Dynamics from Pixels

> **Milestone 2 — Technical Approach (May 22)**
> Solidify method and implementation plan: design choices, intuition, baselines, dataset usage, experimental setup, and evaluation.

## Research Question

Can a compact latent world model trained from pixels learn representations that are useful for manipulating objects with high-dimensional dynamics?

**Approach:** Train and evaluate a LeWorldModel-style action-conditioned JEPA on SoftGym deformable-object tasks.

---

## 1. Data Collection and Environment

### Challenges

- Data coverage will have the biggest impact on the world model.
- Random actions may not generate useful cloth/rope states.
- Expert-only data may not cover failure modes.
- Need exploratory + scripted trajectories covering contact, grasp/release, and goal-relevant deformations.

### SoftGym Setup

- [x] Original SoftGym Docker (CUDA 9 environment)
- [ ] Data collection scripts in `simulation/utils/`
- [ ] Start with **ClothFlatten** and **RopeFlatten** (both use the same 8D two-picker action format)
- [ ] Extend to **ClothFold** and **PourWater** once first two work

### Data Format

Full spec in [`DATA_FORMAT.md`](./DATA_FORMAT.md). Summary:

- [x] **Collector (in Docker):** per-episode NPZ, schema v3 — `pixels`, `action`, `proprio`, `state`, `reward`, `done`, `info_*`, `metadata_json`
- [x] **Trainer (out of Docker):** consolidated HDF5 via `simulation/utils/npz_to_hdf5.py` — flat columnar layout matching LeWM's `keys_to_load` (`pixels`, `action`, `proprio`, `state`)
- [ ] Mirror config YAML structure from LeWM repo (`references/le-wm-main/config/train/data/`)

#### Per-Step Signals (v3)

| Field | Shape | Source |
|---|---|---|
| `pixels` | `(T+1, H, W, 3) uint8` | env render at every step |
| `proprio` | `(T+1, 8) float32` | picker xyz + holding flags |
| `action` | `(T, 8) float32` | raw action (`[dx,dy,dz,grip] × 2`) |
| `state` | `(T+1, 15) float32` | privileged compact state (picker xyz + COM + bbox) |
| `reward` | `(T,) float32` | env reward |
| `done` | `(T,) bool` | terminal flag |
| `info_*` | `(T,) float32` | env scalar metrics (`performance`, `normalized_performance`, …) |
| `full_state_*` *(opt)* | particle arrays | behind `--save-full-state` for MPC `_set_state` |

### Data Volume

| Stage | Episodes | Purpose |
|---|---|---|
| Debug / smoke | 200–500 | First batch training, dataloader validation |
| Full training | 2k–5k per task | Final training runs |
| Stretch goal | 10k | If compute allows |

- [x] Episode length: `horizon=100` env steps (`collect_trajectories.py` default)
- [x] Env `action_repeat=8`, LeWM `frameskip=1` → one model step = 8 physics ticks. See `DATA_FORMAT.md` §2 for rationale.

### Trajectory Mix

| Type | Share | Purpose |
|---|---|---|
| Random / exploratory | 20–30% | Broad local dynamics, failure cases |
| Heuristic / scripted | 40–60% | Meaningful task progress, realistic interactions (e.g. grab both ends of rope and pull apart) |
| Expert / near-expert | 10–30% | High-value planning regions (CEM Dynamics Oracle with full-state solver from SoftAgents) |
| Perturbation rollouts | 5–10% | Recovery from off-distribution states |

#### Data Collection TODOs

- [ ] Random action collector
- [ ] Scripted/heuristic policy collector (geometric grab-and-pull primitives)
- [ ] CEM oracle expert collector (leverage SoftAgents)
- [ ] Perturbation rollout collector (start from messed-up states)
- [ ] Mix sampler that respects the percentage budget above

---

## 2. Training and Model Architecture

### Challenges

- Latent prediction ≠ task completion. LeWM plans by minimizing latent distance to a goal image; "close in embedding space" may not match task success unless the embedding preserves geometry, coverage, endpoints, fluid amount, etc.
- A single RGB frame does not reveal velocities, hidden folds, occluded particles, or contact state.

### LeWM Architecture

#### Vision Encoder — ViT-Tiny (~5M params)

| Hyperparameter | Value |
|---|---|
| Input resolution | 128 x 128 |
| Patch size | 16 x 16 |
| Patch tokens | 64 |
| Transformer layers | 12 |
| Attention heads | 3 |
| Hidden size `D` | 192 |

#### Predictor Transformer (~10M params)

| Hyperparameter | Value |
|---|---|
| Layers | 6 |
| Attention heads | 16 |
| Dropout | 10% |
| Temporal masking | Causal |
| Action conditioning | Adaptive LayerNorm (AdaLN) |

#### Training Objective

`Loss = Prediction MSE + SIGReg`

### Training Pipeline TODOs

- [ ] Convert smoke NPZ data → HDF5
- [ ] Run 2-epoch smoke training
- [ ] Verify dataloader shapes
- [ ] Verify `action_encoder.input_dim`
- [ ] Verify finite losses and non-collapsed embeddings
- [ ] Pilot training on 200–500 episodes
- [ ] Final training on full mixed-policy datasets

### Diagnostics to Log

| Metric | Purpose |
|---|---|
| Prediction loss | Core training signal |
| SIGReg loss | Embedding regularization health |
| Embedding mean / std | Detect collapse |
| Per-dimension latent std | Detect dead dimensions |
| Open-loop latent prediction MSE | Long-horizon stability |

### Ablations for Final Report

- [ ] Full LeWM baseline
- [ ] Trained on different data mixes (random-only, scripted-only, mixed, etc.)
- [ ] No SIGReg / very small SIGReg
- [ ] Custom loss components / architecture changes for SoftGym tasks

---

## 3. Evaluation and Experiments

### Challenges

- Stable-WorldModel does not natively support SoftGym → need to write env-specific MPC planning code.
- Need to train a decoder for latent visualization.
- MPC debugging inside SoftGym.

### Research Questions and Metrics

#### Q1: Did the world model learn basic physical representations and dynamics of object deformation?

| Type | Metric |
|---|---|
| Qualitative | Decoded images for latent state visualization |
| Qualitative | Violation-of-Expectation test |
| Quantitative | Latent Representation MSE |

#### Q2: Do the world model's latent representations translate to success in SoftGym planning tasks?

| Type | Metric |
|---|---|
| Qualitative | Task rollout videos |
| Quantitative | Task completion rate / reward vs. SoftGym-paper RL baselines |
| Quantitative | Planning latency |

### Evaluation TODOs

- [ ] Write SoftGym-specific MPC planner for the learned world model
- [ ] Train decoder for latent visualization
- [ ] Build Violation-of-Expectation evaluation harness
- [ ] Set up benchmarking against SoftGym-paper RL baselines
- [ ] Measure planning latency

---

## Milestone Checklist

- [x] SoftGym env set up (Docker, CUDA 9)
- [x] SoftAgents algorithms imported from original paper
- [x] Experimental trajectory collector scaffolded
- [ ] Finalize data format and config schema
- [ ] Implement all four trajectory collectors
- [ ] Smoke run (NPZ → HDF5 → 2-epoch train)
- [ ] Pilot training (200–500 episodes)
- [ ] Full training (2k–5k episodes per task)
- [ ] MPC planner integrated with SoftGym
- [ ] Decoder + Violation-of-Expectation eval
- [ ] Ablation sweep
- [ ] Final-report figures and videos
