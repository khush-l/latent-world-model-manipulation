# MPC Planning over LeWM Latents

> How to close the loop: take a trained LeWM and use it to actually solve
> SoftGym tasks via Model-Predictive Control in latent space.

## TL;DR

MPC over the trained world model is structurally simple. The learned encoder
+ predictor already gives us "predict the next latent given an action." MPC is
just **gradient-free search over action sequences** that minimizes a latent
goal-distance after rollout. Most heavy lifting is done — we need ~3 small
files of glue code to wire it up.

## Conceptual pipeline

```
  o_t (current obs)                o_g (goal obs)
     │                                  │
     ▼                                  ▼
  encoder ──▶ z_t                  encoder ──▶ z_g   (cached once)
                                       │
                                       ▼
     ┌────────────────────────────────────────────────────────┐
     │  CEM optimizer (Cross-Entropy Method) over actions:    │
     │                                                         │
     │    repeat I iterations:                                 │
     │      sample N action sequences a_{1:H} ~ N(μ, σ²)       │
     │      for each candidate (vectorized on GPU):            │
     │        roll out z_{t+1:t+H} via predictor               │
     │        cost = ‖z_{t+H} − z_g‖²                          │
     │      select top-K elites → refit μ, σ                   │
     └────────────────────────────────────────────────────────┘
                       │
                       ▼
            execute first action(s) in env → o_{t+1}
                       │
                       ▼
                  loop (replan)
```

Everything inside the CEM box happens **entirely in latent space on GPU**.
The env only gets called on the outer loop for execution.

## Components and status

| Component | Purpose | Status | LoC |
|---|---|---|---|
| `JEPA.encode(pixels)` | obs → latent z | ✅ `training/model.py:254` | — |
| `JEPA.predict(z, a)` | one-step predictor | ✅ `training/model.py:266` | — |
| `JEPA.rollout(z₀, action_seq)` | autoregressive latent rollout | ❌ **need to add** | ~50 |
| Goal-cost function | `‖ẑ_H − z_goal‖²` | ❌ trivial | ~10 |
| CEM solver | sample / rollout / elite / refit | ❌ **need to add** | ~80 |
| Goal sampler | pick a goal frame per task | ❌ need to add | ~30 |
| Env interface | reset, step, render | ✅ existing SoftGym envs | — |
| Eval loop | run N episodes, log perf + latency | ❌ need to add | ~150 |
| Replay-buffer goals | pick goal from dataset trajectory's terminal frame | ❌ trivial extension of sampler | ~20 |

**Total new code: ~340 lines** spread across 3–4 files. Heavy work
(learned dynamics, action encoder, AdaLN conditioning) is already trained
and frozen at eval time.

## Files to create

```
eval/
├── rollout.py          # JEPA.rollout() wrapper, mirrors references/le-wm-main/jepa.py:61
├── cem_planner.py      # CEM solver (300 samples × 30 iters × 5-step horizon)
├── sample_goals.py     # pick goal frames from held-out episodes
└── run_mpc.py          # closed-loop: encode → CEM → execute → repeat
```

## Algorithmic details (per paper, Appendix D)

| Knob | Value | Where it comes from |
|---|---|---|
| `num_samples` (N) | 300 | LeWM Sec. 4.1 |
| `n_iters` (I) | 30 (PushT) / 10 (other envs) | LeWM Sec. 4.1 |
| `topk` (elites) | 30 (top 10%) | LeWM Sec. 4.1 |
| `var_scale` | 1.0 | LeWM Sec. 4.1 |
| `plan_horizon` (H) | 5 | LeWM Sec. 4.1 |
| `receding_horizon` (K) | 5 (whole plan executed) | LeWM Sec. 4.1 |
| `action_block` | 5 (= frameskip) | — (we use 1 for SoftGym, so action_block=1) |
| Solver | CEM, fit isotropic Gaussian | LeWM Sec. 4.1 |

**Cost**: terminal latent MSE `‖ẑ_H − z_g‖²`. No per-step shaping, no learned
critic. (Reference also tries this and finds terminal cost works fine.)

## Action-space specifics for SoftGym

Our action is `[dx, dy, dz, grip] × 2` = 8 dims, with `dx,dy,dz ∈ [-0.01, 0.01]`
and `grip ∈ [0, 1]`. CEM samples from a Gaussian in this 8-dim space and
clips to action bounds.

| Sub-dim | Treatment |
|---|---|
| `dx, dy, dz` | Continuous Gaussian sample, clip to `[-0.01, 0.01]` |
| `grip` | Sample continuous, **interpret as `grip > 0.5 → 1, else 0`** when executing in env (env already does this internally) |

This is a minor deviation from continuous-action benchmarks: the grip dim is
effectively binary. CEM handles this fine (the Gaussian distribution finds
the mode), but worth noting in the report.

## Goal definition

For each held-out test episode `e`:

1. Pick a goal frame at step `t_goal` (e.g., `t_goal = horizon - 1` for
   final-state goals, or `t_goal = 25` ahead of start for sub-goal eval).
2. Encode once at episode start: `z_g = encoder(o_{t_goal})`. Cache.
3. Reset env to the same initial config as episode `e`.
4. Run MPC episode of length `eval_budget` (paper uses 50 steps).
5. Score: did final achieved performance reach the goal's performance? Log
   `info_normalized_performance` at terminal step.

## Goal-sampling strategies for RopeFlatten

Three useful goal types, in increasing difficulty:

| Goal type | How it's sampled | What it tests |
|---|---|---|
| **Flattened-rope goal** | Final frame of a high-perf scripted episode | Can the model flatten a curled rope? |
| **Stretched-to-config goal** | Final frame of a manipulate-policy episode | Can it move rope to an arbitrary configuration? |
| **Mid-trajectory goal** | Frame at step k=25 from same trajectory as start | Sub-goal reaching (closer to paper's PushT eval) |

The first is the headline metric. The third is the cleanest direct comparison
to LeWM-paper PushT results.

## Open questions / risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Trained latents may not encode rope-flatness usefully** — collapse, or encode visual texture instead of geometry | High | Check `emb_std` post-training; visualize via decoder (separate side-task); probe with linear regressor on rope endpoint coords |
| **CEM in 8D action space may need more samples than paper's 300** | Medium | Bump `num_samples` to 500–1000 if convergence looks poor; latency scales linearly |
| **5-step horizon may be too short for slow cloth/rope dynamics** | Medium | Bump `plan_horizon` to 8 or 10; latency scales linearly |
| **Discrete grip dim hurts CEM** | Low | Already-clipped at execution; can binarize before scoring; can switch to mixed-action sampler if needed |
| **MPC latency** | Low | Paper: 0.98 s on L40S. Expect ~1.2–1.5 s on L4. Sub-second on H100. |
| **Re-encoding overhead per replan** | Low | One ViT-Tiny forward, ~1 ms |

## What the planner can NOT do

- **Plan with friction/contact awareness beyond what's in the latent.** If
  the learned dynamics fail to capture grip-state transitions (e.g. "the
  picker dropped the rope"), the planner will happily execute plans that
  assume it's still gripping. This is a model-quality bug, not a planner bug.
- **Recover from goals that are unreachable from current state.** CEM will
  find the closest-achievable latent under the model's dynamics, which may
  visually look correct but not actually reach the goal.
- **Beat the SoftGym CEM-with-ground-truth baseline.** That has access to
  the true simulator dynamics; we're using a learned approximation. The
  paper's positioning is "competitive with foundation models at way lower
  compute," not "beats privileged-information baselines."

## Validation order

Once we have a trained model, validate the planner in this order:

1. **Smoke**: feed `z_t = encoder(o_g)` (current obs is the goal). Cost should
   be ~0. Any action sequence works. Confirms cost / rollout / env-loop wiring.
2. **One-step plan**: H=1, can the planner pick an action whose 1-step
   predicted latent matches a known target? Confirms predictor quality.
3. **Open-loop rollout**: H=5 (no replanning), pick the first plan, execute
   in env, measure achieved-vs-predicted final latent. Confirms model
   transfers to env.
4. **Closed-loop MPC** with random reset state. The headline result.

Each stage is a separate runnable script — fail-fast debugging.

## Implementation time estimate

| Task | Time |
|---|---|
| `eval/rollout.py` (port reference's rollout method) | 1 h |
| `eval/cem_planner.py` (Gaussian-sampling CEM) | 2 h |
| `eval/sample_goals.py` + `eval/run_mpc.py` | 4 h |
| Plumbing: arg parsing, output dirs, JSONL eval log | 1 h |
| First end-to-end MPC episode | ~half day debug |
| **Total** | **~1 day to working planner** |

Then 1 day to run baselines + ablations + collect numbers for the report.

## Are all the pieces there?

**Yes.** The training pipeline produces a model with the exact two
forward primitives MPC needs (`encode`, `predict`). The reference repo
has a working `JEPA.rollout` we can port nearly verbatim. CEM is a
well-known algorithm (~80 lines). SoftGym already gives us reset / step.
The 5000-episode datasets we're collecting now give us both training data
and goal frames.

The only meaningful uncertainty is whether the **trained latent dynamics**
are accurate enough on rope to make planning solve the task. That's a model-
quality question, answered by running the planner on a trained model — not
a piece of missing infrastructure.
