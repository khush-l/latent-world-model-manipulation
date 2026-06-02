# CS231N RopeFlatten / LeWM-JEPA — Results Summary

_Last updated: 2026-06-01_

## The system

LeWM-style action-conditioned JEPA world model (ViT encoder → CLS latent →
autoregressive predictor with action(+proprio) conditioning, SIGReg anti-collapse),
trained from pixels, with CEM-MPC planning over the learned latent. The MPC now
**faithfully matches the LeWM reference** (`stable-worldmodel`): receding-horizon
loop, terminal-latent MSE cost, sliding-window autoregressive rollout, CEM with no
candidate box-clipping and no min-std floor — verified line-by-line against
`solver/cem.py` and `wm/lewm/lewm.py`.

## MPC benchmark — headline numbers

All LeWM-matched (horizon 5, receding 5), gray env, seed 0.

| model | data | n=8 | n=50 (robust) | mean peak |
|---|---|---|---|---|
| d192 gray+smooth | no random | 12% | — | 0.55 |
| d384 / d384-deep | gray+smooth, no random | 38% | — | 0.65–0.68 |
| **v2** | original, **33% random** | 75%* | **38%** | 0.67 |
| mega d192 @250k | mega (20% random) | 38% | — | 0.67 |
| **mega + grip-endpoints cheat** | — | **88%** | — | **0.93** |

**\*Honesty correction:** v2's 75% was from only **8 configs**. The n=50 benchmark put
it at **38%**. At scale, v2 ≈ mega ≈ deep ≈ **~38%**; the small-n evals were noisy.
The durable takeaways below — not the 75% — are what hold up.

## Durable findings

1. **Action coverage is the dominant data lever.** Models with the `random` policy
   (~38%) clearly beat no-random gray+smooth (12%). But **composition matters, not
   volume**: the bigger mega (20% random) did *not* beat the smaller, more-random v2
   (33% random). Capacity (d384, d384-deep, more steps) did not move success.

2. **The bottleneck is the GRASP, not the flattening dynamics.** The grip-endpoints
   "cheat" (scripted expert grips the rope *ends*, then MPC takes over with grip held)
   jumps mega from 38% → **88%, peak 0.93** — the hard low-start configs all flatten
   (cfg0 0.23→1.00, cfg3 0.15→0.89, cfg6 0.43→0.99, cfg7 0.43→0.99). The model's
   dynamics *can* flatten the rope; it just can't reliably plan the grab maneuver.

3. **Predictor drift is the core model weakness.** One-step prediction is near-perfect
   (latent drift ≈ 3), but it **explodes ~60× at two steps (≈195)** — a
   `num_preds=1` train/plan mismatch (trained 1-step teacher-forced, MPC rolls 5–24
   steps autoregressively). The encoder is sharp on the start state but reconstructs
   intermediate (visited) states ~2.4× worse; endpoint localization is weak (CLS-token
   global latent). Decode MSE ≈ 0.030 (v2) – 0.046 (d384/deep); capacity does not
   sharpen it.

4. **Planning-side cleverness does not beat fixed-goal MSE.** Expert-demo latent
   subgoals (nearest 50%, time-indexed 38%) and a learned temporal-distance cost (12%)
   both *lost* to plain terminal MSE (~38–75% small-n). Root cause: any cost evaluated
   on the drifted/off-manifold predicted terminal latent is unreliable; MSE is robust
   because it requires the terminal to literally sit near the goal in latent space.
   **The objective is not the problem; the model (predictor fidelity) is.**

## The clean narrative for the writeup

> A faithful LeWM JEPA+MPC flattens rope at ~38% from pixels. Ablations show the
> limiter is **(a) off-policy action coverage in the training data** and **(b) the
> grasp/approach sub-problem** — given a solved grasp, the same model reaches **88%**.
> Capacity, action smoothness, dataset volume, expert-demo subgoals, and a learned
> temporal-distance cost do *not* help; the predictor's multi-step rollout fidelity
> (a 1-step-trained / multi-step-planned mismatch) is the root constraint.

## Deliverables on disk

- **Perf + GIFs**: `eval/runs/charts/{mpc_v2_lewm, mpc_v2_lewm_n50,
  mpc_mega_d192_100k, mpc_mega_d192_250k, mpc_mega_gripcheat,
  mpc_v2_subgoals, mpc_v2_subgoals_time, mpc_v2_tempdist}/summary_k5.json`
- **Latent decode PNGs**: `eval/runs/decode_mega_d192_250k_paper.png` (+ v2, d384, deep)
- **MPC-imagination GIF** (every step decoded, imagined | reality | goal):
  `eval/runs/imagine_mpc_cfg0_mega250.gif`
- **Grip-cheat success GIFs**: `eval/runs/charts/mpc_mega_gripcheat/gifs/`

## Tooling added this session (ablation flags off by default)

- LeWM-faithful `eval/mpc_runner.py`, `eval/solvers.py`, `eval/lewm_cost.py`
  (config-driven arch build; `--cost {mse,tempdist}`; `--subgoals/--subgoal-mode`;
  `--grip-endpoints`).
- `eval/imagine_mpc.py` — visualize the imagined latent rollout during MPC
  (static grid + per-step `--gif-out`; `--scan-episodes` to showcase a real success).
- `training/train_temporal_distance.py` — learned cost-to-go head.
- `simulation/utils/env_server.py` `grip_endpoints` command (+ `env_client` method).
- `simulation/data/rope/rope_mega_dataset.h5` — merged 25k-episode mega dataset
  (random + smooth geom/manip + noisy geom/manip), gray, proprio.

## Open / in flight

- n=50 v2 benchmark: **done (38%)**.
- mega **d384 / d384-deep** capacity sweep on the H100 (capacity ablation on
  random-inclusive data) — the one remaining open question.
- Most-principled forward fix: **rollout / scheduled-sampling training** of the
  predictor (supervise on its own multi-step rollouts) to cut the 3→195 drift, which
  attacks the actual bottleneck and might also make a learned cost viable.
