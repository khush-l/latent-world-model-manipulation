# Baselines: ACT and SmolVLA

Goal: train direct-action imitation baselines on SoftGym RopeFlatten and
ClothFlatten, then compare them against LeWM on task performance, training
time, parameter count, and data requirements.

The baseline stack has three pieces:

1. Convert SoftGym trajectories to LeRobot format.
2. Fine-tune ACT or SmolVLA on the converted dataset.
3. Evaluate the trained policy in SoftGym through a host policy server plus
   SoftGym Docker rollout client.

## Data Policy

For the final report, train every method on the same split for a given
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

The VLA/BC baselines also need a separate high-success demo setting. ACT and
SmolVLA learn direct actions, so random or weak scripted data can produce low
training loss without task success. For the transfer/imitation baseline, collect
candidate scripted trajectories, filter them by SoftGym
`info_normalized_performance`, and train on the successful subset.

Recommended experiment labels:

- `random_debug`: random 5k75 data. Use only to validate conversion/training.
- `scripted_vla_150`: first 150 scripted demos. Useful historical baseline,
  but not strong enough for final claims.
- `success_filtered_v2`: filtered high-success demos. Use this for the stronger
  ACT/SmolVLA comparison.
- `mixed_final`: same random/scripted/oracle/perturbation split used by LeWM.

Do not mix extra teammate data into only one method unless the result is clearly
labeled as a different-data ablation.

## Success-Demo Filtering

Use `queue_success_demo_data.sh` to collect candidates, filter by final task
score, convert to HDF5, and convert to LeRobot:

```bash
cd /home/ubuntu/cs231n-project

# Rope: filter the already-good scripted set.
baselines/queue_success_demo_data.sh rope existing

# Cloth: first run a 50-episode pilot because cloth scripted success is harder.
baselines/queue_success_demo_data.sh cloth pilot

# After the pilot score distribution looks good, scale candidate collection.
baselines/queue_success_demo_data.sh cloth full
```

The filter writes:

| File | Purpose |
|---|---|
| `filter_summary.json` | Count selected episodes and score statistics |
| `filter_report.csv` | Per-episode score, policy, reward, and selected flag |
| copied `.npz` files | Successful trajectories for HDF5 conversion |

Direct filter usage:

```bash
./training/.venv/bin/python simulation/utils/filter_successful_trajectories.py \
  --input-dir simulation/data/trajectories/khush_scripted_rope_vla_150 \
  --output-dir simulation/data/trajectories/khush_rope_success_filtered_v2 \
  --metric info_normalized_performance \
  --threshold 0.80 \
  --top-k 300 \
  --min-episodes 100
```

Current finding: the first rope scripted set is good enough to filter directly
(137/150 demos passed `>= 0.8`). Cloth final-demo collection is paused after
merging origin/main because the shared collector now exposes RopeFlatten
`geometric`/`manipulate` policies; add a ClothFlatten policy there before
rerunning cloth collection.

## LeRobot Conversion

LeRobot stores low-dimensional signals in Parquet, image observations as
videos/images, and metadata such as task labels, stats, and episode boundaries.
The fields we emit are:

| LeRobot key | Source | Notes |
|---|---|---|
| `observation.images.front` | SoftGym `pixels[t]` | RGB uint8 image |
| `observation.state` | concat(`proprio[t]`, `state[t]`) | 8D picker proprio + 15D compact privileged state = 23D |
| `action` | SoftGym `action[t]` | 8D two-picker action |
| `task` | CLI instruction | `"straighten the rope"` or `"flatten the cloth"` |

ACT uses images/state/actions and ignores language. SmolVLA uses the same data
plus `task`.

Dry-run mapping check:

```bash
./training/.venv/bin/python baselines/softgym_hdf5_to_lerobot.py \
  --input training/data/khush_rope_success_filtered_v2.h5 \
  --output-root /home/ubuntu/lerobot_datasets/softgym_ropeflatten_success_filtered_v2 \
  --repo-id khush/softgym-ropeflatten-success-demos \
  --dry-run
```

Write the dataset:

```bash
./training/.venv/bin/python baselines/softgym_hdf5_to_lerobot.py \
  --input training/data/khush_rope_success_filtered_v2.h5 \
  --output-root /home/ubuntu/lerobot_datasets/softgym_ropeflatten_success_filtered_v2 \
  --repo-id khush/softgym-ropeflatten-success-demos \
  --instruction "straighten the rope"
```

## Training

Use the queue scripts for repeatable runs:

```bash
baselines/queue_act_scripted_vla.sh
baselines/queue_smolvla_scripted_vla.sh
baselines/queue_extend_baselines_to_10k.sh
```

For proper ACT tuning, verify the LeRobot optimizer config before launching a
long run. Some LeRobot policy presets override CLI optimizer values; if learning
rate or scheduler changes do not show up in logs, disable the policy training
preset or use a config-file override.

Recommended initial settings:

| Model | Dataset | Notes |
|---|---|---|
| ACT | `success_filtered_v2` | Start with chunk size 50 for 75-step SoftGym episodes |
| SmolVLA | `success_filtered_v2` | Uses the same dataset plus `task` language |
| ACT/SmolVLA | `mixed_final` | Final fair comparison with LeWM |

## Evaluation

Evaluation now uses a bridge:

1. The host Python environment loads the LeRobot policy in
   `baselines/policy_server.py`.
2. SoftGym Docker runs `simulation/utils/eval_lerobot_policy.py`.
3. The Docker rollout client sends observations to the host server and receives
   8D picker actions.

This avoids trying to install modern LeRobot inside the old CUDA 9 SoftGym
Docker image.

Example:

```bash
LEROBOT_POLICY_DEVICE=cuda \
./simulation/docker/eval-lerobot-policy.sh \
  --env-name RopeFlatten \
  --policy-checkpoint outputs/train/act_softgym_rope_scripted_vla_150_tuned_20k/checkpoints/020000/pretrained_model \
  --output-dir data/evals/act_rope_tuned_20k \
  --policy-name act_rope_tuned_20k \
  --num-episodes 10 \
  --horizon 75 \
  --img-size 128 \
  --success-threshold 0.8
```

Outputs:

| Path | Contents |
|---|---|
| `scores.csv` | Per-episode return, final normalized performance, success flag, mean/p90 policy latency |
| `summary.json` | Mean/std task metrics plus mean/p50/p90/p95/max policy latency |
| `videos/*.mp4` | Rollout videos for qualitative inspection |

Latency fields:

| Field family | Meaning |
|---|---|
| `policy_inference_latency_ms_*` | Host-side model inference time, measured inside `policy_server.py` with CUDA synchronization |
| `policy_roundtrip_latency_ms_*` | End-to-end SoftGym Docker request time, including image/state serialization, HTTP, host inference, and response parsing |

Use inference latency when comparing model compute directly. Use round-trip
latency when comparing the deployed bridge end-to-end against MPC wall-clock
latency.

Cleaned plot/table summaries live under `baselines/results/`, grouped by
experiment:

```bash
baselines/results/rope_geometric_5k_v3_20k/
baselines/results/rope_full_mixed_v1_25k/
baselines/results/presentation/
```

## Current Baseline Results

The first ACT/SmolVLA runs trained and evaluated successfully, but they did not
solve the tasks. Treat them as infrastructure baselines, not final model
quality.

| Run | Env | Episodes | Mean final normalized performance | Success rate |
|---|---|---:|---:|---:|
| `act_rope_3k` | RopeFlatten | 10 | 0.397 | 0.00 |
| `act_rope_tuned_20k` | RopeFlatten | 10 | 0.449 | 0.00 |
| `smolvla_rope_final` | RopeFlatten | 10 | 0.326 | 0.00 |
| `act_cloth_3k` | ClothFlatten | 10 | 0.132 | 0.00 |
| `act_cloth_tuned_20k` | ClothFlatten | 10 | 0.165 | 0.00 |
| `smolvla_cloth_final` | ClothFlatten | 10 | 0.101 | 0.00 |

Likely causes:

- The first scripted cloth demos were weak; final normalized performance topped
  out around 0.27.
- BC/VLA policies are sensitive to contact timing and action averaging.
- Training loss alone is not enough; use rollout success and videos.
- Stronger transfer baselines should train on filtered high-success demos or
  the final mixed dataset, not random-only data.


## Rope Dataset Ablations In Progress

The full/geometric baseline queue is meant to produce an early geometric-only
comparison before launching the longer full-mixed runs. This matters for the
LeWM + MPC project because these direct-action BC/VLA baselines are comparison
points, not the main method: they show how much task performance we get from
supervised action prediction without explicit latent world-model planning.

Current rope evals, all on 10 SoftGym episodes with success threshold 0.8:

| Run | Dataset / setting | Mean final normalized performance | Success rate | Notes |
|---|---|---:|---:|---|
| `act_rope_3k` | 150 scripted demos, 3k steps | 0.397 | 0.00 | historical infrastructure baseline |
| `act_rope_tuned_20k` | 150 scripted demos, 20k steps | 0.449 | 0.00 | improved but still no successes |
| `act_rope_success_filtered_v2_8k` | filtered high-success demos, 8k steps | 0.547 | 0.00 | strongest ACT rope result so far |
| `smolvla_rope_success_filtered_v2_20k` | filtered high-success demos, 20k steps | 0.524 | 0.10 | strongest SmolVLA rope result so far |
| `act_rope_geometric_5k_v3_20k` | geometric rope demos, 20k steps | 0.404 | 0.00 | completed; roughly infrastructure-baseline quality |
| `act_rope_full_mixed_v1_50k` | full mixed rope data, 50k steps | 0.392 | 0.00 | previous full-mixed ACT run |
| `smolvla_rope_full_mixed_v1_20k` | full mixed rope data, 20k steps | 0.537 | 0.10 | previous full-mixed SmolVLA run |
| `smolvla_rope_geometric_5k_v3_20k` | geometric rope demos, 20k steps | 0.646 | 0.30 | strongest direct-action baseline for the milestone slide |
| `act_rope_full_mixed_v1_25k` | full mixed rope data, 25k steps | 0.400 | 0.00 | completed after geometric run |
| `smolvla_rope_full_mixed_v1_25k` | full mixed rope data, 25k steps | 0.436 | 0.00 | completed; worse than geometric SmolVLA |

Interpretation: the direct-action baselines can get moderate normalized
performance on rope, but success remains sparse. That is useful context for
LeWM + MPC: the bar to beat is not just training loss, but rollout-level final
normalized performance, success rate, and latency under the same SoftGym eval
seeds. The geometric run is the strongest direct-action comparison so far: SmolVLA reaches 30% success, while ACT remains at 0%, at much lower latency.

## Metrics To Report

- Task success / normalized performance over 50-100 held-out SoftGym episodes.
- Training wall-clock time and GPU type.
- Number of training episodes and policy mix.
- Parameter count.
- Evaluation latency per action or action chunk.
- Qualitative rollout videos for both successes and failures.

For fairness, evaluate LeWM+MPC, ACT, and SmolVLA on the same SoftGym seeds and
same task metrics.

## OpenPI / pi0 Smoke Baseline

`baselines/pi0/` contains a proof-of-concept OpenPI bridge. It deliberately
reuses `simulation/utils/eval_lerobot_policy.py` so pi0 smoke outputs have the
same `scores.csv`, `summary.json`, normalized performance, success threshold,
videos, and latency fields as ACT and SmolVLA.

This is only a zero-shot compatibility smoke on the current T4 instance. OpenPI
inference is feasible here, but LoRA fine-tuning should run on a separate
24-48GB GPU instance. See `baselines/pi0/README.md` for setup and run commands.
