# H100 Direct-Action Baseline Handoff

This is the handoff for running the direct-action baselines on one shared H100 so the comparison is fair. The AWS `g5.2xlarge` run is useful for proving the π0 LoRA pipeline and producing a checkpoint, but final latency should be reported from the same H100 machine used for ACT, SmolVLA, π0 LoRA, and LeWM+MPC.

## TL;DR For Teammate

1. Clone this branch.
2. Install SoftGym/Docker dependencies as usual for this repo.
3. Install OpenPI with `baselines/pi0/bootstrap_openpi.sh /home/ubuntu/openpi`.
4. Put the converted π0 dataset at `/home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi`, or put the raw HDF5 at `training/data/ropeflatten_geometric_5k_v3.h5` and run conversion.
5. Train π0 LoRA with `baselines/pi0/start_pi0_lora_training_tmux.sh`, or use the copied checkpoint.
6. Run all direct-action evals with `baselines/start_h100_direct_action_evals_tmux.sh` after setting checkpoint paths.

## What Should Go In Git

Commit these scripts/docs because they are small and reproducible:

```text
baselines/H100_DIRECT_ACTION_EVAL_README.md
baselines/run_h100_direct_action_evals.sh
baselines/start_h100_direct_action_evals_tmux.sh
baselines/run_h100_direct_action_retrain.sh
baselines/start_h100_direct_action_retrain_tmux.sh
baselines/plot_pi0_rope_lora_charts.py
baselines/pi0/README.md
baselines/pi0/bootstrap_openpi.sh
baselines/pi0/check_pi0_data_paths.sh
baselines/pi0/convert_pi0_softgym_rope_dataset.sh
baselines/pi0/eval_openpi_softgym_smoke.sh
baselines/pi0/eval_pi0_lora_after_training.sh
baselines/pi0/eval_pi0_lora_softgym_rope.sh
baselines/pi0/install_openpi_softgym_config.sh
baselines/pi0/openpi_softgym_policy.py
baselines/pi0/openpi_softgym_server.py
baselines/pi0/rsync_pi0_data_from_old_instance.sh
baselines/pi0/softgym_hdf5_to_openpi_lerobot.py
baselines/pi0/start_pi0_lora_eval_after_training_tmux.sh
baselines/pi0/start_pi0_lora_training_tmux.sh
baselines/pi0/train_pi0_lora_softgym_rope.sh
baselines/pi0/upload_pi0_handoff_to_drive.sh
baselines/pi0/verify_pi0_handoff.sh
```

Useful generated results can also be committed if you want the branch to show the AWS proof run:

```text
baselines/results/pi0_rope_lora_g5/pi0_direct_action_comparison.png
baselines/results/pi0_rope_lora_g5/pi0_direct_action_comparison.pdf
baselines/results/pi0_rope_lora_g5/pi0_rope_lora_dashboard.png
baselines/results/pi0_rope_lora_g5/pi0_rope_lora_dashboard.pdf
baselines/results/pi0_rope_lora_g5/pi0_direct_action_report_table.csv
```

Do not commit large datasets, checkpoints, videos, Docker build output, or raw eval folders unless your repo policy explicitly allows large artifacts.

## What Should Go In Drive Or External Storage

Put these in the shared Drive/rclone folder or another external store. Use a flat layout so the important artifacts are easy to find:

```text
cs231n-pi0-handoff/
  README_PI0_HANDOFF.txt
  softgym-rope-openpi/
  pi0_lora_checkpoint_19999/
  ropeflatten_geometric_5k_v3.h5   # optional if already shared elsewhere
```

The easiest teammate path is to share `softgym-rope-openpi/` and `pi0_lora_checkpoint_19999/` at the top level. The raw HDF5 is useful as source of truth, but does not need to be duplicated if it is already in Drive.

Expected local paths on the H100:

```text
/home/ubuntu/cs231n-project/training/data/ropeflatten_geometric_5k_v3.h5
/home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi
/home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999
```

If using rclone, copy data into those paths before running train/eval:

```bash
mkdir -p /home/ubuntu/cs231n-project/training/data
mkdir -p /home/ubuntu/lerobot_datasets/khush
mkdir -p /home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5

# Only needed if the HDF5 is not already present.
rclone copy <remote>:cs231n-pi0-handoff/ropeflatten_geometric_5k_v3.h5 /home/ubuntu/cs231n-project/training/data/

rclone copy <remote>:cs231n-pi0-handoff/softgym-rope-openpi /home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi
rclone copy <remote>:cs231n-pi0-handoff/pi0_lora_checkpoint_19999 /home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999
```

To upload the flat Drive layout from this instance:

```bash
cd /home/ubuntu/cs231n-project
REMOTE=gdrive:cs231n-pi0-handoff baselines/pi0/upload_pi0_handoff_to_drive.sh
```

By default that uploads only the converted dataset and checkpoint. Set `INCLUDE_RAW_H5=1` if the HDF5 also needs to be uploaded.

## One-Time Setup On H100

```bash
cd /home/ubuntu
git clone https://github.com/nikunjparasar/cs231n-project.git
cd cs231n-project
git checkout khush-baseline-results

baselines/pi0/bootstrap_openpi.sh /home/ubuntu/openpi
baselines/pi0/install_openpi_softgym_config.sh /home/ubuntu/openpi
```

Verify the handoff inputs:

```bash
cd /home/ubuntu/cs231n-project
baselines/pi0/verify_pi0_handoff.sh
```

On a fresh H100, this verifier may report missing data/checkpoints until Drive artifacts are copied. That is fine; missing repo scripts are not fine.

## Dataset Conversion

If the converted dataset is already copied to `/home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi`, skip this step.

If only the raw HDF5 is available:

```bash
cd /home/ubuntu/cs231n-project
tmux new-session -d -s pi0_convert \
  "cd /home/ubuntu/cs231n-project && OVERWRITE=1 baselines/pi0/convert_pi0_softgym_rope_dataset.sh"
```

Expected converted dataset:

```text
5000 episodes / 375000 frames
/home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi
OpenPI repo id in config: khush/softgym-rope-openpi
```

## Optional: Retrain ACT / SmolVLA

Most likely, use the existing ACT and SmolVLA checkpoints and skip this section. If fresh same-machine training is desired, run:

```bash
cd /home/ubuntu/cs231n-project
baselines/start_h100_direct_action_retrain_tmux.sh
```

Defaults target the geometric rope dataset:

```text
DATASET_ROOT=/home/ubuntu/lerobot_datasets/softgym_ropeflatten_geometric_5k_v3
DATASET_REPO=khush/softgym-ropeflatten-geometric-5k-v3
ACT_STEPS=20000
SMOLVLA_STEPS=20000
```

The retrain script prints the resulting `ACT_CHECKPOINT` and `SMOLVLA_CHECKPOINT` exports at the end. Use those when running `baselines/start_h100_direct_action_evals_tmux.sh`.

## Train π0 LoRA

If using the AWS-trained checkpoint, skip training and use that checkpoint for eval. If retraining on the H100:

```bash
cd /home/ubuntu/cs231n-project
OPENPI_ROOT=/home/ubuntu/openpi \
HF_LEROBOT_HOME=/home/ubuntu/lerobot_datasets \
OPENPI_EXP_NAME=softgym_rope_lora_h100 \
OPENPI_BATCH_SIZE=4 \
OPENPI_NUM_WORKERS=4 \
baselines/pi0/start_pi0_lora_training_tmux.sh
```

The installed config reads `OPENPI_BATCH_SIZE`, `OPENPI_NUM_WORKERS`, and `OPENPI_NUM_TRAIN_STEPS` at runtime. The default handoff values are intended for H100 retraining; lower them only if the machine has less memory. 

Monitor:

```bash
tmux attach -t =pi0_lora
tail -f /home/ubuntu/cs231n-project/baselines/logs/pi0_lora_train.log
```

The training script computes OpenPI norm stats if missing, disables W&B, resumes if checkpoint steps already exist, and uses the installed `pi05_softgym_rope_lora` config.

## Run Same-Hardware Direct-Action Evals

Set all checkpoints explicitly:

```bash
export ACT_CHECKPOINT=/path/to/act/pretrained_model
export SMOLVLA_CHECKPOINT=/path/to/smolvla/pretrained_model
export PI0_CHECKPOINT=/home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999
```

Then run in tmux:

```bash
cd /home/ubuntu/cs231n-project
baselines/start_h100_direct_action_evals_tmux.sh
```

Defaults:

```text
Env: RopeFlatten
Episodes: 10
Horizon: 75
Image size: 128
Success threshold: 0.8
Task prompt: straighten the rope
Output root: simulation/data/evals/h100_direct_action
```

Outputs:

```text
simulation/data/evals/h100_direct_action/act_rope_h100/summary.json
simulation/data/evals/h100_direct_action/act_rope_h100/scores.csv
simulation/data/evals/h100_direct_action/smolvla_rope_h100/summary.json
simulation/data/evals/h100_direct_action/smolvla_rope_h100/scores.csv
simulation/data/evals/h100_direct_action/pi0_rope_lora_h100/summary.json
simulation/data/evals/h100_direct_action/pi0_rope_lora_h100/scores.csv
```

## Latency Interpretation

Use the chunk-generation forward-pass latency for fair direct-action latency:

```text
policy_chunk_inference_latency_ms_p50
policy_chunk_inference_latency_ms_p90
```

Do not compare cached action latency as the main latency number. ACT, SmolVLA, and π0 all emit action chunks; most environment steps consume cached actions, which makes per-step latency look artificially tiny.

For π0/OpenPI, the first policy call includes JAX compilation and was about `42.7 s` on the A10G AWS run. The reported p50 chunk latency after compilation was about `200.7 ms` on A10G. That is why the smoke run looked much slower: it included model load / compile / first-call overhead. For final tables, report same-hardware H100 chunk p50/p90 and optionally include a separate cold-start compile latency note.

Also, π0 LoRA here is still a direct-action VLA baseline, not the LeWM world model. LeWM+MPC should get its own same-H100 eval, and the final table should clearly separate direct-action policies from model-based planning.

## What To Report

For each direct-action method, report from `summary.json`:

```text
mean_final_normalized_performance
std_final_normalized_performance
success_rate
policy_chunk_inference_latency_ms_p50
policy_chunk_inference_latency_ms_p90
policy_roundtrip_latency_ms_p50  # optional, includes bridge overhead
checkpoint
created_at
```

Then run the plotting script after H100 results exist, or adapt its input paths:

```bash
/home/ubuntu/openpi/.venv/bin/python baselines/plot_pi0_rope_lora_charts.py
```
