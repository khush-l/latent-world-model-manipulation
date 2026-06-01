# π0 / OpenPI SoftGym Rope Baseline

This folder contains the π0/OpenPI bridge for training and evaluating a direct-action π0 LoRA policy in the same SoftGym rollout pipeline used by ACT and SmolVLA.

The important fairness point: SoftGym still runs `simulation/utils/eval_lerobot_policy.py`, which writes the same `scores.csv`, `summary.json`, videos, success threshold, normalized performance, and latency fields. The only swapped component is the host-side OpenPI policy server.

## Data Format

π0 LoRA uses the converted LeRobot/OpenPI dataset:

```text
/home/ubuntu/lerobot_datasets/khush/softgym-rope-openpi
```

The raw source HDF5 is:

```text
/home/ubuntu/cs231n-project/training/data/ropeflatten_geometric_5k_v3.h5
```

Mapping:

```text
image:  observation/images/front, SoftGym front RGB
state:  observation/state, 23D SoftGym state/proprio
action: 8D picker action
prompt: straighten the rope
```

If the converted dataset is not available, create it with:

```bash
cd /home/ubuntu/cs231n-project
OVERWRITE=1 baselines/pi0/convert_pi0_softgym_rope_dataset.sh
```

For long conversion jobs:

```bash
tmux new-session -d -s pi0_convert \
  "cd /home/ubuntu/cs231n-project && OVERWRITE=1 baselines/pi0/convert_pi0_softgym_rope_dataset.sh"
```

## Setup OpenPI

```bash
cd /home/ubuntu/cs231n-project
baselines/pi0/bootstrap_openpi.sh /home/ubuntu/openpi
baselines/pi0/install_openpi_softgym_config.sh /home/ubuntu/openpi
```

Installed config:

```text
pi05_softgym_rope_lora
```

The installed config is H100-friendly by default but can still be lowered with environment variables:

```text
OPENPI_BATCH_SIZE=4          # default for H100 handoff
OPENPI_NUM_WORKERS=4         # default
OPENPI_NUM_TRAIN_STEPS=20000 # default
LoRA enabled
W&B disabled by the training wrapper
```

## Train LoRA

```bash
cd /home/ubuntu/cs231n-project
OPENPI_ROOT=/home/ubuntu/openpi \
HF_LEROBOT_HOME=/home/ubuntu/lerobot_datasets \
OPENPI_EXP_NAME=softgym_rope_lora_h100 \
baselines/pi0/start_pi0_lora_training_tmux.sh
```

The script resumes automatically if checkpoint step folders already exist under:

```text
/home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/<OPENPI_EXP_NAME>/
```

Monitor:

```bash
tmux attach -t =pi0_lora
tail -f /home/ubuntu/cs231n-project/baselines/logs/pi0_lora_train.log
```

## Evaluate LoRA In SoftGym

```bash
cd /home/ubuntu/cs231n-project
OPENPI_ROOT=/home/ubuntu/openpi \
OPENPI_CHECKPOINT=/home/ubuntu/openpi/checkpoints/pi05_softgym_rope_lora/softgym_rope_lora_g5/19999 \
OPENPI_EVAL_OUT=data/evals/pi0_rope_lora_g5 \
OPENPI_NUM_EPISODES=10 \
OPENPI_HORIZON=75 \
OPENPI_TIMEOUT_S=180 \
baselines/pi0/eval_pi0_lora_softgym_rope.sh
```

Default output:

```text
simulation/data/evals/pi0_rope_lora_g5/scores.csv
simulation/data/evals/pi0_rope_lora_g5/summary.json
simulation/data/evals/pi0_rope_lora_g5/videos/
```

`OPENPI_TIMEOUT_S=180` matters because the first OpenPI/JAX policy call can include compilation. On the A10G run, the first call was about 42.7 seconds, while post-compile chunk-generation p50 was about 200.7 ms.

## H100 Fair Eval

For final latency, run ACT, SmolVLA, π0 LoRA, and LeWM+MPC on the same H100. The direct-action runner is:

```bash
export ACT_CHECKPOINT=/path/to/act/pretrained_model
export SMOLVLA_CHECKPOINT=/path/to/smolvla/pretrained_model
export PI0_CHECKPOINT=/path/to/openpi/checkpoints/pi05_softgym_rope_lora/<exp>/<step>

cd /home/ubuntu/cs231n-project
baselines/start_h100_direct_action_evals_tmux.sh
```

Main latency fields:

```text
policy_chunk_inference_latency_ms_p50
policy_chunk_inference_latency_ms_p90
```

Do not use cached action latency as the headline latency number; action-chunking policies return cached actions for most environment steps.

## Quick Handoff Check

```bash
cd /home/ubuntu/cs231n-project
baselines/pi0/verify_pi0_handoff.sh
```

A fresh machine may report missing data/checkpoints until Drive artifacts are copied, but repo scripts should all be present and executable.
