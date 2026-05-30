# π0 / OpenPI Smoke Baseline

This folder contains the proof-of-concept bridge for evaluating an OpenPI
policy in the same SoftGym rollout pipeline used by the ACT and SmolVLA
baselines.

The important fairness point is that this does not create a separate metric
path: SoftGym still runs `simulation/utils/eval_lerobot_policy.py`, which writes
the same `scores.csv`, `summary.json`, success threshold, normalized
performance, videos, and policy latency fields. The only swapped component is
the host policy server.

## Setup

Clone and install OpenPI outside this repo:

```bash
baselines/pi0/bootstrap_openpi.sh /home/ubuntu/openpi
```

OpenPI's README currently lists single-GPU memory requirements as >8 GB for
inference, >22.5 GB for LoRA fine-tuning, and >70 GB for full fine-tuning. This
T4 instance is therefore only for smoke inference, not LoRA training.

## Local Smoke

Start with a model-load/inference smoke that does not touch SoftGym:

```bash
baselines/pi0/run_openpi_dummy_smoke.sh /home/ubuntu/openpi pi05_droid
```

This downloads the OpenPI checkpoint if needed and runs a dummy DROID-shaped
observation through the policy.

## SoftGym Smoke Evaluation

Run the same SoftGym evaluation client used by the other direct-action
baselines:

```bash
OPENPI_ROOT=/home/ubuntu/openpi \
OPENPI_CONFIG=pi05_droid \
OPENPI_CHECKPOINT=gs://openpi-assets/checkpoints/pi05_droid \
baselines/pi0/eval_openpi_softgym_smoke.sh
```

Default output:

```text
simulation/data/evals/pi0_rope_smoke/
  scores.csv
  summary.json
  videos/
```

The server maps the SoftGym front image into the DROID exterior/wrist image
slots and maps the 23D SoftGym state into an 8D DROID-style state placeholder.
That is only a zero-shot compatibility smoke. The later LoRA instance should
replace this with a real OpenPI data transform trained on our SoftGym LeRobot
dataset.
