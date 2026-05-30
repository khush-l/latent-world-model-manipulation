#!/usr/bin/env python3
"""Run one OpenPI inference on a dummy DROID-shaped observation."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", default="pi05_droid")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--prompt", default="straighten the rope")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = openpi_config.get_config(args.config_name)
    checkpoint = args.checkpoint or "gs://openpi-assets/checkpoints/{}".format(args.config_name)
    checkpoint_dir = download.maybe_download(checkpoint)
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir)
    obs = {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros((7,), dtype=np.float32),
        "observation/gripper_position": np.zeros((1,), dtype=np.float32),
        "prompt": args.prompt,
    }
    start = time.perf_counter()
    out = policy.infer(obs)
    latency_ms = (time.perf_counter() - start) * 1000.0
    actions = np.asarray(out["actions"], dtype=np.float32)
    print(json.dumps({
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "actions_shape": list(actions.shape),
        "latency_ms": latency_ms,
        "first_action": actions.reshape(-1, actions.shape[-1])[0].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
