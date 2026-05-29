"""Replay an open-loop action sequence in SoftGym.

Reads an NPZ (produced by `eval/run_mpc.py --mode replay`) with an action
sequence and a config_id, resets the env to that variation, executes the
action sequence step-by-step, and writes the achieved trajectory back to a
new NPZ for scoring by the host-side eval script.

Designed to run inside the softgym Docker (py3.6). Lightweight: no model
imports, no torch — only numpy + pyflex + softgym + collect_trajectories
helpers (for state extraction).
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)

from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict  # noqa: E402
from utils.collect_trajectories import (  # noqa: E402
    render_rgb_depth, extract_proprio, extract_compact_state,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="NPZ with at least: action_sequence (T, A), config_id (int).")
    p.add_argument("--output", required=True,
                   help="Destination NPZ for the replayed trajectory.")
    p.add_argument("--env-name", default="RopeFlatten")
    p.add_argument("--num-variations", type=int, default=200)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--num-picker", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    plan = np.load(args.input, allow_pickle=True)
    action_sequence = np.asarray(plan["action_sequence"], dtype=np.float32)  # (T, A)
    config_id = int(plan["config_id"]) if "config_id" in plan.files else 0
    if "init_state" not in plan.files:
        init_state_ref = None
    else:
        init_state_ref = np.asarray(plan["init_state"], dtype=np.float32)
    T = action_sequence.shape[0]

    # Build env.
    env_class = SOFTGYM_ENVS[args.env_name]
    kwargs = dict(env_arg_dict[args.env_name])
    kwargs["headless"] = True
    kwargs["render"] = True
    kwargs["num_variations"] = args.num_variations
    kwargs["camera_width"] = 720
    kwargs["camera_height"] = 720
    kwargs["use_cached_states"] = False
    kwargs["save_cached_states"] = False
    env = env_class(**kwargs)

    # Reset to the requested variation.
    cfg = env.cached_configs[config_id % len(env.cached_configs)] \
          if getattr(env, "cached_configs", None) else None
    init = env.cached_init_states[config_id % len(env.cached_init_states)] \
           if getattr(env, "cached_init_states", None) else None
    env.reset(config=cfg, config_id=config_id, initial_state=init)
    num_picker = args.num_picker

    # Buffers.
    pixels = []
    states = []
    proprios = []
    rewards = []
    dones = []
    info_buf = {}

    rgb, _ = render_rgb_depth(env, args.img_size, save_depth=False)
    pixels.append(rgb)
    states.append(extract_compact_state(num_picker))
    proprios.append(extract_proprio(env, num_picker))

    for t in range(T):
        _, reward, done, info = env.step(action_sequence[t])
        rewards.append(float(reward))
        dones.append(bool(done))
        for k, v in info.items():
            if k == "flex_env_recorded_frames":
                continue
            try:
                info_buf.setdefault(k, []).append(float(np.asarray(v)))
            except Exception:
                pass
        rgb, _ = render_rgb_depth(env, args.img_size, save_depth=False)
        pixels.append(rgb)
        states.append(extract_compact_state(num_picker))
        proprios.append(extract_proprio(env, num_picker))
        if done:
            break

    out = {
        "pixels":          np.asarray(pixels, dtype=np.uint8),
        "state":           np.asarray(states, dtype=np.float32),
        "proprio":         np.asarray(proprios, dtype=np.float32),
        "action":          action_sequence[:len(rewards)],
        "reward":          np.asarray(rewards, dtype=np.float32),
        "done":            np.asarray(dones, dtype=np.bool_),
        "config_id":       np.asarray(config_id),
        "metadata_json":   np.asarray(json.dumps({
            "source": args.input,
            "env_name": args.env_name,
            "num_variations": args.num_variations,
            "img_size": args.img_size,
            "init_state_ref": init_state_ref.tolist() if init_state_ref is not None else None,
        }, sort_keys=True)),
    }
    for k, v in info_buf.items():
        out["info_" + k] = np.asarray(v, dtype=np.float32)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, **out)
    final_perf = float(info_buf.get("normalized_performance", [0.0])[-1]) if "normalized_performance" in info_buf else 0.0
    print("wrote %s   (T=%d, final_norm_perf=%.3f)" % (args.output, len(rewards), final_perf))


if __name__ == "__main__":
    main()
