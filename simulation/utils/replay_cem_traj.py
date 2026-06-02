"""Replay CEM-planned trajectories into the v3 NPZ dataset schema.

The SoftAgent CEM runner (cem/run_cem.py) saves `cem_traj.pkl` =
{initial_states, action_trajs, configs} where action_trajs are in the
NormalizedEnv action space ([-1, 1]). This script replays each episode on the
RAW SoftGym env — restoring the exact start state and DENORMALIZING each action
to the raw box (lb + (a+1)/2*(ub-lb)) so the rollout reproduces what CEM
executed — and records pixels/action(raw)/proprio/state/info per step into one
NPZ per episode (same schema as collect_trajectories.py), ready for
npz_to_hdf5 + merge_h5.

Runs inside the softgym Docker (py3.6): numpy + pyflex + softgym only.

Usage (inside container):
    python utils/replay_cem_traj.py \
        --traj data/softagent/cem/cem_traj.pkl \
        --output-dir data/rope_cem/shard0 --start-idx 0 \
        --env-name RopeFlatten --img-size 128
"""

from __future__ import print_function

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)

from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict  # noqa: E402
from utils.collect_trajectories import (  # noqa: E402
    render_rgb_depth, extract_proprio, extract_compact_state,
    SCHEMA_VERSION, STATE_DIM, to_jsonable,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="cem_traj.pkl from run_cem.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start-idx", type=int, default=0,
                   help="global episode_idx of the first episode in this pkl (sharding).")
    p.add_argument("--env-name", default="RopeFlatten")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--num-picker", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.traj, "rb") as f:
        traj = pickle.load(f)
    initial_states = traj["initial_states"]
    action_trajs = traj["action_trajs"]
    configs = traj.get("configs", [None] * len(action_trajs))
    n_eps = len(action_trajs)
    print("replaying %d CEM episodes from %s" % (n_eps, args.traj))

    env_class = SOFTGYM_ENVS[args.env_name]
    kwargs = dict(env_arg_dict[args.env_name])
    kwargs.update(headless=True, render=True, num_variations=1,
                  use_cached_states=False, save_cached_states=False)
    env = env_class(**kwargs)
    num_picker = args.num_picker
    lb = np.asarray(env.action_space.low, dtype=np.float32)
    ub = np.asarray(env.action_space.high, dtype=np.float32)

    def denorm(a):  # NormalizedEnv: [-1,1] -> raw box (matches normalized_env.step)
        a = np.asarray(a, dtype=np.float32)
        return np.clip(lb + (a + 1.0) * 0.5 * (ub - lb), lb, ub)

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(n_eps):
        env.reset(config=configs[i], config_id=0, initial_state=initial_states[i])
        actions_raw, pixels, states, proprios, rewards, dones = [], [], [], [], [], []
        info_buf = {}

        pixels.append(render_rgb_depth(env, args.img_size, include_depth=False)[0])
        states.append(extract_compact_state(num_picker))
        proprios.append(extract_proprio(env, num_picker))

        for a_norm in action_trajs[i]:
            a_raw = denorm(a_norm)
            _, reward, done, info = env.step(a_raw)
            actions_raw.append(a_raw.astype(np.float32))
            rewards.append(float(reward)); dones.append(bool(done))
            for k, v in info.items():
                if k == "flex_env_recorded_frames":
                    continue
                try:
                    info_buf.setdefault(k, []).append(float(np.asarray(v)))
                except Exception:
                    pass
            pixels.append(render_rgb_depth(env, args.img_size, include_depth=False)[0])
            states.append(extract_compact_state(num_picker))
            proprios.append(extract_proprio(env, num_picker))
            if done:
                break

        ep_idx = args.start_idx + i
        T = len(actions_raw)
        metadata = {
            "schema_version": SCHEMA_VERSION, "env_name": args.env_name,
            "episode_idx": ep_idx, "policy": "cem_expert",
            "behavior_policy": "%s_cem_oracle" % args.env_name.lower(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "env_kwargs": to_jsonable(kwargs),
            "config_id": 0,
            "current_config": to_jsonable(env.get_current_config()),
            "action_space_low": lb.tolist(), "action_space_high": ub.tolist(),
            "action_repeat": int(getattr(env, "action_repeat", 8)),
            "horizon": int(getattr(env, "horizon", T)), "img_size": int(args.img_size),
            "num_picker": int(num_picker), "action_dim": int(lb.shape[0]),
            "proprio_dim": int(4 * num_picker), "state_dim": int(STATE_DIM),
            "source_traj": os.path.abspath(args.traj),
            "source_episode_idx": int(i),
        }
        arrays = {
            "pixels": np.asarray(pixels, dtype=np.uint8),
            "action": np.asarray(actions_raw, dtype=np.float32),
            "proprio": np.asarray(proprios, dtype=np.float32),
            "state": np.asarray(states, dtype=np.float32),
            "reward": np.asarray(rewards, dtype=np.float32),
            "done": np.asarray(dones, dtype=np.bool_),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        for k, v in info_buf.items():
            arrays["info_" + k] = np.asarray(v, dtype=np.float32)
        # invariants (pixels/state/proprio have the initial frame -> T+1)
        assert arrays["pixels"].shape[0] == T + 1, (arrays["pixels"].shape, T)
        assert arrays["state"].shape[1] == STATE_DIM, arrays["state"].shape

        path = os.path.join(args.output_dir, "%s_%06d.npz" % (args.env_name, ep_idx))
        np.savez_compressed(path, **arrays)
        perf = info_buf.get("normalized_performance", [0.0])
        print("  ep %d -> %s  (T=%d final_perf=%.3f)" % (ep_idx, path, T, perf[-1]), flush=True)

    print("done: %d episodes -> %s" % (n_eps, args.output_dir))


if __name__ == "__main__":
    main()
