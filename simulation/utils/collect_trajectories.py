#!/usr/bin/env python
"""Collect SoftGym trajectories for latent world-model training.

This script intentionally lives under simulation/ and depends only on the
SoftGym/PyFlex stack. It writes one NPZ file per episode so training code can
consume the resulting dataset from a separate, modern ML environment.
"""

from __future__ import print_function

import argparse
import copy
import json
import os
import random
import sys
import time

import cv2
import gym
import numpy as np


SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)

from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict  # noqa: E402
import pyflex  # noqa: E402


STATE_ARRAY_KEYS = ("particle_pos", "particle_vel", "shape_pos", "phase")
SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect SoftGym trajectories into per-episode NPZ files."
    )
    parser.add_argument("--env-name", choices=sorted(SOFTGYM_ENVS.keys()), default="ClothFlatten")
    parser.add_argument("--output-dir", default="data/trajectories")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--num-variations", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=720)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--observation-mode", default=None)
    parser.add_argument("--action-mode", default=None)
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--use-cached-states", action="store_true")
    parser.add_argument("--save-cached-states", action="store_true")
    parser.add_argument("--eval-split", action="store_true")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("random",),
        default="random",
        help="Behavior policy. Only random is implemented here.",
    )
    return parser.parse_args()


def make_output_dir(path):
    if not os.path.isabs(path):
        path = os.path.join(SIM_ROOT, path)
    os.makedirs(path, exist_ok=True)
    return path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_env_kwargs(args):
    kwargs = copy.deepcopy(env_arg_dict[args.env_name])
    kwargs["headless"] = bool(args.headless)
    kwargs["render"] = bool(args.render)
    kwargs["num_variations"] = args.num_variations
    kwargs["use_cached_states"] = bool(args.use_cached_states)
    kwargs["save_cached_states"] = bool(args.save_cached_states)
    kwargs["camera_width"] = args.camera_width
    kwargs["camera_height"] = args.camera_height

    if args.horizon is not None:
        kwargs["horizon"] = args.horizon
    if args.action_repeat is not None:
        kwargs["action_repeat"] = args.action_repeat
    if args.observation_mode is not None:
        kwargs["observation_mode"] = args.observation_mode
    if args.action_mode is not None:
        kwargs["action_mode"] = args.action_mode
    if args.render_mode is not None:
        kwargs["render_mode"] = args.render_mode
    return kwargs


def to_jsonable(value, max_array_items=64):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, max_array_items) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, max_array_items) for v in value]
    if isinstance(value, np.ndarray):
        if value.size <= max_array_items:
            return value.tolist()
        return {
            "array_shape": list(value.shape),
            "array_dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def render_rgb_depth(env, img_size, include_depth=False):
    img, depth = pyflex.render()
    cam = env.camera_params.get(env.camera_name, env.camera_params.get("default_camera"))
    if cam is None:
        raise RuntimeError("Environment has no camera parameters after reset.")
    raw_width = int(cam["width"])
    raw_height = int(cam["height"])

    rgb = img.reshape(raw_height, raw_width, 4)[::-1, :, :3].astype(np.uint8)
    if img_size != raw_width or img_size != raw_height:
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if not include_depth:
        return rgb, None

    depth = depth.reshape(raw_height, raw_width)[::-1].astype(np.float32)
    if img_size != raw_width or img_size != raw_height:
        depth = cv2.resize(depth, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return rgb, depth


def sample_normalized_action(action_space):
    if not isinstance(action_space, gym.spaces.Box):
        return action_space.sample(), action_space.sample()

    action_norm = np.random.uniform(
        low=-1.0, high=1.0, size=action_space.shape
    ).astype(np.float32)
    low = action_space.low.astype(np.float32)
    high = action_space.high.astype(np.float32)
    action_raw = low + (action_norm + 1.0) * 0.5 * (high - low)
    action_raw = np.clip(action_raw, low, high).astype(np.float32)
    return action_norm, action_raw


def extract_state_arrays(env):
    state = env.get_state()
    arrays = {}
    for key in STATE_ARRAY_KEYS:
        if key in state:
            arrays[key] = np.asarray(state[key]).copy()
    return arrays


def append_state(buffers, state_arrays):
    for key, value in state_arrays.items():
        buffers.setdefault(key, []).append(value)


def stack_state_buffers(state_buffers):
    stacked = {}
    for key, values in state_buffers.items():
        try:
            stacked["state_" + key] = np.stack(values, axis=0)
        except ValueError:
            stacked["state_" + key] = np.asarray(values, dtype=object)
    return stacked


def scalar_info(info):
    scalars = {}
    for key, value in info.items():
        if key == "flex_env_recorded_frames":
            continue
        arr = np.asarray(value)
        if arr.shape == ():
            scalars[key] = float(arr)
    return scalars


def append_info(info_buffers, info):
    for key, value in scalar_info(info).items():
        info_buffers.setdefault(key, []).append(value)


def collect_episode(env, args, episode_idx, env_kwargs):
    obs = env.reset()
    del obs

    rgb_frames = []
    depth_frames = []
    action_norms = []
    action_raws = []
    rewards = []
    dones = []
    info_buffers = {}
    state_buffers = {}

    rgb, depth = render_rgb_depth(env, args.img_size, args.save_depth)
    rgb_frames.append(rgb)
    if args.save_depth:
        depth_frames.append(depth)
    if args.save_state:
        append_state(state_buffers, extract_state_arrays(env))

    for _ in range(env.horizon):
        action_norm, action_raw = sample_normalized_action(env.action_space)
        _, reward, done, info = env.step(action_raw)

        action_norms.append(np.asarray(action_norm, dtype=np.float32).copy())
        action_raws.append(np.asarray(action_raw, dtype=np.float32).copy())
        rewards.append(float(reward))
        dones.append(bool(done))
        append_info(info_buffers, info)

        rgb, depth = render_rgb_depth(env, args.img_size, args.save_depth)
        rgb_frames.append(rgb)
        if args.save_depth:
            depth_frames.append(depth)
        if args.save_state:
            append_state(state_buffers, extract_state_arrays(env))

        if done:
            break

    current_config = env.get_current_config()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "env_name": args.env_name,
        "episode_idx": episode_idx,
        "seed": args.seed,
        "policy": args.policy,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env_kwargs": to_jsonable(env_kwargs),
        "config_id": to_jsonable(getattr(env, "current_config_id", None)),
        "current_config": to_jsonable(current_config),
        "action_space_low": to_jsonable(env.action_space.low),
        "action_space_high": to_jsonable(env.action_space.high),
        "action_repeat": int(env.action_repeat),
        "horizon": int(env.horizon),
        "img_size": int(args.img_size),
    }

    arrays = {
        "rgb": np.asarray(rgb_frames, dtype=np.uint8),
        "action_normalized": np.asarray(action_norms, dtype=np.float32),
        "action_raw": np.asarray(action_raws, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "done": np.asarray(dones, dtype=np.bool_),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if args.save_depth:
        arrays["depth"] = np.asarray(depth_frames, dtype=np.float32)
    for key, values in info_buffers.items():
        arrays["info_" + key] = np.asarray(values, dtype=np.float32)
    if args.save_state:
        arrays.update(stack_state_buffers(state_buffers))

    return arrays, metadata


def save_episode(output_dir, env_name, episode_idx, arrays, compress=True):
    filename = "{}_{:06d}.npz".format(env_name, episode_idx)
    path = os.path.join(output_dir, filename)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    return path


def append_manifest(output_dir, metadata, path):
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    record = copy.deepcopy(metadata)
    record["path"] = os.path.relpath(path, output_dir)
    with open(manifest_path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = make_output_dir(args.output_dir)
    env_kwargs = build_env_kwargs(args)

    env = SOFTGYM_ENVS[args.env_name](**env_kwargs)
    env.eval_flag = bool(args.eval_split)

    try:
        for episode_idx in range(args.num_episodes):
            arrays, metadata = collect_episode(env, args, episode_idx, env_kwargs)
            path = save_episode(
                output_dir,
                args.env_name,
                episode_idx,
                arrays,
                compress=not args.no_compress,
            )
            append_manifest(output_dir, metadata, path)
            print(
                "saved {} steps to {}".format(
                    arrays["action_raw"].shape[0],
                    path,
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
