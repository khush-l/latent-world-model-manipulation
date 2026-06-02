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
SCHEMA_VERSION = 3
STATE_DIM = 15  # see DATA_FORMAT.md §5
GRIP_INDEX = 3  # action[:, 3] = grip per picker


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
    parser.add_argument(
        "--goal-character",
        choices=("S", "O", "M", "C", "U"),
        default=None,
        help="For RopeConfiguration, force resets to this target character.",
    )
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument(
        "--save-full-state",
        action="store_true",
        help="Also dump variable-length particle/shape/phase arrays. "
             "Used for MPC _set_state at eval time; not part of the LeWM "
             "training table. See DATA_FORMAT.md §3.",
    )
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("random", "geometric", "manipulate", "push", "configure", "route_u"),
        default="random",
        help="Behavior policy. 'random' = uniform sample over action space. "
             "'geometric' = scripted state-machine policy (currently only "
             "RopeFlatten — grabs endpoints and stretches). "
             "'manipulate' = scripted free-form policy that grabs random "
             "middle rope particles and walks them through smooth waypoints "
             "(creates diverse rope configurations instead of always flat). "
             "'configure' = goal-conditioned RopeConfiguration oracle that "
             "moves keypoints toward the target character. "
             "'route_u' = U-shaped cable-routing expert for the practical "
             "RopeConfiguration task.",
    )
    parser.add_argument(
        "--script-num-waypoints",
        type=int,
        default=3,
        help="Number of random waypoints per picker for --policy manipulate.",
    )
    parser.add_argument(
        "--script-noise-scale",
        type=float,
        default=0.02,
        help="Gaussian noise sigma on scripted-policy deltas, as fraction of "
             "the commanded magnitude. Auto-shrinks near target so HOLD phases "
             "don't jitter. 0 = deterministic. Used only with --policy geometric.",
    )
    parser.add_argument(
        "--script-lateral-scale",
        type=float,
        default=1.0,
        help="Final endpoint-separation as a fraction of rope natural length. "
             "1.0 = stretch to natural length. Used only with --policy geometric.",
    )
    parser.add_argument(
        "--script-lift-height",
        type=float,
        default=0.05,
        help="How high (meters) to lift gripped endpoints. "
             "Used only with --policy geometric.",
    )
    parser.add_argument(
        "--script-action-gain",
        type=float,
        default=0.3,
        help="Proportional gain for picker position error. <1.0 produces "
             "smooth deceleration near targets (1.0 = bang-bang).",
    )
    parser.add_argument(
        "--script-max-speed-scale",
        type=float,
        default=0.35,
        help="Fraction of env max action delta that the scripted policy is "
             "allowed to command. Lowers picker top speed (e.g. 0.35 = 35%% "
             "of env max → ~2.8 cm per env step instead of 8 cm).",
    )
    parser.add_argument(
        "--script-deadband",
        type=float,
        default=5e-4,
        help="Per-axis commanded-delta deadband (meters/substep). Any |delta| "
             "below this is zeroed — eliminates proportional-control jitter "
             "during HOLD/GRIP phases where the picker should be stationary.",
    )
    parser.add_argument(
        "--min-final-normalized-performance",
        type=float,
        default=None,
        help="If set, only save episodes whose final info_normalized_performance "
             "is at least this value. Failed attempts are retried.",
    )
    parser.add_argument(
        "--stop-on-normalized-performance",
        type=float,
        default=None,
        help="If set, stop recording the episode as soon as "
             "info_normalized_performance reaches this value.",
    )
    parser.add_argument(
        "--max-attempts-per-episode",
        type=int,
        default=20,
        help="Maximum attempts for each saved episode when an acceptance "
             "threshold is set.",
    )
    parser.add_argument(
        "--min-recorded-horizon-for-acceptance",
        type=int,
        default=0,
        help="If an acceptance threshold is set, reject otherwise-successful "
             "episodes shorter than this many recorded actions. Useful for "
             "excluding near-solved expert resets that provide little "
             "interaction data.",
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


def sample_action(action_space):
    if not isinstance(action_space, gym.spaces.Box):
        return action_space.sample().astype(np.float32)
    low = action_space.low.astype(np.float32)
    high = action_space.high.astype(np.float32)
    action_norm = np.random.uniform(low=-1.0, high=1.0, size=action_space.shape).astype(np.float32)
    action_raw = low + (action_norm + 1.0) * 0.5 * (high - low)
    return np.clip(action_raw, low, high).astype(np.float32)


def extract_proprio(env, num_picker):
    """Picker xyz (per picker) + holding flag (0/1) → (4 * num_picker,) float32."""
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    picker_xyz = shape_states[:num_picker, :3].astype(np.float32)
    picked = getattr(env.action_tool, "picked_particles", [None] * num_picker)
    holding = np.array(
        [0.0 if picked[i] is None else 1.0 for i in range(num_picker)],
        dtype=np.float32,
    )
    return np.concatenate([picker_xyz.reshape(-1), holding], axis=0).astype(np.float32)


def extract_compact_state(num_picker):
    """Fixed-dim privileged state — see DATA_FORMAT.md §5."""
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    picker_xyz = shape_states[:num_picker, :3].astype(np.float32).reshape(-1)
    if picker_xyz.shape[0] < 6:
        picker_xyz = np.concatenate(
            [picker_xyz, np.zeros(6 - picker_xyz.shape[0], dtype=np.float32)]
        )
    elif picker_xyz.shape[0] > 6:
        picker_xyz = picker_xyz[:6]

    particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3].astype(np.float32)
    if particle_pos.size == 0:
        com = np.zeros(3, dtype=np.float32)
        bbox_min = np.zeros(3, dtype=np.float32)
        bbox_max = np.zeros(3, dtype=np.float32)
    else:
        com = particle_pos.mean(axis=0)
        bbox_min = particle_pos.min(axis=0)
        bbox_max = particle_pos.max(axis=0)

    state = np.concatenate([picker_xyz, com, bbox_min, bbox_max]).astype(np.float32)
    assert state.shape == (STATE_DIM,), state.shape
    return state


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


def reset_env(env, args, episode_idx=0, attempt_idx=0):
    if args.goal_character is None:
        return env.reset()
    if args.env_name != "RopeConfiguration":
        raise ValueError("--goal-character is only valid for RopeConfiguration")
    candidates = [
        i for i, cfg in enumerate(getattr(env, "cached_configs", []) or [])
        if str(cfg.get("goal_character", "")) == args.goal_character
    ]
    if not candidates:
        raise RuntimeError("no cached RopeConfiguration configs for goal %s" % args.goal_character)
    # Avoid duplicated forced-goal episodes inside expert datasets. This still
    # varies failed retries, but is deterministic for reproducibility.
    offset = int(args.seed) + int(episode_idx) * 997 + int(attempt_idx) * 37
    config_id = int(candidates[offset % len(candidates)])
    return env.reset(config_id=config_id)


def collect_episode(env, args, episode_idx, env_kwargs, attempt_idx=0):
    obs = reset_env(env, args, episode_idx=episode_idx, attempt_idx=attempt_idx)
    del obs

    num_picker = int(env_kwargs.get("num_picker", getattr(env.action_tool, "num_picker", 2)))
    action_dim = 4 * num_picker
    proprio_dim = 4 * num_picker

    pixel_frames = []
    depth_frames = []
    proprios = []
    states = []
    actions = []
    rewards = []
    dones = []
    info_buffers = {}
    full_state_buffers = {}

    pixels, depth = render_rgb_depth(env, args.img_size, args.save_depth)
    pixel_frames.append(pixels)
    proprios.append(extract_proprio(env, num_picker))
    states.append(extract_compact_state(num_picker))
    if args.save_depth:
        depth_frames.append(depth)
    if args.save_full_state:
        append_state(full_state_buffers, extract_state_arrays(env))

    policy_obj = None
    if args.policy in ("geometric", "manipulate", "push", "configure", "route_u"):
        from geometric_policy import make_policy
        policy_obj = make_policy(
            env_name=args.env_name,
            env=env,
            num_picker=num_picker,
            noise_scale=args.script_noise_scale,
            lateral_scale=args.script_lateral_scale,
            lift_height=args.script_lift_height,
            action_gain=args.script_action_gain,
            max_speed_scale=args.script_max_speed_scale,
            deadband=args.script_deadband,
            kind=args.policy,
            num_waypoints=args.script_num_waypoints,
            rng=np.random.RandomState(args.seed + episode_idx),
        )
        policy_obj.reset()

    for _ in range(env.horizon):
        if policy_obj is not None:
            action_raw = policy_obj.get_action()
        else:
            action_raw = sample_action(env.action_space)
        _, reward, done, info = env.step(action_raw)

        actions.append(np.asarray(action_raw, dtype=np.float32).copy())
        rewards.append(float(reward))
        dones.append(bool(done))
        append_info(info_buffers, info)

        pixels, depth = render_rgb_depth(env, args.img_size, args.save_depth)
        pixel_frames.append(pixels)
        proprios.append(extract_proprio(env, num_picker))
        states.append(extract_compact_state(num_picker))
        if args.save_depth:
            depth_frames.append(depth)
        if args.save_full_state:
            append_state(full_state_buffers, extract_state_arrays(env))

        if args.stop_on_normalized_performance is not None:
            perf = scalar_info(info).get("normalized_performance")
            if perf is not None and perf >= args.stop_on_normalized_performance:
                done = True
                dones[-1] = True

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
        "recorded_horizon": int(len(actions)),
        "img_size": int(args.img_size),
        "num_picker": int(num_picker),
        "action_dim": int(action_dim),
        "proprio_dim": int(proprio_dim),
        "state_dim": int(STATE_DIM),
    }
    if args.policy in ("geometric", "manipulate", "push", "configure", "route_u"):
        metadata["behavior_policy"] = "rope_" + args.policy
        metadata["script_noise_scale"] = float(args.script_noise_scale)
        metadata["script_lateral_scale"] = float(args.script_lateral_scale)
        metadata["script_lift_height"] = float(args.script_lift_height)
        metadata["script_action_gain"] = float(args.script_action_gain)
        metadata["script_max_speed_scale"] = float(args.script_max_speed_scale)
        metadata["script_deadband"] = float(args.script_deadband)
        if args.policy == "manipulate":
            metadata["script_num_waypoints"] = int(args.script_num_waypoints)
        if args.policy in ("configure", "route_u"):
            metadata["acceptance_policy"] = "final_normalized_performance"
        if args.policy == "route_u":
            metadata["task_description"] = "u_shaped_cable_routing"

    pixels_arr = np.asarray(pixel_frames, dtype=np.uint8)
    action_arr = np.asarray(actions, dtype=np.float32)
    proprio_arr = np.asarray(proprios, dtype=np.float32)
    state_arr = np.asarray(states, dtype=np.float32)
    reward_arr = np.asarray(rewards, dtype=np.float32)
    done_arr = np.asarray(dones, dtype=np.bool_)

    # Length invariants — see DATA_FORMAT.md §8.
    T = action_arr.shape[0]
    assert pixels_arr.shape[0] == T + 1, (pixels_arr.shape, T)
    assert proprio_arr.shape[0] == T + 1, (proprio_arr.shape, T)
    assert state_arr.shape[0] == T + 1, (state_arr.shape, T)
    assert reward_arr.shape[0] == T and done_arr.shape[0] == T
    assert action_arr.shape[1] == action_dim, (action_arr.shape, action_dim)
    assert proprio_arr.shape[1] == proprio_dim, (proprio_arr.shape, proprio_dim)
    assert state_arr.shape[1] == STATE_DIM, state_arr.shape

    arrays = {
        "pixels": pixels_arr,
        "action": action_arr,
        "proprio": proprio_arr,
        "state": state_arr,
        "reward": reward_arr,
        "done": done_arr,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if args.save_depth:
        arrays["depth"] = np.asarray(depth_frames, dtype=np.float32)
    for key, values in info_buffers.items():
        arrays["info_" + key] = np.asarray(values, dtype=np.float32)
    if args.save_full_state:
        for key, value in stack_state_buffers(full_state_buffers).items():
            arrays["full_" + key] = value

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
            max_attempts = 1
            if args.min_final_normalized_performance is not None:
                max_attempts = max(1, int(args.max_attempts_per_episode))

            for attempt_idx in range(max_attempts):
                arrays, metadata = collect_episode(
                    env, args, episode_idx, env_kwargs, attempt_idx=attempt_idx
                )
                final_perf = None
                if "info_normalized_performance" in arrays:
                    vals = np.asarray(arrays["info_normalized_performance"], dtype=np.float32)
                    finite = vals[np.isfinite(vals)]
                    if finite.size:
                        final_perf = float(finite[-1])
                metadata["attempt_idx"] = int(attempt_idx)
                metadata["accepted_final_normalized_performance"] = (
                    None if final_perf is None else float(final_perf)
                )
                arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

                if args.min_final_normalized_performance is None:
                    break
                recorded_horizon = int(arrays["action"].shape[0])
                long_enough = recorded_horizon >= int(args.min_recorded_horizon_for_acceptance)
                if (
                    final_perf is not None
                    and final_perf >= args.min_final_normalized_performance
                    and long_enough
                ):
                    break
                if final_perf is not None and not long_enough:
                    print(
                        "rejected episode {} attempt {} final_norm_perf={:.4f} "
                        "recorded_horizon={} < min_recorded_horizon={}".format(
                            episode_idx,
                            attempt_idx,
                            final_perf,
                            recorded_horizon,
                            args.min_recorded_horizon_for_acceptance,
                        )
                    )
                    continue
                print(
                    "rejected episode {} attempt {} final_norm_perf={}".format(
                        episode_idx,
                        attempt_idx,
                        "None" if final_perf is None else "{:.4f}".format(final_perf),
                    )
                )
            else:
                raise RuntimeError(
                    "episode {} failed acceptance threshold {} after {} attempts".format(
                        episode_idx,
                        args.min_final_normalized_performance,
                        max_attempts,
                    )
                )

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
                    arrays["action"].shape[0],
                    path,
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
