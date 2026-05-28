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
STATE_DIM = 15  # see DATA_FORMAT.md section 5
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
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument(
        "--save-full-state",
        action="store_true",
        help="Also dump variable-length particle/shape/phase arrays. "
             "Used for MPC _set_state at eval time; not part of the LeWM "
             "training table. See DATA_FORMAT.md section 3.",
    )
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument(
        "--policy",
        choices=(
            "random",
            "rope_geometric",
            "cloth_geometric",
            "rope_geometric_v2",
            "cloth_geometric_v2",
        ),
        default="random",
        help="Behavior policy. rope_geometric and cloth_geometric are scripted demo policies.",
    )
    parser.add_argument(
        "--script-noise-scale",
        type=float,
        default=0.15,
        help="Relative noise scale for scripted policy actions.",
    )
    parser.add_argument(
        "--script-lateral-scale",
        type=float,
        default=0.20,
        help="Relative lateral wiggle scale for scripted rope pulling.",
    )
    parser.add_argument(
        "--cloth-pull-scale",
        type=float,
        default=0.55,
        help="Fraction of current cloth bbox half-diagonal to pull beyond for cloth_geometric.",
    )
    parser.add_argument(
        "--cloth-lift-height",
        type=float,
        default=0.055,
        help="Picker height during cloth_geometric pull stage.",
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


def clip_action(action, action_space):
    if isinstance(action_space, gym.spaces.Box):
        low = action_space.low.astype(np.float32)
        high = action_space.high.astype(np.float32)
        return np.clip(action.astype(np.float32), low, high).astype(np.float32)
    return action.astype(np.float32)


def _rope_axis_from_particles(particle_pos):
    xz = particle_pos[:, [0, 2]]
    centered = xz - xz.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2 or np.linalg.norm(centered) < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0].astype(np.float32)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1.0, 0.0], dtype=np.float32)
    return axis / (np.linalg.norm(axis) + 1e-8)


def _macro_delta_to_action_delta(delta, env, action_space):
    repeat = max(int(getattr(env, "action_repeat", 1)), 1)
    per_repeat_delta = delta / float(repeat)
    if isinstance(action_space, gym.spaces.Box):
        low = action_space.low.reshape(-1, 4)[:, :3]
        high = action_space.high.reshape(-1, 4)[:, :3]
        per_repeat_delta = np.clip(per_repeat_delta, low, high)
    return per_repeat_delta.astype(np.float32)


def _script_noise(target, action_space, scale, vertical_scale=0.25):
    if scale <= 0:
        return target
    if isinstance(action_space, gym.spaces.Box):
        max_delta = np.maximum(np.abs(action_space.high.reshape(-1, 4)[:, :3]), 1e-6)
    else:
        max_delta = np.ones((2, 3), dtype=np.float32) * 0.01
    noise = np.random.normal(loc=0.0, scale=scale, size=(2, 3)).astype(np.float32)
    noise = noise * max_delta.astype(np.float32)
    noise[:, 1] *= vertical_scale
    return target + noise


def rope_geometric_action(env, step_idx, args):
    """Two-picker scripted RopeFlatten policy.

    Stages:
    1. Move pickers above the rope endpoints.
    2. Descend and grasp endpoint particles.
    3. Pull endpoints apart along the rope's principal axis with small lateral
       wiggles/noise to expose local deformable dynamics.
    4. Hold, then release/lift near the end of the episode.
    """
    if args.env_name != "RopeFlatten":
        raise ValueError("rope_geometric policy only supports RopeFlatten")

    action_space = env.action_space
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    picker_pos = shape_states[:2, :3].astype(np.float32)
    particles = np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3].astype(np.float32)

    endpoint0 = particles[0].copy()
    endpoint1 = particles[-1].copy()
    center = particles.mean(axis=0)

    endpoint_axis = endpoint0[[0, 2]] - endpoint1[[0, 2]]
    if np.linalg.norm(endpoint_axis) < 0.03:
        axis_xz = _rope_axis_from_particles(particles)
        if np.dot(axis_xz, endpoint_axis) < 0:
            axis_xz = -axis_xz
    else:
        axis_xz = endpoint_axis / (np.linalg.norm(endpoint_axis) + 1e-8)
    perp_xz = np.array([-axis_xz[1], axis_xz[0]], dtype=np.float32)

    horizon = max(int(env.horizon), 1)
    frac = float(step_idx) / float(horizon)
    rope_length = float(getattr(env, "rope_length", 0.5))
    half_pull = max(0.45 * rope_length, 0.18)

    target = np.zeros((2, 3), dtype=np.float32)
    grip = np.zeros(2, dtype=np.float32)

    if frac < 0.18:
        target[0] = endpoint0 + np.array([0.0, 0.08, 0.0], dtype=np.float32)
        target[1] = endpoint1 + np.array([0.0, 0.08, 0.0], dtype=np.float32)
    elif frac < 0.30:
        target[0] = endpoint0 + np.array([0.0, 0.025, 0.0], dtype=np.float32)
        target[1] = endpoint1 + np.array([0.0, 0.025, 0.0], dtype=np.float32)
        grip[:] = 1.0
    elif frac < 0.82:
        pull_frac = (frac - 0.30) / 0.52
        current_half = 0.5 * np.linalg.norm(endpoint0[[0, 2]] - endpoint1[[0, 2]])
        desired_half = (1.0 - pull_frac) * current_half + pull_frac * half_pull
        lateral = (
            args.script_lateral_scale
            * rope_length
            * 0.08
            * np.sin(2.0 * np.pi * (3.0 * pull_frac + 0.17 * (args.seed + step_idx)))
        )
        target_xz_0 = center[[0, 2]] + axis_xz * desired_half + perp_xz * lateral
        target_xz_1 = center[[0, 2]] - axis_xz * desired_half - perp_xz * lateral
        target[0] = np.array([target_xz_0[0], 0.055, target_xz_0[1]], dtype=np.float32)
        target[1] = np.array([target_xz_1[0], 0.055, target_xz_1[1]], dtype=np.float32)
        grip[:] = 1.0
    elif frac < 0.92:
        target[0] = endpoint0 + np.array([0.0, 0.045, 0.0], dtype=np.float32)
        target[1] = endpoint1 + np.array([0.0, 0.045, 0.0], dtype=np.float32)
        grip[:] = 1.0
    else:
        target[0] = endpoint0 + np.array([0.0, 0.12, 0.0], dtype=np.float32)
        target[1] = endpoint1 + np.array([0.0, 0.12, 0.0], dtype=np.float32)

    if args.script_noise_scale > 0:
        if isinstance(action_space, gym.spaces.Box):
            max_delta = np.maximum(np.abs(action_space.high.reshape(-1, 4)[:, :3]), 1e-6)
        else:
            max_delta = np.ones((2, 3), dtype=np.float32) * 0.01
        noise = np.random.normal(
            loc=0.0,
            scale=args.script_noise_scale * 0.20,
            size=(2, 3),
        ).astype(np.float32) * max_delta.astype(np.float32)
        noise[:, 1] *= 0.25
        target += noise

    delta = _macro_delta_to_action_delta(target - picker_pos, env, action_space)
    action = np.concatenate([delta, grip.reshape(-1, 1)], axis=1).reshape(-1)
    return clip_action(action, action_space)



def rope_geometric_v2_action(env, step_idx, args):
    """Stronger RopeFlatten demo policy with longer, cleaner endpoint pulls."""
    if args.env_name != "RopeFlatten":
        raise ValueError("rope_geometric_v2 policy only supports RopeFlatten")

    action_space = env.action_space
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    picker_pos = shape_states[:2, :3].astype(np.float32)
    particles = np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3].astype(np.float32)

    endpoint0 = particles[0].copy()
    endpoint1 = particles[-1].copy()
    center = particles.mean(axis=0)
    axis_xz = endpoint0[[0, 2]] - endpoint1[[0, 2]]
    if np.linalg.norm(axis_xz) < 0.03:
        axis_xz = _rope_axis_from_particles(particles)
    else:
        axis_xz = axis_xz / (np.linalg.norm(axis_xz) + 1e-8)

    horizon = max(int(env.horizon), 1)
    frac = float(step_idx) / float(horizon)
    rope_length = float(getattr(env, "rope_length", 0.5))
    target_half = max(0.58 * rope_length, 0.25)

    target = np.zeros((2, 3), dtype=np.float32)
    grip = np.zeros(2, dtype=np.float32)
    if frac < 0.12:
        target[0] = endpoint0 + np.array([0.0, 0.09, 0.0], dtype=np.float32)
        target[1] = endpoint1 + np.array([0.0, 0.09, 0.0], dtype=np.float32)
    elif frac < 0.22:
        target[0] = endpoint0 + np.array([0.0, 0.018, 0.0], dtype=np.float32)
        target[1] = endpoint1 + np.array([0.0, 0.018, 0.0], dtype=np.float32)
        grip[:] = 1.0
    elif frac < 0.88:
        pull_frac = (frac - 0.22) / 0.66
        pull_frac = 0.5 - 0.5 * np.cos(np.pi * pull_frac)
        target_xz_0 = center[[0, 2]] + axis_xz * target_half
        target_xz_1 = center[[0, 2]] - axis_xz * target_half
        height = 0.050
        end0 = np.array([target_xz_0[0], height, target_xz_0[1]], dtype=np.float32)
        end1 = np.array([target_xz_1[0], height, target_xz_1[1]], dtype=np.float32)
        start0 = endpoint0 + np.array([0.0, height, 0.0], dtype=np.float32)
        start1 = endpoint1 + np.array([0.0, height, 0.0], dtype=np.float32)
        target[0] = (1.0 - pull_frac) * start0 + pull_frac * end0
        target[1] = (1.0 - pull_frac) * start1 + pull_frac * end1
        grip[:] = 1.0
    elif frac < 0.95:
        target[0] = endpoint0 + np.array([0.0, 0.040, 0.0], dtype=np.float32)
        target[1] = endpoint1 + np.array([0.0, 0.040, 0.0], dtype=np.float32)
        grip[:] = 1.0
    else:
        target[0] = endpoint0 + np.array([0.0, 0.12, 0.0], dtype=np.float32)
        target[1] = endpoint1 + np.array([0.0, 0.12, 0.0], dtype=np.float32)

    target = _script_noise(target, action_space, args.script_noise_scale * 0.08)
    delta = _macro_delta_to_action_delta(target - picker_pos, env, action_space)
    action = np.concatenate([delta, grip.reshape(-1, 1)], axis=1).reshape(-1)
    return clip_action(action, action_space)


def _principal_axis_xz(points_xz):
    centered = points_xz - points_xz.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2 or np.linalg.norm(centered) < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0].astype(np.float32)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1.0, 0.0], dtype=np.float32)
    return axis / (np.linalg.norm(axis) + 1e-8)


def _farthest_pair_indices(points_xz):
    n = points_xz.shape[0]
    if n < 2:
        return 0, 0
    # Cloth has at most ~7k particles in our setup; this O(N^2) search is only
    # called once per stored env step and is acceptable for small VLA demos.
    diff = points_xz[:, None, :] - points_xz[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
    return int(i), int(j)


def _extreme_pair_along_axis(points_xz, axis_xz):
    axis_xz = axis_xz.astype(np.float32)
    axis_xz = axis_xz / (np.linalg.norm(axis_xz) + 1e-8)
    proj = points_xz.dot(axis_xz)
    return int(np.argmax(proj)), int(np.argmin(proj))


def cloth_geometric_action(env, step_idx, args):
    """Two-picker scripted ClothFlatten policy.

    The policy imitates a simple human flattening primitive: find two far-apart
    visible cloth particles, approach from above, grasp them, lift slightly, pull
    outward along the cloth's dominant axis, then release. It is intentionally a
    high-quality demo policy for ACT/SmolVLA fine-tuning, not a broad dynamics
    exploration policy.
    """
    if args.env_name != "ClothFlatten":
        raise ValueError("cloth_geometric policy only supports ClothFlatten")

    action_space = env.action_space
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    picker_pos = shape_states[:2, :3].astype(np.float32)
    particles = np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3].astype(np.float32)
    points_xz = particles[:, [0, 2]]
    center_xz = points_xz.mean(axis=0)

    idx0, idx1 = _farthest_pair_indices(points_xz)
    p0 = particles[idx0].copy()
    p1 = particles[idx1].copy()
    axis_xz = p0[[0, 2]] - p1[[0, 2]]
    if np.linalg.norm(axis_xz) < 0.03:
        axis_xz = _principal_axis_xz(points_xz)
    else:
        axis_xz = axis_xz / (np.linalg.norm(axis_xz) + 1e-8)
    perp_xz = np.array([-axis_xz[1], axis_xz[0]], dtype=np.float32)

    bbox_min = points_xz.min(axis=0)
    bbox_max = points_xz.max(axis=0)
    half_extent = 0.5 * np.linalg.norm(bbox_max - bbox_min)
    pull_half = half_extent * (1.0 + float(args.cloth_pull_scale))
    pull_half = float(np.clip(pull_half, 0.16, 0.42))

    horizon = max(int(env.horizon), 1)
    frac = float(step_idx) / float(horizon)
    target = np.zeros((2, 3), dtype=np.float32)
    grip = np.zeros(2, dtype=np.float32)

    if frac < 0.18:
        target[0] = p0 + np.array([0.0, 0.10, 0.0], dtype=np.float32)
        target[1] = p1 + np.array([0.0, 0.10, 0.0], dtype=np.float32)
    elif frac < 0.30:
        target[0] = p0 + np.array([0.0, 0.025, 0.0], dtype=np.float32)
        target[1] = p1 + np.array([0.0, 0.025, 0.0], dtype=np.float32)
        grip[:] = 1.0
    elif frac < 0.82:
        pull_frac = (frac - 0.30) / 0.52
        lateral = (
            args.script_lateral_scale
            * 0.035
            * np.sin(2.0 * np.pi * (2.0 * pull_frac + 0.11 * (args.seed + step_idx)))
        )
        target_xz_0 = center_xz + axis_xz * pull_half + perp_xz * lateral
        target_xz_1 = center_xz - axis_xz * pull_half - perp_xz * lateral
        height = float(args.cloth_lift_height)
        target[0] = np.array([target_xz_0[0], height, target_xz_0[1]], dtype=np.float32)
        target[1] = np.array([target_xz_1[0], height, target_xz_1[1]], dtype=np.float32)
        grip[:] = 1.0
    elif frac < 0.92:
        target[0] = p0 + np.array([0.0, 0.040, 0.0], dtype=np.float32)
        target[1] = p1 + np.array([0.0, 0.040, 0.0], dtype=np.float32)
        grip[:] = 1.0
    else:
        target[0] = p0 + np.array([0.0, 0.12, 0.0], dtype=np.float32)
        target[1] = p1 + np.array([0.0, 0.12, 0.0], dtype=np.float32)

    if args.script_noise_scale > 0:
        if isinstance(action_space, gym.spaces.Box):
            max_delta = np.maximum(np.abs(action_space.high.reshape(-1, 4)[:, :3]), 1e-6)
        else:
            max_delta = np.ones((2, 3), dtype=np.float32) * 0.01
        noise = np.random.normal(
            loc=0.0,
            scale=args.script_noise_scale * 0.15,
            size=(2, 3),
        ).astype(np.float32) * max_delta.astype(np.float32)
        noise[:, 1] *= 0.25
        target += noise

    delta = _macro_delta_to_action_delta(target - picker_pos, env, action_space)
    action = np.concatenate([delta, grip.reshape(-1, 1)], axis=1).reshape(-1)
    return clip_action(action, action_space)


def cloth_geometric_v2_action(env, step_idx, args):
    """Two-stage ClothFlatten policy for high-success imitation demos.

    Stage one stretches the dominant axis. Stage two releases and stretches the
    perpendicular axis, which usually produces better coverage than a single pull.
    """
    if args.env_name != "ClothFlatten":
        raise ValueError("cloth_geometric_v2 policy only supports ClothFlatten")

    action_space = env.action_space
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    picker_pos = shape_states[:2, :3].astype(np.float32)
    particles = np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3].astype(np.float32)
    points_xz = particles[:, [0, 2]]
    center_xz = points_xz.mean(axis=0)
    principal = _principal_axis_xz(points_xz)
    perp = np.array([-principal[1], principal[0]], dtype=np.float32)

    horizon = max(int(env.horizon), 1)
    frac = float(step_idx) / float(horizon)
    second_stage = frac >= 0.48
    local_frac = frac / 0.48 if not second_stage else (frac - 0.48) / 0.47
    axis = perp if second_stage else principal
    idx0, idx1 = _extreme_pair_along_axis(points_xz, axis)
    p0 = particles[idx0].copy()
    p1 = particles[idx1].copy()

    bbox_min = points_xz.min(axis=0)
    bbox_max = points_xz.max(axis=0)
    half_extent = 0.5 * np.linalg.norm(bbox_max - bbox_min)
    pull_half = float(np.clip(half_extent * (1.0 + float(args.cloth_pull_scale)), 0.18, 0.46))

    target = np.zeros((2, 3), dtype=np.float32)
    grip = np.zeros(2, dtype=np.float32)

    if frac >= 0.95:
        target[0] = p0 + np.array([0.0, 0.12, 0.0], dtype=np.float32)
        target[1] = p1 + np.array([0.0, 0.12, 0.0], dtype=np.float32)
    elif local_frac < 0.18:
        target[0] = p0 + np.array([0.0, 0.09, 0.0], dtype=np.float32)
        target[1] = p1 + np.array([0.0, 0.09, 0.0], dtype=np.float32)
    elif local_frac < 0.30:
        target[0] = p0 + np.array([0.0, 0.020, 0.0], dtype=np.float32)
        target[1] = p1 + np.array([0.0, 0.020, 0.0], dtype=np.float32)
        grip[:] = 1.0
    elif local_frac < 0.86:
        pull_frac = (local_frac - 0.30) / 0.56
        pull_frac = 0.5 - 0.5 * np.cos(np.pi * pull_frac)
        target_xz_0 = center_xz + axis * pull_half
        target_xz_1 = center_xz - axis * pull_half
        height = float(args.cloth_lift_height)
        end0 = np.array([target_xz_0[0], height, target_xz_0[1]], dtype=np.float32)
        end1 = np.array([target_xz_1[0], height, target_xz_1[1]], dtype=np.float32)
        start0 = p0 + np.array([0.0, height, 0.0], dtype=np.float32)
        start1 = p1 + np.array([0.0, height, 0.0], dtype=np.float32)
        target[0] = (1.0 - pull_frac) * start0 + pull_frac * end0
        target[1] = (1.0 - pull_frac) * start1 + pull_frac * end1
        grip[:] = 1.0
    else:
        target[0] = p0 + np.array([0.0, 0.040, 0.0], dtype=np.float32)
        target[1] = p1 + np.array([0.0, 0.040, 0.0], dtype=np.float32)
        grip[:] = 1.0

    target = _script_noise(target, action_space, args.script_noise_scale * 0.07)
    delta = _macro_delta_to_action_delta(target - picker_pos, env, action_space)
    action = np.concatenate([delta, grip.reshape(-1, 1)], axis=1).reshape(-1)
    return clip_action(action, action_space)


def choose_action(env, args, step_idx):
    if args.policy == "random":
        return sample_action(env.action_space)
    if args.policy == "rope_geometric":
        return rope_geometric_action(env, step_idx, args)
    if args.policy == "rope_geometric_v2":
        return rope_geometric_v2_action(env, step_idx, args)
    if args.policy == "cloth_geometric":
        return cloth_geometric_action(env, step_idx, args)
    if args.policy == "cloth_geometric_v2":
        return cloth_geometric_v2_action(env, step_idx, args)
    raise ValueError("unknown policy: {}".format(args.policy))


def extract_proprio(env, num_picker):
    """Picker xyz (per picker) + holding flag (0/1) -> (4 * num_picker,) float32."""
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    picker_xyz = shape_states[:num_picker, :3].astype(np.float32)
    picked = getattr(env.action_tool, "picked_particles", [None] * num_picker)
    holding = np.array(
        [0.0 if picked[i] is None else 1.0 for i in range(num_picker)],
        dtype=np.float32,
    )
    return np.concatenate([picker_xyz.reshape(-1), holding], axis=0).astype(np.float32)


def extract_compact_state(num_picker):
    """Fixed-dim privileged state - see DATA_FORMAT.md section 5."""
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


def collect_episode(env, args, episode_idx, env_kwargs):
    obs = env.reset()
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

    for step_idx in range(env.horizon):
        action_raw = choose_action(env, args, step_idx)
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

        if done:
            break

    current_config = env.get_current_config()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "env_name": args.env_name,
        "episode_idx": episode_idx,
        "seed": args.seed,
        "policy": "scripted" if args.policy in (
            "rope_geometric",
            "cloth_geometric",
            "rope_geometric_v2",
            "cloth_geometric_v2",
        ) else args.policy,
        "behavior_policy": args.policy,
        "script_noise_scale": float(args.script_noise_scale),
        "script_lateral_scale": float(args.script_lateral_scale),
        "cloth_pull_scale": float(args.cloth_pull_scale),
        "cloth_lift_height": float(args.cloth_lift_height),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env_kwargs": to_jsonable(env_kwargs),
        "config_id": to_jsonable(getattr(env, "current_config_id", None)),
        "current_config": to_jsonable(current_config),
        "action_space_low": to_jsonable(env.action_space.low),
        "action_space_high": to_jsonable(env.action_space.high),
        "action_repeat": int(env.action_repeat),
        "horizon": int(env.horizon),
        "img_size": int(args.img_size),
        "num_picker": int(num_picker),
        "action_dim": int(action_dim),
        "proprio_dim": int(proprio_dim),
        "state_dim": int(STATE_DIM),
    }

    pixels_arr = np.asarray(pixel_frames, dtype=np.uint8)
    action_arr = np.asarray(actions, dtype=np.float32)
    proprio_arr = np.asarray(proprios, dtype=np.float32)
    state_arr = np.asarray(states, dtype=np.float32)
    reward_arr = np.asarray(rewards, dtype=np.float32)
    done_arr = np.asarray(dones, dtype=np.bool_)

    # Length invariants - see DATA_FORMAT.md section 8.
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
                    arrays["action"].shape[0],
                    path,
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
