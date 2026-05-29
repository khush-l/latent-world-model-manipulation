"""Sample goal observations from a held-out v3 HDF5 dataset.

For MPC eval we need (initial_state, goal_obs, max_steps) tuples per episode.
This module picks them from a consolidated HDF5 file.

Goal strategies:
  - terminal:    final-frame of an episode (rope flattened, cloth folded, ...)
  - subgoal:     k steps ahead of a random start frame (matches LeWM paper's
                 PushT eval: goal_offset_steps=25)
  - high_perf:   final frame of an episode whose info_normalized_performance
                 ended above a threshold (filters out failed scripted episodes)
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


@dataclass
class GoalSpec:
    """One eval episode specification."""
    episode_idx: int
    start_row: int          # row in the flat HDF5 to begin MPC from
    goal_row: int           # row to use as goal observation
    init_pixels: np.ndarray  # (history_size, C, H, W) uint8
    goal_pixels: np.ndarray  # (C, H, W) uint8
    init_state: np.ndarray   # (15,) compact state at start_row
    goal_perf: float         # info_normalized_performance at goal_row


def _to_imagenet_float(pixels_uint8: np.ndarray) -> np.ndarray:
    """Convert uint8 HWC pixels → float32 CHW ImageNet-normalized."""
    if pixels_uint8.ndim == 3:
        x = pixels_uint8.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))                      # CHW
        x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return x
    # (T, H, W, C) → (T, C, H, W)
    x = pixels_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    x = (x - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    return x


def sample_goals(
    h5_path: str,
    n_goals: int,
    history_size: int = 3,
    strategy: str = "high_perf",
    min_perf: float = 0.85,
    subgoal_offset: int = 25,
    seed: int = 42,
) -> List[GoalSpec]:
    """Sample `n_goals` (start, goal) pairs from a v3 HDF5 dataset.

    Args:
        h5_path: path to consolidated v3 HDF5.
        n_goals: number of episodes to return.
        history_size: number of context frames the planner needs (3 for LeWM).
        strategy: 'terminal' | 'subgoal' | 'high_perf'.
        min_perf: only used when strategy='high_perf' — filter on final
                  info_normalized_performance.
        subgoal_offset: k for strategy='subgoal'.
        seed: RNG seed for episode selection.

    Returns:
        list of `GoalSpec` of length <= n_goals.
    """
    rng = np.random.RandomState(seed)
    out: List[GoalSpec] = []
    with h5py.File(h5_path, "r") as f:
        ep = f["episode_idx"][:]
        perf = f["info_normalized_performance"][:] if "info_normalized_performance" in f else None
        unique_eps = np.unique(ep)
        rng.shuffle(unique_eps)

        for e in unique_eps:
            if len(out) >= n_goals:
                break
            mask = ep == e
            rows = np.where(mask)[0]
            if rows.size < history_size + 1:
                continue
            # Episode terminal row = last non-NaN row (we pad with NaN at end).
            # Use the last action row + 1 as the terminal observation row.
            terminal_row = int(rows[-1])

            # start_row: where we begin executing MPC. Needs history_size frames
            # of context preceding it.
            if strategy == "subgoal":
                ep_len = rows.size
                if ep_len < history_size + subgoal_offset + 1:
                    continue
                start_in_ep = rng.randint(history_size, ep_len - subgoal_offset)
                start_row = int(rows[start_in_ep])
                goal_row = int(rows[start_in_ep + subgoal_offset])
            elif strategy == "terminal":
                start_row = int(rows[history_size])
                goal_row = terminal_row
            elif strategy == "high_perf":
                final_perf = float(perf[terminal_row - 1]) if perf is not None else 1.0
                if final_perf < min_perf:
                    continue
                start_row = int(rows[history_size])
                goal_row = terminal_row
            else:
                raise ValueError("unknown strategy: %r" % strategy)

            init_pixels_uint8 = f["pixels"][start_row - history_size:start_row]   # (H, H_img, W, C)
            goal_pixels_uint8 = f["pixels"][goal_row]                              # (H_img, W, C)
            init_state = f["state"][start_row].astype(np.float32)

            spec = GoalSpec(
                episode_idx=int(e),
                start_row=start_row,
                goal_row=goal_row,
                init_pixels=init_pixels_uint8.copy(),
                goal_pixels=goal_pixels_uint8.copy(),
                init_state=init_state,
                goal_perf=float(perf[goal_row - 1]) if perf is not None and goal_row > 0 else 0.0,
            )
            out.append(spec)
    return out


def goalspec_to_torch_inputs(spec: GoalSpec, device: str = "cuda"):
    """Helper: convert a GoalSpec into the tensors the planner expects."""
    import torch
    init_px_f = _to_imagenet_float(spec.init_pixels)               # (H, C, H_img, W)
    goal_px_f = _to_imagenet_float(spec.goal_pixels)               # (C, H_img, W)
    return {
        "pixels_history": torch.from_numpy(init_px_f).float().to(device),
        "goal_pixels":    torch.from_numpy(goal_px_f).unsqueeze(0).float().to(device),  # (1, C, H, W)
        "init_state":     torch.from_numpy(spec.init_state).float().to(device),
    }


if __name__ == "__main__":
    # Self-test: load goals from the rope HDF5 and print a summary.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="simulation/data/rope/khush_random_rope_5k75.h5")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--strategy", default="terminal")
    args = p.parse_args()

    specs = sample_goals(args.h5, args.n, strategy=args.strategy)
    for s in specs:
        print(f"ep={s.episode_idx:4d}  start_row={s.start_row:6d}  "
              f"goal_row={s.goal_row:6d}  goal_perf={s.goal_perf:.3f}  "
              f"init_state[:3]={s.init_state[:3]}")
