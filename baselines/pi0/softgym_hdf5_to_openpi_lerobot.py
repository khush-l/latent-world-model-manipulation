#!/usr/bin/env python3
"""Convert a SoftGym HDF5 dataset to an OpenPI-friendly LeRobot dataset.

This is separate from `baselines/softgym_hdf5_to_lerobot.py` because OpenPI
training needs keys that are consumed by a SoftGym-specific OpenPI transform:

  observation.images.front  <- pixels[t]
  observation.state         <- concat(proprio[t], state[t]) = 23D
  actions                   <- action[t] = 8D
  task                      <- "straighten the rope"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="SoftGym v3 HDF5 file")
    p.add_argument("--output-root", required=True, help="Local LeRobot dataset root")
    p.add_argument("--repo-id", required=True, help="LeRobot dataset id, e.g. khush/softgym-rope-openpi")
    p.add_argument("--instruction", default="straighten the rope")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def iter_episode_slices(f: h5py.File):
    episode_idx = f["episode_idx"][:]
    n = len(episode_idx)
    i = 0
    # episodes are stored back to back in the hdf5, so walk the runs
    while i < n:
        j = i + 1
        while j < n and episode_idx[j] == episode_idx[i]:
            j += 1
        yield int(episode_idx[i]), i, j
        i = j


def summarize(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        required = ("pixels", "action", "proprio", "state", "episode_idx")
        missing = [name for name in required if name not in f]
        if missing:
            raise KeyError(f"{path} missing required dataset(s): {missing}")
        action = f["action"][:]
        # this catches the padded terminal actions before we write anything
        return {
            "env_name": str(_decode_attr(f.attrs.get("env_name", "unknown"))),
            "image_shape_hwc": list(f["pixels"].shape[1:]),
            "n_rows": int(f["pixels"].shape[0]),
            "n_episodes": int(f.attrs.get("n_episodes", len(np.unique(f["episode_idx"][:])))),
            "valid_action_rows": int(np.isfinite(action).all(axis=1).sum()),
            "proprio_dim": int(f["proprio"].shape[1]),
            "privileged_state_dim": int(f["state"].shape[1]),
            "observation_state_dim": int(f["proprio"].shape[1] + f["state"].shape[1]),
            "action_dim": int(f["action"].shape[1]),
        }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_root).resolve()
    summary = summarize(input_path)

    features = {
        "observation.images.front": {
            "dtype": "image",
            "shape": tuple(summary["image_shape_hwc"]),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (summary["observation_state_dim"],),
            "names": [f"state_{i}" for i in range(summary["observation_state_dim"])],
        },
        "actions": {
            "dtype": "float32",
            "shape": (summary["action_dim"],),
            "names": [f"action_{i}" for i in range(summary["action_dim"])],
        },
    }

    print("source:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("openpi lerobot mapping:")
    print(json.dumps({"repo_id": args.repo_id, "instruction": args.instruction, "features": features}, indent=2))
    if args.dry_run:
        print("dry-run only; no LeRobot dataset written")
        return

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit("LeRobot is not installed. Run this through OpenPI's uv environment.") from exc

    output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=args.fps,
        robot_type="softgym_two_picker",
        features=features,
    )

    n_episodes = 0
    n_frames = 0
    with h5py.File(input_path, "r") as f:
        for _, start, end in iter_episode_slices(f):
            if args.max_episodes is not None and n_episodes >= args.max_episodes:
                break
            rows = np.arange(start, end)
            action = f["action"][rows]
            # last row can have nan action, bc there is no next action target
            rows = rows[np.isfinite(action).all(axis=1)]
            if len(rows) == 0:
                continue

            pixels = f["pixels"][rows]
            proprio = f["proprio"][rows].astype(np.float32)
            state = f["state"][rows].astype(np.float32)
            action = f["action"][rows].astype(np.float32)
            for t in range(len(rows)):
                dataset.add_frame(
                    {
                        "observation.images.front": pixels[t],
                        "observation.state": np.concatenate([proprio[t], state[t]], axis=0).astype(np.float32),
                        "actions": action[t],
                        "task": args.instruction,
                    }
                )
                n_frames += 1
            try:
                dataset.save_episode(task=args.instruction)
            except TypeError:
                dataset.save_episode()
            n_episodes += 1
            if n_episodes % 100 == 0:
                print(f"saved {n_episodes} episodes, {n_frames} frames")

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    print(f"wrote {n_episodes} episodes / {n_frames} frames to {output_root}")


if __name__ == "__main__":
    main()

