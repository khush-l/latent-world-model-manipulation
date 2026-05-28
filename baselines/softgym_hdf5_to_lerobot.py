#!/usr/bin/env python3
"""Convert SoftGym v3 HDF5 trajectories to a LeRobot dataset.

The converter maps each non-terminal SoftGym row to one behavior-cloning
sample:

    observation.images.front  <- pixels[t]
    observation.state         <- concat(proprio[t], state[t])
    action                    <- action[t]
    task                      <- language instruction, e.g. "straighten the rope"

This is intended for ACT and SmolVLA baselines. ACT ignores the natural
language task, while SmolVLA uses it for conditioning.

LeRobot changes quickly, so keep `--dry-run` as the first check. It validates
the source dataset and prints the exact feature mapping without importing
LeRobot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np


TASK_INSTRUCTIONS = {
    "RopeFlatten": "straighten the rope",
    "ClothFlatten": "flatten the cloth",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="SoftGym v3 HDF5 file.")
    p.add_argument(
        "--output-root",
        required=True,
        help="Local root directory for the LeRobot dataset.",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="LeRobot/HF dataset id, e.g. khush/softgym-ropeflatten-random-5k75.",
    )
    p.add_argument(
        "--instruction",
        default=None,
        help="Natural-language task. Defaults from env_name when known.",
    )
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument(
        "--use-videos",
        action="store_true",
        help="Store image observations as MP4 videos. By default LeRobot stores images.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input/mapping only; do not import LeRobot or write output.",
    )
    return p.parse_args()


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def source_summary(h5_path: Path) -> Dict[str, object]:
    with h5py.File(h5_path, "r") as f:
        required = ("pixels", "action", "proprio", "state", "episode_idx", "step_idx")
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"{h5_path} missing required dataset(s): {missing}")

        env_name = str(_decode_attr(f.attrs.get("env_name", "unknown")))
        h, w = int(f["pixels"].shape[1]), int(f["pixels"].shape[2])
        n_rows = int(f["pixels"].shape[0])
        n_episodes = int(f.attrs.get("n_episodes", len(np.unique(f["episode_idx"][:]))))
        action_dim = int(f["action"].shape[1])
        proprio_dim = int(f["proprio"].shape[1])
        privileged_state_dim = int(f["state"].shape[1])
        state_dim = proprio_dim + privileged_state_dim

        valid_action_rows = int(np.isfinite(f["action"][:]).all(axis=1).sum())

    return {
        "env_name": env_name,
        "image_shape_hwc": (h, w, 3),
        "n_rows": n_rows,
        "n_episodes": n_episodes,
        "valid_action_rows": valid_action_rows,
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "privileged_state_dim": privileged_state_dim,
        "observation_state_dim": state_dim,
    }


def iter_episode_slices(f: h5py.File) -> Iterable[Tuple[int, int, int]]:
    """Yield (episode_id, start, end_exclusive) over contiguous episode rows."""
    episode_idx = f["episode_idx"][:]
    n = len(episode_idx)
    i = 0
    while i < n:
        j = i + 1
        while j < n and episode_idx[j] == episode_idx[i]:
            j += 1
        yield int(episode_idx[i]), i, j
        i = j


def make_features(summary: Dict[str, object]) -> Dict[str, dict]:
    h, w, c = summary["image_shape_hwc"]
    state_dim = int(summary["observation_state_dim"])
    action_dim = int(summary["action_dim"])
    return {
        "observation.images.front": {
            "dtype": "image",
            "shape": (h, w, c),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{i}" for i in range(action_dim)],
        },
    }


def convert(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_root).resolve()
    summary = source_summary(input_path)
    instruction = args.instruction or TASK_INSTRUCTIONS.get(
        str(summary["env_name"]), str(summary["env_name"])
    )
    features = make_features(summary)

    print("source:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("lerobot mapping:")
    print(json.dumps({"repo_id": args.repo_id, "instruction": instruction, "features": features}, indent=2))

    if args.dry_run:
        print("dry-run only; no LeRobot dataset written")
        return

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "LeRobot is not installed. Install it in the baseline env first, e.g.\n"
            "  pip install 'lerobot[smolvla]'\n"
            "or follow the official LeRobot source install instructions."
        ) from exc

    output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type="softgym_two_picker",
        features=features,
        root=output_root,
        use_videos=args.use_videos,
    )

    n_written_episodes = 0
    n_written_frames = 0
    with h5py.File(input_path, "r") as f:
        for _, start, end in iter_episode_slices(f):
            if args.max_episodes is not None and n_written_episodes >= args.max_episodes:
                break

            # Skip terminal padded row: it has action = NaN and no action target.
            valid_rows = np.arange(start, end)
            action = f["action"][valid_rows]
            valid_rows = valid_rows[np.isfinite(action).all(axis=1)]
            if len(valid_rows) == 0:
                continue

            pixels = f["pixels"][valid_rows]
            proprio = f["proprio"][valid_rows].astype(np.float32)
            privileged_state = f["state"][valid_rows].astype(np.float32)
            action = f["action"][valid_rows].astype(np.float32)

            for t in range(len(valid_rows)):
                obs_state = np.concatenate([proprio[t], privileged_state[t]], axis=0).astype(np.float32)
                frame = {
                    "observation.images.front": pixels[t],
                    "observation.state": obs_state,
                    "action": action[t],
                    "task": instruction,
                }
                dataset.add_frame(frame)
                n_written_frames += 1

            try:
                dataset.save_episode(task=instruction)
            except TypeError:
                dataset.save_episode()
            n_written_episodes += 1
            if n_written_episodes % 100 == 0:
                print(f"saved {n_written_episodes} episodes, {n_written_frames} frames")

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    print(f"wrote LeRobot dataset under {output_root} ({n_written_episodes} episodes)")


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
