#!/usr/bin/env python3
"""Convert a directory of per-episode NPZs (v3 schema) into a single
consolidated HDF5 file ready for LeWM-style training.

Layout follows DATA_FORMAT.md §4: flat columnar root datasets, one row per
timestep, with `episode_idx` / `step_idx` / `policy_id` book-keeping columns.
Per-step columns (`action`, `reward`, `done`, `info_*`) are right-padded at
the terminal frame so every column has length `T + 1` per episode.

Run this OUTSIDE the SoftGym Docker, in a modern Python env with h5py.
(Compatible with Python 3.6+ so it can also run inside the SoftGym Docker
when h5py is installed there.)
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


SCHEMA_VERSION = 3
EXPECTED_KEYS = ("pixels", "action", "proprio", "state", "reward", "done", "metadata_json")
POLICY_ID_MAP = {
    "random": 0,
    "scripted": 1,
    "cem_expert": 2,
    "perturbation": 3,
    "configure": 4,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Directory containing per-episode NPZs.")
    p.add_argument("--output", required=True, help="Path to write the consolidated .h5 file.")
    p.add_argument("--include-full-state", action="store_true",
                   help="Also copy per-episode full_state_* arrays into /full_state/<ep>/.")
    p.add_argument("--compression", default="gzip", choices=("gzip", "lzf", "none"))
    p.add_argument("--compression-opts", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="Only convert the first N episodes (for smoke tests).")
    return p.parse_args()


def discover_episodes(input_dir: Path):
    paths = sorted(glob.glob(str(input_dir / "*.npz")))
    if not paths:
        raise SystemExit(f"no NPZ files found under {input_dir}")
    return [Path(p) for p in paths]


def load_metadata(npz) -> dict:
    blob = npz["metadata_json"]
    if blob.dtype.kind == "U":
        return json.loads(str(blob))
    return json.loads(blob.item() if blob.shape == () else blob.tolist())


def validate_episode(npz, meta: dict, path: Path):
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {meta.get('schema_version')} != {SCHEMA_VERSION}")
    for k in EXPECTED_KEYS:
        if k not in npz.files:
            raise ValueError(f"{path}: missing required key {k!r}")
    T = int(npz["action"].shape[0])
    if npz["pixels"].shape[0] != T + 1:
        raise ValueError(f"{path}: pixels has {npz['pixels'].shape[0]} rows, expected T+1={T+1}")
    if npz["proprio"].shape[0] != T + 1:
        raise ValueError(f"{path}: proprio has {npz['proprio'].shape[0]} rows, expected T+1={T+1}")
    if npz["state"].shape[0] != T + 1:
        raise ValueError(f"{path}: state has {npz['state'].shape[0]} rows, expected T+1={T+1}")
    if npz["reward"].shape[0] != T or npz["done"].shape[0] != T:
        raise ValueError(f"{path}: reward/done length mismatch (expected {T})")


def pad_to_T1(arr: np.ndarray, fill, T1: int) -> np.ndarray:
    """Right-pad a length-T per-step array up to length T+1."""
    if arr.shape[0] == T1:
        return arr
    pad_shape = (T1 - arr.shape[0],) + arr.shape[1:]
    pad = np.full(pad_shape, fill, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def info_keys_across_episodes(paths):
    keys = set()
    for p in paths:
        with np.load(p, allow_pickle=True) as npz:
            for k in npz.files:
                if k.startswith("info_") and np.asarray(npz[k]).ndim == 1:
                    keys.add(k)
    return sorted(keys)


def compression_kwargs(args):
    if args.compression == "none":
        return {}
    if args.compression == "lzf":
        return {"compression": "lzf"}
    return {"compression": "gzip", "compression_opts": args.compression_opts}


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episode_paths = discover_episodes(input_dir)
    if args.limit is not None:
        episode_paths = episode_paths[: args.limit]

    # First pass: validate, count rows, discover info_* schema, capture global metadata.
    n_total = 0
    n_episodes = len(episode_paths)
    info_keys = info_keys_across_episodes(episode_paths)
    per_episode_meta = []
    global_attrs = None
    img_size = None
    action_dim = proprio_dim = state_dim = None

    for path in episode_paths:
        with np.load(path, allow_pickle=True) as npz:
            meta = load_metadata(npz)
            validate_episode(npz, meta, path)
            T = int(npz["action"].shape[0])
            T1 = T + 1
            n_total += T1
            per_episode_meta.append((path, meta, T1))

            if global_attrs is None:
                global_attrs = {
                    "env_name": str(meta["env_name"]),
                    "num_picker": int(meta["num_picker"]),
                    "action_dim": int(meta["action_dim"]),
                    "proprio_dim": int(meta["proprio_dim"]),
                    "state_dim": int(meta["state_dim"]),
                    "action_repeat": int(meta["action_repeat"]),
                    "horizon": int(meta["horizon"]),
                    "img_size": int(meta["img_size"]),
                    "action_space_low": np.asarray(meta["action_space_low"], dtype=np.float32),
                    "action_space_high": np.asarray(meta["action_space_high"], dtype=np.float32),
                }
                img_size = global_attrs["img_size"]
                action_dim = global_attrs["action_dim"]
                proprio_dim = global_attrs["proprio_dim"]
                state_dim = global_attrs["state_dim"]
            else:
                if int(meta["img_size"]) != img_size:
                    raise ValueError(f"{path}: img_size mismatch ({meta['img_size']} vs {img_size})")
                if int(meta["action_dim"]) != action_dim:
                    raise ValueError(f"{path}: action_dim mismatch")

    print(f"discovered {n_episodes} episodes, {n_total} total rows")
    print(f"info columns: {info_keys}")

    ckwargs = compression_kwargs(args)

    with h5py.File(output_path, "w") as out:
        # Root datasets — preallocate then fill episode-by-episode.
        ds_pixels = out.create_dataset(
            "pixels", shape=(n_total, img_size, img_size, 3), dtype=np.uint8,
            chunks=(1, img_size, img_size, 3), **ckwargs,
        )
        ds_action = out.create_dataset("action", shape=(n_total, action_dim), dtype=np.float32,
                                       chunks=(min(256, n_total), action_dim), **ckwargs)
        ds_proprio = out.create_dataset("proprio", shape=(n_total, proprio_dim), dtype=np.float32,
                                        chunks=(min(256, n_total), proprio_dim), **ckwargs)
        ds_state = out.create_dataset("state", shape=(n_total, state_dim), dtype=np.float32,
                                      chunks=(min(256, n_total), state_dim), **ckwargs)
        ds_reward = out.create_dataset("reward", shape=(n_total,), dtype=np.float32,
                                       chunks=(min(256, n_total),), **ckwargs)
        ds_done = out.create_dataset("done", shape=(n_total,), dtype=np.uint8,
                                     chunks=(min(256, n_total),), **ckwargs)
        ds_ep = out.create_dataset("episode_idx", shape=(n_total,), dtype=np.int32,
                                   chunks=(min(256, n_total),), **ckwargs)
        ds_step = out.create_dataset("step_idx", shape=(n_total,), dtype=np.int32,
                                     chunks=(min(256, n_total),), **ckwargs)
        ds_policy = out.create_dataset("policy_id", shape=(n_total,), dtype=np.int8,
                                       chunks=(min(256, n_total),), **ckwargs)
        ds_info = {
            k: out.create_dataset(k, shape=(n_total,), dtype=np.float32,
                                  chunks=(min(256, n_total),), **ckwargs)
            for k in info_keys
        }

        episodes_grp = out.create_group("episodes")
        full_state_grp = out.create_group("full_state") if args.include_full_state else None

        row = 0
        for ep_idx, (path, meta, T1) in enumerate(per_episode_meta):
            with np.load(path, allow_pickle=True) as npz:
                T = T1 - 1
                end = row + T1

                ds_pixels[row:end] = np.asarray(npz["pixels"], dtype=np.uint8)
                ds_proprio[row:end] = np.asarray(npz["proprio"], dtype=np.float32)
                ds_state[row:end] = np.asarray(npz["state"], dtype=np.float32)

                action = np.asarray(npz["action"], dtype=np.float32)
                ds_action[row:end] = pad_to_T1(action, np.nan, T1)

                reward = np.asarray(npz["reward"], dtype=np.float32)
                ds_reward[row:end] = pad_to_T1(reward, np.nan, T1)

                done = np.asarray(npz["done"], dtype=np.bool_).astype(np.uint8)
                ds_done[row:end] = pad_to_T1(done, 0, T1)

                ds_ep[row:end] = np.full(T1, ep_idx, dtype=np.int32)
                ds_step[row:end] = np.arange(T1, dtype=np.int32)

                policy_name = str(meta.get("policy", "random"))
                pid = POLICY_ID_MAP.get(policy_name, -1)
                if pid < 0:
                    print(f"  WARNING: unknown policy {policy_name!r} in {path.name}, using -1",
                          file=sys.stderr)
                ds_policy[row:end] = np.full(T1, pid, dtype=np.int8)

                for k in info_keys:
                    if k in npz.files:
                        vals = np.asarray(npz[k], dtype=np.float32)
                        ds_info[k][row:end] = pad_to_T1(vals, np.nan, T1)
                    else:
                        ds_info[k][row:end] = np.full(T1, np.nan, dtype=np.float32)

                ep_grp = episodes_grp.create_group(f"{ep_idx:06d}")
                ep_grp.attrs["source_npz"] = str(path.name)
                ep_grp.attrs["row_start"] = row
                ep_grp.attrs["row_end"] = end
                ep_grp.attrs["T"] = T
                for k in ("policy", "seed", "config_id", "created_at"):
                    if k in meta and meta[k] is not None:
                        ep_grp.attrs[k] = str(meta[k]) if not isinstance(meta[k], (int, float)) else meta[k]

                if full_state_grp is not None:
                    fs_keys = [k for k in npz.files if k.startswith("full_state_")]
                    if fs_keys:
                        sub = full_state_grp.create_group(f"{ep_idx:06d}")
                        for k in fs_keys:
                            arr = np.asarray(npz[k])
                            sub.create_dataset(k.replace("full_state_", ""), data=arr, **ckwargs)

                row = end
                print(f"  [{ep_idx + 1}/{n_episodes}] {path.name}: T={T}")

        assert row == n_total, (row, n_total)

        # Root attrs
        out.attrs["schema_version"] = SCHEMA_VERSION
        for k, v in global_attrs.items():
            out.attrs[k] = v
        out.attrs["policy_id_map"] = json.dumps(POLICY_ID_MAP, sort_keys=True)
        out.attrs["n_episodes"] = n_episodes
        out.attrs["n_total_rows"] = n_total
        out.attrs["info_keys"] = json.dumps(info_keys)

        # Stats for normalizer warm-cache (LeWM recomputes these but it lets us verify).
        stats = out.create_group("stats")
        for name, ds in (("action", ds_action), ("proprio", ds_proprio), ("state", ds_state)):
            arr = ds[:]
            finite_mask = np.isfinite(arr).all(axis=tuple(range(1, arr.ndim)))
            valid = arr[finite_mask]
            if valid.size == 0:
                continue
            g = stats.create_group(name)
            g.create_dataset("mean", data=valid.mean(axis=0).astype(np.float32))
            g.create_dataset("std", data=valid.std(axis=0).astype(np.float32))

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
