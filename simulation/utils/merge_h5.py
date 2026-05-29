"""Merge multiple v3 HDF5 trajectory files into one combined file.

Streams pixel chunks (no all-in-RAM) to keep memory low. Adjusts
episode_idx so episodes are globally unique across the merged file, and
re-encodes policy_id with a fresh map per source file.

Usage:
    python simulation/utils/merge_h5.py \\
      --output simulation/data/rope/rope_full_dataset.h5 \\
      --input simulation/data/rope/khush_random_rope_5k75.h5:random \\
      --input simulation/data/rope/ropeflatten_geometric_5k_v3.h5:geometric \\
      --input simulation/data/rope/ropeflatten_manipulate_5k_v3.h5:manipulate
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import h5py
import numpy as np

DEFAULT_CHUNK_ROWS = 4096   # how many rows to stream per copy iter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True)
    p.add_argument("--input", action="append", required=True,
                   help="One per source file. Format: path[:policy_label]")
    p.add_argument("--compression", default="gzip",
                   choices=("gzip", "lzf", "none"))
    p.add_argument("--compression-opts", type=int, default=4)
    p.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    return p.parse_args()


def _parse_input(spec):
    if ":" in spec:
        path, label = spec.rsplit(":", 1)
    else:
        path = spec
        label = os.path.basename(path).replace(".h5", "")
    return path, label


def _required_keys():
    # Top-level columnar datasets that all v3 files share.
    return ("pixels", "action", "proprio", "state",
            "reward", "done", "episode_idx", "step_idx", "policy_id")


def _scan_sources(inputs):
    """Open all sources, sum row counts, collect schema info."""
    infos = []
    total_rows = 0
    total_episodes = 0
    info_keys = None
    for path, label in inputs:
        f = h5py.File(path, "r")
        n_rows = int(f["pixels"].shape[0])
        ep_idx = f["episode_idx"][:]
        n_eps = int(np.unique(ep_idx).size)
        ikeys = sorted(k for k in f.keys() if k.startswith("info_"))
        if info_keys is None:
            info_keys = set(ikeys)
        else:
            info_keys &= set(ikeys)  # intersect: keep only fields present in ALL sources
        infos.append({
            "path": path, "label": label, "file": f,
            "n_rows": n_rows, "n_episodes": n_eps,
        })
        total_rows += n_rows
        total_episodes += n_eps
    return infos, total_rows, total_episodes, sorted(info_keys)


def _create_dataset(out, name, sample, total_rows, compression, compression_opts,
                    chunks_per_row=None):
    """Create a resizable-by-row dataset shaped like (total_rows, *sample.shape[1:])."""
    shape = (total_rows,) + tuple(sample.shape[1:])
    chunks = chunks_per_row if chunks_per_row is not None else \
             ((min(256, total_rows),) + tuple(sample.shape[1:]) if sample.ndim > 1
              else (min(8192, total_rows),))
    kwargs = {"shape": shape, "dtype": sample.dtype, "chunks": chunks}
    if compression != "none":
        kwargs["compression"] = compression
        if compression == "gzip":
            kwargs["compression_opts"] = compression_opts
    return out.create_dataset(name, **kwargs)


def main():
    args = parse_args()
    inputs = [_parse_input(s) for s in args.input]
    infos, total_rows, total_episodes, info_keys = _scan_sources(inputs)

    print(f"merging {len(infos)} sources → {args.output}")
    for info in infos:
        print(f"  {info['label']:12s}  {info['n_episodes']:5d} eps  "
              f"{info['n_rows']:7d} rows  {info['path']}")
    print(f"  TOTAL          {total_episodes:5d} eps  {total_rows:7d} rows")
    print(f"  info_* (kept):  {info_keys}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)

    with h5py.File(args.output, "w") as out:
        # ---- create destination datasets sized for the total row count ----
        first = infos[0]["file"]

        # /pixels — chunked per-frame to mirror the source layout (random reads).
        pix_shape = first["pixels"].shape[1:]
        pix_dtype = first["pixels"].dtype
        out_pixels = out.create_dataset(
            "pixels",
            shape=(total_rows,) + pix_shape,
            dtype=pix_dtype,
            chunks=(1,) + pix_shape,
            compression=(args.compression if args.compression != "none" else None),
            compression_opts=(args.compression_opts if args.compression == "gzip" else None),
        )

        def _make_col(name, sample):
            return _create_dataset(out, name, sample, total_rows,
                                   args.compression, args.compression_opts)

        out_action  = _make_col("action",  first["action"][:1])
        out_proprio = _make_col("proprio", first["proprio"][:1])
        out_state   = _make_col("state",   first["state"][:1])
        out_reward  = _make_col("reward",  first["reward"][:1])
        out_done    = _make_col("done",    first["done"][:1])
        out_epidx   = _make_col("episode_idx", first["episode_idx"][:1])
        out_stepidx = _make_col("step_idx",    first["step_idx"][:1])
        out_polid   = _make_col("policy_id",   first["policy_id"][:1])

        out_infos = {}
        for k in info_keys:
            out_infos[k] = _make_col(k, first[k][:1])

        # ---- stream rows from each source, adjusting episode_idx + policy_id ----
        ep_offset = 0
        row_offset = 0
        new_policy_map = {}  # label → integer id in merged file
        for src_idx, info in enumerate(infos):
            f = info["file"]
            label = info["label"]
            policy_id_for_source = src_idx  # 0=first, 1=second, ...
            new_policy_map[label] = policy_id_for_source

            n_rows = info["n_rows"]
            print(f"\n→ copying {label} ({n_rows} rows) as policy_id={policy_id_for_source}")

            for chunk_start in range(0, n_rows, args.chunk_rows):
                chunk_end = min(chunk_start + args.chunk_rows, n_rows)
                cs = slice(chunk_start, chunk_end)
                ds = slice(row_offset + chunk_start, row_offset + chunk_end)

                out_pixels[ds]  = f["pixels"][cs]
                out_action[ds]  = f["action"][cs]
                out_proprio[ds] = f["proprio"][cs]
                out_state[ds]   = f["state"][cs]
                out_reward[ds]  = f["reward"][cs]
                out_done[ds]    = f["done"][cs]
                out_stepidx[ds] = f["step_idx"][cs]
                # episode_idx: shift by current global offset
                out_epidx[ds]   = f["episode_idx"][cs].astype(np.int64) + ep_offset
                # policy_id: re-encode every row with this source's new id
                out_polid[ds]   = np.full(chunk_end - chunk_start,
                                          policy_id_for_source,
                                          dtype=first["policy_id"].dtype)
                for k in info_keys:
                    out_infos[k][ds] = f[k][cs]

                pct = 100 * (row_offset + chunk_end) / total_rows
                print(f"  [{label}] {chunk_start:7d}/{n_rows:7d}  "
                      f"global pct: {pct:5.1f}%", flush=True)

            row_offset += n_rows
            ep_offset += info["n_episodes"]

        # ---- recompute /stats group on the combined columns ----
        print("\nrecomputing /stats ...")
        stats_grp = out.create_group("stats")
        for name, ds in [("action", out_action),
                         ("proprio", out_proprio),
                         ("state", out_state)]:
            data = ds[:]
            finite = np.isfinite(data).all(axis=tuple(range(1, data.ndim)))
            valid = data[finite]
            mean = valid.mean(axis=0).astype(np.float32) if valid.size else \
                   np.zeros(data.shape[1:], dtype=np.float32)
            std = valid.std(axis=0).astype(np.float32) if valid.size else \
                  np.ones(data.shape[1:], dtype=np.float32)
            std[std < 1e-6] = 1.0
            sub = stats_grp.create_group(name)
            sub.create_dataset("mean", data=mean)
            sub.create_dataset("std", data=std)
            print(f"  /stats/{name} mean[:3]={mean.flatten()[:3]}  std[:3]={std.flatten()[:3]}")

        # ---- root attrs (lightweight metadata) ----
        out.attrs["schema_version"] = 3
        out.attrs["n_total_rows"] = total_rows
        out.attrs["n_episodes"] = total_episodes
        out.attrs["policy_id_map"] = json.dumps(new_policy_map, sort_keys=True)
        out.attrs["sources"] = json.dumps(
            [{"label": i["label"], "path": i["path"],
              "n_episodes": i["n_episodes"], "n_rows": i["n_rows"]}
             for i in infos],
            sort_keys=True,
        )
        # Copy a few env attrs from the first source.
        for k in ("action_dim", "action_repeat", "action_space_high",
                  "action_space_low", "env_name", "horizon", "img_size",
                  "num_picker", "proprio_dim", "state_dim", "info_keys"):
            if k in first.attrs:
                out.attrs[k] = first.attrs[k]

    # Close all source handles.
    for info in infos:
        info["file"].close()

    size_gb = os.path.getsize(args.output) / 1e9
    print(f"\nwrote {args.output}  ({size_gb:.2f} GB, {total_rows} rows, {total_episodes} eps)")
    print(f"policy_id_map: {new_policy_map}")


if __name__ == "__main__":
    main()
