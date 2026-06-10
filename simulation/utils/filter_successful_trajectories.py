#!/usr/bin/env python3
"""Filter SoftGym per-episode NPZ trajectories by final task score.

This is meant for ACT/SmolVLA imitation baselines: collect a larger pool of
scripted candidates, keep the high-success episodes, then convert the filtered
folder with npz_to_hdf5.py and baselines/scripts/convert_lerobot.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Directory containing SoftGym episode .npz files.")
    p.add_argument("--output-dir", required=True, help="Directory to write/copy selected .npz files and reports.")
    p.add_argument("--metric", default="info_normalized_performance", help="NPZ scalar time-series used for ranking.")
    p.add_argument("--threshold", type=float, default=None, help="Keep episodes with final metric >= threshold.")
    p.add_argument("--top-k", type=int, default=None, help="Keep the top K episodes by final metric.")
    p.add_argument("--min-episodes", type=int, default=0, help="If threshold keeps fewer, add best remaining episodes up to this count.")
    p.add_argument("--copy-mode", choices=("copy", "symlink"), default="copy")
    p.add_argument("--dry-run", action="store_true", help="Only print/write reports; do not copy selected episodes.")
    return p.parse_args()


def load_metadata(npz: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "metadata_json" not in npz.files:
        return {}
    blob = npz["metadata_json"]
    raw = blob.item() if blob.shape == () else blob.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def final_finite_value(arr: np.ndarray) -> Optional[float]:
    # some logs have padding/nans, so use the last real value
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def score_episode(path: Path, metric: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        meta = load_metadata(npz)
        score = final_finite_value(npz[metric]) if metric in npz.files else None
        reward = final_finite_value(npz["reward"]) if "reward" in npz.files else None
        total_reward = None
        if "reward" in npz.files:
            r = np.asarray(npz["reward"], dtype=np.float64)
            total_reward = float(np.nansum(r)) if r.size else None
        horizon = int(npz["action"].shape[0]) if "action" in npz.files else None
        behavior_policy = meta.get("behavior_policy", meta.get("policy", "unknown"))
        env_name = meta.get("env_name", "unknown")
    return {
        "path": path,
        "file": path.name,
        "env_name": env_name,
        "behavior_policy": behavior_policy,
        "metric": metric,
        "final_score": score,
        "final_reward": reward,
        "total_reward": total_reward,
        "horizon": horizon,
    }


def discover(input_dir: Path) -> List[Path]:
    paths = sorted(input_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no .npz episodes found under {input_dir}")
    return paths


def select_rows(rows: List[Dict[str, Any]], threshold: Optional[float], top_k: Optional[int], min_episodes: int):
    # rank first, then the threshold/top-k rules are easy to combine
    scored = [r for r in rows if r["final_score"] is not None]
    scored.sort(key=lambda r: r["final_score"], reverse=True)
    selected = []
    if threshold is not None:
        selected = [r for r in scored if r["final_score"] >= threshold]
    else:
        selected = list(scored)
    if top_k is not None:
        selected = selected[:top_k]
    if min_episodes and len(selected) < min_episodes:
        existing = {r["file"] for r in selected}
        for r in scored:
            if r["file"] not in existing:
                selected.append(r)
                existing.add(r["file"])
            if len(selected) >= min_episodes:
                break
    selected.sort(key=lambda r: r["path"].name)
    return selected, scored


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = ["file", "env_name", "behavior_policy", "metric", "final_score", "final_reward", "total_reward", "horizon", "selected"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def copy_selected(selected: List[Dict[str, Any]], output_dir: Path, mode: str) -> None:
    # overwrite selected files so reruns don't leave stale episodes around
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in selected:
        src = r["path"]
        dst = output_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if mode == "symlink":
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [score_episode(p, args.metric) for p in discover(input_dir)]
    selected, scored = select_rows(rows, args.threshold, args.top_k, args.min_episodes)
    selected_files = {r["file"] for r in selected}
    for r in rows:
        r["selected"] = int(r["file"] in selected_files)

    if not args.dry_run:
        copy_selected(selected, output_dir, args.copy_mode)

    write_csv(output_dir / "filter_report.csv", rows)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "metric": args.metric,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "min_episodes": args.min_episodes,
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_selected": len(selected),
        "dry_run": bool(args.dry_run),
    }
    if scored:
        scores = np.asarray([r["final_score"] for r in scored], dtype=np.float64)
        summary.update({
            "score_min": float(np.min(scores)),
            "score_mean": float(np.mean(scores)),
            "score_median": float(np.median(scores)),
            "score_max": float(np.max(scores)),
            "score_p90": float(np.percentile(scores, 90)),
        })
    with (output_dir / "filter_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if selected:
        print("top selected:")
        for r in sorted(selected, key=lambda x: x["final_score"], reverse=True)[:10]:
            print(f"  {r['file']}: {r['final_score']:.4f}")


if __name__ == "__main__":
    main()
