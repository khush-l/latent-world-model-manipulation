"""Score the open-loop replay of action sequences emitted by `run_mpc.py --mode replay`.

For each plan NPZ in <plan_dir>:
  invoke simulation/docker/softgym-local.sh run "python utils/replay_actions.py ..."
  capture the replayed-trajectory NPZ
  collect per-episode achieved info_normalized_performance

Aggregates into a summary table. Doesn't require torch — just numpy + bash.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan-dir", required=True,
                   help="Directory of plan NPZs (output of run_mpc.py --mode replay).")
    p.add_argument("--replay-dir", default="eval/runs/replay_executed",
                   help="Where to put per-episode replayed-trajectory NPZs.")
    p.add_argument("--out", default=None,
                   help="JSON summary path. Defaults to <plan_dir>/replay_scores.json.")
    p.add_argument("--env-name", default="RopeFlatten")
    p.add_argument("--num-variations", type=int, default=200)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only score the first N plan NPZs (for smoke).")
    return p.parse_args()


def main():
    args = parse_args()
    plan_dir = Path(args.plan_dir).resolve()
    replay_dir = Path(args.replay_dir).resolve()
    replay_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else plan_dir / "replay_scores.json"

    plan_npzs = sorted(plan_dir.glob("plan_ep*.npz"))
    if args.limit:
        plan_npzs = plan_npzs[:args.limit]
    if not plan_npzs:
        raise SystemExit(f"no plan NPZs found under {plan_dir}")

    docker_script = PROJECT_ROOT / "simulation" / "docker" / "softgym-local.sh"
    rows = []
    for i, plan_path in enumerate(plan_npzs):
        rel_plan = plan_path.relative_to(PROJECT_ROOT / "simulation") \
                  if plan_path.is_relative_to(PROJECT_ROOT / "simulation") \
                  else plan_path
        rel_replay = (replay_dir / plan_path.name).relative_to(PROJECT_ROOT / "simulation") \
                     if (replay_dir).is_relative_to(PROJECT_ROOT / "simulation") \
                     else (replay_dir / plan_path.name)

        cmd_inside = (
            f"python utils/replay_actions.py "
            f"--input {shlex.quote(str(rel_plan))} "
            f"--output {shlex.quote(str(rel_replay))} "
            f"--env-name {args.env_name} "
            f"--num-variations {args.num_variations} "
            f"--img-size {args.img_size}"
        )
        full_cmd = [str(docker_script), "run", cmd_inside]
        print(f"[{i+1}/{len(plan_npzs)}] {plan_path.name}")
        subprocess.run(full_cmd, check=True)

        # Read both NPZs to compute the headline metric.
        plan = np.load(plan_path, allow_pickle=True)
        repl_npz_path = (PROJECT_ROOT / "simulation" / rel_replay) \
                        if not rel_replay.is_absolute() else rel_replay
        try:
            executed = np.load(repl_npz_path, allow_pickle=True)
        except FileNotFoundError:
            print(f"  WARNING: replay NPZ not found at {repl_npz_path}")
            continue
        perf = executed["info_normalized_performance"] if "info_normalized_performance" in executed.files else None
        achieved = float(perf[-1]) if perf is not None else None
        rows.append({
            "plan_npz": str(plan_path),
            "replay_npz": str(repl_npz_path),
            "episode_idx": int(plan["episode_idx"]),
            "config_id": int(plan["config_id"]),
            "goal_perf": float(plan["goal_perf"]),
            "achieved_perf": achieved,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))

    achieved = np.array([r["achieved_perf"] for r in rows if r["achieved_perf"] is not None])
    goals    = np.array([r["goal_perf"]     for r in rows])
    print("\n========== REPLAY SCORE SUMMARY ==========")
    print(f"episodes:         {len(rows)}")
    if achieved.size:
        print(f"achieved perf:    mean={achieved.mean():.3f}  median={np.median(achieved):.3f}")
        print(f"  > 0.5 success:  {(achieved > 0.5).mean():.1%}")
        print(f"  > 0.8 success:  {(achieved > 0.8).mean():.1%}")
    print(f"goal perf (target): mean={goals.mean():.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
