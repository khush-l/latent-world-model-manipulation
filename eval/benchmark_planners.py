"""Head-to-head benchmark: gradient-based latent planner vs CEM.

Produces the headline experiment for the "fast, parameter-efficient planning"
hypothesis. For a set of held-out (history, goal) pairs, runs both planners and
reports:
    - planning latency (ms/plan)
    - final terminal latent cost achieved (lower = better plan under the model)
    - rollout-step count (a hardware-independent compute proxy)

Both planners are zero-parameter (they only optimize actions), and share the
same frozen 18M-parameter world model — so this isolates the planning algorithm.

Usage:
    python eval/benchmark_planners.py \\
        --ckpt training/runs/rope_full_mixed_v1/ckpt_final.pt \\
        --eval-h5 simulation/data/rope/rope_full_dataset.h5 \\
        --n-episodes 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from model import build_lewm  # noqa: E402
from rollout import encode_pixels  # noqa: E402
from cem_planner import CEMConfig, CEMPlanner  # noqa: E402
from gradient_planner import GradientPlanConfig, GradientPlanner  # noqa: E402
from sample_goals import sample_goals, goalspec_to_torch_inputs  # noqa: E402


ACTION_LOW = torch.tensor([-0.01, -0.01, -0.01, 0.0, -0.01, -0.01, -0.01, 0.0])
ACTION_HIGH = torch.tensor([0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 1.0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval-h5", required=True)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--goal-strategy", default="high_perf")
    p.add_argument("--min-perf", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--plan-horizon", type=int, default=5)

    # CEM config
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-iters", type=int, default=30)
    p.add_argument("--cem-topk", type=int, default=30)

    # Gradient config
    p.add_argument("--grad-restarts", type=int, default=32)
    p.add_argument("--grad-iters", type=int, default=15)
    p.add_argument("--grad-lr", type=float, default=0.1)

    p.add_argument("--out", default=None)
    return p.parse_args()


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    model = build_lewm(
        img_size=cfg.get("img_size", 128), patch_size=cfg.get("patch_size", 16),
        embed_dim=cfg.get("embed_dim", 192),
        predictor_depth=cfg.get("predictor_depth", 6),
        predictor_heads=cfg.get("predictor_heads", 16),
        predictor_dim_head=cfg.get("predictor_dim_head", 64),
        predictor_dropout=cfg.get("predictor_dropout", 0.1),
        history_size=cfg.get("history_size", 3), num_preds=cfg.get("num_preds", 1),
        action_dim=8,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_model(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params (frozen)")

    specs = sample_goals(args.eval_h5, args.n_episodes,
                         history_size=args.history_size,
                         strategy=args.goal_strategy, min_perf=args.min_perf,
                         seed=args.seed)
    print(f"benchmarking on {len(specs)} held-out goals\n")

    cem = CEMPlanner(
        model, CEMConfig(num_samples=args.cem_samples, n_iters=args.cem_iters,
                         topk=args.cem_topk, plan_horizon=args.plan_horizon),
        ACTION_LOW, ACTION_HIGH, args.history_size, device=str(device))
    grad = GradientPlanner(
        model, GradientPlanConfig(plan_horizon=args.plan_horizon,
                                  n_restarts=args.grad_restarts,
                                  n_iters=args.grad_iters, lr=args.grad_lr),
        ACTION_LOW, ACTION_HIGH, args.history_size, device=str(device))

    # Rollout-step compute proxy (hardware independent).
    cem_steps = args.cem_iters * args.cem_samples * args.plan_horizon
    grad_steps = args.grad_restarts * args.grad_iters * args.plan_horizon * 2  # fwd+bwd
    print(f"rollout-step proxy:  CEM={cem_steps:,}   gradient={grad_steps:,}  "
          f"({cem_steps/grad_steps:.1f}× fewer for gradient)\n")

    def final_cost(planner, inp, goal_emb, hist_act):
        # Re-roll the planner's chosen sequence to score terminal cost identically
        # for both planners (uses the no-grad rollout).
        from rollout import rollout, goal_cost
        if isinstance(planner, GradientPlanner):
            seq = planner.plan_full_sequence(inp["pixels_history"], goal_emb, hist_act)
        else:
            planner.plan(inp["pixels_history"], goal_emb, hist_act)
            seq = planner.prev_mean
        full_act = torch.cat([hist_act, seq], dim=0).unsqueeze(0).unsqueeze(0)
        full_px = inp["pixels_history"].unsqueeze(0).unsqueeze(0)
        emb = rollout(model, full_px, full_act, history_size=args.history_size)
        return float(goal_cost(emb, goal_emb.unsqueeze(0)).item())

    rows = []
    for i, spec in enumerate(specs):
        inp = goalspec_to_torch_inputs(spec, device=str(device))
        goal_emb = encode_pixels(model, inp["goal_pixels"].unsqueeze(0)).squeeze(0).squeeze(0)
        hist_act = torch.zeros(args.history_size, 8, device=device)

        # --- CEM ---
        cem.reset()
        _sync(device); t0 = time.perf_counter()
        cem.plan(inp["pixels_history"], goal_emb, hist_act)
        _sync(device); cem_lat = time.perf_counter() - t0
        cem_cost = final_cost(cem, inp, goal_emb, hist_act)

        # --- Gradient ---
        grad.reset()
        _sync(device); t0 = time.perf_counter()
        grad.plan(inp["pixels_history"], goal_emb, hist_act)
        _sync(device); grad_lat = time.perf_counter() - t0
        grad_cost = final_cost(grad, inp, goal_emb, hist_act)

        rows.append({"episode_idx": spec.episode_idx,
                     "cem_latency_s": cem_lat, "cem_cost": cem_cost,
                     "grad_latency_s": grad_lat, "grad_cost": grad_cost})
        print(f"  [{i+1:2d}/{len(specs)}] "
              f"CEM: {cem_lat*1000:6.0f}ms cost={cem_cost:7.3f}  |  "
              f"GRAD: {grad_lat*1000:6.0f}ms cost={grad_cost:7.3f}  |  "
              f"speedup={cem_lat/grad_lat:4.1f}×")

    cem_lat = np.array([r["cem_latency_s"] for r in rows])
    grad_lat = np.array([r["grad_latency_s"] for r in rows])
    cem_cost = np.array([r["cem_cost"] for r in rows])
    grad_cost = np.array([r["grad_cost"] for r in rows])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'':16s}{'CEM':>14s}{'Gradient':>14s}")
    print(f"  {'latency (ms)':16s}{cem_lat.mean()*1000:>14.0f}{grad_lat.mean()*1000:>14.0f}")
    print(f"  {'terminal cost':16s}{cem_cost.mean():>14.3f}{grad_cost.mean():>14.3f}")
    print(f"  {'rollout steps':16s}{cem_steps:>14,}{grad_steps:>14,}")
    print(f"\n  mean speedup:        {(cem_lat/grad_lat).mean():.1f}×")
    print(f"  cost ratio (grad/cem): {grad_cost.mean()/cem_cost.mean():.2f} "
          f"({'gradient better' if grad_cost.mean() < cem_cost.mean() else 'CEM better'})")

    out = Path(args.out) if args.out else Path(args.ckpt).parent / "planner_benchmark.json"
    out.write_text(json.dumps({
        "config": vars(args), "n_params": n_params,
        "cem_rollout_steps": cem_steps, "grad_rollout_steps": grad_steps,
        "rows": rows,
        "summary": {
            "cem_latency_ms_mean": float(cem_lat.mean() * 1000),
            "grad_latency_ms_mean": float(grad_lat.mean() * 1000),
            "cem_cost_mean": float(cem_cost.mean()),
            "grad_cost_mean": float(grad_cost.mean()),
            "mean_speedup": float((cem_lat / grad_lat).mean()),
        },
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
