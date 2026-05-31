"""Receding-horizon MPC over the LeWM cost model (stable-worldmodel style).

Mirrors references/.../policy.py:WorldModelPolicy.get_action: maintain a
history buffer, replan every `receding_horizon` steps with the CEM solver,
warm-start the next plan from the previous tail, execute actions
(denormalized to the env box). Goal = a single fixed final-flat goal image
(the expert reference's last frame), encoded once.

This replaces eval/run_mpc.py:mode_imitate (and the old gradient/cem planners).

Usage:
    python eval/mpc_runner.py --ckpt training/runs/rope_proprio_v1/ckpt_final.pt \
        --n-episodes 8 --seed 0 --planner cem --save-gifs \
        --out-dir eval/runs/charts/mpc_lewm
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from model import build_lewm                       # noqa: E402
from env_client import SubprocessSoftgym           # noqa: E402
from sample_goals import _to_imagenet_float        # noqa: E402
from lewm_cost import LeWMCost                      # noqa: E402
from solvers import CEMSolver                       # noqa: E402

# Env action box for RopeFlatten (2 pickers: delta xyz + grip).
ACTION_LOW = torch.tensor([-0.01, -0.01, -0.01, 0., -0.01, -0.01, -0.01, 0.])
ACTION_HIGH = torch.tensor([0.01, 0.01, 0.01, 1., 0.01, 0.01, 0.01, 1.])


def _column_stats(data):
    """Per-column (mean, std) from rows with no NaNs (matches training/dataset.py)."""
    finite = np.isfinite(data).all(axis=1)
    valid = data[finite]
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def load_stats(h5_path, device):
    """Action + proprio z-score stats from the training data."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        a_mean, a_std = _column_stats(f["action"][:].astype(np.float32))
        if "proprio" in f:
            p_mean, p_std = _column_stats(f["proprio"][:].astype(np.float32))
        else:
            p_mean = p_std = None
    t = lambda x: None if x is None else torch.from_numpy(x).to(device)
    return t(a_mean), t(a_std), t(p_mean), t(p_std)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="training/runs/rope_proprio_v1/ckpt_final.pt")
    p.add_argument("--n-episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-variations", type=int, default=200)
    p.add_argument("--eval-budget", type=int, default=75)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--horizon", type=int, default=5, help="CEM plan length (future actions)")
    p.add_argument("--receding-horizon", type=int, default=1, help="steps executed before replanning")
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-iters", type=int, default=30)
    p.add_argument("--cem-topk", type=int, default=30)
    p.add_argument("--planner", choices=("cem", "random"), default="cem")
    p.add_argument("--confine-2d", action="store_true",
                   help="Confine pickers to the ground plane (freeze vertical action dims).")
    p.add_argument("--grab-steps", type=int, default=0,
                   help="Scripted descend+grip steps before MPC, to put pickers on the rope plane.")
    p.add_argument("--success-thresh", type=float, default=0.8)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--env-name", default="RopeFlatten")
    p.add_argument("--out-dir", default="eval/runs/mpc_lewm")
    p.add_argument("--save-gifs", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_dir = out_dir / "gifs"; gif_dir.mkdir(exist_ok=True)
    H = args.history_size

    # --- model (proprio auto-detected from sibling config.json) ---
    cfg_path = Path(args.ckpt).parent / "config.json"
    train_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    use_proprio = bool(train_cfg.get("use_proprio", False))
    model = build_lewm(history_size=H, use_proprio=use_proprio).to(device)
    state = torch.load(args.ckpt, map_location=device)
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state); model.eval()

    data_h5 = train_cfg.get("data", str(PROJECT_ROOT / "simulation/data/rope/rope_full_dataset.h5"))
    a_mean, a_std, p_mean, p_std = load_stats(data_h5, device)
    print(f"loaded {args.ckpt} (use_proprio={use_proprio}); stats from {data_h5}")

    cost = LeWMCost(model, ACTION_LOW, ACTION_HIGH, history_size=H,
                    action_mean=a_mean, action_std=a_std,
                    proprio_mean=p_mean, proprio_std=p_std, device=device)
    # --confine-2d: pin each picker's vertical delta (action dims 1 & 5, the
    # y/up axis) to raw 0 so the pickers only slide in the ground plane. The
    # frozen value is raw-0 expressed in the solver's normalized space.
    freeze_idx = freeze_val = None
    if args.confine_2d:
        vdims = [1, 5]
        freeze_idx = vdims
        freeze_val = [float((-a_mean[d] / a_std[d]).item()) for d in vdims]
        print(f"  [confine-2d] freezing vertical action dims {vdims} to raw 0")
    solver = CEMSolver(cost, horizon=args.horizon, num_samples=args.cem_samples,
                       n_steps=args.cem_iters, topk=args.cem_topk, device=device, seed=args.seed,
                       freeze_idx=freeze_idx, freeze_val=freeze_val)
    A = cost.action_dim

    env = SubprocessSoftgym(env_name=args.env_name, num_variations=args.num_variations,
                            img_size=args.img_size, num_picker=2)
    rng = np.random.RandomState(args.seed)

    def norm_a(a_raw):  # (..., A) raw -> normalized
        return (a_raw.to(device) - a_mean) / a_std

    summary = []
    try:
        for i in range(args.n_episodes):
            cfg_id = (args.seed + i) % args.num_variations
            obs = env.reset(config_id=cfg_id, seed=args.seed + i)
            start_perf = float(obs["info"].get("normalized_performance",
                                               obs["info"].get("performance", 0.0)))
            # fixed flat goal = expert reference's FLATTEST frame (argmax perf),
            # not its last frame (the scripted expert often drifts after peaking).
            ref_frames, ref_perfs = env.make_goal_trajectory(n_steps=args.eval_budget, seed=1000 + cfg_id)
            expert_max = float(max(ref_perfs)) if ref_perfs else 0.0
            goal_idx = int(np.argmax(ref_perfs)) + 1 if ref_perfs else len(ref_frames) - 1
            goal_idx = min(goal_idx, len(ref_frames) - 1)
            goal_chw = torch.from_numpy(_to_imagenet_float(ref_frames[goal_idx])).float()
            cost.set_goal(goal_chw)

            # history buffers (length H frames/proprio; H-1 past actions)
            px0 = torch.from_numpy(_to_imagenet_float(obs["pixels"])).float()
            pix_hist = collections.deque([px0] * H, maxlen=H)
            prop0 = torch.from_numpy(np.asarray(obs["proprio"], np.float32))
            prop_hist = collections.deque([prop0] * H, maxlen=H)
            act_hist = collections.deque([torch.zeros(A)] * (H - 1), maxlen=H - 1)  # raw past actions

            warm = None
            plan_buf = collections.deque()  # normalized actions to execute
            perfs, frames, lat = [], [obs["pixels"]], []

            # Pre-grab: scripted descent so the pickers reach the rope plane and
            # grab on contact BEFORE the (optionally 2D-confined) MPC takes over.
            # Fixes the "can't descend to grab" failure of plan-time 2D freezing.
            if args.grab_steps > 0:
                descend = np.zeros(A, np.float32)
                descend[1] = -0.01; descend[5] = -0.01     # both pickers move down (y axis)
                descend[3] = 1.0;  descend[7] = 1.0        # grip engaged -> grab on contact
                for _ in range(args.grab_steps):
                    step_out = env.step(descend)
                    perfs.append(float(step_out["info"].get("normalized_performance",
                                       step_out["info"].get("performance", 0.0))))
                    frames.append(step_out["pixels"])
                    pix_hist.append(torch.from_numpy(_to_imagenet_float(step_out["pixels"])).float())
                    prop_hist.append(torch.from_numpy(np.asarray(step_out["proprio"], np.float32)))
                    act_hist.append(torch.from_numpy(descend))
                    if step_out["done"]:
                        break
                held = prop_hist[-1][6:8].tolist()         # hold flags after descent
                print(f"    [pre-grab] {args.grab_steps} steps; picker hold flags={held}")

            for t in range(args.eval_budget):
                if args.planner == "random":
                    a_norm = norm_a(torch.from_numpy(
                        rng.uniform(ACTION_LOW.numpy(), ACTION_HIGH.numpy()).astype(np.float32)))
                else:
                    if not plan_buf:  # replan
                        info = {
                            "hist_emb": cost.encode_obs(torch.stack(list(pix_hist))).unsqueeze(0),  # (1,H,D)
                            "hist_act_norm": norm_a(torch.stack(list(act_hist))).unsqueeze(0).to(device),  # (1,H-1,A)
                        }
                        if cost.uses_proprio:
                            info["start_proprio"] = prop_hist[-1].unsqueeze(0).to(device)
                            info["hist_proprio_norm"] = cost._norm_proprio(
                                torch.stack(list(prop_hist)).to(device)).unsqueeze(0)
                        t0 = time.perf_counter()
                        out = solver.solve(info, init_mean=warm)
                        torch.cuda.synchronize() if device == "cuda" else None
                        lat.append(time.perf_counter() - t0)
                        plan = out["actions"][0]  # (horizon, A) normalized
                        keep = args.receding_horizon
                        for j in range(min(keep, plan.shape[0])):
                            plan_buf.append(plan[j])
                        tail = plan[keep:]
                        warm = tail.unsqueeze(0) if tail.shape[0] > 0 else None
                    a_norm = plan_buf.popleft()

                a_raw = cost.denorm_action(a_norm.to(device))
                a_raw = torch.clamp(a_raw, ACTION_LOW.to(device), ACTION_HIGH.to(device))
                step_out = env.step(a_raw.cpu().numpy().astype(np.float32))
                perf = float(step_out["info"].get("normalized_performance",
                                                  step_out["info"].get("performance", 0.0)))
                perfs.append(perf); frames.append(step_out["pixels"])
                pix_hist.append(torch.from_numpy(_to_imagenet_float(step_out["pixels"])).float())
                prop_hist.append(torch.from_numpy(np.asarray(step_out["proprio"], np.float32)))
                act_hist.append(a_raw.cpu())
                if step_out["done"]:
                    break

            mpc_max = max(perfs) if perfs else start_perf
            mpc_final = perfs[-1] if perfs else start_perf
            success = mpc_max > args.success_thresh
            print(f"  [{i+1}/{args.n_episodes}] cfg={cfg_id} start={start_perf:.3f} "
                  f"max={mpc_max:.3f} final={mpc_final:.3f} success={success} "
                  f"expert_max={expert_max:.3f} lat={np.mean(lat)*1000 if lat else 0:.0f}ms")

            if args.save_gifs:
                goal_img = ref_frames[-1]
                pair = [np.concatenate([f, np.full((args.img_size, 6, 3), 255, np.uint8), goal_img], axis=1)
                        for f in frames]
                iio.mimsave(gif_dir / f"ep{i:03d}_cfg{cfg_id}.gif", pair, duration=0.1, loop=0)

            summary.append({"episode": i, "config_id": cfg_id, "goal_offset": args.horizon,
                            "mpc_final": mpc_final, "mpc_max": mpc_max,
                            "expert_final": float(ref_perfs[-1]) if ref_perfs else 0.0,
                            "expert_max": expert_max, "start_perf": start_perf,
                            "success": bool(success),
                            "mean_plan_lat_s": float(np.mean(lat)) if lat else 0.0,
                            "mpc_perf_curve": [float(x) for x in perfs]})
    finally:
        env.close()

    mx = np.array([s["mpc_max"] for s in summary])
    st = np.array([s["start_perf"] for s in summary])
    sr = np.mean([s["success"] for s in summary])
    print(f"\n==== MPC (LeWM/CEM) ====  n={len(summary)}")
    print(f"success rate (>{args.success_thresh}): {sr:.0%}")
    print(f"mean peak perf: {mx.mean():.3f}   mean start: {st.mean():.3f}   mean Δmax: {(mx-st).mean():+.3f}")
    (out_dir / f"summary_k{args.horizon}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/summary_k{args.horizon}.json")


if __name__ == "__main__":
    main()
