"""MPC evaluation harness for a trained LeWM on SoftGym tasks.

Pipeline:

  load trained model
  load eval HDF5 → sample N (init, goal) episodes
  for each episode:
    encode goal pixels → z_g
    reset env to episode's init config
    history buffer ← initial H frames
    for step in range(eval_budget):
       plan ← CEM(history, z_g)
       action ← plan[0]
       env.step(action)
       update history
    log final info_normalized_performance + planning latency

Three modes:
  - validate:  no env. Just verify the planner reduces latent cost over CEM
               iterations on a fixed (history, goal) pair pulled from HDF5.
               Smoke test for the planner + rollout pipeline.
  - replay:    planner emits a full open-loop action sequence per episode
               into NPZ files. A separate Docker script then replays them in
               SoftGym to measure achieved performance. Decouples py3.12 model
               from py3.6 env.
  - online:    full closed-loop MPC (replan each step). Requires a SoftGym
               env adapter — currently a stub; needs IPC implementation.
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
from rollout import encode_pixels, rollout, goal_cost  # noqa: E402
from cem_planner import CEMConfig, CEMPlanner  # noqa: E402
from gradient_planner import GradientPlanConfig, GradientPlanner  # noqa: E402
from sample_goals import sample_goals, goalspec_to_torch_inputs  # noqa: E402


class RandomPlanner:
    """Baseline: ignore the model + goal, return a uniform-random action.

    Matches the planner interface (.plan / .reset) so the online eval loop
    can swap it in for an apples-to-apples task-performance comparison.
    """
    def __init__(self, action_low, action_high, device):
        self.low = action_low.to(device)
        self.high = action_high.to(device)
        self.device = device

    def reset(self):
        pass

    def plan(self, pixels_history, goal_emb, history_actions):
        u = torch.rand(self.low.numel(), device=self.device)
        return self.low + u * (self.high - self.low)


def _build_planner(args, model, action_low, action_high, device):
    """Construct the planner selected by --planner."""
    if args.planner == "random":
        return RandomPlanner(action_low, action_high, device)
    if args.planner == "cem":
        return CEMPlanner(
            model=model,
            config=CEMConfig(num_samples=args.cem_samples, n_iters=args.cem_iters,
                             topk=args.cem_topk, plan_horizon=args.plan_horizon),
            action_low=action_low, action_high=action_high,
            history_size=args.history_size, device=str(device),
        )
    elif args.planner == "gradient":
        return GradientPlanner(
            model=model,
            config=GradientPlanConfig(plan_horizon=args.plan_horizon,
                                      n_restarts=args.grad_restarts,
                                      n_iters=args.grad_iters, lr=args.grad_lr,
                                      action_reg=args.grad_action_reg),
            action_low=action_low, action_high=action_high,
            history_size=args.history_size, device=str(device),
        )
    raise ValueError("unknown planner: %r" % args.planner)


# ---------------------------------------------------------------------------
# Env-adapter interface (abstract — concrete impls live elsewhere)
# ---------------------------------------------------------------------------

class EnvAdapter:
    """Abstract bridge to a SoftGym env. Concrete impls TBD:
       - InProcessSoftgym (only callable inside py3.6 Docker)
       - SubprocessSoftgym (parent: py3.12, child: py3.6 docker over stdin/stdout)
       - HttpSoftgym (Flask server in Docker, REST client here)
    """
    def reset(self, episode_idx: int):
        raise NotImplementedError
    def step(self, action: np.ndarray):
        raise NotImplementedError
    def render(self) -> np.ndarray:
        raise NotImplementedError
    def info(self) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mode: validate (no env)
# ---------------------------------------------------------------------------

def mode_validate(args, model, device):
    """Run the planner against fixed (history, goal) pairs from HDF5.

    What we verify:
      - the rollout produces finite latents
      - CEM cost decreases across iterations
      - the planner outputs an action in bounds
      - planning latency
    """
    specs = sample_goals(
        h5_path=args.eval_h5,
        n_goals=args.n_episodes,
        history_size=args.history_size,
        strategy=args.goal_strategy,
        min_perf=args.min_perf,
        seed=args.seed,
    )
    print(f"sampled {len(specs)} eval episodes from {args.eval_h5}")

    # Action bounds — read from a sample NPZ's metadata, hardcode defaults
    # if not available. For RopeFlatten with 2 pickers these are well-known.
    action_low = torch.tensor([-0.01, -0.01, -0.01, 0.0,
                               -0.01, -0.01, -0.01, 0.0], dtype=torch.float32)
    action_high = torch.tensor([0.01, 0.01, 0.01, 1.0,
                                0.01, 0.01, 0.01, 1.0], dtype=torch.float32)

    planner = CEMPlanner(
        model=model,
        config=CEMConfig(
            num_samples=args.cem_samples,
            n_iters=args.cem_iters,
            topk=args.cem_topk,
            plan_horizon=args.plan_horizon,
        ),
        action_low=action_low,
        action_high=action_high,
        history_size=args.history_size,
        device=device,
    )

    results = []
    for i, spec in enumerate(specs):
        inputs = goalspec_to_torch_inputs(spec, device=device)
        goal_emb = encode_pixels(model, inputs["goal_pixels"].unsqueeze(0)).squeeze(0).squeeze(0)
        # Zero "history actions" — we don't know what actions led to the
        # initial pixel frames; CEM only uses these to align the predictor's
        # action context, and zeros are a reasonable proxy.
        history_actions = torch.zeros(args.history_size, action_low.numel(), device=device)

        # Profile a plan call
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        first_action = planner.plan(
            pixels_history=inputs["pixels_history"],
            goal_emb=goal_emb,
            history_actions=history_actions,
        )
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - t0

        # Compute an unbiased final cost estimate by re-rolling with the planned
        # mean action sequence.
        action_seq = planner.prev_mean  # (H_plan, A)
        full_act = torch.cat([history_actions, action_seq], dim=0).unsqueeze(0).unsqueeze(0)
        full_px = inputs["pixels_history"].unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            emb = rollout(model, full_px, full_act, history_size=args.history_size)
            cost = float(goal_cost(emb, goal_emb.unsqueeze(0)).item())

        planner.reset()  # fresh state for next episode

        results.append({
            "episode_idx": spec.episode_idx,
            "plan_latency_s": elapsed,
            "final_latent_cost": cost,
            "first_action": first_action.cpu().tolist(),
            "goal_perf": spec.goal_perf,
        })
        print(f"  [{i+1}/{len(specs)}] ep={spec.episode_idx:4d}  "
              f"cost={cost:.3f}  latency={elapsed*1000:.0f}ms  "
              f"first_action={[f'{a:+.4f}' for a in first_action.cpu().tolist()]}")

    mean_lat = np.mean([r["plan_latency_s"] for r in results])
    mean_cost = np.mean([r["final_latent_cost"] for r in results])
    print(f"\nmean plan latency: {mean_lat*1000:.0f} ms")
    print(f"mean final latent cost: {mean_cost:.3f}")
    return results


# ---------------------------------------------------------------------------
# Mode: replay (planner emits NPZ, Docker side replays)
# ---------------------------------------------------------------------------

def mode_replay(args, model, device):
    """Plan open-loop action sequences and dump to NPZ files.

    Outputs one NPZ per episode containing:
      - init_state, goal_pixels, action_sequence (H_plan, A), expected_final_cost
    """
    out_dir = Path(args.replay_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = sample_goals(
        h5_path=args.eval_h5, n_goals=args.n_episodes,
        history_size=args.history_size, strategy=args.goal_strategy,
        min_perf=args.min_perf, seed=args.seed,
    )

    action_low = torch.tensor([-0.01, -0.01, -0.01, 0.0,
                               -0.01, -0.01, -0.01, 0.0], dtype=torch.float32)
    action_high = torch.tensor([0.01, 0.01, 0.01, 1.0,
                                0.01, 0.01, 0.01, 1.0], dtype=torch.float32)

    planner = CEMPlanner(
        model=model,
        config=CEMConfig(num_samples=args.cem_samples, n_iters=args.cem_iters,
                         topk=args.cem_topk, plan_horizon=args.plan_horizon),
        action_low=action_low, action_high=action_high,
        history_size=args.history_size, device=device,
    )

    for i, spec in enumerate(specs):
        inputs = goalspec_to_torch_inputs(spec, device=device)
        goal_emb = encode_pixels(model, inputs["goal_pixels"].unsqueeze(0)).squeeze(0).squeeze(0)
        hist_act = torch.zeros(args.history_size, action_low.numel(), device=device)

        planner.plan(inputs["pixels_history"], goal_emb, hist_act)
        action_seq = planner.prev_mean.cpu().numpy()

        # config_id = episode_idx % num_variations is a deterministic mapping
        # from the source HDF5 episode to a SoftGym variation slot. Replay
        # uses this to reset the env to a comparable starting config.
        config_id = spec.episode_idx % args.num_variations

        out_path = out_dir / f"plan_ep{spec.episode_idx:06d}.npz"
        np.savez(out_path,
                 episode_idx=spec.episode_idx,
                 config_id=config_id,
                 init_state=spec.init_state,
                 init_pixels=spec.init_pixels,
                 goal_pixels=spec.goal_pixels,
                 action_sequence=action_seq.astype(np.float32),
                 goal_perf=spec.goal_perf)
        planner.reset()
        print(f"  [{i+1}/{len(specs)}] wrote {out_path}  config_id={config_id}")


# ---------------------------------------------------------------------------
# Mode: online (closed-loop MPC against SoftGym in Docker)
# ---------------------------------------------------------------------------

def mode_online(args, model, device):
    """Closed-loop MPC.

    For each eval episode:
      reset env → encode goal pixels → run MPC for eval_budget steps,
      replanning at every step. Log final info_normalized_performance and
      per-step planning latency.
    """
    import collections

    from env_client import SubprocessSoftgym
    from sample_goals import _to_imagenet_float  # uses IMAGENET norm

    out_dir = Path(args.online_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample goals from held-out HDF5 (only used for goal_pixels here).
    specs = sample_goals(
        h5_path=args.eval_h5, n_goals=args.n_episodes,
        history_size=args.history_size, strategy=args.goal_strategy,
        min_perf=args.min_perf, seed=args.seed,
    )

    # 2. Action bounds (RopeFlatten defaults).
    action_low = torch.tensor([-0.01, -0.01, -0.01, 0.0,
                               -0.01, -0.01, -0.01, 0.0], dtype=torch.float32)
    action_high = torch.tensor([0.01, 0.01, 0.01, 1.0,
                                0.01, 0.01, 0.01, 1.0], dtype=torch.float32)

    planner = _build_planner(args, model, action_low, action_high, device)
    print(f"planner: {args.planner}")

    env = SubprocessSoftgym(
        env_name=args.env_name,
        num_variations=args.num_variations,
        img_size=args.img_size,
        num_picker=2,
        headless=True,
        render=True,
    )

    summary = []
    try:
        for i, spec in enumerate(specs):
            print(f"\n=== episode {i+1}/{len(specs)}  ep_idx={spec.episode_idx} ===")

            # Reset env. config_id picked from the eval seed so episodes are
            # reproducible but cover different variations.
            cfg_id = (args.seed + i) % args.num_variations
            obs = env.reset(config_id=cfg_id, seed=args.seed + i)

            # Encode goal once.
            goal_px_chw = _to_imagenet_float(spec.goal_pixels)              # (C, H, W) np
            goal_pixels = torch.from_numpy(goal_px_chw).float().to(device)
            goal_emb = encode_pixels(model, goal_pixels.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            # Pixel + action history buffer (deque of length history_size).
            init_px_chw = _to_imagenet_float(obs["pixels"])                  # (C, H, W)
            init_px = torch.from_numpy(init_px_chw).float().to(device)
            pixel_hist = collections.deque([init_px for _ in range(args.history_size)],
                                           maxlen=args.history_size)
            action_hist = collections.deque(
                [torch.zeros(action_low.numel(), device=device)
                 for _ in range(args.history_size)],
                maxlen=args.history_size,
            )

            planner.reset()
            ep_log = []
            final_perf = None

            for t in range(args.eval_budget):
                px_stack = torch.stack(list(pixel_hist), dim=0)               # (H, C, H, W)
                act_stack = torch.stack(list(action_hist), dim=0)              # (H, A)

                t0 = time.perf_counter()
                first_action = planner.plan(px_stack, goal_emb, act_stack)
                if device == "cuda":
                    torch.cuda.synchronize()
                plan_lat = time.perf_counter() - t0

                action_np = first_action.cpu().numpy().astype(np.float32)
                step_out = env.step(action_np)
                final_perf = float(step_out["info"].get("normalized_performance",
                                                        step_out["info"].get("performance", 0.0)))
                # Update history with the new observation + executed action.
                new_px = torch.from_numpy(_to_imagenet_float(step_out["pixels"])).float().to(device)
                pixel_hist.append(new_px)
                action_hist.append(first_action.detach())

                ep_log.append({
                    "t": t, "plan_lat_s": plan_lat,
                    "norm_perf": final_perf, "reward": step_out["reward"],
                })
                if step_out["done"]:
                    break

            mean_lat = float(np.mean([s["plan_lat_s"] for s in ep_log]))
            print(f"  steps: {len(ep_log)}  final_perf: {final_perf:.3f}  "
                  f"mean_plan_lat: {mean_lat*1000:.0f} ms")
            summary.append({
                "episode_idx": spec.episode_idx,
                "config_id": cfg_id,
                "goal_perf": spec.goal_perf,
                "achieved_perf": final_perf,
                "n_steps": len(ep_log),
                "mean_plan_lat_s": mean_lat,
            })
            (out_dir / f"ep{i:03d}.json").write_text(json.dumps({
                "spec_episode_idx": spec.episode_idx, "config_id": cfg_id,
                "goal_perf": spec.goal_perf, "log": ep_log,
            }, indent=2))
    finally:
        env.close()

    # Headline metrics.
    achieved = np.array([s["achieved_perf"] for s in summary])
    goals    = np.array([s["goal_perf"]     for s in summary])
    lat_ms   = np.array([s["mean_plan_lat_s"] * 1000 for s in summary])
    print("\n========== SUMMARY ==========")
    print(f"episodes:           {len(summary)}")
    print(f"achieved perf:      mean={achieved.mean():.3f}  median={np.median(achieved):.3f}")
    print(f"  > 0.5 success:    {(achieved > 0.5).mean():.1%}")
    print(f"  > 0.8 success:    {(achieved > 0.8).mean():.1%}")
    print(f"goal perf (target): mean={goals.mean():.3f}")
    print(f"plan latency:       mean={lat_ms.mean():.0f} ms")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/summary.json")


# ---------------------------------------------------------------------------
# Mode: imitate (LeWM-protocol receding-horizon MPC toward reference subgoals)
# ---------------------------------------------------------------------------

def mode_imitate(args, model, device):
    """Receding-horizon latent MPC toward subgoals from an expert reference
    trajectory generated on the SAME rope (snapshot/restore).

    Per episode:
      1. reset env to a config
      2. make_goal_trajectory() -> expert reference frames F[0..T] (flattens
         the rope), env restored to start
      3. for t in range(budget): goal = encode(F[min(t+k, T)]); plan; step
      4. record achieved perf vs expert; render a GIF of the MPC rollout

    With --goal-offset swept this characterizes the planning horizon.
    """
    import collections
    import imageio.v2 as iio
    from env_client import SubprocessSoftgym
    from sample_goals import _to_imagenet_float

    out_dir = Path(args.imitate_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_dir = out_dir / "gifs"; gif_dir.mkdir(exist_ok=True)

    # Cap the planner's per-step picker motion (the env applies action_repeat=8,
    # so the full +/-0.01 range = 8cm/step lets the planner fling the rope out of
    # frame by exploiting model errors). Scaling the delta bounds — mirroring the
    # scripted policy's max_speed_scale — keeps manipulation gentle and on-screen.
    s = args.plan_action_scale
    action_low = torch.tensor([-0.01*s, -0.01*s, -0.01*s, 0.0, -0.01*s, -0.01*s, -0.01*s, 0.0])
    action_high = torch.tensor([0.01*s, 0.01*s, 0.01*s, 1.0, 0.01*s, 0.01*s, 0.01*s, 1.0])
    planner = _build_planner(args, model, action_low, action_high, device)
    A = action_low.numel()
    k = args.goal_offset

    env = SubprocessSoftgym(env_name=args.env_name, num_variations=args.num_variations,
                            img_size=args.img_size, num_picker=2, headless=True, render=True)

    def enc_goal(frame_uint8):
        chw = _to_imagenet_float(frame_uint8)
        t = torch.from_numpy(chw).float().to(device)
        return encode_pixels(model, t.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    summary = []
    try:
        for i in range(args.n_episodes):
            cfg_id = (args.seed + i) % args.num_variations
            obs = env.reset(config_id=cfg_id, seed=args.seed + i)

            # Expert reference on THIS rope (env restored afterward).
            ref_frames, ref_perfs = env.make_goal_trajectory(n_steps=args.eval_budget)
            expert_final = float(ref_perfs[-1]) if ref_perfs else 0.0
            expert_max = float(max(ref_perfs)) if ref_perfs else 0.0
            T_ref = len(ref_frames)

            # History buffers.
            init_px = torch.from_numpy(_to_imagenet_float(obs["pixels"])).float().to(device)
            pixel_hist = collections.deque([init_px] * args.history_size, maxlen=args.history_size)
            action_hist = collections.deque([torch.zeros(A, device=device)] * args.history_size,
                                            maxlen=args.history_size)
            planner.reset()

            rollout_frames = [obs["pixels"]]
            perfs = []
            lat = []
            for t in range(args.eval_budget):
                goal_idx = min(t + k, T_ref - 1)
                goal_emb = enc_goal(ref_frames[goal_idx])
                px = torch.stack(list(pixel_hist), dim=0)
                act = torch.stack(list(action_hist), dim=0)

                t0 = time.perf_counter()
                first_action = planner.plan(px, goal_emb, act)
                if str(device) == "cuda":
                    torch.cuda.synchronize()
                lat.append(time.perf_counter() - t0)

                step_out = env.step(first_action.cpu().numpy().astype(np.float32))
                perf = float(step_out["info"].get("normalized_performance",
                                                  step_out["info"].get("performance", 0.0)))
                perfs.append(perf)
                rollout_frames.append(step_out["pixels"])
                pixel_hist.append(torch.from_numpy(_to_imagenet_float(step_out["pixels"])).float().to(device))
                action_hist.append(first_action.detach())
                if step_out["done"]:
                    break

            mpc_final = perfs[-1] if perfs else 0.0
            mpc_max = max(perfs) if perfs else 0.0
            print(f"  [{i+1}/{args.n_episodes}] cfg={cfg_id}  "
                  f"MPC final={mpc_final:.3f} max={mpc_max:.3f}  |  "
                  f"expert final={expert_final:.3f} max={expert_max:.3f}  |  "
                  f"lat={np.mean(lat)*1000:.0f}ms")

            # GIF: MPC rollout next to the expert reference (side by side).
            if args.save_gifs:
                n = min(len(rollout_frames), T_ref)
                pair = [np.concatenate([rollout_frames[j],
                                        np.full((args.img_size, 6, 3), 255, np.uint8),
                                        ref_frames[min(j, T_ref - 1)]], axis=1)
                        for j in range(n)]
                iio.mimsave(gif_dir / f"ep{i:03d}_cfg{cfg_id}.gif", pair, duration=0.1, loop=0)

            summary.append({"episode": i, "config_id": cfg_id, "goal_offset": k,
                            "mpc_final": mpc_final, "mpc_max": mpc_max,
                            "expert_final": expert_final, "expert_max": expert_max,
                            "start_perf": float(ref_perfs[0]) if ref_perfs else 0.0,
                            "mean_plan_lat_s": float(np.mean(lat))})
    finally:
        env.close()

    mpc_f = np.array([s["mpc_final"] for s in summary])
    exp_f = np.array([s["expert_final"] for s in summary])
    start = np.array([s["start_perf"] for s in summary])
    print("\n========== IMITATE SUMMARY (goal_offset=%d) ==========" % k)
    print(f"episodes:        {len(summary)}")
    print(f"start perf:      mean={start.mean():.3f}")
    print(f"MPC final perf:  mean={mpc_f.mean():.3f}  median={np.median(mpc_f):.3f}")
    print(f"expert final:    mean={exp_f.mean():.3f}  (upper bound)")
    print(f"MPC improvement over start: {(mpc_f - start).mean():+.3f}")
    print(f"MPC / expert ratio:         {mpc_f.mean()/max(exp_f.mean(),1e-6):.1%}")
    (out_dir / f"summary_k{k}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/summary_k{k}.json  (+ gifs in {gif_dir})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--eval-h5", default=None,
                   help="v3 HDF5 file for goals (required for validate/replay/online; "
                        "unused by imitate, which generates references from the env).")
    p.add_argument("--mode", choices=("validate", "replay", "online", "imitate"), default="validate")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--goal-strategy", choices=("terminal", "subgoal", "high_perf"),
                   default="high_perf")
    p.add_argument("--min-perf", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)

    # planner hyperparams (defaults match LeWM paper)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--planner", choices=("cem", "gradient", "random"), default="gradient",
                   help="Planning algorithm. 'gradient' = our fast gradient-based "
                        "latent planner; 'cem' = the paper's CEM baseline; "
                        "'random' = uniform-random action baseline.")
    p.add_argument("--plan-horizon", type=int, default=5)
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-iters", type=int, default=30)
    p.add_argument("--cem-topk", type=int, default=30)
    p.add_argument("--grad-restarts", type=int, default=32)
    p.add_argument("--grad-iters", type=int, default=15)
    p.add_argument("--grad-lr", type=float, default=0.1)
    p.add_argument("--grad-action-reg", type=float, default=0.0,
                   help="L2 penalty on action magnitude in the gradient planner "
                        "(discourages large/flinging actions).")
    p.add_argument("--plan-action-scale", type=float, default=0.4,
                   help="Scale on the planner's xyz delta bounds (imitate mode). "
                        "0.4 ≈ scripted policy's max_speed_scale; keeps motion gentle.")

    # mode-specific
    p.add_argument("--replay-out-dir", default="eval/runs/replay")
    p.add_argument("--online-out-dir", default="eval/runs/online")
    # imitate mode
    p.add_argument("--imitate-out-dir", default="eval/runs/imitate")
    p.add_argument("--goal-offset", type=int, default=5,
                   help="Subgoal offset k: goal = reference frame min(t+k, T) ahead. "
                        "Sweep this to characterize the planning horizon.")
    p.add_argument("--save-gifs", action="store_true",
                   help="Save MPC-vs-expert side-by-side GIFs per episode (imitate mode).")
    p.add_argument("--env-name", default="RopeFlatten")
    p.add_argument("--num-variations", type=int, default=200)
    p.add_argument("--eval-budget", type=int, default=75,
                   help="Max env steps per episode for --mode online.")

    # model arch (must match the checkpoint)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--num-preds", type=int, default=1)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=16)
    p.add_argument("--predictor-dim-head", type=int, default=64)
    p.add_argument("--predictor-dropout", type=float, default=0.1)
    p.add_argument("--action-dim", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the model with the same arch as training, then load weights.
    model = build_lewm(
        img_size=args.img_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        predictor_dim_head=args.predictor_dim_head,
        predictor_dropout=args.predictor_dropout,
        history_size=args.history_size, num_preds=args.num_preds,
        action_dim=args.action_dim,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded model: {n_params/1e6:.2f}M params from {args.ckpt}")

    {
        "validate": mode_validate,
        "replay":   mode_replay,
        "online":   mode_online,
        "imitate":  mode_imitate,
    }[args.mode](args, model, device)


if __name__ == "__main__":
    main()
