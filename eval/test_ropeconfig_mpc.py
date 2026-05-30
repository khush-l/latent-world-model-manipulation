"""Quick closed-loop MPC smoke test on the RopeConfiguration task.

RopeConfiguration asks the agent to bend the rope into a target character shape
('S','O','M','C','U'). Unlike RopeFlatten there is no scripted expert, but the
env ships a pre-rendered goal-character image — we use that directly as the MPC
goal latent and run the gradient planner closed-loop.

NOTE: our checkpoints are trained on RopeFlatten, so RopeConfiguration is
out-of-distribution. This is a *pipeline* test (env launches, encode/plan/step
loop runs, performance is read out) — not a performance claim.

Usage:
    python eval/test_ropeconfig_mpc.py --ckpt training/runs/rope_full_mixed_v1/ckpt_final.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from model import build_lewm                              # noqa: E402
from rollout import encode_pixels                         # noqa: E402
from gradient_planner import GradientPlanner, GradientPlanConfig  # noqa: E402
from env_client import SubprocessSoftgym                  # noqa: E402
from sample_goals import _to_imagenet_float               # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="training/runs/rope_full_mixed_v1/ckpt_final.pt")
    p.add_argument("--config-id", type=int, default=0)
    p.add_argument("--num-variations", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=25)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--out-dir", default="eval/runs/ropeconfig_test")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild + load model (baseline arch; proprio auto-handled if present).
    import json
    cfg_path = Path(args.ckpt).parent / "config.json"
    use_proprio = bool(json.loads(cfg_path.read_text()).get("use_proprio", False)) if cfg_path.exists() else False
    model = build_lewm(history_size=args.history_size, use_proprio=use_proprio).to(device)
    state = torch.load(args.ckpt, map_location=device)
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state); model.eval()
    print(f"loaded {args.ckpt} (use_proprio={use_proprio})")

    low = torch.tensor([-0.01, -0.01, -0.01, 0., -0.01, -0.01, -0.01, 0.])
    high = torch.tensor([0.01, 0.01, 0.01, 1., 0.01, 0.01, 0.01, 1.])
    planner = GradientPlanner(
        model, GradientPlanConfig(plan_horizon=5, n_restarts=16, n_iters=10),
        low, high, history_size=args.history_size, device=device)

    env = SubprocessSoftgym(env_name="RopeConfiguration", num_variations=args.num_variations,
                            img_size=args.img_size, num_picker=2)
    A = 8
    try:
        obs = env.reset(config_id=args.config_id)
        start_perf = float(obs["info"].get("normalized_performance",
                                           obs["info"].get("performance", 0.0)))
        goal_img = env.get_goal_image()
        print(f"reset cfg {args.config_id}: start perf={start_perf:.3f}  goal img {goal_img.shape}")

        goal_chw = _to_imagenet_float(goal_img)
        goal_emb = encode_pixels(
            model, torch.from_numpy(goal_chw).float().to(device).unsqueeze(0).unsqueeze(0)
        ).squeeze(0).squeeze(0)

        import collections
        init_px = torch.from_numpy(_to_imagenet_float(obs["pixels"])).float().to(device)
        pixel_hist = collections.deque([init_px] * args.history_size, maxlen=args.history_size)
        action_hist = collections.deque([torch.zeros(A, device=device)] * args.history_size,
                                        maxlen=args.history_size)
        planner.reset()

        frames = [obs["pixels"]]
        perfs = []
        for t in range(args.eval_budget):
            px = torch.stack(list(pixel_hist), dim=0)
            act = torch.stack(list(action_hist), dim=0)
            a = planner.plan(px, goal_emb, act)
            out = env.step(a.cpu().numpy().astype(np.float32))
            perf = float(out["info"].get("normalized_performance",
                                         out["info"].get("performance", 0.0)))
            perfs.append(perf)
            frames.append(out["pixels"])
            pixel_hist.append(torch.from_numpy(_to_imagenet_float(out["pixels"])).float().to(device))
            action_hist.append(a.detach())
            print(f"  step {t:2d}: perf={perf:+.3f}")
            if out["done"]:
                break

        gchar = "?"
        best = max(perfs) if perfs else start_perf
        print(f"\nRopeConfiguration MPC: start={start_perf:.3f}  final={perfs[-1]:.3f}  "
              f"best={best:.3f}  (Δbest={best - start_perf:+.3f})")

        # GIF: rollout next to the goal-character image.
        goal_col = np.array(goal_img)
        pair = [np.concatenate([f, np.full((args.img_size, 6, 3), 255, np.uint8), goal_col], axis=1)
                for f in frames]
        gif = out_dir / f"ropeconfig_cfg{args.config_id}.gif"
        iio.mimsave(gif, pair, duration=0.12, loop=0)
        print(f"wrote {gif}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
