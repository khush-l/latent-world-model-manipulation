"""Quick CEM-MPC eval on RopeConfiguration (character-shaping) with the LeWM
cost model. Goal = the env's target-character image. Pure-RGB model.

Run with SOFTGYM_PICKER_COLOR=1,0,0 so the env renders red pickers matching a
red-trained checkpoint. RopeConfiguration is a DIFFERENT task than the
RopeFlatten the model trained on, so this is an out-of-distribution probe.
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np, torch, imageio.v2 as iio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training")); sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from model import build_lewm
from sample_goals import _to_imagenet_float
from lewm_cost import LeWMCost
from solvers import CEMSolver
from env_client import SubprocessSoftgym
from mpc_runner import load_stats, ACTION_LOW, ACTION_HIGH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config-id", type=int, default=0)
    p.add_argument("--num-variations", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--hist", type=int, default=3)
    p.add_argument("--cem-samples", type=int, default=200)
    p.add_argument("--cem-iters", type=int, default=15)
    p.add_argument("--out-dir", default="eval/runs/ropeconfig_red")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / args.out_dir; (out_dir / "gifs").mkdir(parents=True, exist_ok=True)
    cfgp = Path(args.ckpt).parent / "config.json"
    tcfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
    use_proprio = bool(tcfg.get("use_proprio", False))
    model = build_lewm(history_size=args.hist, use_proprio=use_proprio).to(device)
    st = torch.load(args.ckpt, map_location=device); st = st.get("model_state_dict", st) if isinstance(st, dict) else st
    model.load_state_dict(st); model.eval()
    a_mean, a_std, p_mean, p_std = load_stats(tcfg.get("data", "simulation/data/rope/rope_full_dataset_red.h5"), device)
    cost = LeWMCost(model, ACTION_LOW, ACTION_HIGH, history_size=args.hist,
                    action_mean=a_mean, action_std=a_std, proprio_mean=p_mean, proprio_std=p_std, device=device)
    solver = CEMSolver(cost, horizon=5, num_samples=args.cem_samples, n_steps=args.cem_iters, topk=30, device=device, seed=0)
    A = cost.action_dim

    env = SubprocessSoftgym(env_name="RopeConfiguration", num_variations=args.num_variations, img_size=128)
    try:
        obs = env.reset(config_id=args.config_id)
        ch = obs["info"].get("goal_character", "?")
        start = float(obs["info"].get("normalized_performance", obs["info"].get("performance", 0.0)))
        goal_img = env.get_goal_image()
        cost.set_goal(torch.from_numpy(_to_imagenet_float(goal_img)).float())
        print(f"RopeConfiguration cfg{args.config_id} char={ch} start={start:.3f}")

        pix = collections.deque([torch.from_numpy(_to_imagenet_float(obs["pixels"])).float()] * args.hist, maxlen=args.hist)
        acts = collections.deque([torch.zeros(A)] * (args.hist - 1), maxlen=args.hist - 1)
        warm = None; frames = [obs["pixels"]]; perfs = []
        for t in range(args.eval_budget):
            info = {"hist_emb": cost.encode_obs(torch.stack(list(pix))).unsqueeze(0),
                    "hist_act_norm": ((torch.stack(list(acts)).to(device) - a_mean) / a_std).unsqueeze(0)}
            out = solver.solve(info, init_mean=warm)
            plan = out["actions"][0]
            a_norm = plan[0]; warm = plan[1:].unsqueeze(0)
            a_raw = torch.clamp(cost.denorm_action(a_norm.to(device)), ACTION_LOW.to(device), ACTION_HIGH.to(device))
            step = env.step(a_raw.cpu().numpy().astype(np.float32))
            perfs.append(float(step["info"].get("normalized_performance", step["info"].get("performance", 0.0))))
            frames.append(step["pixels"])
            pix.append(torch.from_numpy(_to_imagenet_float(step["pixels"])).float())
            acts.append(a_raw.cpu())
            if step["done"]:
                break
        print(f"  start={start:.3f} max={max(perfs):.3f} final={perfs[-1]:.3f}")
        pair = [np.concatenate([f, np.full((128, 6, 3), 255, np.uint8), goal_img], axis=1) for f in frames]
        big = [np.repeat(np.repeat(x, 2, 0), 2, 1) for x in pair]
        gif = out_dir / f"gifs/ropeconfig_{ch}_cfg{args.config_id}.gif"
        iio.mimsave(gif, big, duration=0.12, loop=0); print(f"wrote {gif}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
