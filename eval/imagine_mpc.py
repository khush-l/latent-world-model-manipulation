"""Visualize the world model's IMAGINED latent rollout DURING receding-horizon MPC.

Runs the real LeWM-matched receding-horizon MPC loop and, at each replan, decodes
the chosen plan's imagined latent trajectory — what the model "thinks" will happen
over the next horizon — and lays it against the frames that ACTUALLY occur next.

To reproduce a SPECIFIC eval episode exactly (mpc_runner.py with --seed S, episode i),
both the env rope RNG and the CEM RNG are call-order dependent, so we replay episodes
0..i in order with one solver seeded once at S (exactly as mpc_runner does), and only
capture/plot the imagination on the target episode i.

Usage (reproduce mpc_runner --seed 0 episode 7 = cfg7):
    python eval/imagine_mpc.py \
        --ckpt training/runs/rope_proprio_v2_250k/ckpt_final.pt \
        --decoder training/runs/rope_proprio_v2_250k/decoders/decoder_step250000.pt \
        --base-seed 0 --episode 7 --horizon 5 --n-replans 5 \
        --out eval/runs/imagine_mpc_cfg7_v2.png
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from env_client import SubprocessSoftgym               # noqa: E402
from sample_goals import _to_imagenet_float            # noqa: E402
from lewm_cost import LeWMCost                          # noqa: E402
from solvers import CEMSolver                           # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD         # noqa: E402
from mpc_runner import ACTION_LOW, ACTION_HIGH, load_stats  # noqa: E402
from model import build_lewm                            # noqa: E402
from train_decoder import TransformerDecoder, Decoder    # noqa: E402


def build_model(ckpt, device, H):
    """Build the JEPA arch from the checkpoint's sibling config.json and load it."""
    cfg = json.loads((Path(ckpt).parent / "config.json").read_text())
    model = build_lewm(
        img_size=cfg.get("img_size", 128), patch_size=cfg.get("patch_size", 16),
        embed_dim=cfg.get("embed_dim", 192), predictor_depth=cfg.get("predictor_depth", 6),
        predictor_heads=cfg.get("predictor_heads", 16), predictor_dim_head=cfg.get("predictor_dim_head", 64),
        predictor_dropout=cfg.get("predictor_dropout", 0.1),
        history_size=H, num_preds=cfg.get("num_preds", 1),
        use_proprio=bool(cfg.get("use_proprio", False)),
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model, cfg


def load_decoder(path, device):
    """Load a saved transformer/conv decoder (from decode_across_checkpoints)."""
    d = torch.load(path, map_location=device, weights_only=False)
    if d["decoder_type"] == "transformer":
        dec = TransformerDecoder(latent_dim=d["latent_dim"], patch=d["patch_size"], out_size=d["img_size"])
    else:
        dec = Decoder(latent_dim=d["latent_dim"], out_size=d["img_size"])
    dec.load_state_dict(d["decoder_state_dict"]); dec.to(device).eval()
    return dec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--decoder", required=True)
    p.add_argument("--base-seed", type=int, default=0, help="the eval's --seed")
    p.add_argument("--episode", type=int, default=7, help="episode index to visualize (cfg=(seed+i)%vars)")
    p.add_argument("--num-variations", type=int, default=200)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--receding-horizon", type=int, default=-1, help="-1 => == horizon")
    p.add_argument("--eval-budget", type=int, default=75)
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-iters", type=int, default=30)
    p.add_argument("--cem-topk", type=int, default=30)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--n-replans", type=int, default=5, help="how many replans to show")
    p.add_argument("--env-name", default="RopeFlatten")
    p.add_argument("--out", default="eval/runs/imagine_mpc.png")
    p.add_argument("--gif-out", default=None,
                   help="also write a per-step GIF: decoded imagined next-state | reality | goal")
    p.add_argument("--scan-episodes", type=int, default=0,
                   help="if >0, run episodes 0..N-1 and use the BEST low-start flatten for the "
                        "figure/GIF (robust to SoftGym physics nondeterminism, which makes "
                        "reproducing one specific episode unreliable).")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = args.history_size
    recede = args.horizon if args.receding_horizon < 0 else args.receding_horizon

    model, cfg = build_model(args.ckpt, device, H)
    dec = load_decoder(args.decoder, device)
    a_mean, a_std, p_mean, p_std = load_stats(cfg["data"], device)
    cost = LeWMCost(model, ACTION_LOW, ACTION_HIGH, history_size=H,
                    action_mean=a_mean, action_std=a_std,
                    proprio_mean=p_mean, proprio_std=p_std, device=device)
    A = cost.action_dim
    # ONE solver, seeded once at base_seed — exactly as mpc_runner (its RNG carries
    # across episodes, so the target episode's plans depend on the warmup).
    solver = CEMSolver(cost, horizon=args.horizon, num_samples=args.cem_samples,
                       n_steps=args.cem_iters, topk=args.cem_topk, device=device, seed=args.base_seed)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    def denorm_img(x):
        return (x * std + mean).clamp(0, 1).detach().cpu().numpy().transpose(0, 2, 3, 1)
    def norm_a(a_raw):
        return (a_raw.to(device) - a_mean) / a_std

    def run_episode(i, capture):
        """One full MPC episode (mirrors mpc_runner). capture=True records the
        decoded imagined rollout at each replan for plotting."""
        cfg_id = (args.base_seed + i) % args.num_variations
        obs = env.reset(config_id=cfg_id, seed=args.base_seed + i)
        start_perf = float(obs["info"].get("normalized_performance", 0.0))
        ref_frames, ref_perfs = env.make_goal_trajectory(n_steps=args.eval_budget, seed=1000 + cfg_id)
        goal_idx = min(int(np.argmax(ref_perfs)) + 1, len(ref_frames) - 1)
        goal_img = ref_frames[goal_idx]
        cost.set_goal(torch.from_numpy(_to_imagenet_float(goal_img)).float())

        px0 = torch.from_numpy(_to_imagenet_float(obs["pixels"])).float()
        pix_hist = collections.deque([px0] * H, maxlen=H)
        prop0 = torch.from_numpy(np.asarray(obs["proprio"], np.float32))
        prop_hist = collections.deque([prop0] * H, maxlen=H)
        act_hist = collections.deque([torch.zeros(A)] * (H - 1), maxlen=H - 1)

        real_frames = [obs["pixels"]]; perfs = [start_perf]; replans = []
        imagined_steps = []                 # decoded imagined NEXT-state per executed step
        cur_imgs = None; exec_k = 0         # current plan's decoded imagined rollout + cursor
        warm = None; plan_buf = collections.deque()

        for t in range(args.eval_budget):
            if not plan_buf:
                info = {
                    "hist_emb": cost.encode_obs(torch.stack(list(pix_hist))).unsqueeze(0),
                    "hist_act_norm": norm_a(torch.stack(list(act_hist))).unsqueeze(0).to(device),
                }
                if cost.uses_proprio:
                    info["start_proprio"] = prop_hist[-1].unsqueeze(0).to(device)
                    info["hist_proprio_norm"] = cost._norm_proprio(
                        torch.stack(list(prop_hist)).to(device)).unsqueeze(0)
                out = solver.solve(info, init_mean=warm)
                plan = out["actions"][0]
                if capture:
                    imagined = cost.imagine(info, plan)   # (horizon+1, D)
                    cur_imgs = denorm_img(dec(imagined))  # (horizon+1, h, w, 3) in [0,1]
                    exec_k = 0
                    replans.append({"t": t, "perf": perfs[-1], "imagined": cur_imgs})
                for j in range(min(recede, plan.shape[0])):
                    plan_buf.append(plan[j])
                tail = plan[recede:]
                warm = tail.unsqueeze(0) if tail.shape[0] > 0 else None

            a_norm = plan_buf.popleft()
            a_raw = torch.clamp(cost.denorm_action(a_norm.to(device)),
                                ACTION_LOW.to(device), ACTION_HIGH.to(device))
            step_out = env.step(a_raw.cpu().numpy().astype(np.float32))
            real_frames.append(step_out["pixels"])
            perfs.append(float(step_out["info"].get("normalized_performance", 0.0)))
            pix_hist.append(torch.from_numpy(_to_imagenet_float(step_out["pixels"])).float())
            prop_hist.append(torch.from_numpy(np.asarray(step_out["proprio"], np.float32)))
            act_hist.append(a_raw.cpu())
            if capture and cur_imgs is not None:
                # decoded imagined state the model predicted for THIS executed step
                imagined_steps.append(cur_imgs[min(exec_k + 1, cur_imgs.shape[0] - 1)])
                exec_k += 1
            if step_out["done"]:
                break
        return dict(cfg_id=cfg_id, start=start_perf, perfs=perfs, replans=replans,
                    real_frames=real_frames, goal_img=goal_img, imagined_steps=imagined_steps)

    env = SubprocessSoftgym(env_name=args.env_name, num_variations=args.num_variations,
                            img_size=args.img_size, num_picker=2)
    try:
        if args.scan_episodes > 0:
            # run several episodes, pick the best LOW-START flatten that actually
            # happens this run (physics is nondeterministic, so we can't reproduce a
            # specific prior episode — we scan and showcase the best real success).
            results = []
            for i in range(args.scan_episodes):
                r = run_episode(i, capture=True)
                pk = max(r["perfs"])
                print(f"  [scan ep{i}] cfg{r['cfg_id']} start={r['start']:.2f} peak={pk:.2f}", flush=True)
                results.append(r)
            flat = [r for r in results if max(r["perfs"]) > 0.8 and r["start"] < 0.7]
            ep = max(flat or results, key=lambda r: max(r["perfs"]))
            print(f"  -> showcasing cfg{ep['cfg_id']} (start {ep['start']:.2f} peak {max(ep['perfs']):.2f})")
        else:
            for i in range(args.episode):   # warmup: replay prior episodes to match RNG state
                r = run_episode(i, capture=False)
                print(f"  [warmup ep{i}] cfg{r['cfg_id']} start={r['start']:.2f} "
                      f"peak={max(r['perfs']):.2f}", flush=True)
            ep = run_episode(args.episode, capture=True)   # target: capture imagination
    finally:
        env.close()

    replans, real_frames, goal_img = ep["replans"], ep["real_frames"], ep["goal_img"]
    peak = max(ep["perfs"])
    print(f"TARGET ep{args.episode} cfg{ep['cfg_id']}: start={ep['start']:.2f} "
          f"peak={peak:.2f} final={ep['perfs'][-1]:.2f}  ({len(replans)} replans)")

    K = min(args.n_replans, len(replans))
    sel = sorted(set(np.linspace(0, len(replans) - 1, K).round().astype(int).tolist()))
    Hh = replans[0]["imagined"].shape[0]
    ncol = Hh + 1

    fig, axes = plt.subplots(2 * len(sel), ncol, figsize=(1.4 * ncol, 2.6 * len(sel)))
    if len(sel) == 1:
        axes = axes.reshape(2, ncol)
    for r in range(2 * len(sel)):
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    for idx, ri in enumerate(sel):
        rp = replans[ri]; t0 = rp["t"]
        rim, rre = 2 * idx, 2 * idx + 1
        for k in range(Hh):
            axes[rim, k].imshow(rp["imagined"][k])
            if idx == 0:
                axes[rim, k].set_title(f"+{k}", fontsize=9)
            ridx = min(t0 + k, len(real_frames) - 1)
            axes[rre, k].imshow(real_frames[ridx].astype(np.float32) / 255.0)
        axes[rim, Hh].imshow(goal_img.astype(np.float32) / 255.0)
        if idx == 0:
            axes[rim, Hh].set_title("goal", fontsize=9)
        axes[rre, Hh].axis("off")
        axes[rim, 0].set_ylabel(f"replan @t={t0}\n(perf {rp['perf']:.2f})\nIMAGINED",
                                fontsize=8, rotation=0, ha="right", va="center")
        axes[rre, 0].set_ylabel("REALITY", fontsize=8, rotation=0, ha="right", va="center")

    fig.suptitle(f"What MPC imagines at each replan vs. what happens — RopeFlatten "
                 f"cfg{ep['cfg_id']} (start {ep['start']:.2f} → peak {peak:.2f})", fontsize=12)
    fig.tight_layout()
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")

    # per-step GIF: decoded imagined next-state | reality | goal, one frame per env step
    if args.gif_out:
        steps = ep["imagined_steps"]
        goal_u8 = goal_img.astype(np.uint8)
        sep = np.full((args.img_size, 6, 3), 255, np.uint8)
        frames = []
        for t, im in enumerate(steps):
            imag_u8 = (np.clip(im, 0, 1) * 255).astype(np.uint8)
            real_u8 = real_frames[min(t + 1, len(real_frames) - 1)].astype(np.uint8)
            frames.append(np.concatenate([imag_u8, sep, real_u8, sep, goal_u8], axis=1))
        gif_path = PROJECT_ROOT / args.gif_out
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        iio.mimsave(gif_path, frames, duration=0.15, loop=0)
        print(f"wrote {gif_path}  ({len(frames)} frames: imagined | reality | goal)")


if __name__ == "__main__":
    main()
