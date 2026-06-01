"""Paper-style figure: decode what the world model IMAGINES on its way to the goal.

For one RopeFlatten config, we (1) plan a long-horizon action sequence with CEM
from the start state, (2) roll the world model forward to get its imagined latent
trajectory, decode each latent with a trained decoder ("the model's dream"), and
(3) execute the SAME plan in the real env to get the actual frames. Laying the two
rows side by side shows imagination-vs-reality on the path to the fixed flat goal.

Usage:
    python eval/imagine_plan.py \
        --ckpt training/runs/rope_sg_proprio_d384_deep_250k/ckpt_final.pt \
        --decoder training/runs/rope_sg_proprio_d384_deep_250k/decoders/decoder_step250000.pt \
        --config-id 5 --seed 5 --horizon 24 --n-cols 7 \
        --out eval/runs/imagine_cfg5.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from model import build_lewm                       # noqa: E402
from env_client import SubprocessSoftgym           # noqa: E402
from sample_goals import _to_imagenet_float        # noqa: E402
from lewm_cost import LeWMCost                      # noqa: E402
from solvers import CEMSolver                       # noqa: E402
from train_decoder import TransformerDecoder, Decoder  # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD     # noqa: E402
from mpc_runner import ACTION_LOW, ACTION_HIGH, load_stats  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--decoder", required=True)
    p.add_argument("--config-id", type=int, default=5)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--num-variations", type=int, default=200)
    p.add_argument("--horizon", type=int, default=24, help="planned (imagined) steps")
    p.add_argument("--n-cols", type=int, default=7, help="columns to show across the horizon")
    p.add_argument("--cem-samples", type=int, default=600)
    p.add_argument("--cem-iters", type=int, default=40)
    p.add_argument("--cem-topk", type=int, default=60)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--eval-budget", type=int, default=75)
    p.add_argument("--env-name", default="RopeFlatten")
    p.add_argument("--out", default="eval/runs/imagine.png")
    return p.parse_args()


def build_model(ckpt, device, H):
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
    d = torch.load(path, map_location=device, weights_only=False)
    cls = TransformerDecoder if d["decoder_type"] == "transformer" else Decoder
    dec = cls(latent_dim=d["latent_dim"], patch=d["patch_size"], out_size=d["img_size"]) \
        if d["decoder_type"] == "transformer" else cls(latent_dim=d["latent_dim"], out_size=d["img_size"])
    dec.load_state_dict(d["decoder_state_dict"]); dec.to(device).eval()
    return dec


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = args.history_size

    model, cfg = build_model(args.ckpt, device, H)
    dec = load_decoder(args.decoder, device)
    a_mean, a_std, p_mean, p_std = load_stats(cfg["data"], device)
    cost = LeWMCost(model, ACTION_LOW, ACTION_HIGH, history_size=H,
                    action_mean=a_mean, action_std=a_std,
                    proprio_mean=p_mean, proprio_std=p_std, device=device)
    solver = CEMSolver(cost, horizon=args.horizon, num_samples=args.cem_samples,
                       n_steps=args.cem_iters, topk=args.cem_topk, device=device, seed=args.seed)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    def denorm_img(x):  # (B,C,H,W) imagenet-norm -> (B,H,W,C) [0,1]
        return (x * std + mean).clamp(0, 1).detach().cpu().numpy().transpose(0, 2, 3, 1)

    env = SubprocessSoftgym(env_name=args.env_name, num_variations=args.num_variations,
                            img_size=args.img_size, num_picker=2)
    try:
        obs = env.reset(config_id=args.config_id, seed=args.seed)
        start_perf = float(obs["info"].get("normalized_performance", 0.0))

        # fixed flat goal = expert reference's flattest frame (same as mpc_runner)
        ref_frames, ref_perfs = env.make_goal_trajectory(n_steps=args.eval_budget, seed=1000 + args.config_id)
        goal_idx = min(int(np.argmax(ref_perfs)) + 1, len(ref_frames) - 1)
        goal_img = ref_frames[goal_idx]
        goal_chw = torch.from_numpy(_to_imagenet_float(goal_img)).float()
        cost.set_goal(goal_chw)

        # history buffers (repeat the start; no past actions)
        A = cost.action_dim
        px0 = torch.from_numpy(_to_imagenet_float(obs["pixels"])).float()
        pix_hist = torch.stack([px0] * H)                       # (H,C,H,W)
        prop0 = torch.from_numpy(np.asarray(obs["proprio"], np.float32))
        info = {
            "hist_emb": cost.encode_obs(pix_hist).unsqueeze(0),                 # (1,H,D)
            "hist_act_norm": torch.zeros(1, H - 1, A, device=device),
        }
        if cost.uses_proprio:
            info["start_proprio"] = prop0.unsqueeze(0).to(device)
            info["hist_proprio_norm"] = cost._norm_proprio(
                torch.stack([prop0] * H).to(device)).unsqueeze(0)

        # plan a long horizon, get the imagined latent trajectory + decode it
        out = solver.solve(info)
        plan = out["actions"][0]                                # (horizon, A) normalized
        print(f"cfg{args.config_id}: start_perf={start_perf:.3f}  CEM cost "
              f"{out['cost_history'][0]:.2f} -> {out['cost_history'][-1]:.2f}")

        imagined = cost.imagine(info, plan)                     # (horizon+1, D): current + imagined
        goal_emb = cost._goal_emb.unsqueeze(0)
        imagined_imgs = denorm_img(dec(imagined))               # (horizon+1, H, W, C)
        goal_recon = denorm_img(dec(goal_emb))[0]               # decoded goal latent

        # execute the SAME plan in the real env for the reality row
        real_frames = [obs["pixels"]]
        for t in range(plan.shape[0]):
            a_raw = torch.clamp(cost.denorm_action(plan[t].to(device)),
                                ACTION_LOW.to(device), ACTION_HIGH.to(device))
            step_out = env.step(a_raw.cpu().numpy().astype(np.float32))
            real_frames.append(step_out["pixels"])
            if step_out["done"]:
                break
        real_perf_final = float(step_out["info"].get("normalized_performance", 0.0))
        print(f"executed plan in env: final_perf={real_perf_final:.3f}")

        # teacher-forced row: ENCODE each real env frame and decode it. Tests the
        # encoder ALONE on the actual intermediate states the rollout visits
        # (separates "encoder can't represent intermediate shapes" from "predictor
        # rollout blurs"). If this row is sharp but the imagined row blurs, the
        # blur is the predictor's latent dynamics, not encoder state-coverage.
        real_chw = torch.stack([torch.from_numpy(_to_imagenet_float(fr)).float()
                                for fr in real_frames]).to(device)
        enc_real = cost.model.projector(cost.model.encoder(real_chw))   # (T_real, D)
        encreal_imgs = denorm_img(dec(enc_real))
        # per-frame recon MSE of the encoder on the real intermediate states
        # (real_chw and the decoder output are both in ImageNet-normalized space)
        enc_mse = float(((dec(enc_real) - real_chw) ** 2).mean().item())
        print(f"encoder recon MSE on real rollout frames: {enc_mse:.4f}")

        # Per-step diagnostic separating the two failure modes:
        #   enc_mse(t)  = decoder recon MSE of encode(real_frame_t)
        #                 -> how well the ENCODER represents the visited state t
        #   drift(t)    = ||predicted_latent_t - encode(real_frame_t)||^2 (latent units)
        #                 -> how far the PREDICTOR's rollout has diverged from truth
        # imagined[t] is the predicted latent after t actions; enc_real[t] is the
        # encoding of the real frame after t actions -> directly aligned.
        recon_per = ((dec(enc_real) - real_chw) ** 2).mean(dim=(1, 2, 3))  # (T_real,)
        Tn = min(imagined.shape[0], enc_real.shape[0])
        drift_per = ((imagined[:Tn] - enc_real[:Tn]) ** 2).sum(dim=-1)     # (Tn,)
        print("  t    enc_recon_mse   predictor_drift(latent L2^2)")
        for t in range(Tn):
            print(f"  {t:2d}   {float(recon_per[t]):.4f}          {float(drift_per[t]):.2f}")
    finally:
        env.close()

    # pick n_cols evenly-spaced timesteps across the horizon (always include 0 and last)
    T = imagined.shape[0]
    cols = sorted(set(np.linspace(0, T - 1, args.n_cols).round().astype(int).tolist()))
    ncol = len(cols) + 1  # +1 for the goal column

    fig, axes = plt.subplots(3, ncol, figsize=(1.5 * ncol, 4.6))
    for r in range(3):
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    for j, t in enumerate(cols):
        axes[0, j].imshow(imagined_imgs[t])
        axes[0, j].set_title(f"t={t}", fontsize=9)
        ri = min(t, len(real_frames) - 1)
        axes[1, j].imshow(encreal_imgs[ri])
        axes[2, j].imshow(real_frames[ri].astype(np.float32) / 255.0)

    # goal column
    axes[0, -1].imshow(goal_recon);                       axes[0, -1].set_title("goal", fontsize=9)
    axes[1, -1].imshow(denorm_img(dec(cost.model.projector(cost.model.encoder(
        torch.from_numpy(_to_imagenet_float(goal_img)).float().unsqueeze(0).to(device)))))[0])
    axes[2, -1].imshow(goal_img.astype(np.float32) / 255.0)

    axes[0, 0].set_ylabel("imagined\n(predicted latent)", fontsize=9, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("encode(reality)\n(decoded latent)", fontsize=9, rotation=0, ha="right", va="center")
    axes[2, 0].set_ylabel("reality\n(env rollout)", fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle(f"Imagined (predictor) vs encode(real) vs reality — RopeFlatten cfg{args.config_id} "
                 f"(start {start_perf:.2f} - goal)", fontsize=11)
    fig.tight_layout()
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
