"""Train a learned temporal-distance head on a FROZEN JEPA encoder.

d(z_a, z_b) ~ number of env steps to get from state a to state b (directed, a
before b), within an episode, normalized to [0,1] by clipping at K steps. Used as
an MPC cost-to-go that replaces the flat terminal-latent MSE metric (which has a
weak gradient both far from and near the goal). Cross-episode pairs are labeled 1
(maximally far) so unrelated states don't look close.

Encoder is frozen; only the small MLP head trains. We precompute latents for a
subset of episodes once (frozen encoder), then sample pairs from RAM — fast.

Usage:
    python training/train_temporal_distance.py \
        --ckpt training/runs/rope_proprio_v2_250k/ckpt_final.pt \
        --data simulation/data/rope/rope_full_dataset.h5 \
        --max-episodes 5000 --K 40 --steps 8000 \
        --out training/runs/rope_proprio_v2_250k/tempdist_head.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from model import build_lewm                       # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD     # noqa: E402


class DistanceHead(nn.Module):
    """MLP estimating normalized directed temporal distance in [0,1]."""
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, za, zb):
        return torch.sigmoid(self.net(torch.cat([za, zb], dim=-1)).squeeze(-1))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--max-episodes", type=int, default=5000, help="episodes to encode for training")
    p.add_argument("--K", type=int, default=40, help="max step-gap; gaps clip here, normalized by it")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--p-cross", type=float, default=0.2, help="fraction of cross-episode (far) pairs")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def build_encoder(ckpt, device):
    cfg = json.loads((Path(ckpt).parent / "config.json").read_text())
    model = build_lewm(
        img_size=cfg.get("img_size", 128), patch_size=cfg.get("patch_size", 16),
        embed_dim=cfg.get("embed_dim", 192), predictor_depth=cfg.get("predictor_depth", 6),
        predictor_heads=cfg.get("predictor_heads", 16), predictor_dim_head=cfg.get("predictor_dim_head", 64),
        predictor_dropout=cfg.get("predictor_dropout", 0.1),
        history_size=cfg.get("history_size", 3), num_preds=cfg.get("num_preds", 1),
        use_proprio=bool(cfg.get("use_proprio", False)),
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def precompute_latents(model, h5_path, ranges, img_size, device):
    """Encode CONTIGUOUS row-ranges (one per episode) -> (Z, ep, st).

    Reads each episode as a contiguous slice px[s:e] (fast sequential gzip read)
    instead of a giant scattered fancy-index (which is CPU-bound and starves the
    GPU). Returns latents (M,D) plus the episode_idx / step_idx arrays aligned to
    the row order of Z.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    Zs, eps, sts = [], [], []
    with h5py.File(h5_path, "r") as f:
        px = f["pixels"]; epd = f["episode_idx"]; stp = f["step_idx"]
        for k, (s, e) in enumerate(ranges):
            x = torch.from_numpy(px[s:e].astype(np.float32) / 255.).permute(0, 3, 1, 2).to(device)
            x = (x - mean) / std
            Zs.append(model.projector(model.encoder(x)))
            eps.append(epd[s:e]); sts.append(stp[s:e])
            if k % 500 == 0:
                print(f"   encoded {k}/{len(ranges)} episodes", flush=True)
    return torch.cat(Zs, 0), np.concatenate(eps), np.concatenate(sts)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    model, cfg = build_encoder(args.ckpt, device)
    D = cfg.get("embed_dim", 192)

    # contiguous (start,end) row-range per episode, in one O(N) pass
    with h5py.File(args.data, "r") as f:
        ep_all = f["episode_idx"][:]
    change = np.where(np.diff(ep_all) != 0)[0] + 1
    starts = np.concatenate([[0], change]); ends = np.concatenate([change, [len(ep_all)]])
    ep_range = {int(ep_all[s]): (int(s), int(e)) for s, e in zip(starts, ends)}
    all_eps = np.array(sorted(ep_range.keys()))
    keep = rng.choice(all_eps, size=min(args.max_episodes, len(all_eps)), replace=False)
    ranges = [ep_range[int(e)] for e in keep]
    nframes = sum(e - s for s, e in ranges)
    print(f"encoding {nframes} frames from {len(ranges)} episodes (contiguous slices) ...", flush=True)
    Z, ep, st = precompute_latents(model, args.data, ranges, args.img_size, device)  # (M,D),(M,),(M,)
    M = Z.shape[0]
    print(f"latents: {tuple(Z.shape)}", flush=True)

    # per-episode position lists (positions into Z), sorted by step
    ep_to_pos = {}
    order = np.argsort(st, kind="stable")
    for pos in order:
        ep_to_pos.setdefault(int(ep[pos]), []).append(int(pos))
    ep_to_pos = {e: np.asarray(v) for e, v in ep_to_pos.items()}
    ep_list = list(ep_to_pos.keys())
    st_t = torch.tensor(st, dtype=torch.float32, device=device)

    head = DistanceHead(D, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    def sample_batch(B):
        """Return (a_idx, b_idx, label) for B pairs (positions into Z)."""
        n_cross = int(B * args.p_cross)
        n_same = B - n_cross
        a = np.empty(B, np.int64); b = np.empty(B, np.int64); lab = np.empty(B, np.float32)
        # within-episode directed pairs (a before b), gap in [0, K] clipped
        k = 0
        while k < n_same:
            e = ep_list[rng.randint(len(ep_list))]
            pos = ep_to_pos[e]
            if len(pos) < 2:
                continue
            i = rng.randint(len(pos) - 1)
            gap = rng.randint(1, args.K + 1)
            j = min(i + gap, len(pos) - 1)
            a[k] = pos[i]; b[k] = pos[j]
            lab[k] = min(j - i, args.K) / args.K
            k += 1
        # cross-episode pairs -> far (label 1)
        a[n_same:] = rng.randint(M, size=n_cross)
        b[n_same:] = rng.randint(M, size=n_cross)
        lab[n_same:] = 1.0
        return (torch.from_numpy(a).to(device), torch.from_numpy(b).to(device),
                torch.from_numpy(lab).to(device))

    head.train()
    for step in range(args.steps):
        ai, bi, lab = sample_batch(args.batch_size)
        pred = head(Z[ai], Z[bi])
        loss = F.mse_loss(pred, lab)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}", flush=True)

    # quick monotonicity sanity: on a few held episodes, does d(z_t, z_flat) drop
    # as t approaches the flattest frame?
    head.eval()
    with torch.no_grad():
        checks = []
        for e in ep_list[:50]:
            pos = ep_to_pos[e]
            if len(pos) < 10:
                continue
            zf = Z[pos[-1]].unsqueeze(0)            # last frame as pseudo-goal
            zt = Z[pos]                             # all frames
            d = head(zt, zf.expand_as(zt))          # (len,)
            # correlation of predicted distance with true remaining steps
            rem = torch.arange(len(pos) - 1, -1, -1, device=device, dtype=torch.float32)
            dd = d - d.mean(); rr = rem - rem.mean()
            corr = (dd * rr).sum() / (dd.norm() * rr.norm() + 1e-8)
            checks.append(float(corr))
        print(f"monotonicity check: mean corr(d(z_t,z_T), steps_remaining) = {np.mean(checks):.3f} "
              f"(want ~ +1)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                "dim": D, "hidden": args.hidden, "K": args.K,
                "ckpt": str(args.ckpt), "data": str(args.data)}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
