"""Decode the same fixed frames from EVERY training checkpoint.

Trains one diagnostic decoder per checkpoint (each frozen encoder has its own
latent space, so the decoder cannot be shared), reconstructs a fixed set of
frames, and assembles a grid: row 0 = originals, then one row per checkpoint
(labeled by training step). Shows how the latent's information content evolves
over training.

Usage:
    python eval/decode_across_checkpoints.py \
        --run-dir training/runs/rope_full_mixed_v1 \
        --data simulation/data/rope/rope_full_dataset.h5 \
        --decoder-steps 4000 --n-frames 6
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from dataset import HDF5SequenceDataset, IMAGENET_MEAN, IMAGENET_STD
from model import build_lewm
from train_decoder import TransformerDecoder, Decoder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Dir with ckpt_step*.pt + ckpt_final.pt")
    p.add_argument("--data", required=True)
    p.add_argument("--decoder-steps", type=int, default=4000)
    p.add_argument("--decoder-type", choices=("transformer", "conv"), default="transformer")
    p.add_argument("--n-frames", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--n-checkpoints", type=int, default=0,
                   help="If >0, evenly subsample to this many checkpoints "
                        "(always including the first and last).")
    return p.parse_args()


def subsample(ckpts, n):
    """Evenly pick n checkpoints, always including first and last."""
    if n <= 0 or n >= len(ckpts):
        return ckpts
    idx = np.linspace(0, len(ckpts) - 1, n).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [ckpts[i] for i in idx]


def find_checkpoints(run_dir):
    """Return [(step, path)] sorted by step; ckpt_final treated as the max step."""
    run = Path(run_dir)
    out = []
    for f in run.glob("ckpt_step*.pt"):
        m = re.search(r"ckpt_step(\d+)\.pt", f.name)
        if m:
            out.append((int(m.group(1)), f))
    final = run / "ckpt_final.pt"
    if final.exists():
        # label final with the configured total steps if available
        import json
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        out.append((int(cfg.get("steps", 10**9)), final))
    return sorted(out)


def build_model(ckpt_path, cfg_fallback, device, img_size, patch_size):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("config", cfg_fallback)
    model = build_lewm(
        img_size=img_size, patch_size=patch_size,
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
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def train_decoder_for(model, loader, fixed_frames, args, device, latent_dim):
    """Train a fresh decoder against this frozen model; return recon of fixed_frames."""
    if args.decoder_type == "transformer":
        dec = TransformerDecoder(latent_dim=latent_dim, patch=args.patch_size,
                                 out_size=args.img_size).to(device)
    else:
        dec = Decoder(latent_dim=latent_dim, out_size=args.img_size).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-4)

    @torch.no_grad()
    def encode(x):
        return model.projector(model.encoder(x))

    dec.train()
    step = 0
    while step < args.decoder_steps:
        for batch in loader:
            if step >= args.decoder_steps:
                break
            x = batch["pixels"][:, 0].to(device, non_blocking=True)
            recon = dec(encode(x))
            loss = F.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

    dec.eval()
    with torch.no_grad():
        rec = dec(encode(fixed_frames))
    return rec, float(loss.item())


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out) if args.out else Path(args.run_dir) / "decode_across_checkpoints.png"

    ckpts = find_checkpoints(args.run_dir)
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.run_dir}")
    ckpts = subsample(ckpts, args.n_checkpoints)
    print(f"using {len(ckpts)} checkpoints: {[s for s,_ in ckpts]}")

    # Dataset + loader built ONCE (independent of the model).
    ds = HDF5SequenceDataset(args.data, num_steps=1, img_size=args.img_size, frameskip=1)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0, pin_memory=True)
    print(f"dataset: {len(ds)} frames")

    # Fixed frames (same across all checkpoints) — deterministic indices.
    rng = np.random.RandomState(args.seed)
    idx = sorted(rng.choice(len(ds), size=args.n_frames, replace=False).tolist())
    fixed = torch.stack([ds[i]["pixels"][0] for i in idx], dim=0).to(device)  # (N,C,H,W)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    def denorm(x):
        return (x * std + mean).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)

    rows = [("original", denorm(fixed))]
    cfg_fallback = {}
    for step, path in ckpts:
        print(f"-> checkpoint step {step}: training decoder ({args.decoder_steps} steps)...", flush=True)
        model, cfg = build_model(path, cfg_fallback, device, args.img_size, args.patch_size)
        cfg_fallback = cfg
        latent_dim = cfg.get("embed_dim", 192)
        rec, mse = train_decoder_for(model, loader, fixed, args, device, latent_dim)
        rows.append((f"step {step}", denorm(rec)))
        print(f"   done (final recon MSE {mse:.4f})", flush=True)
        del model
        torch.cuda.empty_cache()

    # Assemble grid with matplotlib: rows = checkpoints, cols = frames.
    nrow, ncol = len(rows), args.n_frames
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.4 * ncol, 1.4 * nrow))
    if nrow == 1:
        axes = axes[None, :]
    for r, (label, imgs) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(imgs[c]); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Decoder reconstructions across training checkpoints", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
