"""Train a diagnostic decoder: latent z -> reconstructed image.

The encoder + projector are FROZEN (loaded from a trained LeWM checkpoint).
We train only a small ConvTranspose decoder to reconstruct the input frame
from the 192-dim planning latent (post-projector `emb`). This is a probe of
representation quality: if the decoder can reconstruct the rope's shape and
position from the latent, the encoder learned useful geometric features; if
reconstructions are blurry/structureless, it did not — which would explain
poor planning.

Diagnostic only (not used during world-model training), exactly as in the LeWM
paper (their Fig 7/8).

Usage:
    python training/train_decoder.py \
        --ckpt training/runs/rope_full_mixed_v1/ckpt_final.pt \
        --data simulation/data/rope/rope_full_dataset.h5 \
        --steps 6000 --batch-size 64
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import HDF5SequenceDataset, IMAGENET_MEAN, IMAGENET_STD
from model import build_lewm


from einops import rearrange


class DecoderBlock(nn.Module):
    """Transformer decoder block: self-attn among query tokens, cross-attn to
    the latent memory, residual MLP."""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, q, mem):
        qn = self.norm1(q)
        q = q + self.self_attn(qn, qn, qn, need_weights=False)[0]
        q = q + self.cross_attn(self.norm2(q), mem, mem, need_weights=False)[0]
        q = q + self.mlp(self.norm3(q))
        return q


class TransformerDecoder(nn.Module):
    """LeWM-style decoder (paper App D): learnable per-patch query tokens
    cross-attend to the projected latent, then each token is linearly mapped to
    a 16x16x3 pixel patch and rearranged into the full image.
    """
    def __init__(self, latent_dim=192, hidden=384, depth=4, heads=6,
                 patch=16, out_size=128):
        super().__init__()
        self.patch = patch
        self.grid = out_size // patch
        n_patches = self.grid * self.grid
        self.mem_proj = nn.Linear(latent_dim, hidden)
        self.query = nn.Parameter(torch.randn(1, n_patches, hidden) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n_patches, hidden) * 0.02)
        self.layers = nn.ModuleList([DecoderBlock(hidden, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.to_pixels = nn.Linear(hidden, patch * patch * 3)

    def forward(self, z):
        B = z.size(0)
        mem = self.mem_proj(z).unsqueeze(1)                 # (B, 1, hidden)
        q = (self.query + self.pos).expand(B, -1, -1)        # (B, N, hidden)
        for layer in self.layers:
            q = layer(q, mem)
        patches = self.to_pixels(self.norm(q))               # (B, N, patch*patch*3)
        return rearrange(patches, "b (gh gw) (c ph pw) -> b c (gh ph) (gw pw)",
                         gh=self.grid, gw=self.grid, c=3, ph=self.patch, pw=self.patch)


class Decoder(nn.Module):
    """192-dim latent -> (3, img, img) image. DCGAN-style upsampler.

    Decodes from a single 192-d vector to an 8x8 feature map, then 4 transpose
    convolutions take 8 -> 16 -> 32 -> 64 -> 128.
    """
    def __init__(self, latent_dim=192, base=256, out_size=128):
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Linear(latent_dim, base * 8 * 8)
        self.base = base

        def block(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )
        self.net = nn.Sequential(
            block(base, base // 2),       # 8 -> 16
            block(base // 2, base // 4),  # 16 -> 32
            block(base // 4, base // 8),  # 32 -> 64
            nn.ConvTranspose2d(base // 8, 3, 4, stride=2, padding=1),  # 64 -> 128
        )

    def forward(self, z):
        x = self.fc(z).view(-1, self.base, 8, 8)
        return self.net(x)  # (B, 3, 128, 128), ImageNet-normalized space


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--decode-from", choices=("emb", "cls"), default="emb",
                   help="emb = post-projector planning latent (what MPC uses); "
                        "cls = raw encoder CLS token.")
    p.add_argument("--decoder-type", choices=("conv", "transformer"), default="transformer",
                   help="transformer = LeWM-style cross-attention decoder (App D); "
                        "conv = DCGAN-style upsampler.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / "decoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  out: {out_dir}")

    # --- frozen world model ---
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = state.get("config", {})
    model = build_lewm(
        img_size=args.img_size, patch_size=args.patch_size,
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
    print(f"loaded frozen model (step {state.get('step','?')})")

    @torch.no_grad()
    def encode(x):  # x: (B, C, H, W) ImageNet-normed
        cls = model.encoder(x)
        return cls if args.decode_from == "cls" else model.projector(cls)

    # --- data: single frames ---
    ds = HDF5SequenceDataset(args.data, num_steps=1, img_size=args.img_size, frameskip=1)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0, pin_memory=True)
    print(f"dataset: {len(ds)} frames")

    latent_dim = cfg.get("embed_dim", 192)
    if args.decoder_type == "transformer":
        decoder = TransformerDecoder(latent_dim=latent_dim, patch=args.patch_size,
                                     out_size=args.img_size).to(device)
    else:
        decoder = Decoder(latent_dim=latent_dim, out_size=args.img_size).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"decoder: {args.decoder_type}  {n_params/1e6:.2f}M params  (decoding from {args.decode_from})")

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def denorm(x):  # ImageNet-normed -> [0,1] for display
        return (x * std + mean).clamp(0, 1)

    # Hold out a fixed batch for the reconstruction grid.
    fixed = next(iter(loader))["pixels"][:, 0].to(device)  # (B, C, H, W)

    step = 0
    decoder.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            x = batch["pixels"][:, 0].to(device, non_blocking=True)  # single frame
            z = encode(x)
            recon = decoder(z)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % args.log_every == 0 or step == args.steps - 1:
                print(f"step {step:5d} | recon MSE {loss.item():.4f}")
            step += 1

    torch.save({"decoder_state_dict": decoder.state_dict(),
                "decode_from": args.decode_from}, out_dir / "decoder.pt")

    # --- reconstruction grid: original (top) vs reconstruction (bottom) ---
    import imageio.v2 as iio
    decoder.eval()
    with torch.no_grad():
        recon = decoder(encode(fixed))
        orig_img = denorm(fixed).cpu().numpy().transpose(0, 2, 3, 1)  # (B,H,W,3)
        rec_img = denorm(recon).cpu().numpy().transpose(0, 2, 3, 1)
    ncol = min(8, fixed.size(0))
    rows = []
    for r_imgs in (orig_img, rec_img):
        row = np.concatenate([(r_imgs[i] * 255).astype(np.uint8) for i in range(ncol)], axis=1)
        rows.append(row)
    sep = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    grid = np.concatenate([rows[0], sep, rows[1]], axis=0)
    iio.imwrite(out_dir / "reconstructions.png", grid)
    print(f"\nwrote {out_dir}/reconstructions.png  (top: original, bottom: reconstruction)")
    print(f"wrote {out_dir}/decoder.pt")


if __name__ == "__main__":
    main()
