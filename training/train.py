"""LeWM-style JEPA training loop on v3 HDF5 trajectories.

Mirrors references/le-wm-main/train.py:lejepa_forward without the Hydra /
Lightning / stable_pretraining dependencies.

Examples
--------
Overfit smoke test on the 3-episode collection:

    python training/train.py \\
      --data simulation/data/smoke_v3.h5 \\
      --overfit-batches 4 \\
      --batch-size 4 \\
      --img-size 128 --patch-size 16 \\
      --steps 200 --log-every 10

Real run (config matches references/le-wm-main/config/train/lewm.yaml):

    python training/train.py \\
      --data training/data/clothflatten_mixed_v3.h5 \\
      --batch-size 128 --img-size 224 --patch-size 14 \\
      --steps 100000 --lr 5e-5
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import HDF5SequenceDataset
from model import SIGReg, build_lewm, lewm_forward


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Path to v3 HDF5 trajectory file.")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3072)

    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--num-preds", type=int, default=1)
    p.add_argument("--frameskip", type=int, default=1)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=16)
    p.add_argument("--predictor-dropout", type=float, default=0.1)

    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--no-sigreg", action="store_true")

    p.add_argument("--overfit-batches", type=int, default=0,
                   help="If >0, restrict the dataset to N samples and loop indefinitely "
                        "(use --steps to bound training). Smoke test that the model can "
                        "drive pred_loss → 0 on a tiny set.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--run-name", default=None,
                   help="Subfolder under --runs-dir; defaults to a timestamp.")
    p.add_argument("--runs-dir", default="training/runs",
                   help="Root directory for per-run metrics + plots.")
    p.add_argument("--no-log-file", action="store_true",
                   help="Skip writing metrics.jsonl (stdout-only).")
    p.add_argument("--device", default=None)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    num_steps = args.history_size + args.num_preds
    dataset = HDF5SequenceDataset(
        h5_path=args.data,
        num_steps=num_steps,
        img_size=args.img_size,
        frameskip=args.frameskip,
    )
    print(f"dataset: {len(dataset)} windows  (rows={dataset.n_rows}, "
          f"action_dim={dataset.action_dim}, proprio_dim={dataset.proprio_dim}, "
          f"state_dim={dataset.state_dim})")
    print(f"  stats — action: mean[:3]={dataset.stats['action'][0][:3]} std[:3]={dataset.stats['action'][1][:3]}")

    if args.overfit_batches > 0:
        n_keep = min(args.overfit_batches * args.batch_size, len(dataset))
        dataset = Subset(dataset, list(range(n_keep)))
        print(f"overfit mode: using {n_keep} fixed windows")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        drop_last=True, persistent_workers=args.num_workers > 0,
    )

    model = build_lewm(
        img_size=args.img_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        predictor_dropout=args.predictor_dropout,
        history_size=args.history_size, num_preds=args.num_preds,
        action_dim=args.frameskip * 8,  # 8D = 2 pickers × 4
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    sigreg = None if args.no_sigreg else SIGReg().to(device)

    log_file = None
    run_dir = None
    if not args.no_log_file:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.runs_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "metrics.jsonl"
        log_file = log_path.open("w", buffering=1)  # line-buffered
        with (run_dir / "config.json").open("w") as fh:
            json.dump(vars(args), fh, indent=2, sort_keys=True)
        print(f"logging to {log_path}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.precision == "bf16" and device == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    step = 0
    t_start = time.perf_counter()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            model.train()
            with autocast_ctx:
                out = lewm_forward(
                    model, batch,
                    history_size=args.history_size, num_preds=args.num_preds,
                    sigreg=sigreg, sigreg_weight=args.sigreg_weight,
                )
            loss = out["loss"]
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            metrics = {
                "step": step,
                "loss": float(loss.item()),
                "pred_loss": float(out["pred_loss"].item()),
                "sigreg_loss": float(out["sigreg_loss"].item()),
                "emb_std": float(out["emb_std"]),
                "emb_mean": float(out["emb_mean"]),
                "elapsed_s": time.perf_counter() - t_start,
            }
            if log_file is not None:
                log_file.write(json.dumps(metrics) + "\n")
            if step % args.log_every == 0 or step == args.steps - 1:
                rate = (step + 1) / max(metrics["elapsed_s"], 1e-6)
                print(
                    f"step {step:5d} | loss {metrics['loss']:.5f} | "
                    f"pred {metrics['pred_loss']:.5f} | "
                    f"sig {metrics['sigreg_loss']:.5f} | "
                    f"emb_std {metrics['emb_std']:.3f} | "
                    f"{rate:.1f} steps/s"
                )
            step += 1

    if log_file is not None:
        log_file.close()
    print(f"done in {time.perf_counter() - t_start:.1f}s")
    if run_dir is not None:
        print(f"metrics: {run_dir}/metrics.jsonl")
        print(f"plot:    python training/plot_metrics.py {run_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
