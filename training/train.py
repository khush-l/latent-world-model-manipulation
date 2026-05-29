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
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


def _git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _total_param_norm(model):
    with torch.no_grad():
        return float(torch.norm(torch.stack([p.detach().norm() for p in model.parameters()
                                             if p.requires_grad])).item())

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
    p.add_argument("--predictor-dim-head", type=int, default=64,
                   help="Per-head dim in the AR predictor. heads×dim_head = inner attn dim. "
                        "LeWM Tab 6: 64 (ViT-S, 1024 inner) is +15 SR over 12 (ViT-T, 192 inner).")
    p.add_argument("--predictor-dropout", type=float, default=0.1)

    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--no-sigreg", action="store_true")

    p.add_argument("--overfit-batches", type=int, default=0,
                   help="If >0, restrict the dataset to N samples and loop indefinitely "
                        "(use --steps to bound training). Smoke test that the model can "
                        "drive pred_loss → 0 on a tiny set.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=0,
                   help="Save a checkpoint every N steps (0 = only at end).")
    p.add_argument("--run-name", default=None,
                   help="Subfolder under --runs-dir; defaults to a timestamp.")
    p.add_argument("--runs-dir", default="training/runs",
                   help="Root directory for per-run metrics + plots.")
    p.add_argument("--no-log-file", action="store_true",
                   help="Skip writing metrics.jsonl (stdout-only).")
    p.add_argument("--device", default=None)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--cache-pixels", action="store_true",
                   help="Pre-load the entire HDF5 into RAM. Eliminates per-step "
                        "gzip decompression — ~10x throughput on small datasets.")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the model in torch.compile() — ~1.5-2x on H100. "
                        "First step pays a ~1 min compile cost.")

    # wandb (opt-in; requires `pip install wandb` and `wandb login`)
    p.add_argument("--wandb", action="store_true",
                   help="Mirror metrics to Weights & Biases. Requires --wandb-entity and --wandb-project.")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B team or username. Required when --wandb is set.")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name. Required when --wandb is set.")
    p.add_argument("--wandb-name", default=None,
                   help="W&B run display name (defaults to --run-name).")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online",
                   help="Use 'offline' on machines without internet; sync later with `wandb sync`.")
    p.add_argument("--wandb-tags", default=None,
                   help="Comma-separated tags for the W&B run, e.g. 'cloth,smoke'.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Free perf knobs — TF32 matmuls + cuDNN heuristics. Both safe on H100.
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    num_steps = args.history_size + args.num_preds
    dataset = HDF5SequenceDataset(
        h5_path=args.data,
        num_steps=num_steps,
        img_size=args.img_size,
        frameskip=args.frameskip,
        cache_pixels=args.cache_pixels,
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
    steps_per_epoch = max(len(loader), 1)
    print(f"steps/epoch: {steps_per_epoch}  ({args.steps} steps ≈ {args.steps/steps_per_epoch:.2f} epochs)")

    model = build_lewm(
        img_size=args.img_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        predictor_dim_head=args.predictor_dim_head,
        predictor_dropout=args.predictor_dropout,
        history_size=args.history_size, num_preds=args.num_preds,
        action_dim=args.frameskip * 8,  # 8D = 2 pickers × 4
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    if args.compile and device == "cuda":
        print("compiling model with torch.compile() (first step pays ~1 min)...")
        model = torch.compile(model)

    sigreg = None if args.no_sigreg else SIGReg().to(device)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = None
    run_dir = None
    if not args.no_log_file:
        run_dir = Path(args.runs_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "metrics.jsonl"
        log_file = log_path.open("w", buffering=1)  # line-buffered
        cfg_dict = vars(args).copy()
        cfg_dict["git_commit"] = _git_commit_hash()
        cfg_dict["n_params"] = n_params
        with (run_dir / "config.json").open("w") as fh:
            json.dump(cfg_dict, fh, indent=2, sort_keys=True)
        print(f"logging to {log_path}")
        print(f"git_commit: {cfg_dict['git_commit']}")

    wandb_run = None
    if args.wandb:
        if not args.wandb_entity or not args.wandb_project:
            raise SystemExit("--wandb requires --wandb-entity and --wandb-project")
        import wandb  # lazy import; only when --wandb is passed
        tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
        wandb_cfg = vars(args).copy()
        wandb_cfg["git_commit"] = _git_commit_hash()
        wandb_cfg["n_params"] = n_params
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name or run_name,
            mode=args.wandb_mode,
            config=wandb_cfg,
            tags=tags,
            dir=str(run_dir) if run_dir is not None else None,
        )
        wandb_run.summary["model/params_M"] = n_params / 1e6
        wandb_run.summary["data/n_windows"] = len(dataset)
        if torch.cuda.is_available():
            wandb_run.summary["hw/gpu_name"] = torch.cuda.get_device_name(0)
        print(f"wandb: {wandb_run.url}" if args.wandb_mode == "online" else "wandb: offline mode")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
        fused=(device == "cuda"),  # ~5-10% speedup on CUDA, free win
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
            # clip_grad_norm_ returns the *pre-clip* total norm — capture it
            # whether or not we actually clip (use a huge max_norm if --grad-clip is 0).
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip if args.grad_clip else float("inf"),
            ).item())
            optim.step()

            elapsed = time.perf_counter() - t_start
            metrics = {
                "step": step,
                "epoch": step / steps_per_epoch,
                "loss": float(loss.item()),
                "pred_loss": float(out["pred_loss"].item()),
                "sigreg_loss": float(out["sigreg_loss"].item()),
                "emb_std": float(out["emb_std"]),
                "emb_mean": float(out["emb_mean"]),
                "grad_norm": grad_norm,
                "weight_norm": _total_param_norm(model),
                "lr": float(optim.param_groups[0]["lr"]),
                "steps_per_sec": (step + 1) / max(elapsed, 1e-6),
                "elapsed_s": elapsed,
            }
            if device == "cuda":
                metrics["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
            if log_file is not None:
                log_file.write(json.dumps(metrics) + "\n")
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if step % args.log_every == 0 or step == args.steps - 1:
                rate = (step + 1) / max(metrics["elapsed_s"], 1e-6)
                print(
                    f"step {step:5d} | loss {metrics['loss']:.5f} | "
                    f"pred {metrics['pred_loss']:.5f} | "
                    f"sig {metrics['sigreg_loss']:.5f} | "
                    f"emb_std {metrics['emb_std']:.3f} | "
                    f"{rate:.1f} steps/s"
                )
            if args.save_every and run_dir is not None and step > 0 and step % args.save_every == 0:
                _save_ckpt(model, run_dir / f"ckpt_step{step:06d}.pt", args, step)
            step += 1

    if run_dir is not None:
        _save_ckpt(model, run_dir / "ckpt_final.pt", args, step)
    if log_file is not None:
        log_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"done in {time.perf_counter() - t_start:.1f}s")
    if run_dir is not None:
        print(f"metrics: {run_dir}/metrics.jsonl")
        print(f"ckpt:    {run_dir}/ckpt_final.pt")
        print(f"plot:    python training/plot_metrics.py {run_dir}/metrics.jsonl")


def _save_ckpt(model, path, args, step):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "step": step,
    }, path)
    print(f"saved ckpt → {path}")


if __name__ == "__main__":
    main()
