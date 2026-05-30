#!/usr/bin/env python
"""Overlay metrics from an ablation sweep into comparison panels.

Reads every training/runs/abl_*/metrics.jsonl and overlays the key curves
(pred_loss, sigreg_loss, emb_std, total loss) across runs — a report-ready
figure complementing the W&B view.

Usage:
    python training/plot_ablations.py
    python training/plot_ablations.py --runs-dir training/runs --pattern 'abl_*'
    python training/plot_ablations.py --output ablations.png --log-y
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="training/runs")
    p.add_argument("--pattern", default="abl_*", help="Glob for run dirs.")
    p.add_argument("--output", default=None, help="Default: <runs-dir>/ablation_comparison.png")
    p.add_argument("--x-axis", choices=("step", "epoch"), default="step")
    p.add_argument("--log-y", action="store_true", help="Log-scale the loss panels.")
    return p.parse_args()


def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    run_dirs = sorted(d for d in runs_dir.glob(args.pattern) if (d / "metrics.jsonl").exists())
    if not run_dirs:
        raise SystemExit(f"no runs matching {args.runs_dir}/{args.pattern} with metrics.jsonl")

    runs = {}
    for d in run_dirs:
        rows = load_jsonl(d / "metrics.jsonl")
        if rows:
            # Strip the "abl_<n>_" prefix for cleaner legend labels.
            label = d.name
            for pre in ("abl_",):
                if label.startswith(pre):
                    label = label[len(pre):]
            runs[label] = rows
    print(f"loaded {len(runs)} runs: {', '.join(runs)}")

    panels = [
        ("pred_loss",   "Prediction MSE"),
        ("sigreg_loss", "SIGReg loss"),
        ("emb_std",     "Embedding std (1.0 = ideal)"),
        ("loss",        "Total loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, title) in zip(axes.flat, panels):
        for label, rows in runs.items():
            x = [r.get(args.x_axis, r["step"]) for r in rows if key in r]
            y = [r[key] for r in rows if key in r]
            if y:
                ax.plot(x, y, lw=1.2, label=label, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(args.x_axis)
        ax.grid(alpha=0.3)
        if args.log_y and "loss" in key:
            ax.set_yscale("log")
        if key == "emb_std":
            ax.axhline(1.0, color="k", ls="--", lw=0.7, alpha=0.5)
    axes.flat[0].legend(fontsize=8, loc="upper right")

    fig.suptitle("Ablation sweep comparison", fontsize=13)
    fig.tight_layout()
    out = Path(args.output) if args.output else runs_dir / "ablation_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
