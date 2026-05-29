#!/usr/bin/env python
"""Render training charts from a metrics.jsonl file.

Usage:
    python training/plot_metrics.py training/runs/<run>/metrics.jsonl
    python training/plot_metrics.py training/runs/<run>/metrics.jsonl --output chart.png
    python training/plot_metrics.py training/runs/<run>/metrics.jsonl --log-y
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path):
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="Path to metrics.jsonl")
    p.add_argument("--output", default=None, help="Output PNG path (default: <jsonl_dir>/charts.png)")
    p.add_argument("--log-y", action="store_true", help="Log-scale y axis for loss panels.")
    p.add_argument("--x-axis", choices=("step", "epoch"), default="step",
                   help="X-axis units. Defaults to training steps.")
    return p.parse_args()


def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl).resolve()
    records = load_jsonl(jsonl_path)
    if not records:
        raise SystemExit(f"no records in {jsonl_path}")

    has_epoch = "epoch" in records[0]
    use_epoch = has_epoch and args.x_axis == "epoch"
    x_vals = [r["epoch"] if use_epoch else r["step"] for r in records]
    x_label = "epoch" if use_epoch else "step"

    series = {
        "loss":        ("Total loss",      [r["loss"] for r in records]),
        "pred_loss":   ("Prediction MSE",  [r["pred_loss"] for r in records]),
        "sigreg_loss": ("SIGReg",          [r["sigreg_loss"] for r in records]),
        "emb_std":     ("Embedding std",   [r["emb_std"] for r in records]),
        "emb_mean":    ("Embedding mean",  [r["emb_mean"] for r in records]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    plot_order = ["loss", "pred_loss", "sigreg_loss", "emb_std", "emb_mean"]
    for ax, key in zip(axes.flat, plot_order):
        title, vals = series[key]
        ax.plot(x_vals, vals, lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.3)
        if args.log_y and "loss" in key:
            ax.set_yscale("log")

    # Hide the unused 6th subplot
    axes.flat[-1].axis("off")
    range_str = (f"epochs: {x_vals[0]:.3f}..{x_vals[-1]:.3f}" if use_epoch
                 else f"steps: {x_vals[0]}..{x_vals[-1]}")
    axes.flat[-1].text(
        0.0, 1.0,
        f"source: {jsonl_path}\n"
        f"n_records: {len(records)}\n"
        f"{range_str}\n"
        f"total steps: {records[-1]['step']}\n"
        f"final loss: {records[-1]['loss']:.5f}\n"
        f"final pred_loss: {records[-1]['pred_loss']:.5f}\n"
        f"final emb_std: {records[-1]['emb_std']:.3f}",
        family="monospace", fontsize=9, va="top",
    )

    fig.suptitle(f"Training metrics — {jsonl_path.parent.name}", fontsize=12)
    fig.tight_layout()

    out_path = Path(args.output) if args.output else jsonl_path.parent / "charts.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
