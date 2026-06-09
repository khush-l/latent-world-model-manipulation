#!/usr/bin/env python3
"""Render LeRobot text logs in the same dashboard style as training/plot_metrics.py."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_RE = re.compile(
    r"step:(?P<step>[0-9K]+).*?epch:(?P<epoch>[0-9.]+) "
    r"loss:(?P<loss>[0-9.]+) grdn:(?P<grad_norm>[0-9.]+) "
    r"lr:(?P<lr>[0-9.e+-]+) updt_s:(?P<update_s>[0-9.]+) data_s:(?P<data_s>[0-9.]+)"
)


def step_value(value: str) -> int:
    if value.endswith("K"):
        return int(float(value[:-1]) * 1000)
    return int(value)


def moving_avg(xs, window):
    if window <= 1 or window > len(xs):
        return xs
    out = []
    total = 0.0
    # rolling sum keeps this simple and fast enough for long logs
    for i, x in enumerate(xs):
        total += x
        if i >= window:
            total -= xs[i - window]
        out.append(total / min(i + 1, window))
    return out


def load_records(path: Path):
    records = []
    text = path.read_text(errors="replace").replace("\r", "\n")
    # lerobot logs are plain text, so regex is less brittle than split cols
    for match in LOG_RE.finditer(text):
        row = match.groupdict()
        records.append(
            {
                "step": step_value(row["step"]),
                "epoch": float(row["epoch"]),
                "loss": float(row["loss"]),
                "grad_norm": float(row["grad_norm"]),
                "lr": float(row["lr"]),
                "update_s": float(row["update_s"]),
                "data_s": float(row["data_s"]),
            }
        )

    # LeRobot prints abbreviated step labels like 1K for several nearby rows.
    deduped = {}
    for record in records:
        deduped[record["step"]] = record
    return [deduped[k] for k in sorted(deduped)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", help="LeRobot train log file.")
    p.add_argument("--output", default=None, help="Output PNG path.")
    p.add_argument("--run-name", default=None, help="Title run name.")
    p.add_argument("--smooth", type=int, default=1, help="Moving-average smoothing window.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log).resolve()
    records = load_records(log_path)
    if not records:
        raise SystemExit(f"no LeRobot metric records found in {log_path}")

    steps = [r["step"] for r in records]
    panels = [
        ("loss", "Total loss"),
        ("grad_norm", "Gradient norm"),
        ("lr", "Learning rate"),
        ("update_s", "Update time"),
        ("data_s", "Data time"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (key, title) in zip(axes.flat, panels):
        vals = [r[key] for r in records]
        if args.smooth > 1:
            vals = moving_avg(vals, args.smooth)
        ax.plot(steps, vals, lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if key == "lr":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes.flat[-1].axis("off")
    # use the last panel as a tiny run card instead of making another plot
    axes.flat[-1].text(
        0.0,
        1.0,
        f"source: {log_path}\n"
        f"n_records: {len(records)}\n"
        f"steps: {steps[0]}..{steps[-1]}\n"
        f"final loss: {records[-1]['loss']:.5f}\n"
        f"final grad_norm: {records[-1]['grad_norm']:.3f}\n"
        f"final epoch: {records[-1]['epoch']:.2f}\n"
        f"smoothing window: {args.smooth}",
        family="monospace",
        fontsize=9,
        va="top",
    )

    run_name = args.run_name or log_path.stem
    fig.suptitle(f"Training metrics — {run_name}", fontsize=12)
    fig.tight_layout()

    out_path = Path(args.output) if args.output else log_path.with_name(f"{log_path.stem}_charts.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
