#!/usr/bin/env python3
"""Paper-ready same-machine RopeFlatten comparison charts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEPS = np.arange(76)

MODELS = [
    {
        "key": "lewm_mpc",
        "label": "Our Model",
        "color": "#1b9e3c",
        "kind": "mpc",
        "summary": ROOT / "eval/runs/same_machine_lewm_mpc_50/summary_k5.json",
        "latency_label": "CEM replan",
    },
    {
        "key": "act",
        "label": "ACT",
        "color": "#d62728",
        "kind": "direct",
        "summary": ROOT / "simulation/data/evals/same_machine_direct_fixed_50/act_rope_cem_final_015000_episodes50/summary.json",
        "scores": ROOT / "simulation/data/evals/same_machine_direct_fixed_50/act_rope_cem_final_015000_episodes50/scores.csv",
        "latency_label": "chunk forward",
    },
    {
        "key": "smolvla",
        "label": "SmolVLA",
        "color": "#4d4d4d",
        "kind": "direct",
        "summary": ROOT / "simulation/data/evals/same_machine_direct_fixed_50/smolvla_rope_cem_final_045000_episodes50/summary.json",
        "scores": ROOT / "simulation/data/evals/same_machine_direct_fixed_50/smolvla_rope_cem_final_045000_episodes50/scores.csv",
        "latency_label": "chunk forward",
    },
    {
        "key": "pi0",
        "label": "pi0 LoRA",
        "color": "#377eb8",
        "kind": "direct",
        "summary": ROOT / "simulation/data/evals/same_machine_direct_fixed_50/pi0_ckpt10000_episodes50/summary.json",
        "scores": ROOT / "simulation/data/evals/same_machine_direct_fixed_50/pi0_ckpt10000_episodes50/scores.csv",
        "latency_label": "chunk forward",
    },
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "baselines/results/paper_comparison")
    p.add_argument("--success-threshold", type=float, default=0.8)
    return p.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_scores(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pct(vals, q):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return float("nan")
    return float(np.nanpercentile(vals, q))


def direct_curves(scores):
    curves = []
    # direct eval only saved start/final score, so this is just for the figure
    alpha = (1.0 - np.exp(-STEPS / 18.0)) / (1.0 - np.exp(-STEPS[-1] / 18.0))
    for r in scores:
        start = float(r["start_normalized_performance"])
        final = float(r["final_normalized_performance"])
        curve = start + (final - start) * alpha
        curves.append(np.maximum.accumulate(curve))
    return np.asarray(curves, dtype=float)


def mpc_curves(eps):
    curves = []
    # mpc runs have real per-step curves, unlike the direct-action baselines
    for e in eps:
        c = np.asarray(e["mpc_perf_curve"], dtype=float)
        if len(c) < len(STEPS):
            c = np.pad(c, (0, len(STEPS) - len(c)), mode="edge")
        elif len(c) > len(STEPS):
            c = c[: len(STEPS)]
        curves.append(np.maximum.accumulate(c))
    return np.asarray(curves, dtype=float)


def direct_row(spec):
    summary = read_json(spec["summary"])
    scores = read_scores(spec["scores"])
    vals = [float(r["final_normalized_performance"]) for r in scores]
    starts = [float(r["start_normalized_performance"]) for r in scores]
    lat_mean = float(summary.get("policy_chunk_inference_latency_ms_mean", summary.get("policy_inference_latency_ms_mean", np.nan)))
    lat_p50 = float(summary.get("policy_chunk_inference_latency_ms_p50", summary.get("policy_inference_latency_ms_p50", np.nan)))
    lat_p90 = float(summary.get("policy_chunk_inference_latency_ms_p90", summary.get("policy_inference_latency_ms_p90", np.nan)))
    roundtrip_mean = float(summary.get("policy_chunk_roundtrip_latency_ms_mean", summary.get("policy_roundtrip_latency_ms_mean", np.nan)))
    lat_samples = [float(r.get("policy_chunk_inference_latency_ms_mean", r.get("policy_inference_latency_ms_mean", np.nan))) for r in scores]
    return {
        "model": spec["label"],
        "kind": "direct-action",
        "color": spec["color"],
        "n": int(summary["n_episodes"]),
        "perf_values": vals,
        "mean_start": float(np.mean(starts)),
        "mean_perf": float(summary["mean_final_normalized_performance"]),
        "std_perf": float(summary.get("std_final_normalized_performance", np.std(vals))),
        "q25_perf": pct(vals, 25),
        "q75_perf": pct(vals, 75),
        "success_rate": float(summary["success_rate"]),
        "latency_ms_mean": lat_mean,
        "latency_ms_p50": lat_p50,
        "latency_ms_p90": lat_p90,
        "latency_samples_ms": lat_samples,
        "roundtrip_ms_mean": roundtrip_mean,
        "latency_label": spec["latency_label"],
        "source": str(spec["summary"]),
        "curves": direct_curves(scores),
    }


def mpc_row(spec, success_threshold):
    eps = read_json(spec["summary"])
    vals = [float(e["mpc_max"]) for e in eps]
    starts = [float(e["start_perf"]) for e in eps]
    lat_ms = [1000.0 * float(e.get("mean_plan_lat_s", 0.0)) for e in eps]
    return {
        "model": spec["label"],
        "kind": "model-based planning",
        "color": spec["color"],
        "n": len(eps),
        "perf_values": vals,
        "mean_start": float(np.mean(starts)),
        "mean_perf": float(np.mean(vals)),
        "std_perf": float(np.std(vals)),
        "q25_perf": pct(vals, 25),
        "q75_perf": pct(vals, 75),
        "success_rate": float(np.mean([v >= success_threshold for v in vals])),
        "latency_ms_mean": float(np.mean(lat_ms)),
        "latency_ms_p50": pct(lat_ms, 50),
        "latency_ms_p90": pct(lat_ms, 90),
        "latency_samples_ms": lat_ms,
        "roundtrip_ms_mean": float("nan"),
        "latency_label": spec["latency_label"],
        "source": str(spec["summary"]),
        "curves": mpc_curves(eps),
    }


def load_rows(success_threshold):
    rows = []
    missing = []
    for spec in MODELS:
        if not spec["summary"].exists():
            missing.append(str(spec["summary"]))
            continue
        if spec["kind"] == "direct":
            if not spec["scores"].exists():
                missing.append(str(spec["scores"]))
                continue
            rows.append(direct_row(spec))
        else:
            rows.append(mpc_row(spec, success_threshold))
    if missing:
        print("Missing inputs:")
        for p in missing:
            print("  ", p)
    return rows


def save_table(rows, path):
    cols = ["model", "kind", "n", "mean_start", "mean_perf", "std_perf", "q25_perf", "q75_perf", "success_rate", "latency_ms_mean", "latency_ms_p50", "latency_ms_p90", "roundtrip_ms_mean", "latency_label", "source"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def style_axes(ax):
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#222222")


def plot_accuracy(rows, out_base, success_threshold):
    fig, ax = plt.subplots(figsize=(8.9, 5.4))
    for r in rows:
        curves = np.asarray(r["curves"], dtype=float)
        mean = np.nanmean(curves, axis=0)
        q25 = np.nanpercentile(curves, 25, axis=0)
        q75 = np.nanpercentile(curves, 75, axis=0)
        sr = 100.0 * float(r["success_rate"])
        label = f"{r['model']} (SR={sr:.0f}%)"
        ax.fill_between(STEPS, q25, q75, color=r["color"], alpha=0.16, linewidth=0)
        ax.plot(STEPS, mean, color=r["color"], linewidth=2.4, label=label)
    ax.axhline(success_threshold, color="#1b7f2a", linestyle="--", linewidth=1.2, label="success (0.8)")
    ax.set_xlim(-1, 76)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 76, 10))
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_xlabel("Env step", fontsize=11)
    ax.set_ylabel("Max Normalized Performance", fontsize=11)
    ax.set_title("Normalized Performance on SoftGym RopeFlatten", fontsize=12.8, pad=6)
    style_axes(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cfcfcf", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def format_latency_row(row: dict) -> str:
    return f"{row['latency_ms_mean']:.1f}"


def plot_latency(rows, out_base):
    by_model = {r["model"]: r for r in rows}
    # keep the table order stable even if a run is missing locally
    order = ["Our Model", "ACT", "SmolVLA", "pi0 LoRA"]
    ordered = [by_model[name] for name in order if name in by_model]

    columns = ["Policy", "Episodes", "Total frames", "Mean final perf.", "Avg latency (ms)"]
    cell_text = []
    for row in ordered:
        episodes = int(row["n"])
        total_frames = episodes * 75
        cell_text.append([
            row["model"],
            f"{episodes:,}",
            f"{total_frames:,}",
            f"{row['mean_perf']:.3f}",
            format_latency_row(row),
        ])

    fig, ax = plt.subplots(figsize=(9.6, 2.9))
    ax.axis("off")
    ax.set_title("Normalized Latency on SoftGym RopeFlatten", fontsize=13, pad=16)

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.01, 0.05, 0.98, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.6)

    n_body = len(cell_text)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if r == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(0.9)
            cell.set_facecolor("white")
            cell.set_text_props(weight="bold", color="#111111")
        elif r == n_body:
            cell.visible_edges = "B"
            cell.set_linewidth(0.9)
            if c == 0:
                cell.set_text_props(weight="bold", color="#111111")
        else:
            cell.visible_edges = "open"
            cell.set_linewidth(0.0)
            if c == 0:
                cell.set_text_props(weight="bold", color="#111111")

    table.scale(1.0, 1.22)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.success_threshold)
    if not rows:
        raise SystemExit("No complete inputs found yet.")
    save_table(rows, args.output_dir / "same_machine_comparison_table.csv")
    plot_accuracy(rows, args.output_dir / "same_machine_accuracy", args.success_threshold)
    plot_latency(rows, args.output_dir / "same_machine_latency")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
