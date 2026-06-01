#!/usr/bin/env python3
"""Create presentation charts for the pi0 SoftGym Rope LoRA eval."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

STEP_RE = re.compile(
    r"Step (?P<step>\d+): grad_norm=(?P<grad_norm>[0-9.]+), "
    r"loss=(?P<loss>[0-9.]+), param_norm=(?P<param_norm>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pi0-eval-dir",
        default=ROOT / "simulation/data/evals/pi0_rope_lora_g5",
        type=Path,
    )
    parser.add_argument(
        "--geometric-report",
        default=ROOT / "baselines/results/rope_geometric_5k_v3_20k/eval_rope_geometric_5k_v3_report_table.csv",
        type=Path,
    )
    parser.add_argument(
        "--full-mixed-report",
        default=ROOT / "baselines/results/rope_full_mixed_v1_25k/eval_rope_full_mixed_25k_report_table.csv",
        type=Path,
    )
    parser.add_argument(
        "--train-log",
        default=ROOT / "baselines/logs/pi0_lora_train.log",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT / "baselines/results/pi0_rope_lora_g5",
        type=Path,
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_pi0_summary(eval_dir: Path) -> dict:
    data = json.loads((eval_dir / "summary.json").read_text())
    data["policy"] = "pi0_rope_lora_g5"
    data["dataset"] = "geometric 5k v3"
    return data


def read_pi0_scores(eval_dir: Path) -> list[dict[str, str]]:
    rows = read_csv(eval_dir / "scores.csv")
    for row in rows:
        row["policy"] = "pi0_rope_lora_g5"
    return rows


def baseline_rows(geometric_report: Path, full_mixed_report: Path) -> list[dict]:
    rows: list[dict] = []
    for source, dataset in (
        (geometric_report, "geometric 5k v3"),
        (full_mixed_report, "full mixed v1"),
    ):
        for row in read_csv(source):
            row = dict(row)
            row["dataset"] = dataset
            rows.append(row)
    return rows


def as_float(row: dict, key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def pretty_policy(name: str) -> str:
    labels = {
        "act_rope_geometric_5k_v3_20k": "ACT\ngeo 20k",
        "smolvla_rope_geometric_5k_v3_20k": "SmolVLA\ngeo 20k",
        "act_rope_full_mixed_v1_25k": "ACT\nmixed 25k",
        "smolvla_rope_full_mixed_v1_25k": "SmolVLA\nmixed 25k",
        "pi0_rope_lora_g5": "pi0 LoRA\ngeo 20k",
        "pi0_rope_smoke": "pi0 LoRA\ngeo 20k",
    }
    return labels.get(name, name.replace("_", " "))


def annotate_bars(ax, bars, fmt: str, dy: float = 4.0) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def load_train_records(path: Path) -> list[dict[str, float]]:
    records = []
    text = path.read_text(errors="replace")
    for match in STEP_RE.finditer(text):
        row = match.groupdict()
        records.append({key: float(value) for key, value in row.items()})
    deduped = {int(row["step"]): row for row in records}
    return [deduped[step] for step in sorted(deduped)]


def moving_average(values: list[float], window: int = 7) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(arr, kernel, mode="same")


def plot_comparison(rows: list[dict], out_base: Path) -> None:
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    labels = [pretty_policy(row["policy"]) for row in rows]
    x = np.arange(len(rows))

    final_norm = [as_float(row, "mean_final_normalized_performance") for row in rows]
    success = [100.0 * as_float(row, "success_rate") for row in rows]
    latency = [as_float(row, "policy_forward_pass_latency_ms_p50", as_float(row, "policy_chunk_inference_latency_ms_p50")) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.patch.set_facecolor("white")

    specs = [
        (axes[0], final_norm, "Final normalized performance", "higher is better", "{:.3f}", (0, 0.82)),
        (axes[1], success, "Success rate", "final norm >= 0.8", "{:.0f}%", (0, 35)),
        (axes[2], latency, "Forward-pass latency p50", "chunk generation only", "{:.0f} ms", (0, 380)),
    ]
    for ax, values, title, subtitle, fmt, ylim in specs:
        bars = ax.bar(x, values, color=colors[: len(rows)], width=0.68)
        annotate_bars(ax, bars, fmt)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=8, color="#666666")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("SoftGym RopeFlatten Direct-Action Baselines", fontsize=13, fontweight="bold", y=1.04)
    fig.text(
        0.01,
        -0.03,
        "Note: pi0 LoRA was trained/evaluated on g5.2xlarge A10G; rerun final latency on shared H100 for fair hardware comparison.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_dashboard(pi0_summary: dict, pi0_scores: list[dict[str, str]], train_records: list[dict[str, float]], out_base: Path) -> None:
    episodes = [int(row["episode"]) for row in pi0_scores]
    final_norm = [as_float(row, "final_normalized_performance") for row in pi0_scores]
    chunk_p50 = as_float(pi0_summary, "policy_chunk_inference_latency_ms_p50")
    chunk_max = as_float(pi0_summary, "policy_chunk_inference_latency_ms_max")
    cached_p50 = as_float(pi0_summary, "policy_cached_roundtrip_latency_ms_p50")

    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_perf = fig.add_subplot(gs[0, 0])
    ax_latency = fig.add_subplot(gs[0, 1])
    ax_train = fig.add_subplot(gs[1, 0])
    ax_cards = fig.add_subplot(gs[1, 1])

    colors = ["#54A24B" if value >= 0.8 else "#4C78A8" for value in final_norm]
    bars = ax_perf.bar(episodes, final_norm, color=colors)
    ax_perf.axhline(0.8, color="#E45756", linestyle="--", linewidth=1.2, label="success threshold")
    ax_perf.set_title("pi0 LoRA episode outcomes", loc="left", fontweight="bold")
    ax_perf.set_xlabel("episode")
    ax_perf.set_ylabel("final normalized performance")
    ax_perf.set_ylim(0, max(0.85, max(final_norm) * 1.16))
    ax_perf.grid(axis="y", alpha=0.25)
    ax_perf.legend(frameon=False, fontsize=8)
    annotate_bars(ax_perf, bars, "{:.2f}", dy=3)

    latency_labels = ["chunk p50", "cached\nroundtrip p50", "first compile\nmax"]
    latency_values = [chunk_p50, cached_p50, chunk_max]
    latency_colors = ["#F58518", "#72B7B2", "#E45756"]
    lat_bars = ax_latency.bar(np.arange(3), latency_values, color=latency_colors, width=0.62)
    ax_latency.set_yscale("log")
    ax_latency.set_xticks(np.arange(3))
    ax_latency.set_xticklabels(latency_labels, fontsize=8)
    ax_latency.set_ylabel("ms, log scale")
    ax_latency.set_title("Latency shape", loc="left", fontweight="bold")
    ax_latency.grid(axis="y", alpha=0.25, which="both")
    annotate_bars(ax_latency, lat_bars, "{:.1f} ms", dy=3)

    if train_records:
        steps = [row["step"] for row in train_records]
        loss = moving_average([row["loss"] for row in train_records], window=7)
        grad = moving_average([row["grad_norm"] for row in train_records], window=7)
        ax_train.plot(steps, loss, color="#4C78A8", label="loss, smoothed")
        ax_train.set_ylabel("loss")
        ax_train_twin = ax_train.twinx()
        ax_train_twin.plot(steps, grad, color="#F58518", alpha=0.75, label="grad norm, smoothed")
        ax_train_twin.set_ylabel("grad norm")
        lines, labels = ax_train.get_legend_handles_labels()
        lines2, labels2 = ax_train_twin.get_legend_handles_labels()
        ax_train.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8)
    ax_train.set_title("LoRA training curve", loc="left", fontweight="bold")
    ax_train.set_xlabel("step")
    ax_train.grid(alpha=0.25)

    ax_cards.axis("off")
    card_text = (
        f"Checkpoint\n{pi0_summary['checkpoint']}\n\n"
        f"Mean final norm: {as_float(pi0_summary, 'mean_final_normalized_performance'):.3f}\n"
        f"Std final norm: {as_float(pi0_summary, 'std_final_normalized_performance'):.3f}\n"
        f"Success: {100.0 * as_float(pi0_summary, 'success_rate'):.0f}% ({int(sum(v >= 0.8 for v in final_norm))}/{len(final_norm)})\n"
        f"Mean return: {as_float(pi0_summary, 'mean_return'):.2f}\n\n"
        f"Chunk p50: {chunk_p50:.1f} ms\n"
        f"Chunk p90: {as_float(pi0_summary, 'policy_chunk_inference_latency_ms_p90'):.1f} ms\n"
        f"Cached roundtrip p50: {cached_p50:.1f} ms\n"
        f"First compile max: {chunk_max / 1000.0:.1f} s\n\n"
        f"Output dir\nsimulation/data/evals/pi0_rope_lora_g5"
    )
    ax_cards.text(
        0.02,
        0.98,
        card_text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        bbox={"facecolor": "#F7F7F7", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.6"},
    )

    for ax in (ax_perf, ax_latency, ax_train):
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("pi0 LoRA on SoftGym RopeFlatten", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def write_report(rows: list[dict], pi0_summary: dict, out: Path) -> None:
    cols = [
        "policy",
        "dataset",
        "n_episodes",
        "mean_final_normalized_performance",
        "std_final_normalized_performance",
        "success_rate",
        "policy_forward_pass_latency_ms_p50",
        "policy_chunk_inference_latency_ms_p50",
        "policy_chunk_inference_latency_ms_max",
        "checkpoint",
        "created_at",
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if row["policy"] == "pi0_rope_lora_g5":
                row["policy_forward_pass_latency_ms_p50"] = pi0_summary.get("policy_chunk_inference_latency_ms_p50", "")
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pi0_summary = read_pi0_summary(args.pi0_eval_dir)
    pi0_scores = read_pi0_scores(args.pi0_eval_dir)
    train_records = load_train_records(args.train_log)

    comparison_rows = [
        row
        for row in baseline_rows(args.geometric_report, args.full_mixed_report)
        if row["policy"] in {
            "act_rope_geometric_5k_v3_20k",
            "smolvla_rope_geometric_5k_v3_20k",
            "act_rope_full_mixed_v1_25k",
            "smolvla_rope_full_mixed_v1_25k",
        }
    ]
    comparison_rows.append(pi0_summary)

    plot_comparison(comparison_rows, args.output_dir / "pi0_direct_action_comparison")
    plot_dashboard(pi0_summary, pi0_scores, train_records, args.output_dir / "pi0_rope_lora_dashboard")
    write_report(comparison_rows, pi0_summary, args.output_dir / "pi0_direct_action_report_table.csv")

    print(f"wrote {args.output_dir / 'pi0_direct_action_comparison.png'}")
    print(f"wrote {args.output_dir / 'pi0_direct_action_comparison.pdf'}")
    print(f"wrote {args.output_dir / 'pi0_rope_lora_dashboard.png'}")
    print(f"wrote {args.output_dir / 'pi0_rope_lora_dashboard.pdf'}")
    print(f"wrote {args.output_dir / 'pi0_direct_action_report_table.csv'}")


if __name__ == "__main__":
    main()
