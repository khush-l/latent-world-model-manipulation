#!/usr/bin/env python3
"""Aggregate SoftGym LeRobot eval outputs into tables and summary plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", default="simulation/data/evals")
    p.add_argument("--output-dir", default="baselines/plots")
    p.add_argument("--prefix", default="eval_success_filtered")
    p.add_argument(
        "--include-policy",
        action="append",
        default=[],
        help="Only include this policy name. May be passed multiple times.",
    )
    return p.parse_args()


def read_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["eval_dir"] = str(path.parent)
    return data


def read_scores(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))



def _float_or_none(value):
    try:
        if value in (None, ""):
            return None
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if val == val else None


def add_latency_aliases(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        row = dict(row)
        for stat in ("mean", "p50", "p90", "p95", "max"):
            chunk_key = f"policy_chunk_inference_latency_ms_{stat}"
            fallback_key = f"policy_inference_latency_ms_{stat}"
            value = _float_or_none(row.get(chunk_key))
            if value is None:
                value = _float_or_none(row.get(fallback_key))
            if value is not None:
                row[f"policy_forward_pass_latency_ms_{stat}"] = value
        for stat in ("mean", "p50", "p90", "p95", "max"):
            chunk_key = f"policy_chunk_roundtrip_latency_ms_{stat}"
            fallback_key = f"policy_roundtrip_latency_ms_{stat}"
            value = _float_or_none(row.get(chunk_key))
            if value is None:
                value = _float_or_none(row.get(fallback_key))
            if value is not None:
                row[f"policy_forward_pass_roundtrip_ms_{stat}"] = value
        return_note = "chunk_generation_latency_if_available_else_per_step_latency"
        row["latency_reporting"] = return_note
        enriched.append(row)
    return enriched

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)



def write_report_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "policy",
        "env",
        "n_episodes",
        "mean_final_normalized_performance",
        "std_final_normalized_performance",
        "success_rate",
        "policy_forward_pass_latency_ms_p50",
        "policy_forward_pass_latency_ms_mean",
        "policy_forward_pass_latency_ms_p90",
        "policy_forward_pass_roundtrip_ms_p50",
        "policy_forward_pass_roundtrip_ms_mean",
        "policy_inference_latency_ms_mean",
        "policy_cached_action_latency_ms_mean",
        "latency_reporting",
        "checkpoint",
        "created_at",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def policy_order(name: str) -> tuple[int, str]:
    order = {
        "act_rope_random_5k75_5k": 0,
        "act_rope_success_filtered_v2_8k": 1,
        "smolvla_rope_success_filtered_v2_20k": 2,
        "act_cloth_random_5k75_5k": 3,
        "act_cloth_success_filtered_v2_8k": 4,
        "smolvla_cloth_success_filtered_v2_20k": 5,
    }
    return (order.get(name, 999), name)


def plot_bars(summaries: list[dict], out: Path) -> None:
    rows = sorted(summaries, key=lambda r: policy_order(r.get("policy", "")))
    labels = [r.get("policy", Path(r.get("eval_dir", "")).name) for r in rows]
    final_norm = [float(r.get("mean_final_normalized_performance", 0.0)) for r in rows]
    success = [float(r.get("success_rate", 0.0)) for r in rows]
    infer_ms = [float(r.get("policy_forward_pass_latency_ms_p50", r.get("policy_forward_pass_latency_ms_mean", 0.0))) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (final_norm, "Mean Final Normalized Performance", True, "{:.3f}"),
        (success, "Success Rate (final norm >= 0.8)", False, "{:.2f}"),
        (infer_ms, "Forward Pass Latency (ms, p50)", True, "{:.1f} ms"),
    ]
    for ax, (values, title, label_values, label_fmt) in zip(axes, panels):
        bars = ax.bar(range(len(rows)), values)
        ax.set_title(title)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if label_values:
            for bar, value in zip(bars, values):
                ax.annotate(
                    label_fmt.format(value),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            if values:
                ax.set_ylim(top=max(values) * 1.18 if max(values) > 0 else 1.0)
    fig.suptitle("SoftGym Baseline Evaluation Summary", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")


def plot_box(scores: list[dict], out: Path) -> None:
    groups: dict[str, list[float]] = {}
    for row in scores:
        policy = row.get("policy", "unknown")
        try:
            val = float(row.get("final_normalized_performance", "nan"))
        except ValueError:
            continue
        if val == val:
            groups.setdefault(policy, []).append(val)
    labels = sorted(groups, key=policy_order)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot([groups[label] for label in labels], labels=labels, showmeans=True)
    ax.set_title("Final Normalized Performance by Policy")
    ax.set_ylabel("final normalized performance")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [read_summary(p) for p in sorted(eval_root.glob("*/summary.json"))]
    scores = []
    for scores_path in sorted(eval_root.glob("*/scores.csv")):
        scores.extend(read_scores(scores_path))
    if args.include_policy:
        include = set(args.include_policy)
        summaries = [r for r in summaries if r.get("policy") in include]
        scores = [r for r in scores if r.get("policy") in include]

    summary_csv = out_dir / f"{args.prefix}_summary_table.csv"
    scores_csv = out_dir / f"{args.prefix}_all_scores.csv"
    report_csv = out_dir / f"{args.prefix}_report_table.csv"
    summaries = add_latency_aliases(summaries)
    scores = add_latency_aliases(scores)
    write_csv(summary_csv, summaries)
    write_report_csv(report_csv, summaries)
    write_csv(scores_csv, scores)
    if summaries:
        plot_bars(summaries, out_dir / f"{args.prefix}_comparison.png")
    if scores:
        plot_box(scores, out_dir / f"{args.prefix}_boxplots.png")

    print(f"wrote {summary_csv}")
    print(f"wrote {report_csv}")
    print(f"wrote {scores_csv}")


if __name__ == "__main__":
    main()
