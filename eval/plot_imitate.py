"""Presentation charts for the imitate-MPC evaluation.

Reads the per-run summary JSONs written by eval/run_mpc.py (--mode imitate)
and produces three figures:

  1. method_bar.png      — peak improvement (Δmax = max - start) by method,
                            at a fixed goal-offset k. Shows random vs CEM
                            (paper) vs our gradient planner, baseline vs proprio.
  2. horizon_curve.png   — Δmax vs goal-offset k (planning-horizon sweep),
                            one line per model (baseline / proprio).
  3. within_episode.png  — normalized performance vs env step (mean ± SEM),
                            MPC baseline / proprio vs the expert reference.

Run directory naming convention (under --runs-dir):
    random_k{K}, cem_base_k{K}, cem_prop_k{K},
    grad_base_k{K}, grad_prop_k{K}
Each holds a single summary_k{K}.json.

Usage:
    python eval/plot_imitate.py --runs-dir eval/runs/charts --k-bar 5
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- loading


def _load_run(run_dir):
    """Load the single summary_k*.json in a run dir → list[episode dict] or None."""
    hits = glob.glob(str(Path(run_dir) / "summary_k*.json"))
    if not hits:
        return None
    return json.loads(Path(hits[0]).read_text())


def _delta_max(ep):
    """Peak improvement over the true start performance."""
    return ep["mpc_max"] - ep["start_perf"]


def _delta_final(ep):
    return ep["mpc_final"] - ep["start_perf"]


def _frac_expert(ep):
    """Fraction of the expert's achievable improvement that MPC captured."""
    denom = ep["expert_max"] - ep["start_perf"]
    if denom <= 1e-3:
        return np.nan
    return (ep["mpc_max"] - ep["start_perf"]) / denom


def _agg(eps, fn):
    """mean, SEM over episodes for metric fn (NaN-safe)."""
    v = np.array([fn(e) for e in eps], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.0
    return float(v.mean()), float(v.std() / max(np.sqrt(v.size), 1))


# ---------------------------------------------------------------- figures


def fig_method_bar(runs_dir, k, out):
    """Bar chart: Δmax by method at offset k."""
    specs = [
        ("random_k%d" % k,     "Random",            "#9e9e9e"),
        ("cem_base_k%d" % k,   "CEM (paper)\nbaseline", "#ff9800"),
        ("cem_prop_k%d" % k,   "CEM (paper)\nproprio",  "#fb8c00"),
        ("grad_base_k%d" % k,  "Gradient (ours)\nbaseline", "#1e88e5"),
        ("grad_prop_k%d" % k,  "Gradient (ours)\nproprio",  "#43a047"),
    ]
    labels, means, sems, colors = [], [], [], []
    for d, name, c in specs:
        eps = _load_run(Path(runs_dir) / d)
        if not eps:
            continue
        m, s = _agg(eps, _delta_max)
        labels.append(name); means.append(m); sems.append(s); colors.append(c)
    if not labels:
        print("  [method_bar] no runs found — skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sems, color=colors, capsize=4, edgecolor="black", linewidth=0.6)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.005, f"{m:+.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Peak improvement  Δ = max − start")
    ax.set_title(f"RopeFlatten: improvement by planning method (k={k}, n=8)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def fig_success_bar(runs_dir, k, out, thresh=0.8):
    """Success rate by method at offset k. Success = the MPC rollout reached a
    normalized performance > `thresh` (i.e. a flattened rope)."""
    specs = [
        ("mpc_random_k%d" % k, "Random",  "#9e9e9e"),
        ("mpc_lewm_k%d" % k,   "MPC",     "#1e88e5"),
    ]
    labels, rates, ns, colors = [], [], [], []
    for d, name, c in specs:
        eps = _load_run(Path(runs_dir) / d)
        if not eps:
            continue
        succ = np.array([e["mpc_max"] > thresh for e in eps], dtype=float)
        labels.append(name); rates.append(100 * succ.mean()); ns.append(len(eps)); colors.append(c)
    if not labels:
        print("  [success_bar] no runs found — skipping")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
    for xi, r, n in zip(x, rates, ns):
        ax.text(xi, r + 1.5, f"{r:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(xi, 2, f"n={n}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("MPC rollout success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"RopeFlatten: MPC success rate (k={k})\n"
                 f"success = reached normalized performance > {thresh:g}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def fig_success_latency(runs_dir, k, out, thresh=0.8):
    """Two bars for the MPC planner: success rate (%) and mean planning
    latency per step (s), on a twin y-axis."""
    eps = _load_run(Path(runs_dir) / ("mpc_lewm_k%d" % k))
    if not eps:
        print("  [success_latency] no MPC run found — skipping")
        return
    sr = 100 * np.mean([e["mpc_max"] > thresh for e in eps])
    lat = float(np.mean([e["mean_plan_lat_s"] for e in eps]))
    n = len(eps)

    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax2 = ax1.twinx()
    ax1.bar([0], [sr], width=0.6, color="#1e88e5", edgecolor="black", linewidth=0.6)
    ax2.bar([1], [lat], width=0.6, color="#ff9800", edgecolor="black", linewidth=0.6)

    ax1.text(0, sr + 2, f"{sr:.0f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax2.text(1, lat + 0.03, f"{lat:.2f} s", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["Success rate", "Planning latency"], fontsize=11)
    ax1.set_ylabel("Success rate (%)", color="#1e88e5")
    ax2.set_ylabel("Latency per step (s)", color="#ff9800")
    ax1.set_ylim(0, 100); ax2.set_ylim(0, max(lat * 1.6, 0.5))
    ax1.tick_params(axis="y", colors="#1e88e5"); ax2.tick_params(axis="y", colors="#ff9800")
    ax1.set_title(f"MPC on RopeFlatten (CEM, n={n})")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def fig_success_vs_k(runs_dir, ks, out, thresh=0.8):
    """Success rate vs planning horizon k, with the random baseline as a line.
    Success = reached normalized performance > `thresh`."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tmpl, name, c, mk in [("grad_prop_k%d", "MPC", "#1e88e5", "o")]:
        xs, ys = [], []
        for k in ks:
            eps = _load_run(Path(runs_dir) / (tmpl % k))
            if not eps:
                continue
            xs.append(k); ys.append(100 * np.mean([e["mpc_max"] > thresh for e in eps]))
        if xs:
            ax.plot(xs, ys, marker=mk, color=c, label=name, linewidth=2, markersize=7)
    rnd = _load_run(Path(runs_dir) / "random_k5")
    if rnd:
        r = 100 * np.mean([e["mpc_max"] > thresh for e in rnd])
        ax.axhline(r, color="#9e9e9e", linestyle="--", linewidth=1.6, label=f"Random ({r:.0f}%)")
    ax.set_xlabel("Goal offset k"); ax.set_ylabel("MPC success rate (%)")
    ax.set_ylim(0, 100); ax.set_title(f"MPC success rate vs planning horizon (perf > {thresh:g})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def fig_horizon_curve(runs_dir, ks, out):
    """Δmax vs goal-offset k, one line per model."""
    models = [("grad_base_k%d", "Baseline", "#1e88e5", "o"),
              ("grad_prop_k%d", "Proprio",  "#43a047", "s")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for tmpl, name, c, mk in models:
        xs, ys, es = [], [], []
        for k in ks:
            eps = _load_run(Path(runs_dir) / (tmpl % k))
            if not eps:
                continue
            m, s = _agg(eps, _delta_max)
            xs.append(k); ys.append(m); es.append(s)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker=mk, color=c, label=name,
                        capsize=3, linewidth=2, markersize=7)
            plotted = True
    if not plotted:
        print("  [horizon_curve] no runs found — skipping")
        plt.close(fig); return
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Goal offset k (steps ahead on the expert reference)")
    ax.set_ylabel("Peak improvement  Δ = max − start")
    ax.set_title("RopeFlatten: planning-horizon sweep (gradient planner, n=8)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def _mean_curve(eps, key, T):
    """Stack per-episode perf curves to length T (pad with last value), mean±SEM."""
    rows = []
    for e in eps:
        c = list(e.get(key, []))
        if not c:
            continue
        c = (c + [c[-1]] * T)[:T]
        rows.append(c)
    if not rows:
        return None, None
    M = np.array(rows, dtype=float)
    return M.mean(0), M.std(0) / max(np.sqrt(M.shape[0]), 1)


def fig_within_episode(runs_dir, k, out):
    """Normalized performance vs env step (mean ± SEM), MPC vs expert."""
    T = 76
    fig, ax = plt.subplots(figsize=(7, 4.5))
    eps = _load_run(Path(runs_dir) / ("mpc_lewm_k%d" % k))
    rnd = _load_run(Path(runs_dir) / ("mpc_random_k%d" % k))
    if not eps:
        print("  [within_episode] no runs found — skipping")
        plt.close(fig); return
    mean, sem = _mean_curve(eps, "mpc_perf_curve", T)
    t = np.arange(len(mean))
    start = float(np.mean([e["start_perf"] for e in eps]))
    ax.axhline(start, color="#888", linestyle=":", linewidth=1.3, label=f"start ({start:.2f})")
    ax.plot(t, mean, color="#1e88e5", label="MPC", linewidth=2.2)
    ax.fill_between(t, mean - sem, mean + sem, color="#1e88e5", alpha=0.2)
    if rnd:
        rmean, _ = _mean_curve(rnd, "mpc_perf_curve", T)
        ax.plot(np.arange(len(rmean)), rmean, color="#9e9e9e", label="Random", linewidth=1.8)
    ax.axhline(0.8, color="#43a047", linestyle="--", linewidth=1.3, label="success (0.8)")
    ax.set_xlabel("Env step"); ax.set_ylabel("Normalized performance")
    ax.set_ylim(0, 1)
    ax.set_title(f"RopeFlatten: within-episode performance (MPC vs Random, mean ± SEM, n={len(eps)})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="eval/runs/charts")
    p.add_argument("--out-dir", default=None, help="default: <runs-dir>/plots")
    p.add_argument("--k-bar", type=int, default=5, help="goal-offset for the method bar + within-episode curve")
    p.add_argument("--ks", default="1,5,10,25,50", help="goal-offsets for the horizon curve")
    args = p.parse_args()

    out_dir = Path(args.out_dir or (Path(args.runs_dir) / "plots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",")]

    fig_success_bar(args.runs_dir, args.k_bar, out_dir / "success_rate.png")
    fig_success_latency(args.runs_dir, args.k_bar, out_dir / "success_latency.png")
    fig_success_vs_k(args.runs_dir, ks, out_dir / "success_vs_k.png")
    fig_method_bar(args.runs_dir, args.k_bar, out_dir / "method_bar.png")
    fig_horizon_curve(args.runs_dir, ks, out_dir / "horizon_curve.png")
    fig_within_episode(args.runs_dir, args.k_bar, out_dir / "within_episode.png")
    print(f"\nplots in {out_dir}")


if __name__ == "__main__":
    main()
