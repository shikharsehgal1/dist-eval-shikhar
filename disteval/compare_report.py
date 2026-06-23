"""Head-to-head agent comparison report — the ranking inversion demo.

Usage (2 agents):
    python -m disteval.compare_report <jobs_dir_A> <jobs_dir_B> \\
        --agent-a <name> --agent-b <name> --tasks-dir <dir> --output-dir <dir>

Usage (3 agents):
    python -m disteval.compare_report <jobs_dir_A> <jobs_dir_B> <jobs_dir_C> \\
        --agent-a <name> --agent-b <name> --agent-c <name> \\
        --tasks-dir <dir> --output-dir <dir>

Produces:
  1. Terminal ranking table: Harbor leaderboard vs disteval ranking
  2. Side-by-side metric comparison chart (PNG)
  3. Per-difficulty breakdown chart (PNG)
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .adapters.harbor_jobs import load_harbor_job
from .records import RecordStore
from . import metrics


PALETTE = ["#e74c3c", "#3498db", "#2ecc71"]   # red, blue, green
DIFF_COLORS = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"}
DIFF_ORDER = ["easy", "medium", "hard"]


def _color(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def _hr(title: str, width: int = 72) -> None:
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print('═' * width)


def build_agent_summary(store: RecordStore, agent_name: str) -> dict:
    df = store.df()
    scores = df["score"].to_numpy(float)
    summary = {
        "agent": agent_name,
        "n": int(len(scores)),
        "mean":          float(scores.mean()),
        "iqm":           metrics.iqm(scores),
        "median":        float(np.median(scores)),
        "std":           float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "cvar@0.1":      metrics.cvar(scores, 0.1),
        "var@0.1":       metrics.var_at(scores, 0.1),
        "pass@3":        metrics.pass_at_k(df, 3),
        "pass^3":        metrics.pass_hat_k(df, 3),
        "success_rate":  float(df["success"].mean()),
    }
    strata = {}
    if "s_difficulty" in df.columns:
        for diff in DIFF_ORDER:
            sub = df[df["s_difficulty"] == diff]
            if sub.empty:
                continue
            s = sub["score"].to_numpy(float)
            strata[diff] = {
                "n":        int(len(s)),
                "mean":     float(s.mean()),
                "cvar@0.1": metrics.cvar(s, 0.1),
                "pass^3":   metrics.pass_hat_k(sub, 3),
            }
    summary["strata"] = strata
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Terminal output
# ─────────────────────────────────────────────────────────────────────────────

def print_leaderboard(summaries: list[dict]) -> None:
    """Print Harbor-style (mean-only) ranking vs disteval ranking side by side."""
    _hr(f"DISTEVAL 3-AGENT LEADERBOARD  ({len(summaries)} agents, {summaries[0]['n']}–{summaries[-1]['n']} trials each)")

    # Harbor ranking
    harbor_rank = sorted(summaries, key=lambda s: -s["mean"])
    print("\n  ┌──── Harbor sees only this ─────────────────────────────────┐")
    for i, s in enumerate(harbor_rank, 1):
        bar = "█" * int(s["mean"] * 20)
        print(f"  │  #{i}  {s['agent']:<30}  mean={s['mean']:.3f}  {bar}")
    print("  └────────────────────────────────────────────────────────────┘")

    # Full metric table
    METRICS = [
        ("Mean",            "mean",         "Harbor leaderboard metric"),
        ("IQM",             "iqm",          "Outlier-resistant center"),
        ("CVaR@0.1",        "cvar@0.1",     "Expected score in worst 10% of runs"),
        ("pass@3",          "pass@3",       "Can it ever solve the task?"),
        ("pass^3",          "pass^3",       "Does it ALWAYS solve the task?"),
        ("Success rate",    "success_rate", "Fraction of episodes fully passing"),
    ]

    _hr("FULL DISTRIBUTIONAL COMPARISON", width=72)
    names = [s["agent"][:16] for s in summaries]
    header = f"  {'Metric':<18}" + "".join(f"  {n:>16}" for n in names) + "  Best"
    print(header)
    print("  " + "─" * (18 + len(summaries) * 18 + 8))

    ranking_wins = {s["agent"]: 0 for s in summaries}
    for label, key, desc in METRICS:
        vals = [s[key] for s in summaries]
        best_val = max(vals)
        best_agents = [s["agent"] for s in summaries if abs(s[key] - best_val) < 1e-9]
        row = f"  {label:<18}"
        for s in summaries:
            v = s[key]
            cell = f"{v:>16.3f}"
            if s["agent"] in best_agents and len(best_agents) < len(summaries):
                cell = _color(cell, "32")  # green = winner
            row += "  " + cell
        best_str = "/".join(a[:10] for a in best_agents) if len(best_agents) < len(summaries) else "tie"
        row += f"  {_color(best_str, '32')}"
        print(row)
        for a in best_agents:
            if len(best_agents) < len(summaries):
                ranking_wins[a] += 1

    # Disteval ranking
    disteval_rank = sorted(summaries, key=lambda s: -(s["cvar@0.1"] * 2 + s["pass^3"] + s["iqm"]))
    print()
    print("  ┌──── disteval reliability ranking ─────────────────────────┐")
    for i, s in enumerate(disteval_rank, 1):
        wins = ranking_wins[s["agent"]]
        harbor_pos = next(j+1 for j, h in enumerate(harbor_rank) if h["agent"] == s["agent"])
        flip = f"  ↕ (was #{harbor_pos} on Harbor)" if i != harbor_pos else ""
        print(f"  │  #{i}  {s['agent']:<28}  wins {wins}/5 metrics{flip}")
    print("  └────────────────────────────────────────────────────────────┘")

    # Inversion check
    if harbor_rank[0]["agent"] != disteval_rank[0]["agent"]:
        winner_harbor = harbor_rank[0]["agent"]
        winner_dist   = disteval_rank[0]["agent"]
        _hr("⚡ RANKING INVERSION DETECTED", width=72)
        print(f"  Harbor #1: {_color(winner_harbor, '33')}  (mean={harbor_rank[0]['mean']:.3f})")
        print(f"  disteval #1: {_color(winner_dist, '32')}  (more reliable tail + consistency)")
        cvar_gap = summaries[next(i for i,s in enumerate(summaries) if s['agent']==winner_dist)]['cvar@0.1'] \
                 - summaries[next(i for i,s in enumerate(summaries) if s['agent']==winner_harbor)]['cvar@0.1']
        print(f"  CVaR gap: {winner_dist} has {abs(cvar_gap):.3f} better tail floor")
        print(f"  → The mean rewarded {winner_harbor} for peak shots; disteval found {winner_dist} is more consistent in deployment")
    else:
        print(f"\n  Harbor and disteval agree on #1: {_color(harbor_rank[0]['agent'], '32')}")
        print("  But distributional metrics reveal important per-agent tail risk differences ↑")

    # Per-difficulty breakdown
    _hr("PER-DIFFICULTY BREAKDOWN", width=72)
    print(f"  {'Diff':<8}  {'Metric':<12}" + "".join(f"  {s['agent'][:14]:>14}" for s in summaries))
    print("  " + "─" * (22 + len(summaries) * 16))
    for diff in DIFF_ORDER:
        first = True
        for metric, key in [("Mean", "mean"), ("CVaR@0.1", "cvar@0.1"), ("pass^3", "pass^3")]:
            diff_label = diff.capitalize() if first else ""
            first = False
            vals = []
            row = f"  {diff_label:<8}  {metric:<12}"
            for s in summaries:
                v = s["strata"].get(diff, {}).get(key, None)
                vals.append(v)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    row += f"  {'—':>14}"
                else:
                    row += f"  {v:>14.3f}"
            print(row)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_3way_metrics(summaries: list[dict], out_path: str) -> plt.Figure:
    """6-metric grouped bar chart for all agents."""
    metric_labels = ["Mean", "IQM", "CVaR@0.1", "pass@3", "pass^3", "Success"]
    metric_keys   = ["mean", "iqm", "cvar@0.1", "pass@3", "pass^3", "success_rate"]

    n_agents = len(summaries)
    n_metrics = len(metric_labels)
    x = np.arange(n_metrics)
    total_width = 0.7
    width = total_width / n_agents
    offsets = np.linspace(-total_width/2 + width/2, total_width/2 - width/2, n_agents)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, (s, color) in enumerate(zip(summaries, PALETTE)):
        vals = [s[k] for k in metric_keys]
        bars = ax.bar(x + offsets[i], vals, width, label=s["agent"],
                      color=color, edgecolor="white", linewidth=1.0, alpha=0.88)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                        color=color, fontweight="bold")

    # Shade the "Harbor metric" column
    ax.axvspan(-0.5, 0.5, alpha=0.07, color="orange", zorder=0)
    ax.text(0, 1.10, "Harbor metric\n(all they show)", ha="center", va="top",
            fontsize=7.5, color="darkorange", transform=ax.get_xaxis_transform())
    ax.axvspan(1.5, 5.5, alpha=0.04, color="steelblue", zorder=0)
    ax.text(3.5, 1.10, "disteval adds these →", ha="center", va="top",
            fontsize=7.5, color="steelblue", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.set_title("3-Agent Benchmark: Harbor Mean vs Distributional Metrics (disteval)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_difficulty_breakdown(summaries: list[dict], out_path: str) -> plt.Figure:
    """CVaR@0.1 by difficulty for each agent — shows where each agent collapses."""
    diffs = [d for d in DIFF_ORDER if any(d in s["strata"] for s in summaries)]
    n_diffs = len(diffs)
    n_agents = len(summaries)
    x = np.arange(n_diffs)
    total_width = 0.65
    width = total_width / n_agents
    offsets = np.linspace(-total_width/2 + width/2, total_width/2 - width/2, n_agents)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, (metric_key, metric_label) in enumerate([("cvar@0.1", "CVaR@0.1 (tail risk)"), ("pass^3", "pass^3 (consistency)")]):
        ax = axes[ax_idx]
        for i, (s, color) in enumerate(zip(summaries, PALETTE)):
            vals = [s["strata"].get(d, {}).get(metric_key, 0.0) for d in diffs]
            bars = ax.bar(x + offsets[i], vals, width, label=s["agent"],
                          color=color, edgecolor="white", linewidth=1.0, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                        color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in diffs], fontsize=12)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_ylim(0, 1.2)
        ax.set_title(f"{metric_label} by Difficulty", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Where do agents ACTUALLY fail? (Harbor can't tell you this)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="disteval multi-agent comparison")
    parser.add_argument("jobs_dirs", nargs="+", help="Harbor jobs dirs (2 or 3 agents)")
    parser.add_argument("--agents", nargs="+", default=None,
                        help="Agent display names (same count as jobs_dirs)")
    parser.add_argument("--tasks-dir", "-t", default=None)
    parser.add_argument("--output-dir", "-o", default="disteval_comparison")
    parser.add_argument("--completed-only", action="store_true", default=True)
    args = parser.parse_args(argv)

    agent_names = args.agents or [f"Agent {chr(65+i)}" for i in range(len(args.jobs_dirs))]
    if len(agent_names) != len(args.jobs_dirs):
        parser.error("--agents must have same count as jobs_dirs")

    summaries = []
    print("Loading agents...")
    for job_dir, name in zip(args.jobs_dirs, agent_names):
        store = load_harbor_job(job_dir, run_id="run0", tasks_dir=args.tasks_dir)
        if args.completed_only:
            records = [r for r in store._records if r.failure_mode != "missing_reward"]
            store = RecordStore(records)
        n_infra = sum(1 for r in load_harbor_job(job_dir, run_id="x", tasks_dir=args.tasks_dir)._records
                      if r.failure_mode == "missing_reward")
        print(f"  {name}: {len(store)} completed trials ({n_infra} infra errors excluded)")
        summaries.append(build_agent_summary(store, name))

    print_leaderboard(summaries)

    os.makedirs(args.output_dir, exist_ok=True)

    p1 = os.path.join(args.output_dir, "comparison_metrics.png")
    plot_3way_metrics(summaries, p1)
    print(f"\nSaved: {p1}")

    p2 = os.path.join(args.output_dir, "comparison_by_difficulty.png")
    plot_difficulty_breakdown(summaries, p2)
    print(f"Saved: {p2}")


if __name__ == "__main__":
    main()
