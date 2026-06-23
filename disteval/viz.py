"""Visualization layer for disteval distribution reports.

Generates three key plots:
  1. Performance profile (fraction of tasks scoring >= tau, all strata)
  2. CVaR vs Mean bar chart by difficulty stratum -- the "mean lied" visual
  3. Bootstrap CI vs true repeat spread -- the "error bars are wrong" visual

All plots saved to an output directory; also returned as fig objects for
inline display. Zero deps beyond matplotlib + numpy.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from .records import RecordStore
from . import metrics, bootstrap


COLORS = {
    "easy":   "#2ecc71",   # green
    "medium": "#f39c12",   # amber
    "hard":   "#e74c3c",   # red
    "all":    "#3498db",   # blue
}
DIFF_ORDER = ["easy", "medium", "hard"]


# --------------------------------------------------------------------------- #
# 1. Performance profile                                                        #
# --------------------------------------------------------------------------- #
def plot_performance_profile(
    store: RecordStore,
    out_path: str,
    title: str = "Performance Profile by Difficulty",
) -> plt.Figure:
    """Fraction of episodes scoring >= tau, plotted for each stratum.

    This is the whole distribution in one curve — tails included. Rliable's
    primary visualization; we extend it with per-stratum breakdown.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    df = store.df()

    # Overall
    scores_all = df["score"].to_numpy(float)
    taus, fracs = bootstrap.performance_profile(scores_all)
    ax.plot(taus, fracs, color=COLORS["all"], lw=2.5, label="All tasks", zorder=5)
    ax.fill_between(taus, fracs, alpha=0.08, color=COLORS["all"])

    # Per stratum
    if "s_difficulty" in df.columns:
        for diff in DIFF_ORDER:
            sub = df[df["s_difficulty"] == diff]
            if sub.empty:
                continue
            s = sub["score"].to_numpy(float)
            taus_d, fracs_d = bootstrap.performance_profile(s)
            ax.plot(taus_d, fracs_d, color=COLORS.get(diff, "gray"),
                    lw=2, ls="--", label=f"{diff.capitalize()} tasks")

    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Score threshold (τ)", fontsize=12)
    ax.set_ylabel("Fraction of episodes scoring ≥ τ", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    # Derive x-limits from data so scores outside [0, 1] are still visible.
    x_min = scores_all.min() - 0.05
    x_max = scores_all.max() + 0.05
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# 2. CVaR vs Mean bar chart                                                    #
# --------------------------------------------------------------------------- #
def plot_mean_vs_cvar(
    store: RecordStore,
    out_path: str,
    alpha: float = 0.1,
    title: str = "Mean vs CVaR@0.1 — The Mean Hides Tail Risk",
) -> plt.Figure:
    """Side-by-side bars of mean and CVaR@0.1 for each difficulty stratum.

    This is the core visual: mean looks plausible, CVaR exposes collapse on
    hard tasks. The gap between the two bars IS the hidden tail risk.
    """
    df = store.df()
    strata = ["all"] + [d for d in DIFF_ORDER if "s_difficulty" in df.columns
                        and d in df["s_difficulty"].values]
    means, cvars, labels, colors = [], [], [], []

    for s in strata:
        if s == "all":
            sub = df
        else:
            sub = df[df["s_difficulty"] == s]
        if sub.empty:
            continue
        sc = sub["score"].to_numpy(float)
        means.append(float(sc.mean()))
        cvars.append(metrics.cvar(sc, alpha=alpha))
        labels.append("All" if s == "all" else s.capitalize())
        colors.append(COLORS.get(s, "gray"))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_mean = ax.bar(x - width / 2, means, width, label="Mean reward",
                       color=[c + "cc" for c in colors], edgecolor="white", linewidth=1.2)
    bars_cvar = ax.bar(x + width / 2, cvars, width, label=f"CVaR@{alpha} (worst {int(alpha*100)}%)",
                       color=colors, edgecolor="white", linewidth=1.2)

    # Annotate values
    for bar in bars_mean:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9, color="gray")
    for bar in bars_cvar:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    # Annotation arrow on hard if present
    if "Hard" in labels:
        idx = labels.index("Hard")
        m, c = means[idx], cvars[idx]
        if m - c > 0.15:
            ax.annotate("Mean lies here\n↕ gap = hidden tail risk",
                        xy=(x[idx] + width / 2, c + 0.02),
                        xytext=(x[idx] + width / 2 + 0.6, c + 0.25),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                        fontsize=8.5, color="red")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# 3. Pass@k vs Pass^k reliability gap                                          #
# --------------------------------------------------------------------------- #
def plot_pass_reliability(
    store: RecordStore,
    out_path: str,
    k: int = 3,
    title: str = "pass@k vs pass^k — Capability vs Consistency",
) -> plt.Figure:
    """Bar chart showing pass@k (can it ever?) vs pass^k (does it always?).

    The gap IS the reliability deficit — what the mean completely hides.
    Broken down by difficulty stratum.
    """
    df = store.df()
    strata = ["all"] + [d for d in DIFF_ORDER if "s_difficulty" in df.columns
                        and d in df["s_difficulty"].values]
    pat_vals, pht_vals, labels, colors = [], [], [], []

    for s in strata:
        sub = df if s == "all" else df[df["s_difficulty"] == s]
        if sub.empty:
            continue
        pat_vals.append(metrics.pass_at_k(sub, k))
        pht_vals.append(metrics.pass_hat_k(sub, k))
        labels.append("All" if s == "all" else s.capitalize())
        colors.append(COLORS.get(s, "gray"))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width / 2, pat_vals, width, label=f"pass@{k} (peak capability — can it ever?)",
           color=[c + "99" for c in colors], edgecolor="white", linewidth=1.2)
    ax.bar(x + width / 2, pht_vals, width, label=f"pass^{k} (consistency — does it always?)",
           color=colors, edgecolor="white", linewidth=1.2)

    for i, (p, h) in enumerate(zip(pat_vals, pht_vals)):
        ax.text(x[i] - width / 2, p + 0.01, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + width / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        # Gap arrow
        if p - h > 0.1:
            ax.annotate("", xy=(x[i] + width / 2, h + 0.005),
                        xytext=(x[i] - width / 2, p - 0.005),
                        arrowprops=dict(arrowstyle="<->", color="red", lw=1.3))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# 4. Eval reliability: bootstrap CI vs true spread                             #
# --------------------------------------------------------------------------- #
def plot_eval_reliability(
    boot_width: float,
    repeat_width: float,
    out_path: str,
    agent_name: str = "Agent",
    title: str = "Eval Reliability: Reported CI vs True Run-to-Run Spread",
) -> plt.Figure:
    """Visual showing how much the bootstrap CI understates true variance."""
    fig, ax = plt.subplots(figsize=(6, 4))

    categories = ["Published\n(bootstrap CI)", "True spread\n(repeated eval)"]
    widths = [boot_width, repeat_width]
    bar_colors = ["#3498db", "#e74c3c"]

    bars = ax.bar(categories, widths, color=bar_colors, width=0.45, edgecolor="white", linewidth=1.5)
    for bar, w in zip(bars, widths):
        ax.text(bar.get_x() + bar.get_width() / 2, w + 0.001,
                f"±{w:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ratio = repeat_width / boot_width if boot_width > 0 else float("inf")
    ax.set_ylabel("95% CI width", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(widths) * 1.4)
    ax.grid(axis="y", alpha=0.25)

    ax.text(0.5, 0.88, f"True spread is {ratio:.1f}× wider than published CI\n"
            f"→ Published error bars are overconfident by {ratio:.1f}×",
            transform=ax.transAxes, ha="center", fontsize=10,
            color="red", bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Combined report: generate all plots                                          #
# --------------------------------------------------------------------------- #
def generate_all_plots(
    store: RecordStore,
    output_dir: str,
    boot_width: Optional[float] = None,
    repeat_width: Optional[float] = None,
    agent_name: str = "Agent",
) -> dict[str, str]:
    """Generate all plots for a RecordStore. Returns dict of {name: path}."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    p1 = os.path.join(output_dir, "01_performance_profile.png")
    plot_performance_profile(store, p1)
    paths["performance_profile"] = p1

    p2 = os.path.join(output_dir, "02_mean_vs_cvar.png")
    plot_mean_vs_cvar(store, p2)
    paths["mean_vs_cvar"] = p2

    p3 = os.path.join(output_dir, "03_pass_reliability.png")
    plot_pass_reliability(store, p3)
    paths["pass_reliability"] = p3

    if boot_width is not None and repeat_width is not None:
        p4 = os.path.join(output_dir, "04_eval_reliability.png")
        plot_eval_reliability(boot_width, repeat_width, p4, agent_name)
        paths["eval_reliability"] = p4

    return paths
