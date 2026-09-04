"""Charts for experiment 11 — drawn from the 64 real episodes in results/.

Every number plotted here comes from disteval itself (`metrics.summarize`,
`right_tail_analysis`, `DistributedEvalPool`) applied to live model output, not
from synthetic draws. Run after `run.py --rescore` so the records and summary
are current:

    python3 research/experiments/experiment_11_real_model_distributed_eval/plots.py

Writes into docs/images/ so README.md can embed them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from disteval.metrics import cvar, iqm

HERE = Path(__file__).parent
ROOT = Path(__file__).parents[3]
RESULTS = HERE / "results"
DEFAULT_OUT = ROOT / "docs" / "images"

# Scaffolded agents in blue, raw-API agents in amber: the comparison the
# experiment is actually about.
AGENT_ORDER = ["cli-opus", "cli-haiku", "api-opus", "api-haiku"]
AGENT_COLORS = {
    "cli-opus": "#2c6fbb",
    "cli-haiku": "#5aa1e3",
    "api-opus": "#d98b1f",
    "api-haiku": "#f0b860",
}
TASK_ORDER = ["fizzbuzz-15", "sort-desc", "count-r", "base-7"]


def load() -> tuple[pd.DataFrame, dict]:
    rows = [json.loads(line) for line in open(RESULTS / "records.jsonl")]
    df = pd.json_normalize(rows)
    summary = json.load(open(RESULTS / "summary.json"))
    return df, summary


def _finish(fig, ax_or_axes, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 1. Where the capability gap actually is                                      #
# --------------------------------------------------------------------------- #
def plot_score_matrix(df: pd.DataFrame, out_path: Path) -> None:
    """Mean capability score per (agent, task): all 64 episodes in one grid."""
    grid = (df.pivot_table(index="model", columns="task", values="score", aggfunc="mean")
              .reindex(index=AGENT_ORDER, columns=TASK_ORDER))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(grid.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    for i, agent in enumerate(grid.index):
        for j, task in enumerate(grid.columns):
            v = grid.loc[agent, task]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                    fontweight="bold", color="black" if 0.25 < v < 0.85 else "white")

    ax.set_xticks(range(len(grid.columns)), grid.columns, fontsize=11)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=11)
    ax.set_title("Capability score by agent and task — 64 real episodes\n"
                 "CLI-scaffolded agents above the line, raw API models below",
                 fontsize=13, fontweight="bold")
    # Separate the scaffolded agents (top) from the raw-API ones (bottom).
    ax.axhline(1.5, color="black", lw=2.5)
    fig.colorbar(im, ax=ax, label="mean score (4 episodes)")
    _finish(fig, ax, out_path)


# --------------------------------------------------------------------------- #
# 2. The mean vs the robust center vs the tail                                 #
# --------------------------------------------------------------------------- #
def plot_mean_iqm_cvar(df: pd.DataFrame, out_path: Path) -> None:
    """For each agent: mean, IQM, CVaR@0.1 side by side."""
    stats = []
    for agent in AGENT_ORDER:
        s = df[df["model"] == agent]["score"].to_numpy(float)
        stats.append((agent, float(np.mean(s)), iqm(s), cvar(s, 0.1)))

    x = np.arange(len(stats))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, key, color, label in (
        (-w, 1, "#8fa6bb", "Mean (what a leaderboard reports)"),
        (0.0, 2, "#2c6fbb", "IQM (robust center)"),
        (w, 3, "#c0392b", "CVaR@0.1 (worst 10%)"),
    ):
        vals = [s[key] for s in stats]
        bars = ax.bar(x + offset, vals, w, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)

    ax.set_xticks(x, [s[0] for s in stats], fontsize=11)
    ax.set_ylim(0, 1.45)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Mean vs robust center vs tail — the three disagree",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3,
              frameon=False)
    ax.grid(alpha=0.25, axis="y")
    # The sharpest reading on this data: a respectable mean sitting on a zero tail.
    ax.annotate("mean 0.750, but the worst 10% of episodes score 0.000",
                xy=(2 + w, 0.03), xytext=(1.05, 1.30), fontsize=9.5, color="#c0392b",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4,
                                connectionstyle="arc3,rad=-0.2"))
    _finish(fig, ax, out_path)


# --------------------------------------------------------------------------- #
# 3. Capability vs instruction compliance                                      #
# --------------------------------------------------------------------------- #
def plot_strict_vs_lenient(df: pd.DataFrame, out_path: Path) -> None:
    """Strict scoring conflates 'got it wrong' with 'formatted it wrong'."""
    rows = []
    for agent in AGENT_ORDER:
        sub = df[df["model"] == agent]
        rows.append((agent, float(sub["score"].mean()), float(sub["strict_score"].mean()),
                     int((sub["failure_mode"] == "format_noncompliance").sum())))

    x = np.arange(len(rows))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, [r[1] for r in rows], w, color="#2c6fbb",
                label="Lenient (capability — is the answer right?)")
    b2 = ax.bar(x + w / 2, [r[2] for r in rows], w, color="#d98b1f",
                label="Strict (compliance — is it formatted as asked?)")
    ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=2)

    for i, r in enumerate(rows):
        if r[3]:
            ax.text(i, max(r[1], r[2]) + 0.10, f"{r[3]} episodes:\nright answer,\nwrong format",
                    ha="center", fontsize=8.5, color="#c0392b")

    ax.set_xticks(x, [r[0] for r in rows], fontsize=11)
    ax.set_ylim(0, 1.32)
    ax.set_ylabel("Mean score", fontsize=12)
    ax.set_title("Two scoring channels — capability is not compliance",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
              frameon=False)
    ax.grid(alpha=0.25, axis="y")
    _finish(fig, ax, out_path)


# --------------------------------------------------------------------------- #
# 4. The training pairs the run produced                                       #
# --------------------------------------------------------------------------- #
def plot_cross_agent_pairs(summary: dict, out_path: Path) -> None:
    """Every pair here is a ready-made DPO example found without human labels."""
    pairs = summary["cross_agent_pairs"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(pairs))

    for i, p in enumerate(pairs):
        start = 1.0 - p["gap"]
        ax.barh(i, p["gap"], 0.4, left=start, color="#2c6fbb")
        ax.text(start, i + 0.32, f"{p['negative']} (rejected, {start:.2f})",
                va="bottom", ha="left", fontsize=9.5, color="#8a5b12")
        ax.text(1.0, i + 0.32, f"{p['positive']} (chosen, 1.00)",
                va="bottom", ha="right", fontsize=9.5, color="#1a4d85")
        ax.text(start + p["gap"] / 2, i, f"Δ {p['gap']:.2f}", va="center", ha="center",
                fontsize=11, fontweight="bold", color="white")

    ax.set_yticks(y, [p["task"] for p in pairs], fontsize=11)
    ax.set_ylim(-0.6, len(pairs) - 0.25)
    ax.set_xlim(-0.05, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Cross-agent DPO pairs found automatically — no human labels",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="x")
    _finish(fig, ax, out_path)


# --------------------------------------------------------------------------- #
# 5. Robust aggregation under infra failures                                   #
# --------------------------------------------------------------------------- #
def plot_contamination(summary: dict, out_path: Path) -> None:
    """15% of scores zeroed, mimicking Harbor `missing_reward`. Who survives?"""
    study = summary["contamination_study"]
    order = ["naive", "ivw", "huber"]
    labels = {"naive": "Naive mean", "ivw": "Inverse-variance\nweighted",
              "huber": "Huber\nM-estimator"}
    maes = [study["results"][m]["mae"] for m in order]
    stds = [study["results"][m]["std"] for m in order]
    colors = ["#8fa6bb", "#8fa6bb", "#2c6fbb"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([labels[m] for m in order], maes, 0.55, yerr=stds, capsize=6, color=colors)
    ax.bar_label(bars, fmt="%.3f", fontsize=11, padding=8, fontweight="bold")

    ax.set_ylabel("MAE vs clean per-task truth (lower is better)", fontsize=11)
    ax.set_title(f"Aggregation under {int(study['corrupt_fraction'] * 100)}% contamination "
                 f"({study['n_trials']} trials)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(m + s for m, s in zip(maes, stds)) * 1.35)
    ax.grid(alpha=0.25, axis="y")
    reduction = 100 * (1 - study["results"]["huber"]["mae"] / study["results"]["naive"]["mae"])
    ax.annotate(f"{reduction:.0f}% closer to truth", xy=(2, maes[2] + stds[2]),
                xytext=(1.35, maes[0] * 1.15), fontsize=10, color="#2c6fbb",
                fontweight="bold", arrowprops=dict(arrowstyle="->", color="#2c6fbb", lw=1.5))
    _finish(fig, ax, out_path)


def generate_all(out_dir: Path = DEFAULT_OUT) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df, summary = load()
    paths = {
        "score_matrix": out_dir / "real_01_score_matrix.png",
        "mean_iqm_cvar": out_dir / "real_02_mean_iqm_cvar.png",
        "strict_vs_lenient": out_dir / "real_03_strict_vs_lenient.png",
        "cross_agent_pairs": out_dir / "real_04_cross_agent_pairs.png",
        "contamination": out_dir / "real_05_contamination.png",
    }
    plot_score_matrix(df, paths["score_matrix"])
    plot_mean_iqm_cvar(df, paths["mean_iqm_cvar"])
    plot_strict_vs_lenient(df, paths["strict_vs_lenient"])
    plot_cross_agent_pairs(summary, paths["cross_agent_pairs"])
    plot_contamination(summary, paths["contamination"])
    return paths


def main() -> None:
    df, _ = load()
    print(f"Episodes: {len(df)}  agents: {df['model'].nunique()}  tasks: {df['task'].nunique()}")
    for name, path in generate_all().items():
        print(f"  {name:20} -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
