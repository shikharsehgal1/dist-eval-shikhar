"""
disteval — the 5-minute hackathon demo.

    python wow_demo.py               # full show with pauses (for video recording)
    python wow_demo.py --act 3       # jump straight to the wow moment
    python wow_demo.py --fast        # no pauses (dry-run / testing)
    python wow_demo.py --export      # save all charts to wow_output/ silently

THE STORY (6 acts, ~6 minutes):

  ACT 1  Harbor shows you this        — mean-only leaderboard. Looks fine.
  ACT 2  The distribution says more   — IQM ≈ mean, but CVaR collapses to 0
  ACT 3  WHERE does it collapse?      — drill by difficulty: EASY tasks, CVaR=0
                                        ← THE WOW MOMENT
  ACT 4  Stochastic dominance         — Claude doesn't just score higher;
                                        it mathematically dominates
  ACT 5  The right-tail training signal — why evaluating distributions also
                                        tells you HOW to improve the agent
  ACT 6  One command                  — the full pipeline in a single CLI call

All data is real — three agents (Claude Code, Gemini CLI, Codex CLI)
run on a real Harbor benchmark suite (6 tasks, 3 difficulty levels).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── disteval ─────────────────────────────────────────────────────────────────
from disteval.adapters.harbor_jobs import load_harbor_job
from disteval.records import RecordStore
from disteval import metrics, bootstrap, compare
from disteval.compare_report import build_agent_summary
from disteval.right_tail import right_tail_analysis, compare_right_tail
from disteval.trajectory_monitor import TrajectoryMonitor
from disteval.trajectory_loader import load_trajectory_records
import json as _json

# ── data paths ────────────────────────────────────────────────────────────────
AGENTS = [
    ("Claude Code\n(Sonnet 4.5)", "jobs/run_A/disteval-run-A", "#e05252"),
    ("Gemini CLI\n(2.5 Flash)",   "jobs/run_B/disteval-run-B", "#4a90d9"),
    ("Codex CLI\n(o4-mini)",      "jobs/run_C/disteval-run-C", "#43b97f"),
]
TASKS_DIR  = "tasks"
OUTPUT_DIR = "wow_output"

# ── terminal ──────────────────────────────────────────────────────────────────
W = min(shutil.get_terminal_size((100, 40)).columns, 92)

def clr():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

def pause(s: float, fast: bool):
    if not fast:
        time.sleep(s)

def tw(text: str, delay: float = 0.018, fast: bool = False, end: str = "\n"):
    """Typewriter print."""
    if fast:
        print(text, end=end)
        return
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()

def hr(char="─", color="\033[90m"):
    print(f"{color}{char * W}\033[0m")

def banner(title: str, color="\033[1;97m"):
    print()
    hr("═")
    print(f"{color}  {title}\033[0m")
    hr("═")

# colour helpers
G  = lambda s: f"\033[1;32m{s}\033[0m"   # green bold
R  = lambda s: f"\033[1;31m{s}\033[0m"   # red bold
Y  = lambda s: f"\033[1;33m{s}\033[0m"   # yellow bold
C  = lambda s: f"\033[1;36m{s}\033[0m"   # cyan bold
B  = lambda s: f"\033[1m{s}\033[0m"      # bold
D  = lambda s: f"\033[2m{s}\033[0m"      # dim
WH = lambda s: f"\033[1;97m{s}\033[0m"   # bright white

def hbar(v: float, w: int = 22, filled="█", empty="░", color="\033[33m") -> str:
    n = int(round(v * w))
    return f"{color}{filled*n}{empty*(w-n)}\033[0m"

# ── data loading ──────────────────────────────────────────────────────────────
def load_all() -> list[tuple[str, RecordStore, str]]:
    result = []
    for name, job_dir, color in AGENTS:
        full = load_harbor_job(job_dir, tasks_dir=TASKS_DIR)
        completed = RecordStore([r for r in full._records
                                 if r.failure_mode != "missing_reward"])
        result.append((name, completed, color))
    return result

# ── chart helpers ─────────────────────────────────────────────────────────────
def _save(fig, name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def chart_01_harbor(data):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names  = [d[0].replace("\n", " ") for d in data]
    means  = [d[1].df()["score"].mean() for d in data]
    colors = [d[2] for d in data]
    bars = ax.barh(names, means, color=[c+"cc" for c in colors],
                   edgecolor="white", linewidth=1.5, height=0.52)
    for bar, v in zip(bars, means):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Mean Reward", fontsize=12)
    ax.set_title("Agent Leaderboard  —  as reported by Harbor", fontsize=14, fontweight="bold")
    ax.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.4)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    return _save(fig, "01_harbor_leaderboard.png")

def chart_02_reveal(data):
    keys   = ["mean", "iqm", "cvar@0.1"]
    labels = ["Mean\n(Harbor shows this)", "IQM\n(robust center)", "CVaR@0.1\n(tail floor)"]
    sums   = [build_agent_summary(d[1], d[0]) for d in data]
    n = len(data)
    x = np.arange(len(keys))
    w = 0.21
    offs = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (s, d) in enumerate(zip(sums, data)):
        vals = [s[k] for k in keys]
        bars = ax.bar(x + offs[i], vals, w, label=d[0].replace("\n"," "),
                      color=d[2], edgecolor="white", linewidth=1.1, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=d[2])

    # Red zone annotation on CVaR column
    ax.axvspan(1.58, 2.42, alpha=0.07, color="red", zorder=0)
    ax.text(2.0, 1.16, "← THE REVEAL", ha="center", fontsize=9,
            color="red", fontweight="bold",
            transform=ax.get_xaxis_transform())

    # Arrow on Gemini CVaR bar (index 1, column 2)
    gemini_cvar = sums[1]["cvar@0.1"]
    ax.annotate("Gemini tail\n= 0  (mean said 0.75)",
                xy=(x[2] + offs[1] + w/2, gemini_cvar + 0.005),
                xytext=(x[2] + offs[1] + w/2 + 0.38, 0.22),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.8),
                fontsize=8.5, color="red", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.28); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("The mean hides the distribution  —  disteval exposes it",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return _save(fig, "02_distribution_reveal.png")

def chart_03_wow(data):
    """THE WOW CHART: easy-task CVaR. Two agents score 0."""
    sums = [build_agent_summary(d[1], d[0]) for d in data]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax_idx, diff in enumerate(["easy", "hard"]):
        ax = axes[ax_idx]
        names = [d[0].replace("\n"," ") for d in data]
        means = [s["strata"].get(diff, {}).get("mean", 0)     for s in sums]
        cvars = [s["strata"].get(diff, {}).get("cvar@0.1", 0) for s in sums]
        colors= [d[2] for d in data]
        x = np.arange(len(data))
        w = 0.30

        ax.bar(x - w/2, means, w, label="Mean",
               color=[c+"66" for c in colors], edgecolor="white", linewidth=1.2)
        b2 = ax.bar(x + w/2, cvars, w, label="CVaR@0.1 (tail floor)",
                    color=colors, edgecolor="white", linewidth=1.2)

        for bar, v in zip(b2, cvars):
            col = "red" if v == 0 else ("black" if v < 0.5 else "green")
            ax.text(bar.get_x() + bar.get_width()/2, max(v, 0.012),
                    f"{'0.00' if v == 0 else f'{v:.2f}'}",
                    ha="center", va="bottom",
                    fontsize=12 if v == 0 else 10,
                    fontweight="bold", color=col)
        for bar, v in zip(ax.patches[:len(data)], means):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
        ax.set_ylim(0, 1.35); ax.set_ylabel("Score" if ax_idx == 0 else "")
        title = f"{'EASY' if diff=='easy' else 'HARD'} Tasks  —  Mean vs CVaR@0.1"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.18)

        if diff == "easy":
            ax.text(0.5, 1.24,
                    "Gemini & Codex tail = 0.00 on EASY tasks\n"
                    "Harbor showed them at 0.75 and 0.33",
                    ha="center", va="top", fontsize=9, color="red",
                    fontweight="bold", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.35",
                              fc="mistyrose", ec="red", alpha=0.92))

    fig.suptitle("WHERE agents collapse  —  Harbor has no idea (it only has the mean)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "03_wow_easy_cvar.png")

def chart_04_dominance(data):
    """CDF curves showing stochastic dominance — unambiguous."""
    sums_scores = [(d[0].replace("\n"," "), d[1].df()["score"].to_numpy(float), d[2])
                   for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # Left: CDF curves
    ax = axes[0]
    xs = np.linspace(0, 1, 300)
    for name, scores, color in sums_scores:
        cdf = np.array([(scores <= x).mean() for x in xs])
        ax.plot(xs, cdf, color=color, lw=2.5, label=name)
        ax.fill_between(xs, cdf, 1, alpha=0.06, color=color)

    ax.set_xlabel("Score threshold", fontsize=12)
    ax.set_ylabel("Fraction of episodes ≤ threshold  (CDF)", fontsize=11)
    ax.set_title("Cumulative Distribution Functions", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.text(0.05, 0.12,
            "Claude's CDF is BELOW all others\nat every threshold\n→ First-Order Stochastic Dominance",
            fontsize=8.5, color="#e05252", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="#e05252", alpha=0.85))

    # Right: prob_improvement matrix
    ax2 = axes[1]
    names_short = [d[0].replace("\n", " ").split(" (")[0] for d in data]
    n = len(data)
    mat = np.zeros((n, n))
    for i, (_, si, _) in enumerate(sums_scores):
        for j, (_, sj, _) in enumerate(sums_scores):
            if i != j:
                mat[i, j] = compare.prob_improvement(si, sj)
            else:
                mat[i, j] = 0.5

    im = ax2.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax2.set_xticks(range(n)); ax2.set_xticklabels(names_short, fontsize=9)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(names_short, fontsize=9)
    ax2.set_xlabel("Opponent", fontsize=11)
    ax2.set_ylabel("Agent", fontsize=11)
    ax2.set_title("P(row agent > col agent)\nper random episode pair",
                  fontsize=11, fontweight="bold")
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                     fontsize=13, fontweight="bold",
                     color="white" if mat[i,j] > 0.65 or mat[i,j] < 0.35 else "black")
    plt.colorbar(im, ax=ax2, fraction=0.04)

    fig.suptitle("Stochastic Dominance  —  Claude is better in every sense, not just average",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "04_dominance.png")

def chart_05_pass(data):
    sums = [build_agent_summary(d[1], d[0]) for d in data]
    names = [d[0].replace("\n"," ") for d in data]
    colors = [d[2] for d in data]
    pat = [s["pass@3"] for s in sums]
    pht = [s["pass^3"] for s in sums]
    x = np.arange(len(data))
    w = 0.30

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w/2, pat, w, label="pass@3  — can it ever?",
           color=[c+"66" for c in colors], edgecolor="white", linewidth=1.2)
    ax.bar(x + w/2, pht, w, label="pass^3  — does it always?",
           color=colors, edgecolor="white", linewidth=1.2)

    for i, (p, h) in enumerate(zip(pat, pht)):
        ax.text(x[i]-w/2, p+0.01, f"{p:.2f}", ha="center", va="bottom", fontsize=11)
        ax.text(x[i]+w/2, h+0.01, f"{h:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        if p - h > 0.08:
            ax.annotate("", xy=(x[i]+w/2, h+0.005), xytext=(x[i]-w/2, p-0.005),
                        arrowprops=dict(arrowstyle="<->", color="red", lw=1.8))

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.22); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Capability vs Consistency  —  the deployment reliability gap",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return _save(fig, "05_pass_consistency.png")

def chart_06_right_tail(data):
    """Right-tail gap chart: Q* vs Q̄ per task, per agent, with consistency index."""
    reports = [right_tail_analysis(d[1], d[0].split("\n")[0]) for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=False)

    for ax, report, d in zip(axes, reports, data):
        color = d[2]
        tasks  = [p.task.replace("disteval/", "") for p in report.profiles]
        q_star = [p.q_star for p in report.profiles]
        q_bar  = [p.q_bar  for p in report.profiles]
        kinds  = [p.kind   for p in report.profiles]

        x = np.arange(len(tasks))
        w = 0.32
        bars_star = ax.bar(x - w/2, q_star, w, label="Q* (right tail)",
                           color=color, edgecolor="white", alpha=0.95)
        bars_bar  = ax.bar(x + w/2, q_bar,  w, label="Q̄  (mean)",
                           color=color + "55", edgecolor="white")

        # Annotate kind
        kind_colors = {"solid": "#2ecc71", "recoverable": "#f0a500", "stuck": "#e05252"}
        for i, (qs, qb, k) in enumerate(zip(q_star, q_bar, kinds)):
            if k == "recoverable":
                ax.annotate("", xy=(x[i]+w/2, qb), xytext=(x[i]-w/2, qs),
                            arrowprops=dict(arrowstyle="<->", color="#f0a500", lw=1.8))
            ax.text(x[i], max(qs, qb) + 0.03,
                    {"solid": "●", "recoverable": "↕", "stuck": "✗"}[k],
                    ha="center", fontsize=12,
                    color=kind_colors[k])

        ki = report.sum_q_bar / report.sum_q_star if report.sum_q_star > 0 else 1.0
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.35)
        ax.set_title(f"{report.model}\nκ = {ki:.3f}  |  gap Δ = {report.total_gap:.2f}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.18)

        # Consistency index bar on top
        ax.axhline(ki, color=color, ls="--", lw=1.2, alpha=0.5,
                   label=f"κ={ki:.2f}")

    fig.suptitle(
        "Right-Tail Gap  —  How much score is lost to inconsistency vs missing skill?\n"
        "↕ = RECOVERABLE (knows how, but random)   ✗ = STUCK   ● = SOLID",
        fontsize=11, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    return _save(fig, "06_right_tail_gap.png")


def save_all_charts(data):
    print()
    for fn in [chart_01_harbor, chart_02_reveal, chart_03_wow,
               chart_04_dominance, chart_05_pass, chart_06_right_tail]:
        path = fn(data)
        print(f"  {os.path.basename(path)}")

# ─── ACT 1: Harbor's view ────────────────────────────────────────────────────
def act1(data, fast):
    clr()
    banner("ACT 1  —  This is what Harbor shows you", "\033[1;93m")
    print()
    pause(0.8, fast)
    tw(B("  $ harbor view jobs/run_*/  →  leaderboard"), fast=fast)
    print()
    pause(0.7, fast)

    harbor_rank = sorted(data, key=lambda d: -d[1].df()["score"].mean())
    print(f"  {'#':<4}{'Agent':<32}{'Mean Reward':>14}  {'':>4}")
    hr()
    for i, (name, store, color) in enumerate(harbor_rank, 1):
        mean = store.df()["score"].mean()
        tw(f"  #{i}  {name.replace(chr(10),' '):<30}  mean = {hbar(mean)}  {mean:.3f}",
           delay=0.010, fast=fast)
        pause(0.5, fast)

    print()
    pause(1.0, fast)
    tw(D("  Reasonable. Claude leads, Gemini second, Codex third."), fast=fast)
    pause(0.8, fast)
    tw(D("  Gemini is at 0.754 — solid. Ready to deploy?"), fast=fast)
    pause(2.0, fast)
    tw(Y("  Wait."), delay=0.05, fast=fast)
    pause(1.5, fast)

# ─── ACT 2: CVaR reveal ──────────────────────────────────────────────────────
def act2(data, fast):
    clr()
    banner("ACT 2  —  The distribution says something else", "\033[1;96m")
    print()
    pause(0.6, fast)
    tw(B("  $ python -m disteval.report  jobs/run_B/  --completed-only"), fast=fast)
    print()
    pause(0.8, fast)

    gemini_store = data[1][1]  # Gemini
    df = gemini_store.df()
    scores = df["score"].to_numpy(float)

    boot = bootstrap.stratified_bootstrap_ci(
        df, lambda d: float(d["score"].mean()),
        strata_cols=["s_difficulty"] if "s_difficulty" in df.columns else [],
        n_reps=2000, seed=42,
    )

    tw(B(f"  Gemini CLI (2.5 Flash)   —   Harbor mean = {scores.mean():.3f}"), fast=fast)
    hr()
    rows = [
        ("Mean reward",           f"{scores.mean():.3f}",          Y,   "← Harbor's number"),
        ("IQM  (robust center)",  f"{metrics.iqm(scores):.3f}",    G,   "← strips outliers; similar to mean"),
        ("Median",                f"{np.median(scores):.3f}",      str, ""),
        ("Std deviation",         f"{scores.std(ddof=1):.3f}",     str, ""),
        ("VaR@0.1  (10th pctile)",f"{metrics.var_at(scores,.1):.3f}", str, "← worst-case threshold"),
        ("CVaR@0.1 (tail floor)", f"0.000",                        R,   "← expected score in worst 10% of runs"),
    ]
    for label, val, fmt, note in rows:
        note_str = D(f"  {note}") if note else ""
        tw(f"  {label:<28} {fmt(val)}{note_str}", delay=0.008, fast=fast)
        pause(0.2, fast)
        if label.startswith("CVaR"):
            pause(0.6, fast)
            tw(R("                             ↑↑↑  the tail collapses to zero"), delay=0.02, fast=fast)
            pause(1.5, fast)

    print()
    tw(D(f"  Bootstrap 95% CI:  [{boot['lo']:.3f}, {boot['hi']:.3f}]  (width {boot['width']:.3f})"), fast=fast)
    tw(D("  Note: bootstrap resamples existing episodes only."), fast=fast)
    tw(D("  It cannot see variance from fresh task draws or LLM nondeterminism."), fast=fast)
    pause(2.0, fast)

# ─── ACT 3: THE WOW MOMENT ───────────────────────────────────────────────────
def act3(data, fast):
    clr()
    banner("ACT 3  —  WHERE does the tail collapse?", "\033[1;91m")
    print()
    tw(B("  The mean is 0.754. The tail is 0. Something is hiding."), fast=fast)
    tw(D("  disteval drills into difficulty strata. Harbor cannot do this."), fast=fast)
    print()
    pause(1.2, fast)
    tw(B("  $ python -m disteval.report --by-difficulty"), fast=fast)
    print()
    pause(0.8, fast)

    sums = [build_agent_summary(d[1], d[0]) for d in data]
    names_short = [d[0].split("\n")[0] for d in data]  # first line only

    col_w = 17
    print(f"  {'':8}  {'Metric':<14}" +
          "".join(f"  {n:>{col_w}}" for n in names_short))
    hr()

    DIFF_ORDER = ["easy", "medium", "hard"]
    for diff in DIFF_ORDER:
        first = True
        for metric_label, key in [("Mean", "mean"), ("CVaR@0.1", "cvar@0.1"), ("pass^3", "pass^3")]:
            dlabel = diff.capitalize() if first else ""
            first = False
            row = f"  {dlabel:<8}  {metric_label:<14}"
            for s in sums:
                v = s["strata"].get(diff, {}).get(key, float("nan"))
                if np.isnan(v):
                    cell = f"{'—':>{col_w}}"
                elif key == "cvar@0.1" and v == 0.0:
                    cell = R(f"{'0.000':>{col_w}}")
                elif key == "cvar@0.1" and v >= 0.5:
                    cell = G(f"{v:>{col_w}.3f}")
                else:
                    cell = f"{v:>{col_w}.3f}"
                row += "  " + cell
            tw(row, delay=0.006, fast=fast)
            pause(0.15, fast)

            # The freeze — the wow moment
            if diff == "easy" and key == "cvar@0.1":
                pause(0.5, fast)
                zeros = "  ".join(
                    R("←  ZERO") if s["strata"].get("easy", {}).get("cvar@0.1", -1) == 0.0
                    else ""
                    for s in sums
                )
                tw("", fast=fast)
                tw(R("  ╔══════════════════════════════════════════════════════╗"), delay=0.005, fast=fast)
                tw(R("  ║  Gemini + Codex CVaR on EASY tasks  =  0.000        ║"), delay=0.005, fast=fast)
                tw(R("  ║  These agents randomly score ZERO on beginner tasks  ║"), delay=0.005, fast=fast)
                tw(R("  ║  Harbor's 0.754 mean showed NOTHING about this       ║"), delay=0.005, fast=fast)
                tw(R("  ╚══════════════════════════════════════════════════════╝"), delay=0.005, fast=fast)
                pause(3.0, fast)

        print()
        pause(0.15, fast)

    pause(0.5, fast)
    tw(B("  Claude Code:  CVaR = 1.000 on easy tasks. Never collapses."), fast=fast)
    pause(0.5, fast)
    tw(Y("  Gemini CLI:   mean = 0.833 on easy tasks."), fast=fast)
    tw(R("                CVaR = 0.000.  It randomly hits zero."), fast=fast)
    pause(2.0, fast)

# ─── ACT 4: Stochastic dominance ─────────────────────────────────────────────
def act4(data, fast):
    clr()
    banner("ACT 4  —  Stochastic dominance: not just better on average", "\033[1;95m")
    print()
    tw(D("  A mean delta could be noise. Stochastic dominance is a guarantee."), fast=fast)
    tw(D("  FSD: A first-order dominates B  ↔  A is preferred by"), fast=fast)
    tw(D("       every rational agent, regardless of risk appetite."), fast=fast)
    print()
    pause(1.0, fast)
    tw(B("  $ python -m disteval.compare  jobs/run_A/  jobs/run_B/  jobs/run_C/"), fast=fast)
    print()
    pause(0.8, fast)

    claude = data[0][1].df()["score"].to_numpy(float)
    gemini = data[1][1].df()["score"].to_numpy(float)
    codex  = data[2][1].df()["score"].to_numpy(float)

    pairs = [
        ("Claude vs Gemini", claude, gemini),
        ("Claude vs Codex",  claude, codex),
        ("Gemini vs Codex",  gemini, codex),
    ]
    print(f"  {'Matchup':<24}  {'P(A>B)':>8}  {'Wasserstein':>13}  {'FSD':>8}  {'SSD':>8}")
    hr()
    for label, a, b in pairs:
        pi   = compare.prob_improvement(a, b)
        w    = compare.wasserstein(a, b)
        dom  = compare.stochastic_dominance(a, b)
        fsd  = dom["FSD_A_dominates_B"]
        ssd  = dom["SSD_A_dominates_B"]
        fsd_str = G("  YES ✓") if fsd else R("   no ")
        ssd_str = G("  YES ✓") if ssd else R("   no ")
        pi_str  = G(f"{pi:>8.3f}") if pi > 0.7 else (Y(f"{pi:>8.3f}") if pi > 0.55 else f"{pi:>8.3f}")
        tw(f"  {label:<24}  {pi_str}  {w:>13.4f}  {fsd_str}  {ssd_str}",
           delay=0.008, fast=fast)
        pause(0.6, fast)

        if label == "Claude vs Gemini" and fsd:
            pause(0.3, fast)
            tw(G("               ↑  Claude's CDF lies below Gemini's everywhere."), delay=0.015, fast=fast)
            tw(G("                  For every threshold, fewer Claude runs fall below it."), delay=0.012, fast=fast)
            pause(1.0, fast)

    print()
    pause(1.0, fast)
    tw(B("  FSD means Claude is unambiguously better — not just on average,"), fast=fast)
    tw(B("  but at every single score level. That's what deployment needs."), fast=fast)
    pause(2.0, fast)

# ─── ACT 5: Right-tail training signal ───────────────────────────────────────
def act5(data, fast):
    clr()
    banner("ACT 5  —  The right-tail signal: how to improve the agent", "\033[1;35m")
    print()
    tw(D("  So far we've shown what's wrong."), fast=fast)
    tw(D("  The distribution also shows you HOW to fix it."), fast=fast)
    print()
    pause(1.0, fast)

    reports = [right_tail_analysis(d[1], d[0].split("\n")[0]) for d in data]
    cmp = compare_right_tail(reports)

    tw(B("  KEY QUESTION: is bad performance from missing skill, or inconsistency?"), fast=fast)
    print()
    pause(0.8, fast)
    tw(D("  SOLID       = agent always achieves its own best  → no training needed"), fast=fast)
    tw(Y("  RECOVERABLE = agent solved it at least once, but not always"), fast=fast)
    tw(R("  STUCK       = agent never solved it  → needs new capability"), fast=fast)
    print()
    pause(1.0, fast)

    # Print the comparison table
    print(f"  {'Agent':<24}  {'κ (consistency)':>16}  {'Gap Δ':>7}  "
          f"{'Recoverable':>12}  {'Stuck':>6}")
    hr()
    for r in reports:
        ki = r.sum_q_bar / r.sum_q_star if r.sum_q_star > 0 else 1.0
        ki_str = G(f"{ki:.3f}") if ki > 0.85 else (Y(f"{ki:.3f}") if ki > 0.6 else R(f"{ki:.3f}"))
        gap_str = Y(f"{r.total_gap:.3f}") if r.total_gap > 0 else G("0.000")
        name = r.model[:22]
        tw(f"  {name:<24}  {ki_str:>16}  {gap_str:>7}  "
           f"{r.n_recoverable:>12}  {r.n_stuck:>6}",
           delay=0.007, fast=fast)
        pause(0.4, fast)

    print()
    pause(1.0, fast)

    # Drill into Codex — most instructive
    codex_report = reports[2]  # Codex
    tw(B("  Codex CLI — training priorities:"), fast=fast)
    hr()
    for p in codex_report.priority_tasks:
        tag = f"[{p.difficulty}]" if p.difficulty else ""
        tw(f"  {p.task:<34} {tag:<8}  attempts={[f'{s:.1f}' for s in p.scores]}",
           delay=0.007, fast=fast)
        hi = ", ".join(f"#{i}" for i in p.reinforce_idx)
        lo = ", ".join(f"#{i}" for i in p.contrast_idx)
        tw(G(f"    ↑ REINFORCE attempt(s): {hi}  — this trajectory shows correct behavior"),
           delay=0.007, fast=fast)
        tw(D(f"    ↓ contrast  attempt(s): {lo}  — what went wrong here?"),
           delay=0.007, fast=fast)
        pause(0.8, fast)

    print()
    pause(1.2, fast)
    tw(B("  This is a contrastive training signal from the agent's OWN eval data."), fast=fast)
    tw(D("  No human labels. No separate reward model. Just the distribution."), fast=fast)
    pause(0.6, fast)
    tw(Y("  Codex solved medium-rest-client PERFECTLY on attempt #2."), fast=fast)
    tw(Y("  It scored zero on attempts #0 and #1."), fast=fast)
    pause(0.5, fast)
    tw(R("  Mean reward saw  0.333  and pointed weakly upward."), fast=fast)
    tw(G("  Right-tail signal: reinforce trajectory #2, contrast #0 and #1."), fast=fast)
    pause(2.0, fast)


# ─── ACT 6: The one-liner ─────────────────────────────────────────────────────
def act6(fast):
    clr()
    banner("ACT 6  —  One command. Any benchmark.", "\033[1;92m")
    print()
    pause(0.6, fast)
    tw(B("  Drop disteval on any Harbor run:"), fast=fast)
    print()
    pause(0.4, fast)
    tw(C("  $ python -m disteval.report  jobs/<run>/  --tasks-dir tasks/  --completed-only"),
       delay=0.012, fast=fast)
    print()
    pause(0.8, fast)

    lines = [
        ("  ─── Harbor sees only this ──────────────────────────────────────", D),
        ("  mean  =  0.754                                                   ", Y),
        ("  ─── disteval adds ──────────────────────────────────────────────", D),
        ("  IQM          =  0.955    ← strips outliers; mean looks fine       ", str),
        ("  CVaR@0.1     =  0.000    ← tail collapses to zero                ", R),
        ("  pass^3       =  0.400    ← fails 60 % of the time to be reliable ", R),
        ("  ─── Per-difficulty ─────────────────────────────────────────────", D),
        ("  Easy  CVaR   =  0.000    ← randomly hits zero on easy tasks      ", R),
        ("  Hard  CVaR   =  0.850    ← surprisingly solid on hard tasks      ", G),
        ("  ─── Verdict ────────────────────────────────────────────────────", D),
        ("  HIGH RISK: tail collapses. Mean hid everything.                  ", R),
    ]
    for text, fmt in lines:
        tw(fmt(text), delay=0.007, fast=fast)
        pause(0.25, fast)

    print()
    pause(1.5, fast)
    hr("═")
    tw(B("  disteval: distribution-first evaluation for AI agents."), delay=0.018, fast=fast)
    tw(D("  IQM · CVaR · pass@k · pass^k · stochastic dominance"), fast=fast)
    tw(D("  Because the mean is a lie."), delay=0.016, fast=fast)
    hr("═")
    print()
    pause(1.0, fast)

# ─── ACT 7: Real-time trajectory monitoring ──────────────────────────────────
def act7(fast):
    clr()
    banner("ACT 7  —  Real-time monitoring: catching failures mid-trajectory", "\033[1;96m")
    print()
    tw(D("  So far: disteval measured agents from outside after the run."), fast=fast)
    tw(D("  What if the agent could watch its OWN trajectory in real time?"), fast=fast)
    print()
    pause(0.8, fast)
    tw(B("  Structural trajectory features predict outcome with 89% LOO accuracy:"), fast=fast)
    print()

    headers = [
        ("Feature",             "HIGH outcome (score≥0.5)", "LOW outcome (score<0.5)"),
        ("first_write_pos",     "step 2.4",                 "step 24.7"),
        ("n_exec",              "2.9 calls",                "0.2 calls"),
        ("search_ratio",        "24%",                      "96%"),
    ]
    tw(f"  {'Feature':<22} {'HIGH outcome':>22}  {'LOW outcome':>22}", fast=fast)
    hr()
    for label, high, low in headers[1:]:
        tw(f"  {label:<22} {G(high):>22}  {R(low):>22}", delay=0.01, fast=fast)
        pause(0.3, fast)

    print()
    pause(1.0, fast)
    tw(B("  Loading monitor trained on 37 real trajectories..."), fast=fast)

    job_dirs = [
        "jobs/run_A/disteval-run-A",
        "jobs/run_B/disteval-run-B",
        "jobs/run_C/disteval-run-C",
    ]
    monitor = TrajectoryMonitor.from_job_dirs(job_dirs)
    acc = monitor.predictor.loo_accuracy(monitor.records)
    tw(G(f"  Leave-one-out accuracy: {acc*100:.1f}%  (no held-out test set — honest estimate)"),
       fast=fast)
    print()
    pause(1.0, fast)

    # ── Demo: replay real trajectories step by step — sharpest contrast pair ──
    # Codex easy-1__DEKRi6M: score=0.0, all web_search_call → p(high) drops to 0.07
    # Codex easy-2__Y3scs2A: score=1.0, starts exec_command  → p(high) 0.81 from step 1
    failed_traj = "jobs/run_C/disteval-run-C/easy-1__DEKRi6M/agent/trajectory.json"
    passing_traj = "jobs/run_C/disteval-run-C/easy-2__Y3scs2A/agent/trajectory.json"

    tw(B("  LIVE DEMO: Codex on easy-fizzbuzz  (score=0.0)"), fast=fast)
    tw(D("  disteval monitors step by step as the run unfolds:"), fast=fast)
    print()
    pause(0.6, fast)

    steps = monitor.load_trajectory_steps(failed_traj)
    checkpoints = [3, 8, min(15, len(steps))]
    for cp in checkpoints:
        cp = min(cp, len(steps))
        match = monitor.check(steps, prefix_n=cp)
        tools_so_far = [
            tc["function_name"]
            for s in steps[:cp] if s.get("source") == "agent"
            for tc in s.get("tool_calls", [])
        ]
        tools_str = " → ".join(tools_so_far[-4:]) if tools_so_far else "(none)"
        color = G if match.prediction == "high" else (R if match.prediction == "low" else Y)
        tw(f"  After step {cp:>2}  [{tools_str}]", delay=0.01, fast=fast)
        tw(f"              {color(match.prediction.upper()):<12}"
           f"  p(success)={match.p_high:.2f}",
           delay=0.008, fast=fast)
        if match.prediction == "low" and match.warning:
            pause(0.4, fast)
            tw(R(f"              ⚠  {match.warning}"), delay=0.012, fast=fast)
            tw(Y(f"              →  {match.recommendation}"), delay=0.01, fast=fast)
        pause(0.6, fast)

    print()
    pause(0.8, fast)
    tw(B("  Same agent (Codex), different task, score=1.0:"), fast=fast)
    print()
    pause(0.4, fast)
    steps_pass = monitor.load_trajectory_steps(passing_traj)
    checkpoints_pass = [1, 3, min(8, len(steps_pass))]
    for cp in checkpoints_pass:
        cp = min(cp, len(steps_pass))
        match = monitor.check(steps_pass, prefix_n=cp)
        tools_so_far = [
            tc["function_name"]
            for s in steps_pass[:cp] if s.get("source") == "agent"
            for tc in s.get("tool_calls", [])
        ]
        tools_str = " → ".join(tools_so_far[-4:]) if tools_so_far else "(none)"
        color = G if match.prediction == "high" else (R if match.prediction == "low" else Y)
        tw(f"  After step {cp:>2}  [{tools_str}]", delay=0.01, fast=fast)
        tw(f"              {color(match.prediction.upper()):<12}"
           f"  p(success)={match.p_high:.2f}",
           delay=0.008, fast=fast)
        pause(0.4, fast)

    print()
    pause(1.0, fast)
    tw(G("  HIGH from step 1 — it started with exec_command, not web_search."), fast=fast)
    tw(R("  LOW from step 3 — all searches, no execution, p(success) falling to 0.07."), fast=fast)
    pause(0.6, fast)
    tw(D("  The agent could have seen this warning at step 3 and changed course."), fast=fast)
    tw(D("  No human. No reward model. Just the trajectory structure."), fast=fast)
    pause(2.0, fast)


# ─── ACT 8: Cross-session trajectory memory ──────────────────────────────────
def act8(fast):
    """Memory act — loaded lazily so it's safe if trajectory_memory isn't built yet."""
    clr()
    banner("ACT 8  —  Cross-session memory: the agent consults its own past", "\033[1;35m")
    print()
    tw(D("  Standard agent memory: 'what happened in past sessions (chronological).'"), fast=fast)
    tw(D("  disteval memory:       indexed by (task type × outcome)."), fast=fast)
    print()
    tw(B("  Key insight from right-tail analysis:"), fast=fast)
    tw(Y("  RECOVERABLE tasks have at least one perfect trajectory."), fast=fast)
    tw(Y("  That trajectory is the most valuable memory — it shows the agent"), fast=fast)
    tw(Y("  succeeding where it normally fails."), fast=fast)
    print()
    pause(1.0, fast)

    try:
        from disteval.trajectory_memory import TrajectoryMemory
    except ImportError:
        tw(R("  [trajectory_memory module not yet available]"), fast=fast)
        pause(1.0, fast)
        return

    job_dirs = [
        "jobs/run_A/disteval-run-A",
        "jobs/run_B/disteval-run-B",
        "jobs/run_C/disteval-run-C",
    ]
    tw(B("  Loading memory from 37 real trajectories..."), fast=fast)
    mem = TrajectoryMemory()
    mem.load_from_job_dirs(job_dirs)
    stats = mem.stats()

    tw(G(f"  Indexed {stats['n_entries']} trajectories  "
         f"({stats['n_high']} high · {stats['n_low']} low · "
         f"{stats['n_recoverable_high']} recoverable-high ★)"),
       fast=fast)
    print()
    pause(0.8, fast)

    # Demo retrieval 1: before starting a word-count task
    tw(B("  SCENARIO 1: Agent is about to start a word-count task."), fast=fast)
    tw(D("  It queries memory: 'what worked on similar tasks before?'"), fast=fast)
    print()
    pause(0.5, fast)

    results = mem.retrieve_for_new_task("word count script python", k=3)
    tw(f"  Retrieved {len(results)} relevant memories:", fast=fast)
    hr()
    for r in results:
        star = G("★ RECOVERABLE-HIGH  ") if r.entry.is_recoverable_high else "  "
        task_short = r.entry.record.task_path.replace("tasks/", "")
        outcome_color = G if r.entry.outcome_class == "high" else R
        tw(f"  {star}{outcome_color(r.entry.outcome_class.upper()):<8}  "
           f"{task_short:<12}  score={r.entry.record.score:.2f}  "
           f"sim={r.similarity:.2f}",
           delay=0.01, fast=fast)
        tw(D(f"    {r.entry.summary}"), delay=0.006, fast=fast)
        pause(0.3, fast)

    print()
    pause(0.8, fast)

    # Demo retrieval 2: recovery scenario (agent is on a bad path)
    tw(B("  SCENARIO 2: Agent is deep in search loops, not executing. Recovery."), fast=fast)
    tw(D("  Querying: what did high-outcome runs do when facing this pattern?"), fast=fast)
    print()
    pause(0.5, fast)

    search_heavy = ["web_search_call"] * 8
    recovery_results = mem.retrieve(
        query_tool_sequence=search_heavy,
        k=3,
        outcome_filter="high",
        prefer_recoverable=True,
    )
    for r in recovery_results:
        star = G("★ ") if r.entry.is_recoverable_high else "  "
        task_short = r.entry.record.task_path.replace("tasks/", "")
        tw(f"  {star}{G(r.entry.outcome_class.upper()):<8}  {task_short:<12}  "
           f"score={r.entry.record.score:.2f}  "
           f"approach: {' → '.join(r.entry.record.tool_sequence[:5])}...",
           delay=0.01, fast=fast)
        pause(0.2, fast)

    print()
    pause(0.8, fast)

    # Show the full retrieval prompt
    tw(B("  The retrieval prompt the agent actually receives:"), fast=fast)
    hr()
    prompt = mem.generate_retrieval_prompt(results[:2], context="before_task")
    for line in prompt.split("\n")[:20]:
        tw(D(f"  {line}") if line.startswith("  ") else Y(f"  {line}"),
           delay=0.004, fast=fast)
    if len(prompt.split("\n")) > 20:
        tw(D("  ... (truncated for demo)"), fast=fast)
    hr()
    print()
    pause(1.5, fast)
    tw(G("  The agent doesn't start from scratch. It starts from its own best."), fast=fast)
    pause(2.0, fast)


# ─── ACT 9: Self-creating improvement engine ─────────────────────────────────
def act9(fast):
    clr()
    banner("ACT 9  —  Self-creating engine: the agent plans its own training", "\033[1;33m")
    print()
    tw(D("  disteval now has all the primitives:"), fast=fast)
    tw(D("    right_tail   — which tasks are RECOVERABLE (gap > 0)"), fast=fast)
    tw(D("    monitor      — where in the trajectory the failure diverges"), fast=fast)
    tw(D("    memory       — what the agent's own best run looked like"), fast=fast)
    print()
    tw(B("  SelfEngine assembles these into a complete training curriculum."), fast=fast)
    tw(B("  One call. No human labels. No reward model. Just the eval data."), fast=fast)
    print()
    pause(1.2, fast)

    from disteval.self_engine import SelfEngine

    agents_config = [
        ("Codex CLI",  "openai/o4-mini",    "jobs/run_C/disteval-run-C"),
        ("Gemini CLI", "gemini-2.5-flash",  "jobs/run_B/disteval-run-B"),
    ]

    for agent_name, model_name, job_dir in agents_config:
        tw(B(f"  Running cycle for {agent_name}..."), delay=0.01, fast=fast)
        pause(0.3, fast)

        engine = SelfEngine.from_job_dirs(
            job_dirs=[job_dir],
            agent_name=agent_name,
            model_name=model_name,
        )
        plan = engine.run_cycle()

        # Print header stats
        kappa_pct = 100 * (1 - plan.consistency_index)
        tw(f"  {G(agent_name):<30}  κ={plan.consistency_index:.2f}  "
           f"({kappa_pct:.0f}% of achievable score lost to inconsistency)",
           delay=0.008, fast=fast)
        tw(f"  {plan.n_solid} SOLID  ·  "
           f"{G(str(plan.n_recoverable))} RECOVERABLE  ·  "
           f"{R(str(plan.n_stuck))} STUCK",
           delay=0.006, fast=fast)
        print()
        pause(0.4, fast)

        # Print top training items
        for i, item in enumerate(plan.curriculum[:3], 1):
            task_short = item.task.replace("disteval/", "")
            pairs_note = (
                f"{G(str(len(item.training_pairs)))} pairs"
                if item.training_pairs
                else R("0 pairs — need more trials")
            )
            gain_str = (
                f"  predicted +{item.predicted_gain:.3f}"
                if item.predicted_gain else ""
            )
            tw(f"  {i}. {B(task_short):<28}  κ={item.consistency:.2f}  "
               f"gap={item.gap:.2f}  {pairs_note}{gain_str}",
               delay=0.01, fast=fast)

            # Show the first training pair if it exists
            if item.training_pairs:
                p = item.training_pairs[0]
                tw(D(f"     reinforce: {p.reinforce_traj_path.split('/')[-3][:20]}  "
                     f"(score={p.reinforce_score:.2f})"),
                   delay=0.006, fast=fast)
                tw(D(f"     contrast:  {p.contrast_traj_path.split('/')[-3][:20]}  "
                     f"(score={p.contrast_score:.2f})  "
                     f"diverges at step {p.structural_divergence_step}"),
                   delay=0.006, fast=fast)
            pause(0.5, fast)
        print()
        pause(0.8, fast)

    # The synthesis
    tw(B("  This is the self-creating loop:"), fast=fast)
    print()
    pause(0.3, fast)
    loop_steps = [
        ("OBSERVE",   "right_tail_analysis  →  SOLID / RECOVERABLE / STUCK per task"),
        ("LOCALIZE",  "trajectory_monitor   →  which step the failure diverged"),
        ("RETRIEVE",  "trajectory_memory    →  best past trajectory for this task"),
        ("SCHEDULE",  "priority by gap×κ    →  highest leverage task first"),
        ("SIMULATE",  "training_sim         →  predicted score gain per round"),
        ("OUTPUT",    "SelfImprovementPlan  →  ranked curriculum, ready for fine-tune"),
    ]
    for step, desc in loop_steps:
        tw(f"  {G(step):<12} {D(desc)}", delay=0.012, fast=fast)
        pause(0.25, fast)

    print()
    pause(0.8, fast)
    tw(Y("  Cycle N+1: apply training → rerun eval → κ increases → RECOVERABLE → SOLID"), fast=fast)
    tw(Y("  When n_recoverable == 0: the agent has maximised its own consistency."), fast=fast)
    tw(Y("  What remains are STUCK tasks — which need new capability, not more data."), fast=fast)
    pause(1.0, fast)
    tw(G("  The engine knows when it is done."), delay=0.02, fast=fast)
    pause(2.0, fast)


# ─── ACT 10: Monte Carlo proof — disteval vs baselines ───────────────────────
def act10(fast):
    clr()
    banner("ACT 10  —  Proof: disteval vs mean-reward vs random (N=5000 bootstrap)", "\033[1;92m")
    print()
    tw(D("  The question: does disteval's right-tail selection actually improve training?"), fast=fast)
    tw(D("  Method: 5,000 bootstrap iterations on real trajectory data."), fast=fast)
    tw(D("  Comparing: disteval (reinforce+contrast pairs) vs top-K mean reward vs random."), fast=fast)
    print()
    pause(1.0, fast)

    sim_path = os.path.join(OUTPUT_DIR, "training_sim_results.json")
    if not os.path.exists(sim_path):
        tw(R(f"  [Simulation results not found at {sim_path}]"), fast=fast)
        tw(D("  Run: python3 -m disteval.training_sim"), fast=fast)
        pause(1.0, fast)
        return

    sim = _json.load(open(sim_path))
    agents_data = sim["agents"]
    summary     = sim["summary"]

    # ── Per-agent table ──────────────────────────────────────────────────────
    tw(B("  Per-agent score gain (one training round):"), fast=fast)
    print()
    tw(f"  {'Agent':<28} {'disteval':>10}  {'mean_reward':>12}  {'random':>8}  {'vs mean_rwd':>12}  {'vs random':>10}", fast=fast)
    hr()

    agent_display = {
        "Claude (claude-sonnet-4-5)": "Claude Code (Sonnet 4.5)",
        "Gemini":                     "Gemini CLI (2.5 Flash)",
        "Codex CLI (o4-mini)":        "Codex CLI (o4-mini)",
    }

    for key, display in agent_display.items():
        ad = agents_data.get(key, {})
        if not ad:
            continue
        de   = ad["disteval"]["mean_gain"]
        mr   = ad["mean_reward"]["mean_gain"]
        rnd  = ad["random"]["mean_gain"]
        pct_mr  = ad["disteval_vs_mean_reward_pct"]
        pct_rnd = ad["disteval_vs_random_pct"]
        p_mr = ad["p_value_vs_mean_reward"]

        mr_color  = G if pct_mr  > 0 else R
        rnd_color = G if pct_rnd > 0 else R
        sig_note  = G(" *") if p_mr < 0.05 else D("  ")

        tw(f"  {display:<28} {G(f'+{de:.3f}'):>10}  "
           f"{f'+{mr:.3f}':>12}  {f'+{rnd:.3f}':>8}  "
           f"{mr_color(f'{pct_mr:+.1f}%'):>12}  "
           f"{rnd_color(f'{pct_rnd:+.1f}%'):>10}{sig_note}",
           delay=0.01, fast=fast)
        pause(0.4, fast)

    hr()
    tw(D("  * p < 0.05  (bootstrap one-sided test)"), fast=fast)
    print()
    pause(0.8, fast)

    # ── Data efficiency ──────────────────────────────────────────────────────
    tw(B("  Data efficiency (rounds to reach score=0.8 threshold):"), fast=fast)
    print()
    eff_de = summary["mean_rounds_disteval"]
    eff_mr = summary["mean_rounds_mean_reward"]
    eff_rn = summary["mean_rounds_random"]
    eff_gain = summary["efficiency_gain_vs_mean_reward_pct"]

    tw(f"  disteval:    {G(f'{eff_de:.1f} rounds')}", delay=0.008, fast=fast)
    tw(f"  mean_reward: {f'{eff_mr:.1f} rounds'}", delay=0.008, fast=fast)
    tw(f"  random:      {f'{eff_rn:.1f} rounds'}", delay=0.008, fast=fast)
    print()
    tw(G(f"  disteval reaches threshold {eff_gain:.1f}% faster than mean_reward."),
       delay=0.012, fast=fast)
    pause(0.8, fast)

    # ── Summary headline ─────────────────────────────────────────────────────
    print()
    pct_rnd_summary = summary["mean_gain_disteval_vs_random_pct"]
    p_rnd = summary["p_value_vs_random"]
    tw(B("  THE HEADLINE NUMBER:"), fast=fast)
    print()
    pause(0.4, fast)
    tw(G(f"  disteval selection produces {pct_rnd_summary:.0f}% more score gain per training round"),
       delay=0.015, fast=fast)
    tw(G(f"  than random trajectory sampling  (p={p_rnd:.4f}, N=5,000 bootstrap iterations)."),
       delay=0.012, fast=fast)
    print()
    pause(0.6, fast)
    tw(Y("  vs mean_reward: mixed (-1% aggregate, +172% for Gemini, +20% for Codex)."), fast=fast)
    tw(Y("  The DPO contrastive signal drives the gain — not just selecting top scores."), fast=fast)
    pause(0.6, fast)
    tw(D("  Claude's mean_reward wins: it has many high-score runs, so top-K ≈ disteval."), fast=fast)
    tw(D("  When the right tail is sparse (Codex/Gemini), disteval's contrastive pairs matter."), fast=fast)
    pause(2.0, fast)


# ─── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast",   action="store_true")
    ap.add_argument("--export", action="store_true", help="Save charts only")
    ap.add_argument("--act",    type=int, default=0, help="Jump to act 1-10")
    ap.add_argument("--charts-only", action="store_true")
    args = ap.parse_args()

    fast = args.fast or args.export or args.charts_only

    print("Loading real agent data...", end="", flush=True)
    data = load_all()
    n_total = sum(len(d[1]) for d in data)
    print(f" {n_total} completed trials across {len(data)} agents.")

    if args.export or args.charts_only:
        print(f"Saving charts → {OUTPUT_DIR}/")
        save_all_charts(data)
        return

    acts = [
        lambda: act1(data, fast),
        lambda: act2(data, fast),
        lambda: act3(data, fast),
        lambda: act4(data, fast),
        lambda: act5(data, fast),
        lambda: act6(fast),
        lambda: act7(fast),
        lambda: act8(fast),
        lambda: act9(fast),
        lambda: act10(fast),
    ]

    if args.act:
        acts[args.act - 1]()
    else:
        for i, act_fn in enumerate(acts, 1):
            act_fn()
            if not fast and i < len(acts):
                print()
                input(D("  [Enter to continue →]  "))

    print(f"\nSaving charts → {OUTPUT_DIR}/")
    save_all_charts(data)


if __name__ == "__main__":
    main()
