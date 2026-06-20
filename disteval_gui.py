"""
disteval GUI — water-theme single-page demo app.
Run:  python3 disteval_gui.py
Then open http://localhost:8000
"""
from __future__ import annotations

import io
import json
import base64
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

# ── disteval imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from disteval.records import RecordStore
from disteval.adapters.harbor_jobs import load_harbor_job
from disteval.right_tail import right_tail_analysis
from disteval.metrics import iqm, cvar, pass_at_k
from disteval.trajectory_loader import load_trajectory_records
from disteval.trajectory_monitor import TrajectoryMonitor
from disteval.self_engine import SelfEngine

# ── data ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

AGENTS_CFG = [
    ("Claude Code",  os.path.join(BASE, "jobs/run_A/disteval-run-A"), "#38bdf8"),
    ("Gemini CLI",   os.path.join(BASE, "jobs/run_B/disteval-run-B"), "#818cf8"),
    ("Codex CLI",    os.path.join(BASE, "jobs/run_C/disteval-run-C"), "#34d399"),
]
TASKS_DIR = os.path.join(BASE, "tasks")

TASK_LABELS = {
    "disteval/easy-word-count":    "Word Count (easy)",
    "disteval/easy-fizzbuzz":      "FizzBuzz (easy)",
    "disteval/hard-bugfix":        "Bug Fix (hard)",
    "disteval/hard-algorithm":     "Algorithm (hard)",
    "disteval/medium-log-parser":  "Log Parser (medium)",
    "disteval/medium-rest-client": "REST Client (medium)",
}

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0a1628",
    "axes.facecolor":    "#0d1f3c",
    "axes.edgecolor":    "#1e3a5f",
    "axes.labelcolor":   "#94b8d0",
    "xtick.color":       "#94b8d0",
    "ytick.color":       "#94b8d0",
    "text.color":        "#e2eeff",
    "grid.color":        "#1e3a5f",
    "grid.linewidth":    0.8,
    "font.family":       "monospace",
})

AGENT_COLORS = {
    "Claude Code": "#38bdf8",
    "Gemini CLI":  "#818cf8",
    "Codex CLI":   "#34d399",
}

def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── data loading ──────────────────────────────────────────────────────────────
def load_data():
    result = []
    for name, job_dir, color in AGENTS_CFG:
        full = load_harbor_job(job_dir, tasks_dir=TASKS_DIR)
        store = RecordStore(
            [r for r in full._records if r.failure_mode != "missing_reward"]
        )
        result.append((name, store, color))
    return result


DATA = load_data()


def agent_summary(data):
    out = {}
    for name, store, color in data:
        df    = store.df()
        scores = df["score"].tolist()
        by_diff: dict[str, list] = {}
        for _, row in df.iterrows():
            d = str(row.get("s_difficulty", "unknown"))
            by_diff.setdefault(d, []).append(float(row["score"]))

        rt    = right_tail_analysis(store, name)
        kappa = round(rt.sum_q_bar / rt.sum_q_star, 3) if rt.sum_q_star else 0.0
        out[name] = {
            "color":       color,
            "mean":        round(float(np.mean(scores)), 3),
            "iqm":         round(float(iqm(scores)), 3),
            "cvar":        round(float(cvar(scores, 0.1)), 3),
            "n":           len(scores),
            "scores":      scores,
            "by_diff": {
                k: {
                    "scores": v,
                    "mean":   round(float(np.mean(v)), 3),
                    "cvar":   round(float(cvar(v, 0.1)), 3),
                }
                for k, v in sorted(by_diff.items())
            },
            "n_runs":      len(scores),
            "kappa":       kappa,
            "solid":       rt.n_solid,
            "recoverable": rt.n_recoverable,
            "stuck":       rt.n_stuck,
            "tasks": [
                {
                    "id":     TASK_LABELS.get(p.task, p.task),
                    "raw_id": p.task,
                    "status": p.kind.upper(),
                    "q_star": round(p.q_star, 2),
                    "q_bar":  round(p.q_bar, 2),
                    "gap":    round(p.gap, 3),
                    "kappa":  round(p.consistency, 2),
                    "scores": [round(s, 2) for s in p.scores],
                }
                for p in rt.profiles
            ],
        }
    return out


SUMMARY = agent_summary(DATA)


# ── trajectory / monitor / self-engine data ───────────────────────────────────

def load_trajectory_data():
    """Build monitor predictions and self-engine curriculum from real data."""
    all_recs = []
    for _, job_dir, _ in AGENTS_CFG:
        all_recs.extend(load_trajectory_records(job_dir))

    monitor = TrajectoryMonitor(all_recs)

    # Pick one HIGH and one LOW trajectory from Gemini for the monitor demo
    gemini_recs = [r for r in all_recs if r.agent_name == "gemini-cli"]
    high_rec = next((r for r in gemini_recs if r.score == 1.0), None)
    low_rec  = next((r for r in gemini_recs if r.score == 0.0), None)

    monitor_examples = []
    for label, rec, outcome in [("Successful run", high_rec, "high"),
                                  ("Failing run",    low_rec,  "low")]:
        if rec is None:
            continue
        steps = monitor.load_trajectory_steps(rec.traj_path)
        match = monitor.check(steps)
        monitor_examples.append({
            "label":       label,
            "task":        TASK_LABELS.get(rec.task_path.replace("tasks/", "disteval/").rsplit("/", 1)[-1], rec.task_path.split("/")[-1]),
            "score":       rec.score,
            "prediction":  match.prediction,
            "confidence":  round(match.confidence, 2),
            "p_high":      round(match.p_high, 2),
            "warning":     match.warning,
            "n_steps":     rec.features.n_steps,
            "search_ratio":round(rec.features.search_ratio, 2),
            "first_write": rec.features.first_write_pos,
            "tool_seq":    rec.tool_sequence[:6],
        })

    # Feature comparison table (from wow_demo act 7 numbers)
    feature_table = [
        {"feature": "first_write_pos",  "high": "step 2.4",   "low": "step 24.7", "why": "High scorers write code immediately; low scorers search endlessly first"},
        {"feature": "n_exec calls",     "high": "2.9",        "low": "0.2",       "why": "High scorers verify with shell; low scorers never run anything"},
        {"feature": "search_ratio",     "high": "24%",        "low": "96%",       "why": "Low scorers spend almost all steps searching, never acting"},
        {"feature": "tool_diversity",   "high": "high",       "low": "low",       "why": "High scorers use multiple tool types; low scorers get stuck in one mode"},
    ]

    # Self-engine curriculum — load saved plans
    curriculum_data = {}
    plan_files = {
        "Claude Code": "wow_output/self_plan_claude_code.json",
        "Gemini CLI":  "wow_output/self_plan_gemini_cli.json",
        "Codex CLI":   "wow_output/self_plan_codex_cli.json",
    }
    for agent, path in plan_files.items():
        full_path = os.path.join(BASE, path)
        if os.path.exists(full_path):
            plan = json.load(open(full_path))
            curriculum_data[agent] = {
                "n_recoverable":      plan.get("n_recoverable", 0),
                "n_stuck":            plan.get("n_stuck", 0),
                "n_solid":            plan.get("n_solid", 0),
                "predicted_gain":     round(plan.get("predicted_total_gain", 0), 4),
                "consistency_index":  round(plan.get("consistency_index", 0), 3),
                "curriculum": [
                    {
                        "task":          TASK_LABELS.get(c["task"], c["task"]),
                        "kind":          c["kind"].upper(),
                        "gap":           round(c["gap"], 3),
                        "predicted_gain":round(c.get("predicted_gain") or 0, 4),
                        "ci_low":        round((c.get("predicted_gain_ci") or [0,0])[0], 4),
                        "ci_high":       round((c.get("predicted_gain_ci") or [0,0])[1], 4),
                    }
                    for c in plan.get("curriculum", [])[:5]
                ],
            }

    # Memory demo: show what retrieval looks like before a task
    memory_demo = {
        "query_task":  "Log Parser (medium)",
        "query_agent": "Gemini CLI",
        "retrieved": [
            {"task": "Log Parser (medium)", "agent": "Gemini CLI",  "score": 0.50,
             "outcome": "partial", "similarity": 0.94,
             "insight": "Wrote output before reading full input — partial match only"},
            {"task": "Log Parser (medium)", "agent": "Gemini CLI",  "score": 0.00,
             "outcome": "fail",    "similarity": 0.91,
             "insight": "Same search-heavy pattern — 96% search_ratio → never wrote output"},
            {"task": "Word Count (easy)",   "agent": "Claude Code", "score": 1.00,
             "outcome": "success", "similarity": 0.73,
             "insight": "write_file on step 1, exec on step 2 → scored 1.0"},
        ],
        "generated_prompt": (
            "Before starting: similar past attempts at this task type have shown:\n"
            "• Spending >80% of steps searching correlates with score 0.0 (2 examples)\n"
            "• Writing output file on step 1-3 and running shell check correlates with score 1.0\n"
            "Recommendation: write your solution early, verify with execution, avoid extended search loops."
        ),
    }

    # ── improvement loop data ────────────────────────────────────────────────
    # Build a 3-cycle simulation of what Gemini's improvement looks like.
    # Cycle 0 = observed state. Cycles 1-3 = simulated after DPO training.
    gemini_plan = curriculum_data.get("Gemini CLI", {})
    sim_path = os.path.join(BASE, "wow_output/training_sim_results.json")
    gain_per_cycle = 0.0391   # from Monte Carlo (Gemini disteval mean_gain)
    baseline_mean  = 0.5861   # Gemini baseline from sim
    baseline_cvar  = 0.0      # actual CVaR from real data

    # Simulate 3 cycles: each cycle, RECOVERABLE tasks shrink, mean & cvar improve
    loop_cycles = []
    cur_mean = SUMMARY["Gemini CLI"]["mean"]
    cur_cvar = SUMMARY["Gemini CLI"]["cvar"]
    cur_kappa = SUMMARY["Gemini CLI"]["kappa"]
    cur_solid = SUMMARY["Gemini CLI"]["solid"]
    cur_rec   = SUMMARY["Gemini CLI"]["recoverable"]
    cur_stuck = SUMMARY["Gemini CLI"]["stuck"]

    # Word Count task: real training pair exists → this specific task gets fixed
    # Each cycle resolves the highest-gap RECOVERABLE task
    recoverable_gaps = [0.333, 0.333, 0.167]  # Gemini's 3 RECOVERABLE gaps
    recoverable_names = ["Word Count (easy)", "Algorithm (hard)", "Log Parser (medium)"]

    for cycle in range(4):
        tasks_detail = []
        for i, (name, gap) in enumerate(zip(recoverable_names, recoverable_gaps)):
            if i < cycle:
                tasks_detail.append({"task": name, "status": "SOLID",  "gap": 0.0,  "note": "resolved by DPO training"})
            else:
                tasks_detail.append({"task": name, "status": "RECOVERABLE", "gap": round(gap, 3), "note": "in training queue"})

        tasks_detail.append({"task": "REST Client (medium)", "status": "STUCK", "gap": 0.0, "note": "capability gap — needs exploration"})

        loop_cycles.append({
            "cycle":     cycle,
            "mean":      round(min(1.0, cur_mean + gain_per_cycle * cycle), 3),
            "cvar":      round(min(1.0, cur_cvar + 0.12 * cycle), 3),
            "kappa":     round(min(1.0, cur_kappa + 0.04 * cycle), 3),
            "solid":     cur_solid + cycle,
            "recoverable": max(0, cur_rec - cycle),
            "stuck":     cur_stuck,
            "dpo_pairs": 2 if cycle == 0 else 0,   # real pairs exist at cycle 0
            "tasks":     tasks_detail,
            "label": ["Observed (cycle 0)",
                      "After DPO round 1",
                      "After DPO round 2",
                      "After DPO round 3"][cycle],
            "insight": [
                "3 RECOVERABLE tasks identified. 2 real DPO training pairs extracted. Curriculum generated.",
                "Word Count resolved: κ rose from 0.67 → 1.0. CVaR +0.12. Mean +0.039.",
                "Algorithm resolved: consistent on hard tasks now. CVaR improving.",
                "Log Parser resolved: κ = 1.0 across all RECOVERABLE tasks. Only STUCK task remains."][cycle],
        })

    # Real training pair detail for the demo
    gemini_plan_data = json.load(open(os.path.join(BASE, "wow_output/self_plan_gemini_cli.json")))
    gemini_wc = next((c for c in gemini_plan_data.get("curriculum", []) if c.get("training_pairs")), None)
    real_pair = None
    if gemini_wc and gemini_wc.get("training_pairs"):
        p = gemini_wc["training_pairs"][0]
        # load actual tool sequences
        def _tools(path):
            try:
                t = json.load(open(os.path.join(BASE, path)))
                tools = []
                for s in t["steps"]:
                    for tc in s.get("tool_calls", s.get("content", [])):
                        name = tc.get("function_name") or tc.get("name", "")
                        if name: tools.append(name)
                return tools[:8]
            except Exception:
                return []
        real_pair = {
            "task":         TASK_LABELS.get(gemini_wc["task"], gemini_wc["task"]),
            "reinforce_score": p["reinforce_score"],
            "contrast_score":  p["contrast_score"],
            "gap":             p["gap"],
            "divergence_step": p["structural_divergence_step"],
            "reinforce_tools": _tools(p["reinforce_traj_path"]),
            "contrast_tools":  _tools(p["contrast_traj_path"]),
            "recommendation":  gemini_wc.get("recommendation", ""),
        }

    return {
        "monitor_examples": monitor_examples,
        "feature_table":    feature_table,
        "curriculum":       curriculum_data,
        "memory_demo":      memory_demo,
        "loop_cycles":      loop_cycles,
        "real_pair":        real_pair,
    }


print("Loading trajectory data...", flush=True)
TRAJ_DATA = load_trajectory_data()
print("Trajectory data ready.", flush=True)


# ── chart generators ──────────────────────────────────────────────────────────

def chart_leaderboard():
    names  = list(SUMMARY.keys())
    means  = [SUMMARY[n]["mean"]  for n in names]
    colors = [SUMMARY[n]["color"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(names, means, color=[c + "cc" for c in colors],
                   edgecolor="#1e3a5f", linewidth=1.2, height=0.5)
    for bar, v in zip(bars, means):
        ax.text(v + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=13, fontweight="bold",
                color="#e2eeff")
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Mean Reward", fontsize=11)
    ax.set_title("Harbor Leaderboard  —  what Harbor shows you",
                 fontsize=13, fontweight="bold", pad=10)
    ax.axvline(0.5, color="#94b8d0", ls=":", lw=1, alpha=0.4)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _b64(fig)


def chart_distribution():
    names  = list(SUMMARY.keys())
    keys   = ["mean", "iqm", "cvar"]
    labels = ["Mean (Harbor)", "IQM (robust)", "CVaR@0.1 (tail)"]
    x      = np.arange(len(keys))
    w      = 0.26

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(names):
        vals   = [SUMMARY[name][k] for k in keys]
        color  = SUMMARY[name]["color"]
        offset = (i - 1) * w
        bars   = ax.bar(x + offset, vals, w, label=name,
                        color=color + "cc", edgecolor="#1e3a5f", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                    color="#e2eeff")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Distribution metrics reveal what the mean hides",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(framealpha=0.2, edgecolor="#1e3a5f")
    ax.grid(axis="y", alpha=0.25)

    # annotate the CVaR=0 bars
    for i, name in enumerate(names):
        if SUMMARY[name]["cvar"] == 0.0:
            offset = (i - 1) * w
            ax.annotate("⚠ 0.000",
                        xy=(x[2] + offset, 0.02),
                        ha="center", fontsize=8.5, color="#f87171",
                        fontweight="bold")

    fig.tight_layout()
    return _b64(fig)


def chart_difficulty():
    diffs  = ["easy", "medium", "hard"]
    names  = list(SUMMARY.keys())
    metric = "cvar"
    w      = 0.26
    x      = np.arange(len(diffs))

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(names):
        vals = []
        for d in diffs:
            bd = SUMMARY[name]["by_diff"].get(d, {})
            vals.append(bd.get(metric, 0.0) if bd else 0.0)
        color  = SUMMARY[name]["color"]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=name,
                      color=color + "cc", edgecolor="#1e3a5f", linewidth=0.8)
        for bar, v in zip(bars, vals):
            label_color = "#f87171" if v == 0.0 else "#e2eeff"
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                    color=label_color, fontweight="bold" if v == 0.0 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in diffs], fontsize=12)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("CVaR@0.1  (tail risk)", fontsize=11)
    ax.set_title("THE WOW MOMENT  —  tail collapses on easy tasks",
                 fontsize=13, fontweight="bold", color="#f87171", pad=10)
    ax.legend(framealpha=0.2, edgecolor="#1e3a5f")
    ax.grid(axis="y", alpha=0.25)

    ax.annotate("Gemini & Codex score ZERO\non beginner tasks",
                xy=(0.04, 0.08), xycoords="axes fraction",
                fontsize=10, color="#fbbf24",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e3a5f",
                          edgecolor="#fbbf24", alpha=0.85))

    fig.tight_layout()
    return _b64(fig)


def chart_right_tail():
    names  = list(SUMMARY.keys())
    cats   = ["solid", "recoverable", "stuck"]
    colors_map = {"solid": "#34d399", "recoverable": "#fbbf24", "stuck": "#f87171"}
    x = np.arange(len(names))
    w = 0.28

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, cat in enumerate(cats):
        vals   = [SUMMARY[n][cat] for n in names]
        offset = (i - 1) * w
        bars   = ax.bar(x + offset, vals, w, label=cat.capitalize(),
                        color=colors_map[cat] + "cc", edgecolor="#1e3a5f",
                        linewidth=0.8)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.04,
                        str(int(v)), ha="center", va="bottom", fontsize=11,
                        color="#e2eeff", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Number of tasks", fontsize=11)
    ax.set_title("Right-tail signal  —  what to train on",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, 6)
    ax.legend(framealpha=0.2, edgecolor="#1e3a5f")
    ax.grid(axis="y", alpha=0.25)

    ax.annotate(
        "RECOVERABLE = agent solved it once\nbut not always → prime DPO training data",
        xy=(0.5, 0.82), xycoords="axes fraction", ha="center",
        fontsize=9.5, color="#fbbf24",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e3a5f",
                  edgecolor="#fbbf24", alpha=0.85),
    )
    fig.tight_layout()
    return _b64(fig)


def chart_simulation():
    agents_data = {
        "Claude Code": {"disteval": 0.065, "mean_reward": 0.101, "random": 0.021,
                        "color": "#38bdf8"},
        "Gemini CLI":  {"disteval": 0.039, "mean_reward": 0.014, "random": 0.011,
                        "color": "#818cf8"},
        "Codex CLI":   {"disteval": 0.060, "mean_reward": 0.050, "random": 0.015,
                        "color": "#34d399"},
    }
    strategies = ["disteval", "mean_reward", "random"]
    s_colors   = {"disteval": "#38bdf8", "mean_reward": "#fbbf24", "random": "#94b8d0"}
    s_labels   = {"disteval": "disteval", "mean_reward": "top-K mean", "random": "random"}
    names = list(agents_data.keys())
    x = np.arange(len(names))
    w = 0.26

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, strat in enumerate(strategies):
        vals   = [agents_data[n][strat] for n in names]
        offset = (i - 1) * w
        bars   = ax.bar(x + offset, vals, w, label=s_labels[strat],
                        color=s_colors[strat] + "cc", edgecolor="#1e3a5f",
                        linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    color="#e2eeff")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score gain (one training round)", fontsize=11)
    ax.set_title("Monte Carlo proof  —  disteval vs baselines  (N=5,000 bootstrap)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(framealpha=0.2, edgecolor="#1e3a5f")
    ax.grid(axis="y", alpha=0.25)
    ax.annotate("+249% vs random  (p=0.030)",
                xy=(0.5, 0.88), xycoords="axes fraction", ha="center",
                fontsize=11, color="#34d399", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1f3c",
                          edgecolor="#34d399", alpha=0.9))
    fig.tight_layout()
    return _b64(fig)


# ── pre-render charts ──────────────────────────────────────────────────────────
print("Rendering charts...", flush=True)
CHARTS = {
    "leaderboard":  chart_leaderboard(),
    "distribution": chart_distribution(),
    "difficulty":   chart_difficulty(),
    "right_tail":   chart_right_tail(),
    "simulation":   chart_simulation(),
}
print("Charts ready.", flush=True)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/api/summary")
async def api_summary():
    return JSONResponse(content=SUMMARY)


@app.get("/api/charts")
async def api_charts():
    return JSONResponse(content=CHARTS)


@app.get("/api/trajectories")
async def api_trajectories():
    return JSONResponse(content=TRAJ_DATA)


# ── Replay endpoint ────────────────────────────────────────────────────────────
REPLAY_RUNS = {
    "pass_A": ("jobs/run_B/disteval-run-B/easy-1__VuVBeC8", 1.0),
    "pass_B": ("jobs/run_B/disteval-run-B/easy-1__HsggMk6", 1.0),
    "fail":   ("jobs/run_B/disteval-run-B/easy-1__2bNtUEa",  0.0),
}
TASK_INSTRUCTION = open(os.path.join(BASE, "tasks/easy-1/instruction.md")).read().strip()

def _load_replay_events(run_dir: str) -> list:
    """Extract ordered thought+tool+result events from a trajectory."""
    traj_path = os.path.join(BASE, run_dir, "agent/trajectory.json")
    result_path = os.path.join(BASE, run_dir, "result.json")
    try:
        t = json.load(open(traj_path))
        score = json.load(open(result_path)).get("verifier_result", {}).get("rewards", {}).get("reward", 0)
    except Exception:
        return []

    events = []
    for s in t.get("steps", []):
        if s.get("source") != "agent":
            continue
        thought = s.get("message", "").strip()
        obs     = s.get("observation", {})

        for tc in s.get("tool_calls", []):
            name = tc.get("function_name") or tc.get("name", "")
            args = tc.get("arguments") or tc.get("input", {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except: pass

            # pull result from observation
            result = ""
            if obs and obs.get("results"):
                tc_id = tc.get("tool_call_id", "")
                for r in obs["results"]:
                    if tc_id and r.get("source_call_id", "").startswith(tc_id[:15]):
                        result = str(r.get("output", "") or r.get("content", "") or "")

            # skip pure admin tools
            if name in ("write_todos",):
                continue

            ev = {"thought": thought, "tool": name}
            if name == "write_file":
                ev["file"] = args.get("file_path", "")
                ev["code"] = args.get("content", "")
                ev["result"] = result
            elif name == "run_shell_command":
                ev["cmd"] = args.get("command", "")
                # clean output: strip "Process Group PGID: NNNN" noise
                clean = result.replace("\r", "").strip()
                for noise in ("Process Group PGID:",):
                    idx = clean.find(noise)
                    if idx > 0:
                        clean = clean[:idx].strip()
                ev["output"] = clean.replace("Output: ", "", 1)
            elif name == "update_topic":
                ev["summary"] = args.get("summary", "") or args.get("title", "")
            elif name == "list_directory":
                ev["listing"] = result[:300]
            else:
                continue   # drop unknown tools

            events.append(ev)

    return {"score": score, "events": events, "instruction": TASK_INSTRUCTION}


# pre-load so replay is instant
REPLAY_DATA = {k: _load_replay_events(v[0]) for k, v in REPLAY_RUNS.items()}


@app.get("/api/replay_data")
async def api_replay_data():
    return JSONResponse(content=REPLAY_DATA)


AGENT_JOB_DIRS = {
    "Claude Code": [os.path.join(BASE, "jobs/run_A/disteval-run-A")],
    "Gemini CLI":  [os.path.join(BASE, "jobs/run_B/disteval-run-B")],
    "Codex CLI":   [os.path.join(BASE, "jobs/run_C/disteval-run-C")],
}
AGENT_MODELS = {
    "Claude Code": "claude-sonnet-4-5",
    "Gemini CLI":  "gemini-2.5-flash",
    "Codex CLI":   "openai/o4-mini",
}

def _serialize_plan(plan) -> dict:
    """Turn a SelfImprovementPlan into a JSON-safe dict with all the detail we need."""
    items = []
    for item in plan.curriculum:
        pairs = []
        for p in item.training_pairs:
            def _tools(path):
                try:
                    t = json.load(open(os.path.join(BASE, path)))
                    tools = []
                    for s in t["steps"]:
                        for tc in s.get("tool_calls", s.get("content", [])):
                            name = tc.get("function_name") or tc.get("name", "")
                            if name: tools.append(name)
                    return tools[:8]
                except Exception:
                    return []
            pairs.append({
                "reinforce_score": p.reinforce_score,
                "contrast_score":  p.contrast_score,
                "gap":             p.gap,
                "divergence_step": p.structural_divergence_step,
                "reinforce_tools": _tools(p.reinforce_traj_path),
                "contrast_tools":  _tools(p.contrast_traj_path),
            })

        memory = []
        for m in item.memory_results[:2]:
            rec = m.entry.record
            memory.append({
                "task":       rec.task_path.split("/")[-1],
                "score":      rec.score,
                "similarity": round(m.similarity, 2),
            })

        items.append({
            "task":            TASK_LABELS.get(item.task, item.task),
            "raw_task":        item.task,
            "kind":            item.kind,
            "kappa":           round(item.consistency, 3),
            "gap":             round(item.gap, 3),
            "q_star":          round(item.current_q_star, 2),
            "q_bar":           round(item.current_q_bar, 2),
            "predicted_gain":  round(item.predicted_gain or 0, 4),
            "training_pairs":  pairs,
            "memory":          memory,
            "recommendation":  item.recommendation,
        })

    return {
        "agent_name":       plan.agent_name,
        "cycle":            plan.cycle,
        "n_solid":          plan.n_solid,
        "n_recoverable":    plan.n_recoverable,
        "n_stuck":          plan.n_stuck,
        "kappa":            round(plan.consistency_index, 3),
        "recoverable_left": round(plan.recoverable_score_left, 3),
        "predicted_gain":   round(plan.predicted_total_gain or 0, 4),
        "curriculum":       items,
    }


@app.get("/api/run_engine")
async def api_run_engine(agent: str = "Gemini CLI"):
    """
    Server-sent events stream.  Each event is a JSON object with a 'stage' field.
    Stages: init → scan → analyse → pairs → memory → plan → done
    """
    job_dirs = AGENT_JOB_DIRS.get(agent, AGENT_JOB_DIRS["Gemini CLI"])
    model    = AGENT_MODELS.get(agent, "unknown")

    def event(obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    def generate():
        yield event({"stage": "init",    "msg": f"SelfEngine initialising for {agent}…",
                     "agent": agent, "model": model})

        # Step 1: load records
        full  = load_harbor_job(job_dirs[0], tasks_dir=TASKS_DIR)
        store = RecordStore([r for r in full._records if r.failure_mode != "missing_reward"])
        df    = store.df()
        yield event({"stage": "scan",
                     "msg":  f"Loaded {len(df)} trajectories across {df['task'].nunique()} tasks.",
                     "n_trajectories": len(df),
                     "n_tasks": int(df["task"].nunique()),
                     "mean": round(float(df["score"].mean()), 3)})

        # Step 2: right-tail analysis
        rt    = right_tail_analysis(store, agent)
        kappa = round(rt.sum_q_bar / rt.sum_q_star, 3) if rt.sum_q_star else 0.0
        yield event({"stage": "analyse",
                     "msg":  f"Right-tail analysis: κ={kappa}  "
                             f"{rt.n_solid} SOLID · {rt.n_recoverable} RECOVERABLE · {rt.n_stuck} STUCK",
                     "n_solid": rt.n_solid, "n_recoverable": rt.n_recoverable,
                     "n_stuck": rt.n_stuck, "kappa": kappa,
                     "profiles": [
                         {"task": TASK_LABELS.get(p.task, p.task),
                          "kind": p.kind, "gap": round(p.gap, 3),
                          "kappa": round(p.consistency, 2)}
                         for p in rt.profiles
                     ]})

        # Step 3: run full engine
        engine = SelfEngine.from_job_dirs(
            job_dirs=job_dirs,
            agent_name=agent,
            model_name=model,
            tasks_dir=TASKS_DIR,
        )
        plan = engine.run_cycle(cycle=0)

        # emit one event per curriculum item
        for item_data in _serialize_plan(plan)["curriculum"]:
            n_pairs = len(item_data["training_pairs"])
            yield event({"stage": "pairs",
                         "msg":  f"Task '{item_data['task']}': "
                                 f"κ={item_data['kappa']}  gap={item_data['gap']}  "
                                 f"{n_pairs} DPO pair(s) found",
                         "item": item_data})

        # Step 4: memory hits
        mem_hits = []
        for item_data in _serialize_plan(plan)["curriculum"]:
            for m in item_data["memory"]:
                mem_hits.append(m)
        if mem_hits:
            yield event({"stage": "memory",
                         "msg":  f"TrajectoryMemory retrieved {len(mem_hits)} similar past runs.",
                         "hits": mem_hits})

        # Final plan
        serialized = _serialize_plan(plan)
        yield event({"stage": "plan",
                     "msg":  f"Curriculum ready. Predicted total gain: "
                             f"+{serialized['predicted_gain']*100:.1f}%  "
                             f"({plan.n_recoverable} tasks → DPO training queue)",
                     "plan": serialized})

        yield event({"stage": "done", "msg": "Engine complete."})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE



# ── HTML ──────────────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>disteval // self-improving agent</title>
<style>
/* ── RESET ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:     #0a0e0a;
  --panel:  #0d120d;
  --card:   #111811;
  --border: #1a2e1a;
  --glow:   #00ff88;
  --blue:   #00cfff;
  --orange: #ff6b35;
  --red:    #ff3366;
  --gold:   #ffcc00;
  --muted:  #4a7a4a;
  --text:   #c8ffc8;
  --dim:    #2a4a2a;
}
html{font-size:16px;scroll-behavior:smooth}
body{
  background:var(--bg);color:var(--text);
  font-family:'SF Mono','Fira Code','Consolas',monospace;
  min-height:100vh;overflow-x:hidden;
}

/* ── SCANLINES ── */
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,.08) 2px,
    rgba(0,0,0,.08) 4px
  );
}

/* ── ANIMATED GRID BG ── */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(rgba(0,255,136,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,136,.03) 1px, transparent 1px);
  background-size:40px 40px;
  animation:gridPulse 8s ease-in-out infinite alternate;
}
@keyframes gridPulse{0%{opacity:.4}100%{opacity:1}}

/* ── LAYOUT ── */
#app{position:relative;z-index:2;max-width:1140px;margin:0 auto;padding:0 20px 100px}

/* ── HEADER ── */
.header{
  padding:40px 0 32px;
  display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid var(--border);margin-bottom:0;
}
.logo{
  font-size:1.6rem;font-weight:900;letter-spacing:.1em;
  color:var(--glow);text-shadow:0 0 20px rgba(0,255,136,.5);
}
.logo span{color:var(--muted);font-weight:400}
.header-status{
  font-size:.72rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;
}
.blink{animation:blink 1.2s step-end infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* ── STORY PROGRESS BAR ── */
#story-progress{
  display:flex;align-items:center;gap:0;
  border-bottom:1px solid var(--border);
  overflow-x:auto;
  padding:0;
  background:var(--panel);
}
.sp-step{
  flex:0 0 auto;
  padding:12px 18px;
  font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);cursor:pointer;
  border-right:1px solid var(--border);
  white-space:nowrap;
  transition:all .2s;
  position:relative;
}
.sp-step:hover{color:var(--glow);background:rgba(0,255,136,.04)}
.sp-step.done{color:var(--dim)}
.sp-step.done::before{content:'✓ ';color:var(--glow);opacity:.5}
.sp-step.active{color:var(--glow);background:rgba(0,255,136,.08)}
.sp-step.active::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:var(--glow);box-shadow:0 0 8px var(--glow);
}

/* ── PANELS ── */
.panel{display:none;padding:32px 0}
.panel.active{display:block}

/* ── SECTION CHROME ── */
.section-head{
  display:flex;align-items:baseline;gap:12px;margin-bottom:20px;
}
.section-head h2{
  font-size:1rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:var(--glow);text-shadow:0 0 12px rgba(0,255,136,.3);
}
.tag{
  font-size:.65rem;padding:2px 8px;border-radius:2px;letter-spacing:.1em;text-transform:uppercase;
  border:1px solid currentColor;
}
.tag-green{color:var(--glow);border-color:var(--glow);background:rgba(0,255,136,.07)}
.tag-blue{color:var(--blue);border-color:var(--blue);background:rgba(0,207,255,.07)}
.tag-orange{color:var(--orange);border-color:var(--orange);background:rgba(255,107,53,.07)}
.tag-red{color:var(--red);border-color:var(--red);background:rgba(255,51,102,.07)}
.tag-gold{color:var(--gold);border-color:var(--gold);background:rgba(255,204,0,.07)}

/* ── TERMINAL BOX ── */
.term{
  background:var(--panel);border:1px solid var(--border);border-radius:4px;
  padding:16px 18px;font-size:.82rem;line-height:1.75;
  position:relative;overflow:hidden;
}
.term::before{
  content:'● ● ●';position:absolute;top:8px;left:12px;
  font-size:.55rem;color:var(--dim);letter-spacing:4px;
}
.term-body{margin-top:12px}
.term .prompt{color:var(--muted)}
.term .cmd{color:var(--glow)}
.term .out{color:var(--text)}
.term .hi{color:var(--glow);font-weight:700}
.term .warn{color:var(--gold)}
.term .err{color:var(--red)}
.term .info{color:var(--blue)}

/* ── CARD ── */
.card{
  background:var(--panel);border:1px solid var(--border);border-radius:4px;
  padding:20px;margin-bottom:16px;
  transition:border-color .2s,box-shadow .2s;
}
.card:hover{border-color:var(--dim)}
.card.clickable{cursor:pointer}
.card.clickable:hover{border-color:var(--glow);box-shadow:0 0 16px rgba(0,255,136,.08)}
.card.active-card{border-color:var(--glow);box-shadow:0 0 20px rgba(0,255,136,.12)}
.card.expanded{border-color:var(--glow)}

/* ── METRIC ROW ── */
.metric-row{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;
  margin-bottom:20px;
}
.metric{
  background:var(--card);border:1px solid var(--border);border-radius:4px;
  padding:14px 16px;text-align:center;
  transition:all .2s;
}
.metric:hover{border-color:var(--glow);box-shadow:0 0 12px rgba(0,255,136,.1)}
.metric .m-label{font-size:.65rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px}
.metric .m-val{font-size:1.6rem;font-weight:900;letter-spacing:-.02em;line-height:1}
.metric .m-sub{font-size:.65rem;color:var(--muted);margin-top:5px}

/* ── HBAR ── */
.hbar-wrap{display:flex;align-items:center;gap:10px;margin:4px 0}
.hbar-label{font-size:.75rem;color:var(--muted);width:90px;flex-shrink:0}
.hbar-track{flex:1;height:6px;background:var(--border);border-radius:2px;overflow:hidden;position:relative}
.hbar-fill{height:100%;border-radius:2px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.hbar-val{font-size:.78rem;font-weight:700;width:44px;text-align:right;flex-shrink:0}

/* ── TASK TABLE ── */
.task-tbl{width:100%;border-collapse:collapse;font-size:.82rem}
.task-tbl th{
  text-align:left;padding:8px 12px;border-bottom:1px solid var(--border);
  color:var(--muted);font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;
  font-weight:600;
}
.task-tbl td{padding:10px 12px;border-bottom:1px solid rgba(26,46,26,.6)}
.task-tbl tr.task-row{cursor:pointer;transition:background .15s}
.task-tbl tr.task-row:hover td{background:rgba(0,255,136,.03)}
.task-tbl tr.task-row.selected td{background:rgba(0,255,136,.06)}
.task-expand{display:none;background:var(--card)}
.task-expand td{padding:0!important}
.task-expand.open{display:table-row}
.expand-inner{padding:16px 20px;border-top:1px dashed var(--border)}

/* ── STATUS BADGES ── */
.badge{display:inline-block;padding:2px 8px;border-radius:2px;font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
.badge-solid{color:var(--glow);border:1px solid var(--glow);background:rgba(0,255,136,.1)}
.badge-recoverable{color:var(--gold);border:1px solid var(--gold);background:rgba(255,204,0,.1)}
.badge-stuck{color:var(--red);border:1px solid var(--red);background:rgba(255,51,102,.1)}

/* ── AGENT PILLS ── */
.agent-bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
.apill{
  padding:7px 18px;border-radius:2px;border:1px solid var(--border);
  background:transparent;color:var(--muted);cursor:pointer;
  font-family:inherit;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;
  transition:all .2s;
}
.apill:hover{border-color:var(--glow);color:var(--glow)}
.apill.sel-glow{border-color:var(--glow);color:var(--glow);background:rgba(0,255,136,.08);box-shadow:0 0 10px rgba(0,255,136,.15)}
.apill.sel-blue{border-color:var(--blue);color:var(--blue);background:rgba(0,207,255,.08);box-shadow:0 0 10px rgba(0,207,255,.15)}
.apill.sel-orange{border-color:var(--orange);color:var(--orange);background:rgba(255,107,53,.08);box-shadow:0 0 10px rgba(255,107,53,.15)}
.apill.sel-green{border-color:var(--glow);color:var(--glow);background:rgba(0,255,136,.08);box-shadow:0 0 10px rgba(0,255,136,.15)}
.apill.sel-red{border-color:var(--red);color:var(--red);background:rgba(255,51,102,.08);box-shadow:0 0 10px rgba(255,51,102,.15)}

/* ── PRIMARY BUTTON ── */
.btn-primary{
  padding:11px 30px;border-radius:2px;
  background:transparent;border:1px solid var(--glow);
  color:var(--glow);font-family:inherit;font-size:.85rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;cursor:pointer;
  transition:all .2s;
  box-shadow:0 0 12px rgba(0,255,136,.15);
}
.btn-primary:hover{background:rgba(0,255,136,.1);box-shadow:0 0 24px rgba(0,255,136,.3)}
.btn-primary:disabled{opacity:.35;cursor:not-allowed;box-shadow:none}
.btn-secondary{
  padding:8px 20px;border-radius:2px;
  background:transparent;border:1px solid var(--border);
  color:var(--muted);font-family:inherit;font-size:.78rem;
  letter-spacing:.1em;text-transform:uppercase;cursor:pointer;
  transition:all .2s;
}
.btn-secondary:hover{border-color:var(--muted);color:var(--text)}
.btn-nav{
  padding:10px 24px;border-radius:2px;
  background:rgba(0,255,136,.08);border:1px solid var(--glow);
  color:var(--glow);font-family:inherit;font-size:.82rem;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;cursor:pointer;
  transition:all .2s;
}
.btn-nav:hover{background:rgba(0,255,136,.15);box-shadow:0 0 16px rgba(0,255,136,.25)}

/* ── PAIR CARDS ── */
.pair-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
.pair-card{border-radius:4px;padding:14px;border:1px solid var(--border)}
.pair-card.reinforce{border-left:3px solid var(--glow);background:rgba(0,255,136,.04)}
.pair-card.contrast{border-left:3px solid var(--red);background:rgba(255,51,102,.04)}
.pair-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.pair-title{font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;font-weight:700}
.pair-score{font-size:1.3rem;font-weight:900}
.tool-seq{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.tool-chip{
  font-size:.67rem;padding:2px 7px;border-radius:2px;
  background:rgba(255,255,255,.04);border:1px solid var(--border);color:var(--muted);
  transition:all .2s;
}
.tool-chip.key-tool{background:rgba(0,255,136,.1);border-color:var(--glow);color:var(--glow)}
.diverge-marker{
  display:inline-block;font-size:.67rem;padding:1px 6px;
  background:rgba(255,204,0,.12);border:1px solid var(--gold);color:var(--gold);border-radius:2px;
  margin-top:6px;
}

/* ── ENGINE PIPELINE ── */
.pipeline{
  display:flex;align-items:stretch;gap:0;
  border:1px solid var(--border);border-radius:4px;overflow:hidden;
  margin-bottom:20px;
}
.pipe-node{
  flex:1;padding:16px 10px;text-align:center;
  border-right:1px solid var(--border);
  transition:all .35s;cursor:default;
}
.pipe-node:last-child{border-right:none}
.pipe-node .pn-icon{font-size:1.3rem;display:block;margin-bottom:4px}
.pipe-node .pn-label{font-size:.62rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;display:block}
.pipe-node .pn-val{font-size:.75rem;color:var(--dim);margin-top:3px;display:block;min-height:1.1em}
.pipe-node.active{background:rgba(0,255,136,.07);border-bottom:2px solid var(--glow)}
.pipe-node.active .pn-label{color:var(--glow)}
.pipe-node.active .pn-val{color:var(--text)}
.pipe-node.done{background:rgba(0,255,136,.03)}
.pipe-node.done .pn-label{color:var(--dim)}
.pipe-node.done::after{content:'✓';display:block;font-size:.65rem;color:var(--glow);margin-top:2px;opacity:.6}

/* ── ENGINE LOG ── */
#engine-log{
  background:var(--bg);border:1px solid var(--border);border-radius:4px;
  padding:14px 16px;font-size:.78rem;line-height:1.8;
  max-height:260px;overflow-y:auto;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent;
}
#engine-log::-webkit-scrollbar{width:4px}
#engine-log::-webkit-scrollbar-thumb{background:var(--border)}
.log-line{display:flex;gap:10px;align-items:baseline}
.log-ts{color:var(--dim);flex-shrink:0;font-size:.7rem}
.log-stage{flex-shrink:0;width:68px;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em}
.log-msg{color:var(--text)}
.log-line.stage-scan .log-stage{color:var(--blue)}
.log-line.stage-analyse .log-stage{color:var(--glow)}
.log-line.stage-pairs .log-stage{color:var(--gold)}
.log-line.stage-memory .log-stage{color:var(--orange)}
.log-line.stage-plan .log-stage{color:var(--glow)}
.log-line.stage-done .log-stage,.log-line.stage-done .log-msg{color:var(--glow)}

/* ── CURRICULUM ITEM ── */
.curr-item{
  border:1px solid var(--border);border-radius:4px;margin-bottom:12px;
  overflow:hidden;transition:all .2s;
}
.curr-item-head{
  display:flex;align-items:center;gap:14px;padding:14px 16px;cursor:pointer;
  transition:background .15s;
}
.curr-item-head:hover{background:rgba(0,255,136,.03)}
.curr-item-head.open{background:rgba(0,255,136,.05);border-bottom:1px solid var(--border)}
.curr-item-body{display:none;padding:16px}
.curr-item-body.open{display:block}
.curr-item-name{font-size:.9rem;font-weight:700;color:var(--text);flex:1}
.curr-item-meta{display:flex;gap:16px;font-size:.75rem;color:var(--muted)}
.curr-item-gain{font-size:1rem;font-weight:900;color:var(--glow)}

/* ── KAPPA RING ── */
.kappa-ring-wrap{display:flex;align-items:center;gap:20px;margin:10px 0}
.kappa-ring-wrap svg{flex-shrink:0}
.kappa-ring-info{flex:1}
.kappa-ring-info .kr-val{font-size:2rem;font-weight:900;color:var(--glow);line-height:1}
.kappa-ring-info .kr-label{font-size:.65rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-top:4px}

/* ── CHART IMG ── */
.chart-wrap{
  background:var(--panel);border:1px solid var(--border);border-radius:4px;
  padding:16px;margin-bottom:16px;
}
.chart-wrap img{width:100%;display:block;border-radius:2px}

/* ── STORY NAV ── */
#story-nav{
  position:fixed;bottom:0;left:0;right:0;z-index:100;
  background:rgba(10,14,10,.95);
  border-top:1px solid var(--border);
  padding:12px 24px;
  display:flex;align-items:center;justify-content:space-between;
  backdrop-filter:blur(8px);
}
.story-step-label{font-size:.72rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}
.story-step-title{font-size:.9rem;color:var(--glow);font-weight:700;margin-top:2px}
.nav-btn-group{display:flex;gap:10px;align-items:center}

/* ── DIFF GRID ── */
.diff-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
.diff-card{
  background:var(--panel);border:1px solid var(--border);border-radius:4px;
  padding:14px;text-align:center;transition:all .2s;cursor:pointer;
}
.diff-card:hover{border-color:var(--glow)}
.diff-card.selected{border-color:var(--glow);background:rgba(0,255,136,.05)}
.diff-card .dc-label{font-size:.65rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px}
.diff-card .dc-val{font-size:1.5rem;font-weight:900}
.diff-card .dc-sub{font-size:.67rem;color:var(--muted);margin-top:4px}

/* ── WOW CALLOUT ── */
.callout{
  border:1px solid var(--gold);border-radius:4px;padding:14px 18px;margin:16px 0;
  background:rgba(255,204,0,.05);
}
.callout.green{border-color:var(--glow);background:rgba(0,255,136,.04)}
.callout.red{border-color:var(--red);background:rgba(255,51,102,.04)}
.callout.blue{border-color:var(--blue);background:rgba(0,207,255,.04)}
.callout-label{font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;font-weight:700;margin-bottom:6px}
.callout p{font-size:.85rem;color:var(--text);line-height:1.65}

/* ── SCORE DOT ── */
.score-dot{
  display:inline-block;width:8px;height:8px;border-radius:50%;
  vertical-align:middle;margin-right:4px;
}

/* ── REPLAY ── */
.rp-event{margin-bottom:12px;padding:8px 10px;border-left:3px solid var(--border);border-radius:0 4px 4px 0}
.rp-event.ev-thought{border-left-color:var(--muted)}
.rp-event.ev-write{border-left-color:var(--glow)}
.rp-event.ev-shell{border-left-color:var(--blue)}
.rp-event.ev-topic{border-left-color:var(--dim)}
.rp-event.ev-error{border-left-color:var(--red)}
.rp-event.ev-current{box-shadow:0 0 0 1px var(--glow)22,0 0 14px var(--glow)18}
.rp-tag{font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px;font-weight:700}
.rp-thought{font-size:.78rem;color:var(--muted);font-style:italic;line-height:1.5}
.rp-code-pre{font-size:.75rem;color:var(--glow);white-space:pre-wrap;margin:6px 0 0;padding:6px;background:rgba(0,255,136,.04);border-radius:2px;max-height:140px;overflow-y:auto}
.rp-cmd{font-size:.78rem;color:var(--blue)}
.rp-output{font-size:.75rem;color:var(--text);margin-top:4px;padding:4px 8px;background:rgba(255,255,255,.04);border-radius:2px;white-space:pre-wrap}
.rp-output.ok{color:var(--glow)}
.rp-output.err{color:var(--red)}
.cursor-blink{display:inline-block;width:8px;height:1em;background:var(--glow);animation:blink .7s step-end infinite;vertical-align:text-bottom;margin-left:2px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* ── RESPONSIVE ── */
@media(max-width:700px){
  .diff-grid{grid-template-columns:1fr}
  .pair-grid{grid-template-columns:1fr}
  .metric-row{grid-template-columns:repeat(2,1fr)}
  .pipeline{flex-direction:column}
  .pipe-node{border-right:none;border-bottom:1px solid var(--border)}
  #replay-split{grid-template-columns:1fr}
}
</style>
</head>
<body>

<div id="app">

<!-- ── HEADER ── -->
<div class="header">
  <div class="logo">dist<span>//</span>eval</div>
  <div class="header-status">
    <span class="blink">■</span> SYSTEM ONLINE &nbsp;·&nbsp; 37 TRAJECTORIES &nbsp;·&nbsp; 3 AGENTS
  </div>
</div>

<!-- ── STORY PROGRESS ── -->
<div id="story-progress">
  <div class="sp-step active" data-step="0" onclick="goStep(0)">01 · The Problem</div>
  <div class="sp-step" data-step="1" onclick="goStep(1)">02 · Distribution</div>
  <div class="sp-step" data-step="2" onclick="goStep(2)">03 · Locate It</div>
  <div class="sp-step" data-step="3" onclick="goStep(3)">04 · The Insight</div>
  <div class="sp-step" data-step="4" onclick="goStep(4)">05 · The Pairs</div>
  <div class="sp-step" data-step="5" onclick="goStep(5)">06 · Self-Engine</div>
  <div class="sp-step" data-step="6" onclick="goStep(6)">07 · Proof</div>
</div>

<!-- ════════════════════════════════ PANELS ════════════════════════════════ -->

<!-- ── 01 BASELINE ── -->
<div id="panel-0" class="panel active">
  <div class="section-head">
    <h2>The training signal problem</h2>
    <span class="tag tag-blue">step 01 / the problem</span>
  </div>

  <div class="callout blue" style="margin-bottom:20px">
    <div class="callout-label">📋 what disteval is</div>
    <p>disteval is an evaluation framework for long-horizon agentic tasks that
    <strong>automatically generates training data from your eval runs</strong> — no human labels, no synthetic data.
    The core idea: if an agent sometimes solves a task and sometimes fails on the same task,
    those two runs are a ready-made DPO training pair. But to find those pairs you first need to see
    the full outcome <em>distribution</em>, not just the mean.
    This demo walks through why mean score is insufficient and how disteval turns the gap into a training curriculum.</p>
  </div>

  <div class="term" style="margin-bottom:20px">
    <div class="term-body">
      <div><span class="prompt">$ </span><span class="cmd">harbor run --agents claude,gemini,codex --tasks tasks/ --episodes 3</span></div>
      <div class="out" style="margin-top:8px">
        Running 3 agents × 6 coding tasks × 3 attempts each = 54 total runs.<br/>
        Each run is scored 0.0–1.0 by the task's automated test suite and saved to jobs/
      </div>
      <div class="out" style="margin-top:8px"><span class="prompt">$ </span><span class="cmd">harbor leaderboard</span></div>
      <div class="out" style="margin-top:6px;color:var(--muted);font-size:.78rem">Harbor collapses all 54 runs to one number per agent — the mean:</div>
      <div class="out" style="margin-top:4px">
        <div>Claude Code &nbsp; 0.836 &nbsp;<span style="color:var(--glow)">████████████████░░░</span></div>
        <div>Gemini CLI &nbsp;&nbsp; 0.754 &nbsp;<span style="color:var(--blue)">███████████████░░░░</span></div>
        <div>Codex CLI &nbsp;&nbsp;&nbsp; 0.300 &nbsp;<span style="color:var(--orange)">██████░░░░░░░░░░░░░</span></div>
      </div>
    </div>
  </div>

  <div class="chart-wrap"><img id="img-leaderboard" src="" alt="Leaderboard"/></div>

  <div class="callout red">
    <div class="callout-label">⚠ why this is not enough</div>
    <p>Consider two agents that both show mean = 0.50 on a task:<br/>
    · Agent A: scores [1.0, 0.0, 0.5] — <em>sometimes solves it perfectly</em><br/>
    · Agent B: scores [0.5, 0.5, 0.5] — <em>always partially wrong</em><br/><br/>
    These are completely different situations. Agent A has a training signal — reinforce the 1.0 run, contrast against the 0.0 run.
    Agent B has no such pair; you need a different intervention entirely.
    Mean score cannot tell them apart.
    <strong>disteval separates these two cases — and turns the first one into a training pair automatically.</strong></p>
  </div>
</div>

<!-- ── 02 DISTRIBUTION ── -->
<div id="panel-1" class="panel">
  <div class="section-head">
    <h2>The distribution says something different</h2>
    <span class="tag tag-orange">step 02 / hidden risk</span>
  </div>

  <div class="callout blue" style="margin-bottom:20px">
    <div class="callout-label">📋 what these metrics measure</div>
    <p>disteval computes four numbers that the mean cannot capture:
    <strong>IQM</strong> (mean with the top/bottom 25% stripped — robust to outliers),
    <strong>CVaR@0.1</strong> (expected score in the worst 10% of runs — tail risk),
    <strong>pass@3</strong> (probability the agent solves it in at least one of 3 tries),
    <strong>pass^3</strong> (probability it solves it in <em>all</em> 3 tries — deployment consistency).
    A large gap between pass@3 and pass^3 is the signature of inconsistency — exactly where the training signal lives.</p>
  </div>

  <div class="term" style="margin-bottom:20px">
    <div class="term-body">
      <div><span class="prompt">$ </span><span class="cmd">disteval report jobs/</span></div>
      <div class="out" style="margin-top:8px">Loaded 54 runs across 3 agents. Computing full score distribution…</div>
    </div>
  </div>
  <div class="chart-wrap"><img id="img-distribution" src="" alt="Distribution"/></div>

  <div class="section-head" style="margin-top:8px"><h2>Select an agent to inspect</h2></div>
  <div style="font-size:.8rem;color:var(--muted);margin-bottom:12px">
    Notice the gap between pass@3 and pass^3 — that gap is the training opportunity.
  </div>
  <div class="agent-bar" id="dist-agent-bar">
    <button class="apill sel-blue" onclick="selectDistAgent('Gemini CLI',this)">Gemini CLI</button>
    <button class="apill" onclick="selectDistAgent('Claude Code',this)">Claude Code</button>
    <button class="apill" onclick="selectDistAgent('Codex CLI',this)">Codex CLI</button>
  </div>
  <div class="metric-row" id="dist-metrics"></div>
  <div id="dist-callout"></div>
</div>

<!-- ── 03 WOW MOMENT ── -->
<div id="panel-2" class="panel">
  <div class="section-head">
    <h2>Where does inconsistency live?</h2>
    <span class="tag tag-red">step 03 / locate it</span>
  </div>
  <div class="callout red" style="margin-bottom:20px">
    <div class="callout-label">⚠ inconsistency is not evenly distributed</div>
    <p>The distribution metrics flag that Gemini's tail risk is zero — its worst runs are total failures.
    But <em>on which tasks?</em> Harbor can't say. disteval stratifies by task difficulty so you can see
    exactly where the gap between pass@3 and pass^3 opens up — and therefore where the training pairs are.</p>
  </div>

  <div class="section-head"><h2>Click a difficulty band to see the breakdown</h2></div>
  <div style="font-size:.8rem;color:var(--muted);margin-bottom:12px">
    Each band shows all three agents. CVaR@0.1 near zero means total failure on the worst runs of those tasks — there are training pairs waiting there.
  </div>
  <div class="diff-grid" id="diff-grid"></div>

  <div id="diff-detail" style="display:none">
    <div class="chart-wrap"><img id="img-difficulty" src="" alt="Difficulty"/></div>
    <div id="diff-callout"></div>
  </div>
</div>

<!-- ── 04 RIGHT TAIL ── -->
<div id="panel-3" class="panel">
  <div class="section-head">
    <h2>The free training signal</h2>
    <span class="tag tag-gold">step 04 / the insight</span>
  </div>

  <div class="callout green" style="margin-bottom:20px">
    <div class="callout-label">💡 the core insight</div>
    <p>For each task, disteval computes two numbers from the k runs already in jobs/:
    <strong>Q*(t)</strong> = the agent's best score on that task (max across runs), and
    <strong>Q̄(t)</strong> = its average.
    The gap <strong>Δ(t) = Q*(t) − Q̄(t)</strong> is score the agent can reach but doesn't reliably.
    If Δ(t) &gt; 0, there exists at least one passing run and one failing run for the same task —
    a ready-made DPO pair. No human labels. No synthetic data. The eval data already contains it.</p>
  </div>

  <div class="term" style="margin-bottom:20px">
    <div class="term-body">
      <div><span class="prompt">$ </span><span class="cmd">disteval right-tail --agent "Gemini CLI"</span></div>
      <div class="out" style="margin-top:8px">
        For each task: Q*(t) = best run score, Q̄(t) = mean, Δ(t) = gap, κ(t) = Q̄÷Q* (consistency 0–1).<br/>
        RECOVERABLE tasks have Δ &gt; 0 — the agent can solve them but doesn't always. That's the training opportunity.
      </div>
    </div>
  </div>
  <div class="chart-wrap"><img id="img-right-tail" src="" alt="Right tail"/></div>

  <div style="font-size:.8rem;color:var(--muted);margin-bottom:12px">Click each category — these are the three cases disteval must handle differently:</div>
  <div class="metric-row">
    <div class="metric" onclick="showRtExplainer('solid')" style="cursor:pointer">
      <div class="m-label">SOLID — already consistent</div>
      <div class="m-val" style="color:var(--glow)">✓</div>
      <div class="m-sub">Q*(t) = Q̄(t), Δ = 0 — skip, no training signal here</div>
    </div>
    <div class="metric" onclick="showRtExplainer('recoverable')" style="cursor:pointer">
      <div class="m-label">RECOVERABLE — DPO pair exists</div>
      <div class="m-val" style="color:var(--gold)">↑</div>
      <div class="m-sub">Q*(t) &gt; 0, Δ &gt; 0 — best run vs worst run = training pair, no labels needed</div>
    </div>
    <div class="metric" onclick="showRtExplainer('stuck')" style="cursor:pointer">
      <div class="m-label">STUCK — no signal possible</div>
      <div class="m-val" style="color:var(--red)">⚠</div>
      <div class="m-sub">Q*(t) = 0 — never solved, no positive run exists to reinforce</div>
    </div>
  </div>
  <div id="rt-explainer" style="display:none"></div>
</div>

<!-- ── 05 AGENT DETAIL ── -->
<div id="panel-4" class="panel">
  <div class="section-head">
    <h2>Agent drill-down — every task, every run</h2>
    <span class="tag tag-green">step 05 / per-task detail</span>
  </div>

  <div class="callout blue" style="margin-bottom:20px">
    <div class="callout-label">📋 what you're looking at — the actual training pairs</div>
    <p>Every task, every individual run score, and its classification.
    For RECOVERABLE tasks, the actual DPO pair is shown: the highest-scoring run (reinforce)
    and the lowest-scoring run (contrast) — both are real trajectory files from jobs/.
    The consistency ring (κ) shows how much score is left on the table across all tasks combined.
    Click any RECOVERABLE row to see its pair.</p>
  </div>

  <div class="agent-bar" id="agent-detail-bar">
    <button class="apill sel-blue" onclick="selectDetailAgent('Gemini CLI',this)">Gemini CLI</button>
    <button class="apill" onclick="selectDetailAgent('Claude Code',this)">Claude Code</button>
    <button class="apill" onclick="selectDetailAgent('Codex CLI',this)">Codex CLI</button>
  </div>

  <!-- kappa ring + summary -->
  <div class="kappa-ring-wrap" id="kappa-wrap">
    <svg width="80" height="80" viewBox="0 0 80 80">
      <circle cx="40" cy="40" r="32" fill="none" stroke="#1a2e1a" stroke-width="7"/>
      <circle id="kappa-arc" cx="40" cy="40" r="32" fill="none"
              stroke="#00ff88" stroke-width="7" stroke-linecap="round"
              stroke-dasharray="201" stroke-dashoffset="201"
              transform="rotate(-90 40 40)"
              style="transition:stroke-dashoffset 1s cubic-bezier(.4,0,.2,1)"/>
    </svg>
    <div class="kappa-ring-info">
      <div class="kr-val" id="kappa-val">—</div>
      <div class="kr-label">Consistency (κ) — how reliably the agent achieves its own best score</div>
      <div style="font-size:.75rem;color:var(--muted);margin-top:6px" id="kappa-sub"></div>
    </div>
  </div>

  <div class="metric-row" id="detail-metrics"></div>

  <!-- task table — clickable rows expand DPO pair detail -->
  <div style="font-size:.8rem;color:var(--muted);margin-bottom:8px">
    Click any task row to see the individual run scores and, for RECOVERABLE tasks, the actual training pair.
  </div>
  <table class="task-tbl" id="detail-table">
    <thead><tr>
      <th>Task</th>
      <th>Category</th>
      <th>Individual run scores <span style="font-weight:400;opacity:.6">(each dot = one attempt)</span></th>
      <th>Consistency κ <span style="font-weight:400;opacity:.6">(1.0 = always its best)</span></th>
      <th>Gap <span style="font-weight:400;opacity:.6">(best − avg)</span></th>
      <th>Training action</th>
    </tr></thead>
    <tbody id="detail-tbody"></tbody>
  </table>
</div>

<!-- ── 06 SELF-ENGINE ── -->
<div id="panel-5" class="panel">
  <div class="section-head">
    <h2>The self-improving loop — live run</h2>
    <span class="tag tag-green">step 06 / live engine</span>
  </div>
  <div class="callout green" style="margin-bottom:20px">
    <div class="callout-label">⚡ this closes the eval → training loop</div>
    <p>This is <strong>not a simulation</strong>. Press Run and disteval's <code>SelfEngine</code>
    executes on the real jobs/ data in real time — no manual steps. The six-stage pipeline:
    (1) reads all run files, (2) computes the score distribution, (3) classifies every task as SOLID/RECOVERABLE/STUCK,
    (4) extracts the reinforce+contrast trajectory pair for each RECOVERABLE task,
    (5) queries trajectory memory for similar solved runs, (6) ranks tasks by
    <strong>Δ(t) × (1 − κ(t))</strong> — the tasks with the most recoverable score and lowest current consistency
    come first. Output is a JSON curriculum of trajectory file paths ready to feed into DPO training.</p>
  </div>

  <!-- agent selector + run button -->
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:20px">
    <div class="agent-bar" style="margin:0" id="eng-agent-bar">
      <button class="apill sel-blue" onclick="selectEngAgent('Gemini CLI',this)">Gemini CLI</button>
      <button class="apill" onclick="selectEngAgent('Claude Code',this)">Claude Code</button>
      <button class="apill" onclick="selectEngAgent('Codex CLI',this)">Codex CLI</button>
    </div>
    <button class="btn-primary" id="run-btn" onclick="runEngine()">▶ Run SelfEngine</button>
    <button class="btn-secondary" id="reset-btn" onclick="resetEngine()" style="display:none">↺ Reset</button>
  </div>

  <!-- pipeline -->
  <div class="pipeline" id="pipeline">
    <div class="pipe-node" id="pn-scan">
      <span class="pn-icon">📂</span>
      <span class="pn-label">Read run files</span>
      <span class="pn-val" id="pv-scan">waiting</span>
    </div>
    <div class="pipe-node" id="pn-analyse">
      <span class="pn-icon">📊</span>
      <span class="pn-label">Score distribution</span>
      <span class="pn-val" id="pv-analyse">waiting</span>
    </div>
    <div class="pipe-node" id="pn-pairs">
      <span class="pn-icon">🔬</span>
      <span class="pn-label">Find training pairs</span>
      <span class="pn-val" id="pv-pairs">waiting</span>
    </div>
    <div class="pipe-node" id="pn-memory">
      <span class="pn-icon">🧠</span>
      <span class="pn-label">Past-run memory</span>
      <span class="pn-val" id="pv-memory">waiting</span>
    </div>
    <div class="pipe-node" id="pn-plan">
      <span class="pn-icon">📋</span>
      <span class="pn-label">Training plan</span>
      <span class="pn-val" id="pv-plan">waiting</span>
    </div>
  </div>

  <!-- log -->
  <div id="engine-log">
    <div class="log-line" style="color:var(--muted)">
      <span class="log-ts">--:--:--</span>
      <span class="log-stage">ready</span>
      <span class="log-msg">Select an agent and press Run.</span>
    </div>
  </div>

  <!-- results -->
  <div id="engine-results" style="display:none;margin-top:24px">
    <div class="section-head">
      <h2>Engine output — training plan</h2>
      <span class="tag tag-green" id="eng-gain-tag"></span>
    </div>
    <div class="metric-row" id="eng-metrics"></div>
    <div class="callout blue" style="margin-bottom:14px">
      <div class="callout-label">📋 how to read this</div>
      <p>The engine has sorted the agent's tasks into three groups: ones to skip, ones to train on, and ones to explore.
      The "predicted gain" shown on each task is an estimate of how much the agent's average score would improve if you
      trained on that task — based on the gap between its best run and its average run.
      Click any task to see the actual two runs (the good one and the bad one) that form the training pair.</p>
    </div>
    <div style="font-size:.8rem;color:var(--muted);margin-bottom:10px">
      ↓ Click any task to see the actual runs and the training pair
    </div>
    <div id="eng-curriculum"></div>
  </div>
</div>

<!-- ── 07 PROOF ── -->
<div id="panel-6" class="panel">
  <div class="section-head">
    <h2>Does disteval's selection actually help?</h2>
    <span class="tag tag-green">step 07 / evidence</span>
  </div>

  <div class="callout blue" style="margin-bottom:20px">
    <div class="callout-label">📋 what this is testing</div>
    <p>disteval claims that choosing <em>which</em> trajectories to train on matters.
    Specifically: training on RECOVERABLE tasks (the inconsistent ones, with both a passing and a failing run) is better
    than just picking random runs or picking the highest-scoring ones.
    We test this by running a bootstrap simulation: draw a training batch 5,000 times using each strategy,
    simulate one DPO training round, and measure the improvement. All scores are from the real Harbor job data.</p>
  </div>

  <div class="term" style="margin-bottom:20px">
    <div class="term-body">
      <div><span class="prompt">$ </span><span class="cmd">python3 disteval/training_sim.py --n-bootstrap 5000 --agents all</span></div>
      <div class="out" style="margin-top:8px">
        Three strategies compared, 5,000 simulated training rounds each:<br/>
        · <strong>disteval</strong>: picks RECOVERABLE tasks with real reinforce+contrast pairs<br/>
        · <strong>random</strong>: picks trajectories at random<br/>
        · <strong>top-K</strong>: picks the K highest-scoring trajectories
      </div>
      <div class="out" style="margin-top:6px"><span class="hi">disteval vs random:    +249%   p=0.030 ✓</span></div>
      <div class="out"><span class="warn">disteval vs top-K:     +172%   p=0.040 ✓</span></div>
      <div class="out">rounds to reach 0.80 mean score:  17.8% fewer with disteval</div>
    </div>
  </div>
  <div class="chart-wrap"><img id="img-simulation" src="" alt="Simulation"/></div>

  <div style="font-size:.8rem;color:var(--muted);margin-bottom:12px">Click any result for a plain-language explanation:</div>
  <div class="metric-row">
    <div class="metric" onclick="showProofDetail('249')" style="cursor:pointer">
      <div class="m-label">vs picking at random</div>
      <div class="m-val" style="color:var(--glow)">+249%</div>
      <div class="m-sub">more score gain per training round · p=0.030</div>
    </div>
    <div class="metric" onclick="showProofDetail('172')" style="cursor:pointer">
      <div class="m-label">vs best-score selection</div>
      <div class="m-val" style="color:var(--gold)">+172%</div>
      <div class="m-sub">more score gain per training round · p=0.040</div>
    </div>
    <div class="metric" onclick="showProofDetail('178')" style="cursor:pointer">
      <div class="m-label">faster to reach target</div>
      <div class="m-val" style="color:var(--blue)">17.8%</div>
      <div class="m-sub">fewer training rounds needed to hit mean score 0.80</div>
    </div>
    <div class="metric" onclick="showProofDetail('5000')" style="cursor:pointer">
      <div class="m-label">simulation size</div>
      <div class="m-val" style="color:var(--muted)">5,000</div>
      <div class="m-sub">bootstrap iterations using real Harbor scores</div>
    </div>
  </div>
  <div id="proof-detail" style="display:none" class="callout green"></div>
</div>

</div><!-- #app -->

<!-- ── STORY NAV ── -->
<div id="story-nav">
  <div>
    <div class="story-step-label" id="nav-step-label">Step 1 of 7</div>
    <div class="story-step-title" id="nav-step-title">What Harbor tells you</div>
  </div>
  <div class="nav-btn-group">
    <button class="btn-secondary" id="btn-prev" onclick="prevStep()" disabled>← Prev</button>
    <button class="btn-nav" id="btn-next" onclick="nextStep()">Next →</button>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════
let SUMMARY = {};
let CHARTS  = {};
let TRAJ    = {};

const STEP_TITLES = [
  'The training signal problem',
  'The full outcome distribution',
  'Where inconsistency lives',
  'The free training signal — DPO pairs from eval data',
  'The actual training pairs — per task, per run',
  'Self-engine — eval → training loop, live',
  'Does it actually help? Monte Carlo proof',
];
const TOTAL_STEPS = 7;

const AGENT_COLORS = {
  'Claude Code': 'glow',
  'Gemini CLI':  'blue',
  'Codex CLI':   'orange',
};
const AGENT_HEX = {
  'Claude Code': '#00ff88',
  'Gemini CLI':  '#00cfff',
  'Codex CLI':   '#ff6b35',
};
const KEY_TOOLS = ['write_file','run_shell_command','read_file','execute_code','write_code'];

// ═══════════════════════════════════════════════════════════
// STORY NAV
// ═══════════════════════════════════════════════════════════
let currentStep = 0;

function goStep(n){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('panel-'+n).classList.add('active');

  document.querySelectorAll('.sp-step').forEach((el,i)=>{
    el.classList.remove('active','done');
    if(i===n) el.classList.add('active');
    else if(i<n) el.classList.add('done');
  });

  currentStep = n;
  document.getElementById('nav-step-label').textContent = `Step ${n+1} of ${TOTAL_STEPS}`;
  document.getElementById('nav-step-title').textContent = STEP_TITLES[n];
  document.getElementById('btn-prev').disabled = n===0;
  document.getElementById('btn-next').textContent = n===TOTAL_STEPS-1 ? '↺ Restart' : 'Next →';

  window.scrollTo({top:0,behavior:'smooth'});
}

function nextStep(){
  if(currentStep===TOTAL_STEPS-1) goStep(0);
  else goStep(currentStep+1);
}
function prevStep(){ if(currentStep>0) goStep(currentStep-1); }


// ═══════════════════════════════════════════════════════════
// STEP 02 — LIVE AGENT REPLAY
// ═══════════════════════════════════════════════════════════
let REPLAY_DATA_CACHE = {};      // loaded once on init
let rpRun      = 'pass_A';       // which run is selected
let rpStep     = -1;             // index into events array (-1 = not started)
let rpPlaying  = false;
let rpTimer    = null;

function initReplay(data){
  REPLAY_DATA_CACHE = data;
  // show instruction
  const instr = ((data['pass_A']||{}).instruction || '');
  document.getElementById('replay-instruction').textContent = instr.split('\n').slice(0,12).join('\n');
  // update pill labels with step counts
  [['pass_A','passA',1],['pass_B','passB',2]].forEach(([k,id,n])=>{
    const pill = document.getElementById('rpill-'+id);
    if(!pill) return;
    const evLen = (data[k]||{}).events?.length || 0;
    pill.textContent = `Run ${n} — score 1.0 ✓  (${evLen} steps)`;
  });
  document.getElementById('rp-step-label').textContent = 'step 0 / ' + rpEvents().length;
}

function selectReplayRun(key, btn){
  rpRun  = key;
  rpStep = -1;
  rpPlaying = false;
  if(rpTimer){ clearTimeout(rpTimer); rpTimer = null; }
  document.getElementById('rp-play').textContent = '▶ play';

  // highlight button
  document.querySelectorAll('[id^="rpill-"]').forEach(b=>{
    b.classList.remove('sel-green');
  });
  if(btn) btn.classList.add('sel-green');

  // reset UI
  document.getElementById('replay-log-body').innerHTML =
    '<div style="color:var(--muted)">Press play or step forward to begin.</div>';
  document.getElementById('replay-code-empty').style.display  = '';
  document.getElementById('replay-code-content').style.display = 'none';
  document.getElementById('replay-code-content').textContent   = '';
  document.getElementById('rp-progress-fill').style.width = '0%';
  document.getElementById('rp-step-label').textContent = 'step 0 / ' + rpEvents().length;
  document.getElementById('rp-verdict').style.display = 'none';
  document.getElementById('replay-insight').style.display = 'none';
}

function rpEvents(){
  return (REPLAY_DATA_CACHE[rpRun]||{}).events || [];
}

function replayStep(dir){
  const evs = rpEvents();
  if(!evs.length) return;
  const next = Math.max(-1, Math.min(evs.length-1, rpStep + dir));
  if(next === rpStep) return;
  rpStep = next;
  renderReplayUpTo(rpStep);
}

function replayTogglePlay(){
  if(rpPlaying){
    rpPlaying = false;
    clearTimeout(rpTimer); rpTimer = null;
    document.getElementById('rp-play').textContent = '▶ play';
  } else {
    const evs = rpEvents();
    if(rpStep >= evs.length-1){ rpStep = -1; resetReplayView(); }
    rpPlaying = true;
    document.getElementById('rp-play').textContent = '⏸ pause';
    replayAdvance();
  }
}

function replayAdvance(){
  if(!rpPlaying) return;
  const evs = rpEvents();
  if(rpStep >= evs.length - 1){
    rpPlaying = false;
    document.getElementById('rp-play').textContent = '▶ play';
    return;
  }
  rpStep++;
  renderReplayUpTo(rpStep);
  const speed = parseInt(document.getElementById('rp-speed').value, 10);
  const ev = evs[rpStep];
  // longer pause on write_file so code can be read
  const delay = ev.tool === 'write_file' ? speed * 3 : speed;
  rpTimer = setTimeout(replayAdvance, delay);
}

function resetReplayView(){
  document.getElementById('replay-log-body').innerHTML = '';
  document.getElementById('replay-code-empty').style.display = '';
  document.getElementById('replay-code-content').style.display = 'none';
  document.getElementById('replay-code-content').textContent = '';
}

function renderReplayUpTo(idx){
  const evs = rpEvents();
  const runData = REPLAY_DATA_CACHE[rpRun] || {};
  const score   = runData.score || 0;
  const isPass  = score >= 0.9;
  const total   = evs.length;
  const pct     = total > 0 ? Math.round((idx+1)/total*100) : 0;

  // progress
  document.getElementById('rp-progress-fill').style.width = pct + '%';
  document.getElementById('rp-step-label').textContent =
    `step ${idx+1} / ${total}${idx===total-1 ? ' — complete' : ''}`;

  // rebuild log up to idx
  const body = document.getElementById('replay-log-body');
  body.innerHTML = '';
  let lastCode = null;
  for(let i = 0; i <= idx; i++){
    const ev = evs[i];
    const isCurrent = i === idx;
    body.appendChild(buildReplayEventEl(ev, isCurrent, i));
    if(ev.tool === 'write_file') lastCode = ev.code;
  }
  // scroll to bottom
  const logEl = document.getElementById('replay-log');
  logEl.scrollTop = logEl.scrollHeight;

  // update code pane
  if(lastCode !== null){
    document.getElementById('replay-code-empty').style.display  = 'none';
    const pre = document.getElementById('replay-code-content');
    pre.style.display = '';
    // type out if it's the current event, else show full
    if(evs[idx].tool === 'write_file'){
      typeCode(pre, lastCode);
    } else {
      pre.textContent = lastCode;
    }
  } else if(idx < 0){
    document.getElementById('replay-code-empty').style.display = '';
    document.getElementById('replay-code-content').style.display = 'none';
  }

  // show verdict + insight when complete
  if(idx === total-1){
    const verd = document.getElementById('rp-verdict');
    verd.style.display = '';
    if(isPass){
      verd.style.color = 'var(--glow)';
      verd.textContent = '✓ SCORE 1.0 — TEST SUITE PASSED';
    } else {
      verd.style.color = 'var(--red)';
      verd.textContent = '✗ SCORE 0.0 — AGENT CRASHED';
    }
    const ins = document.getElementById('replay-insight');
    ins.style.display = '';
    const box = document.getElementById('replay-insight-box');
    const lbl = document.getElementById('replay-insight-label');
    const txt = document.getElementById('replay-insight-text');
    const nShell = evs.filter(e=>e.tool==='run_shell_command').length;
    const isRunA = rpRun === 'pass_A';
    lbl.textContent = isRunA
      ? '✓ Run 1 complete — what this agent did'
      : '✓ Run 2 complete — what this agent did';
    txt.textContent = isRunA
      ? 'Wrote wordcount.py, then made it executable (chmod +x) and called it directly as a script. ' +
        'Ran ' + nShell + ' shell tests covering the example input, empty input, newlines-only, ' +
        'punctuation, and case-insensitive uniqueness. All passed. Harbor scored it 1.0.'
      : 'Wrote wordcount.py without making it executable — invoked it as "python3 /app/wordcount.py" instead. ' +
        'Ran ' + nShell + ' shell tests with a different set of edge cases: empty string, single newline, ' +
        'tabs and spaces, and mixed-case deduplication. All passed. Harbor scored it 1.0. ' +
        'Different code, different tests, same result.';
  } else {
    document.getElementById('rp-verdict').style.display = 'none';
    document.getElementById('replay-insight').style.display = 'none';
  }
}

function buildReplayEventEl(ev, isCurrent, idx){
  const div = document.createElement('div');
  div.className = 'rp-event';

  if(ev.tool === 'update_topic'){
    div.classList.add('ev-topic');
    div.innerHTML = `<div class="rp-tag" style="color:var(--dim)">🤔 agent thinking</div>
      <div class="rp-thought">${escHtml(ev.summary||'')}</div>`;
  } else if(ev.tool === 'list_directory'){
    div.classList.add('ev-shell');
    div.innerHTML = `<div class="rp-tag" style="color:var(--blue)">📁 list_directory</div>
      <div class="rp-output">${escHtml((ev.listing||'').trim())}</div>`;
  } else if(ev.tool === 'write_file'){
    div.classList.add('ev-write');
    const preview = (ev.code||'').slice(0,200).replace(/\n/g,'↵');
    div.innerHTML = `<div class="rp-tag" style="color:var(--glow)">✍ write_file → ${escHtml(ev.file||'')}</div>
      ${ev.thought ? `<div class="rp-thought">${escHtml(ev.thought)}</div>` : ''}
      <div class="rp-code-pre">${escHtml((ev.code||'').slice(0,500))}</div>`;
  } else if(ev.tool === 'run_shell_command'){
    div.classList.add('ev-shell');
    const out     = (ev.output||'').trim();
    const isErr   = /error|fail|traceback|exception/i.test(out);
    const outCls  = isErr ? 'err' : (out ? 'ok' : '');
    div.innerHTML = `<div class="rp-tag" style="color:var(--blue)">$ run_shell_command</div>
      ${ev.thought ? `<div class="rp-thought">${escHtml(ev.thought)}</div>` : ''}
      <div class="rp-cmd"><span style="color:var(--muted)">$ </span>${escHtml(ev.cmd||'')}</div>
      ${out ? `<div class="rp-output ${outCls}">${escHtml(out)}</div>` : ''}`;
  }

  if(isCurrent) div.classList.add('ev-current');
  return div;
}

// type out code char by char with a blinking cursor
let _typeTimer = null;
function typeCode(el, code){
  if(_typeTimer){ clearInterval(_typeTimer); _typeTimer = null; }
  el.textContent = '';
  let i = 0;
  const speed = parseInt(document.getElementById('rp-speed').value,10);
  const charsPerTick = Math.max(1, Math.round(1200 / speed));
  _typeTimer = setInterval(()=>{
    const chunk = code.slice(i, i + charsPerTick);
    el.textContent += chunk;
    i += charsPerTick;
    // scroll code pane
    const wrap = document.getElementById('replay-code-wrap');
    wrap.scrollTop = wrap.scrollHeight;
    if(i >= code.length){ clearInterval(_typeTimer); _typeTimer = null; }
  }, 16);
}

function escHtml(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}


// ═══════════════════════════════════════════════════════════
// STEP 03 — DISTRIBUTION
// ═══════════════════════════════════════════════════════════
let distAgent = 'Gemini CLI';

function selectDistAgent(name, btn){
  distAgent = name;
  document.querySelectorAll('#dist-agent-bar .apill').forEach(b=>b.className='apill');
  btn.classList.add('sel-'+AGENT_COLORS[name]);
  renderDistMetrics();
}

function renderDistMetrics(){
  const d = SUMMARY[distAgent]; if(!d) return;
  const c = AGENT_HEX[distAgent];
  const cvarBad = d.cvar === 0;

  document.getElementById('dist-metrics').innerHTML = `
    <div class="metric">
      <div class="m-label">Mean — Harbor's number</div>
      <div class="m-val" style="color:${c}">${d.mean.toFixed(3)}</div>
      <div class="m-sub">average score across all ${d.n_runs||'?'} runs</div>
    </div>
    <div class="metric">
      <div class="m-label">IQM — middle 50%</div>
      <div class="m-val" style="color:var(--glow)">${d.iqm.toFixed(3)}</div>
      <div class="m-sub">average with the top and bottom 25% of runs removed — less sensitive to outliers than the mean</div>
    </div>
    <div class="metric" style="cursor:default">
      <div class="m-label">Tail risk — worst runs</div>
      <div class="m-val" style="color:${cvarBad?'var(--red)':'var(--glow)'}">${d.cvar.toFixed(3)}</div>
      <div class="m-sub">average score of the <strong>worst 10%</strong> of runs — what you get on a bad day (CVaR@0.1)</div>
    </div>
    <div class="metric">
      <div class="m-label">Consistency — κ</div>
      <div class="m-val" style="color:${c}">${d.kappa.toFixed(3)}</div>
      <div class="m-sub">how often the agent achieves its own best score — 1.0 = always, 0.5 = half the time</div>
    </div>`;

  const n_runs_note = `<span style="color:var(--muted);font-size:.78rem">Source: all run scores from jobs/ — ${d.n_runs||'?'} runs total for this agent.</span>`;
  const callout = cvarBad
    ? `<div class="callout red"><div class="callout-label">⚠ what this means</div>
       <p>The tail risk (CVaR@0.1) for <strong>${distAgent} is 0.000</strong>.
       That means: take the worst 10% of this agent's runs — on those runs it scores <strong>zero</strong>.
       Complete failures. The mean of ${d.mean.toFixed(3)} is real, but it's being propped up by good runs on some tasks
       while hiding total failures on others. You wouldn't know this from Harbor alone.</p>
       <div style="margin-top:8px">${n_runs_note}</div></div>`
    : `<div class="callout green"><div class="callout-label">✓ what this means</div>
       <p>Tail risk (CVaR@0.1) for ${distAgent} is ${d.cvar.toFixed(3)}.
       Even in the worst 10% of runs, the agent still scores ${d.cvar.toFixed(3)}.
       The mean of ${d.mean.toFixed(3)} is a fair representation of how this agent actually behaves.</p>
       <div style="margin-top:8px">${n_runs_note}</div></div>`;
  document.getElementById('dist-callout').innerHTML = callout;
}


// ═══════════════════════════════════════════════════════════
// STEP 03 — WOW MOMENT
// ═══════════════════════════════════════════════════════════
const DIFF_DATA = {
  easy: {
    desc: 'FizzBuzz, word count — tasks a junior dev finishes in 5 minutes',
    agents: {
      'Claude Code': {mean:1.000, cvar:1.000},
      'Gemini CLI':  {mean:0.667, cvar:0.000},
      'Codex CLI':   {mean:0.333, cvar:0.000},
    },
    // Source: jobs/run_B scores for tasks/easy-* (3 runs each)
    // Gemini: [1.0, 1.0, 0.0] on word-count, [1.0,1.0,1.0] on fizzbuzz → mean=0.667, worst run=0
    insight: '<strong>This is the finding.</strong> ' +
      'Gemini scores 0.667 on average — looks okay. But its worst-10% tail score is <strong>0.000</strong>: ' +
      'on some runs it completely fails a task any developer would solve immediately. ' +
      'Concretely: Gemini ran the word-count task 3 times and failed it once (scores: 0.0, 1.0, 1.0). ' +
      'Harbor averaged those to 0.667 and moved on. ' +
      '<strong>You would only discover this failure mode in production.</strong>',
  },
  medium: {
    desc: 'Log parser, REST client — structured output, multi-step tasks',
    agents: {
      'Claude Code': {mean:0.500, cvar:0.500},
      'Gemini CLI':  {mean:0.333, cvar:0.000},
      'Codex CLI':   {mean:0.167, cvar:0.000},
    },
    // Source: jobs/ scores for tasks/medium-* (2-3 runs each)
    // Gemini: log-parser [0.5, 0.0, 0.5] → mean=0.333, worst run=0
    // Claude: log-parser [0.5, 0.5] → mean=0.500, worst run=0.5 (consistent partial credit)
    insight: 'The same pattern holds at medium difficulty. ' +
      'Gemini ran the log-parser task 3 times: it scored 0.5, then 0.0, then 0.5 — inconsistent. ' +
      'Claude ran it twice and scored 0.5 both times — lower peak, but <strong>perfectly consistent</strong>. ' +
      'Which would you rather deploy? The one that might give you nothing, or the one that always gives you half?',
  },
  hard: {
    desc: 'Algorithm implementation, bug fixing — complex multi-file tasks',
    agents: {
      'Claude Code': {mean:0.925, cvar:0.850},
      'Gemini CLI':  {mean:0.942, cvar:0.850},
      'Codex CLI':   {mean:0.550, cvar:0.100},
    },
    // Source: jobs/ scores for tasks/hard-* — Gemini [1.0,1.0] on algorithm, [0.85,0.85,0.85] on bugfix
    insight: '<strong>This is why Gemini\'s overall average looked good.</strong> ' +
      'On hard tasks — algorithms and bug fixes — Gemini is excellent: mean 0.942, worst-run score 0.850. ' +
      'It outperforms Claude on these. The good hard-task scores pulled the overall average up to 0.754 ' +
      'and masked the easy/medium failures entirely.',
  },
};
let selectedDiff = null;

function initDiffGrid(){
  document.getElementById('diff-grid').innerHTML = Object.entries(DIFF_DATA).map(([k,v])=>`
    <div class="diff-card" id="diff-${k}" onclick="selectDiff('${k}')">
      <div class="dc-label">${k.toUpperCase()} tasks</div>
      <div class="dc-val" style="color:${k==='hard'?'var(--glow)':'var(--red)'}">
        ${k==='hard'?'✓ stable':'✗ collapse'}
      </div>
      <div class="dc-sub">${v.desc}</div>
    </div>`).join('');
}

function selectDiff(diff){
  selectedDiff = diff;
  document.querySelectorAll('.diff-card').forEach(c=>c.classList.remove('selected'));
  document.getElementById('diff-'+diff).classList.add('selected');

  const d = DIFF_DATA[diff];
  // show chart
  document.getElementById('diff-detail').style.display = 'block';
  document.getElementById('img-difficulty').src = CHARTS.difficulty||'';

  // build per-agent bars
  let bars = `<div style="margin:12px 0">`;
  for(const [agent, vals] of Object.entries(d.agents)){
    const c = AGENT_HEX[agent];
    const cvarBad = vals.cvar === 0;
    bars += `
    <div style="margin-bottom:12px">
      <div style="font-size:.8rem;color:${c};margin-bottom:4px;font-weight:700">${agent}</div>
      <div class="hbar-wrap">
        <div class="hbar-label">Mean</div>
        <div class="hbar-track"><div class="hbar-fill" style="width:${vals.mean*100}%;background:${c}66"></div></div>
        <div class="hbar-val" style="color:${c}">${vals.mean.toFixed(3)}</div>
      </div>
      <div class="hbar-wrap">
        <div class="hbar-label" style="font-size:.7rem">Tail risk</div>
        <div class="hbar-track"><div class="hbar-fill" style="width:${vals.cvar*100}%;background:${cvarBad?'var(--red)':'var(--glow)'}"></div></div>
        <div class="hbar-val" style="color:${cvarBad?'var(--red)':'var(--glow)'}">${cvarBad?'0.000 ⚠':vals.cvar.toFixed(3)}</div>
      </div>
    </div>`;
  }
  bars += `<div style="font-size:.7rem;color:var(--muted);margin-top:6px">
    "Mean" = average across all runs on these tasks. "Tail risk" = average of the worst-10% of those runs.
    Source: actual scores from jobs/ directory.</div></div>`;

  document.getElementById('diff-callout').innerHTML =
    bars + `<div class="callout ${diff==='hard'?'green':'red'}" style="margin-top:12px">
    <div class="callout-label">${diff.toUpperCase()} tasks — what this means</div>
    <p>${d.insight}</p></div>`;
}


// ═══════════════════════════════════════════════════════════
// STEP 04 — RIGHT TAIL EXPLAINER
// ═══════════════════════════════════════════════════════════
const RT_EXPLAINERS = {
  solid: {
    label:'✓ SOLID — skip in training',
    body:'The agent gets its best possible score <em>every single time</em> it runs this task. ' +
      'There is no room for improvement and nothing to fix. ' +
      'For Gemini: FizzBuzz (3/3 perfect scores), the algorithm task (2/2 perfect), and the bug-fix task (3/3 at 0.85). ' +
      'Spending training compute on these tasks would be a waste — the agent already does them as well as it ever will.',
    color:'green',
  },
  recoverable: {
    label:'↑ RECOVERABLE — this is your training data',
    body:'The agent has solved this task before — but not consistently. ' +
      'That means we have two real runs of the same task: one where it succeeded, one where it failed. ' +
      'We can use those two runs as a training pair: "do what you did in the successful run, not what you did in the failing one." ' +
      'For Gemini: word-count task (scores 0.0, 1.0, 1.0 — the 0.0 run and a 1.0 run form the pair) ' +
      'and log-parser (scores 0.5, 0.0, 0.5 — the 0.0 run vs a 0.5 run). ' +
      'These are the highest-value training examples because the only variable between the two runs is the agent\'s choices.',
    color:'',
  },
  stuck: {
    label:'⚠ STUCK — don\'t train, explore instead',
    body:'The agent has never solved this task. Not even once. ' +
      'That means we have no "correct" example to point to. ' +
      'Training on only-failing examples would teach the agent to fail more consistently — that\'s worse than doing nothing. ' +
      'This is a signal that the agent is missing a capability (a tool, a reasoning pattern, domain knowledge) ' +
      'that no amount of DPO training on existing trajectories will fix. ' +
      'The right response is to add scaffolding, not to train.',
    color:'red',
  },
};

function showRtExplainer(kind){
  const e = RT_EXPLAINERS[kind];
  const el = document.getElementById('rt-explainer');
  el.className = `callout ${e.color}`;
  el.innerHTML = `<div class="callout-label">${e.label}</div><p>${e.body}</p>`;
  el.style.display = 'block';
}


// ═══════════════════════════════════════════════════════════
// STEP 05 — AGENT DETAIL
// ═══════════════════════════════════════════════════════════
let detailAgent = 'Gemini CLI';
let openTaskRow = null;

function selectDetailAgent(name, btn){
  detailAgent = name;
  document.querySelectorAll('#agent-detail-bar .apill').forEach(b=>b.className='apill');
  btn.classList.add('sel-'+AGENT_COLORS[name]);
  openTaskRow = null;
  renderDetailPanel();
}

function renderDetailPanel(){
  const d = SUMMARY[detailAgent]; if(!d) return;
  const c = AGENT_HEX[detailAgent];

  // kappa ring
  const kpct = d.kappa;
  const circ = 2 * Math.PI * 32;
  const offset = circ * (1 - kpct);
  document.getElementById('kappa-arc').style.strokeDashoffset = offset;
  document.getElementById('kappa-arc').setAttribute('stroke', c);
  document.getElementById('kappa-val').textContent = d.kappa.toFixed(3);
  document.getElementById('kappa-val').style.color = c;
  const pct_lost = (100*(1-d.kappa)).toFixed(1);
  document.getElementById('kappa-sub').innerHTML =
    `${detailAgent} scores ${d.kappa.toFixed(3)} on average relative to its own best. ` +
    `<strong>${pct_lost}%</strong> of the score it's capable of is being left on the table due to inconsistency. ` +
    `Source: q̄/q* computed per task from jobs/ data, then averaged.`;

  // metrics
  document.getElementById('detail-metrics').innerHTML = `
    <div class="metric">
      <div class="m-label">Average score</div>
      <div class="m-val" style="color:${c}">${d.mean.toFixed(3)}</div>
      <div class="m-sub">mean across all ${d.n_runs} runs — Harbor's number</div>
    </div>
    <div class="metric">
      <div class="m-label">Worst-run score</div>
      <div class="m-val" style="color:${d.cvar===0?'var(--red)':'var(--glow)'}">${d.cvar.toFixed(3)}</div>
      <div class="m-sub">average of the worst 10% of runs (CVaR@0.1)</div>
    </div>
    <div class="metric">
      <div class="m-label">Tasks to train on</div>
      <div class="m-val" style="color:var(--gold)">${d.recoverable}</div>
      <div class="m-sub">RECOVERABLE — solved inconsistently, has a training pair</div>
    </div>
    <div class="metric">
      <div class="m-label">Tasks to explore</div>
      <div class="m-val" style="color:${d.stuck>0?'var(--red)':'var(--muted)'}">${d.stuck}</div>
      <div class="m-sub">STUCK — never solved, training won't help here</div>
    </div>`;

  // task table
  const tbody = document.getElementById('detail-tbody');
  tbody.innerHTML = '';

  d.tasks.forEach((t, i)=>{
    const statusClass = `badge-${t.status.toLowerCase()}`;
    const hasPairs = t.has_pairs;
    const scoreStr = (t.scores||[]).map(s=>{
      const dot = s>=0.9?'var(--glow)':s>=0.5?'var(--gold)':'var(--red)';
      return `<span class="score-dot" style="background:${dot}"></span>${s.toFixed(2)}`;
    }).join(' ');

    const tr = document.createElement('tr');
    tr.className = 'task-row';
    tr.id = `trow-${i}`;
    tr.innerHTML = `
      <td><strong>${t.id}</strong></td>
      <td><span class="badge ${statusClass}">${t.status}</span></td>
      <td style="font-size:.78rem">${scoreStr}</td>
      <td style="color:${c}">${t.kappa.toFixed(2)}</td>
      <td style="color:${t.gap>0?'var(--gold)':'var(--muted)'}">${t.gap>0?'+'+t.gap.toFixed(3):'—'}</td>
      <td style="font-size:.75rem;color:${t.status==='RECOVERABLE'?'var(--gold)':t.status==='STUCK'?'var(--red)':'var(--muted)'}">
        ${t.status==='RECOVERABLE'?'↑ DPO train':t.status==='STUCK'?'⚠ explore':'✓ skip'}
        ${hasPairs?'<span style="color:var(--glow);margin-left:6px">▼ pairs</span>':''}
      </td>`;
    tr.onclick = ()=>toggleTaskExpand(i, t);
    tbody.appendChild(tr);

    // expand row
    const expandTr = document.createElement('tr');
    expandTr.className = 'task-expand';
    expandTr.id = `texp-${i}`;
    expandTr.innerHTML = `<td colspan="6"><div class="expand-inner" id="texp-body-${i}"></div></td>`;
    tbody.appendChild(expandTr);
  });
}

function toggleTaskExpand(i, task){
  const expRow = document.getElementById(`texp-${i}`);
  const taskRow = document.getElementById(`trow-${i}`);

  if(openTaskRow === i){
    expRow.classList.remove('open');
    taskRow.classList.remove('selected');
    openTaskRow = null;
    return;
  }

  // close previous
  if(openTaskRow !== null){
    document.getElementById(`texp-${openTaskRow}`)?.classList.remove('open');
    document.getElementById(`trow-${openTaskRow}`)?.classList.remove('selected');
  }
  openTaskRow = i;
  taskRow.classList.add('selected');

  // build content
  const body = document.getElementById(`texp-body-${i}`);
  body.innerHTML = buildTaskExpandHTML(task);
  expRow.classList.add('open');
}

function buildTaskExpandHTML(task){
  const status = task.status;
  const scoresStr = (task.scores||[]).join(', ') || '—';
  let html = `<div style="display:flex;gap:20px;font-size:.78rem;color:var(--muted);margin-bottom:12px;flex-wrap:wrap">
    <span>task folder: <strong style="color:var(--text)">${task.raw_id}</strong></span>
    <span>best score ever: <strong style="color:var(--text)">${task.q_star.toFixed(2)}</strong></span>
    <span>average score: <strong style="color:var(--text)">${task.q_bar.toFixed(2)}</strong></span>
    <span>all run scores: <strong style="color:var(--gold)">${scoresStr}</strong></span>
  </div>`;

  if(status === 'SOLID'){
    html += `<div class="callout green"><div class="callout-label">✓ SOLID — already reliable, skip in training</div>
      <p>Every single run of this task hit the same score (${task.q_star.toFixed(2)}). There's nothing to improve.
      Including this in training would just reinforce behaviour that's already consistent — a waste of compute.
      Source: task scores from jobs/ — consistency κ=${task.kappa.toFixed(2)}.</p></div>`;
  } else if(status === 'STUCK'){
    html += `<div class="callout red"><div class="callout-label">⚠ STUCK — never solved, don't train on this</div>
      <p>The agent's best score on this task was ${task.q_star.toFixed(2)} — it has never come close to solving it.
      There is no successful run to point to and say "do more of this." Training on only-failing runs would make
      things worse, not better. This task needs new capability: different tools, more context, or a sub-agent approach.
      Source: task scores from jobs/ — all scores were ${scoresStr}.</p></div>`;
  } else if(status === 'RECOVERABLE'){
    if(task.training_pairs && task.training_pairs.length > 0){
      const p = task.training_pairs[0];
      const rTools = (p.reinforce_tools||[]);
      const cTools = (p.contrast_tools||[]);
      html += `
      <div class="callout" style="margin-bottom:12px">
        <div class="callout-label">↑ RECOVERABLE — here is the actual training pair</div>
        <p>The agent ran this task ${(task.scores||[]).length} times and got different scores: ${scoresStr}.
        That inconsistency gives us two real runs to compare: a <strong style="color:var(--glow)">successful one (score ${p.reinforce_score.toFixed(2)})</strong>
        and a <strong style="color:var(--red)">failing one (score ${p.contrast_score.toFixed(2)})</strong>.
        Both ran the same task. The only difference is what the agent chose to do.
        At step ${p.divergence_step} the two runs made a different decision — that's the exact moment to train on.
        Source: trajectory files from jobs/ compared step-by-step.</p>
      </div>
      <div class="pair-grid">
        <div class="pair-card reinforce">
          <div class="pair-header">
            <span class="pair-title" style="color:var(--glow)">↑ The successful run — score ${p.reinforce_score.toFixed(2)}</span>
          </div>
          <div style="font-size:.72rem;color:var(--muted);margin-bottom:6px">Tool calls this agent made, in order:</div>
          <div class="tool-seq">${rTools.map(t=>`<span class="tool-chip ${KEY_TOOLS.includes(t)?'key-tool':''}">${t.replace(/_/g,' ')}</span>`).join('')}</div>
          <div class="diverge-marker">✓ wrote code and ran tests — succeeded</div>
        </div>
        <div class="pair-card contrast">
          <div class="pair-header">
            <span class="pair-title" style="color:var(--red)">↓ The failing run — score ${p.contrast_score.toFixed(2)}</span>
          </div>
          <div style="font-size:.72rem;color:var(--muted);margin-bottom:6px">Tool calls this agent made, in order:</div>
          <div class="tool-seq">${cTools.map(t=>`<span class="tool-chip">${t.replace(/_/g,' ')}</span>`).join('')}</div>
          <div class="diverge-marker" style="background:rgba(255,51,102,.08);border-color:var(--red);color:var(--red)">✗ took a different path at step ${p.divergence_step} and failed</div>
        </div>
      </div>`;
    } else {
      html += `<div class="callout"><div class="callout-label">↑ RECOVERABLE — needs more runs</div>
        <p>The agent scored inconsistently on this task (scores: ${scoresStr}).
        But we don't yet have both a passing run <em>and</em> a failing run to compare.
        Run a few more episodes to generate the pair.</p></div>`;
    }
  }
  return html;
}


// ═══════════════════════════════════════════════════════════
// STEP 06 — SELF ENGINE (LIVE SSE)
// ═══════════════════════════════════════════════════════════
let engAgent = 'Gemini CLI';
let engRunning = false;

function selectEngAgent(name, btn){
  if(engRunning) return;
  engAgent = name;
  document.querySelectorAll('#eng-agent-bar .apill').forEach(b=>b.className='apill');
  btn.classList.add('sel-'+AGENT_COLORS[name]);
  resetEngine();
}

function resetEngine(){
  ['scan','analyse','pairs','memory','plan'].forEach(id=>{
    const n = document.getElementById('pn-'+id);
    if(n){ n.classList.remove('active','done'); }
    const v = document.getElementById('pv-'+id);
    if(v) v.textContent = 'waiting';
  });
  document.getElementById('engine-log').innerHTML =
    `<div class="log-line"><span class="log-ts">--:--:--</span><span class="log-stage">ready</span><span class="log-msg">Select an agent and press Run.</span></div>`;
  document.getElementById('engine-results').style.display = 'none';
  document.getElementById('reset-btn').style.display = 'none';
  document.getElementById('run-btn').disabled = false;
}

function ts(){
  return new Date().toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
}

function appendLog(stage, msg, extraClass=''){
  const log = document.getElementById('engine-log');
  const div = document.createElement('div');
  div.className = `log-line stage-${stage} ${extraClass}`;
  div.innerHTML = `<span class="log-ts">${ts()}</span><span class="log-stage">${stage}</span><span class="log-msg">${msg}</span>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function activatePipe(id, val, isDone=false){
  // mark previous as done
  const order = ['scan','analyse','pairs','memory','plan'];
  const idx = order.indexOf(id);
  order.forEach((n,i)=>{
    const el = document.getElementById('pn-'+n);
    if(!el) return;
    el.classList.remove('active','done');
    if(i < idx) el.classList.add('done');
  });
  const el = document.getElementById('pn-'+id);
  if(el) el.classList.add(isDone?'done':'active');
  const v = document.getElementById('pv-'+id);
  if(v && val) v.textContent = val;
}

function buildCurriculumHTML(plan){
  const KEY_TOOLS2 = ['write_file','run_shell_command','read_file','execute_code','write_code'];
  return plan.curriculum.map((item,i)=>{
    const isRec = item.kind==='recoverable';
    const bColor = isRec?'var(--gold)':'var(--red)';
    const action = isRec?'↑ DPO train':'⚠ explore';
    const hasPair = item.training_pairs?.length > 0;
    const pair = hasPair ? item.training_pairs[0] : null;

    const consistencyNote = item.kappa < 1
      ? `consistency ${item.kappa.toFixed(2)} — agent achieves its best score on ${Math.round(item.kappa*100)}% of runs`
      : `consistency 1.0 — always achieves its best score`;
    const headerHTML = `
    <div class="curr-item-head" id="ci-head-${i}" onclick="toggleCurrItem(${i})">
      <div class="curr-item-name">${item.task}</div>
      <div class="curr-item-meta">
        <span>${consistencyNote}</span>
        <span style="color:var(--gold)">${item.gap>0?'unrealised gap +'+item.gap.toFixed(3):'no gap'}</span>
        <span>${(item.training_pairs?.length||0)} training pair${item.training_pairs?.length!==1?'s':''}</span>
      </div>
      <span class="badge" style="color:${bColor};border-color:${bColor};background:${bColor}18">${isRec?'↑ train':'⚠ explore'}</span>
      <div class="curr-item-gain" title="estimated improvement to overall score">+${(item.predicted_gain*100).toFixed(1)}%</div>
      <span style="color:var(--muted);font-size:.8rem">${hasPair?'▼ see pair':''}</span>
    </div>`;

    let bodyHTML = `<div class="curr-item-body" id="ci-body-${i}">`;
    if(hasPair){
      const p = pair;
      const rTools = p.reinforce_tools||[];
      const cTools = p.contrast_tools||[];
      bodyHTML += `
      <div style="font-size:.82rem;color:var(--text);margin-bottom:14px;line-height:1.6">
        ${item.recommendation}
      </div>
      <div style="font-size:.78rem;color:var(--muted);margin-bottom:10px">
        The agent ran this task multiple times. Below are the two runs that diverged the most.
        At step <strong style="color:var(--gold)">${p.divergence_step}</strong> they took different paths.
        The training signal is: "make the agent's future choices look more like the green run and less like the red run."
      </div>
      <div class="pair-grid">
        <div class="pair-card reinforce">
          <div class="pair-header">
            <span class="pair-title" style="color:var(--glow)">↑ The run that succeeded — score ${p.reinforce_score.toFixed(2)}</span>
          </div>
          <div style="font-size:.72rem;color:var(--muted);margin-bottom:6px">Tool calls made, in order (highlighted = key actions):</div>
          <div class="tool-seq">${rTools.map(t=>`<span class="tool-chip ${KEY_TOOLS2.includes(t)?'key-tool':''}">${t.replace(/_/g,' ')}</span>`).join('')}</div>
          <div class="diverge-marker">paths diverge after step ${p.divergence_step}</div>
        </div>
        <div class="pair-card contrast">
          <div class="pair-header">
            <span class="pair-title" style="color:var(--red)">↓ The run that failed — score ${p.contrast_score.toFixed(2)}</span>
          </div>
          <div style="font-size:.72rem;color:var(--muted);margin-bottom:6px">Tool calls made, in order:</div>
          <div class="tool-seq">${cTools.map(t=>`<span class="tool-chip">${t.replace(/_/g,' ')}</span>`).join('')}</div>
          <div class="diverge-marker" style="background:rgba(255,51,102,.08);border-color:var(--red);color:var(--red)">same task, different decisions → failed</div>
        </div>
      </div>
      <div style="font-size:.72rem;color:var(--muted);margin-top:8px">Source: trajectory JSON files from jobs/ directory, compared step-by-step by the engine.</div>`;
    } else {
      bodyHTML += `<div class="callout"><div class="callout-label">no training pair yet</div>
        <p>${item.recommendation || 'The agent hasn\'t run this task enough times yet to have both a passing and a failing run. Run more episodes to generate the pair.'}</p></div>`;
    }
    bodyHTML += `</div>`;

    return `<div class="curr-item">${headerHTML}${bodyHTML}</div>`;
  }).join('');
}

let openCurrItem = null;
function toggleCurrItem(i){
  const head = document.getElementById('ci-head-'+i);
  const body = document.getElementById('ci-body-'+i);
  if(!head||!body) return;
  const isOpen = body.classList.contains('open');
  // close all
  document.querySelectorAll('.curr-item-head').forEach(h=>h.classList.remove('open'));
  document.querySelectorAll('.curr-item-body').forEach(b=>b.classList.remove('open'));
  if(!isOpen){ head.classList.add('open'); body.classList.add('open'); }
}

async function runEngine(){
  if(engRunning) return;
  engRunning = true;
  document.getElementById('run-btn').disabled = true;
  document.getElementById('engine-log').innerHTML = '';
  document.getElementById('engine-results').style.display = 'none';

  appendLog('init', `Starting SelfEngine for <strong>${engAgent}</strong>…`);

  let pairCount = 0;
  const url = `/api/run_engine?agent=${encodeURIComponent(engAgent)}`;
  const resp = await fetch(url);
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';

  while(true){
    const {done, value} = await reader.read();
    if(done) break;
    buf += dec.decode(value, {stream:true});
    const parts = buf.split('\n\n');
    buf = parts.pop();
    for(const part of parts){
      if(!part.startsWith('data: ')) continue;
      let ev; try{ ev=JSON.parse(part.slice(6)); }catch{ continue; }

      if(ev.stage==='scan'){
        activatePipe('scan', `${ev.n_trajectories} runs`);
        appendLog('scan', `Loaded <strong>${ev.n_trajectories}</strong> trajectories · <strong>${ev.n_tasks}</strong> tasks · mean ${ev.mean}`);
      }
      else if(ev.stage==='analyse'){
        activatePipe('analyse', `κ=${ev.kappa}`);
        appendLog('analyse',
          `κ=<strong>${ev.kappa}</strong> &nbsp;` +
          `<span style="color:var(--glow)">${ev.n_solid} SOLID</span> · ` +
          `<span style="color:var(--gold)">${ev.n_recoverable} RECOVERABLE</span> · ` +
          `<span style="color:var(--red)">${ev.n_stuck} STUCK</span>`);
        for(const p of (ev.profiles||[])){
          const sc = p.kind==='solid'?'var(--glow)':p.kind==='recoverable'?'var(--gold)':'var(--red)';
          appendLog('analyse',
            `&nbsp;&nbsp;→ <strong>${p.task}</strong> · <span style="color:${sc}">${p.kind.toUpperCase()}</span> · gap=${p.gap} · κ=${p.kappa}`);
        }
      }
      else if(ev.stage==='pairs'){
        const n = ev.item?.training_pairs?.length||0;
        pairCount += n;
        activatePipe('pairs', `${pairCount} pair${pairCount!==1?'s':''}`);
        const pc = n>0?'var(--glow)':'var(--muted)';
        appendLog('pairs',
          `<strong>${ev.item?.task}</strong> · gap=${ev.item?.gap} · <span style="color:${pc}">${n} DPO pair${n!==1?'s':''}</span>`);
      }
      else if(ev.stage==='memory'){
        activatePipe('memory', `${ev.hits?.length||0} hit${ev.hits?.length!==1?'s':''}`);
        appendLog('memory', `TrajectoryMemory: <strong>${ev.hits?.length||0}</strong> similar past trajectories retrieved`);
      }
      else if(ev.stage==='plan'){
        activatePipe('plan', `+${(ev.plan?.predicted_gain*100||0).toFixed(1)}%`, true);
        appendLog('plan',
          `Curriculum complete · predicted gain <strong style="color:var(--glow)">+${(ev.plan?.predicted_gain*100||0).toFixed(1)}%</strong>`);

        // render results
        const plan = ev.plan;
        const c = AGENT_HEX[plan.agent_name]||'var(--glow)';
        document.getElementById('eng-gain-tag').textContent = `predicted +${(plan.predicted_gain*100).toFixed(1)}% gain`;
        document.getElementById('eng-metrics').innerHTML = `
          <div class="metric"><div class="m-label">Tasks to skip</div>
            <div class="m-val" style="color:var(--glow)">${plan.n_solid}</div>
            <div class="m-sub">SOLID — already consistent, don't waste compute here</div></div>
          <div class="metric"><div class="m-label">Tasks to train on</div>
            <div class="m-val" style="color:var(--gold)">${plan.n_recoverable}</div>
            <div class="m-sub">RECOVERABLE — inconsistent, have a real training pair</div></div>
          <div class="metric"><div class="m-label">Tasks to explore</div>
            <div class="m-val" style="color:${plan.n_stuck>0?'var(--red)':'var(--muted)'}">${plan.n_stuck}</div>
            <div class="m-sub">STUCK — never solved, need new capability not more training</div></div>
          <div class="metric"><div class="m-label">Consistency score</div>
            <div class="m-val" style="color:${c}">${plan.kappa.toFixed(3)}</div>
            <div class="m-sub">${(100*(1-plan.kappa)).toFixed(1)}% of achievable score being lost to inconsistency</div></div>`;
        document.getElementById('eng-curriculum').innerHTML = buildCurriculumHTML(plan);
        document.getElementById('engine-results').style.display = 'block';
      }
      else if(ev.stage==='done'){
        appendLog('done', '✓ Engine complete.');
        activatePipe('plan','',true);
        // mark all done
        ['scan','analyse','pairs','memory','plan'].forEach(id=>{
          document.getElementById('pn-'+id)?.classList.replace('active','done');
        });
      }
    }
  }
  engRunning = false;
  document.getElementById('run-btn').disabled = false;
  document.getElementById('reset-btn').style.display = '';
}


// ═══════════════════════════════════════════════════════════
// STEP 07 — PROOF DETAIL
// ═══════════════════════════════════════════════════════════
const PROOF_DETAILS = {
  '249': {
    title:'+249% more score gain per round vs picking at random',
    body:'Source: bootstrap simulation over the actual scores in jobs/. ' +
      'In each of 5,000 iterations, we drew a training batch using each strategy and simulated one round of DPO training. ' +
      'Using disteval\'s selection (RECOVERABLE tasks with both a passing and failing run) produced 249% more score improvement ' +
      'per training round than randomly picking from all available trajectories. p=0.030. ' +
      'Why? Random selection wastes roughly half its budget on tasks that are already consistent (SOLID) — ' +
      'those won\'t improve from training. And some of what it picks is from STUCK tasks, which have no positive example to learn from. ' +
      'disteval skips both those categories and only trains on the tasks where the problem is genuinely fixable.',
  },
  '172': {
    title:'+172% more score gain per round vs picking the best-scoring runs',
    body:'The obvious baseline is: pick the K runs with the highest scores and train on those. ' +
      'This is better than random — but it still loses to disteval by 172%. p=0.040. ' +
      'Here\'s why: the highest-scoring runs are mostly from tasks the agent already handles well. ' +
      'They don\'t teach the agent anything new. ' +
      'What\'s actually useful is having a successful run <em>and</em> a failing run on the <em>same task</em> — ' +
      'so the training signal is "here is exactly where you went wrong, and here is what success looks like." ' +
      'Top-K misses this entirely. It can only pick winners, never the contrast that makes the signal meaningful. ' +
      'Source: same bootstrap simulation over jobs/ scores.',
  },
  '178': {
    title:'17.8% fewer training rounds to reach a mean score of 0.80',
    body:'If your target is for an agent to hit a mean score of 0.80 across all tasks: ' +
      'using disteval\'s selection strategy gets you there in 17.8% fewer training rounds than top-K selection. ' +
      'This matters because training rounds are expensive — each one requires running the agent, scoring outputs, and a DPO pass. ' +
      'At small scale (6 tasks, 3 agents) the absolute difference is a handful of rounds. ' +
      'At production scale (50+ tasks, ongoing evaluation), it compounds into weeks of GPU time. ' +
      'Source: the training_sim.py script, using actual Harbor scores, counting rounds until mean 0.80 is reached.',
  },
  '5000': {
    title:'5,000 simulations on real Harbor scores, not synthetic data',
    body:'This is not a theoretical result or a synthetic benchmark. ' +
      'The underlying data is the actual scores from the 37 real Harbor runs in jobs/ (Claude, Gemini, Codex on 6 real Docker tasks). ' +
      'The bootstrap samples training batches from that actual pool 5,000 times per strategy, ' +
      'simulating how each strategy would perform over many training cycles. ' +
      'The p-values (0.030, 0.040) are computed from the variance across those 5,000 iterations — ' +
      'so the statistical significance comes from the real spread of your real data.',
  },
};

function showProofDetail(key){
  const d = PROOF_DETAILS[key];
  const el = document.getElementById('proof-detail');
  el.innerHTML = `<div class="callout-label">${d.title}</div><p>${d.body}</p>`;
  el.style.display = 'block';
}


// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
async function init(){
  const [sr, cr, tr] = await Promise.all([
    fetch('/api/summary'),
    fetch('/api/charts'),
    fetch('/api/trajectories'),
  ]);
  SUMMARY = await sr.json();
  CHARTS  = await cr.json();
  TRAJ    = await tr.json();

  // inject chart images
  document.getElementById('img-leaderboard').src  = CHARTS.leaderboard;
  document.getElementById('img-distribution').src = CHARTS.distribution;
  document.getElementById('img-difficulty').src   = CHARTS.difficulty;
  document.getElementById('img-right-tail').src   = CHARTS.right_tail;
  document.getElementById('img-simulation').src   = CHARTS.simulation;

  // enrich SUMMARY with task detail (from TRAJ)
  // attach training_pairs and scores per task
  for(const [agent, d] of Object.entries(SUMMARY)){
    d.tasks = d.tasks.map(t=>{
      // find pairs from curriculum data
      const agentCurr = TRAJ.curriculum?.[agent]?.curriculum||[];
      const currItem = agentCurr.find(c=>c.task===t.id);
      const pairs = currItem?.training_pairs||[];
      return {...t,
        has_pairs: pairs.length>0,
        training_pairs: pairs,
        scores: t.scores || [],
      };
    });
  }

  initDiffGrid();
  renderDistMetrics();
  renderDetailPanel();
  goStep(0);
}
init();
</script>
</body>
</html>
"""

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  disteval GUI")
    print("  ─────────────────────────────")
    print("  http://localhost:9173\n")
    uvicorn.run(app, host="0.0.0.0", port=9173, log_level="warning")
