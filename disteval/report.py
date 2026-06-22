"""disteval report CLI — the distribution-first agent reliability report.

Usage:
    python -m disteval.report <jobs_dir> [--output-dir <dir>] [--agent <name>]

Reads a Harbor jobs/ directory, builds a RecordStore from real trial output,
and produces:
  1. A rich terminal report (mean vs IQM vs CVaR, pass@k vs pass^k, failure modes)
  2. Three matplotlib plots saved to --output-dir
  3. A machine-readable summary JSON

This is the end-to-end proof that distributional eval surfaces what mean hides.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .adapters.harbor_jobs import load_harbor_job
from .records import RecordStore
from . import metrics, bootstrap, failure
from .right_tail import right_tail_analysis
from .viz import generate_all_plots


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _hr(title: str, width: int = 72) -> None:
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print('═' * width)


def _row(label: str, value: str, width: int = 30) -> None:
    print(f"  {label:<{width}} {value}")


def _color(text: str, code: str) -> str:
    """ANSI color if stdout is a tty."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def _fmt_score(v: float) -> str:
    if v >= 0.75:
        return _color(f"{v:.3f}", "32")   # green
    if v >= 0.4:
        return _color(f"{v:.3f}", "33")   # yellow
    return _color(f"{v:.3f}", "31")       # red


# --------------------------------------------------------------------------- #
# Main report logic                                                            #
# --------------------------------------------------------------------------- #
def build_report(store: RecordStore, agent_name: str = "Agent") -> dict:
    """Compute all report fields from a RecordStore. Returns summary dict."""
    df = store.df()
    scores = df["score"].to_numpy(float)

    # --- global metrics ---
    glob = {
        "n_episodes": int(len(scores)),
        "mean":       float(scores.mean()),
        "iqm":        metrics.iqm(scores),
        "median":     float(np.median(scores)),
        "std":        float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "var@0.1":    metrics.var_at(scores, 0.1),
        "cvar@0.1":   metrics.cvar(scores, 0.1),
        "success_rate": float(df["success"].mean()),
        "pass@1":     metrics.pass_at_k(df, 1),
        "pass@3":     metrics.pass_at_k(df, 3),
        "pass^1":     metrics.pass_hat_k(df, 1),
        "pass^3":     metrics.pass_hat_k(df, 3),
    }

    # --- bootstrap CI on mean ---
    boot = bootstrap.stratified_bootstrap_ci(
        df, lambda d: float(d["score"].mean()),
        strata_cols=["s_difficulty"] if "s_difficulty" in df.columns else [],
        n_reps=2000, seed=42,
    )

    # --- per-stratum breakdown ---
    strata_report = {}
    if "s_difficulty" in df.columns:
        for diff in ["easy", "medium", "hard"]:
            sub = df[df["s_difficulty"] == diff]
            if sub.empty:
                continue
            s = sub["score"].to_numpy(float)
            strata_report[diff] = {
                "n": int(len(s)),
                "mean":     float(s.mean()),
                "cvar@0.1": metrics.cvar(s, 0.1),
                "pass^3":   metrics.pass_hat_k(sub, 3),
                "success_rate": float(sub["success"].mean()),
            }

    # --- failure mode distribution ---
    fail_df = failure.failure_distribution(
        df,
        by=["s_difficulty"] if "s_difficulty" in df.columns else None,
    )

    # --- right-tail training signal ---
    rt_report = right_tail_analysis(store, model_name=agent_name)
    rt_summary = {
        "n_solid":        rt_report.n_solid,
        "n_recoverable":  rt_report.n_recoverable,
        "n_stuck":        rt_report.n_stuck,
        "total_gap":      rt_report.total_gap,
        "sum_q_star":     rt_report.sum_q_star,
        "sum_q_bar":      rt_report.sum_q_bar,
        "consistency_index": (rt_report.sum_q_bar / rt_report.sum_q_star
                              if rt_report.sum_q_star > 0 else 1.0),
        "priority_tasks": [
            {
                "task":        p.task,
                "difficulty":  p.difficulty,
                "scores":      p.scores,
                "q_star":      p.q_star,
                "q_bar":       p.q_bar,
                "gap":         p.gap,
                "consistency": p.consistency,
                "reinforce":   [f"#{i}(score={p.scores[i]:.2f})"
                                for i in p.reinforce_idx],
                "contrast":    [f"#{i}(score={p.scores[i]:.2f})"
                                for i in p.contrast_idx],
            }
            for p in rt_report.priority_tasks
        ],
    }

    return {
        "agent": agent_name,
        "global": glob,
        "bootstrap_ci": boot,
        "strata": strata_report,
        "failure_distribution": fail_df.to_dict(orient="records"),
        "right_tail": rt_summary,
    }


def print_report(rep: dict) -> None:
    g = rep["global"]
    b = rep["bootstrap_ci"]

    _hr(f"DISTEVAL AGENT RELIABILITY REPORT — {rep['agent']}")

    # Harbor's view (just the mean)
    mean_str = _color(f"{g['mean']:.3f}", '33')
    print(f"\n  {'Harbor reports only this:':30} mean = {mean_str}")
    print(f"  {'(their leaderboard number)':30}")

    _hr("WHAT THE DISTRIBUTION ACTUALLY SHOWS", width=72)
    _row("Episodes evaluated:", str(g["n_episodes"]))
    print()
    _row("Mean reward:",     _fmt_score(g["mean"]) + "  ← what Harbor reports")
    _row("IQM (robust center):", _fmt_score(g["iqm"]) + "  ← outlier-resistant")
    _row("Median:",          _fmt_score(g["median"]))
    _row("Std dev:",         f"{g['std']:.3f}")
    print()
    _row("VaR@0.1 (10th pctile):", _color(f"{g['var@0.1']:.3f}", "31"))
    _row("CVaR@0.1 (worst 10%):",  _color(f"{g['cvar@0.1']:.3f}", "31") +
         "  ← tail risk the mean hides")
    print()
    _row("Success rate:",    _fmt_score(g["success_rate"]))
    _row("pass@1:",          _fmt_score(g["pass@1"]) + "  ← peak capability")
    _row("pass@3:",          _fmt_score(g["pass@3"]))
    _row("pass^1:",          _fmt_score(g["pass^1"]))
    _row("pass^3:",          _fmt_score(g["pass^3"]) + "  ← does it ALWAYS succeed?")

    gap = g["pass@3"] - g["pass^3"]
    if gap > 0.05:
        print(f"\n  {_color('⚠ RELIABILITY GAP', '31')}  pass@3 − pass^3 = {gap:.3f}  "
              f"(the agent can do it but not consistently)")

    _hr("DIFFICULTY STRATIFICATION", width=72)
    if rep["strata"]:
        header = f"  {'Stratum':<10}  {'N':>5}  {'Mean':>6}  {'CVaR@0.1':>9}  {'pass^3':>7}  {'Status'}"
        print(header)
        print("  " + "-" * 60)
        for diff in ["easy", "medium", "hard"]:
            s = rep["strata"].get(diff)
            if not s:
                continue
            status = "✓ reliable" if s["cvar@0.1"] > 0.3 else ("~ unstable" if s["cvar@0.1"] > 0.05 else "✗ collapses")
            color = "32" if "✓" in status else ("33" if "~" in status else "31")
            print(f"  {diff.capitalize():<10}  {s['n']:>5}  {s['mean']:>6.3f}  "
                  f"{s['cvar@0.1']:>9.3f}  {s['pass^3']:>7.3f}  "
                  f"{_color(status, color)}")
    else:
        print("  (no difficulty strata found in task metadata)")

    _hr("FAILURE MODE BREAKDOWN", width=72)
    fail_records = rep["failure_distribution"]
    if fail_records:
        for rec in fail_records[:12]:
            parts = []
            if "s_difficulty" in rec:
                parts.append(f"{str(rec['s_difficulty']).capitalize():<8}")
            parts.append(f"{str(rec.get('failure_mode', 'unknown')):<35}")
            parts.append(f"n={rec['n']:>3}  {rec['share_of_failures']*100:>5.1f}%")
            print("  " + "  ".join(parts))
    else:
        print("  No failures recorded (all tasks succeeded).")

    _hr("EVAL RELIABILITY AUDIT", width=72)
    print(f"  Single-run bootstrap 95% CI:  ±{b['width']/2:.4f}  ← what Harbor implicitly uses")
    print(f"  Bootstrap CI range:           [{b['lo']:.3f}, {b['hi']:.3f}]")
    print()
    print(f"  {_color('NOTE', '33')}: This bootstrap CI only resamples episodes already collected.")
    print("  It CANNOT capture variance from fresh task draws, env seeds, or")
    print("  LLM nondeterminism. Run disteval repeat-eval to see true spread.")

    if "right_tail" in rep:
        rt = rep["right_tail"]
        _hr("RIGHT-TAIL SIGNAL (training leverage)", width=72)
        ci = rt["sum_q_bar"] / rt["sum_q_star"] if rt["sum_q_star"] > 0 else 1.0
        print(f"  Consistency index κ = {_color(f'{ci:.3f}', '32' if ci > 0.85 else '33' if ci > 0.6 else '31')}"
              f"  (Q̄/Q*; 1.0 = always achieves its own best)")
        print(f"  Total right-tail gap = {rt['total_gap']:.3f}  "
              f"← score recoverable through consistency, not new skills")
        print()
        for kind_label, key, color in [
            ("SOLID       ", "n_solid",       "32"),
            ("RECOVERABLE ", "n_recoverable", "33"),
            ("STUCK       ", "n_stuck",       "31"),
        ]:
            n = rt[key]
            desc = {"n_solid": "always at its best",
                    "n_recoverable": "demonstrated capability, inconsistent → training priority",
                    "n_stuck": "never solved → needs new skill or exploration"}[key]
            print(f"  {_color(kind_label, color)}  {n:>2} task(s)  {_color(desc, color)}")
        if rt["priority_tasks"]:
            print("\n  Top training targets (RECOVERABLE, ranked by gap):")
            for p in rt["priority_tasks"]:
                tag = f" [{p['difficulty']}]" if p.get('difficulty') else ""
                print(f"    {p['task'] + tag:<38}  "
                      f"Q*={p['q_star']:.2f}  κ={p['consistency']:.2f}  "
                      f"gap={_color('{:.3f}'.format(p['gap']), '33')}")
                if p.get("reinforce"):
                    print(f"    {'':38}  "
                          f"{_color('↑ reinforce: ' + ', '.join(p['reinforce']), '32')}")
                if p.get("contrast"):
                    print(f"    {'':38}  "
                          f"\033[2m↓ contrast:  {', '.join(p['contrast'])}\033[0m")
        print()

    _hr("VERDICT", width=72)
    c = g["cvar@0.1"]
    p3 = g["pass^3"]
    if c < 0.1:
        verdict = _color("HIGH RISK", "31") + f": CVaR@0.1={c:.3f} — agent catastrophically fails on tail tasks"
    elif c < 0.4:
        verdict = _color("MODERATE RISK", "33") + f": CVaR@0.1={c:.3f} — significant tail underperformance"
    else:
        verdict = _color("LOW RISK", "32") + f": CVaR@0.1={c:.3f} — reliable across difficulty levels"

    print(f"\n  {verdict}")
    if p3 < 0.5:
        print(f"  {_color('UNRELIABLE', '31')}: pass^3={p3:.3f} — agent fails to consistently complete tasks")
    elif p3 < 0.75:
        print(f"  {_color('INCONSISTENT', '33')}: pass^3={p3:.3f} — reliability needs improvement")
    else:
        print(f"  {_color('CONSISTENT', '32')}: pass^3={p3:.3f} — agent reliably completes tasks")

    print()


# --------------------------------------------------------------------------- #
# CLI entry point                                                              #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="disteval: distribution-first agent reliability report"
    )
    parser.add_argument("jobs_dir", help="Harbor jobs/ directory for one agent run")
    parser.add_argument("--output-dir", "-o", default="disteval_output",
                        help="Directory to save plots and JSON summary")
    parser.add_argument("--agent", "-a", default="Agent",
                        help="Agent name (for display)")
    parser.add_argument("--tasks-dir", "-t", default=None,
                        help="Tasks directory (to read difficulty metadata from task.toml)")
    parser.add_argument("--completed-only", action="store_true",
                        help="Exclude infra-error trials (missing_reward) — show only trials the agent actually ran")
    parser.add_argument("--success-threshold", type=float, default=0.99,
                        help="Score >= this is a success (default: 0.99)")
    parser.add_argument("--reward-key", default=None,
                        help="Named reward key to use as primary score")
    args = parser.parse_args(argv)

    # Load the Harbor job
    print(f"Loading Harbor job from: {args.jobs_dir}")
    store = load_harbor_job(
        args.jobs_dir,
        run_id="run0",
        tasks_dir=args.tasks_dir,
        success_threshold=args.success_threshold,
        reward_key=args.reward_key,
    )
    n_total = len(store)
    n_infra = len(store.df()[store.df()["failure_mode"] == "missing_reward"])

    if args.completed_only and n_infra > 0:
        from .records import RecordStore
        completed_records = [r for r in store._records if r.failure_mode != "missing_reward"]
        store = RecordStore(completed_records)
        print(f"Loaded {n_total} episode records total ({n_infra} infra errors excluded).")
        print(f"Reporting on {len(store)} completed trials (agent actually ran).")
    else:
        print(f"Loaded {n_total} episode records ({n_infra} infra errors included as score=0).")

    if len(store) == 0:
        print("No records found. Check that the jobs directory contains completed trials.")
        sys.exit(1)

    # Build and print report
    rep = build_report(store, agent_name=args.agent)
    print_report(rep)

    # Save plots
    print(f"\nGenerating plots → {args.output_dir}/")
    paths = generate_all_plots(store, args.output_dir, agent_name=args.agent)
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Save JSON summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {k: v for k, v in rep.items() if k != "failure_distribution"}
    summary["failure_distribution"] = rep["failure_distribution"]
    # Make numpy types serializable
    def _serial(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_serial)
    print(f"  summary: {summary_path}")
    print()


if __name__ == "__main__":
    main()
