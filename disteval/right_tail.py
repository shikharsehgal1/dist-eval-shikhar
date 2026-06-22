"""
disteval.right_tail — Right-tail training signal for agentic tasks.

CORE IDEA
─────────
Standard RL collapses each episode to a single scalar reward and optimizes
E[R]. When an agent runs the same task k times, E[R] treats every attempt
equally — a perfect run and a zero run are averaged into 0.5.

This is wrong for agentic evals. The right question is:

    "The agent solved this task ONCE. Why can't it solve it EVERY time?"

The right tail of the agent's own outcome distribution on a task
(Q* = max over attempts) represents demonstrated, verified capability.
The gap (Q* - Q_i) on a failed attempt is NOT a missing skill — it is
inconsistency. These require completely different interventions.

DEFINITIONS
───────────
For agent A on task t with k attempts producing scores q_1 ... q_k:

    Q*(t)    = max_i q_i          right tail / demonstrated best
    Q̄(t)     = (1/k) Σ q_i       standard mean (what RL optimizes)
    δ_i(t)   = Q*(t) - q_i       right-tail residual for attempt i
    Δ(t)     = Q*(t) - Q̄(t)      right-tail gap for task t   ≥ 0

    Total right-tail gap:  Δ_total = Σ_t Δ(t)

    Consistency score:     κ(t) = 1 - Δ(t) / max(Q*(t), ε)
                                 = Q̄(t) / Q*(t)   when Q*(t) > 0
                           κ = 1 → always achieves its best
                           κ = 0 → achieved its best only once

TASK TAXONOMY
─────────────
Given attempts q_1 ... q_k on task t:

    SOLID        Q*(t) > 0,  Δ(t) = 0     — consistently achieves its best
    RECOVERABLE  Q*(t) > 0,  Δ(t) > 0     — knows how, but inconsistent
    STUCK        Q*(t) = 0                 — has never solved it; needs new skill

The right-tail signal only applies to RECOVERABLE tasks.
For STUCK tasks, the agent needs different training (exploration, new examples).

MATHEMATICAL ARGUMENT FOR RIGHT-TAIL TRAINING
──────────────────────────────────────────────
Let π_θ be the agent policy. On task t, it produces outcome distribution
F_t(q; θ). Standard RL maximizes:

    J_mean(θ) = E_{t,q ~ F_t} [q]

Right-tail training instead maximizes:

    J_rt(θ) = E_t [ E_{q ~ F_t} [q | q ≥ VaR_{1-α}(F_t)] ]
             = E_t [ CVaR_{1-α}(F_t) ]          (upper-tail CVaR)

Why this is better for agentic tasks:

1. CONSISTENCY vs CAPABILITY separation
   J_mean rewards a lucky high run the same as a consistent high run.
   J_rt specifically penalizes variance — you only get credit for the
   *expected* score in the top-α fraction, so you must be consistently good,
   not just occasionally good.

2. The recoverable-gap gradient
   For a RECOVERABLE task, ∂J_rt/∂θ points toward making low attempts look
   like high attempts (reducing δ_i for the low runs). This is exactly the
   trajectory-level counterfactual: "you solved this on attempt 2 — what
   did you do differently? Do that every time."

3. Natural curriculum
   Tasks sort into SOLID > RECOVERABLE > STUCK. The right-tail signal is
   zero for SOLID (nothing to improve there) and undefined for STUCK (no
   demonstrated upper bound to pull toward). It is maximally informative
   for RECOVERABLE tasks — a non-zero, achievable target exists.

4. Connection to CVaR-RL (distributional RL)
   Optimizing E[CVaR_{1-α}(F_t)] over tasks is equivalent to a risk-seeking
   objective over the AGENT'S OWN RETURN DISTRIBUTION, which has been shown
   to improve tail performance in distributional RL (Bellemare et al., 2017;
   Dabney et al., 2018). disteval makes this concrete and measurable without
   needing to modify the training loop — we compute the signal from eval data
   and show which trajectories to reinforce.

PRACTICAL USE
─────────────
Given a RecordStore from k attempts per task, right_tail_analysis() returns:

  - Per-task classification (SOLID / RECOVERABLE / STUCK)
  - δ_i for each episode (how far it fell below its own best)
  - κ(t) consistency score per task
  - Δ_total: total recoverable score left on the table
  - ranked list of RECOVERABLE tasks by gap (highest-leverage training targets)
  - For RECOVERABLE episodes: which specific attempts to REINFORCE (the high ones)
    and which to CONTRAST (the low ones)

The reinforcement target: the high-scoring trajectory within the same task
is the positive example. Low-scoring trajectories on the same task are
negative examples. This is a distributional contrastive signal derived
entirely from the agent's own eval data — no human labels required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from .records import EpisodeRecord, RecordStore


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TaskOutcomeProfile:
    """All outcomes for one (agent, task) combination."""
    task: str
    model: str
    scores: list[float]           # all k attempts, in order
    q_star: float                 # max score = right tail
    q_bar: float                  # mean score = what RL sees
    gap: float                    # q_star - q_bar
    consistency: float            # q_bar / q_star  (0–1; 1 = perfectly consistent)
    kind: str                     # "solid" | "recoverable" | "stuck"
    difficulty: Optional[str]

    # Episode-level residuals
    residuals: list[float] = field(default_factory=list)   # q_star - q_i per attempt
    # Which episode indices to REINFORCE (high) vs CONTRAST (low)
    reinforce_idx: list[int] = field(default_factory=list)
    contrast_idx:  list[int] = field(default_factory=list)

    # Recursive self-improvement extensions (optional, default-disabled)
    parent_task: Optional[str] = None        # parent task if this is a sub-task
    sub_task_depth: int = 0                  # recursion depth (0 = root task)
    sub_task_profiles: list["TaskOutcomeProfile"] = field(default_factory=list)
    recursive_gap: float = 0.0               # gap propagated from sub-task gaps


@dataclass
class RightTailReport:
    """Full right-tail analysis for one agent across all tasks."""
    model: str
    n_tasks: int
    n_episodes: int

    # Task-level breakdown
    profiles: list[TaskOutcomeProfile]

    # Aggregate stats
    n_solid:        int
    n_recoverable:  int
    n_stuck:        int
    total_gap:      float   # sum of gaps across all tasks
    pct_recoverable: float  # n_recoverable / n_tasks

    # The key insight number
    recoverable_score_left: float  # how much score exists if inconsistency is fixed
    sum_q_star: float              # theoretical max if always at right tail
    sum_q_bar:  float              # actual mean-aggregated score

    # Ranked training targets
    priority_tasks: list[TaskOutcomeProfile]  # RECOVERABLE, sorted by gap desc

    # Recursive self-improvement extensions (optional, default-disabled)
    sub_task_profiles: dict[str, list[TaskOutcomeProfile]] = field(default_factory=dict)
    recursive_gap: float = 0.0               # total gap propagated from sub-tasks


# ── Core analysis ─────────────────────────────────────────────────────────────

def task_outcome_profile(
    task: str,
    scores: list[float],
    model: str,
    difficulty: Optional[str] = None,
    reinforce_threshold: float = 0.9,   # fraction of q_star to count as "high"
    parent_task: Optional[str] = None,
    sub_task_depth: int = 0,
) -> TaskOutcomeProfile:
    """
    Compute the right-tail profile for one (agent, task) cell.

    reinforce_threshold: attempts scoring >= reinforce_threshold * q_star
    are candidates for reinforcement. The rest are contrast examples.
    """
    arr = np.array(scores, dtype=float)
    q_star = float(arr.max())
    q_bar  = float(arr.mean())
    gap    = q_star - q_bar

    if q_star == 0:
        kind = "stuck"
        consistency = 0.0
    elif gap < 1e-9:
        kind = "solid"
        consistency = 1.0
    else:
        kind = "recoverable"
        consistency = q_bar / q_star if q_star > 0 else 0.0

    residuals = [q_star - float(s) for s in scores]

    threshold = reinforce_threshold * q_star
    reinforce_idx = [i for i, s in enumerate(scores) if float(s) >= threshold and q_star > 0]
    contrast_idx  = [i for i, s in enumerate(scores) if float(s) < threshold]

    return TaskOutcomeProfile(
        task=task, model=model, scores=list(scores),
        q_star=q_star, q_bar=q_bar, gap=gap, consistency=consistency,
        kind=kind, difficulty=difficulty,
        residuals=residuals,
        reinforce_idx=reinforce_idx,
        contrast_idx=contrast_idx,
        parent_task=parent_task,
        sub_task_depth=sub_task_depth,
    )


def right_tail_analysis(
    store: RecordStore,
    model_name: Optional[str] = None,
    reinforce_threshold: float = 0.9,
) -> RightTailReport:
    """
    Full right-tail analysis for one agent's RecordStore.

    Groups episodes by (model, task), computes per-task profiles,
    and returns a RightTailReport with training priorities.
    """
    df = store.df()
    if df.empty:
        raise ValueError("RecordStore is empty")

    model = model_name or (df["model"].iloc[0] if "model" in df.columns else "agent")

    profiles: list[TaskOutcomeProfile] = []
    diff_col = "s_difficulty" if "s_difficulty" in df.columns else None

    for task, group in df.groupby("task"):
        scores = group["score"].tolist()
        diff   = group[diff_col].iloc[0] if diff_col else None
        prof   = task_outcome_profile(
            task=str(task), scores=scores, model=model,
            difficulty=diff, reinforce_threshold=reinforce_threshold,
        )
        profiles.append(prof)

    n_solid       = sum(1 for p in profiles if p.kind == "solid")
    n_recoverable = sum(1 for p in profiles if p.kind == "recoverable")
    n_stuck       = sum(1 for p in profiles if p.kind == "stuck")
    total_gap     = sum(p.gap for p in profiles)
    sum_q_star    = sum(p.q_star for p in profiles)
    sum_q_bar     = sum(p.q_bar  for p in profiles)
    pct_recov     = n_recoverable / len(profiles) if profiles else 0.0

    priority = sorted(
        [p for p in profiles if p.kind == "recoverable"],
        key=lambda p: -p.gap,
    )

    return RightTailReport(
        model=model,
        n_tasks=len(profiles),
        n_episodes=len(df),
        profiles=profiles,
        n_solid=n_solid,
        n_recoverable=n_recoverable,
        n_stuck=n_stuck,
        total_gap=total_gap,
        pct_recoverable=pct_recov,
        recoverable_score_left=total_gap,
        sum_q_star=sum_q_star,
        sum_q_bar=sum_q_bar,
        priority_tasks=priority,
    )


# ── Comparison across agents ──────────────────────────────────────────────────

def compare_right_tail(reports: list[RightTailReport]) -> pd.DataFrame:
    """
    Build a comparison DataFrame across multiple agents.

    Columns:
        model, n_tasks, n_recoverable, pct_recoverable,
        total_gap, sum_q_star, sum_q_bar, consistency_index
    """
    rows = []
    for r in reports:
        ci = r.sum_q_bar / r.sum_q_star if r.sum_q_star > 0 else 1.0
        rows.append({
            "model":             r.model,
            "n_tasks":           r.n_tasks,
            "n_recoverable":     r.n_recoverable,
            "pct_recoverable":   round(r.pct_recoverable, 3),
            "total_gap":         round(r.total_gap, 4),
            "sum_q_star":        round(r.sum_q_star, 4),
            "sum_q_bar":         round(r.sum_q_bar, 4),
            "consistency_index": round(ci, 4),   # Q̄_total / Q*_total
        })
    return pd.DataFrame(rows).sort_values("consistency_index", ascending=False)


# ── Terminal display ──────────────────────────────────────────────────────────

_KIND_COLOR = {
    "solid":       "\033[1;32m",   # green
    "recoverable": "\033[1;33m",   # yellow
    "stuck":       "\033[1;31m",   # red
}
_RESET = "\033[0m"
_DIM   = "\033[2m"
_BOLD  = "\033[1m"

def _ckind(kind: str) -> str:
    return f"{_KIND_COLOR.get(kind,'')}{kind.upper():<12}{_RESET}"


def print_right_tail_report(report: RightTailReport, width: int = 80) -> None:
    """Rich terminal display of a RightTailReport."""
    hr = "─" * width
    EQ = "═" * width

    print(f"\n{EQ}")
    print(f"  RIGHT-TAIL SIGNAL ANALYSIS  —  {report.model}")
    print(EQ)

    ci = report.sum_q_bar / report.sum_q_star if report.sum_q_star > 0 else 1.0
    print(f"\n  Episodes evaluated:       {report.n_episodes}")
    print(f"  Tasks analysed:           {report.n_tasks}")
    print(f"  {'SOLID':<14}  {report.n_solid:>3}  (always achieves its best)")
    print(f"  {'RECOVERABLE':<14}  "
          f"\033[1;33m{report.n_recoverable:>3}\033[0m  "
          f"(demonstrated capability, but inconsistent)")
    print(f"  {'STUCK':<14}  "
          f"\033[1;31m{report.n_stuck:>3}\033[0m  "
          f"(has never solved this task)")

    print(f"\n  {hr}")
    print(f"  Sum of right-tail Q*:     {report.sum_q_star:.3f}  ← if always at best")
    print(f"  Sum of mean Q̄:            {report.sum_q_bar:.3f}  ← what RL sees")
    print(f"  Total right-tail gap:  \033[1;33m{report.total_gap:>7.3f}\033[0m  "
          f"← score recoverable through consistency training")
    print(f"  Consistency index κ:   \033[1;{'32' if ci > 0.85 else '33' if ci > 0.6 else '31'}m{ci:>7.3f}\033[0m  "
          f"  (Q̄/Q*; 1.0 = perfect)")

    if report.priority_tasks:
        print(f"\n  {hr}")
        print(f"  {_BOLD}TOP TRAINING PRIORITIES (RECOVERABLE tasks, ranked by gap){_RESET}")
        print(f"  {hr}")
        print(f"  {'Task':<34} {'Attempts':<26} {'Q*':>5} {'κ':>5} {'Gap':>6}")
        print(f"  {hr}")
        for p in report.priority_tasks:
            attempts_str = str([f"{s:.1f}" for s in p.scores])
            diff_tag = f" [{p.difficulty}]" if p.difficulty else ""
            print(f"  {p.task + diff_tag:<34} {attempts_str:<26} "
                  f"{p.q_star:>5.2f} {p.consistency:>5.2f} "
                  f"\033[1;33m{p.gap:>6.3f}\033[0m")
            # Show which trajectories to reinforce vs contrast
            hi = [f"#{i}({p.scores[i]:.1f})" for i in p.reinforce_idx]
            lo = [f"#{i}({p.scores[i]:.1f})" for i in p.contrast_idx]
            if hi:
                print(f"  {'':34}   "
                      f"\033[1;32m↑ reinforce: {', '.join(hi)}\033[0m")
            if lo:
                print(f"  {'':34}   "
                      f"\033[2m↓ contrast:  {', '.join(lo)}\033[0m")
        print(f"  {hr}")

    verdict_parts = []
    if report.n_recoverable > 0:
        pct = report.pct_recoverable * 100
        verdict_parts.append(
            f"\033[1;33m{report.n_recoverable} recoverable task(s) ({pct:.0f}%) — "
            f"{report.total_gap:.3f} score points are inconsistency, not missing skill\033[0m"
        )
    if report.n_stuck > 0:
        verdict_parts.append(
            f"\033[1;31m{report.n_stuck} stuck task(s) — genuine capability gap, needs new training\033[0m"
        )
    if not verdict_parts:
        verdict_parts.append("\033[1;32mAll tasks SOLID — no right-tail gap\033[0m")

    print(f"\n  VERDICT")
    for v in verdict_parts:
        print(f"    {v}")
    print(f"\n{EQ}\n")
