"""
disteval.right_tail — LEGACY max-based task taxonomy (kept for compatibility).

.. deprecated:: 0.2
   Use :mod:`disteval.reliability` instead. This module is retained because its
   API is referenced by ``self_engine``, ``report``, and external callers, and
   because ``Q*``/``kappa`` remain useful *descriptive* statistics. It should not
   be used as a statistical estimate of latent capability. Nothing here is
   removed; :func:`right_tail_analysis` now also attaches posterior estimates
   (see ``profile.posterior`` / ``profile.label``) so migration is incremental.

WHAT THIS MODULE COMPUTES
─────────────────────────
For agent A on task t with k attempts producing scores q_1 ... q_k:

    Q*(t)    = max_i q_i          observed best (a descriptive statistic)
    Q̄(t)     = (1/k) Σ q_i       observed mean
    δ_i(t)   = Q*(t) - q_i       residual of attempt i below the observed best
    Δ(t)     = Q*(t) - Q̄(t)      observed spread below the best, ≥ 0
    κ(t)     = Q̄(t) / Q*(t)      observed consistency ratio ∈ [0,1]

These are honest summaries of the sample. They are *not* estimates of anything
latent, for two reasons this module previously got wrong:

1. **The maximum is biased upward and grows with k.** E[max of k draws] is
   monotonically increasing in k, so the same agent evaluated 20 times looks
   strictly more "capable" than evaluated 3 times. A single lucky success is
   enough to set Q* = 1 and declare the task mastered. This is the classic
   multiple-comparisons / winner's-curse problem, and it is why
   :mod:`disteval.reliability.posterior` estimates a posterior over p_t instead.

2. **Δ(t) = 0 does not mean "reliable".** A task run once always has Δ = 0 and
   is classified SOLID by the rules below. Uncertainty is invisible in this
   taxonomy; :mod:`disteval.reliability.classify` adds an UNCERTAIN category
   precisely for this case.

LEGACY TASK TAXONOMY (retained, superseded)
───────────────────────────────────────────
    SOLID        Q*(t) > 0,  Δ(t) = 0
    RECOVERABLE  Q*(t) > 0,  Δ(t) > 0
    STUCK        Q*(t) = 0

Compare :func:`disteval.reliability.classify.diagnose`, which defines the same
three names as posterior probability statements and adds UNCERTAIN.

CORRECTION: TAIL RISK
─────────────────────
Earlier versions of this module and of ``THEORY.md`` claimed that maximising the
*upper*-tail CVaR — E[q | q ≥ VaR_{1-α}] — penalises unreliable low-scoring
runs. **That claim is false.** Upper-tail CVaR ignores the lower tail by
construction. Concretely, at α = 1/3 on three runs::

    scores [0, 0, 1]  →  upper-tail CVaR_{2/3} = 1.0
    scores [1, 1, 1]  →  upper-tail CVaR_{2/3} = 1.0

An agent that fails two runs out of three and one that never fails are
indistinguishable under that objective, which is the opposite of what a
reliability metric must do. Reliability is a statement about the *bad* tail.
Use :func:`disteval.metrics.cvar` with ``tail="lower"`` (its default), or the
explicitly named helpers in :mod:`disteval.metrics`: ``lower_cvar``,
``worst_case``, ``lower_quantile``. Optimising an upper-tail functional is a
risk-*seeking* objective; it is a reasonable thing to want when the goal is
"reach the frontier at least sometimes", but it is not a reliability objective
and this repository no longer describes it as one.

WHAT SURVIVES
─────────────
The useful, correct ideas here are unchanged and now live on a sounder footing:

  - Tasks separate into "can do it but not consistently" versus "cannot do it",
    and those need different interventions.
  - Within a task, high-scoring and low-scoring attempts form a matched
    contrastive pair with no human labels required.
  - Ranking tasks by how much reliability is missing gives a training curriculum.

See ``THEORY.md`` and :mod:`disteval.reliability` for the corrected treatment,
and :mod:`disteval.selection` for the preference-pair construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from .records import RecordStore
from .reliability.classify import ReliabilityThresholds, diagnose
from .reliability.posterior import JEFFREYS_PRIOR, BetaPrior, TaskPosterior, posterior_from_scores


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

    # Information-theoretic extension
    outcome_entropy: float = 0.0             # H[Y_t] = uncertainty of score distribution

    # Recursive self-improvement extensions (optional, default-disabled)
    parent_task: Optional[str] = None        # parent task if this is a sub-task
    sub_task_depth: int = 0                  # recursion depth (0 = root task)
    sub_task_profiles: list["TaskOutcomeProfile"] = field(default_factory=list)
    recursive_gap: float = 0.0               # gap propagated from sub-task gaps

    # Posterior estimates (disteval.reliability). These are the statistically
    # defensible replacements for q_star/gap/kind; they are attached here so
    # existing consumers of this dataclass can migrate incrementally.
    posterior: Optional["TaskPosterior"] = None   # latent-performance posterior
    label: Optional[str] = None                   # SOLID/RECOVERABLE/STUCK/UNCERTAIN
    capability: Optional[float] = None            # C_t = P(p_t > tau_cap)
    reliability: Optional[float] = None           # R_t = P(p_t > tau_rel)
    recoverability: Optional[float] = None        # expected reliability headroom


def _outcome_entropy(scores: list[float], n_bins: int = 5) -> float:
    """Empirical Shannon entropy of the score distribution.

    H[Y] = -Σ p(y) log p(y)

    Discretizes scores into ``n_bins`` histogram bins and returns the entropy in nats.
    """
    arr = np.array(scores, dtype=float)
    if len(arr) < 2:
        return 0.0
    counts, _ = np.histogram(arr, bins=n_bins, range=(arr.min(), arr.max()))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


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

    @property
    def consistency_index(self) -> float:
        """κ = Σq̄ / Σq* — fraction of achievable (right-tail) score actually realized.

        1.0 means the agent performs at its own peak on every task; lower values
        quantify the score lost to run-to-run inconsistency rather than missing
        capability. Returns 1.0 when there is no achievable score (Σq* == 0).
        """
        return self.sum_q_bar / self.sum_q_star if self.sum_q_star > 0 else 1.0


# ── Core analysis ─────────────────────────────────────────────────────────────

def task_outcome_profile(
    task: str,
    scores: list[float],
    model: str,
    difficulty: Optional[str] = None,
    reinforce_threshold: float = 0.9,   # fraction of q_star to count as "high"
    parent_task: Optional[str] = None,
    sub_task_depth: int = 0,
    thresholds: Optional[ReliabilityThresholds] = None,
    prior: BetaPrior = JEFFREYS_PRIOR,
) -> TaskOutcomeProfile:
    """
    Compute the (legacy descriptive) right-tail profile for one (agent, task) cell,
    with posterior estimates attached.

    reinforce_threshold: attempts scoring >= reinforce_threshold * q_star
    are candidates for reinforcement. The rest are contrast examples.

    ``kind`` remains the legacy max-based label. ``label`` carries the posterior
    classification from :mod:`disteval.reliability.classify` and is what new code
    should read.
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
    contrast_idx = [i for i, s in enumerate(scores) if float(s) < threshold and q_star > 0]

    post = posterior_from_scores(scores, prior)
    diag = diagnose(post, task=task, model=model, thresholds=thresholds, scores=scores)

    return TaskOutcomeProfile(
        task=task, model=model, scores=list(scores),
        q_star=q_star, q_bar=q_bar, gap=gap, consistency=consistency,
        kind=kind, difficulty=difficulty,
        residuals=residuals,
        reinforce_idx=reinforce_idx,
        contrast_idx=contrast_idx,
        outcome_entropy=_outcome_entropy(scores),
        parent_task=parent_task,
        sub_task_depth=sub_task_depth,
        posterior=post,
        label=diag.label,
        capability=diag.capability,
        reliability=diag.reliability,
        recoverability=diag.recoverability,
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

    print("\n  VERDICT")
    for v in verdict_parts:
        print(f"    {v}")
    print(f"\n{EQ}\n")
