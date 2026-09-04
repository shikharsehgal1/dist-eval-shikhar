"""Comparing two agents properly: paired, distributional, and with uncertainty.

What is wrong with comparing means
-----------------------------------
Two agents evaluated on the same tasks are a *paired* design, and treating them
as two independent samples throws away most of the power: the between-task
variance (some tasks are just hard) dominates the between-agent variance, and an
unpaired test buries the signal in it. Worse, a difference in means says nothing
about whether one agent is more *reliable* -- the property this repository cares
about. An agent can win on mean and lose on every low-tail measure.

So this module provides:

**Paired task-level tests.** Bootstrap and exact permutation on the per-task
difference, plus the Wilcoxon signed-rank test. Permutation is the default
because under the null "the labels A/B are exchangeable within each task" it is
exact for any statistic, requires no distributional assumption, and handles the
small-T regime (20-60 tasks) where asymptotics are unreliable.

**Posterior P(A more reliable than B).** With per-task posteriors already in
hand, the directly interesting quantity is not a p-value but
``P(p_A > p_B | data)``, computed per task by posterior sampling and aggregated.
This answers "how confident am I", which is what a decision needs.

**Stochastic dominance.** A checks whether A's score distribution dominates B's
first-order (``F_A(x) <= F_B(x)`` everywhere), which is a much stronger claim
than a higher mean: it means A is at least as good at *every* threshold, so no
choice of pass bar reverses the conclusion. Second-order dominance is also
tested, which is the risk-averse version and the one that usually settles
reliability comparisons.

**Effect sizes, always.** Every test returns Cohen's d, the rank-biserial
correlation, and the raw mean difference with an interval. A p-value without an
effect size invites reporting a 0.4-point difference across 400 tasks as a
finding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import stats

from .classify import PosteriorLike

__all__ = [
    "PairedComparison",
    "paired_comparison",
    "posterior_dominance",
    "stochastic_dominance",
    "reliability_profile_comparison",
]


@dataclass
class PairedComparison:
    """Result of a paired task-level comparison of two agents."""

    model_a: str
    model_b: str
    n_tasks: int
    mean_a: float
    mean_b: float
    mean_diff: float
    diff_ci: tuple[float, float]
    permutation_p: float
    wilcoxon_p: float
    cohens_d: float
    rank_biserial: float
    n_a_better: int
    n_b_better: int
    n_tied: int
    metric: str = "score"
    meta: dict = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero. Not a substitute for the effect size."""
        return self.diff_ci[0] > 0 or self.diff_ci[1] < 0

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "n_tasks": self.n_tasks,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_diff": self.mean_diff,
            "diff_ci_lo": self.diff_ci[0],
            "diff_ci_hi": self.diff_ci[1],
            "permutation_p": self.permutation_p,
            "wilcoxon_p": self.wilcoxon_p,
            "cohens_d": self.cohens_d,
            "rank_biserial": self.rank_biserial,
            "n_a_better": self.n_a_better,
            "n_b_better": self.n_b_better,
            "n_tied": self.n_tied,
            "significant": self.significant,
        }


def paired_comparison(
    a: Sequence[float],
    b: Sequence[float],
    model_a: str = "A",
    model_b: str = "B",
    *,
    metric: str = "score",
    n_boot: int = 5000,
    n_perm: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Paired comparison of two agents over the same tasks.

    ``a`` and ``b`` are per-task values in the same task order -- one number per
    task, not per run. Aggregate runs to a task-level statistic first (posterior
    mean, pass^k, lower-tail CVaR) and the comparison inherits whatever
    reliability property that statistic encodes.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size != y.size:
        raise ValueError("paired comparison needs one value per task for each agent")
    if x.size < 2:
        raise ValueError("need at least 2 tasks")
    d = x - y
    rng = np.random.default_rng(seed)

    # Paired bootstrap over tasks.
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    tail = (1 - level) / 2
    ci = (float(np.quantile(boots, tail)), float(np.quantile(boots, 1 - tail)))

    # Exact-in-the-limit permutation: flip the sign of each task's difference,
    # which is exactly the null that the A/B labels are exchangeable per task.
    obs = float(np.abs(d.mean()))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    perm_p = float((np.sum(null >= obs) + 1) / (n_perm + 1))

    try:
        w_p = float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)
    except ValueError:
        w_p = float("nan")

    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    n_a = int(np.sum(d > 0))
    n_b = int(np.sum(d < 0))
    n_t = int(np.sum(d == 0))
    return PairedComparison(
        model_a=model_a,
        model_b=model_b,
        n_tasks=int(d.size),
        mean_a=float(x.mean()),
        mean_b=float(y.mean()),
        mean_diff=float(d.mean()),
        diff_ci=ci,
        permutation_p=perm_p,
        wilcoxon_p=w_p,
        cohens_d=float(d.mean() / sd) if sd > 0 else float("nan"),
        rank_biserial=float((n_a - n_b) / d.size),
        n_a_better=n_a,
        n_b_better=n_b,
        n_tied=n_t,
        metric=metric,
        meta={"n_boot": n_boot, "n_perm": n_perm},
    )


def posterior_dominance(
    posteriors_a: Mapping[str, PosteriorLike],
    posteriors_b: Mapping[str, PosteriorLike],
    *,
    n_samples: int = 20000,
    seed: int = 0,
    threshold: Optional[float] = None,
) -> dict:
    """``P(A more reliable than B)`` from the per-task posteriors.

    Two readings are returned:

    * ``per_task`` -- ``P(p_A > p_B)`` for each shared task, by joint sampling.
    * ``prob_a_better_overall`` -- ``P(mean_t p_A,t > mean_t p_B,t)``, sampling
      all tasks jointly so the aggregate inherits every task's uncertainty rather
      than comparing two point estimates.

    With ``threshold`` set, ``prob_a_meets_bar_more_often`` compares the expected
    number of tasks each agent clears at that reliability bar -- often the
    decision-relevant question ("which ships more tasks at 90% reliability").
    """
    shared = sorted(set(posteriors_a) & set(posteriors_b))
    if not shared:
        return {"n_tasks": 0, "per_task": {}, "prob_a_better_overall": float("nan")}
    rng = np.random.default_rng(seed)
    A = np.vstack([posteriors_a[t].sample(n_samples, rng) for t in shared])
    B = np.vstack([posteriors_b[t].sample(n_samples, rng) for t in shared])
    per_task = {t: float(np.mean(A[i] > B[i])) for i, t in enumerate(shared)}
    out = {
        "n_tasks": len(shared),
        "per_task": per_task,
        "prob_a_better_overall": float(np.mean(A.mean(axis=0) > B.mean(axis=0))),
        "mean_prob_a_better": float(np.mean(list(per_task.values()))),
        "n_tasks_a_favoured": int(sum(1 for v in per_task.values() if v > 0.5)),
    }
    if threshold is not None:
        out["prob_a_meets_bar_more_often"] = float(
            np.mean((threshold < A).sum(axis=0) > (threshold < B).sum(axis=0))
        )
        out["threshold"] = threshold
    return out


def stochastic_dominance(
    a: Sequence[float], b: Sequence[float], n_grid: int = 512
) -> dict:
    """First- and second-order stochastic dominance tests.

    First order (``F_A(x) <= F_B(x)`` for all x) means A is at least as good at
    every threshold: no choice of pass bar reverses the conclusion, which a mean
    comparison cannot promise. Second order (the integrated CDF condition) is the
    risk-averse version and is the one that usually settles reliability
    comparisons -- it holds when A is preferred by every risk-averse decision
    maker, even where first-order dominance fails.

    ``violation`` is the largest amount by which the condition fails, so a tiny
    violation reads as "essentially dominates" rather than a flat no.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("both samples must be non-empty")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if hi <= lo:
        hi = lo + 1e-9
    grid = np.linspace(lo, hi, n_grid)
    Fa = np.array([(x <= g).mean() for g in grid])
    Fb = np.array([(y <= g).mean() for g in grid])
    d1 = Fa - Fb                       # A dominates when this is <= 0 everywhere
    step = (hi - lo) / (n_grid - 1)
    Ia, Ib = np.cumsum(Fa) * step, np.cumsum(Fb) * step
    d2 = Ia - Ib
    return {
        "first_order_a_dominates": bool(np.all(d1 <= 1e-12)),
        "first_order_b_dominates": bool(np.all(d1 >= -1e-12)),
        "first_order_violation": float(max(d1.max(), 0.0)),
        "second_order_a_dominates": bool(np.all(d2 <= 1e-12)),
        "second_order_b_dominates": bool(np.all(d2 >= -1e-12)),
        "second_order_violation": float(max(d2.max(), 0.0)),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
    }


def reliability_profile_comparison(
    per_task_a: Mapping[str, Sequence[float]],
    per_task_b: Mapping[str, Sequence[float]],
    model_a: str = "A",
    model_b: str = "B",
    **kwargs,
) -> dict:
    """Compare two agents on several reliability statistics at once.

    Runs the paired test on the mean, on the lower-tail CVaR, and on the worst
    run, over the shared tasks. The point is that these can disagree: an agent
    that wins on the mean and loses on the lower tail is trading reliability for
    average performance, and reporting only the mean would hide the trade.
    """
    from ..metrics import lower_cvar

    shared = sorted(set(per_task_a) & set(per_task_b))
    if len(shared) < 2:
        return {"n_tasks": len(shared), "comparisons": {}, "note": "too few shared tasks"}

    stats_ = {
        "mean": lambda v: float(np.mean(v)),
        "lower_cvar": lambda v: float(lower_cvar(np.asarray(v, dtype=float), 0.25)),
        "worst_run": lambda v: float(np.min(v)),
    }
    out = {}
    for name, fn in stats_.items():
        a = [fn(per_task_a[t]) for t in shared]
        b = [fn(per_task_b[t]) for t in shared]
        out[name] = paired_comparison(
            a, b, model_a, model_b, metric=name, **kwargs
        ).to_dict()
    directions = {k: np.sign(v["mean_diff"]) for k, v in out.items()}
    return {
        "n_tasks": len(shared),
        "comparisons": out,
        "consistent_direction": len(set(directions.values())) == 1,
        "note": ""
        if len(set(directions.values())) == 1
        else "the agents rank differently on mean and on tail statistics; "
             "reporting only the mean would hide a reliability trade-off",
    }
