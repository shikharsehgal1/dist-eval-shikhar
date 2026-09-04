"""Cost-aware reliability: reliability you can afford.

An agent that succeeds 95% of the time using 40 tool calls and $2.10 per task is
not obviously better than one that succeeds 88% of the time for $0.15. Which you
prefer depends on the deployment, and collapsing both into a single score picks
that trade-off on the reader's behalf without telling them.

So the default output here is a **Pareto frontier**, not a score. The scalarised
utility exists, but it takes explicit coefficients and is never computed with
defaults that pretend to be neutral.

Quantities tracked
------------------
Whatever the harness logs, keyed by name: ``tokens``, ``usd``, ``tool_calls``,
``seconds``, ``retries``. Everything below is generic over the key.

Metrics
-------
``reliable_success_per_cost``
    ``P(success) / E[cost]``. Simple, and the right first number, but note it
    rewards a cheap unreliable agent and an expensive reliable one identically at
    equal ratio -- read it alongside the raw pair, not instead of it.

``cost_conditional``
    ``E[cost | success]`` and ``E[cost | failure]`` separately. These often differ
    by a factor of several: failed runs burn budget flailing, or terminate early.
    Which of the two it is says a lot about the agent, and the pooled mean hides
    both.

``risk_adjusted_utility``
    ``U = E[R] - lambda * CVaR_alpha^loss - gamma * E[cost]`` with all three
    coefficients required, no defaults. The loss CVaR term is the lower-tail
    shortfall ``E[(target - q)^+ | worst alpha]``, so ``lambda`` prices bad runs
    over and above their effect on the mean -- which is exactly what a
    reliability-sensitive deployment wants and what expected reward alone misses.

``pareto_frontier``
    The non-dominated set over (reliability up, cost down). The intended headline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..metrics import cvar

__all__ = [
    "CostProfile",
    "cost_profile",
    "reliable_success_per_cost",
    "risk_adjusted_utility",
    "pareto_frontier",
    "pareto_table",
]


@dataclass
class CostProfile:
    """Cost and reliability summary for one agent (optionally on one task)."""

    model: str
    cost_key: str
    n_runs: int
    success_rate: float
    mean_cost: float
    median_cost: float
    p90_cost: float
    total_cost: float
    mean_cost_given_success: float
    mean_cost_given_failure: float
    success_per_cost: float
    task: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def cost_ratio_failure_to_success(self) -> float:
        """How much more a failed run costs than a successful one.

        >1 means failures burn budget; <1 means they bail early. Both are common
        and they call for different fixes.
        """
        if not np.isfinite(self.mean_cost_given_success) or self.mean_cost_given_success <= 0:
            return float("nan")
        return self.mean_cost_given_failure / self.mean_cost_given_success

    def to_dict(self) -> dict:
        d = {
            "model": self.model,
            "task": self.task,
            "cost_key": self.cost_key,
            "n_runs": self.n_runs,
            "success_rate": self.success_rate,
            "mean_cost": self.mean_cost,
            "median_cost": self.median_cost,
            "p90_cost": self.p90_cost,
            "total_cost": self.total_cost,
            "mean_cost_given_success": self.mean_cost_given_success,
            "mean_cost_given_failure": self.mean_cost_given_failure,
            "cost_ratio_failure_to_success": self.cost_ratio_failure_to_success,
            "success_per_cost": self.success_per_cost,
        }
        d.update(self.extras)
        return d


def cost_profile(
    costs: Sequence[float],
    successes: Sequence[float],
    cost_key: str = "usd",
    model: str = "",
    task: str = "",
) -> CostProfile:
    """Summarise cost and reliability together for one set of runs."""
    c = np.asarray(costs, dtype=float)
    y = np.asarray(successes, dtype=float)
    if c.size != y.size:
        raise ValueError("costs and successes must be the same length")
    if c.size == 0:
        raise ValueError("no runs supplied")
    ok = y > 0.5
    mean_c = float(c.mean())
    return CostProfile(
        model=model,
        task=task,
        cost_key=cost_key,
        n_runs=int(c.size),
        success_rate=float(y.mean()),
        mean_cost=mean_c,
        median_cost=float(np.median(c)),
        p90_cost=float(np.quantile(c, 0.9)),
        total_cost=float(c.sum()),
        mean_cost_given_success=float(c[ok].mean()) if ok.any() else float("nan"),
        mean_cost_given_failure=float(c[~ok].mean()) if (~ok).any() else float("nan"),
        success_per_cost=float(y.mean() / mean_c) if mean_c > 0 else float("inf"),
    )


def reliable_success_per_cost(
    successes: Sequence[float], costs: Sequence[float]
) -> float:
    """``P(success) / E[cost]``. See the module docstring for how to read it."""
    c = np.asarray(costs, dtype=float)
    y = np.asarray(successes, dtype=float)
    m = float(c.mean())
    return float(y.mean() / m) if m > 0 else float("inf")


def risk_adjusted_utility(
    scores: Sequence[float],
    costs: Sequence[float],
    *,
    lam: float,
    gamma: float,
    alpha: float = 0.2,
    target: float = 1.0,
) -> dict:
    """``U = E[R] - lam * CVaR_alpha^loss - gamma * E[cost]``.

    ``lam`` and ``gamma`` are required, deliberately: there is no neutral
    exchange rate between a reliability point and a dollar, and defaulting one
    would smuggle in a preference. The loss CVaR is the mean shortfall below
    ``target`` in the worst ``alpha`` fraction of runs, so it is zero for an
    agent whose bad runs are still good and grows as its tail gets worse.

    Returns the components alongside the total so the trade-off stays visible.
    """
    q = np.asarray(scores, dtype=float)
    c = np.asarray(costs, dtype=float)
    if q.size == 0:
        raise ValueError("no runs supplied")
    loss = np.clip(target - q, 0.0, None)
    # Upper tail of the loss = lower tail of the score: the bad runs.
    tail_loss = cvar(loss, alpha=alpha, tail="upper")
    e_r = float(q.mean())
    e_c = float(c.mean()) if c.size else 0.0
    return {
        "utility": float(e_r - lam * tail_loss - gamma * e_c),
        "expected_reward": e_r,
        "cvar_loss": float(tail_loss),
        "expected_cost": e_c,
        "lambda": lam,
        "gamma": gamma,
        "alpha": alpha,
        "target": target,
    }


def pareto_frontier(
    points: Sequence[Mapping],
    *,
    benefit_key: str = "success_rate",
    cost_key: str = "mean_cost",
) -> list[int]:
    """Indices of the non-dominated points (higher benefit, lower cost).

    A point is dominated when another is at least as good on both axes and
    strictly better on one. Ties on both axes are all kept -- arbitrarily
    dropping one of two identical options would be a silent choice.
    """
    b = np.array([float(p[benefit_key]) for p in points], dtype=float)
    c = np.array([float(p[cost_key]) for p in points], dtype=float)
    keep = []
    for i in range(len(points)):
        dominated = np.any(
            (b >= b[i]) & (c <= c[i]) & ((b > b[i]) | (c < c[i]))
        )
        if not dominated:
            keep.append(i)
    return keep


def pareto_table(profiles: Sequence[CostProfile], **kwargs) -> "object":
    """Cost profiles as a DataFrame with a ``pareto_optimal`` flag.

    This -- not a scalarised score -- is the intended headline output. The reader
    picks the trade-off; the framework's job is to show which options are on the
    frontier at all.
    """
    import pandas as pd

    rows = [p.to_dict() for p in profiles]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    front = set(pareto_frontier(rows, **kwargs))
    df["pareto_optimal"] = [i in front for i in range(len(rows))]
    return df.sort_values(["pareto_optimal", "success_rate"], ascending=[False, False]).reset_index(drop=True)
