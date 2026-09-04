"""Separating "doesn't know how" from "knows how but executes badly".

The temptation and the problem
------------------------------
It is tempting to write

    P(Y = 1) = q * r

with ``q`` = capability and ``r`` = execution reliability, and call it a richer
model. **From binary run outcomes alone this is not identifiable.** Only the
product is observed, and any ``(q, r)`` with the same product fits the data
equally well: 8 successes in 10 runs is exactly as consistent with (q=0.8, r=1.0)
as with (q=1.0, r=0.8). Fitting such a model and reporting its two numbers would
be reporting the prior, not the data.

So this module does the only honest thing: it requires an **additional
observable** and states exactly what has to be true of it.

The identifiable formulation
----------------------------
Let ``A`` indicate that a run reached a designated *capability milestone* -- an
observable that shows the agent found the right approach, whether or not it
finished. Then

    q_{m,t} = P(A = 1)              "does it know the approach"
    r_{m,t} = P(Y = 1 | A = 1)      "given the right approach, does it execute"
    P(Y = 1) = q * r                 under assumption (A1) below

Both factors are now directly estimable from observed counts -- ``q`` from
milestone attainment across runs, ``r`` from the *conditional* success rate among
runs that reached the milestone -- and each gets its own Beta-Binomial posterior.
The decomposition is exact, not fitted.

Assumptions, stated so they can be checked and argued with
----------------------------------------------------------
**(A1) Necessity.** ``P(Y = 1 | A = 0) = 0``: a run cannot succeed without
reaching the milestone. If this is violated, the factorisation is not exact.
:class:`Decomposition` counts the violations and reports
``necessity_violations``; a non-zero count means the milestone is mis-specified
and the numbers should not be trusted.

**(A2) Validity.** Reaching the milestone genuinely indicates the agent found the
right approach. This is a modelling *choice* embodied in the milestone
definition, and it cannot be checked from the data. Choose milestones that are
substantive (a key sub-goal, a designated set of rubric criteria) rather than
trivially attainable, and report which definition was used -- it travels on the
result object.

**(A3) Within-task homogeneity.** Runs of a task are exchangeable. The same
assumption every other estimator here makes; see
:mod:`disteval.reliability.correlated` for the diagnostic.

Estimability
------------
``r`` is undefined when no run reached the milestone (``n_milestone = 0``): there
is nothing to condition on. The result marks ``r_estimable = False`` rather than
returning a prior-driven number, and downstream consumers must handle it. This
is common for genuinely STUCK tasks and is informative in itself.

Milestone definitions
---------------------
:func:`milestone_from_rubric`, :func:`milestone_from_events` and
:func:`milestone_from_score` cover the usual cases; any callable
``Trajectory -> bool`` works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from ..trajectory.events import Trajectory
from .posterior import JEFFREYS_PRIOR, BetaPrior, TaskPosterior, binary_posterior

__all__ = [
    "Milestone",
    "Decomposition",
    "decompose",
    "milestone_from_rubric",
    "milestone_from_events",
    "milestone_from_score",
]

Milestone = Callable[[Trajectory], bool]


def milestone_from_rubric(
    criteria: Sequence[str], min_satisfied: Optional[int] = None, threshold: float = 1.0
) -> Milestone:
    """Milestone = at least ``min_satisfied`` of ``criteria`` are met.

    Defaults to requiring all of them. Use for rubrics where a designated subset
    of criteria establishes that the right approach was taken.
    """
    need = len(criteria) if min_satisfied is None else int(min_satisfied)

    def _m(t: Trajectory) -> bool:
        got = sum(1 for c in criteria if float(t.rubric_scores.get(c, 0.0)) >= threshold)
        return got >= need

    _m.__doc__ = f"rubric milestone: >={need} of {list(criteria)} satisfied"
    return _m


def milestone_from_events(
    predicate: Callable[[Trajectory], bool], description: str = ""
) -> Milestone:
    """Milestone defined by an arbitrary trajectory predicate."""
    predicate.__doc__ = description or (predicate.__doc__ or "event milestone")
    return predicate


def milestone_from_score(threshold: float) -> Milestone:
    """Milestone = the run's partial score reached ``threshold``.

    The weakest of the three definitions: partial score is a coarse proxy for
    "found the right approach" and correlates with success by construction, which
    inflates ``r``. Use it only when nothing better is logged, and say so.
    """

    def _m(t: Trajectory) -> bool:
        return float(t.score) >= threshold

    _m.__doc__ = f"score milestone: score >= {threshold}"
    return _m


@dataclass
class Decomposition:
    """The estimated capability / execution-reliability split for one cell."""

    task: str
    model: str
    n_runs: int
    n_milestone: int
    n_success: int
    n_success_given_milestone: int
    capability: TaskPosterior            # q = P(A=1)
    execution: Optional[TaskPosterior]   # r = P(Y=1|A=1); None when unestimable
    milestone_description: str
    necessity_violations: int
    prior: BetaPrior

    @property
    def r_estimable(self) -> bool:
        return self.execution is not None

    @property
    def q_mean(self) -> float:
        return self.capability.mean

    @property
    def r_mean(self) -> float:
        return self.execution.mean if self.execution else float("nan")

    @property
    def implied_success_rate(self) -> float:
        """``q * r``: the model's fitted success probability."""
        return self.q_mean * self.r_mean if self.execution else float("nan")

    @property
    def bottleneck(self) -> str:
        """Which factor is the binding constraint: ``knowledge`` or ``execution``.

        Reported only when both are estimable and they differ by more than their
        combined posterior sd, otherwise ``"indeterminate"`` -- the honest answer
        when the data cannot separate them.
        """
        if not self.execution:
            return "knowledge" if self.q_mean < 0.5 else "indeterminate"
        gap = self.q_mean - self.r_mean
        noise = self.capability.sd + self.execution.sd
        if abs(gap) <= noise:
            return "indeterminate"
        return "knowledge" if gap < 0 else "execution"

    @property
    def valid(self) -> bool:
        """Whether assumption (A1) held in the observed data."""
        return self.necessity_violations == 0

    def to_dict(self) -> dict:
        qlo, qhi = self.capability.credible_interval()
        rlo, rhi = self.execution.credible_interval() if self.execution else (float("nan"),) * 2
        return {
            "task": self.task,
            "model": self.model,
            "n_runs": self.n_runs,
            "n_milestone": self.n_milestone,
            "n_success": self.n_success,
            "capability_q": self.q_mean,
            "capability_ci_lo": qlo,
            "capability_ci_hi": qhi,
            "execution_r": self.r_mean,
            "execution_ci_lo": rlo,
            "execution_ci_hi": rhi,
            "r_estimable": self.r_estimable,
            "implied_success_rate": self.implied_success_rate,
            "observed_success_rate": self.n_success / self.n_runs if self.n_runs else float("nan"),
            "bottleneck": self.bottleneck,
            "necessity_violations": self.necessity_violations,
            "valid": self.valid,
            "milestone": self.milestone_description,
        }


def decompose(
    trajectories: Sequence[Trajectory],
    milestone: Milestone,
    task: str = "",
    model: str = "",
    *,
    prior: BetaPrior = JEFFREYS_PRIOR,
) -> Decomposition:
    """Split observed success into capability and execution-reliability factors.

    See the module docstring for the identifiability argument and the three
    assumptions. Check ``result.valid`` before using the numbers.
    """
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("no trajectories supplied")
    task = task or trajectories[0].task
    model = model or trajectories[0].model

    reached = [bool(milestone(t)) for t in trajectories]
    succeeded = [bool(t.success) for t in trajectories]

    n = len(trajectories)
    n_ms = sum(reached)
    n_su = sum(succeeded)
    n_su_ms = sum(1 for a, y in zip(reached, succeeded) if a and y)
    violations = sum(1 for a, y in zip(reached, succeeded) if y and not a)

    q = binary_posterior(n_ms, n, prior)
    r = binary_posterior(n_su_ms, n_ms, prior) if n_ms > 0 else None

    return Decomposition(
        task=task,
        model=model,
        n_runs=n,
        n_milestone=n_ms,
        n_success=n_su,
        n_success_given_milestone=n_su_ms,
        capability=q,
        execution=r,
        milestone_description=(getattr(milestone, "__doc__", "") or "custom milestone").strip(),
        necessity_violations=violations,
        prior=prior,
    )
