"""Per-criterion reliability: which parts of the task are unstable?

Why overall success is too coarse
---------------------------------
A task with ten rubric criteria where nine are always satisfied and one is a coin
flip is a completely different object from one where all ten are half-right. Both
have the same overall pass rate and the same mean score. The first has one
localised execution defect and is an excellent preference-training target; the
second suggests the agent does not really know the task.

So each criterion gets its own Beta-Binomial posterior across the task's runs,
and criteria are classified the same way tasks are:

``STABLE_PASS``     strong posterior evidence the criterion is nearly always met
``STOCHASTIC``      evidence it is sometimes met and sometimes not
``STABLE_FAIL``     strong evidence it is essentially never met
``UNCERTAIN``       too few runs to say

Dependency-aware counting
-------------------------
Criteria are not independent. If a wrong retrieval causes five downstream
criteria to fail, counting five capability deficits overstates the problem
fivefold. When a :class:`~disteval.diagnosis.causality.CriterionGraph` is
supplied, :func:`root_criterion_profile` reports the *root* failing criteria --
those with no failing declared parent -- alongside the raw count, and the
amplification between them.

The dependency graph is declared by the benchmark, not inferred. Inferring it
from co-occurrence across a handful of runs would confuse "these fail together
because one causes the other" with "these fail together because the task is
hard", and there is no way to separate those without intervention.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from ..diagnosis.causality import CriterionGraph
from .classify import ReliabilityThresholds
from .posterior import JEFFREYS_PRIOR, BetaPrior, TaskPosterior, binary_posterior

__all__ = [
    "STABLE_PASS",
    "STOCHASTIC",
    "STABLE_FAIL",
    "UNCERTAIN_CRITERION",
    "CriterionProfile",
    "RubricProfile",
    "criterion_posteriors",
    "profile_rubric",
    "root_criterion_profile",
    "rubric_matrix",
]

STABLE_PASS = "STABLE_PASS"
STOCHASTIC = "STOCHASTIC"
STABLE_FAIL = "STABLE_FAIL"
UNCERTAIN_CRITERION = "UNCERTAIN"


@dataclass
class CriterionProfile:
    """Posterior reliability of one rubric criterion on one task."""

    criterion: str
    task: str
    model: str
    n_runs: int
    n_satisfied: int
    posterior: TaskPosterior
    label: str
    is_root: Optional[bool] = None    # None when no dependency graph was supplied

    @property
    def rate(self) -> float:
        return self.n_satisfied / self.n_runs if self.n_runs else float("nan")

    def to_dict(self) -> dict:
        lo, hi = self.posterior.credible_interval()
        return {
            "task": self.task,
            "model": self.model,
            "criterion": self.criterion,
            "n_runs": self.n_runs,
            "n_satisfied": self.n_satisfied,
            "observed_rate": self.rate,
            "posterior_mean": self.posterior.mean,
            "posterior_sd": self.posterior.sd,
            "ci_lo": lo,
            "ci_hi": hi,
            "label": self.label,
            "is_root": self.is_root,
        }


@dataclass
class RubricProfile:
    """All criteria for one (model, task), plus the aggregate reading."""

    task: str
    model: str
    criteria: list[CriterionProfile]
    n_runs: int
    criterion_graph: Optional[CriterionGraph] = None
    meta: dict = field(default_factory=dict)

    def by_label(self, label: str) -> list[CriterionProfile]:
        return [c for c in self.criteria if c.label == label]

    @property
    def n_stochastic(self) -> int:
        return len(self.by_label(STOCHASTIC))

    @property
    def n_stable_fail(self) -> int:
        return len(self.by_label(STABLE_FAIL))

    @property
    def instability_concentration(self) -> float:
        """Fraction of *non-stable-pass* criteria that are merely stochastic.

        1.0 means every shortfall is an execution-reliability problem on a
        criterion the agent can satisfy; 0.0 means every shortfall is a criterion
        it never satisfies. This is the rubric-level analogue of the
        RECOVERABLE vs STUCK distinction and is a much sharper reading of a task
        than its overall score.
        """
        unstable = self.n_stochastic + self.n_stable_fail
        if unstable == 0:
            return float("nan")
        return self.n_stochastic / unstable

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "model": self.model,
            "n_runs": self.n_runs,
            "n_criteria": len(self.criteria),
            "n_stable_pass": len(self.by_label(STABLE_PASS)),
            "n_stochastic": self.n_stochastic,
            "n_stable_fail": self.n_stable_fail,
            "n_uncertain": len(self.by_label(UNCERTAIN_CRITERION)),
            "instability_concentration": self.instability_concentration,
            **self.meta,
        }


def criterion_posteriors(
    runs: Sequence[Mapping[str, float]],
    *,
    prior: BetaPrior = JEFFREYS_PRIOR,
    satisfied_threshold: float = 1.0,
) -> dict[str, TaskPosterior]:
    """Beta-Binomial posterior per criterion from per-run criterion scores.

    ``runs`` is one mapping per run from criterion id to score. Criteria missing
    from a run are treated as not observed in that run (not as failures), so a
    rubric that grows between runs does not fabricate failures.
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [n_obs, n_sat]
    for r in runs:
        for k, v in (r or {}).items():
            counts[k][0] += 1
            counts[k][1] += int(float(v) >= satisfied_threshold)
    return {
        k: binary_posterior(sat, n, prior) for k, (n, sat) in counts.items()
    }


def _criterion_label(
    post: TaskPosterior, th: ReliabilityThresholds
) -> str:
    if post.n_obs < th.min_runs:
        return UNCERTAIN_CRITERION
    d = th.confidence
    if post.prob_above(th.tau_rel) >= d:
        return STABLE_PASS
    if post.prob_above(th.tau_stuck) <= 1.0 - d:
        return STABLE_FAIL
    if post.prob_above(th.tau_cap) >= d and post.prob_above(th.tau_rel) <= 1.0 - d:
        return STOCHASTIC
    return UNCERTAIN_CRITERION


def profile_rubric(
    runs: Sequence[Mapping[str, float]],
    task: str = "",
    model: str = "",
    *,
    thresholds: Optional[ReliabilityThresholds] = None,
    prior: BetaPrior = JEFFREYS_PRIOR,
    satisfied_threshold: float = 1.0,
    criterion_graph: Optional[CriterionGraph] = None,
) -> RubricProfile:
    """Full per-criterion reliability profile for one (model, task)."""
    th = thresholds or ReliabilityThresholds()
    posts = criterion_posteriors(
        runs, prior=prior, satisfied_threshold=satisfied_threshold
    )
    failing = {
        k for k, p in posts.items()
        if _criterion_label(p, th) in (STABLE_FAIL, STOCHASTIC)
    }
    profiles = []
    for crit in sorted(posts):
        p = posts[crit]
        is_root = None
        if criterion_graph is not None:
            is_root = not bool(set(criterion_graph.parents(crit)) & failing)
        profiles.append(
            CriterionProfile(
                criterion=crit,
                task=task,
                model=model,
                n_runs=p.n_obs,
                n_satisfied=int(round(p.alpha - prior.alpha)),
                posterior=p,
                label=_criterion_label(p, th),
                is_root=is_root,
            )
        )
    return RubricProfile(
        task=task,
        model=model,
        criteria=profiles,
        n_runs=len(runs),
        criterion_graph=criterion_graph,
    )


def root_criterion_profile(profile: RubricProfile) -> dict:
    """Separate root criterion failures from downstream ones.

    Requires the profile to carry a :class:`CriterionGraph`; without one, every
    criterion is its own root and the amplification is 1.0 by construction, which
    the result states rather than implying a finding.
    """
    failing = [c for c in profile.criteria if c.label in (STABLE_FAIL, STOCHASTIC)]
    if profile.criterion_graph is None:
        return {
            "n_failing": len(failing),
            "n_root_failing": len(failing),
            "amplification": 1.0 if failing else float("nan"),
            "roots": [c.criterion for c in failing],
            "note": "no criterion dependency graph supplied; every failing "
                    "criterion is treated as its own root",
        }
    roots = [c.criterion for c in failing if c.is_root]
    return {
        "n_failing": len(failing),
        "n_root_failing": len(roots),
        "amplification": (len(failing) / len(roots)) if roots else float("nan"),
        "roots": roots,
        "downstream": [c.criterion for c in failing if not c.is_root],
        "note": "",
    }


def rubric_matrix(
    profiles: Sequence[RubricProfile],
) -> "object":
    """Tasks x criteria posterior-mean matrix, for the reliability heatmap.

    Returns a DataFrame indexed by ``(model, task)`` with one column per
    criterion. Criteria absent from a task are NaN, which the plotting code
    renders as a gap rather than as a zero.
    """
    import pandas as pd

    rows = {}
    for p in profiles:
        rows[(p.model, p.task)] = {
            c.criterion: c.posterior.mean for c in p.criteria
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["model", "task"])
    return df.sort_index(axis=1)
