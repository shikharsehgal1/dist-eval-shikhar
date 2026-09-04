"""Criterion-level reliability, and the capability--reliability gap.

This is the framework's **primary** estimation path. The task-level estimators in
:mod:`disteval.reliability.posterior` remain available and are what you get when
a benchmark reports only a scalar score, but where per-criterion grading exists
it carries strictly more information and should be used.

Why criterion level
-------------------
A task with ten rubric criteria where nine are always satisfied and one is a coin
flip has the same overall pass rate as one where all ten are half-right. They are
completely different objects: the first has a single localised execution defect,
the second suggests the agent does not really know the task. Collapsing the
rubric to one number destroys that distinction before any analysis begins.

The model
---------
For task ``t`` and criterion ``j``, with ``p_{t,j}`` the latent probability that
criterion ``j`` is satisfied on a single run::

    logit(p_{t,j}) = mu + alpha_t + beta_j + gamma_{d(t)} + eps_{t,j}

fitted by the same crossed random-effects machinery as the task-level model
(:mod:`disteval.reliability.hierarchical`). Partial pooling matters more here
than at task level: a criterion observed on eight runs of one task borrows
strength from the same criterion across other tasks in its family, and from the
task's other criteria.

An empirical-Bayes path (:func:`empirical_bayes_criteria`) is provided for speed
and for the case where the hierarchy is not worth fitting; it pools each
criterion toward a Beta prior fitted across tasks by moments.

The capability--reliability gap
-------------------------------
Define, at a capability threshold ``tau_cap`` and a deployment threshold
``tau_rel``::

    c_{t,j} = P(p_{t,j} > tau_cap | D)      evidence the criterion is within reach
    r_{t,j} = P(p_{t,j} > tau_rel | D)      evidence it is reliably met

    C_t = (1/J) sum_j c_{t,j}               expected fraction within reach
    R_t = (1/J) sum_j r_{t,j}               expected fraction reliably met
    G_t = C_t - R_t                          the capability--reliability gap

``G_t`` in [0, 1] is the fraction of the rubric the agent can demonstrably satisfy
but does not satisfy dependably. It is a function of posteriors only: no observed
maximum appears anywhere, so it does not grow with the number of runs, and a
single lucky success on one criterion moves it only as far as one run of evidence
warrants.

**This metric is not claimed to be novel.** A gap between demonstrated and
dependable performance is the same idea as the pass@k / pass^k gap (tau-bench),
as "capability overhang" in evaluation practice, and as the difference between
best-of-n and single-sample performance. What is stated here is a specific
posterior definition of it at criterion granularity, and the open question is
whether it carries *incremental* predictive value for training-data selection
over difficulty and learning progress -- see :mod:`disteval.selection`.

Criterion-level and task-level capability are different things
---------------------------------------------------------------
``C_t`` asks whether each criterion is *individually* within reach. A task can
score high on it while the agent has **never once satisfied every criterion
simultaneously** -- each piece is achievable, the conjunction is not. That is a
real and useful finding, not an artefact: a task-level score reports it only as
"always fails", which is the less actionable description.

Because the two readings can diverge, :attr:`GapProfile.joint_reliability`
reports ``P(all criteria satisfied on one run)`` from the observed all-satisfied
count, and :attr:`GapProfile.joint_capability` the evidence that the task is ever
fully solvable. Both are estimated directly rather than by multiplying the
per-criterion posteriors, because criteria are not independent and the product is
badly biased downward. Selection strategies that need task-level evidence of
success should read these; ``G_t`` alone will happily rank a never-fully-solved
task highly.

Reading the gap
---------------
``G_t`` alone does not say whether the shortfall is one unstable criterion or ten
mediocre ones, and those call for different interventions. :attr:`GapProfile`
therefore also reports:

* ``gap_concentration`` -- share of the total gap contributed by its largest
  single criterion. Near 1 means one criterion accounts for the shortfall.
* ``dominant_criterion`` -- which one.
* ``n_unstable`` -- criteria with evidence of capability but not of reliability.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .classify import PosteriorLike, ReliabilityThresholds
from .hierarchical import HierarchicalSpec, fit_hierarchical
from .posterior import JEFFREYS_PRIOR, BetaPrior, TaskPosterior, binary_posterior

__all__ = [
    "CriterionEstimate",
    "GapProfile",
    "empirical_bayes_criteria",
    "fit_criterion_model",
    "gap_profile",
    "gap_profiles",
    "capability_reliability_gap",
]


@dataclass
class CriterionEstimate:
    """Posterior for one (task, criterion) cell."""

    task: str
    criterion: str
    n_runs: int
    n_satisfied: float
    posterior: PosteriorLike
    capability: float          # c_{t,j}
    reliability: float         # r_{t,j}
    shrinkage: Optional[float] = None
    domain: Optional[str] = None

    @property
    def gap(self) -> float:
        """``c - r`` for this criterion: its contribution to the task gap."""
        return float(self.capability - self.reliability)

    @property
    def observed_rate(self) -> float:
        return self.n_satisfied / self.n_runs if self.n_runs else float("nan")

    def to_dict(self) -> dict:
        lo, hi = self.posterior.credible_interval()
        return {
            "task": self.task,
            "criterion": self.criterion,
            "domain": self.domain,
            "n_runs": self.n_runs,
            "n_satisfied": self.n_satisfied,
            "observed_rate": self.observed_rate,
            "posterior_mean": self.posterior.mean,
            "posterior_sd": self.posterior.sd,
            "ci_lo": lo,
            "ci_hi": hi,
            "capability": self.capability,
            "reliability": self.reliability,
            "gap": self.gap,
            "shrinkage": self.shrinkage,
        }


@dataclass
class GapProfile:
    """The capability--reliability gap for one task, with its structure."""

    task: str
    model: str
    criteria: list[CriterionEstimate]
    capability: float          # C_t
    reliability: float         # R_t
    thresholds: ReliabilityThresholds
    domain: Optional[str] = None
    n_runs: int = 0
    #: Posterior for "every criterion satisfied on a single run", estimated
    #: directly from the observed all-satisfied count rather than by multiplying
    #: the per-criterion posteriors -- criteria are not independent and the
    #: product would be badly biased downward.
    joint: Optional[PosteriorLike] = None
    meta: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        """``G_t = C_t - R_t``."""
        return float(self.capability - self.reliability)

    @property
    def n_criteria(self) -> int:
        return len(self.criteria)

    @property
    def joint_reliability(self) -> float:
        """``P(all criteria satisfied on one run)`` -- task-level reliability.

        Reported alongside the criterion aggregates because they can diverge
        sharply, and the divergence is the interesting part. A task can have high
        ``C_t`` (most criteria are individually within reach) while never once
        satisfying all of them together, which is precisely the pattern
        criterion-level analysis exists to surface and which a task-level score
        reports only as "always fails".
        """
        return self.joint.mean if self.joint is not None else float("nan")

    @property
    def joint_capability(self) -> float:
        """``P(p_joint > tau_cap)``: evidence the task is ever fully solvable."""
        if self.joint is None:
            return float("nan")
        return self.joint.prob_above(self.thresholds.tau_cap)

    @property
    def gap_concentration(self) -> float:
        """Share of the total gap contributed by its single largest criterion.

        Near 1: one criterion accounts for the whole shortfall, which is the case
        a targeted intervention can plausibly address. Near 1/J: the shortfall is
        spread evenly across the rubric. NaN when there is no gap to apportion.
        """
        gaps = [c.gap for c in self.criteria]
        total = float(sum(gaps))
        if not gaps or total <= 1e-12:
            return float("nan")
        return float(max(gaps) / total)

    @property
    def dominant_criterion(self) -> Optional[str]:
        if not self.criteria:
            return None
        return max(self.criteria, key=lambda c: c.gap).criterion

    def n_unstable(self, delta: Optional[float] = None) -> int:
        """Criteria with evidence of capability but not of reliability."""
        d = delta if delta is not None else self.thresholds.confidence
        return sum(1 for c in self.criteria
                   if c.capability >= d and c.reliability <= 1 - d)

    def n_never(self, delta: Optional[float] = None) -> int:
        """Criteria with no evidence of capability."""
        d = delta if delta is not None else self.thresholds.confidence
        return sum(1 for c in self.criteria if c.capability <= 1 - d)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "model": self.model,
            "domain": self.domain,
            "n_runs": self.n_runs,
            "n_criteria": self.n_criteria,
            "capability": self.capability,
            "reliability": self.reliability,
            "gap": self.gap,
            "gap_concentration": self.gap_concentration,
            "dominant_criterion": self.dominant_criterion,
            "n_unstable_criteria": self.n_unstable(),
            "n_never_satisfied_criteria": self.n_never(),
            "joint_reliability": self.joint_reliability,
            "joint_capability": self.joint_capability,
            **self.meta,
        }


# --------------------------------------------------------------------------- #
# Estimation                                                                  #
# --------------------------------------------------------------------------- #
def _counts(
    runs_by_task: Mapping[str, Sequence[Mapping[str, float]]],
    satisfied_threshold: float,
) -> dict[tuple[str, str], list[float]]:
    """(task, criterion) -> [n_observed, n_satisfied]."""
    out: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for task, runs in runs_by_task.items():
        for r in runs:
            for crit, val in (r or {}).items():
                cell = out[(task, str(crit))]
                cell[0] += 1.0
                cell[1] += float(float(val) >= satisfied_threshold)
    return dict(out)


def empirical_bayes_criteria(
    runs_by_task: Mapping[str, Sequence[Mapping[str, float]]],
    *,
    satisfied_threshold: float = 1.0,
    pool_by: str = "criterion",
) -> dict[tuple[str, str], TaskPosterior]:
    """Empirical-Bayes criterion posteriors, pooled by criterion (or globally).

    The fast path. A Beta prior is fitted by method of moments across the
    population -- by default, across all tasks *for each criterion*, so
    "did it verify its work" borrows strength from every task where that
    criterion appears. The moment fit subtracts within-cell binomial variance
    before matching, so the prior reflects genuine between-task spread rather
    than few-runs noise (see :func:`disteval.shrinkage.fit_beta_binomial_mom`).

    Use this when the hierarchy is not worth fitting; use
    :func:`fit_criterion_model` when you also want domain effects, task effects,
    and honest joint uncertainty.
    """
    from ..shrinkage import fit_beta_binomial_mom

    cells = _counts(runs_by_task, satisfied_threshold)
    if not cells:
        return {}
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (task, crit) in cells:
        key = crit if pool_by == "criterion" else ("task" if pool_by == "task" else "_all")
        groups[crit if pool_by == "criterion" else (task if pool_by == "task" else "_all")].append((task, crit))

    out: dict[tuple[str, str], TaskPosterior] = {}
    for _key, members in groups.items():
        n = np.array([cells[m][0] for m in members], dtype=float)
        s = np.array([cells[m][1] for m in members], dtype=float)
        if n.size >= 2 and np.all(n > 0):
            prior_d = fit_beta_binomial_mom(s, n)
            prior = BetaPrior(max(prior_d["alpha"], 1e-3), max(prior_d["beta"], 1e-3))
        else:
            prior = JEFFREYS_PRIOR
        for m, si, ni in zip(members, s, n):
            out[m] = binary_posterior(int(round(si)), int(round(ni)), prior)
    return out


def fit_criterion_model(
    runs_by_task: Mapping[str, Sequence[Mapping[str, float]]],
    *,
    domains: Optional[Mapping[str, str]] = None,
    satisfied_threshold: float = 1.0,
    spec: Optional[HierarchicalSpec] = None,
    backend: str = "laplace",
    seed: int = 0,
):
    """Fit ``logit(p_{t,j}) = mu + alpha_t + beta_j + gamma_d + eps_{t,j}``.

    Returns the :class:`~disteval.reliability.hierarchical.HierarchicalFit`, whose
    cells are keyed ``(task, criterion)``. The task takes the "model" slot and the
    criterion the "task" slot of the underlying crossed model, which is exactly
    the same algebra under different names.

    The interaction term ``eps_{t,j}`` is what preserves "this criterion is
    specifically unstable *on this task*" -- without it, criterion behaviour would
    be forced through additive main effects and the very cells this framework
    cares about would be shrunk away.
    """
    cells = _counts(runs_by_task, satisfied_threshold)
    if not cells:
        raise ValueError("no criterion observations to fit")
    tasks = [t for (t, _c) in cells]
    crits = [c for (_t, c) in cells]
    succ = [cells[k][1] for k in cells]
    trials = [cells[k][0] for k in cells]
    doms = [(domains or {}).get(t, "_all") for t in tasks]
    return fit_hierarchical(
        tasks, crits, succ, trials, doms, spec=spec, backend=backend, seed=seed
    )


def gap_profile(
    task: str,
    runs: Sequence[Mapping[str, float]],
    *,
    model: str = "policy",
    thresholds: Optional[ReliabilityThresholds] = None,
    posteriors: Optional[Mapping[str, PosteriorLike]] = None,
    satisfied_threshold: float = 1.0,
    domain: Optional[str] = None,
    shrinkage: Optional[Mapping[str, float]] = None,
) -> GapProfile:
    """Capability--reliability gap for one task from its per-criterion runs.

    ``posteriors`` lets a pooled fit be supplied; without it, each criterion gets
    an independent Jeffreys-prior Beta posterior.
    """
    th = thresholds or ReliabilityThresholds()
    counts = _counts({task: runs}, satisfied_threshold)
    estimates: list[CriterionEstimate] = []
    for (t, crit), (n, s) in sorted(counts.items()):
        post = (posteriors or {}).get(crit)
        if post is None:
            post = binary_posterior(int(round(s)), int(round(n)), JEFFREYS_PRIOR)
        estimates.append(
            CriterionEstimate(
                task=t,
                criterion=crit,
                n_runs=int(round(n)),
                n_satisfied=float(s),
                posterior=post,
                capability=post.prob_above(th.tau_cap),
                reliability=post.prob_above(th.tau_rel),
                shrinkage=(shrinkage or {}).get(crit),
                domain=domain,
            )
        )
    if estimates:
        C = float(np.mean([e.capability for e in estimates]))
        R = float(np.mean([e.reliability for e in estimates]))
    else:
        C = R = float("nan")

    n_all = sum(
        1 for r in runs
        if r and all(float(v) >= satisfied_threshold for v in r.values())
    )
    joint = binary_posterior(n_all, len(runs), JEFFREYS_PRIOR) if runs else None

    return GapProfile(
        task=task,
        model=model,
        criteria=estimates,
        capability=C,
        reliability=R,
        thresholds=th,
        domain=domain,
        n_runs=len(runs),
        joint=joint,
    )


def gap_profiles(
    runs_by_task: Mapping[str, Sequence[Mapping[str, float]]],
    *,
    model: str = "policy",
    thresholds: Optional[ReliabilityThresholds] = None,
    domains: Optional[Mapping[str, str]] = None,
    satisfied_threshold: float = 1.0,
    estimator: str = "hierarchical",
    backend: str = "laplace",
    seed: int = 0,
) -> list[GapProfile]:
    """Gap profiles for every task, sharing one pooled criterion model.

    ``estimator``:

    * ``"hierarchical"`` (default) -- fit the crossed random-effects model. Cells
      borrow strength across tasks and criteria; uncertainty is joint.
    * ``"empirical_bayes"`` -- moment-fitted Beta prior per criterion. Faster,
      and adequate when there are too few tasks for the hierarchy.
    * ``"independent"`` -- no pooling. The honest baseline, and what the
      ablations compare pooling against.
    """
    th = thresholds or ReliabilityThresholds()
    posts: dict[str, dict[str, PosteriorLike]] = defaultdict(dict)
    shrink: dict[str, dict[str, float]] = defaultdict(dict)

    if estimator == "hierarchical":
        n_cells = len(_counts(runs_by_task, satisfied_threshold))
        if n_cells >= 4:
            fit = fit_criterion_model(
                runs_by_task, domains=domains,
                satisfied_threshold=satisfied_threshold, backend=backend, seed=seed,
            )
            for (task, crit), cell in fit.cells.items():
                posts[task][crit] = cell
                shrink[task][crit] = fit.shrinkage(task, crit)
        else:
            estimator = "independent"
    if estimator == "empirical_bayes":
        for (task, crit), p in empirical_bayes_criteria(
            runs_by_task, satisfied_threshold=satisfied_threshold
        ).items():
            posts[task][crit] = p

    return [
        gap_profile(
            task, runs, model=model, thresholds=th,
            posteriors=posts.get(task) or None,
            satisfied_threshold=satisfied_threshold,
            domain=(domains or {}).get(task),
            shrinkage=shrink.get(task) or None,
        )
        for task, runs in sorted(runs_by_task.items())
    ]


def capability_reliability_gap(
    runs_by_task: Mapping[str, Sequence[Mapping[str, float]]], **kwargs
) -> dict[str, float]:
    """Convenience: ``{task: G_t}``."""
    return {p.task: p.gap for p in gap_profiles(runs_by_task, **kwargs)}
