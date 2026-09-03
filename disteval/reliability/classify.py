"""Capability, reliability, and the SOLID / RECOVERABLE / STUCK / UNCERTAIN taxonomy.

The taxonomy is inherited from ``disteval.right_tail`` because it is genuinely
useful, but the *definitions* are replaced. The legacy rules were::

    SOLID        max(q) > 0 and max(q) == mean(q)
    RECOVERABLE  max(q) > 0 and max(q) >  mean(q)
    STUCK        max(q) == 0

which are functions of the observed maximum and therefore depend on how many
times the task happened to be run. Under those rules a task with runs ``[0, 1]``
is "RECOVERABLE" with the same confidence as one with ``[0,0,0,0,1,1,1,1]``, and
a task run once is always SOLID or STUCK -- never uncertain.

Posterior definitions
---------------------
Everything is a probability statement about the latent parameter ``p_t``:

    C_t = P(p_t > tau_cap | D_t)     estimated capability
    R_t = P(p_t > tau_rel | D_t)     estimated deployment reliability
    K_t = P(p_t < tau_stuck | D_t)   estimated evidence of no capability

with ``tau_stuck <= tau_cap < tau_rel``. ``tau_cap`` is deliberately low: it asks
"is there real evidence the agent can do this at all", not "is it good at it".
``tau_rel`` is a deployment bar. A confidence level ``delta`` says how much
posterior mass is required before an assertion is made:

    SOLID        C_t >= delta and R_t >= delta
    STUCK        K_t >= delta
    RECOVERABLE  C_t >= delta and R_t <= 1 - delta
                 (evidence it *can*, and evidence it is *not* consistent)
    UNCERTAIN    anything else -- typically too few runs to assert either way

Note that UNCERTAIN is a real, common outcome at n=2-3, and that is the point:
the discrete label never hides the uncertainty, and the continuous scores
``C_t``, ``R_t``, ``K_t`` and the credible interval are always reported alongside
it.

Recoverability
--------------
Three posterior-only estimators are provided; none is asserted to be best, and
:mod:`disteval.experiments` compares them empirically.

``posterior_gap``
    ``C_t * (1 - R_t)``. The literal reading of "demonstrated capability but not
    reliable". Simple and monotone in the right directions, but saturates: any
    task with clear capability and clear unreliability scores ~1 regardless of
    how much reliability is actually missing.

``expected_headroom``
    ``E[ (tau_rel - p_t)^+ * 1{p_t > tau_cap} ]``, i.e. the posterior-expected
    amount of reliability that is missing *given* the agent is capable. This is
    the quantity a reliability-maximising planner would actually want: it is
    largest for tasks that are demonstrably capable and far from the bar, and
    small both for tasks near the bar (little to gain) and for tasks with no
    evidence of capability (the indicator kills them). Computed by quadrature
    against the posterior, so it inherits the uncertainty automatically.

``evidence_weighted``
    ``posterior_gap * n / (n + n0)``. An explicit sample-size discount on top of
    the posterior gap, for when you want ranking to be conservative about
    low-``n`` tasks beyond what the posterior already does. ``n0`` is the number
    of runs at which a task is given half weight.

All three are functions of the posterior alone. Signals derived from
trajectories (failure entropy, intervention distance, neighbourhood distance)
are combined with these in :mod:`disteval.reliability.recoverability`.

A limitation worth stating plainly: when scores are continuous, the latent
parameter is the *mean score*, so a task that scores exactly 0.5 on every run is
labelled RECOVERABLE even though nothing about it is stochastic. That is a real
consequence of collapsing a rubric to one number, not a bug in the classifier.
If the question is specifically about execution reliability, either dichotomise
at a rubric threshold (``posterior_from_scores(..., success_threshold=...)``) or
use the per-criterion model in :mod:`disteval.reliability.rubric`, which
separates "consistently partially wrong" from "intermittently fully wrong".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np

__all__ = [
    "SOLID",
    "RECOVERABLE",
    "STUCK",
    "UNCERTAIN",
    "CATEGORIES",
    "ReliabilityThresholds",
    "TaskDiagnosis",
    "PosteriorLike",
    "diagnose",
    "diagnose_many",
    "posterior_gap",
    "expected_headroom",
    "evidence_weighted_gap",
    "rank_by_recoverability",
]

SOLID = "SOLID"
RECOVERABLE = "RECOVERABLE"
STUCK = "STUCK"
UNCERTAIN = "UNCERTAIN"
CATEGORIES = (SOLID, RECOVERABLE, STUCK, UNCERTAIN)


class PosteriorLike(Protocol):
    """The interface every posterior in this package satisfies."""

    n_obs: int

    @property
    def mean(self) -> float: ...
    @property
    def sd(self) -> float: ...
    @property
    def entropy(self) -> float: ...
    def prob_above(self, threshold: float) -> float: ...
    def credible_interval(self, level: float = 0.95) -> tuple[float, float]: ...
    def quantile(self, q: float) -> float: ...
    def sample(self, size: int, rng=None) -> np.ndarray: ...


@dataclass(frozen=True)
class ReliabilityThresholds:
    """Configurable thresholds for the taxonomy. No universal constants.

    The defaults below are a starting point for binary task success on a
    long-horizon agent benchmark, not a recommendation for every setting:

    * ``tau_cap = 0.15`` -- "there is real evidence the agent can do this".
      Roughly: better than one-in-seven. Lower it for extremely hard suites.
    * ``tau_rel = 0.90`` -- a deployment bar. Raise it toward 0.99 for
      safety-critical use, lower it for research triage.
    * ``tau_stuck = 0.10`` -- below this the agent effectively cannot do the task.
      Note the interaction with sample size: asserting ``P(p < 0.10) >= 0.80``
      needs about 8 consecutive failures under a Jeffreys prior, and asserting
      ``P(p < 0.05) >= 0.80`` would need about 15. Lower thresholds are not free.
    * ``confidence = 0.80`` -- posterior mass needed to make an assertion. With
      n=8 runs this is attainable; with n=2-3 most tasks stay UNCERTAIN, correctly.
    * ``min_runs = 3`` -- fewer runs than this is always UNCERTAIN. Two runs with
      one success would otherwise be labelled RECOVERABLE on the strength of a
      single lucky attempt, which is precisely the inference this module exists
      to prevent.
    """

    tau_cap: float = 0.15
    tau_rel: float = 0.90
    tau_stuck: float = 0.10
    confidence: float = 0.80
    min_runs: int = 3
    credible_level: float = 0.95
    #: Runs at which ``evidence_weighted`` gives a task half weight.
    evidence_n0: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_stuck <= self.tau_cap < self.tau_rel <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= tau_stuck <= tau_cap < tau_rel <= 1"
            )
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0.5, 1)")
        if self.min_runs < 1:
            raise ValueError("min_runs must be >= 1")


# --------------------------------------------------------------------------- #
# Recoverability estimators (posterior-only)                                  #
# --------------------------------------------------------------------------- #
def posterior_gap(capability: float, reliability: float) -> float:
    """``C_t * (1 - R_t)``: capability present, reliability absent."""
    return float(capability * (1.0 - reliability))


def expected_headroom(
    post: PosteriorLike, thresholds: ReliabilityThresholds, n_grid: int = 512
) -> float:
    """``E[(tau_rel - p)^+ * 1{p > tau_cap}]`` by quadrature on the posterior.

    Normalised by ``tau_rel - tau_cap`` so the score lands in [0, 1] and is
    comparable across threshold settings.
    """
    lo, hi = 1e-6, 1 - 1e-6
    grid = np.linspace(lo, hi, n_grid)
    # posterior CDF differences give the probability mass in each cell
    cdf = np.array([1.0 - post.prob_above(x) for x in grid])
    mass = np.diff(cdf)
    mid = 0.5 * (grid[:-1] + grid[1:])
    integrand = np.clip(thresholds.tau_rel - mid, 0.0, None) * (mid > thresholds.tau_cap)
    span = thresholds.tau_rel - thresholds.tau_cap
    return float(np.sum(mass * integrand) / span) if span > 0 else 0.0


def evidence_weighted_gap(
    capability: float, reliability: float, n_obs: int, n0: float = 4.0
) -> float:
    """Posterior gap discounted by an explicit sample-size weight ``n/(n+n0)``."""
    w = n_obs / (n_obs + n0) if n_obs > 0 else 0.0
    return float(posterior_gap(capability, reliability) * w)


# --------------------------------------------------------------------------- #
# Diagnosis                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class TaskDiagnosis:
    """Everything known about one (agent, task) cell after estimation."""

    task: str
    model: str
    n_runs: int
    n_success: float

    # continuous posterior scores -- always reported alongside the label
    capability: float          # C_t = P(p > tau_cap)
    reliability: float         # R_t = P(p > tau_rel)
    stuck_evidence: float      # K_t = P(p < tau_stuck)
    posterior_mean: float
    posterior_sd: float
    ci_lo: float
    ci_hi: float
    posterior_entropy: float

    # recoverability estimators
    recoverability: float          # the configured primary estimator
    recoverability_gap: float
    recoverability_headroom: float
    recoverability_evidence: float

    label: str
    thresholds: ReliabilityThresholds

    # descriptive statistics -- explicitly NOT the latent estimate
    observed_max: float = float("nan")
    observed_mean: float = float("nan")
    domain: Optional[str] = None
    shrinkage: Optional[float] = None
    extras: dict = field(default_factory=dict)

    @property
    def ci_width(self) -> float:
        return self.ci_hi - self.ci_lo

    def to_dict(self) -> dict:
        d = {
            "task": self.task,
            "model": self.model,
            "domain": self.domain,
            "n_runs": self.n_runs,
            "n_success": self.n_success,
            "capability": self.capability,
            "reliability": self.reliability,
            "stuck_evidence": self.stuck_evidence,
            "posterior_mean": self.posterior_mean,
            "posterior_sd": self.posterior_sd,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "ci_width": self.ci_width,
            "posterior_entropy": self.posterior_entropy,
            "recoverability": self.recoverability,
            "recoverability_gap": self.recoverability_gap,
            "recoverability_headroom": self.recoverability_headroom,
            "recoverability_evidence": self.recoverability_evidence,
            "label": self.label,
            "observed_max": self.observed_max,
            "observed_mean": self.observed_mean,
            "shrinkage": self.shrinkage,
        }
        d.update(self.extras)
        return d


def _label(
    capability: float,
    reliability: float,
    stuck_evidence: float,
    n_runs: int,
    th: ReliabilityThresholds,
) -> str:
    if n_runs < th.min_runs:
        return UNCERTAIN
    d = th.confidence
    if capability >= d and reliability >= d:
        return SOLID
    if stuck_evidence >= d:
        return STUCK
    if capability >= d and reliability <= 1.0 - d:
        return RECOVERABLE
    return UNCERTAIN


def diagnose(
    post: PosteriorLike,
    task: str = "",
    model: str = "",
    thresholds: Optional[ReliabilityThresholds] = None,
    *,
    scores: Optional[Sequence[float]] = None,
    domain: Optional[str] = None,
    primary: str = "headroom",
    shrinkage: Optional[float] = None,
) -> TaskDiagnosis:
    """Turn a posterior into a full task diagnosis.

    ``primary`` selects which recoverability estimator populates the
    ``recoverability`` field: ``"gap"``, ``"headroom"`` (default) or
    ``"evidence"``. All three are always computed and reported.
    """
    th = thresholds or ReliabilityThresholds()
    cap = post.prob_above(th.tau_cap)
    rel = post.prob_above(th.tau_rel)
    stuck = 1.0 - post.prob_above(th.tau_stuck)
    lo, hi = post.credible_interval(th.credible_level)

    r_gap = posterior_gap(cap, rel)
    r_head = expected_headroom(post, th)
    r_evid = evidence_weighted_gap(cap, rel, post.n_obs, th.evidence_n0)
    primary_value = {"gap": r_gap, "headroom": r_head, "evidence": r_evid}
    if primary not in primary_value:
        raise ValueError(f"unknown recoverability estimator {primary!r}")

    arr = np.asarray(list(scores), dtype=float) if scores is not None else None
    n_success = float(getattr(post, "n_success", float("nan")))
    if arr is not None and arr.size and not np.isfinite(n_success):
        n_success = float(arr.sum())

    return TaskDiagnosis(
        task=task,
        model=model,
        n_runs=int(post.n_obs),
        n_success=n_success,
        capability=cap,
        reliability=rel,
        stuck_evidence=stuck,
        posterior_mean=post.mean,
        posterior_sd=post.sd,
        ci_lo=lo,
        ci_hi=hi,
        posterior_entropy=post.entropy,
        recoverability=primary_value[primary],
        recoverability_gap=r_gap,
        recoverability_headroom=r_head,
        recoverability_evidence=r_evid,
        label=_label(cap, rel, stuck, int(post.n_obs), th),
        thresholds=th,
        observed_max=float(arr.max()) if arr is not None and arr.size else float("nan"),
        observed_mean=float(arr.mean()) if arr is not None and arr.size else float("nan"),
        domain=domain,
        shrinkage=shrinkage,
    )


def diagnose_many(
    posteriors: dict[tuple[str, str], PosteriorLike],
    thresholds: Optional[ReliabilityThresholds] = None,
    *,
    scores: Optional[dict[tuple[str, str], Sequence[float]]] = None,
    domains: Optional[dict[str, str]] = None,
    primary: str = "headroom",
    shrinkage: Optional[dict[tuple[str, str], float]] = None,
) -> list[TaskDiagnosis]:
    """Diagnose every ``(model, task)`` cell. Keys are ``(model, task)`` tuples."""
    out = []
    for (model, task), post in posteriors.items():
        out.append(
            diagnose(
                post,
                task=task,
                model=model,
                thresholds=thresholds,
                scores=(scores or {}).get((model, task)),
                domain=(domains or {}).get(task),
                primary=primary,
                shrinkage=(shrinkage or {}).get((model, task)),
            )
        )
    return out


def rank_by_recoverability(
    diagnoses: Iterable[TaskDiagnosis],
    estimator: str = "headroom",
    *,
    labels: Optional[Sequence[str]] = (RECOVERABLE,),
) -> list[TaskDiagnosis]:
    """Rank tasks as post-training targets, highest estimated value first.

    ``labels=None`` ranks every task; the default restricts to RECOVERABLE, which
    is the population the research hypothesis is about.
    """
    field_name = {
        "gap": "recoverability_gap",
        "headroom": "recoverability_headroom",
        "evidence": "recoverability_evidence",
        "primary": "recoverability",
    }[estimator]
    items = [d for d in diagnoses if labels is None or d.label in labels]
    return sorted(items, key=lambda d: -getattr(d, field_name))
