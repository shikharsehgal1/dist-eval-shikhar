"""Choosing which tasks to build training data from.

This is where the research hypothesis lives. Everything upstream estimates task
properties; this module turns those estimates into a training set, and the
experiment downstream measures whether the choice mattered.

    HYPOTHESIS: within-task failure structure adds predictive value for
    training-data selection beyond difficulty and learning progress.

Note carefully what is *not* claimed. It is not claimed that the
capability--reliability gap is a novel metric -- it is a posterior restatement of
a familiar demonstrated-versus-dependable comparison. It is not claimed that
tasks with intermediate success rates are inherently more trainable; that is a
plausible intuition with obvious failure modes (such tasks may be nearly solved
with little headroom, or fail for irreducibly stochastic reasons), and treating
it as established would be exactly the error this framework is built to avoid.

The claim under test is narrower and falsifiable: that knowing *how* a task fails
-- which criterion, how consistently, how far the failed runs sit from the
successful ones -- predicts training value over and above knowing *how hard* it is
and *how fast it is currently improving*. The comparison that settles it is
``gap_plus_structure`` against ``difficulty`` and ``learning_progress``, not
against random, which is too weak a baseline to support the claim.

The six curriculum strategies
-----------------------------
``uniform``                     uniform random. The control.
``difficulty``                  lowest estimated performance first.
``uncertainty``                 largest posterior sd first.
``learning_progress``           PAC-style: observed progress where a training
                                history exists, else the learnability proxy
                                ``E[p(1-p)]``.
``capability_reliability_gap``  largest criterion-level ``G_t = C_t - R_t``.
``gap_plus_structure``          ``G_t`` weighted by within-task failure structure.

The research question is whether the last one beats ``difficulty`` and
``learning_progress`` -- an incremental-validity question, not a comparison
against random.

Additional arms (legacy names retained, still functional)
---------------------------------------------------------
``random``              alias behaviour of ``uniform``
``hardest``             alias behaviour of ``difficulty``
``lowest_mean``         lowest observed mean score (no posterior); shrinkage ablation
``highest_variance``    highest observed run-to-run variance
``success_failure``     any task with both a success and a failure, at random
``recoverability``      legacy task-level recoverability score
``uncertainty_aware``   legacy recoverability discounted by uncertainty
``oracle``              highest true training benefit (simulation only, upper bound)

Fair comparison
---------------
Every selector returns the same number of *tasks*, and pair construction then
caps pairs per task, so the strategies are compared at equal data volume rather
than one of them silently getting more examples. ``SelectionResult`` records the
requested and delivered counts, and shortfalls are reported rather than
backfilled from another strategy -- if recoverability selection can only find 12
qualifying tasks when asked for 50, that is a finding about the method, not
something to paper over.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from ..reliability.classify import RECOVERABLE, TaskDiagnosis

__all__ = [
    "SelectionResult",
    "DataSelector",
    "RandomSelector",
    "HardestSelector",
    "LowestMeanSelector",
    "HighestVarianceSelector",
    "SuccessFailureSelector",
    "RecoverabilitySelector",
    "UncertaintyAwareSelector",
    "OracleSelector",
    "SELECTORS",
    "make_selector",
    "UniformSelector",
    "DifficultySelector",
    "UncertaintySelector",
    "LearningProgressSelector",
    "CapabilityReliabilityGapSelector",
    "GapStructureSelector",
    "CURRICULUM_STRATEGIES",
]


@dataclass
class SelectionResult:
    """Selected tasks plus the accounting needed to compare strategies fairly."""

    strategy: str
    tasks: list[str]
    requested: int
    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    n_eligible: int = 0
    note: str = ""
    params: dict = field(default_factory=dict)

    @property
    def delivered(self) -> int:
        return len(self.tasks)

    @property
    def shortfall(self) -> int:
        return max(self.requested - self.delivered, 0)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "requested": self.requested,
            "delivered": self.delivered,
            "shortfall": self.shortfall,
            "n_eligible": self.n_eligible,
            "tasks": list(self.tasks),
            "note": self.note,
            **{f"param_{k}": v for k, v in self.params.items()},
        }


class DataSelector(ABC):
    """Plugin interface: diagnoses in, chosen task ids out."""

    name = "selector"
    #: Whether the selector needs matched success and failure trajectories.
    requires_pairs = False

    def __init__(self, seed: int = 0, **params):
        self.seed = seed
        self.params = params

    @abstractmethod
    def _rank(
        self, diagnoses: Sequence[TaskDiagnosis], rng: np.random.Generator
    ) -> list[tuple[str, float, str]]:
        """Return ``(task, score, reason)`` triples, best first."""

    def eligible(self, d: TaskDiagnosis) -> bool:
        """Whether a task can be used at all. Overridden by pair-requiring selectors."""
        return d.n_runs > 0

    def select(self, diagnoses: Sequence[TaskDiagnosis], n: int) -> SelectionResult:
        rng = np.random.default_rng(self.seed)
        pool = [d for d in diagnoses if self.eligible(d)]
        ranked = self._rank(pool, rng)
        chosen = ranked[:n]
        note = ""
        if len(chosen) < n:
            note = (
                f"only {len(chosen)} of {n} requested tasks qualify under "
                f"{self.name!r} ({len(pool)} eligible of {len(diagnoses)}); "
                "the shortfall is reported rather than backfilled"
            )
        return SelectionResult(
            strategy=self.name,
            tasks=[t for t, _, _ in chosen],
            requested=n,
            scores={t: s for t, s, _ in chosen},
            reasons={t: r for t, _, r in chosen},
            n_eligible=len(pool),
            note=note,
            params=dict(self.params),
        )


class RandomSelector(DataSelector):
    """Uniform random. The baseline that is hardest to beat by accident."""

    name = "random"

    def _rank(self, diagnoses, rng):
        order = rng.permutation(len(diagnoses))
        return [
            (diagnoses[i].task, 0.0, "selected uniformly at random") for i in order
        ]


class HardestSelector(DataSelector):
    """Lowest posterior mean first: train on what the agent is worst at.

    The obvious curriculum heuristic, and a real competitor. Its weakness is the
    one the hypothesis targets: the hardest tasks are disproportionately ones the
    agent has never solved, so there is no successful trajectory to contrast
    against and preference training has nothing to learn from.
    """

    name = "hardest"

    def _rank(self, diagnoses, rng):
        return [
            (d.task, -d.posterior_mean,
             f"posterior mean {d.posterior_mean:.3f} (lowest first)")
            for d in sorted(diagnoses, key=lambda d: d.posterior_mean)
        ]


class LowestMeanSelector(DataSelector):
    """Lowest *observed* mean score first. The no-posterior version of `hardest`.

    Included as an ablation: it isolates how much the Bayesian machinery
    contributes over the raw sample mean.
    """

    name = "lowest_mean"

    def _rank(self, diagnoses, rng):
        keyed = sorted(
            diagnoses,
            key=lambda d: (d.observed_mean if np.isfinite(d.observed_mean) else 1e9),
        )
        return [
            (d.task, -float(d.observed_mean), f"observed mean {d.observed_mean:.3f}")
            for d in keyed
        ]


class HighestVarianceSelector(DataSelector):
    """Highest run-to-run variance first.

    The most important baseline in the set. It captures "inconsistent" directly
    from the observed scores with no Bayesian apparatus at all. If recoverability
    selection cannot beat this, the posterior machinery is not earning its place.
    """

    name = "highest_variance"

    def _rank(self, diagnoses, rng):
        def var(d: TaskDiagnosis) -> float:
            p = d.n_success / d.n_runs if d.n_runs else 0.0
            return float(p * (1.0 - p))

        return [
            (d.task, var(d), f"outcome variance {var(d):.3f} (highest first)")
            for d in sorted(diagnoses, key=var, reverse=True)
        ]


class SuccessFailureSelector(DataSelector):
    """Any task with at least one success and one failure, chosen at random.

    This is the generic "make preference pairs where you can" strategy and is the
    closest baseline to existing trajectory-preference work. The hypothesis is
    specifically that *ranking within* this pool by estimated recoverability
    beats sampling from it uniformly -- so this is the baseline that isolates the
    contribution being claimed.
    """

    name = "success_failure"
    requires_pairs = True

    def eligible(self, d):
        return d.n_runs >= 2 and 0 < d.n_success < d.n_runs

    def _rank(self, diagnoses, rng):
        order = rng.permutation(len(diagnoses))
        return [
            (diagnoses[i].task, 0.0,
             f"has {diagnoses[i].n_success:.0f} successes and "
             f"{diagnoses[i].n_runs - diagnoses[i].n_success:.0f} failures")
            for i in order
        ]


class RecoverabilitySelector(DataSelector):
    """Highest recoverability first. The method under test.

    ``estimator`` picks which recoverability score to rank by, and
    ``restrict_to_recoverable`` decides whether to require the RECOVERABLE label
    or merely rank by the continuous score. Both are exposed because they are
    separate claims: "the label is useful" and "the score is useful" can come
    apart, and the ablations test them independently.
    """

    name = "recoverability"
    requires_pairs = True

    def __init__(
        self,
        seed: int = 0,
        estimator: str = "headroom",
        restrict_to_recoverable: bool = False,
        require_pairs: bool = True,
        **params,
    ):
        super().__init__(seed, estimator=estimator,
                         restrict_to_recoverable=restrict_to_recoverable,
                         require_pairs=require_pairs, **params)
        self.estimator = estimator
        self.restrict = restrict_to_recoverable
        self.require_pairs = require_pairs

    _FIELDS = {
        "gap": "recoverability_gap",
        "headroom": "recoverability_headroom",
        "evidence": "recoverability_evidence",
        "primary": "recoverability",
    }

    def eligible(self, d):
        if self.require_pairs and not (d.n_runs >= 2 and 0 < d.n_success < d.n_runs):
            return False
        if self.restrict and d.label != RECOVERABLE:
            return False
        return d.n_runs > 0

    def _rank(self, diagnoses, rng):
        f = self._FIELDS[self.estimator]
        keyed = sorted(diagnoses, key=lambda d: -getattr(d, f))
        return [
            (d.task, float(getattr(d, f)),
             f"{self.estimator} recoverability {getattr(d, f):.3f}, "
             f"C={d.capability:.2f}, R={d.reliability:.2f}, label={d.label}")
            for d in keyed
        ]


class UncertaintyAwareSelector(RecoverabilitySelector):
    """Recoverability discounted by posterior uncertainty.

    Multiplies the recoverability score by ``1 - 2*posterior_sd``, so a task whose
    high score rests on two runs is ranked below one with the same score and
    eight runs. This is the direct test of whether uncertainty correction helps
    or merely adds a knob.
    """

    name = "uncertainty_aware"

    def _rank(self, diagnoses, rng):
        f = self._FIELDS[self.estimator]

        def score(d: TaskDiagnosis) -> float:
            w = float(np.clip(1.0 - 2.0 * d.posterior_sd, 0.0, 1.0))
            return float(getattr(d, f)) * w

        keyed = sorted(diagnoses, key=lambda d: -score(d))
        return [
            (d.task, score(d),
             f"{self.estimator} {getattr(d, f):.3f} x evidence weight "
             f"{np.clip(1 - 2*d.posterior_sd, 0, 1):.3f} (n={d.n_runs})")
            for d in keyed
        ]


class OracleSelector(DataSelector):
    """Ranks by *true* training benefit. Simulation only; an upper bound.

    Reported alongside the real selectors so the gap between the best feasible
    method and the ceiling is visible. A recoverability selector that captures
    80% of the oracle's benefit is a strong result; one that captures 15% is not,
    even if it beats random.
    """

    name = "oracle"

    def __init__(self, benefit: Mapping[str, float], seed: int = 0, **params):
        super().__init__(seed, **params)
        self.benefit = dict(benefit)

    def eligible(self, d):
        return d.task in self.benefit

    def _rank(self, diagnoses, rng):
        keyed = sorted(diagnoses, key=lambda d: -self.benefit[d.task])
        return [
            (d.task, float(self.benefit[d.task]),
             f"true training benefit {self.benefit[d.task]:.3f}")
            for d in keyed
        ]


SELECTORS: dict[str, type] = {
    "random": RandomSelector,
    "hardest": HardestSelector,
    "lowest_mean": LowestMeanSelector,
    "highest_variance": HighestVarianceSelector,
    "success_failure": SuccessFailureSelector,
    "recoverability": RecoverabilitySelector,
    "uncertainty_aware": UncertaintyAwareSelector,
    "oracle": OracleSelector,
}


def make_selector(name: str, **kwargs) -> DataSelector:
    """Construct a selector by name, for the config-driven experiment runner."""
    if name not in SELECTORS:
        raise ValueError(f"unknown selector {name!r}; have {sorted(SELECTORS)}")
    return SELECTORS[name](**kwargs)


# --------------------------------------------------------------------------- #
# The six curriculum baselines of the current methodology                     #
# --------------------------------------------------------------------------- #
# The selectors above remain available and unchanged. The six below are the set
# the experiment config and the ablations are built around, and they are the ones
# a comparison should report. `uniform` and `difficulty` are new names for the
# existing `random` and `hardest` behaviours; the old names still work.


class UniformSelector(RandomSelector):
    """Uniform random over eligible tasks. The control condition.

    Renamed from ``random`` for the curriculum vocabulary; identical behaviour.
    """

    name = "uniform"


class DifficultySelector(HardestSelector):
    """Lowest estimated performance first.

    The standard difficulty-based curriculum, and the first thing any new
    selection signal has to beat. Renamed from ``hardest``; identical behaviour.
    """

    name = "difficulty"


class UncertaintySelector(DataSelector):
    """Most uncertain tasks first.

    Ranks by posterior standard deviation -- where the estimate is least settled.
    This is a genuinely different hypothesis from difficulty or gap: it says the
    value of a task is in what we do not yet know about it. It is also the
    baseline most likely to be confounded with low sample size, which is why the
    result table reports run counts alongside it.
    """

    name = "uncertainty"

    def _rank(self, diagnoses, rng):
        return [
            (d.task, float(d.posterior_sd),
             f"posterior sd {d.posterior_sd:.3f} (n={d.n_runs}, CI width "
             f"{d.ci_width:.3f})")
            for d in sorted(diagnoses, key=lambda d: -d.posterior_sd)
        ]


class LearningProgressSelector(DataSelector):
    """PAC-style learning progress: where is performance still moving?

    Two modes, because what is computable depends on what was logged.

    **With a history** (``history={task: [score_at_checkpoint, ...]}``) this is
    absolute learning progress: the magnitude of the change between an earlier
    and a recent window,

        ALP_t = | mean(recent w scores) - mean(previous w scores) |

    which is the standard automatic-curriculum signal (Oudeyer & Kaplan's
    competence progress; Graves et al. 2017 for the bandit formulation). It is
    the right measure when it is available, because it observes progress rather
    than predicting it.

    **Without a history** -- the usual case for a one-shot evaluation -- progress
    cannot be observed, so this falls back to a *learnability* proxy:

        L_t = E[p_t (1 - p_t)]

    the posterior-expected outcome variance, available in closed form from a Beta
    posterior as ``E[p] - E[p^2]``. This is the Fisher information of a Bernoulli
    up to a constant, and it is the standard "zone of proximal development"
    quantity: maximised where the outcome is most informative, near p = 0.5,
    and vanishing where the task is already solved or hopeless.

    **The fallback measures potential, not progress**, and the two are not the
    same thing -- a task can be maximally informative and still be one the policy
    cannot move. ``used_history`` is recorded on the result so a reader always
    knows which was computed.
    """

    name = "learning_progress"

    def __init__(
        self,
        seed: int = 0,
        history: Optional[Mapping[str, Sequence[float]]] = None,
        window: int = 3,
        **params,
    ):
        super().__init__(seed, window=window, **params)
        self.history = {k: list(v) for k, v in (history or {}).items()}
        self.window = int(window)

    def _alp(self, task: str) -> Optional[float]:
        h = self.history.get(task)
        if not h or len(h) < 2:
            return None
        w = max(1, min(self.window, len(h) // 2))
        recent = float(np.mean(h[-w:]))
        earlier = float(np.mean(h[-2 * w:-w]))
        return abs(recent - earlier)

    @staticmethod
    def _learnability(d: TaskDiagnosis) -> float:
        """``E[p(1-p)]`` under the posterior: ``E[p] - (Var[p] + E[p]^2)``."""
        m, s = d.posterior_mean, d.posterior_sd
        return float(max(m - (s * s + m * m), 0.0))

    def _rank(self, diagnoses, rng):
        out = []
        for d in diagnoses:
            alp = self._alp(d.task)
            if alp is not None:
                out.append((d.task, float(alp),
                            f"absolute learning progress {alp:.3f} over the last "
                            f"{self.window} checkpoints"))
            else:
                v = self._learnability(d)
                out.append((d.task, v,
                            f"learnability proxy E[p(1-p)]={v:.3f} (no training "
                            f"history; this is potential, not observed progress)"))
        return sorted(out, key=lambda x: (-x[1], x[0]))

    def select(self, diagnoses, n):
        res = super().select(diagnoses, n)
        used = sum(1 for t in res.tasks if self._alp(t) is not None)
        res.params["used_history"] = used > 0
        res.params["n_with_history"] = used
        if used == 0 and res.tasks:
            res.note = (
                (res.note + " ") if res.note else ""
            ) + (
                "no training history was supplied, so this ranked by the "
                "learnability proxy E[p(1-p)] rather than by observed progress"
            ).strip()
        return res


class CapabilityReliabilityGapSelector(DataSelector):
    """Rank by the criterion-level capability--reliability gap ``G_t``.

    Requires gap profiles from :mod:`disteval.reliability.criterion`, supplied as
    ``gap={task: GapProfile}`` or ``gap={task: float}``. Falls back to the
    task-level posterior gap when no criterion-level profile is available for a
    task, and records how many tasks used the fallback -- the two are not the
    same quantity and mixing them silently would be misleading.

    ``require_joint_capability`` additionally demands posterior evidence that the
    task is *ever* fully solved. This matters: criterion-level ``G_t`` is high for
    a task whose criteria are each individually within reach but never satisfied
    together, and whether such a task belongs in a curriculum is exactly the sort
    of thing the ablation should decide rather than the default assume.
    """

    name = "capability_reliability_gap"

    def __init__(
        self,
        seed: int = 0,
        gap: Optional[Mapping[str, object]] = None,
        require_joint_capability: Optional[float] = None,
        **params,
    ):
        super().__init__(seed, require_joint_capability=require_joint_capability, **params)
        self.gap = dict(gap or {})
        self.require_joint = require_joint_capability

    def _value(self, d: TaskDiagnosis) -> tuple[float, str, bool]:
        g = self.gap.get(d.task)
        if g is None:
            fallback = float(d.capability * (1.0 - d.reliability))
            return fallback, "task-level posterior gap (no criterion profile)", True
        if isinstance(g, (int, float)):
            return float(g), "criterion-level gap G_t", False
        return (
            float(getattr(g, "gap", float("nan"))),
            f"criterion-level gap G_t (C={g.capability:.2f}, R={g.reliability:.2f}, "
            f"dominant={g.dominant_criterion})",
            False,
        )

    def eligible(self, d):
        if not super().eligible(d):
            return False
        if self.require_joint is not None:
            g = self.gap.get(d.task)
            jc = getattr(g, "joint_capability", None) if g is not None else None
            if jc is not None and jc < self.require_joint:
                return False
        return True

    def _rank(self, diagnoses, rng):
        rows = []
        for d in diagnoses:
            v, why, _ = self._value(d)
            if np.isfinite(v):
                rows.append((d.task, v, f"{why}: {v:.3f}"))
        return sorted(rows, key=lambda x: (-x[1], x[0]))

    def select(self, diagnoses, n):
        res = super().select(diagnoses, n)
        fell_back = sum(1 for t in res.tasks if self._value_by_task(t, diagnoses))
        res.params["n_task_level_fallback"] = fell_back
        if fell_back:
            res.note = ((res.note + " ") if res.note else "") + (
                f"{fell_back} of {res.delivered} selected tasks had no "
                "criterion-level profile and were ranked by the task-level "
                "posterior gap instead"
            )
        return res

    def _value_by_task(self, task: str, diagnoses) -> bool:
        for d in diagnoses:
            if d.task == task:
                return self._value(d)[2]
        return False


class GapStructureSelector(CapabilityReliabilityGapSelector):
    """Capability--reliability gap combined with within-task failure structure.

    This is the selector that carries the actual research question. It ranks by

        score_t = G_t * (w_g + w_s * S_t)

    where ``S_t`` in [0, 1] summarises the *structure* of the task's failures:
    how concentrated its failure modes are, how close its failed runs sit to its
    successful ones in trajectory space, and how small an intervention separates
    them. Structure is supplied as ``structure={task: {...}}`` with any of the
    keys ``failure_concentration``, ``neighbourhood``, ``intervention``,
    ``gap_concentration``; whichever are present are averaged.

    The hypothesis this operationalises is deliberately narrow:

        **Does within-task failure structure add predictive value for
        training-data selection beyond difficulty and learning progress?**

    That is an incremental-validity question, and it is answered by comparing
    this selector against ``difficulty`` and ``learning_progress`` -- not against
    random alone, which is too weak a comparison to support the claim. It is not
    assumed that structured failures are more trainable; the ablation exists to
    find out, and the simulator includes worlds in which they are not.
    """

    name = "gap_plus_structure"

    #: Which structure signals to average, when present.
    STRUCTURE_KEYS = (
        "failure_concentration", "neighbourhood", "intervention", "gap_concentration",
    )

    def __init__(
        self,
        seed: int = 0,
        gap: Optional[Mapping[str, object]] = None,
        structure: Optional[Mapping[str, Mapping[str, float]]] = None,
        gap_weight: float = 0.5,
        structure_weight: float = 0.5,
        **params,
    ):
        super().__init__(seed, gap=gap, gap_weight=gap_weight,
                         structure_weight=structure_weight, **params)
        self.structure = {k: dict(v) for k, v in (structure or {}).items()}
        self.w_g = float(gap_weight)
        self.w_s = float(structure_weight)

    def _structure_score(self, task: str) -> tuple[float, list[str]]:
        raw = self.structure.get(task, {})
        vals, used = [], []
        for k in self.STRUCTURE_KEYS:
            v = raw.get(k)
            if v is None or not np.isfinite(v):
                continue
            # `intervention` is a cost: smaller is better, so invert it.
            v = 1.0 / (1.0 + float(v)) if k == "intervention" else float(np.clip(v, 0, 1))
            vals.append(v)
            used.append(k)
        if not vals:
            return float("nan"), []
        return float(np.mean(vals)), used

    def _rank(self, diagnoses, rng):
        rows = []
        for d in diagnoses:
            g, why, _ = self._value(d)
            if not np.isfinite(g):
                continue
            s, used = self._structure_score(d.task)
            if np.isfinite(s):
                score = g * (self.w_g + self.w_s * s)
                reason = (f"{why}: G={g:.3f}, structure={s:.3f} "
                          f"from {'+'.join(used)} -> {score:.3f}")
            else:
                # No structure signals: fall back to the gap alone rather than
                # imputing a value, which would demote exactly the tasks with
                # sparse trajectory data.
                score = g * (self.w_g + self.w_s * 0.5)
                reason = f"{why}: G={g:.3f}, no structure signals available"
            rows.append((d.task, float(score), reason))
        return sorted(rows, key=lambda x: (-x[1], x[0]))

    def select(self, diagnoses, n):
        res = DataSelector.select(self, diagnoses, n)
        n_struct = sum(
            1 for t in res.tasks if np.isfinite(self._structure_score(t)[0])
        )
        res.params["n_with_structure"] = n_struct
        if n_struct < res.delivered:
            res.note = ((res.note + " ") if res.note else "") + (
                f"{res.delivered - n_struct} of {res.delivered} selected tasks had "
                "no trajectory-structure signals and were ranked on the gap alone"
            )
        return res


#: The six curriculum strategies the current methodology compares. Everything
#: else in SELECTORS is either a legacy alias or an extra ablation arm.
CURRICULUM_STRATEGIES = (
    "uniform",
    "difficulty",
    "uncertainty",
    "learning_progress",
    "capability_reliability_gap",
    "gap_plus_structure",
)

SELECTORS.update(
    {
        "uniform": UniformSelector,
        "difficulty": DifficultySelector,
        "uncertainty": UncertaintySelector,
        "learning_progress": LearningProgressSelector,
        "capability_reliability_gap": CapabilityReliabilityGapSelector,
        "gap_plus_structure": GapStructureSelector,
    }
)
