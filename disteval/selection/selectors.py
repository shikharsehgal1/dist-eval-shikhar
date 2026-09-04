"""Choosing which tasks to build training data from.

This is where the research hypothesis lives. Everything upstream estimates task
properties; this module turns those estimates into a training set, and the
experiment downstream measures whether the choice mattered.

    HYPOTHESIS: at a fixed labelling/training budget, selecting tasks where the
    agent has demonstrated capability but remains unreliable yields more
    reliability improvement than random selection or hardest-first selection.

The baselines are not strawmen. "Hardest tasks" is what most curricula actually
do and is a genuinely reasonable heuristic; "highest variance" is a direct
competitor that captures inconsistency without any Bayesian machinery; and
random is the one that is hardest to beat by accident. If recoverability
selection cannot beat *variance* selection, the Bayesian apparatus is not
earning its place, and the experiment is set up to say so.

Selectors
---------
``random``              uniform over tasks with usable data
``hardest``             lowest posterior mean first
``lowest_mean``         lowest observed mean score first (no posterior)
``highest_variance``    highest observed run-to-run variance first
``success_failure``     any task with both a success and a failure, at random
``recoverability``      highest recoverability score first
``uncertainty_aware``   recoverability discounted by posterior uncertainty
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
from typing import Callable, Mapping, Optional, Sequence

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
