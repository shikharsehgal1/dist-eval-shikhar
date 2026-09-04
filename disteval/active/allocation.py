"""Deciding which task to run next, under a fixed evaluation budget.

The setting
-----------
You have ``B`` agent executions to spend across ``T`` tasks. Uniform allocation
spends ``B/T`` on each, which is the right thing to do when you know nothing --
and demonstrably wasteful once you have a few runs, because most of the budget
then goes to confirming labels that are already settled.

Every policy here implements the same interface and every one respects the
budget exactly. That last property is enforced in
:meth:`AllocationPolicy.allocate` rather than left to each policy, and it is
property-tested.

Policies
--------
``UniformAllocation``
    The baseline every adaptive policy must beat. Not a strawman: it is
    unbiased, trivially parallel, and immune to the feedback pathology below.

``GreedyValuePolicy``
    Repeatedly spend the next run on the task with the highest value-of-
    information score (:mod:`disteval.active.value`), re-scoring after each award
    against a sampled rollout of the outcome. This is a genuine sequential
    policy, not a one-shot ranking. Default criterion is multi-step label-flip
    probability.

``ThompsonAllocation``
    Sample each task's latent value from its posterior and allocate to the
    argmax. Naturally balances exploration and exploitation and, unlike greedy
    VOI, does not get stuck when several tasks tie.

``StratifiedUniform``
    Uniform within declared strata (domain, difficulty) with proportional
    budgets. The right compromise when you need per-stratum estimates.

A pathology worth naming
------------------------
Adaptive allocation makes run counts *outcome-dependent*, which breaks the
i.i.d.-across-tasks assumption behind aggregate statistics like pass^k averaged
over tasks. A task that got 20 runs because it looked interesting is not
exchangeable with one that got 3. This is why :class:`AllocationResult` records
the full allocation trace, and why the evaluator reports aggregate metrics on the
uniform prefix separately. Adaptive sampling is for *per-task* estimates and
classification; treat suite-level headline numbers computed from adaptively
sampled data with suspicion.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from ..reliability.classify import ReliabilityThresholds
from ..reliability.posterior import JEFFREYS_PRIOR, BetaPrior, TaskPosterior, binary_posterior
from .value import bald, expected_entropy_reduction, flip_probability, label_of

__all__ = [
    "TaskState",
    "AllocationResult",
    "AllocationPolicy",
    "UniformAllocation",
    "GreedyValuePolicy",
    "ThompsonAllocation",
    "StratifiedUniform",
    "VALUE_CRITERIA",
    "make_policy",
]


@dataclass
class TaskState:
    """Mutable running state of one task during adaptive evaluation."""

    task: str
    n_runs: int = 0
    n_success: float = 0.0
    domain: Optional[str] = None
    prior: BetaPrior = JEFFREYS_PRIOR
    stopped: bool = False
    stop_reason: str = ""
    scores: list[float] = field(default_factory=list)

    @property
    def posterior(self) -> TaskPosterior:
        return TaskPosterior(
            alpha=self.prior.alpha + self.n_success,
            beta=self.prior.beta + (self.n_runs - self.n_success),
            n_obs=self.n_runs,
            n_eff=float(self.n_runs),
            binary=True,
            prior=self.prior,
        )

    def observe(self, score: float, success: Optional[bool] = None) -> None:
        self.n_runs += 1
        self.scores.append(float(score))
        self.n_success += float(success if success is not None else score >= 1.0)


VALUE_CRITERIA: dict[str, Callable] = {
    "flip": lambda st, th: flip_probability(st.posterior, th),
    "entropy": lambda st, th: expected_entropy_reduction(st.posterior),
    "bald": lambda st, th: bald(st.posterior),
}


@dataclass
class AllocationResult:
    """An allocation plus the trace needed to audit it."""

    allocation: dict[str, int]
    order: list[str]
    budget: int
    policy: str
    scores_trace: list[tuple[str, float]] = field(default_factory=list)

    @property
    def spent(self) -> int:
        return sum(self.allocation.values())

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "budget": self.budget,
            "spent": self.spent,
            "n_tasks_touched": sum(1 for v in self.allocation.values() if v > 0),
            "allocation": dict(self.allocation),
        }


class AllocationPolicy(ABC):
    """Base class. Subclasses implement :meth:`_order`; the budget is enforced here."""

    name = "policy"

    def __init__(self, thresholds: Optional[ReliabilityThresholds] = None):
        self.thresholds = thresholds or ReliabilityThresholds()

    @abstractmethod
    def _order(self, states: Sequence[TaskState], budget: int, rng) -> list[str]:
        """Return exactly ``budget`` task ids, in the order runs should be spent."""

    def allocate(
        self, states: Sequence[TaskState], budget: int, seed: int = 0
    ) -> AllocationResult:
        """Allocate ``budget`` runs across ``states``, never exceeding it.

        Tasks marked ``stopped`` are excluded. If every task has stopped, the
        remaining budget is simply not spent -- returning early is the correct
        behaviour and is what makes the stopping rules save anything.
        """
        if budget < 0:
            raise ValueError("budget must be non-negative")
        rng = np.random.default_rng(seed)
        live = [s for s in states if not s.stopped]
        if not live or budget == 0:
            return AllocationResult({}, [], budget, self.name)
        order = self._order(live, budget, rng)[:budget]
        alloc: dict[str, int] = {}
        for t in order:
            alloc[t] = alloc.get(t, 0) + 1
        assert sum(alloc.values()) <= budget, "policy exceeded its budget"
        return AllocationResult(alloc, order, budget, self.name)


class UniformAllocation(AllocationPolicy):
    """Round-robin. The baseline every adaptive policy has to beat."""

    name = "uniform"

    def _order(self, states, budget, rng):
        ids = [s.task for s in states]
        reps = int(np.ceil(budget / len(ids)))
        return [t for _ in range(reps) for t in ids][:budget]


class GreedyValuePolicy(AllocationPolicy):
    """Spend each run where the value of information is highest, re-scoring as it goes."""

    name = "greedy_voi"

    def __init__(
        self,
        criterion: str = "flip",
        thresholds: Optional[ReliabilityThresholds] = None,
        tie_break: str = "entropy",
    ):
        super().__init__(thresholds)
        if criterion not in VALUE_CRITERIA:
            raise ValueError(f"unknown criterion {criterion!r}; have {sorted(VALUE_CRITERIA)}")
        self.criterion = criterion
        self.tie_break = tie_break
        self.name = f"greedy_{criterion}"

    def _order(self, states, budget, rng):
        # Work on copies so scoring does not mutate the caller's state.
        work = {
            s.task: TaskState(s.task, s.n_runs, s.n_success, s.domain, s.prior)
            for s in states
        }
        th = self.thresholds
        primary = VALUE_CRITERIA[self.criterion]
        secondary = VALUE_CRITERIA.get(self.tie_break, VALUE_CRITERIA["entropy"])
        order: list[str] = []
        for _ in range(budget):
            scored = [
                (primary(st, self.thresholds), secondary(st, self.thresholds), t)
                for t, st in work.items()
            ]
            # Tiny jitter breaks exact ties without changing any real ordering.
            best = max(scored, key=lambda x: (x[0], x[1], rng.random()))
            t = best[2]
            order.append(t)
            # Advance the hypothetical state by a *sampled* outcome from the
            # posterior predictive, not by the fractional expectation. This
            # matters: advancing by the expectation keeps the posterior mean
            # fixed and only shrinks its variance, so a borderline task's
            # flip probability plateaus instead of resolving, and greedy pours
            # the entire budget into it. Sampling a concrete 0/1 outcome moves
            # the state the way a real run would, so the criterion actually
            # decreases as the task becomes decided. The rng is seeded, so the
            # plan is reproducible.
            st = work[t]
            st.n_runs += 1
            st.n_success += float(rng.random() < st.posterior.mean)
        return order


class ThompsonAllocation(AllocationPolicy):
    """Sample from each posterior, allocate to the argmax of a target functional.

    Default target is the recoverability-relevant quantity ``p*(1-p)``: variance
    of the run outcome, maximised at p=0.5, which is where reliability
    information is densest. Pass ``target`` to optimise something else.
    """

    name = "thompson"

    def __init__(
        self,
        target: Optional[Callable[[float], float]] = None,
        thresholds: Optional[ReliabilityThresholds] = None,
    ):
        super().__init__(thresholds)
        self.target = target or (lambda p: p * (1.0 - p))

    def _order(self, states, budget, rng):
        ids = [s.task for s in states]
        posts = [s.posterior for s in states]
        order = []
        counts = {t: 0 for t in ids}
        for _ in range(budget):
            draws = [self.target(float(rng.beta(p.alpha, p.beta))) for p in posts]
            i = int(np.argmax(draws))
            order.append(ids[i])
            counts[ids[i]] += 1
        return order


class StratifiedUniform(AllocationPolicy):
    """Uniform within strata, budget split proportionally to stratum size."""

    name = "stratified_uniform"

    def _order(self, states, budget, rng):
        strata: dict[str, list[str]] = {}
        for s in states:
            strata.setdefault(s.domain or "_all", []).append(s.task)
        total = sum(len(v) for v in strata.values())
        order: list[str] = []
        for key in sorted(strata):
            ids = strata[key]
            share = int(round(budget * len(ids) / total))
            reps = int(np.ceil(max(share, 1) / len(ids)))
            order.extend(([t for _ in range(reps) for t in ids])[:share])
        # Any rounding shortfall goes round-robin over everything.
        i = 0
        all_ids = [s.task for s in states]
        while len(order) < budget:
            order.append(all_ids[i % len(all_ids)])
            i += 1
        return order[:budget]


_POLICIES = {
    "uniform": UniformAllocation,
    "greedy_voi": GreedyValuePolicy,
    "thompson": ThompsonAllocation,
    "stratified_uniform": StratifiedUniform,
}


def make_policy(name: str, **kwargs) -> AllocationPolicy:
    """Construct a policy by name. Used by the config-driven experiment runner."""
    if name not in _POLICIES:
        raise ValueError(f"unknown allocation policy {name!r}; have {sorted(_POLICIES)}")
    return _POLICIES[name](**kwargs)
