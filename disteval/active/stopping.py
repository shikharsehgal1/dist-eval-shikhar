"""Sequential stopping: stop running a task once more runs cannot change anything.

Running every task exactly ``k`` times is the wrong default once you accept that
tasks differ. A task at 0/8 is settled; an eleventh run on it buys nothing. A
task at 4/8 with a wide interval is where the budget should go.

Rules
-----
Each rule answers ``should_stop(state) -> (bool, reason)``. They compose with
:class:`AnyOf` / :class:`AllOf`, and every rule reports *why* it fired so a
truncated evaluation is auditable rather than mysterious.

``ConfidentLabel``    stop when P(label is correct) exceeds a confidence
``IntervalWidth``     stop when the credible interval is narrower than a target
``MaxRuns``           a hard cap, always present in practice
``MinRuns``           a floor, composed with AllOf, so nothing stops too early
``NoValue``           stop when the value of another run falls below a floor

A caution about optional stopping
----------------------------------
Stopping when a *frequentist* criterion is met inflates type-I error: you are
effectively testing repeatedly and stopping on a favourable result. The rules
here are stated on **posterior** quantities, which are not subject to that
particular pathology -- a Bayesian posterior conditions on the data actually
observed and its interpretation does not depend on the stopping rule, provided
the rule depends only on the observed data (it does here).

That is *not* a claim that adaptive stopping is free. It changes the *sample* in
a way that matters for suite-level aggregates: tasks that stopped early are
systematically the easy and the hopeless ones, so a pass^k averaged over
adaptively-sampled tasks is not comparable to one averaged over uniformly
sampled tasks. :class:`StoppingLedger` records every stop so the report can say
which aggregates are safe. Compute headline suite numbers from the uniform
prefix.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..reliability.classify import RECOVERABLE, ReliabilityThresholds
from .allocation import TaskState
from .value import flip_probability, label_of

__all__ = [
    "StoppingRule",
    "ConfidentLabel",
    "IntervalWidth",
    "MaxRuns",
    "MinRuns",
    "NoValue",
    "AnyOf",
    "AllOf",
    "StoppingLedger",
    "default_stopping",
]


class StoppingRule(ABC):
    """Base class. ``should_stop`` returns ``(stop, reason)``."""

    name = "rule"

    @abstractmethod
    def should_stop(self, state: TaskState) -> tuple[bool, str]: ...

    def __or__(self, other: "StoppingRule") -> "AnyOf":
        return AnyOf([self, other])

    def __and__(self, other: "StoppingRule") -> "AllOf":
        return AllOf([self, other])


@dataclass
class ConfidentLabel(StoppingRule):
    """Stop once the posterior confidently supports a definite label.

    Concretely, stop when ``P(p > tau_rel) >= confidence`` (definitely SOLID) or
    ``P(p < tau_stuck) >= confidence`` (definitely STUCK).

    RECOVERABLE is held to a stricter standard than the other two, deliberately.
    Classification confidence alone is not enough there: a task at 2/4 already
    supports RECOVERABLE at 96% (we are confident it *can* and confident it is
    not reliable) and its label is stable against the next few runs -- so a
    label-stability rule would stop it. But RECOVERABLE tasks are the ones that
    get *ranked* and fed into training-data selection, and ranking needs the
    posterior located, not just labelled. So this rule additionally requires the
    credible interval to be narrower than ``recoverable_ci_width``.

    Note what the SOLID condition implies about cost: confirming
    ``P(p > 0.9) >= 0.95`` takes roughly 20 consecutive successes under a
    Jeffreys prior. High reliability is genuinely expensive to certify, and no
    stopping rule can make it cheap.
    """

    thresholds: ReliabilityThresholds = field(default_factory=ReliabilityThresholds)
    confidence: float = 0.95
    recoverable_flip_tol: float = 0.05
    recoverable_ci_width: float = 0.35
    name: str = "confident_label"

    def should_stop(self, state: TaskState) -> tuple[bool, str]:
        th = self.thresholds
        post = state.posterior
        if state.n_runs < th.min_runs:
            return False, ""
        if post.prob_above(th.tau_rel) >= self.confidence:
            return True, (
                f"P(p > {th.tau_rel}) = {post.prob_above(th.tau_rel):.3f} "
                f">= {self.confidence}: confidently SOLID"
            )
        if 1.0 - post.prob_above(th.tau_stuck) >= self.confidence:
            return True, (
                f"P(p < {th.tau_stuck}) = {1 - post.prob_above(th.tau_stuck):.3f} "
                f">= {self.confidence}: confidently STUCK"
            )
        if label_of(post, th) == RECOVERABLE:
            flip = flip_probability(post, th)
            lo, hi = post.credible_interval(th.credible_level)
            width = hi - lo
            confident = (
                post.prob_above(th.tau_cap) >= self.confidence
                and post.prob_above(th.tau_rel) <= 1.0 - self.confidence
            )
            if confident and flip <= self.recoverable_flip_tol and width <= self.recoverable_ci_width:
                return True, (
                    f"confidently RECOVERABLE (C={post.prob_above(th.tau_cap):.3f}, "
                    f"R={post.prob_above(th.tau_rel):.3f}), P(label changes)="
                    f"{flip:.3f}, CI width {width:.3f} <= {self.recoverable_ci_width}"
                )
        return False, ""


@dataclass
class IntervalWidth(StoppingRule):
    """Stop once the credible interval is narrower than ``target_width``."""

    target_width: float = 0.25
    level: float = 0.95
    name: str = "interval_width"

    def should_stop(self, state: TaskState) -> tuple[bool, str]:
        lo, hi = state.posterior.credible_interval(self.level)
        w = hi - lo
        if w <= self.target_width:
            return True, (
                f"{int(self.level*100)}% CI width {w:.3f} <= target {self.target_width}"
            )
        return False, ""


@dataclass
class MaxRuns(StoppingRule):
    """A hard cap. Always include one: every other rule can in principle never fire."""

    max_runs: int = 32
    name: str = "max_runs"

    def should_stop(self, state: TaskState) -> tuple[bool, str]:
        if state.n_runs >= self.max_runs:
            return True, f"reached the run cap ({self.max_runs})"
        return False, ""


@dataclass
class MinRuns(StoppingRule):
    """A floor: never stop before ``min_runs``. Compose with :class:`AllOf`."""

    min_runs: int = 3
    name: str = "min_runs"

    def should_stop(self, state: TaskState) -> tuple[bool, str]:
        if state.n_runs >= self.min_runs:
            return True, ""
        return False, f"only {state.n_runs} runs; floor is {self.min_runs}"


@dataclass
class NoValue(StoppingRule):
    """Stop when the value of further runs falls below ``tol``.

    ``horizon`` is the crux and there is no horizon-free version of this rule:
    "the label will not change" is only meaningful relative to how many more runs
    you would have been willing to spend. At ``horizon=2`` a task at 2/4 looks
    settled, because two more runs genuinely cannot move it -- but eight more
    could. Set ``horizon`` to roughly the additional per-task budget you would
    otherwise commit; the default of 8 matches a typical cap of 24 against a
    floor of 3.
    """

    tol: float = 0.02
    thresholds: ReliabilityThresholds = field(default_factory=ReliabilityThresholds)
    horizon: int = 8
    name: str = "no_value"

    def should_stop(self, state: TaskState) -> tuple[bool, str]:
        v = flip_probability(state.posterior, self.thresholds, horizon=self.horizon)
        if v <= self.tol:
            return True, (
                f"P(label changes within {self.horizon} more runs) = {v:.3f} "
                f"<= {self.tol}"
            )
        return False, ""


@dataclass
class AnyOf(StoppingRule):
    """Stop when any child rule fires."""

    rules: list[StoppingRule]
    name: str = "any_of"

    def should_stop(self, state: TaskState) -> tuple[bool, str]:
        for r in self.rules:
            stop, why = r.should_stop(state)
            if stop:
                return True, f"{r.name}: {why}"
        return False, ""


@dataclass
class AllOf(StoppingRule):
    """Stop only when every child rule fires."""

    rules: list[StoppingRule]
    name: str = "all_of"

    def should_stop(self, state: TaskState) -> tuple[bool, str]:
        reasons = []
        for r in self.rules:
            stop, why = r.should_stop(state)
            if not stop:
                return False, ""
            if why:
                reasons.append(f"{r.name}: {why}")
        return True, "; ".join(reasons)


def default_stopping(
    thresholds: Optional[ReliabilityThresholds] = None,
    *,
    min_runs: int = 3,
    max_runs: int = 24,
    confidence: float = 0.95,
    no_value_horizon: int = 8,
) -> StoppingRule:
    """The recommended composition: a floor, a cap, and a confident-label rule.

    ``(MinRuns AND (ConfidentLabel OR NoValue)) OR MaxRuns`` -- never stop before
    the floor, stop once the label is settled or another run is worthless, and
    always stop at the cap.
    """
    th = thresholds or ReliabilityThresholds()
    return AnyOf(
        [
            AllOf(
                [
                    MinRuns(min_runs),
                    AnyOf(
                        [
                            ConfidentLabel(th, confidence),
                            NoValue(0.02, th, horizon=no_value_horizon),
                        ]
                    ),
                ]
            ),
            MaxRuns(max_runs),
        ]
    )


@dataclass
class StoppingLedger:
    """Record of every stop, so a truncated evaluation is auditable."""

    entries: list[dict] = field(default_factory=list)

    def record(self, state: TaskState, reason: str) -> None:
        self.entries.append(
            {
                "task": state.task,
                "n_runs": state.n_runs,
                "n_success": state.n_success,
                "posterior_mean": state.posterior.mean,
                "reason": reason,
            }
        )

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.entries)

    def summary(self) -> dict:
        if not self.entries:
            return {"n_stopped": 0}
        import numpy as np

        runs = [e["n_runs"] for e in self.entries]
        return {
            "n_stopped": len(self.entries),
            "mean_runs_at_stop": float(np.mean(runs)),
            "min_runs_at_stop": int(min(runs)),
            "max_runs_at_stop": int(max(runs)),
        }
