"""Approximate minimal interventions: what would have had to change?

The question
------------
Given a failed run, what is the smallest edit that would plausibly have made it
succeed?

    Delta*(tau_f) = argmin_Delta cost(Delta)  s.t.  P(success | tau_f + Delta) > eta

Solving this properly requires a model of ``P(success | trajectory)`` and the
ability to counterfactually re-execute the environment. Neither is available
from logged evaluation data. What *is* available is a set of successful runs of
the same task in the same environment, and those are counterfactual evidence: an
alignment between a failed run and a successful one is literally an edit script
turning one into the other.

So the approximation is:

    Delta* ~= the cheapest edit script, over all successful runs of this task,
              that reconciles the failed run's actions with a successful one.

This is an upper bound on the true minimal intervention cost -- there may be a
cheaper fix that no observed success happens to exhibit -- and it is only
defined when the task has at least one success. Both facts are stated on the
result object, and :attr:`InterventionEstimate.is_bound` marks it.

The cost model
--------------
Edits are drawn from an explicit vocabulary with configurable costs, ordered by
how much of the agent's behaviour has to change:

===================== ==== ===============================================
edit                  cost meaning
===================== ==== ===============================================
``retarget``          1.0  same tool, different file/document/URL
``rearg``             1.2  same tool and target, different other arguments
``insert_verify``     1.5  add a verification step the success had
``substitute_tool``   2.5  use a different tool at this point
``insert_step``       3.0  add a step the failed run never took
``delete_step``       2.0  remove a step the failed run took
===================== ==== ===============================================

The ordering encodes a claim worth arguing with: changing *what you point a tool
at* is a smaller behavioural change than changing *which tool you reach for*,
which is smaller than restructuring the plan. Costs are configurable precisely
because that claim is domain-dependent.

What the number is for
----------------------
``intervention_distance`` is one of the candidate recoverability signals. The
hypothesis it encodes -- a failure one retarget away from success is a better
preference-training target than one needing three inserted steps -- is tested,
not assumed. Note the deliberate asymmetry with the taxonomy: a run can have low
intervention distance and still be labelled a planning failure, and disagreement
between the two signals is informative rather than a bug.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .align import Alignment, align
from .events import EventType, Trajectory, TrajectoryEvent

__all__ = [
    "EditCosts",
    "Edit",
    "InterventionEstimate",
    "minimal_intervention",
    "intervention_distance",
]


@dataclass(frozen=True)
class EditCosts:
    """Cost of each edit primitive. Domain-dependent; override freely."""

    retarget: float = 1.0
    rearg: float = 1.2
    insert_verify: float = 1.5
    substitute_tool: float = 2.5
    insert_step: float = 3.0
    delete_step: float = 2.0

    def of(self, kind: str) -> float:
        return float(getattr(self, kind))


@dataclass(frozen=True)
class Edit:
    """One proposed change to the failed trajectory."""

    kind: str
    failure_step: Optional[int]
    success_step: Optional[int]
    cost: float
    description: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "failure_step": self.failure_step,
            "success_step": self.success_step,
            "cost": self.cost,
            "description": self.description,
        }


@dataclass
class InterventionEstimate:
    """The cheapest reconciling edit script found, and its provenance."""

    task: str
    failure_id: str
    reference_id: str
    edits: list[Edit]
    total_cost: float
    normalized_cost: float
    n_references_considered: int
    is_bound: bool = True
    note: str = ""

    @property
    def n_edits(self) -> int:
        return len(self.edits)

    @property
    def dominant_edit(self) -> Optional[str]:
        """The edit kind carrying the most cost: what mainly has to change."""
        if not self.edits:
            return None
        by_kind: dict[str, float] = {}
        for e in self.edits:
            by_kind[e.kind] = by_kind.get(e.kind, 0.0) + e.cost
        return max(by_kind, key=by_kind.get)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "failure_id": self.failure_id,
            "reference_id": self.reference_id,
            "n_edits": self.n_edits,
            "total_cost": self.total_cost,
            "normalized_cost": self.normalized_cost,
            "dominant_edit": self.dominant_edit,
            "n_references_considered": self.n_references_considered,
            "is_bound": self.is_bound,
            "note": self.note,
            "edits": [e.to_dict() for e in self.edits],
        }


def _edits_from_alignment(
    success: Trajectory, failure: Trajectory, al: Alignment, costs: EditCosts
) -> list[Edit]:
    """Read an alignment as an edit script transforming ``failure`` into ``success``."""
    edits: list[Edit] = []
    for p in al.pairs:
        if p.i is None and p.j is not None:
            fe = failure.events[p.j]
            edits.append(
                Edit("delete_step", p.j, None, costs.of("delete_step"),
                     f"drop step {p.j} ({fe.action_key})")
            )
        elif p.j is None and p.i is not None:
            se = success.events[p.i]
            kind = (
                "insert_verify"
                if se.event_type == EventType.VERIFICATION
                else "insert_step"
            )
            edits.append(
                Edit(kind, None, p.i, costs.of(kind),
                     f"insert {se.action_key} (present in the successful run at step {p.i})")
            )
        elif p.i is not None and p.j is not None:
            se, fe = success.events[p.i], failure.events[p.j]
            if se.arg_key == fe.arg_key:
                continue
            if se.action_key != fe.action_key:
                edits.append(
                    Edit("substitute_tool", p.j, p.i, costs.of("substitute_tool"),
                         f"use {se.action_key} instead of {fe.action_key} at step {p.j}",
                         {"from": fe.action_key, "to": se.action_key})
                )
            elif se.target != fe.target:
                edits.append(
                    Edit("retarget", p.j, p.i, costs.of("retarget"),
                         f"point {fe.action_key} at {se.target!r} instead of {fe.target!r} "
                         f"at step {p.j}",
                         {"from": fe.target, "to": se.target})
                )
            else:
                diff = sorted(
                    k for k in set(se.tool_args) | set(fe.tool_args)
                    if se.tool_args.get(k) != fe.tool_args.get(k)
                )
                edits.append(
                    Edit("rearg", p.j, p.i, costs.of("rearg"),
                         f"change argument(s) {diff} of {fe.action_key} at step {p.j}",
                         {"changed": diff})
                )
    return edits


def minimal_intervention(
    failure: Trajectory,
    successes: Sequence[Trajectory],
    *,
    costs: Optional[EditCosts] = None,
    method: str = "needleman_wunsch",
    align_kwargs: Optional[dict] = None,
) -> InterventionEstimate:
    """Cheapest edit script reconciling ``failure`` with any of ``successes``.

    Every successful run is considered, not one arbitrary reference: which
    success a failure is closest to is itself the answer to "what should it have
    done", and taking the minimum over references is what makes this an upper
    bound on the intervention cost rather than an arbitrary one.
    """
    costs = costs or EditCosts()
    successes = [s for s in successes if len(s.events)]
    if not successes:
        return InterventionEstimate(
            task=failure.task, failure_id=failure.trajectory_id, reference_id="",
            edits=[], total_cost=float("nan"), normalized_cost=float("nan"),
            n_references_considered=0, is_bound=False,
            note="no successful run of this task; the intervention distance is "
                 "undefined, not zero",
        )

    best: Optional[tuple[float, list[Edit], Trajectory]] = None
    for s in successes:
        al = align(s, failure, method=method, **(align_kwargs or {}))
        edits = _edits_from_alignment(s, failure, al, costs)
        total = sum(e.cost for e in edits)
        if best is None or total < best[0]:
            best = (total, edits, s)

    total, edits, ref = best
    denom = max(len(failure.events), len(ref.events), 1)
    return InterventionEstimate(
        task=failure.task,
        failure_id=failure.trajectory_id,
        reference_id=ref.trajectory_id,
        edits=edits,
        total_cost=float(total),
        normalized_cost=float(total / denom),
        n_references_considered=len(successes),
        is_bound=True,
        note="upper bound: the cheapest fix exhibited by an observed success, "
             "which may not be the cheapest fix that exists",
    )


def intervention_distance(
    failures: Sequence[Trajectory],
    successes: Sequence[Trajectory],
    **kwargs,
) -> dict:
    """Task-level intervention-distance summary over all failed runs.

    ``mean_normalized_cost`` is the recoverability signal. ``consistency`` --
    the fraction of failures whose cheapest fix is the same edit kind -- is the
    counterfactual analogue of low failure entropy, and is arguably the more
    actionable of the two: it says the same *fix* applies, not merely that the
    same thing goes wrong.
    """
    ests = [minimal_intervention(f, successes, **kwargs) for f in failures]
    valid = [e for e in ests if e.is_bound]
    if not valid:
        return {
            "n_failures": len(failures), "n_estimated": 0,
            "mean_cost": float("nan"), "mean_normalized_cost": float("nan"),
            "consistency": float("nan"), "dominant_edit": None,
            "estimates": [e.to_dict() for e in ests],
        }
    kinds = [e.dominant_edit for e in valid if e.dominant_edit]
    dominant = max(set(kinds), key=kinds.count) if kinds else None
    return {
        "n_failures": len(failures),
        "n_estimated": len(valid),
        "mean_cost": float(np.mean([e.total_cost for e in valid])),
        "mean_normalized_cost": float(np.mean([e.normalized_cost for e in valid])),
        "min_normalized_cost": float(np.min([e.normalized_cost for e in valid])),
        "mean_n_edits": float(np.mean([e.n_edits for e in valid])),
        "consistency": (kinds.count(dominant) / len(kinds)) if kinds else float("nan"),
        "dominant_edit": dominant,
        "estimates": [e.to_dict() for e in ests],
    }
