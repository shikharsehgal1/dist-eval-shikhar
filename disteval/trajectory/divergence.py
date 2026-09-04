"""Where a failed run stopped resembling a successful one, and what followed.

An alignment tells you *which* steps correspond. This module answers the
question the research actually needs: **where did the failed run go wrong, and
did anything downstream depend on it?**

Candidate divergence points
---------------------------
Rather than reporting one number, the analysis emits every structurally
identifiable candidate, each with the evidence that produced it, and then picks
a primary one by a documented rule. The candidates are:

``first_action_divergence``
    First aligned column where the coarse move differs (different tool, or a gap).

``first_argument_divergence``
    First aligned column where the same move was made with different arguments or
    against a different target. This is the "read the wrong file" case, and it is
    the most common real cause in document/spreadsheet agents.

``first_failed_execution``
    First step in the failed run that returned an error, whether or not the
    successful run has anything there.

``first_state_divergence``
    First column where the numeric environment-state summary separates by more
    than ``state_tol`` in normalised distance. Catches silent corruption: the
    actions look identical but the world no longer matches.

``first_rubric_divergence``
    First column where a rubric criterion holds in one run and not the other.
    The most decision-relevant candidate when incremental grading is available.

``last_common_state``
    The final column where the two runs still agreed on everything. Everything
    after this is downstream of the divergence.

Choosing the primary
--------------------
Precedence is rubric > state > argument > action > execution error, restricted
to candidates at or before the earliest of them, and then the *earliest* wins.
The reasoning: a rubric or state divergence is direct evidence of a consequence,
whereas an action divergence may be a harmless alternative route; but a later
consequence is always downstream of an earlier cause, so earliness dominates
within the evidence hierarchy. The rule is a heuristic and is recorded on the
result (``primary_rule``) so downstream analyses can override it.

Consequence and recoverability
------------------------------
Divergence alone is not causation. Two things are measured afterwards:

* ``consequence_strength`` -- how much of the remaining trajectory is affected,
  as the fraction of post-divergence columns that fail to match. A divergence
  followed by full re-convergence is weak evidence of causation.
* ``recovered`` -- whether the failed run genuinely returned to the successful
  run's path after diverging: the alignment tail matches cleanly *and* no state
  or rubric divergence appears later. Action realignment alone is not recovery,
  since an agent that read the wrong file and then ran the identical downstream
  steps on wrong data looks realigned but is not. A run that diverges, truly
  recovers, and still fails did not fail because of that divergence, and the
  analysis says so.

  Caveat worth stating: with no state features and no incremental rubric, the
  only evidence available is the action sequence, so ``recovered`` will be
  optimistic. Populate ``state_features`` or ``rubric_state`` on your events and
  this signal becomes much sharper.

Semantic comparison is optional
-------------------------------
Everything above is structural and runs with no model API. When free-text
observations or intermediate answers need semantic comparison,
:class:`SemanticComparator` is the plug point; :class:`NullComparator` (the
default) declines to judge, and :class:`EmbeddingComparator` uses any
user-supplied embedding function. No provider is imported anywhere in this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from .align import Alignment, align
from .events import EventType, Trajectory, TrajectoryEvent

__all__ = [
    "SemanticComparator",
    "NullComparator",
    "EmbeddingComparator",
    "DivergenceCandidate",
    "DivergenceReport",
    "analyse_divergence",
    "divergence_matrix",
]


# --------------------------------------------------------------------------- #
# Pluggable semantic comparison                                               #
# --------------------------------------------------------------------------- #
class SemanticComparator(Protocol):
    """Judges whether two free-text observations mean the same thing.

    Returns ``(similarity in [0,1], confidence in [0,1])``. A confidence of 0
    means "I decline to judge" and callers must fall back to structure.
    """

    def compare(self, a: str, b: str) -> tuple[float, float]: ...


class NullComparator:
    """Declines to judge. The default: the package never requires a model API."""

    def compare(self, a: str, b: str) -> tuple[float, float]:
        return (0.0, 0.0)


class EmbeddingComparator:
    """Cosine similarity under any user-supplied embedding function.

    ``embed`` maps a list of strings to an array of vectors. Any provider works;
    none is imported here. Results are cached per string so repeated comparisons
    across a task's runs cost one embedding call per distinct observation.
    """

    def __init__(self, embed: Callable[[Sequence[str]], np.ndarray], confidence: float = 0.8):
        self._embed = embed
        self._cache: dict[str, np.ndarray] = {}
        self._confidence = float(confidence)

    def _vec(self, s: str) -> np.ndarray:
        if s not in self._cache:
            self._cache[s] = np.asarray(self._embed([s]))[0]
        return self._cache[s]

    def compare(self, a: str, b: str) -> tuple[float, float]:
        if not a or not b:
            return (0.0, 0.0)
        va, vb = self._vec(a), self._vec(b)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-12 or nb < 1e-12:
            return (0.0, 0.0)
        cos = float(np.dot(va, vb) / (na * nb))
        return (float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0)), self._confidence)


# --------------------------------------------------------------------------- #
# Results                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DivergenceCandidate:
    """One structurally identified point at which two runs part ways."""

    kind: str
    column: int                       # index into the alignment's pair list
    i: Optional[int]                  # step index in the successful run
    j: Optional[int]                  # step index in the failed run
    evidence: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "column": self.column,
            "success_step": self.i,
            "failure_step": self.j,
            "evidence": self.evidence,
            **{f"detail_{k}": v for k, v in self.detail.items()},
        }


@dataclass
class DivergenceReport:
    """Full divergence analysis of one (successful, failed) trajectory pair."""

    task: str
    model: str
    success_id: str
    failure_id: str
    alignment: Alignment
    candidates: list[DivergenceCandidate]
    primary: Optional[DivergenceCandidate]
    primary_rule: str
    common_prefix: int
    consequence_strength: float
    recovered: bool
    n_post_divergence: int

    def candidate(self, kind: str) -> Optional[DivergenceCandidate]:
        for c in self.candidates:
            if c.kind == kind:
                return c
        return None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "model": self.model,
            "success_id": self.success_id,
            "failure_id": self.failure_id,
            "alignment_method": self.alignment.method,
            "alignment_distance": self.alignment.distance(),
            "common_prefix": self.common_prefix,
            "primary_kind": self.primary.kind if self.primary else None,
            "primary_column": self.primary.column if self.primary else None,
            "primary_failure_step": self.primary.j if self.primary else None,
            "primary_evidence": self.primary.evidence if self.primary else None,
            "primary_rule": self.primary_rule,
            "consequence_strength": self.consequence_strength,
            "recovered": self.recovered,
            "n_post_divergence": self.n_post_divergence,
            "n_candidates": len(self.candidates),
        }


# --------------------------------------------------------------------------- #
# Analysis                                                                    #
# --------------------------------------------------------------------------- #
def _state_distance(a: TrajectoryEvent, b: TrajectoryEvent) -> Optional[float]:
    keys = set(a.state_features) | set(b.state_features)
    if not keys:
        return None
    va = np.array([a.state_features.get(k, 0.0) for k in sorted(keys)], dtype=float)
    vb = np.array([b.state_features.get(k, 0.0) for k in sorted(keys)], dtype=float)
    denom = np.linalg.norm(va) + np.linalg.norm(vb)
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm(va - vb) / denom)


def _rubric_divergence(a: TrajectoryEvent, b: TrajectoryEvent) -> Optional[str]:
    keys = set(a.rubric_state) | set(b.rubric_state)
    for k in sorted(keys):
        if a.rubric_state.get(k) != b.rubric_state.get(k):
            return k
    return None


#: Evidence hierarchy: how directly a candidate demonstrates a consequence.
_PRECEDENCE = {
    "first_rubric_divergence": 4,
    "first_state_divergence": 3,
    "first_argument_divergence": 2,
    "first_action_divergence": 1,
    "first_failed_execution": 0,
}


def analyse_divergence(
    success: Trajectory,
    failure: Trajectory,
    *,
    method: str = "needleman_wunsch",
    match_threshold: float = 0.999,
    state_tol: float = 0.05,
    comparator: Optional[SemanticComparator] = None,
    align_kwargs: Optional[dict] = None,
) -> DivergenceReport:
    """Locate where ``failure`` stopped resembling ``success`` and what followed."""
    comparator = comparator or NullComparator()
    al = align(success, failure, method=method, **(align_kwargs or {}))
    pairs = al.pairs
    candidates: list[DivergenceCandidate] = []

    seen: set[str] = set()

    def _add(kind: str, col: int, p, evidence: str, **detail) -> None:
        if kind in seen:
            return
        seen.add(kind)
        candidates.append(DivergenceCandidate(kind, col, p.i, p.j, evidence, detail))

    for col, p in enumerate(pairs):
        ea = success.events[p.i] if p.i is not None else None
        eb = failure.events[p.j] if p.j is not None else None

        if eb is not None and (eb.ok is False or eb.event_type == EventType.ERROR):
            _add(
                "first_failed_execution", col, p,
                f"failed run errored at step {p.j}"
                + (f" in {eb.tool_name}" if eb.tool_name else ""),
                tool=eb.tool_name, observation=eb.observation,
            )

        if ea is None or eb is None:
            side = "success" if eb is None else "failure"
            _add("first_action_divergence", col, p,
                 f"step present only in the {side} run", gap_side=side)
            continue

        if ea.action_key != eb.action_key:
            _add("first_action_divergence", col, p,
                 f"different move: {ea.action_key!r} vs {eb.action_key!r}",
                 success_action=ea.action_key, failure_action=eb.action_key)
        elif ea.arg_key != eb.arg_key:
            sim, conf = (
                comparator.compare(ea.observation or "", eb.observation or "")
                if (ea.observation and eb.observation) else (0.0, 0.0)
            )
            _add("first_argument_divergence", col, p,
                 f"same move {ea.action_key!r}, different arguments/target: "
                 f"{ea.target!r} vs {eb.target!r}",
                 success_target=ea.target, failure_target=eb.target,
                 success_args=dict(ea.tool_args), failure_args=dict(eb.tool_args),
                 semantic_similarity=sim, semantic_confidence=conf)

        sd = _state_distance(ea, eb)
        if sd is not None and sd > state_tol:
            _add("first_state_divergence", col, p,
                 f"environment state separated (normalised distance {sd:.3f})",
                 state_distance=sd)

        rk = _rubric_divergence(ea, eb)
        if rk is not None:
            _add("first_rubric_divergence", col, p,
                 f"rubric criterion {rk!r} differs between the runs", criterion=rk)

    # -- pick the primary ---------------------------------------------------
    primary: Optional[DivergenceCandidate] = None
    rule = "none"
    if candidates:
        earliest_col = min(c.column for c in candidates)
        # Among candidates at the earliest column, the strongest evidence wins;
        # earliness dominates across columns because a later consequence is by
        # construction downstream of an earlier cause.
        at_earliest = [c for c in candidates if c.column == earliest_col]
        primary = max(at_earliest, key=lambda c: _PRECEDENCE.get(c.kind, -1))
        rule = "earliest column, then strongest evidence"

    # -- consequence and recovery ------------------------------------------
    prefix = al.common_prefix_length(match_threshold)
    if primary is None:
        consequence, recovered, n_post = 0.0, False, 0
    else:
        after = pairs[primary.column + 1:]
        n_post = len(after)
        if n_post == 0:
            consequence, recovered = 0.0, False
        else:
            mismatched = sum(
                1 for p in after
                if not (p.is_match and p.similarity >= match_threshold)
            )
            consequence = mismatched / n_post
            # "Recovered" means the run genuinely got back on track: the tail of
            # the alignment matched cleanly AND no state or rubric divergence
            # appeared later. Action realignment alone is not recovery -- an
            # agent that read the wrong file and then ran the identical
            # downstream steps on wrong data looks realigned but is not.
            tail = after[max(0, int(len(after) * 2 / 3)):]
            actions_realigned = bool(tail) and all(
                p.is_match and p.similarity >= match_threshold for p in tail
            )
            later_consequence = any(
                c.column > primary.column
                and c.kind in ("first_state_divergence", "first_rubric_divergence")
                for c in candidates
            )
            recovered = bool(actions_realigned and not later_consequence)

    return DivergenceReport(
        task=failure.task,
        model=failure.model,
        success_id=success.trajectory_id,
        failure_id=failure.trajectory_id,
        alignment=al,
        candidates=candidates,
        primary=primary,
        primary_rule=rule,
        common_prefix=prefix,
        consequence_strength=float(consequence),
        recovered=bool(recovered),
        n_post_divergence=int(n_post),
    )


def divergence_matrix(
    successes: Sequence[Trajectory],
    failures: Sequence[Trajectory],
    **kwargs,
) -> list[DivergenceReport]:
    """Analyse every (success, failure) pair for one task.

    Every failure is compared against every success rather than against one
    arbitrary reference, because which successful run a failure is closest to is
    itself informative -- and because a divergence that shows up against *all*
    successful runs is much stronger evidence than one that shows up against one.
    """
    out = []
    for s in successes:
        for f in failures:
            out.append(analyse_divergence(s, f, **kwargs))
    return out
