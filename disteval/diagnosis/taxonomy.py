"""An extensible failure taxonomy for long-horizon agent runs.

Why a taxonomy at all
---------------------
"The run failed" is not actionable. The research question -- which failures are
locally recoverable -- needs to distinguish a run that fetched the wrong document
from one that had no idea how to approach the task. Those need different
interventions, and only the first is plausibly fixable by preference training on
matched trajectories.

Design constraints
------------------
1. **Extensible, not hard-coded.** The default categories below cover the stages
   of a tool-using agent run, but a benchmark with a domain-specific failure kind
   registers it with :func:`register_mode` rather than editing this file.
2. **Heuristics first, models optional.** Every default mode ships with a purely
   structural detector. LLM attribution is supported via
   :class:`FailureAttributor` but never required, and when it is used the raw
   evidence and a confidence are retained alongside the label so a reader can
   audit it.
3. **Stage-ordered.** Modes carry a ``stage`` index describing where in a run
   they occur. This is what lets the causality graph orient edges without a
   learned causal discovery step -- retrieval precedes reasoning precedes
   synthesis, and an earlier-stage failure is a candidate cause of a later one.

The default modes correspond to the standard decomposition of an agentic run:
perception/retrieval, planning, reasoning, tool selection, tool execution, state
tracking, memory, verification, recovery, and final synthesis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..trajectory.events import EventType, Trajectory, TrajectoryEvent

__all__ = [
    "FailureMode",
    "TAXONOMY",
    "register_mode",
    "get_mode",
    "stage_of",
    "FailureLabel",
    "classify_failure",
    "classify_events",
    "UNKNOWN",
    "FailureAttributor",
]

UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureMode:
    """One category in the taxonomy."""

    name: str
    #: Position in the run pipeline. Lower numbers happen earlier and are
    #: therefore candidate causes of higher-numbered failures.
    stage: int
    description: str
    #: Structural detector: given an event and its trajectory, does this mode
    #: apply? Returns a confidence in [0, 1]; 0 means "not this mode".
    detector: Optional[Callable[[TrajectoryEvent, Trajectory], float]] = None
    #: Whether preference training on matched trajectories is a plausible fix.
    #: Used as a prior by the recoverability model, never as a hard filter.
    locally_correctable: bool = True


TAXONOMY: dict[str, FailureMode] = {}


def register_mode(mode: FailureMode) -> FailureMode:
    """Add or replace a failure mode. Benchmarks extend the taxonomy this way."""
    TAXONOMY[mode.name] = mode
    return mode


def get_mode(name: str) -> Optional[FailureMode]:
    return TAXONOMY.get(name)


def stage_of(name: str) -> int:
    m = TAXONOMY.get(name)
    return m.stage if m else 99


# --------------------------------------------------------------------------- #
# Structural detectors                                                        #
# --------------------------------------------------------------------------- #
_RETRIEVAL_TYPES = {EventType.RETRIEVAL, EventType.BROWSER}
_ERROR_WORDS = (
    "not found", "no such file", "404", "permission denied", "timeout",
    "connection", "rate limit", "invalid", "traceback", "exception",
)


def _d_retrieval(e: TrajectoryEvent, t: Trajectory) -> float:
    if e.event_type in _RETRIEVAL_TYPES and e.ok is False:
        return 0.9
    obs = (e.observation or "").lower()
    if e.event_type in _RETRIEVAL_TYPES and any(w in obs for w in ("not found", "404", "empty")):
        return 0.7
    return 0.0


def _d_tool_execution(e: TrajectoryEvent, t: Trajectory) -> float:
    if e.ok is False and e.event_type in (EventType.TOOL_CALL, EventType.TOOL_RESULT):
        return 0.9
    if e.event_type == EventType.ERROR:
        return 0.6
    return 0.0


def _d_tool_selection(e: TrajectoryEvent, t: Trajectory) -> float:
    # An immediately-abandoned tool: called once, errored, and never used again.
    if e.ok is False and e.tool_name:
        later = [x for x in t.events[e.index + 1:] if x.tool_name == e.tool_name]
        if not later:
            return 0.5
    return 0.0


def _d_state_tracking(e: TrajectoryEvent, t: Trajectory) -> float:
    if e.event_type == EventType.STATE_CHANGE and e.ok is False:
        return 0.8
    return 0.0


def _d_recovery(e: TrajectoryEvent, t: Trajectory) -> float:
    # Repeated identical retries are a recovery failure, not the original fault.
    if e.event_type != EventType.RETRY and e.ok is not False:
        return 0.0
    same = sum(1 for x in t.events if x.arg_key == e.arg_key and x.ok is False)
    return 0.7 if same >= 3 else (0.4 if same == 2 else 0.0)


def _d_verification(e: TrajectoryEvent, t: Trajectory) -> float:
    # A final answer emitted with no verification anywhere in the run.
    if e.event_type == EventType.FINAL_ANSWER and t.n_verifications() == 0:
        return 0.5
    if e.event_type == EventType.VERIFICATION and e.ok is False:
        return 0.7
    return 0.0


def _d_synthesis(e: TrajectoryEvent, t: Trajectory) -> float:
    if e.event_type == EventType.FINAL_ANSWER and e.ok is False:
        return 0.8
    return 0.0


def _d_planning(e: TrajectoryEvent, t: Trajectory) -> float:
    if e.event_type == EventType.PLAN and e.ok is False:
        return 0.7
    return 0.0


def _d_reasoning(e: TrajectoryEvent, t: Trajectory) -> float:
    if e.event_type in (EventType.REASONING, EventType.INTERMEDIATE_ANSWER) and e.ok is False:
        return 0.6
    return 0.0


def _d_memory(e: TrajectoryEvent, t: Trajectory) -> float:
    # The agent re-fetched something it had already successfully fetched.
    if e.event_type in _RETRIEVAL_TYPES and e.target:
        earlier = [
            x for x in t.events[: e.index]
            if x.target == e.target and x.ok is not False
        ]
        if len(earlier) >= 2:
            return 0.4
    return 0.0


for _m in (
    FailureMode("retrieval", 0,
                "Failed or wrong perception/retrieval: document, file, record or page.",
                _d_retrieval, locally_correctable=True),
    FailureMode("planning", 1,
                "The run's decomposition or ordering of subtasks was wrong.",
                _d_planning, locally_correctable=False),
    FailureMode("reasoning", 2,
                "An intermediate inference or computation was wrong.",
                _d_reasoning, locally_correctable=False),
    FailureMode("tool_selection", 3,
                "The wrong tool was chosen for the subtask.",
                _d_tool_selection, locally_correctable=True),
    FailureMode("tool_execution", 4,
                "The right tool was called and it failed or was called wrongly.",
                _d_tool_execution, locally_correctable=True),
    FailureMode("state_tracking", 5,
                "The agent's model of the environment diverged from its actual state.",
                _d_state_tracking, locally_correctable=True),
    FailureMode("memory", 6,
                "Information established earlier in the run was lost or re-derived.",
                _d_memory, locally_correctable=True),
    FailureMode("verification", 7,
                "The agent did not check its work, or its check was wrong.",
                _d_verification, locally_correctable=True),
    FailureMode("recovery", 8,
                "An error occurred and the agent failed to recover from it.",
                _d_recovery, locally_correctable=True),
    FailureMode("synthesis", 9,
                "The final artifact or answer was malformed or incomplete.",
                _d_synthesis, locally_correctable=True),
):
    register_mode(_m)


# --------------------------------------------------------------------------- #
# Labelling                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FailureLabel:
    """A failure mode attributed to a specific event, with its evidence."""

    mode: str
    event_index: int
    confidence: float
    evidence: str
    source: str = "heuristic"   # "heuristic" | "llm" | "manual"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "event_index": self.event_index,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source": self.source,
        }


def classify_events(trajectory: Trajectory, min_confidence: float = 0.3) -> list[FailureLabel]:
    """Run every registered structural detector over every event.

    Returns *all* labels above ``min_confidence``, not one per event: a single
    step can legitimately be both a tool-execution failure and (because it was
    the third identical retry) a recovery failure, and the causality graph needs
    both.
    """
    out: list[FailureLabel] = []
    for e in trajectory.events:
        for name, mode in TAXONOMY.items():
            if mode.detector is None:
                continue
            c = float(mode.detector(e, trajectory))
            if c >= min_confidence:
                out.append(
                    FailureLabel(
                        mode=name,
                        event_index=e.index,
                        confidence=c,
                        evidence=(
                            f"{e.event_type}"
                            + (f":{e.tool_name}" if e.tool_name else "")
                            + (f" target={e.target!r}" if e.target else "")
                            + (f" obs={e.observation[:80]!r}" if e.observation else "")
                        ),
                    )
                )
    return out


def classify_failure(
    trajectory: Trajectory, min_confidence: float = 0.3
) -> tuple[str, list[FailureLabel]]:
    """Attribute a single primary failure mode to a run, plus all evidence.

    The primary is the **earliest-stage** label, breaking ties by confidence and
    then by event order. Earliest-stage rather than highest-confidence because a
    retrieval failure that causes a synthesis failure should be reported as a
    retrieval failure -- the downstream label is a symptom. Returns
    ``("unknown", [])`` when no detector fires, which is honest and common:
    plenty of failures leave no structural trace.
    """
    labels = classify_events(trajectory, min_confidence)
    if not labels:
        return UNKNOWN, []
    primary = min(
        labels, key=lambda l: (stage_of(l.mode), -l.confidence, l.event_index)
    )
    return primary.mode, labels


class FailureAttributor:
    """Pluggable attribution over the structural baseline.

    ``judge`` receives a trajectory and the heuristic labels and returns a list
    of ``FailureLabel`` (typically with ``source="llm"``). It is never called
    unless supplied. Heuristic labels are always retained alongside any model
    labels so the model's contribution is auditable and removable.
    """

    def __init__(self, judge: Optional[Callable[[Trajectory, list[FailureLabel]], list[FailureLabel]]] = None):
        self._judge = judge

    def label(self, trajectory: Trajectory, min_confidence: float = 0.3) -> list[FailureLabel]:
        heuristic = classify_events(trajectory, min_confidence)
        if self._judge is None:
            return heuristic
        extra = list(self._judge(trajectory, heuristic))
        return heuristic + extra
