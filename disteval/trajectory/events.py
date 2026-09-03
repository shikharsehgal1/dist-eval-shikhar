"""A canonical, benchmark-agnostic representation of agent trajectories.

Why a canonical form
--------------------
Every trajectory analysis in this package -- alignment, divergence detection,
failure attribution, counterfactuals, embeddings -- needs to ask the same
questions of a run: *what did the agent do at this point, with what arguments,
what came back, and what did that change about the world and the rubric?* If
those questions are answered by benchmark-specific dict-poking, each analysis
gets its own adapter and none of them compose.

So there is exactly one event type. Benchmark adapters map into it; every
analysis reads only it.

The event vocabulary
--------------------
``EventType`` deliberately covers what long-horizon professional-agent runs
actually contain, not just tool calls: retrieval, browser actions, intermediate
answers, environment state changes, verification steps, errors and retries.
It is a plain string enum, and :func:`register_event_type` lets an adapter add
one without editing this file -- a benchmark with a genuinely novel event kind
should not have to fork the taxonomy.

What is deliberately *not* here
-------------------------------
No model outputs beyond a short text field, no full observation blobs by
default. Trajectories at APEX scale are millions of events; the analyses in this
package need the *structure* (which tool, which arguments, which document,
success or failure), and keeping raw payloads out of the in-memory
representation is what lets the alignment and embedding code run over a whole
benchmark. Raw content stays addressable via ``raw_ref``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator, Optional, Sequence

__all__ = [
    "EventType",
    "EVENT_TYPES",
    "register_event_type",
    "TrajectoryEvent",
    "Trajectory",
    "TrajectorySet",
    "from_generic_steps",
]


class EventType:
    """Canonical event vocabulary. Extend with :func:`register_event_type`."""

    TOOL_CALL = "tool_call"          # the agent invoked a tool
    TOOL_RESULT = "tool_result"      # the tool returned (ok or error)
    RETRIEVAL = "retrieval"          # a document/file/record was fetched
    BROWSER = "browser"              # navigation, click, form fill
    PLAN = "plan"                    # an explicit planning/decomposition step
    REASONING = "reasoning"          # intermediate reasoning, no external effect
    INTERMEDIATE_ANSWER = "intermediate_answer"
    STATE_CHANGE = "state_change"    # environment/world state transition
    RUBRIC_UPDATE = "rubric_update"  # a rubric criterion changed status
    ERROR = "error"                  # an error surfaced to the agent
    RETRY = "retry"                  # the agent retried a failed action
    VERIFICATION = "verification"    # the agent checked its own work
    FINAL_ANSWER = "final_answer"    # terminal artifact / answer
    OTHER = "other"


EVENT_TYPES: set[str] = {
    v for k, v in vars(EventType).items() if not k.startswith("_") and isinstance(v, str)
}


def register_event_type(name: str) -> str:
    """Add a benchmark-specific event type to the canonical vocabulary."""
    if not name or not name.isidentifier():
        raise ValueError("event type must be a non-empty identifier-like string")
    EVENT_TYPES.add(name)
    return name


@dataclass
class TrajectoryEvent:
    """One structured step of an agent run.

    Only ``index`` and ``event_type`` are required. Everything else is optional
    because real adapters have partial information, and every analysis in this
    package degrades gracefully rather than requiring a fully populated event.

    Attributes
    ----------
    index:
        Position within the trajectory, 0-based. The *raw* clock. Alignment
        exists because comparing two runs by this index alone is weak.
    timestamp:
        Wall-clock seconds since run start, when available. Used by the
        cost/latency metrics, never by alignment (agents differ in speed).
    event_type:
        One of :class:`EventType`.
    tool_name, tool_args:
        The action identity. ``tool_args`` is canonicalised (sorted keys, scalars
        only at the top level) so two calls with the same semantics hash equal.
    observation:
        Short textual summary of what came back. Truncated by adapters.
    ok:
        Whether the step succeeded, when the environment reports it. ``None``
        means unknown -- distinct from ``False``.
    target:
        The object acted on: file path, document id, URL, table name. This is
        what makes "retrieved the wrong document" detectable.
    state_features:
        Numeric environment-state summary; the basis of the state-divergence
        signal and part of the trajectory embedding.
    rubric_state:
        Per-criterion status at this point, when the grader is incremental.
    cost:
        Per-step resource usage (tokens, dollars, seconds); see
        :mod:`disteval.reliability.cost`.
    raw_ref:
        Pointer back to the untruncated source record.
    """

    index: int
    event_type: str = EventType.OTHER
    timestamp: Optional[float] = None
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    observation: Optional[str] = None
    ok: Optional[bool] = None
    target: Optional[str] = None
    state_features: dict[str, float] = field(default_factory=dict)
    rubric_state: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    raw_ref: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- identity used by alignment and divergence ---------------------------
    @property
    def action_key(self) -> str:
        """Coarse identity: *what kind of thing* the agent did.

        Two events with the same ``action_key`` are the same move at the level a
        human would describe it ("called search", "read a file"). Alignment
        matches on this; divergence detection then drills into arguments.
        """
        if self.tool_name:
            return f"{self.event_type}:{self.tool_name}"
        return self.event_type

    @property
    def arg_key(self) -> str:
        """Fine identity: action plus canonicalised arguments and target."""
        payload = {"t": self.target, "a": _canonical(self.tool_args)}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return f"{self.action_key}|{hashlib.blake2s(blob.encode(), digest_size=8).hexdigest()}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryEvent":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def _canonical(args: dict) -> dict:
    """Normalise arguments so semantically identical calls hash identically."""
    out = {}
    for k, v in sorted((args or {}).items()):
        if isinstance(v, str):
            out[k] = v.strip()
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = json.dumps(v, sort_keys=True, default=str)
    return out


@dataclass
class Trajectory:
    """One run of one agent on one task, as a sequence of canonical events."""

    trajectory_id: str
    task: str
    model: str
    events: list[TrajectoryEvent] = field(default_factory=list)
    score: float = 0.0
    success: bool = False
    run_id: str = ""
    episode: int = 0
    domain: Optional[str] = None
    rubric_scores: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[TrajectoryEvent]:
        return iter(self.events)

    def __getitem__(self, i):
        return self.events[i]

    @property
    def action_keys(self) -> list[str]:
        return [e.action_key for e in self.events]

    @property
    def arg_keys(self) -> list[str]:
        return [e.arg_key for e in self.events]

    @property
    def tool_sequence(self) -> list[str]:
        return [e.tool_name for e in self.events if e.tool_name]

    def first_error(self) -> Optional[TrajectoryEvent]:
        """First event that failed or was an explicit error. None if never."""
        for e in self.events:
            if e.ok is False or e.event_type == EventType.ERROR:
                return e
        return None

    def n_retries(self) -> int:
        return sum(1 for e in self.events if e.event_type == EventType.RETRY)

    def n_errors(self) -> int:
        return sum(
            1 for e in self.events if e.ok is False or e.event_type == EventType.ERROR
        )

    def n_verifications(self) -> int:
        return sum(1 for e in self.events if e.event_type == EventType.VERIFICATION)

    def total_cost(self, key: str) -> float:
        if key in self.cost:
            return float(self.cost[key])
        return float(sum(e.cost.get(key, 0.0) for e in self.events))

    def to_dict(self) -> dict:
        d = {
            k: v for k, v in asdict(self).items() if k != "events"
        }
        d["events"] = [e.to_dict() for e in self.events]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        events = [TrajectoryEvent.from_dict(e) for e in d.get("events", [])]
        known = {f for f in cls.__dataclass_fields__} - {"events"}
        return cls(events=events, **{k: v for k, v in d.items() if k in known})


class TrajectorySet:
    """A queryable collection of trajectories, grouped by (model, task).

    Kept intentionally thin: an in-memory index plus JSONL persistence. For
    benchmark-scale corpora the events live in Parquet and are streamed --
    see :mod:`disteval.store`.
    """

    def __init__(self, trajectories: Optional[Iterable[Trajectory]] = None):
        self._items: list[Trajectory] = list(trajectories or [])

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._items)

    def add(self, t: Trajectory) -> None:
        self._items.append(t)

    def extend(self, ts: Iterable[Trajectory]) -> None:
        self._items.extend(ts)

    def for_task(self, task: str, model: Optional[str] = None) -> list[Trajectory]:
        return [
            t for t in self._items
            if t.task == task and (model is None or t.model == model)
        ]

    def group(self) -> dict[tuple[str, str], list[Trajectory]]:
        out: dict[tuple[str, str], list[Trajectory]] = {}
        for t in self._items:
            out.setdefault((t.model, t.task), []).append(t)
        return out

    def successes(self, task: str, model: Optional[str] = None) -> list[Trajectory]:
        return [t for t in self.for_task(task, model) if t.success]

    def failures(self, task: str, model: Optional[str] = None) -> list[Trajectory]:
        return [t for t in self.for_task(task, model) if not t.success]

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for t in self._items:
                f.write(json.dumps(t.to_dict(), default=str) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "TrajectorySet":
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(Trajectory.from_dict(json.loads(line)))
        return cls(items)


# --------------------------------------------------------------------------- #
# Generic adapter                                                             #
# --------------------------------------------------------------------------- #
_TOOL_KEYS = ("tool", "tool_name", "name", "action", "function")
_ARG_KEYS = ("args", "tool_args", "arguments", "input", "parameters")
_OBS_KEYS = ("observation", "output", "result", "content", "response")
_TARGET_KEYS = ("target", "path", "file", "url", "document", "doc_id", "resource")


def _first(d: dict, keys: Sequence[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def from_generic_steps(
    steps: Sequence[dict],
    trajectory_id: str,
    task: str,
    model: str,
    *,
    score: float = 0.0,
    success: Optional[bool] = None,
    success_threshold: float = 1.0,
    observation_chars: int = 400,
    **kwargs,
) -> Trajectory:
    """Best-effort mapping from loosely-structured step dicts to a Trajectory.

    Handles the common shapes agent harnesses emit (``tool``/``args``/``output``,
    ``action``/``input``/``observation``, ...). Where the type cannot be inferred
    it falls back to ``EventType.OTHER`` rather than guessing; analyses that need
    a specific type will simply see fewer events of it.
    """
    events: list[TrajectoryEvent] = []
    for i, raw in enumerate(steps):
        if not isinstance(raw, dict):
            events.append(TrajectoryEvent(index=i, observation=str(raw)[:observation_chars]))
            continue
        tool = _first(raw, _TOOL_KEYS)
        args = _first(raw, _ARG_KEYS, {}) or {}
        if not isinstance(args, dict):
            args = {"value": args}
        obs = _first(raw, _OBS_KEYS)
        etype = raw.get("event_type") or raw.get("type")
        ok = raw.get("ok")
        if ok is None and raw.get("error"):
            ok = False
        if etype not in EVENT_TYPES:
            if raw.get("error") or (isinstance(obs, str) and obs.lower().startswith("error")):
                etype = EventType.ERROR
            elif tool:
                etype = EventType.TOOL_CALL
            elif obs is not None:
                etype = EventType.REASONING
            else:
                etype = EventType.OTHER
        events.append(
            TrajectoryEvent(
                index=i,
                event_type=etype,
                timestamp=raw.get("timestamp"),
                tool_name=str(tool) if tool else None,
                tool_args=_canonical(args),
                observation=(str(obs)[:observation_chars] if obs is not None else None),
                ok=ok,
                target=_first(raw, _TARGET_KEYS) or _first(args, _TARGET_KEYS),
                state_features={
                    k: float(v) for k, v in (raw.get("state_features") or {}).items()
                },
                rubric_state=raw.get("rubric_state") or {},
                cost={k: float(v) for k, v in (raw.get("cost") or {}).items()},
                raw_ref=raw.get("raw_ref"),
            )
        )
    if success is None:
        success = score >= success_threshold
    return Trajectory(
        trajectory_id=trajectory_id,
        task=task,
        model=model,
        events=events,
        score=float(score),
        success=bool(success),
        **kwargs,
    )
