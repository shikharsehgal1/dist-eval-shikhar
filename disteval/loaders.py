"""Generic entry points: ``load_runs``, ``load_tasks``, ``load_trajectories``.

This repository is not affiliated with Mercor and has no access to APEX data or
infrastructure. What it can do -- and what this module is for -- is be
*structurally compatible* with the kind of data a long-horizon professional-agent
benchmark produces, so that plugging real trajectories in is a matter of writing
one adapter rather than reshaping the analysis.

The shape assumed
-----------------
Three tables, which is the natural decomposition of any repeated-run agent eval:

``tasks``        task id, domain/world, rubric criteria, declared complexity,
                 environment/variant id, arbitrary metadata
``runs``         one row per execution: task id, model id, run index, final
                 score, per-criterion rubric scores, cost, environment/seed,
                 trajectory reference
``trajectories`` one record per execution: the structured event log

Every field except the ids and the score is optional, and every analysis degrades
gracefully when a field is absent -- rubric analysis is skipped without rubric
scores, trajectory signals are skipped without trajectories, and the posterior
estimation works from outcomes alone. A benchmark that logs only ``(task, score)``
still gets capability/reliability/recoverability estimates; richer logs unlock
more.

Formats
-------
JSON, JSONL and Parquet are read natively; ``format="auto"`` picks by extension.
Custom sources register an adapter with :func:`register_adapter` and are then
addressable by name, so the CLI and config files can reference them without this
module importing anything benchmark-specific.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .records import EpisodeRecord, RecordStore
from .trajectory.events import Trajectory, TrajectorySet, from_generic_steps

__all__ = [
    "TaskSpec",
    "load_tasks",
    "load_runs",
    "load_trajectories",
    "register_adapter",
    "ADAPTERS",
    "runs_to_store",
    "describe_dataset",
]


@dataclass
class TaskSpec:
    """A task definition, independent of any run of it."""

    task_id: str
    domain: str = ""
    environment: str = ""
    instruction: str = ""
    rubric_criteria: list[str] = field(default_factory=list)
    #: Declared dependencies between criteria, as (parent, child) pairs.
    criterion_dependencies: list[tuple[str, str]] = field(default_factory=list)
    #: Task-intrinsic complexity, preferred over observed step count for the
    #: horizon analysis (see disteval.reliability.scaling for why).
    complexity: Optional[float] = None
    tools: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def tool_signature(self) -> str:
        """Stable identity of the tool set, for tool-composition splits."""
        return "+".join(sorted(self.tools)) or "_none"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "environment": self.environment,
            "rubric_criteria": list(self.rubric_criteria),
            "complexity": self.complexity,
            "tools": list(self.tools),
            "tool_signature": self.tool_signature,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "TaskSpec":
        known = {
            "task_id", "domain", "environment", "instruction", "rubric_criteria",
            "criterion_dependencies", "complexity", "tools",
        }
        tid = d.get("task_id") or d.get("task") or d.get("id")
        if not tid:
            raise ValueError(f"task record has no task_id/task/id field: {dict(d)!r}")
        deps = [tuple(x) for x in d.get("criterion_dependencies", [])]
        return cls(
            task_id=str(tid),
            domain=str(d.get("domain") or d.get("world") or ""),
            environment=str(d.get("environment") or d.get("variant") or ""),
            instruction=str(d.get("instruction") or ""),
            rubric_criteria=list(d.get("rubric_criteria") or d.get("criteria") or []),
            criterion_dependencies=deps,
            complexity=d.get("complexity"),
            tools=list(d.get("tools") or []),
            metadata={k: v for k, v in d.items() if k not in known and k not in ("task", "id")},
        )


# --------------------------------------------------------------------------- #
# Reading                                                                     #
# --------------------------------------------------------------------------- #
def _read_records(path: str | Path, fmt: str = "auto") -> list[dict]:
    """Read JSON / JSONL / Parquet into a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if fmt == "auto":
        fmt = {".jsonl": "jsonl", ".ndjson": "jsonl", ".json": "json",
               ".parquet": "parquet", ".pq": "parquet"}.get(p.suffix.lower(), "json")
    if fmt == "jsonl":
        out = []
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{p}:{i} is not valid JSON: {exc}") from exc
        return out
    if fmt == "json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Accept {"runs": [...]} / {"tasks": [...]} wrappers.
            for key in ("runs", "tasks", "trajectories", "records", "data"):
                if key in data and isinstance(data[key], list):
                    return list(data[key])
            return [data]
        if not isinstance(data, list):
            raise ValueError(f"{p} must contain a list or a wrapped list")
        return data
    if fmt == "parquet":
        import pandas as pd

        return pd.read_parquet(p).to_dict("records")
    raise ValueError(f"unknown format {fmt!r}")


ADAPTERS: dict[str, Callable[..., Any]] = {}


def register_adapter(name: str, fn: Callable[..., Any]) -> None:
    """Register a named source adapter, addressable from configs and the CLI."""
    ADAPTERS[name] = fn


def load_tasks(source: str | Path, fmt: str = "auto", **kwargs) -> dict[str, TaskSpec]:
    """Load task definitions, keyed by task id.

    ``source`` is a path, or the name of a registered adapter.
    """
    if isinstance(source, str) and source in ADAPTERS:
        return ADAPTERS[source](kind="tasks", **kwargs)
    return {t.task_id: t for t in map(TaskSpec.from_dict, _read_records(source, fmt))}


def load_runs(
    source: str | Path,
    fmt: str = "auto",
    *,
    success_threshold: float = 1.0,
    default_model: str = "agent",
    **kwargs,
) -> list[dict]:
    """Load per-execution run records into a normalised list of dicts.

    Accepts the loose field names real harnesses emit (``task``/``task_id``,
    ``score``/``reward``, ``model``/``agent``) and normalises them. Missing
    ``success`` is derived from the score and the threshold; missing ``episode``
    is assigned per (model, task) in file order.
    """
    if isinstance(source, str) and source in ADAPTERS:
        return ADAPTERS[source](kind="runs", **kwargs)
    raw = _read_records(source, fmt)
    counters: dict[tuple[str, str], int] = {}
    out = []
    for i, r in enumerate(raw):
        task = r.get("task_id") or r.get("task")
        if task is None:
            raise ValueError(f"run record {i} has no task/task_id field: {r!r}")
        model = str(r.get("model") or r.get("agent") or default_model)
        score = r.get("score", r.get("reward"))
        if score is None:
            raise ValueError(f"run record {i} for task {task!r} has no score/reward")
        score = float(score)
        key = (model, str(task))
        ep = r.get("episode")
        if ep is None:
            ep = counters.get(key, 0)
        counters[key] = max(counters.get(key, 0), int(ep)) + 1
        out.append(
            {
                "task": str(task),
                "model": model,
                "run_id": str(r.get("run_id") or "run_0"),
                "episode": int(ep),
                "score": score,
                "success": bool(r["success"]) if "success" in r
                else score >= success_threshold,
                "domain": r.get("domain") or r.get("world"),
                "environment": r.get("environment") or r.get("variant") or r.get("seed"),
                "rubric_scores": dict(r.get("rubric_scores") or r.get("criteria") or {}),
                "cost": dict(r.get("cost") or {}),
                "trajectory_ref": r.get("trajectory_ref") or r.get("trajectory"),
                "failure_mode": r.get("failure_mode"),
                "n_steps": r.get("n_steps"),
                "metadata": {
                    k: v for k, v in r.items()
                    if k not in {
                        "task", "task_id", "model", "agent", "score", "reward",
                        "success", "episode", "run_id", "domain", "world",
                        "environment", "variant", "seed", "rubric_scores",
                        "criteria", "cost", "trajectory_ref", "trajectory",
                        "failure_mode", "n_steps",
                    }
                },
            }
        )
    return out


def load_trajectories(
    source: str | Path,
    fmt: str = "auto",
    *,
    success_threshold: float = 1.0,
    **kwargs,
) -> TrajectorySet:
    """Load structured event logs into canonical :class:`Trajectory` objects.

    Accepts either fully-formed trajectory records (with an ``events`` list) or
    loose harness output (a ``steps``/``trajectory`` list), mapping the latter
    through :func:`disteval.trajectory.events.from_generic_steps`.
    """
    if isinstance(source, str) and source in ADAPTERS:
        return ADAPTERS[source](kind="trajectories", **kwargs)
    out = TrajectorySet()
    for i, r in enumerate(_read_records(source, fmt)):
        if "events" in r:
            out.add(Trajectory.from_dict(r))
            continue
        steps = r.get("steps") or r.get("trajectory") or []
        task = r.get("task_id") or r.get("task")
        if task is None:
            raise ValueError(f"trajectory record {i} has no task/task_id field")
        score = float(r.get("score", r.get("reward", 0.0)))
        traj = from_generic_steps(
            steps,
            trajectory_id=str(r.get("trajectory_id") or r.get("id") or f"traj_{i}"),
            task=str(task),
            model=str(r.get("model") or r.get("agent") or "agent"),
            score=score,
            success=r.get("success"),
            success_threshold=success_threshold,
            run_id=str(r.get("run_id") or "run_0"),
            episode=int(r.get("episode") or 0),
            domain=r.get("domain") or r.get("world"),
        )
        traj.rubric_scores = dict(r.get("rubric_scores") or {})
        traj.cost = dict(r.get("cost") or {})
        traj.meta = dict(r.get("metadata") or {})
        out.add(traj)
    return out


def runs_to_store(runs: Sequence[Mapping]) -> RecordStore:
    """Convert normalised run dicts into the existing :class:`RecordStore`.

    Preserved as the bridge to the pre-existing analysis code (``metrics``,
    ``failure``, ``compare``, ``report``), so the new loaders feed the old
    pipeline unchanged.
    """
    store = RecordStore()
    for r in runs:
        strata = {}
        if r.get("domain"):
            strata["domain"] = r["domain"]
        if r.get("environment") is not None:
            strata["environment"] = r["environment"]
        for k, v in (r.get("metadata") or {}).items():
            if isinstance(v, (str, int, float, bool)):
                strata[k] = v
        store.add(
            EpisodeRecord(
                run_id=r.get("run_id", "run_0"),
                model=r["model"],
                task=r["task"],
                episode=int(r.get("episode", 0)),
                score=float(r["score"]),
                success=bool(r["success"]),
                strata=strata,
                failure_mode=r.get("failure_mode"),
                length=r.get("n_steps"),
                trajectory_ref=r.get("trajectory_ref"),
                meta={"rubric_scores": r.get("rubric_scores", {}), "cost": r.get("cost", {})},
            )
        )
    return store


def describe_dataset(
    runs: Sequence[Mapping],
    tasks: Optional[Mapping[str, TaskSpec]] = None,
    trajectories: Optional[TrajectorySet] = None,
) -> dict:
    """What is present in this dataset, and therefore which analyses can run.

    Reported at the top of every generated report so a reader knows which
    sections are missing because the data lacked a field rather than because the
    analysis found nothing.
    """
    import numpy as np

    by_task: dict[str, int] = {}
    models, domains = set(), set()
    n_rubric = n_cost = 0
    for r in runs:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
        models.add(r["model"])
        if r.get("domain"):
            domains.add(r["domain"])
        n_rubric += bool(r.get("rubric_scores"))
        n_cost += bool(r.get("cost"))
    counts = list(by_task.values()) or [0]
    return {
        "n_runs": len(runs),
        "n_tasks": len(by_task),
        "n_models": len(models),
        "n_domains": len(domains),
        "runs_per_task_min": int(min(counts)),
        "runs_per_task_max": int(max(counts)),
        "runs_per_task_mean": float(np.mean(counts)),
        "unequal_run_counts": len(set(counts)) > 1,
        "has_rubric_scores": n_rubric > 0,
        "pct_runs_with_rubric": n_rubric / max(len(runs), 1),
        "has_cost": n_cost > 0,
        "has_trajectories": bool(trajectories and len(trajectories)),
        "n_trajectories": len(trajectories) if trajectories else 0,
        "has_task_specs": bool(tasks),
        "has_criterion_dependencies": bool(
            tasks and any(t.criterion_dependencies for t in tasks.values())
        ),
        "enabled_analyses": _enabled(runs, tasks, trajectories),
    }


def _enabled(runs, tasks, trajectories) -> dict[str, bool]:
    has_rubric = any(r.get("rubric_scores") for r in runs)
    has_traj = bool(trajectories and len(trajectories))
    return {
        "posterior_reliability": bool(runs),
        "classification": bool(runs),
        "recoverability_posterior": bool(runs),
        "pass_at_k": bool(runs),
        "hierarchical_pooling": len({r["task"] for r in runs}) >= 4,
        "rubric_reliability": has_rubric,
        "criterion_root_cause": bool(
            tasks and any(t.criterion_dependencies for t in tasks.values())
        ),
        "trajectory_divergence": has_traj,
        "failure_taxonomy": has_traj,
        "counterfactual_intervention": has_traj,
        "trajectory_embedding": has_traj,
        "survival_analysis": has_traj,
        "capability_execution_split": has_rubric or has_traj,
        "cost_analysis": any(r.get("cost") for r in runs),
        "preference_pairs": has_traj,
    }
