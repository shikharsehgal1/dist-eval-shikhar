"""
disteval.distributed_eval — Aggregate multi-agent evaluations into a shared pool.

OVERVIEW
────────
Distributed evaluation is the third phase of recursive self-improvement:
instead of training from a single agent's evaluations, pool evaluations from
multiple agents on the same tasks and generate cross-agent training pairs.

This module provides:

  - DistributedEvalRecord: one agent's outcome on one task, with optional
    per-checkpoint breakdowns and trajectory references.
  - DistributedEvalPool: aggregate, query, and compare records across agents.
  - Cross-agent training pair generation: find tasks where agents disagree,
    and use the high-performing agent's trajectory as the positive example
    for the low-performing agent (and vice versa).
  - Disagreement attribution: identify which checkpoints/sub-tasks drive
    the disagreement between agents.

The pool is default-disabled and opt-in; existing single-agent workflows are
unchanged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DistributedEvalRecord:
    """One agent's outcome on one task."""

    agent_name: str
    model_name: str
    task: str
    score: float
    checkpoint_scores: dict[str, float] = field(default_factory=dict)
    trajectory_ref: Optional[str] = None
    success: bool = False
    failure_mode: Optional[str] = None
    sub_task_depth: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "task": self.task,
            "score": self.score,
            "checkpoint_scores": self.checkpoint_scores,
            "trajectory_ref": self.trajectory_ref,
            "success": self.success,
            "failure_mode": self.failure_mode,
            "sub_task_depth": self.sub_task_depth,
        }


@dataclass
class CrossAgentPair:
    """A cross-agent (positive, negative) training pair for one task."""

    task: str
    positive_agent: str
    negative_agent: str
    positive_score: float
    negative_score: float
    positive_trajectory_ref: Optional[str]
    negative_trajectory_ref: Optional[str]
    gap: float
    disagreement_checkpoints: list[str] = field(default_factory=list)
    attribution: str = ""  # e.g., "positive_agent succeeds on checkpoint X"

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "positive_agent": self.positive_agent,
            "negative_agent": self.negative_agent,
            "positive_score": self.positive_score,
            "negative_score": self.negative_score,
            "positive_trajectory_ref": self.positive_trajectory_ref,
            "negative_trajectory_ref": self.negative_trajectory_ref,
            "gap": self.gap,
            "disagreement_checkpoints": self.disagreement_checkpoints,
            "attribution": self.attribution,
        }


@dataclass
class TaskAggregate:
    """Aggregate statistics for one task across agents."""

    task: str
    n_records: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    best_agent: str
    worst_agent: str
    disagreement_score: float  # max_score - min_score

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "n_records": self.n_records,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "best_agent": self.best_agent,
            "worst_agent": self.worst_agent,
            "disagreement_score": self.disagreement_score,
        }


class DistributedEvalPool:
    """
    Shared pool of evaluations from multiple agents.

    Usage:
        pool = DistributedEvalPool()
        pool.add(DistributedEvalRecord(...))
        agg = pool.aggregate_by_task()
        pairs = pool.generate_cross_agent_pairs()
    """

    def __init__(self) -> None:
        self.records: list[DistributedEvalRecord] = []
        self._task_records: dict[str, list[DistributedEvalRecord]] = {}

    def add(self, record: DistributedEvalRecord) -> None:
        """Add a record to the pool."""
        self.records.append(record)
        self._task_records.setdefault(record.task, []).append(record)

    def add_from_store(
        self,
        store,
        agent_name: str,
        model_name: str,
        checkpoint_scores: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        """
        Add all records from a disteval RecordStore.

        checkpoint_scores: optional map {task: {checkpoint_id: score}}.
        """
        df = store.df()
        for _, row in df.iterrows():
            task = row["task"]
            cscores = (checkpoint_scores or {}).get(task, {})
            self.add(
                DistributedEvalRecord(
                    agent_name=agent_name,
                    model_name=model_name,
                    task=task,
                    score=float(row["score"]),
                    checkpoint_scores=cscores,
                    trajectory_ref=row.get("trajectory_ref"),
                    success=bool(row.get("success", False)),
                    failure_mode=row.get("failure_mode"),
                )
            )

    def tasks(self) -> list[str]:
        """Return all tasks with at least one record."""
        return sorted(self._task_records.keys())

    def agents(self) -> list[str]:
        """Return all agents with at least one record."""
        return sorted({r.agent_name for r in self.records})

    def aggregate_by_task(self) -> list[TaskAggregate]:
        """Aggregate scores per task across agents."""
        aggregates = []
        for task, records in self._task_records.items():
            scores = np.array([r.score for r in records], dtype=float)
            best_idx = int(np.argmax(scores))
            worst_idx = int(np.argmin(scores))
            aggregates.append(
                TaskAggregate(
                    task=task,
                    n_records=len(records),
                    mean_score=float(np.mean(scores)),
                    std_score=float(np.std(scores)),
                    min_score=float(np.min(scores)),
                    max_score=float(np.max(scores)),
                    best_agent=records[best_idx].agent_name,
                    worst_agent=records[worst_idx].agent_name,
                    disagreement_score=float(np.max(scores) - np.min(scores)),
                )
            )
        return sorted(aggregates, key=lambda a: -a.disagreement_score)

    def generate_cross_agent_pairs(
        self,
        min_gap: float = 0.1,
        require_checkpoints: bool = False,
    ) -> list[CrossAgentPair]:
        """
        Generate cross-agent training pairs for tasks with score disagreement.

        For each task with at least one high and one low scoring agent, produce
        a pair where the higher-scoring agent provides the positive trajectory
        and the lower-scoring agent provides the negative trajectory.

        Parameters
        ----------
        min_gap : float
            Minimum score gap to generate a pair.
        require_checkpoints : bool
            If True, only generate pairs when checkpoint breakdowns are available.

        Returns
        -------
        list[CrossAgentPair]
        """
        pairs: list[CrossAgentPair] = []
        for task, records in self._task_records.items():
            if len(records) < 2:
                continue

            # Use the best and worst agent on this task.
            best = max(records, key=lambda r: r.score)
            worst = min(records, key=lambda r: r.score)
            gap = best.score - worst.score

            if gap < min_gap:
                continue

            disagreement_checkpoints, attribution = self._attribute_disagreement(
                best, worst
            )

            if require_checkpoints and not disagreement_checkpoints:
                continue

            pairs.append(
                CrossAgentPair(
                    task=task,
                    positive_agent=best.agent_name,
                    negative_agent=worst.agent_name,
                    positive_score=best.score,
                    negative_score=worst.score,
                    positive_trajectory_ref=best.trajectory_ref,
                    negative_trajectory_ref=worst.trajectory_ref,
                    gap=gap,
                    disagreement_checkpoints=disagreement_checkpoints,
                    attribution=attribution,
                )
            )
        return sorted(pairs, key=lambda p: -p.gap)

    def _attribute_disagreement(
        self,
        positive: DistributedEvalRecord,
        negative: DistributedEvalRecord,
    ) -> tuple[list[str], str]:
        """Identify which checkpoints/sub-tasks drive the disagreement."""
        keys = set(positive.checkpoint_scores.keys()) | set(
            negative.checkpoint_scores.keys()
        )
        if not keys:
            return [], "No checkpoint breakdowns available"

        disagreements = []
        for k in sorted(keys):
            pos = positive.checkpoint_scores.get(k, 0.0)
            neg = negative.checkpoint_scores.get(k, 0.0)
            if pos > neg:
                disagreements.append(k)

        if disagreements:
            attribution = (
                f"{positive.agent_name} succeeds on "
                f"{', '.join(disagreements)} where {negative.agent_name} fails"
            )
        else:
            attribution = "Agents disagree only in total score, not in checkpoint breakdown"

        return disagreements, attribution

    def to_dict(self) -> dict:
        """Serialize the pool to a dict."""
        return {
            "n_records": len(self.records),
            "agents": self.agents(),
            "tasks": self.tasks(),
            "records": [r.to_dict() for r in self.records],
        }

    def save(self, path: str) -> None:
        """Save the pool to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str) -> "DistributedEvalPool":
        """Load the pool from JSON."""
        self.records = []
        self._task_records = {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data.get("records", []):
            self.add(
                DistributedEvalRecord(
                    agent_name=row["agent_name"],
                    model_name=row["model_name"],
                    task=row["task"],
                    score=row["score"],
                    checkpoint_scores=row.get("checkpoint_scores", {}),
                    trajectory_ref=row.get("trajectory_ref"),
                    success=row.get("success", False),
                    failure_mode=row.get("failure_mode"),
                    sub_task_depth=row.get("sub_task_depth", 0),
                )
            )
        return self

    def __len__(self) -> int:
        return len(self.records)
