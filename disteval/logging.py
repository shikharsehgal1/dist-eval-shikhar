"""disteval.logging — structured observability for self-improvement cycles.

The recursive self-improvement loop produces a curriculum each cycle, but
researchers need to track whether the loop is actually improving the agent.
This module provides a lightweight, JSON/CSV-exportable cycle logger that
captures the key metrics and decisions without depending on external
observability backends.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class TaskLog:
    """One task's log entry within a cycle."""
    task: str
    kind: str
    gap: float
    priority_score: float
    difficulty: Optional[str] = None
    n_training_pairs: int = 0
    divergence_step: Optional[int] = None
    predicted_gain: Optional[float] = None


@dataclass
class CycleLog:
    """Full log for one self-improvement cycle."""
    cycle: int
    agent_name: str
    model_name: str
    started_at: str
    n_tasks_total: int = 0
    n_solid: int = 0
    n_recoverable: int = 0
    n_stuck: int = 0
    consistency_index: float = 0.0
    recoverable_score_left: float = 0.0
    predicted_total_gain: Optional[float] = None
    cycle_complete: bool = False
    recursion_enabled: bool = False
    n_decomposed: int = 0
    tasks: list[TaskLog] = field(default_factory=list)
    ended_at: Optional[str] = None
    delta_kappa: Optional[float] = None
    plateau_detected: bool = False


class CycleLogger:
    """Structured logger for self-improvement cycles.

    Usage::

        logger = CycleLogger(agent_name="agent-A", model_name="gpt-4")
        logger.log_cycle_start(1, n_tasks=10, kappa=0.65)
        logger.log_task_improvement("task-1", "recoverable", gap=0.3,
                                    priority_score=0.15)
        logger.log_cycle_end(1, kappa_new=0.72, delta_kappa=0.07,
                             plateau_detected=False)
        logger.export_json("run_log.json")
    """

    def __init__(self, agent_name: str = "agent", model_name: str = "unknown"):
        self.agent_name = agent_name
        self.model_name = model_name
        self.cycles: list[CycleLog] = []
        self._current: Optional[CycleLog] = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_cycle_start(self, cycle: int, n_tasks: int, kappa: float) -> None:
        """Start logging a new cycle."""
        self._current = CycleLog(
            cycle=cycle,
            agent_name=self.agent_name,
            model_name=self.model_name,
            started_at=self._now(),
            n_tasks_total=n_tasks,
            consistency_index=kappa,
        )

    def log_taxonomy(
        self,
        n_solid: int,
        n_recoverable: int,
        n_stuck: int,
        recoverable_score_left: float,
    ) -> None:
        """Record the right-tail taxonomy counts."""
        if self._current is None:
            return
        self._current.n_solid = n_solid
        self._current.n_recoverable = n_recoverable
        self._current.n_stuck = n_stuck
        self._current.recoverable_score_left = recoverable_score_left

    def log_task_improvement(
        self,
        task: str,
        kind: str,
        gap: float,
        priority_score: float,
        *,
        difficulty: Optional[str] = None,
        n_training_pairs: int = 0,
        divergence_step: Optional[int] = None,
        predicted_gain: Optional[float] = None,
    ) -> None:
        """Log a single task improvement decision."""
        if self._current is None:
            return
        self._current.tasks.append(
            TaskLog(
                task=task,
                kind=kind,
                gap=gap,
                priority_score=priority_score,
                difficulty=difficulty,
                n_training_pairs=n_training_pairs,
                divergence_step=divergence_step,
                predicted_gain=predicted_gain,
            )
        )

    def log_training_pair(
        self,
        task: str,
        pair_idx: int,
        divergence_step: int,
        reinforce_score: float,
        contrast_score: float,
    ) -> None:
        """Log the creation of a reinforce/contrast training pair."""
        if self._current is None:
            return
        # Store pair-level details in cycle metadata keyed by task.
        pairs = self._current.__dict__.setdefault("training_pairs", [])
        pairs.append(
            {
                "task": task,
                "pair_idx": pair_idx,
                "divergence_step": divergence_step,
                "reinforce_score": reinforce_score,
                "contrast_score": contrast_score,
            }
        )

    def log_cycle_end(
        self,
        cycle: int,
        kappa_new: float,
        delta_kappa: float,
        plateau_detected: bool,
        *,
        predicted_total_gain: Optional[float] = None,
        cycle_complete: bool = False,
        recursion_enabled: bool = False,
        n_decomposed: int = 0,
    ) -> None:
        """Finalize the current cycle log."""
        if self._current is None:
            self.log_cycle_start(cycle, n_tasks=0, kappa=kappa_new)
        assert self._current is not None
        self._current.ended_at = self._now()
        self._current.consistency_index = kappa_new
        self._current.delta_kappa = delta_kappa
        self._current.plateau_detected = plateau_detected
        self._current.predicted_total_gain = predicted_total_gain
        self._current.cycle_complete = cycle_complete
        self._current.recursion_enabled = recursion_enabled
        self._current.n_decomposed = n_decomposed
        self.cycles.append(self._current)
        self._current = None

    def export_json(self, path: str) -> dict:
        """Export the full cycle history to JSON and return it."""
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    def export_csv(self, path: str) -> None:
        """Export one flat CSV row per task per cycle."""
        rows = []
        for cycle in self.cycles:
            base = {
                "cycle": cycle.cycle,
                "agent_name": cycle.agent_name,
                "model_name": cycle.model_name,
                "started_at": cycle.started_at,
                "n_tasks_total": cycle.n_tasks_total,
                "n_solid": cycle.n_solid,
                "n_recoverable": cycle.n_recoverable,
                "n_stuck": cycle.n_stuck,
                "consistency_index": cycle.consistency_index,
                "delta_kappa": cycle.delta_kappa,
                "plateau_detected": cycle.plateau_detected,
                "predicted_total_gain": cycle.predicted_total_gain,
                "cycle_complete": cycle.cycle_complete,
                "recursion_enabled": cycle.recursion_enabled,
            }
            for task in cycle.tasks:
                row = {**base}
                row.update(
                    {
                        "task": task.task,
                        "kind": task.kind,
                        "difficulty": task.difficulty,
                        "gap": task.gap,
                        "priority_score": task.priority_score,
                        "n_training_pairs": task.n_training_pairs,
                        "divergence_step": task.divergence_step,
                        "predicted_gain": task.predicted_gain,
                    }
                )
                rows.append(row)
        if not rows:
            return
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def to_dict(self) -> dict[str, Any]:
        """Return the full cycle history as a plain dict."""
        def _task_to_dict(t: TaskLog) -> dict:
            return {
                "task": t.task,
                "kind": t.kind,
                "difficulty": t.difficulty,
                "gap": t.gap,
                "priority_score": t.priority_score,
                "n_training_pairs": t.n_training_pairs,
                "divergence_step": t.divergence_step,
                "predicted_gain": t.predicted_gain,
            }

        def _cycle_to_dict(c: CycleLog) -> dict:
            return {
                "cycle": c.cycle,
                "agent_name": c.agent_name,
                "model_name": c.model_name,
                "started_at": c.started_at,
                "ended_at": c.ended_at,
                "n_tasks_total": c.n_tasks_total,
                "n_solid": c.n_solid,
                "n_recoverable": c.n_recoverable,
                "n_stuck": c.n_stuck,
                "consistency_index": c.consistency_index,
                "recoverable_score_left": c.recoverable_score_left,
                "delta_kappa": c.delta_kappa,
                "plateau_detected": c.plateau_detected,
                "predicted_total_gain": c.predicted_total_gain,
                "cycle_complete": c.cycle_complete,
                "recursion_enabled": c.recursion_enabled,
                "n_decomposed": c.n_decomposed,
                "tasks": [_task_to_dict(t) for t in c.tasks],
            }

        return {
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "n_cycles": len(self.cycles),
            "cycles": [_cycle_to_dict(c) for c in self.cycles],
        }
