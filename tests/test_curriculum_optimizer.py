"""Tests for disteval.curriculum_optimizer."""
from __future__ import annotations

from dataclasses import dataclass

from disteval.curriculum_optimizer import CurriculumValueIterator, MPCCurriculumPlanner


@dataclass
class _Task:
    task_id: str
    gap: float
    consistency: float


def test_value_iterator_prefers_high_gap_task():
    """With a single high-gap task, the planner should keep selecting it."""
    tasks = [
        _Task("t1", gap=0.4, consistency=0.5),
        _Task("t2", gap=0.1, consistency=0.5),
    ]
    planner = CurriculumValueIterator(tasks, alpha=0.4, gamma=0.99, n_bins=5)
    plan = planner.plan(initial_kappas=[0.5, 0.5], horizon=3)
    assert plan[0] == "t1"


def test_value_iterator_converges():
    """Value iteration should converge in a reasonable number of iterations."""
    tasks = [_Task("t1", gap=0.3, consistency=0.6)]
    planner = CurriculumValueIterator(tasks, alpha=0.4, gamma=0.9, epsilon=1e-3, n_bins=3)
    result = planner.solve(horizon=5)
    assert result["iterations"] < 1000
    assert len(result["V"]) > 0


def test_mpc_prefers_better_task():
    """MPC should pick the task with higher predicted gain."""
    tasks = [
        _Task("t1", gap=0.4, consistency=0.5),
        _Task("t2", gap=0.1, consistency=0.5),
    ]
    planner = MPCCurriculumPlanner(tasks, alpha=0.4, gamma=0.99, horizon=3)
    action = planner.next_action([0.5, 0.5])
    assert action == 0


def test_mpc_evaluates_sequence():
    """Cumulative reward should be positive for any non-empty sequence."""
    tasks = [_Task("t1", gap=0.3, consistency=0.6)]
    planner = MPCCurriculumPlanner(tasks, alpha=0.4, gamma=0.99, horizon=2)
    value = planner.evaluate_sequence([0.6], (0, 0))
    assert value > 0.0
