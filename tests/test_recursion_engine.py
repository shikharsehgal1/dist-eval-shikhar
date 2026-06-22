"""Tests for disteval.recursion_engine."""
from __future__ import annotations

import pytest

from disteval.recursion_engine import (
    RecursionEngine,
    RecursionEngineConfig,
    RMDPNode,
)
from disteval.right_tail import RightTailReport, TaskOutcomeProfile, task_outcome_profile


class FakeRecord:
    def __init__(self, task_path: str, traj_path: str, score: float, tool_sequence: list):
        self.task_path = task_path
        self.traj_path = traj_path
        self.score = score
        self.tool_sequence = tool_sequence
        self.n_steps = len(tool_sequence)
        self.n_exec = sum(1 for t in tool_sequence if t in ("run_shell_command", "exec_command"))
        self.n_search = 0
        self.first_write_pos = 0


class FakeMonitor:
    """Minimal monitor stand-in for unit tests."""

    def __init__(self, records: list = None):
        self.records = records or []

    def find_phase_boundaries(self, traj_path: str, **kwargs):
        # Return deterministic boundaries for unit tests.
        return [
            {"step_index": 1, "tool_name": "read_file", "p_high": 0.5, "phase_tag": "read", "confidence": 0.5},
            {"step_index": 3, "tool_name": "write_file", "p_high": 0.8, "phase_tag": "write", "confidence": 0.8},
            {"step_index": 5, "tool_name": "read_file", "p_high": 0.6, "phase_tag": "read", "confidence": 0.6},
            {"step_index": 6, "tool_name": "run_shell_command", "p_high": 0.85, "phase_tag": "exec", "confidence": 0.85},
            {"step_index": 8, "tool_name": "write_file", "p_high": 0.9, "phase_tag": "write", "confidence": 0.9},
        ]

    def load_trajectory_steps(self, traj_path: str):
        return []


@pytest.fixture
def profiles() -> list[TaskOutcomeProfile]:
    return [
        task_outcome_profile("tasks/easy-1", [0.0, 0.0, 0.0], "m"),
        task_outcome_profile("tasks/easy-2", [1.0, 0.0, 1.0], "m"),
        task_outcome_profile("tasks/medium-2", [0.8, 0.5, 0.6], "m"),
    ]


@pytest.fixture
def report(profiles) -> RightTailReport:
    return RightTailReport(
        model="m",
        n_tasks=len(profiles),
        n_episodes=9,
        profiles=profiles,
        n_solid=0,
        n_recoverable=2,
        n_stuck=1,
        total_gap=sum(p.gap for p in profiles),
        pct_recoverable=2 / 3,
        recoverable_score_left=sum(p.gap for p in profiles),
        sum_q_star=sum(p.q_star for p in profiles),
        sum_q_bar=sum(p.q_bar for p in profiles),
        priority_tasks=[p for p in profiles if p.kind == "recoverable"],
    )


@pytest.fixture
def engine(report) -> RecursionEngine:
    records = [
        FakeRecord("easy-1", "t1", 0.0, ["read_file", "read_file", "write_file", "run_shell_command", "run_shell_command"]),
        FakeRecord("easy-2", "t2", 1.0, ["write_file", "run_shell_command", "write_file"]),
        FakeRecord("medium-2", "t3", 0.8, ["read_file", "read_file", "write_file", "run_shell_command", "write_file", "run_shell_command"]),
    ]
    return RecursionEngine(
        monitor=FakeMonitor(records),
        memory=None,
        config=RecursionEngineConfig(max_depth=2),
        agent_name="test",
        model_name="m",
    )


def test_decompose_stuck_task_produces_sub_tasks(engine, report):
    graph = engine.decompose(report)
    # easy-1 is STUCK and should be decomposed.
    easy1_subs = [s for s in graph.sub_tasks if s.parent_task == "tasks/easy-1"]
    assert len(easy1_subs) > 0
    for sub in easy1_subs:
        assert sub.sub_task_depth == 1
        assert sub.estimated_q_star == 0.0  # stuck parent => all sub-tasks stuck
        assert sub.kind == "stuck"


def test_decompose_recoverable_task_produces_weighted_sub_tasks(engine, report):
    graph = engine.decompose(report)
    medium_subs = [s for s in graph.sub_tasks if s.parent_task == "tasks/medium-2"]
    assert len(medium_subs) > 0
    # We should have checkpoint weights from the test suite parser.
    weights = [s.weight for s in medium_subs]
    assert sum(weights) == pytest.approx(1.0, abs=1e-3)


def test_sub_task_graph_serializable(engine, report):
    graph = engine.decompose(report)
    d = graph.to_dict()
    assert "parent_tasks" in d
    assert "sub_tasks" in d
    assert "edges" in d
    assert any(e[0] == "tasks/medium-2" for e in d["edges"])


def test_compute_recursive_gap(engine, report):
    graph = engine.decompose(report)
    root = RMDPNode(
        task="tasks/medium-2",
        profile=report.profiles[2],
        children=[
            RMDPNode(
                task=s.sub_task_id,
                profile=graph.profiles[s.sub_task_id],
                sub_task=s,
            )
            for s in graph.sub_tasks
            if s.parent_task == "tasks/medium-2"
        ],
    )
    gap = engine.compute_recursive_gap(root)
    assert gap >= 0.0
    assert gap <= sum(p.gap for p in report.profiles) + 1e-6


def test_checkpoint_specs_loaded(engine):
    assert "medium-2" in engine._checkpoint_specs
    assert len(engine._checkpoint_specs["medium-2"]) == 5


def test_sub_task_graph_save_load(tmp_path, engine, report):
    graph = engine.decompose(report)
    path = tmp_path / "graph.json"
    graph.save(str(path))
    assert path.exists()
    assert len(path.read_text()) > 0


def test_recursion_depth_cap(engine):
    # With max_depth=2, a stuck task should not produce depth > 2 sub-tasks.
    deep_profiles = [
        task_outcome_profile("tasks/easy-1", [0.0] * 5, "m")
    ]
    deep_report = RightTailReport(
        model="m",
        n_tasks=1,
        n_episodes=5,
        profiles=deep_profiles,
        n_solid=0,
        n_recoverable=0,
        n_stuck=1,
        total_gap=0.0,
        pct_recoverable=0.0,
        recoverable_score_left=0.0,
        sum_q_star=0.0,
        sum_q_bar=0.0,
        priority_tasks=[],
    )
    graph = engine.decompose(deep_report)
    max_depth = max((s.sub_task_depth for s in graph.sub_tasks), default=0)
    assert max_depth <= engine.config.max_depth
