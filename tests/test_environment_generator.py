"""Tests for disteval.environment_generator and environment_registry."""
from __future__ import annotations

import numpy as np
import pytest

from disteval.environment_generator import EnvironmentGenerator
from disteval.environment_registry import EnvironmentRegistry
from disteval.recursion_engine import (
    RecursionEngine,
    RecursionEngineConfig,
    SubTaskGraph,
)
from disteval.right_tail import task_outcome_profile
from disteval.training_sim import simulate_recursive_gains


class FakeMonitor:
    def __init__(self, records=None):
        self.records = records or []

    def find_phase_boundaries(self, traj_path, **kwargs):
        return [
            {"step_index": 1, "tool_name": "read_file", "p_high": 0.5, "phase_tag": "read", "confidence": 0.5},
            {"step_index": 3, "tool_name": "write_file", "p_high": 0.8, "phase_tag": "write", "confidence": 0.8},
            {"step_index": 6, "tool_name": "run_shell_command", "p_high": 0.85, "phase_tag": "exec", "confidence": 0.85},
        ]

    def load_trajectory_steps(self, traj_path):
        return []


class FakeRecord:
    def __init__(self, task_path, traj_path, score, tool_sequence):
        self.task_path = task_path
        self.traj_path = traj_path
        self.score = score
        self.tool_sequence = tool_sequence
        self.n_steps = len(tool_sequence)
        self.n_exec = 0
        self.n_search = 0
        self.first_write_pos = 0


@pytest.fixture
def sub_task_graph() -> SubTaskGraph:
    engine = RecursionEngine(
        monitor=FakeMonitor([
            FakeRecord("easy-1", "t1", 0.0, ["read_file", "write_file", "run_shell_command"]),
            FakeRecord("easy-2", "t2", 1.0, ["write_file", "run_shell_command"]),
            FakeRecord("medium-2", "t3", 0.8, ["read_file", "write_file", "run_shell_command"]),
        ]),
        config=RecursionEngineConfig(max_depth=2),
        agent_name="test",
        model_name="m",
    )
    profiles = [
        task_outcome_profile("tasks/easy-1", [0.0, 0.0], "m"),
        task_outcome_profile("tasks/easy-2", [1.0, 1.0], "m"),
        task_outcome_profile("tasks/medium-2", [0.8, 0.5], "m"),
    ]
    report = type("R", (), {
        "profiles": profiles,
        "priority_tasks": [p for p in profiles if p.kind == "recoverable"],
        "n_tasks": len(profiles),
        "n_solid": 1,
        "n_recoverable": 1,
        "n_stuck": 1,
    })()
    return engine.decompose(report)


def test_generate_environments(sub_task_graph):
    generator = EnvironmentGenerator()
    bundle = generator.generate(sub_task_graph)
    assert len(bundle.environments) > 0
    env = bundle.environments[0]
    assert env.task_id
    assert env.parent_task
    assert env.instruction
    assert env.reward.type in ("checkpoint", "structural")


def test_bundle_serialization(sub_task_graph, tmp_path):
    generator = EnvironmentGenerator()
    bundle = generator.generate(sub_task_graph)
    path = tmp_path / "bundle.json"
    bundle.save(str(path))
    assert path.exists()


def test_registry_register_and_lookup(sub_task_graph):
    generator = EnvironmentGenerator()
    bundle = generator.generate(sub_task_graph)
    registry = EnvironmentRegistry()
    registry.register_bundle(bundle)
    assert len(registry) == len(bundle.environments)
    for env in bundle.environments:
        assert env.task_id in registry
        looked_up = registry.get(env.task_id)
        assert looked_up.task_id == env.task_id
        assert looked_up.parent_task == env.parent_task


def test_registry_save_and_load(sub_task_graph, tmp_path):
    generator = EnvironmentGenerator()
    bundle = generator.generate(sub_task_graph)
    registry = EnvironmentRegistry()
    registry.register_bundle(bundle)
    path = tmp_path / "registry.json"
    registry.save(str(path))
    new_registry = EnvironmentRegistry()
    new_registry.load(str(path))
    assert len(new_registry) == len(registry)
    assert sorted(new_registry.list_task_ids()) == sorted(registry.list_task_ids())


def test_simulate_recursive_gains():
    scores = {
        "t1": np.array([0.0, 0.0, 0.0]),
        "t2": np.array([0.8, 0.5, 0.6]),
    }
    kinds = {"t1": "stuck", "t2": "recoverable"}
    updated = simulate_recursive_gains(scores, kinds, max_rounds=3)
    assert "t1" in updated
    assert "t2" in updated
    assert all(updated["t2"] >= 0)
    assert all(updated["t2"] <= 1.0)
