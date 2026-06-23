"""Tests for disteval.environment_registry lifecycle management."""
import json
import os
import tempfile

from disteval.environment_generator import (
    GenEnv,
    InitialState,
    RewardSpec,
    TerminationSpec,
)
from disteval.environment_registry import EnvironmentRegistry


def make_env(task_id: str, parent: str = "root", boundary_hash: str = "abc") -> GenEnv:
    return GenEnv(
        task_id=task_id,
        parent_task=parent,
        sub_task_depth=1,
        instruction="do thing",
        entry_step=0,
        exit_step=5,
        phase_tag="phase-1",
        reward=RewardSpec(type="checkpoint", weight=1.0, success_threshold=0.99),
        initial_state=InitialState(),
        termination=TerminationSpec(),
        metadata={"boundary_hash": boundary_hash},
    )


class TestRegistryLifecycle:
    def test_retire_marks_status(self):
        reg = EnvironmentRegistry()
        env = make_env("t1")
        reg.register(env)
        assert reg.retire("t1", reason="solid") is True
        assert env.metadata["status"] == "retired"
        assert env.metadata["retired_reason"] == "solid"
        assert "retired_at" in env.metadata

    def test_retire_missing_returns_false(self):
        reg = EnvironmentRegistry()
        assert reg.retire("missing") is False

    def test_recompute_status(self):
        reg = EnvironmentRegistry()
        env = make_env("t1")
        reg.register(env)
        assert reg.recompute_status("t1", {"kind": "recoverable", "gap": 0.3}) is True
        assert env.metadata["status"] == "recoverable"
        assert env.metadata["last_profile"]["kind"] == "recoverable"

    def test_invalidate_by_boundary_hash(self):
        reg = EnvironmentRegistry()
        reg.register(make_env("t1", boundary_hash="h1"))
        reg.register(make_env("t2", boundary_hash="h1"))
        reg.register(make_env("t3", boundary_hash="h2"))
        removed = reg.invalidate_by_boundary_hash("h1")
        assert sorted(removed) == ["t1", "t2"]
        assert len(reg) == 1

    def test_export_curriculum_skips_retired(self):
        reg = EnvironmentRegistry()
        reg.register(make_env("t1"))
        reg.register(make_env("t2"))
        reg.retire("t1")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            curriculum = reg.export_curriculum(path)
            assert curriculum["n_active"] == 1
            assert curriculum["n_retired"] == 1
            assert len(curriculum["environments"]) == 1
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["n_active"] == 1
        finally:
            if os.path.exists(path):
                os.unlink(path)
