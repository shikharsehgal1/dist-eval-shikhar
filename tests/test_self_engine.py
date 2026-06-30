"""Integration tests for disteval.self_engine."""
import os
from pathlib import Path

import pytest

from disteval.self_engine import SelfEngine


JOBS_ROOT = Path(__file__).parent.parent / "jobs"


class TestSelfEngineFromJobDirs:
    @pytest.mark.integration
    @pytest.mark.requires_harbor
    def test_loads_from_real_harbor_dir(self):
        job_dir = str(JOBS_ROOT / "run_A" / "disteval-run-A")
        if not os.path.isdir(job_dir):
            pytest.skip("Harbor job directory not present")
        engine = SelfEngine.from_job_dirs([job_dir])
        assert len(engine.store) > 0
        assert engine.agent_name == "agent"
        assert engine.monitor is not None

    @pytest.mark.integration
    @pytest.mark.requires_harbor
    def test_run_cycle_returns_plan(self):
        job_dir = str(JOBS_ROOT / "run_A" / "disteval-run-A")
        if not os.path.isdir(job_dir):
            pytest.skip("Harbor job directory not present")
        engine = SelfEngine.from_job_dirs([job_dir])
        plan = engine.run_cycle(1)
        assert plan.agent_name == "agent"
        assert plan.cycle == 1
        assert plan.n_tasks_total > 0
        assert plan.consistency_index >= 0.0
        assert plan.summary() is not None
        assert isinstance(plan.to_dict(), dict)

    def test_run_cycle_with_logger_does_not_crash(self):
        """Regression: _build_curriculum referenced a nonexistent
        TaskImprovement.divergence_step, crashing run_cycle whenever a logger
        was attached (no test exercised the logger path). Build an engine
        directly with injected (empty) monitor/memory and a synthetic store
        containing one RECOVERABLE task."""
        from disteval.logging import CycleLogger
        from disteval.records import EpisodeRecord, RecordStore
        from disteval.trajectory_memory import TrajectoryMemory

        store = RecordStore()
        # A RECOVERABLE task: high peak (q*≈1) but inconsistent mean.
        for ep, score in enumerate([1.0, 0.0, 1.0, 0.0]):
            store.add(EpisodeRecord(
                run_id="r0", model="m", task="flaky_task", episode=ep,
                score=score, success=score >= 0.99,
            ))

        logger = CycleLogger(agent_name="a", model_name="m")
        # Inject monitor=object() / empty memory so __init__ skips Harbor loading;
        # with no trajectory records the curriculum has empty training_pairs,
        # which is exactly the branch the divergence_step fix must handle.
        engine = SelfEngine(
            store=store, job_dirs=[], agent_name="a", model_name="m",
            monitor=object(), memory=TrajectoryMemory(), logger=logger,
        )
        plan = engine.run_cycle(1)  # must not raise AttributeError
        assert plan.cycle == 1
        assert plan.n_recoverable >= 1
        assert logger.cycles, "logger should have captured the cycle"
        assert logger.cycles[0].tasks, "logger should have captured the task"

    def test_plan_serialisation_roundtrip(self):
        # Minimal unit test: build a plan directly without filesystem deps.
        from disteval.self_engine import SelfImprovementPlan
        plan = SelfImprovementPlan(
            agent_name="a",
            model_name="m",
            cycle=1,
            n_tasks_total=1,
            n_solid=1,
            n_recoverable=0,
            n_stuck=0,
            consistency_index=0.9,
            recoverable_score_left=0.0,
            curriculum=[],
        )
        d = plan.to_dict()
        assert d["agent_name"] == "a"
        assert d["cycle"] == 1
        assert "consistency_index" in d
