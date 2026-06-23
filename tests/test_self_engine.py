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
