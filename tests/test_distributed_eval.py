"""Tests for disteval.distributed_eval."""
from __future__ import annotations

import pytest

from disteval.distributed_eval import (
    DistributedEvalPool,
    DistributedEvalRecord,
)


@pytest.fixture
def pool() -> DistributedEvalPool:
    p = DistributedEvalPool()
    p.add(
        DistributedEvalRecord(
            agent_name="agent-A",
            model_name="model-A",
            task="tasks/easy-1",
            score=1.0,
            checkpoint_scores={"ck0": 1.0, "ck1": 1.0},
            trajectory_ref="traj-A-1",
        )
    )
    p.add(
        DistributedEvalRecord(
            agent_name="agent-B",
            model_name="model-B",
            task="tasks/easy-1",
            score=0.0,
            checkpoint_scores={"ck0": 0.0, "ck1": 0.0},
            trajectory_ref="traj-B-1",
        )
    )
    p.add(
        DistributedEvalRecord(
            agent_name="agent-A",
            model_name="model-A",
            task="tasks/easy-2",
            score=1.0,
            checkpoint_scores={"ck0": 1.0},
            trajectory_ref="traj-A-2",
        )
    )
    p.add(
        DistributedEvalRecord(
            agent_name="agent-B",
            model_name="model-B",
            task="tasks/easy-2",
            score=0.9,
            checkpoint_scores={"ck0": 0.9},
            trajectory_ref="traj-B-2",
        )
    )
    return p


def test_aggregate_by_task(pool):
    aggregates = pool.aggregate_by_task()
    assert len(aggregates) == 2
    easy1 = next(a for a in aggregates if a.task == "tasks/easy-1")
    assert easy1.best_agent == "agent-A"
    assert easy1.worst_agent == "agent-B"
    assert easy1.disagreement_score == pytest.approx(1.0)


def test_generate_cross_agent_pairs(pool):
    pairs = pool.generate_cross_agent_pairs(min_gap=0.1)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.task == "tasks/easy-1"
    assert pair.positive_agent == "agent-A"
    assert pair.negative_agent == "agent-B"
    assert pair.gap == pytest.approx(1.0)
    assert pair.positive_trajectory_ref == "traj-A-1"
    assert pair.negative_trajectory_ref == "traj-B-1"


def test_disagreement_attribution(pool):
    pairs = pool.generate_cross_agent_pairs(min_gap=0.1)
    assert len(pairs) == 1
    pair = pairs[0]
    assert "ck0" in pair.disagreement_checkpoints
    assert "ck1" in pair.disagreement_checkpoints
    assert "agent-A" in pair.attribution
    assert "agent-B" in pair.attribution


def test_pool_save_load(tmp_path, pool):
    path = tmp_path / "pool.json"
    pool.save(str(path))
    new_pool = DistributedEvalPool()
    new_pool.load(str(path))
    assert len(new_pool) == len(pool)
    assert set(new_pool.agents()) == set(pool.agents())
    assert set(new_pool.tasks()) == set(pool.tasks())


def test_no_pairs_when_agreement(pool):
    pool.add(
        DistributedEvalRecord(
            agent_name="agent-C",
            model_name="model-C",
            task="tasks/easy-2",
            score=0.95,
            checkpoint_scores={"ck0": 0.95},
            trajectory_ref="traj-C-2",
        )
    )
    pairs = pool.generate_cross_agent_pairs(min_gap=0.2)
    # easy-2 max-min gap is now 0.1 (1.0 - 0.9), which is < 0.2.
    assert all(p.task != "tasks/easy-2" for p in pairs)


def test_require_checkpoints(pool):
    pool.add(
        DistributedEvalRecord(
            agent_name="agent-C",
            model_name="model-C",
            task="tasks/hard-1",
            score=1.0,
            checkpoint_scores={},
            trajectory_ref="traj-C-3",
        )
    )
    pool.add(
        DistributedEvalRecord(
            agent_name="agent-D",
            model_name="model-D",
            task="tasks/hard-1",
            score=0.0,
            checkpoint_scores={},
            trajectory_ref="traj-D-3",
        )
    )
    pairs = pool.generate_cross_agent_pairs(min_gap=0.1, require_checkpoints=True)
    assert all(p.task != "tasks/hard-1" for p in pairs)
