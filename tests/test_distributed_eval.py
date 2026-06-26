"""Tests for disteval.distributed_eval."""
from __future__ import annotations

import pytest

from disteval.distributed_eval import (
    DistributedEvalPool,
    DistributedEvalRecord,
)


def _make_pool() -> DistributedEvalPool:
    pool = DistributedEvalPool()
    pool.add(DistributedEvalRecord(agent_name="A", model_name="mA", task="t1", score=0.9))
    pool.add(DistributedEvalRecord(agent_name="A", model_name="mA", task="t2", score=0.8))
    pool.add(DistributedEvalRecord(agent_name="B", model_name="mB", task="t1", score=0.5))
    pool.add(DistributedEvalRecord(agent_name="B", model_name="mB", task="t2", score=0.4))
    return pool


# ---------------------------------------------------------------------------
# Consensus graph aggregation
# ---------------------------------------------------------------------------

class TestConsensusGraph:
    def test_ingest_and_build_consensus(self):
        pool = DistributedEvalPool()
        graph = {
            "sub_tasks": [
                {
                    "sub_task_id": "t1::p0",
                    "parent_task": "t1",
                    "phase_tag": "explore",
                    "entry_step": 3,
                    "exit_step": 8,
                    "source": "structural_divergence",
                    "confidence": 0.8,
                }
            ]
        }
        pool.ingest("agent-A", {"n_tasks": 1}, graph)
        pool.ingest("agent-B", {"n_tasks": 1}, graph)
        consensus = pool.build_consensus_graph(min_votes=2)
        assert len(consensus) == 1
        assert consensus[0].parent_task == "t1"
        assert consensus[0].phase_tag == "explore"
        assert consensus[0].n_votes == 2
        assert consensus[0].mean_confidence == pytest.approx(0.8)

    def test_consensus_requires_min_votes(self):
        pool = DistributedEvalPool()
        graph = {
            "sub_tasks": [
                {"parent_task": "t1", "phase_tag": "explore", "entry_step": 3, "exit_step": 8}
            ]
        }
        pool.ingest("agent-A", {}, graph)
        consensus = pool.build_consensus_graph(min_votes=2)
        assert consensus == []

    def test_boundary_tolerance_clusters(self):
        pool = DistributedEvalPool()
        pool.ingest("agent-A", {}, {
            "sub_tasks": [{"parent_task": "t1", "phase_tag": "explore", "entry_step": 3, "exit_step": 8}]
        })
        pool.ingest("agent-B", {}, {
            "sub_tasks": [{"parent_task": "t1", "phase_tag": "explore", "entry_step": 4, "exit_step": 9}]
        })
        consensus = pool.build_consensus_graph(min_votes=2, entry_tolerance=2)
        assert len(consensus) == 1
        assert consensus[0].entry_step == pytest.approx(3.5, abs=1.0)


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


# ---------------------------------------------------------------------------
# Robust aggregation and weighted consensus
# ---------------------------------------------------------------------------


def test_ivw_aggregation_reduces_to_mean_with_equal_variance():
    pool = _make_pool()
    agg_simple = {a.task: a.mean_score for a in pool.aggregate_by_task()}
    agg_ivw = {a.task: a.mean_score for a in pool.aggregate_by_task_ivw()}
    assert agg_simple == pytest.approx(agg_ivw)


def test_ivw_aggregation_weights_low_variance_agent_more():
    pool = DistributedEvalPool()
    # Agent A is consistent (low variance across tasks); agent B is noisy.
    for score in [0.81, 0.80, 0.79]:
        pool.add(DistributedEvalRecord(agent_name="A", model_name="mA", task="t1", score=score))
    for score in [0.95, 0.50, 0.65]:
        pool.add(DistributedEvalRecord(agent_name="B", model_name="mB", task="t1", score=score))
    agg = pool.aggregate_by_task_ivw()
    assert len(agg) == 1
    # IVW mean should be closer to A's mean (0.80) than B's mean (0.70).
    assert agg[0].mean_score > 0.75


def test_robust_aggregation_downweights_outlier():
    pool = DistributedEvalPool()
    for agent, score in [("A", 0.80), ("B", 0.81), ("C", 0.82), ("D", 0.79), ("E", 0.05)]:
        pool.add(DistributedEvalRecord(agent_name=agent, model_name=agent, task="t1", score=score))
    robust = pool.aggregate_by_task_robust(loss="huber")
    simple = pool.aggregate_by_task()
    assert robust[0].mean_score > simple[0].mean_score


def test_weighted_consensus_prefers_high_confidence():
    pool = DistributedEvalPool()
    pool.ingest("agent-A", {"n_tasks": 1}, {
        "sub_tasks": [
            {"parent_task": "t1", "phase_tag": "explore", "entry_step": 10, "exit_step": 20, "confidence": 0.9}
        ]
    })
    pool.ingest("agent-B", {"n_tasks": 1}, {
        "sub_tasks": [
            {"parent_task": "t1", "phase_tag": "explore", "entry_step": 4, "exit_step": 14, "confidence": 0.1}
        ]
    })
    consensus = pool.build_consensus_graph(min_votes=2, entry_tolerance=10, use_confidence_weights=True)
    assert len(consensus) == 1
    # Weighted median should be closer to 10 than to 4.
    assert consensus[0].entry_step > 7
