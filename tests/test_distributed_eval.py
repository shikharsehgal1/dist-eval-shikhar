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


def test_robust_center_point_gets_full_weight():
    """Regression: a score with residual exactly 0 (at the center) used to get
    IRLS weight ~0 instead of 1 (psi/(scaled+eps) at scaled=0). The estimate
    must treat the center point as maximally trusted."""
    pool = DistributedEvalPool()
    for agent, s in [("A", 0.0), ("B", 0.5), ("C", 1.0)]:
        pool.add(DistributedEvalRecord(agent_name=agent, model_name=agent, task="t", score=s))
    r = pool.aggregate_by_task_robust()
    assert r[0].mean_score == pytest.approx(0.5)
    # robust_std must include the center point's (zero) deviation at full
    # weight; with the old bug the center was dropped and std was inflated.
    assert r[0].std_score < 0.5


def test_robust_bisquare_loss_and_unknown_loss():
    pool = DistributedEvalPool()
    for agent, s in [("A", 0.80), ("B", 0.81), ("C", 0.82), ("D", 0.79), ("E", 0.05)]:
        pool.add(DistributedEvalRecord(agent_name=agent, model_name=agent, task="t", score=s))
    bis = pool.aggregate_by_task_robust(loss="bisquare")
    assert bis[0].mean_score > pool.aggregate_by_task()[0].mean_score
    with pytest.raises(ValueError):
        pool.aggregate_by_task_robust(loss="tukey-typo")


def test_cross_agent_pairs_require_distinct_agents():
    """Regression: with multiple episodes per agent, best/worst could both be
    the same agent — violating the cross-agent contract."""
    pool = DistributedEvalPool()
    # Agent A is flaky (1.0 and 0.0 episodes); agent B is steady at 0.6.
    pool.add(DistributedEvalRecord(agent_name="A", model_name="mA", task="t", score=1.0,
                                   trajectory_ref="a-hi"))
    pool.add(DistributedEvalRecord(agent_name="A", model_name="mA", task="t", score=0.0,
                                   trajectory_ref="a-lo"))
    pool.add(DistributedEvalRecord(agent_name="B", model_name="mB", task="t", score=0.6,
                                   trajectory_ref="b"))
    pairs = pool.generate_cross_agent_pairs(min_gap=0.1)
    assert len(pairs) == 1
    # Max cross-agent gap: B's steady 0.6 vs A's failed 0.0 episode (gap 0.6)
    # beats A's 1.0 vs B's 0.6 (gap 0.4). Agents must differ either way.
    assert pairs[0].positive_agent != pairs[0].negative_agent
    assert pairs[0].positive_agent == "B" and pairs[0].negative_agent == "A"
    assert pairs[0].gap == pytest.approx(0.6)
    assert pairs[0].negative_trajectory_ref == "a-lo"
    # opt-out reproduces the old raw best-vs-worst behavior (same-agent allowed)
    raw = pool.generate_cross_agent_pairs(min_gap=0.1, require_distinct_agents=False)
    assert raw[0].positive_agent == "A" and raw[0].negative_agent == "A"


def test_cross_agent_pairs_max_gap_under_ties():
    """Regression from a real-model run: when two agents tie at the max (both
    have a 1.0 episode), the pair generator must still find the max cross-agent
    gap (tie-holder's 1.0 vs the other agent's worst episode), not report gap 0."""
    pool = DistributedEvalPool()
    # opus: flaky — has both a perfect and a bad episode. haiku: perfect twice.
    pool.add(DistributedEvalRecord(agent_name="opus", model_name="o", task="t", score=1.0,
                                   trajectory_ref="o-hi"))
    pool.add(DistributedEvalRecord(agent_name="opus", model_name="o", task="t", score=0.2,
                                   trajectory_ref="o-lo"))
    pool.add(DistributedEvalRecord(agent_name="haiku", model_name="h", task="t", score=1.0,
                                   trajectory_ref="h-1"))
    pool.add(DistributedEvalRecord(agent_name="haiku", model_name="h", task="t", score=1.0,
                                   trajectory_ref="h-2"))
    pairs = pool.generate_cross_agent_pairs(min_gap=0.1)
    assert len(pairs) == 1
    assert pairs[0].positive_agent == "haiku"
    assert pairs[0].negative_agent == "opus"
    assert pairs[0].gap == pytest.approx(0.8)
    assert pairs[0].negative_trajectory_ref == "o-lo"


def test_cross_agent_pairs_skip_single_agent_task():
    pool = DistributedEvalPool()
    pool.add(DistributedEvalRecord(agent_name="A", model_name="mA", task="t", score=1.0))
    pool.add(DistributedEvalRecord(agent_name="A", model_name="mA", task="t", score=0.0))
    assert pool.generate_cross_agent_pairs(min_gap=0.1) == []


def test_weighted_median_basic_and_skewed():
    wm = DistributedEvalPool._weighted_median
    assert wm([1, 2, 3], [1.0, 1.0, 1.0]) == 2
    # A dominant weight pulls the median to its value.
    assert wm([1, 2, 30], [0.1, 0.1, 10.0]) == 30
    assert wm([5], [1.0]) == 5


def test_add_from_store_roundtrip():
    from disteval.records import EpisodeRecord, RecordStore

    store = RecordStore()
    for ep, score in enumerate([1.0, 0.5]):
        store.add(EpisodeRecord(run_id="r", model="m", task="t", episode=ep,
                                score=score, success=score >= 0.99))
    pool = DistributedEvalPool()
    pool.add_from_store(store, agent_name="agent-X", model_name="m")
    assert len(pool) == 2
    assert pool.agents() == ["agent-X"]
    scores = sorted(r.score for r in pool.records)
    assert scores == [0.5, 1.0]


def test_ivw_multiple_episodes_per_agent():
    """IVW with several episodes per agent per task: weights are per-agent,
    applied per-record; the aggregate must stay within the observed range."""
    pool = DistributedEvalPool()
    for s in [0.9, 0.8, 1.0, 0.85]:
        pool.add(DistributedEvalRecord(agent_name="steady", model_name="m", task="t", score=s))
    for s in [0.1, 0.9, 0.5, 0.99]:
        pool.add(DistributedEvalRecord(agent_name="noisy", model_name="m", task="t", score=s))
    agg = pool.aggregate_by_task_ivw()
    assert 0.1 <= agg[0].mean_score <= 1.0
    # The steady agent (lower variance) must pull the IVW mean above the naive mean.
    naive = pool.aggregate_by_task()[0].mean_score
    assert agg[0].mean_score > naive


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
