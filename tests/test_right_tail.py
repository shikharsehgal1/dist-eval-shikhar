"""Tests for disteval.right_tail — right-tail training signal."""
import pytest
import pandas as pd

from disteval.records import EpisodeRecord, RecordStore
from disteval.right_tail import (
    task_outcome_profile,
    right_tail_analysis,
    compare_right_tail,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_store(task_attempts: dict[str, list[float]], model: str = "agent") -> RecordStore:
    """Build a RecordStore from a dict of task → score list."""
    store = RecordStore()
    for task, scores in task_attempts.items():
        for i, score in enumerate(scores):
            store.add(EpisodeRecord(
                run_id="run0", model=model, task=task,
                episode=i, score=score, success=score >= 0.99,
            ))
    return store


# ── task_outcome_profile ──────────────────────────────────────────────────────

class TestTaskOutcomeProfile:

    def test_solid_all_ones(self):
        p = task_outcome_profile("t", [1.0, 1.0, 1.0], model="a")
        assert p.kind == "solid"
        assert p.q_star == pytest.approx(1.0)
        assert p.q_bar  == pytest.approx(1.0)
        assert p.gap    == pytest.approx(0.0)
        assert p.consistency == pytest.approx(1.0)

    def test_solid_uniform_partial(self):
        p = task_outcome_profile("t", [0.5, 0.5], model="a")
        assert p.kind == "solid"
        assert p.q_star == pytest.approx(0.5)
        assert p.consistency == pytest.approx(1.0)

    def test_recoverable_mixed(self):
        p = task_outcome_profile("t", [0.0, 1.0, 1.0], model="a")
        assert p.kind == "recoverable"
        assert p.q_star == pytest.approx(1.0)
        assert p.q_bar  == pytest.approx(2/3)
        assert p.gap    == pytest.approx(1/3)
        assert p.consistency == pytest.approx(2/3)

    def test_recoverable_single_high(self):
        p = task_outcome_profile("t", [0.0, 0.0, 1.0], model="a")
        assert p.kind == "recoverable"
        assert p.q_star == pytest.approx(1.0)
        assert p.gap    == pytest.approx(2/3)
        assert p.consistency == pytest.approx(1/3)

    def test_stuck_all_zeros(self):
        p = task_outcome_profile("t", [0.0, 0.0, 0.0], model="a")
        assert p.kind == "stuck"
        assert p.q_star == pytest.approx(0.0)
        assert p.consistency == pytest.approx(0.0)

    def test_residuals_correct(self):
        p = task_outcome_profile("t", [0.0, 0.5, 1.0], model="a")
        assert p.residuals == pytest.approx([1.0, 0.5, 0.0])

    def test_reinforce_contrast_split(self):
        # threshold=0.9 → attempts >= 0.9*q_star=0.9 are reinforce
        p = task_outcome_profile("t", [0.0, 0.5, 1.0], model="a",
                                 reinforce_threshold=0.9)
        assert 2 in p.reinforce_idx   # score=1.0 ≥ 0.9
        assert 0 in p.contrast_idx    # score=0.0 < 0.9
        assert 1 in p.contrast_idx    # score=0.5 < 0.9

    def test_reinforce_threshold_lower(self):
        # With threshold=0.5, score=0.5 qualifies as reinforce (0.5 >= 0.5*1.0)
        p = task_outcome_profile("t", [0.0, 0.5, 1.0], model="a",
                                 reinforce_threshold=0.5)
        assert 1 in p.reinforce_idx
        assert 2 in p.reinforce_idx
        assert 0 in p.contrast_idx

    def test_single_attempt(self):
        p = task_outcome_profile("t", [0.7], model="a")
        assert p.kind == "solid"
        assert p.q_star == pytest.approx(0.7)
        assert p.gap    == pytest.approx(0.0)

    def test_difficulty_stored(self):
        p = task_outcome_profile("t", [1.0], model="a", difficulty="hard")
        assert p.difficulty == "hard"

    def test_gap_non_negative(self):
        for scores in [[0.0, 0.5], [1.0, 0.0], [0.3, 0.3, 0.3]]:
            p = task_outcome_profile("t", scores, model="a")
            assert p.gap >= -1e-12

    def test_stuck_task_has_empty_reinforce_and_contrast(self):
        p = task_outcome_profile("t", [0.0, 0.0, 0.0], model="a")
        assert p.kind == "stuck"
        assert p.reinforce_idx == []
        assert p.contrast_idx == []


# ── right_tail_analysis ───────────────────────────────────────────────────────

class TestRightTailAnalysis:

    def _store_all_solid(self):
        return make_store({
            "task-a": [1.0, 1.0, 1.0],
            "task-b": [0.5, 0.5],
        })

    def _store_mixed(self):
        return make_store({
            "task-easy":   [1.0, 1.0, 1.0],   # solid
            "task-medium": [0.0, 1.0, 1.0],   # recoverable
            "task-hard":   [0.0, 0.0, 0.0],   # stuck
        })

    def test_all_solid(self):
        store = self._store_all_solid()
        r = right_tail_analysis(store)
        assert r.n_solid == 2
        assert r.n_recoverable == 0
        assert r.n_stuck == 0
        assert r.total_gap == pytest.approx(0.0)
        assert r.priority_tasks == []

    def test_mixed_classification(self):
        store = self._store_mixed()
        r = right_tail_analysis(store)
        assert r.n_solid == 1
        assert r.n_recoverable == 1
        assert r.n_stuck == 1

    def test_total_gap_correct(self):
        store = make_store({"t": [0.0, 1.0]})
        r = right_tail_analysis(store)
        # q_star=1.0, q_bar=0.5, gap=0.5
        assert r.total_gap == pytest.approx(0.5)

    def test_sum_q_star_and_q_bar(self):
        store = make_store({
            "t1": [1.0, 1.0],         # q_star=1, q_bar=1
            "t2": [0.0, 0.0, 1.0],   # q_star=1, q_bar=1/3
        })
        r = right_tail_analysis(store)
        assert r.sum_q_star == pytest.approx(2.0)
        assert r.sum_q_bar  == pytest.approx(1 + 1/3)

    def test_consistency_index(self):
        store = make_store({
            "t1": [1.0, 1.0],
            "t2": [0.0, 1.0],
        })
        r = right_tail_analysis(store)
        # sum_q_star = 2.0, sum_q_bar = 1.0 + 0.5 = 1.5
        assert r.sum_q_star == pytest.approx(2.0)
        assert r.sum_q_bar  == pytest.approx(1.5)

    def test_priority_tasks_sorted_by_gap(self):
        store = make_store({
            "small-gap": [0.9, 1.0],   # gap=0.05
            "big-gap":   [0.0, 1.0],   # gap=0.5
        })
        r = right_tail_analysis(store)
        assert r.priority_tasks[0].task == "big-gap"
        assert r.priority_tasks[1].task == "small-gap"

    def test_model_name_override(self):
        store = make_store({"t": [1.0]})
        r = right_tail_analysis(store, model_name="my-agent")
        assert r.model == "my-agent"

    def test_empty_store_raises(self):
        with pytest.raises(ValueError):
            right_tail_analysis(RecordStore())

    def test_n_episodes_count(self):
        store = make_store({
            "t1": [1.0, 0.5, 0.0],
            "t2": [1.0, 1.0],
        })
        r = right_tail_analysis(store)
        assert r.n_episodes == 5

    def test_n_tasks_count(self):
        store = make_store({
            "t1": [1.0],
            "t2": [0.5],
            "t3": [0.0],
        })
        r = right_tail_analysis(store)
        assert r.n_tasks == 3

    def test_pct_recoverable(self):
        store = make_store({
            "solid": [1.0, 1.0],
            "recov": [0.0, 1.0],
        })
        r = right_tail_analysis(store)
        assert r.pct_recoverable == pytest.approx(0.5)


# ── compare_right_tail ────────────────────────────────────────────────────────

class TestCompareRightTail:

    def test_returns_dataframe(self):
        s1 = make_store({"t": [1.0, 1.0]}, model="A")
        s2 = make_store({"t": [0.0, 1.0]}, model="B")
        r1 = right_tail_analysis(s1, "A")
        r2 = right_tail_analysis(s2, "B")
        df = compare_right_tail([r1, r2])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_columns_present(self):
        s = make_store({"t": [1.0]}, model="X")
        r = right_tail_analysis(s, "X")
        df = compare_right_tail([r])
        assert "model" in df.columns
        assert "consistency_index" in df.columns
        assert "total_gap" in df.columns
        assert "n_recoverable" in df.columns

    def test_sorted_by_consistency_desc(self):
        # A is fully consistent, B has gap
        s_a = make_store({"t": [1.0, 1.0]}, model="A")
        s_b = make_store({"t": [0.0, 1.0]}, model="B")
        r_a = right_tail_analysis(s_a, "A")
        r_b = right_tail_analysis(s_b, "B")
        df = compare_right_tail([r_b, r_a])   # pass in wrong order
        assert df.iloc[0]["model"] == "A"     # A should be first (κ=1.0)
        assert df.iloc[1]["model"] == "B"

    def test_consistency_index_values(self):
        s_a = make_store({"t": [1.0, 1.0]}, model="A")  # κ=1.0
        s_b = make_store({"t": [0.0, 1.0]}, model="B")  # κ=0.5
        r_a = right_tail_analysis(s_a, "A")
        r_b = right_tail_analysis(s_b, "B")
        df = compare_right_tail([r_a, r_b]).set_index("model")
        assert df.loc["A", "consistency_index"] == pytest.approx(1.0)
        assert df.loc["B", "consistency_index"] == pytest.approx(0.5)
