"""
tests/test_training_sim.py — Tests for disteval.training_sim

Covers (≥12 tests):
  1.  bootstrap_resample_within_tasks: shape preserved, within-task resampling
  2.  bootstrap_resample_within_tasks: task structure preserved (no cross-task contamination)
  3.  select_disteval_right_tail: returns only RECOVERABLE task trajectories
  4.  select_disteval_right_tail: respects k cap
  5.  select_mean_reward: returns top-K by score descending
  6.  select_mean_reward: returns all if k >= n
  7.  select_random: returns exactly k (or all if k >= n)
  8.  apply_training_effect: RECOVERABLE task improves correctly (formula check)
  9.  apply_training_effect: STUCK task improves by small amount
  10. apply_training_effect: SOLID task does NOT improve
  11. apply_training_effect: scores clipped to [0, 1]
  12. run_bootstrap_simulation: disteval_gains shape == (n_bootstrap,) per strategy
  13. run_bootstrap_simulation: reproducibility (same seed → same results)
  14. build_json_output: schema keys present for each agent
  15. build_json_output: CI ordering (ci_low <= mean_gain <= ci_high)
  16. pct_improvement / efficiency_pct_improvement: math correctness
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disteval.records import EpisodeRecord, RecordStore
from disteval.right_tail import right_tail_analysis
from disteval.training_sim import (
    ALPHA,
    DPO_BONUS,
    STUCK_FACTOR,
    N_BOOTSTRAP,
    apply_training_effect,
    bootstrap_resample_within_tasks,
    build_json_output,
    efficiency_pct_improvement,
    pct_improvement,
    run_bootstrap_simulation,
    select_disteval_right_tail,
    select_mean_reward,
    select_random,
    StrategyResult,
    AgentResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_episode(task: str, score: float, episode: int = 0, model: str = "test-model") -> EpisodeRecord:
    return EpisodeRecord(
        run_id="test-run",
        model=model,
        task=task,
        episode=episode,
        score=score,
        success=score >= 0.99,
    )


def _make_store_with_profile() -> RecordStore:
    """
    Store with:
      - task_A: RECOVERABLE [0.0, 1.0]   (q*=1, q_bar=0.5, gap=0.5)
      - task_B: SOLID       [1.0, 1.0]   (q*=1, q_bar=1.0, gap=0.0)
      - task_C: STUCK       [0.0, 0.0]   (q*=0)
    """
    records = [
        _make_episode("task_A", 0.0, episode=0),
        _make_episode("task_A", 1.0, episode=1),
        _make_episode("task_B", 1.0, episode=0),
        _make_episode("task_B", 1.0, episode=1),
        _make_episode("task_C", 0.0, episode=0),
        _make_episode("task_C", 0.0, episode=1),
    ]
    return RecordStore(records)


def _get_report(store: RecordStore):
    return right_tail_analysis(store)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: bootstrap_resample_within_tasks preserves total record count
# ─────────────────────────────────────────────────────────────────────────────
def test_bootstrap_resample_preserves_count():
    store = _make_store_with_profile()
    records = store._records
    rng = np.random.default_rng(0)
    resampled = bootstrap_resample_within_tasks(records, rng)
    assert len(resampled) == len(records), (
        f"Expected {len(records)} records after bootstrap, got {len(resampled)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: bootstrap_resample_within_tasks preserves task structure
# ─────────────────────────────────────────────────────────────────────────────
def test_bootstrap_resample_preserves_task_counts():
    store = _make_store_with_profile()
    records = store._records
    rng = np.random.default_rng(1)

    # Count per task before
    from collections import Counter
    before = Counter(r.task for r in records)

    resampled = bootstrap_resample_within_tasks(records, rng)
    after = Counter(r.task for r in resampled)

    assert dict(before) == dict(after), (
        f"Task counts changed: before={dict(before)}, after={dict(after)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: bootstrap resampling only resamples within tasks (no cross-task scores)
# ─────────────────────────────────────────────────────────────────────────────
def test_bootstrap_resample_no_cross_task_contamination():
    """task_B is all 1.0 and task_C is all 0.0 — resampled tasks must keep those scores."""
    store = _make_store_with_profile()
    records = store._records
    rng = np.random.default_rng(2)
    resampled = bootstrap_resample_within_tasks(records, rng)

    for rec in resampled:
        if rec.task == "task_B":
            assert rec.score == 1.0, "task_B should always be 1.0 (SOLID)"
        if rec.task == "task_C":
            assert rec.score == 0.0, "task_C should always be 0.0 (STUCK)"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: select_disteval_right_tail only picks from RECOVERABLE tasks
# ─────────────────────────────────────────────────────────────────────────────
def test_select_disteval_only_recoverable_tasks():
    store = _make_store_with_profile()
    records = store._records
    report = _get_report(store)
    selected = select_disteval_right_tail(records, report, k=10)
    selected_tasks = {r.task for r in selected}
    # task_B is SOLID, task_C is STUCK — neither should appear
    assert "task_B" not in selected_tasks, "SOLID task should not be selected"
    assert "task_C" not in selected_tasks, "STUCK task should not be selected"
    assert "task_A" in selected_tasks, "RECOVERABLE task_A should be selected"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: select_disteval_right_tail respects k cap
# ─────────────────────────────────────────────────────────────────────────────
def test_select_disteval_respects_k():
    store = _make_store_with_profile()
    records = store._records
    report = _get_report(store)
    selected = select_disteval_right_tail(records, report, k=1)
    assert len(selected) <= 1, f"Expected at most 1 selected, got {len(selected)}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: select_mean_reward returns top-K by score descending
# ─────────────────────────────────────────────────────────────────────────────
def test_select_mean_reward_top_k():
    records = [
        _make_episode("task_A", 0.2, episode=0),
        _make_episode("task_A", 0.9, episode=1),
        _make_episode("task_B", 0.5, episode=0),
        _make_episode("task_B", 0.1, episode=1),
    ]
    selected = select_mean_reward(records, k=2)
    assert len(selected) == 2
    scores = [r.score for r in selected]
    assert scores[0] >= scores[1], "Should be sorted descending"
    assert max(scores) == 0.9, "Top score should be 0.9"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: select_mean_reward returns all if k >= n
# ─────────────────────────────────────────────────────────────────────────────
def test_select_mean_reward_k_ge_n():
    records = [_make_episode("task_A", float(i) * 0.1, episode=i) for i in range(3)]
    selected = select_mean_reward(records, k=10)
    assert len(selected) == len(records), "Should return all when k >= n"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: select_random returns exactly k (without replacement when k < n)
# ─────────────────────────────────────────────────────────────────────────────
def test_select_random_count():
    records = [_make_episode("task_A", float(i) * 0.1, episode=i) for i in range(10)]
    rng = np.random.default_rng(42)
    selected = select_random(records, k=3, rng=rng)
    assert len(selected) == 3
    # All selected must be from original records
    for rec in selected:
        assert rec in records


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: apply_training_effect — RECOVERABLE task improvement formula
# ─────────────────────────────────────────────────────────────────────────────
def test_apply_training_effect_recoverable():
    """
    task_A is RECOVERABLE with scores [0.0, 1.0].
    current_mean = 0.5. selected all with mean=0.5.
    expected improvement = 0.4 * 0.5 * (1 - 0.5) = 0.1
    new scores for task_A = [0.0+0.1, 1.0+0.1] = [0.1, 1.0] (1.0 clipped)
    """
    store = _make_store_with_profile()
    records = store._records
    report = _get_report(store)

    # Select all task_A records as "selected"
    task_a_recs = [r for r in records if r.task == "task_A"]
    new_scores = apply_training_effect(records, task_a_recs, report, alpha=ALPHA)

    # task_A records: indices 0 (score 0.0) and 1 (score 1.0)
    task_a_new = [s for r, s in zip(records, new_scores) if r.task == "task_A"]
    current_mean = 0.5
    sel_mean = 0.5
    expected_improvement = ALPHA * sel_mean * (1.0 - current_mean)  # 0.4*0.5*0.5 = 0.1
    for orig_r, new_s in zip(task_a_recs, task_a_new):
        expected = min(1.0, orig_r.score + expected_improvement)
        assert abs(new_s - expected) < 1e-9, (
            f"RECOVERABLE improvement mismatch: got {new_s:.6f}, expected {expected:.6f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: apply_training_effect — STUCK task improves by small amount
# ─────────────────────────────────────────────────────────────────────────────
def test_apply_training_effect_stuck():
    """
    task_C is STUCK with scores [0.0, 0.0].
    Select task_A records (score=[0.0, 1.0]), sel_mean=0.5 for task_A.
    STUCK improvement = alpha * STUCK_FACTOR * sel_mean_FOR_THIS_TASK
    Since task_C has no selected records, improvement = 0.
    But if we select stuck task records themselves ... they're 0.0 so improvement=0.
    """
    store = _make_store_with_profile()
    records = store._records
    report = _get_report(store)

    # Select task_A records only — task_C gets no selected records → improvement=0
    selected = [r for r in records if r.task == "task_A"]
    new_scores = apply_training_effect(records, selected, report, alpha=ALPHA)

    task_c_new = [s for r, s in zip(records, new_scores) if r.task == "task_C"]
    # STUCK with no selected trajectories = 0 improvement
    for s in task_c_new:
        assert s == 0.0, f"STUCK task with no selected should stay 0.0, got {s}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: apply_training_effect — SOLID task does NOT improve
# ─────────────────────────────────────────────────────────────────────────────
def test_apply_training_effect_solid_no_change():
    """task_B is SOLID with scores [1.0, 1.0]. No change expected."""
    store = _make_store_with_profile()
    records = store._records
    report = _get_report(store)

    # Select all records
    new_scores = apply_training_effect(records, records, report, alpha=ALPHA)

    task_b_new = [s for r, s in zip(records, new_scores) if r.task == "task_B"]
    for s in task_b_new:
        assert s == 1.0, f"SOLID task should stay at 1.0, got {s}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: apply_training_effect — scores clipped to [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def test_apply_training_effect_score_clipping():
    """With high improvement, scores should be clipped to 1.0 (not exceed it)."""
    records = [
        _make_episode("task_A", 0.95, episode=0),
        _make_episode("task_A", 1.0,  episode=1),
    ]
    store = RecordStore(records)
    report = right_tail_analysis(store)

    # Select with mean=0.975, current_mean=0.975
    # improvement = 0.4 * 0.975 * (1 - 0.975) = very small
    # Use large alpha to force over 1.0
    new_scores = apply_training_effect(records, records, report, alpha=10.0)
    for s in new_scores:
        assert s <= 1.0, f"Score should be clipped to 1.0, got {s}"
        assert s >= 0.0, f"Score should be >= 0.0, got {s}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: run_bootstrap_simulation — output shapes and types
# ─────────────────────────────────────────────────────────────────────────────
def test_run_bootstrap_simulation_shapes():
    store = _make_store_with_profile()
    records = store._records
    report = _get_report(store)
    n_bs = 50  # small for test speed

    de_gains, mr_gains, rand_gains = run_bootstrap_simulation(
        records, report, n_bootstrap=n_bs, seed=42
    )
    assert de_gains.shape == (n_bs,), f"Expected ({n_bs},), got {de_gains.shape}"
    assert mr_gains.shape == (n_bs,), f"Expected ({n_bs},), got {mr_gains.shape}"
    assert rand_gains.shape == (n_bs,), f"Expected ({n_bs},), got {rand_gains.shape}"
    assert de_gains.dtype in (np.float64, np.float32), "Gains should be floats"


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: run_bootstrap_simulation — reproducibility with same seed
# ─────────────────────────────────────────────────────────────────────────────
def test_run_bootstrap_simulation_reproducible():
    store = _make_store_with_profile()
    records = store._records
    report = _get_report(store)
    n_bs = 30

    de1, mr1, rnd1 = run_bootstrap_simulation(records, report, n_bootstrap=n_bs, seed=99)
    de2, mr2, rnd2 = run_bootstrap_simulation(records, report, n_bootstrap=n_bs, seed=99)

    np.testing.assert_array_equal(de1, de2,  err_msg="disteval gains not reproducible")
    np.testing.assert_array_equal(mr1, mr2,  err_msg="mean_reward gains not reproducible")
    np.testing.assert_array_equal(rnd1, rnd2, err_msg="random gains not reproducible")


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: StrategyResult.from_gains — CI ordering
# ─────────────────────────────────────────────────────────────────────────────
def test_strategy_result_ci_ordering():
    gains = np.random.default_rng(0).normal(0.1, 0.02, 1000)
    sr = StrategyResult.from_gains("disteval_right_tail", gains)
    assert sr.ci_low <= sr.mean_gain, f"ci_low ({sr.ci_low}) > mean_gain ({sr.mean_gain})"
    assert sr.mean_gain <= sr.ci_high, f"mean_gain ({sr.mean_gain}) > ci_high ({sr.ci_high})"
    assert abs(sr.mean_gain - float(np.mean(gains))) < 1e-9, "mean_gain mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: build_json_output — required keys present
# ─────────────────────────────────────────────────────────────────────────────
def test_build_json_output_schema():
    # Build a minimal AgentResult
    n = 50
    rng = np.random.default_rng(42)
    gains = rng.normal(0.1, 0.02, n)
    de_sr   = StrategyResult.from_gains("disteval_right_tail", gains + 0.02)
    mr_sr   = StrategyResult.from_gains("mean_reward",         gains)
    rnd_sr  = StrategyResult.from_gains("random_sampling",     gains - 0.03)

    res = AgentResult(
        agent_name="TestAgent",
        baseline=0.5,
        disteval=de_sr,
        mean_reward=mr_sr,
        random=rnd_sr,
        disteval_vs_mean_reward_pct=20.0,
        disteval_vs_random_pct=50.0,
        p_value_vs_mean_reward=0.01,
        p_value_vs_random=0.001,
        efficiency_disteval=3.0,
        efficiency_mean_reward=5.0,
        efficiency_random=8.0,
        efficiency_disteval_ci=(2.0, 4.0),
        efficiency_mean_reward_ci=(4.0, 6.0),
        efficiency_random_ci=(6.0, 10.0),
        efficiency_gain_vs_mean_reward_pct=40.0,
        efficiency_gain_vs_random_pct=62.5,
    )

    output = build_json_output([res], n_bootstrap=n)

    # Top-level keys
    assert "n_bootstrap" in output
    assert "agents" in output
    assert "summary" in output
    assert output["n_bootstrap"] == n

    # Agent-level keys
    agent_data = output["agents"]["TestAgent"]
    required_agent_keys = [
        "baseline", "disteval", "mean_reward", "random",
        "disteval_vs_mean_reward_pct", "disteval_vs_random_pct",
        "p_value_vs_mean_reward", "p_value_vs_random",
        "data_efficiency_disteval", "data_efficiency_mean_reward", "data_efficiency_random",
        "efficiency_gain_vs_mean_reward_pct", "efficiency_gain_vs_random_pct",
    ]
    for key in required_agent_keys:
        assert key in agent_data, f"Missing agent key: {key}"

    # Strategy sub-keys
    for strat in ["disteval", "mean_reward", "random"]:
        strat_data = agent_data[strat]
        for k in ["mean_gain", "ci_low", "ci_high"]:
            assert k in strat_data, f"Missing {strat}.{k}"
        assert strat_data["ci_low"] <= strat_data["mean_gain"], \
            f"{strat}: ci_low > mean_gain"
        assert strat_data["mean_gain"] <= strat_data["ci_high"], \
            f"{strat}: mean_gain > ci_high"

    # Summary keys
    summary_keys = [
        "mean_gain_disteval_vs_mean_reward_pct",
        "mean_gain_disteval_vs_random_pct",
        "p_value_vs_mean_reward",
        "p_value_vs_random",
    ]
    for key in summary_keys:
        assert key in output["summary"], f"Missing summary key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: pct_improvement — math correctness
# ─────────────────────────────────────────────────────────────────────────────
def test_pct_improvement_math():
    # (0.1 - 0.05) / 0.05 * 100 = 100%
    assert abs(pct_improvement(0.1, 0.05) - 100.0) < 1e-9
    # same values → 0%
    assert pct_improvement(0.07, 0.07) == 0.0
    # zero denominator → inf (a > b)
    result = pct_improvement(0.1, 0.0)
    assert result == float("inf")
    # zero denominator, a=b=0 → 0
    assert pct_improvement(0.0, 0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 18: efficiency_pct_improvement — math correctness
# ─────────────────────────────────────────────────────────────────────────────
def test_efficiency_pct_improvement_math():
    # disteval needs 4 rounds, baseline needs 8 → 50% fewer
    assert abs(efficiency_pct_improvement(4.0, 8.0) - 50.0) < 1e-9
    # equal → 0%
    assert efficiency_pct_improvement(5.0, 5.0) == 0.0
    # zero denominator → 0
    assert efficiency_pct_improvement(5.0, 0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 19: build_json_output — JSON serializable (no numpy types)
# ─────────────────────────────────────────────────────────────────────────────
def test_build_json_output_serializable():
    n = 20
    rng = np.random.default_rng(7)
    gains = rng.normal(0.1, 0.02, n)
    de_sr  = StrategyResult.from_gains("disteval_right_tail", gains + 0.01)
    mr_sr  = StrategyResult.from_gains("mean_reward",         gains)
    rnd_sr = StrategyResult.from_gains("random_sampling",     gains - 0.01)

    res = AgentResult(
        agent_name="SerializationTest",
        baseline=0.5,
        disteval=de_sr,
        mean_reward=mr_sr,
        random=rnd_sr,
        disteval_vs_mean_reward_pct=10.0,
        disteval_vs_random_pct=20.0,
        p_value_vs_mean_reward=0.05,
        p_value_vs_random=0.02,
        efficiency_disteval=3.0,
        efficiency_mean_reward=4.0,
        efficiency_random=7.0,
        efficiency_disteval_ci=(2.0, 4.0),
        efficiency_mean_reward_ci=(3.0, 5.0),
        efficiency_random_ci=(5.0, 9.0),
        efficiency_gain_vs_mean_reward_pct=25.0,
        efficiency_gain_vs_random_pct=57.1,
    )

    output = build_json_output([res], n_bootstrap=n)
    # Should not raise
    serialized = json.dumps(output)
    reloaded = json.loads(serialized)
    assert reloaded["agents"]["SerializationTest"]["baseline"] == round(0.5, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Test 20: select_disteval_right_tail — with multiple RECOVERABLE tasks, sorted by gap
# ─────────────────────────────────────────────────────────────────────────────
def test_select_disteval_multiple_recoverable_tasks():
    """
    With two RECOVERABLE tasks, higher-gap task should be prioritized first.
    task_high_gap: [0.0, 1.0] → gap=0.5
    task_low_gap:  [0.5, 1.0] → gap=0.25
    k=1 → should select from task_high_gap
    """
    records = [
        _make_episode("task_high_gap", 0.0, episode=0),
        _make_episode("task_high_gap", 1.0, episode=1),
        _make_episode("task_low_gap",  0.5, episode=0),
        _make_episode("task_low_gap",  1.0, episode=1),
    ]
    store = RecordStore(records)
    report = right_tail_analysis(store)
    selected = select_disteval_right_tail(records, report, k=1)
    # The highest-gap task's high-outcome trajectory should be selected first
    assert len(selected) == 1
    assert selected[0].task == "task_high_gap", (
        f"Expected task_high_gap to be selected first, got {selected[0].task}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 21: apply_training_effect — STUCK task with selected trajectories
# ─────────────────────────────────────────────────────────────────────────────
def test_apply_training_effect_stuck_with_selected():
    """
    STUCK task with selected trajectories should improve by small amount.
    task_stuck: [0.0] → STUCK (q*=0)
    BUT we can manually test the stuck branch by crafting a scenario:
    task with q*=0 has all zeros. If something was selected from that task,
    improvement = alpha * STUCK_FACTOR * sel_mean.
    """
    # Create a store where task_stuck is STUCK but we assign it sel trajectories with score 0.5
    # We have to use a separate design: task_recoverable helps drive stuck task
    # Actually test: for STUCK task, if we select task_stuck trajectories (all 0.0),
    # improvement = alpha * STUCK_FACTOR * 0.0 = 0.0
    records = [
        _make_episode("task_stuck", 0.0, episode=0),
    ]
    store = RecordStore(records)
    report = right_tail_analysis(store)
    # Select the stuck record itself (score=0.0)
    new_scores = apply_training_effect(records, records, report, alpha=ALPHA)
    assert new_scores[0] == 0.0, "STUCK with 0.0 selected scores stays 0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Test 22: run_bootstrap_simulation with different seeds gives different results
# ─────────────────────────────────────────────────────────────────────────────
def test_run_bootstrap_simulation_different_seeds():
    store = _make_store_with_profile()
    records = store._records
    report = _get_report(store)
    n_bs = 30

    de1, _, _ = run_bootstrap_simulation(records, report, n_bootstrap=n_bs, seed=1)
    de2, _, _ = run_bootstrap_simulation(records, report, n_bootstrap=n_bs, seed=999)

    # With different seeds, results should NOT be identical (with overwhelming probability)
    assert not np.array_equal(de1, de2), "Different seeds produced identical results"


# ─────────────────────────────────────────────────────────────────────────────
# Test 23: apply_training_effect — disteval DPO bonus for paired examples
# ─────────────────────────────────────────────────────────────────────────────
def test_apply_training_effect_dpo_bonus():
    """
    When disteval_right_tail strategy is used AND both reinforce (high) AND
    contrast (low) trajectories are selected for a RECOVERABLE task, the
    improvement should use DPO_BONUS * q_star (not just mean(selected)).
    """
    records = [
        _make_episode("task_A", 0.0, episode=0),
        _make_episode("task_A", 1.0, episode=1),
    ]
    store = RecordStore(records)
    report = right_tail_analysis(store)

    # For disteval: both 0.0 (contrast) and 1.0 (reinforce) selected
    # current_mean = 0.5, q_star = 1.0
    # expected = ALPHA * DPO_BONUS * q_star * (1 - current_mean)
    #          = 0.4 * 1.5 * 1.0 * 0.5 = 0.3
    new_scores_dpo = apply_training_effect(records, records, report, alpha=ALPHA, strategy="disteval_right_tail")
    # expected improvement = 0.3
    expected_imp = ALPHA * DPO_BONUS * 1.0 * 0.5
    for orig_r, new_s in zip(records, new_scores_dpo):
        expected = min(1.0, orig_r.score + expected_imp)
        assert abs(new_s - expected) < 1e-9, (
            f"DPO bonus mismatch: orig={orig_r.score}, got {new_s:.6f}, expected {expected:.6f}"
        )

    # Generic strategy (no DPO bonus): uses mean(selected)=0.5
    # expected = 0.4 * 0.5 * 0.5 = 0.1
    new_scores_bc = apply_training_effect(records, records, report, alpha=ALPHA, strategy="generic")
    expected_bc = ALPHA * 0.5 * 0.5
    for orig_r, new_s in zip(records, new_scores_bc):
        expected = min(1.0, orig_r.score + expected_bc)
        assert abs(new_s - expected) < 1e-9, (
            f"Generic BC mismatch: orig={orig_r.score}, got {new_s:.6f}, expected {expected:.6f}"
        )

    # DPO bonus should be larger than BC
    dpo_gain = float(np.mean(new_scores_dpo))
    bc_gain  = float(np.mean(new_scores_bc))
    assert dpo_gain > bc_gain, (
        f"DPO strategy should produce larger gain than BC: DPO={dpo_gain:.4f}, BC={bc_gain:.4f}"
    )
