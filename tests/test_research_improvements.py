"""Tests for the research-driven additions and bug fixes.

Covers:
  - compare: adjust_pvalues, min_detectable_effect, required_n, score_length_bias,
    and the empty/single-input guards.
  - metrics: var alias, reliability_decay, variance_amplification_factor,
    grpo_advantages.
  - bootstrap: confidence_sequence (anytime-valid coverage).
  - right_tail: RightTailReport.consistency_index.
  - repeat: meta_distribution empty-input guard.
  - bayesian_optimization: ThompsonSamplingScheduler posterior correctness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from disteval import compare, metrics
from disteval.bootstrap import confidence_sequence
from disteval.repeat import meta_distribution


# ---------------------------------------------------------------------------
# compare.adjust_pvalues
# ---------------------------------------------------------------------------

class TestAdjustPvalues:
    def test_benjamini_hochberg_known_values(self):
        p = [0.01, 0.04, 0.03, 0.005]
        adj = compare.adjust_pvalues(p, method="bh")
        np.testing.assert_allclose(adj, [0.02, 0.04, 0.04, 0.02], rtol=1e-9)

    def test_holm_known_values(self):
        p = [0.01, 0.04, 0.03, 0.005]
        adj = compare.adjust_pvalues(p, method="holm")
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02], rtol=1e-9)

    def test_adjusted_at_least_raw_and_bounded(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 50)
        for method in ("bh", "holm"):
            adj = compare.adjust_pvalues(p, method=method)
            assert np.all(adj >= p - 1e-12)
            assert np.all(adj <= 1.0 + 1e-12)
            assert np.all(adj >= 0.0)

    def test_empty_input(self):
        assert compare.adjust_pvalues([]).size == 0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            compare.adjust_pvalues([0.1, 0.2], method="bonferroni-typo")


# ---------------------------------------------------------------------------
# compare.min_detectable_effect / required_n
# ---------------------------------------------------------------------------

class TestPowerAnalysis:
    def test_mde_positive_and_decreasing_in_n(self):
        small = compare.min_detectable_effect(5, 5)
        large = compare.min_detectable_effect(100, 100)
        assert small > large > 0

    def test_required_n_roundtrips_with_mde(self):
        n = 30
        d = compare.min_detectable_effect(n, n)
        assert compare.required_n(d) == pytest.approx(n, abs=1)

    def test_required_n_larger_for_smaller_effect(self):
        assert compare.required_n(0.2) > compare.required_n(0.8)

    def test_zero_effect_is_capped_large(self):
        assert compare.required_n(0.0) > 10**6

    def test_mde_invalid_n(self):
        assert np.isnan(compare.min_detectable_effect(0, 5))


# ---------------------------------------------------------------------------
# compare.score_length_bias
# ---------------------------------------------------------------------------

class TestScoreLengthBias:
    def test_strong_positive_correlation_flagged(self):
        lengths = np.arange(1, 31, dtype=float)
        scores = lengths / 30.0  # perfectly monotone with length
        res = compare.score_length_bias(scores, lengths)
        assert res["rho"] == pytest.approx(1.0)
        assert res["flagged"] is True

    def test_no_correlation_not_flagged(self):
        rng = np.random.default_rng(1)
        scores = rng.uniform(0, 1, 200)
        lengths = rng.uniform(1, 100, 200)
        res = compare.score_length_bias(scores, lengths)
        assert res["flagged"] is False

    def test_constant_length_safe(self):
        res = compare.score_length_bias([0.1, 0.5, 0.9], [10, 10, 10])
        assert res["flagged"] is False
        assert np.isnan(res["rho"])

    def test_too_few_points(self):
        res = compare.score_length_bias([0.1, 0.9], [1, 2])
        assert res["flagged"] is False
        assert res["n"] == 2


# ---------------------------------------------------------------------------
# compare guards (bug fixes)
# ---------------------------------------------------------------------------

class TestCompareGuards:
    def test_effect_size_single_element_is_zero_not_nan(self):
        # std(ddof=1) on a single element is nan (warns); the guard maps it to 0.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert compare.effect_size([1.0], [0.0]) == 0.0

    def test_prob_improvement_empty_is_nan(self):
        assert np.isnan(compare.prob_improvement([], [1.0, 2.0]))


# ---------------------------------------------------------------------------
# metrics additions
# ---------------------------------------------------------------------------

class TestMetricsVarAlias:
    def test_var_is_var_at(self):
        assert metrics.var is metrics.var_at
        x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        assert metrics.var(x, 0.1) == metrics.var_at(x, 0.1)


def _decay_df(per_task_success):
    """Build a (task, success) frame from {task: [bool, ...]}."""
    rows = []
    for task, succ in per_task_success.items():
        for i, s in enumerate(succ):
            rows.append({"task": task, "episode": i, "success": bool(s)})
    return pd.DataFrame(rows)


class TestReliabilityDecay:
    def test_decay_is_non_increasing_and_slope_non_positive(self):
        # One task, 4 trials, half succeed -> pass^k falls as k grows.
        df = _decay_df({"t1": [1, 1, 0, 0], "t2": [1, 1, 1, 0]})
        res = metrics.reliability_decay(df, ks=(1, 2, 4))
        vals = res["pass_hat_k"]
        assert vals == sorted(vals, reverse=True)  # non-increasing
        assert res["slope"] <= 0.0
        assert res["total_drop"] >= 0.0
        assert res["ks"] == [1, 2, 4]

    def test_perfect_agent_flat_curve(self):
        df = _decay_df({"t1": [1, 1, 1, 1]})
        res = metrics.reliability_decay(df, ks=(1, 2, 4))
        assert all(v == pytest.approx(1.0) for v in res["pass_hat_k"])
        assert res["slope"] == pytest.approx(0.0)


class TestVarianceAmplification:
    def test_long_horizon_more_variance(self):
        short = np.array([0.5, 0.5, 0.5, 0.6, 0.4])
        long = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
        assert metrics.variance_amplification_factor(short, long) > 1.0

    def test_zero_short_variance_is_nan(self):
        assert np.isnan(metrics.variance_amplification_factor([0.5, 0.5, 0.5], [0.1, 0.9, 0.5]))

    def test_too_few_points_is_nan(self):
        assert np.isnan(metrics.variance_amplification_factor([0.5], [0.1, 0.9]))


class TestGrpoAdvantages:
    def test_global_advantages_zero_mean(self):
        adv = metrics.grpo_advantages([0.0, 0.5, 1.0])
        assert adv.mean() == pytest.approx(0.0, abs=1e-9)
        assert adv[2] > 0 > adv[0]

    def test_grouped_normalization_independent(self):
        scores = np.array([0.0, 1.0, 10.0, 20.0])
        groups = np.array(["a", "a", "b", "b"])
        adv = metrics.grpo_advantages(scores, groups)
        # Within each group the higher score is positive, lower is negative.
        assert adv[0] < 0 < adv[1]
        assert adv[2] < 0 < adv[3]
        # Group b's large raw spread is normalized to the same scale as group a.
        assert adv[1] == pytest.approx(adv[3], rel=1e-6)

    def test_empty_input(self):
        assert metrics.grpo_advantages([]).size == 0


# ---------------------------------------------------------------------------
# bootstrap.confidence_sequence
# ---------------------------------------------------------------------------

class TestConfidenceSequence:
    def test_running_mean_matches_cumulative_mean(self):
        x = np.array([0.0, 1.0, 0.0, 1.0])
        res = confidence_sequence(x)
        np.testing.assert_allclose(res["running_mean"], [0.0, 0.5, 1 / 3, 0.5])
        assert res["point"] == pytest.approx(0.5)

    def test_width_shrinks_with_more_data(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(0, 1, 500)
        res = confidence_sequence(x)
        widths = np.array(res["running_hi"]) - np.array(res["running_lo"])
        assert widths[-1] < widths[10]

    def test_anytime_coverage(self):
        # Across many sequences, the true mean should lie inside the running
        # interval at EVERY t for >= (1 - alpha) of runs (conservative -> ~all).
        rng = np.random.default_rng(7)
        p = 0.3
        n_runs, n = 300, 80
        covered_everywhere = 0
        for _ in range(n_runs):
            x = (rng.uniform(0, 1, n) < p).astype(float)
            res = confidence_sequence(x, ci=0.9)
            lo = np.array(res["running_lo"])
            hi = np.array(res["running_hi"])
            if np.all((lo <= p) & (p <= hi)):
                covered_everywhere += 1
        assert covered_everywhere / n_runs >= 0.9

    def test_empty_input(self):
        res = confidence_sequence([])
        assert np.isnan(res["point"]) and res["n"] == 0

    def test_degenerate_range_raises(self):
        with pytest.raises(ValueError):
            confidence_sequence([0.5, 0.5], lo=1.0, hi=1.0)


# ---------------------------------------------------------------------------
# right_tail.RightTailReport.consistency_index
# ---------------------------------------------------------------------------

class TestConsistencyIndex:
    def _report(self, sum_q_bar, sum_q_star):
        from disteval.right_tail import RightTailReport
        return RightTailReport(
            model="m", n_tasks=1, n_episodes=1, profiles=[],
            n_solid=0, n_recoverable=1, n_stuck=0, total_gap=0.0,
            pct_recoverable=1.0, recoverable_score_left=0.0,
            sum_q_star=sum_q_star, sum_q_bar=sum_q_bar, priority_tasks=[],
        )

    def test_ratio(self):
        assert self._report(0.6, 0.8).consistency_index == pytest.approx(0.75)

    def test_zero_q_star_defaults_to_one(self):
        assert self._report(0.0, 0.0).consistency_index == 1.0


# ---------------------------------------------------------------------------
# repeat.meta_distribution empty guard
# ---------------------------------------------------------------------------

def test_meta_distribution_empty():
    res = meta_distribution([], lambda df: df["score"].mean())
    assert res["n_repeats"] == 0
    assert np.isnan(res["mean"])
    assert res["values"] == []


# ---------------------------------------------------------------------------
# report CLI accepts a generic .jsonl file (the "bring your own agent" path)
# ---------------------------------------------------------------------------

def test_report_main_accepts_generic_jsonl(tmp_path):
    import json

    import matplotlib
    matplotlib.use("Agg")
    from disteval import report

    recs = [
        {"run_id": "r", "model": "m", "task": f"t{i % 3}", "episode": i,
         "score": float(i % 2), "success": (i % 2) == 1}
        for i in range(12)
    ]
    jsonl = tmp_path / "runs.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in recs))
    out = tmp_path / "out"

    report.main([str(jsonl), "-o", str(out), "--agent", "test"])
    assert (out / "summary.json").exists()
