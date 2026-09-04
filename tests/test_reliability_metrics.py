"""Property and edge-case tests for the distributional metrics.

Includes the properties the brief asks for explicitly: pass@k cannot decrease in
k, pass^k cannot increase in k, and lower-tail CVaR must respond to degradation
in the worst tail while upper-tail CVaR must not.
"""
import numpy as np
import pandas as pd
import pytest

from disteval import metrics as M
from disteval import metrics_spec


def _df(per_task):
    rows = []
    for task, scores in per_task.items():
        for s in scores:
            rows.append({"task": task, "score": float(s), "success": float(s) >= 1.0})
    return pd.DataFrame(rows)


class TestPassAtKProperties:
    def test_pass_at_k_is_non_decreasing_in_k(self):
        rng = np.random.default_rng(0)
        df = _df({f"t{i}": rng.integers(0, 2, 12).tolist() for i in range(15)})
        vals = [M.pass_at_k(df, k) for k in range(1, 9)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))

    def test_pass_hat_k_is_non_increasing_in_k(self):
        rng = np.random.default_rng(1)
        df = _df({f"t{i}": rng.integers(0, 2, 12).tolist() for i in range(15)})
        vals = [M.pass_hat_k(df, k) for k in range(1, 9)]
        assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))

    def test_pass_at_1_equals_pass_hat_1_equals_success_rate(self):
        df = _df({"a": [1, 0, 1, 1], "b": [0, 0, 1, 0]})
        assert M.pass_at_k(df, 1) == pytest.approx(M.pass_hat_k(df, 1))
        assert M.pass_at_k(df, 1) == pytest.approx(df.groupby("task")["success"].mean().mean())

    def test_all_success_gives_one_for_both(self):
        df = _df({"a": [1] * 8, "b": [1] * 8})
        assert M.pass_at_k(df, 4) == pytest.approx(1.0)
        assert M.pass_hat_k(df, 4) == pytest.approx(1.0)

    def test_all_failure_gives_zero_for_both(self):
        df = _df({"a": [0] * 8, "b": [0] * 8})
        assert M.pass_at_k(df, 4) == pytest.approx(0.0)
        assert M.pass_hat_k(df, 4) == pytest.approx(0.0)

    def test_unbiased_estimator_matches_the_closed_form(self):
        # 8 runs, 3 successes: pass@2 = 1 - C(5,2)/C(8,2) = 1 - 10/28
        df = _df({"a": [1, 1, 1, 0, 0, 0, 0, 0]})
        assert M.pass_at_k(df, 2) == pytest.approx(1 - 10 / 28)
        assert M.pass_hat_k(df, 2) == pytest.approx(3 / 28)

    def test_falls_back_when_fewer_runs_than_k(self):
        df = _df({"a": [1, 0]})
        assert M.pass_at_k(df, 8) == pytest.approx(1.0)
        assert M.pass_hat_k(df, 8) == pytest.approx(0.0)

    def test_unequal_run_counts_are_averaged_per_task(self):
        df = _df({"a": [1] * 2, "b": [0] * 10})
        assert M.pass_at_k(df, 2) == pytest.approx(0.5)

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError):
            M.pass_at_k(pd.DataFrame({"task": ["a"]}), 1)

    def test_reliability_gap_is_non_negative(self):
        rng = np.random.default_rng(2)
        df = _df({f"t{i}": rng.integers(0, 2, 8).tolist() for i in range(20)})
        assert M.reliability_gap(df, 4) >= -1e-12


class TestTailRisk:
    def test_lower_cvar_responds_to_worst_tail_degradation(self):
        base = np.array([0.2, 0.5, 0.6, 0.9, 1.0])
        worse = np.array([0.0, 0.5, 0.6, 0.9, 1.0])
        assert M.lower_cvar(worse, 0.2) < M.lower_cvar(base, 0.2)

    def test_upper_cvar_does_not_respond_to_the_lower_tail(self):
        """The counterexample from THEORY.md, as an executable test."""
        a = np.array([0.0, 0.0, 1.0])
        b = np.array([1.0, 1.0, 1.0])
        assert M.upper_cvar(a, 1 / 3) == pytest.approx(M.upper_cvar(b, 1 / 3))
        assert M.lower_cvar(a, 1 / 3) < M.lower_cvar(b, 1 / 3)

    def test_lower_cvar_never_exceeds_the_mean(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            x = rng.random(30)
            assert M.lower_cvar(x, 0.25) <= x.mean() + 1e-12

    def test_upper_cvar_is_at_least_the_mean(self):
        rng = np.random.default_rng(4)
        for _ in range(20):
            x = rng.random(30)
            assert M.upper_cvar(x, 0.25) >= x.mean() - 1e-12

    def test_constant_sample_collapses_to_the_constant(self):
        x = np.full(10, 0.42)
        assert M.lower_cvar(x, 0.2) == pytest.approx(0.42)
        assert M.upper_cvar(x, 0.2) == pytest.approx(0.42)
        assert M.worst_case(x) == pytest.approx(0.42)

    def test_empty_returns_nan_not_zero(self):
        assert np.isnan(M.cvar(np.array([]), 0.2))
        assert np.isnan(M.worst_case(np.array([])))
        assert np.isnan(M.iqm(np.array([])))

    def test_bootstrap_ci_brackets_the_point_estimate(self):
        rng = np.random.default_rng(5)
        x = rng.random(60)
        lo, hi = M.bootstrap_ci(x, n_boot=800, seed=0)
        assert lo <= M.lower_cvar(x, 0.2) <= hi

    def test_bootstrap_ci_is_reproducible(self):
        x = np.random.default_rng(6).random(40)
        assert M.bootstrap_ci(x, n_boot=400, seed=7) == M.bootstrap_ci(x, n_boot=400, seed=7)

    def test_bootstrap_ci_on_empty(self):
        lo, hi = M.bootstrap_ci(np.array([]))
        assert np.isnan(lo) and np.isnan(hi)


class TestIQM:
    def test_ignores_extreme_tails(self):
        clean = np.array([0.4, 0.5, 0.5, 0.5, 0.6])
        dirty = np.array([0.0, 0.5, 0.5, 0.5, 1.0])
        assert M.iqm(clean) == pytest.approx(M.iqm(dirty), abs=0.05)
        assert abs(clean.mean() - dirty.mean()) < 0.05  # means agree here too

    def test_single_value(self):
        assert M.iqm(np.array([0.7])) == pytest.approx(0.7)


class TestRegistry:
    def test_every_registered_metric_is_complete(self):
        for name, spec in metrics_spec.REGISTRY.items():
            assert spec.summary and spec.definition and spec.domain
            assert spec.assumptions, f"{name} declares no assumptions"
            assert spec.edge_cases, f"{name} declares no edge cases"
            assert spec.estimator and spec.uncertainty

    def test_prior_work_is_attributed_where_it_exists(self):
        for name in ("pass@k", "pass^k", "lower_cvar", "iqm"):
            assert metrics_spec.REGISTRY[name].prior_work, f"{name} lacks attribution"

    def test_recoverability_is_labelled_as_a_hypothesis(self):
        text = " ".join(metrics_spec.REGISTRY["recoverability_headroom"].assumptions)
        assert "HYPOTHESIS" in text.upper()

    def test_duplicate_registration_raises(self):
        spec = metrics_spec.REGISTRY["mean"]
        with pytest.raises(ValueError):
            metrics_spec.register(spec)

    def test_markdown_renders_every_metric(self):
        md = metrics_spec.to_markdown()
        for name in metrics_spec.REGISTRY:
            assert f"`{name}`" in md

    def test_frame_shape(self):
        assert len(metrics_spec.as_frame()) == len(metrics_spec.REGISTRY)
