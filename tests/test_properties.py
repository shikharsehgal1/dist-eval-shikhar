"""Property-based tests, using Hypothesis where a generator adds real coverage.

These target invariants that must hold for *any* input, not just the hand-picked
cases in the unit tests -- the class of bug that survives example-based testing.
"""
import numpy as np
import pandas as pd
import pytest

from disteval import metrics as M
from disteval.active.allocation import TaskState, make_policy
from disteval.active.value import flip_probability
from disteval.reliability.classify import (
    CATEGORIES,
    ReliabilityThresholds,
    diagnose,
    expected_headroom,
)
from disteval.reliability.posterior import binary_posterior, posterior_from_scores

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

counts = st.integers(min_value=0, max_value=60)
scores = st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False,
                            allow_infinity=False), min_size=1, max_size=40)


@st.composite
def success_counts(draw):
    n = draw(st.integers(min_value=1, max_value=60))
    s = draw(st.integers(min_value=0, max_value=n))
    return s, n


@st.composite
def task_table(draw):
    n_tasks = draw(st.integers(min_value=1, max_value=8))
    n_runs = draw(st.integers(min_value=1, max_value=12))
    rows = []
    for t in range(n_tasks):
        for _ in range(n_runs):
            ok = draw(st.booleans())
            rows.append({"task": f"t{t}", "score": float(ok), "success": ok})
    return pd.DataFrame(rows), n_runs


class TestPosteriorProperties:
    @SETTINGS
    @given(success_counts())
    def test_posterior_mean_is_strictly_inside_the_unit_interval(self, sn):
        s, n = sn
        p = binary_posterior(s, n)
        assert 0.0 < p.mean < 1.0

    @SETTINGS
    @given(success_counts())
    def test_credible_interval_brackets_the_median(self, sn):
        s, n = sn
        p = binary_posterior(s, n)
        lo, hi = p.credible_interval()
        assert lo <= p.median <= hi
        assert 0.0 <= lo <= hi <= 1.0

    @SETTINGS
    @given(success_counts())
    def test_prob_above_is_monotone_non_increasing_in_the_threshold(self, sn):
        s, n = sn
        p = binary_posterior(s, n)
        vals = [p.prob_above(x) for x in np.linspace(0.01, 0.99, 25)]
        assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))

    @SETTINGS
    @given(success_counts())
    def test_posterior_predictive_is_normalised(self, sn):
        s, n = sn
        assert binary_posterior(s, n).posterior_predictive(6).sum() == pytest.approx(1.0)

    @SETTINGS
    @given(st.integers(min_value=1, max_value=40))
    def test_more_successes_never_lower_the_posterior_mean(self, n):
        means = [binary_posterior(s, n).mean for s in range(n + 1)]
        assert all(b >= a - 1e-12 for a, b in zip(means, means[1:]))

    @SETTINGS
    @given(st.integers(min_value=2, max_value=40))
    def test_uncertainty_shrinks_with_more_data_at_a_fixed_rate(self, n):
        a = binary_posterior(n, 2 * n)
        b = binary_posterior(2 * n, 4 * n)
        assert b.sd < a.sd

    @SETTINGS
    @given(scores)
    def test_any_bounded_score_vector_yields_a_valid_posterior(self, xs):
        p = posterior_from_scores(xs)
        assert 0.0 < p.mean < 1.0
        assert np.isfinite(p.sd) and p.sd >= 0
        lo, hi = p.credible_interval()
        assert 0.0 <= lo <= hi <= 1.0


class TestMetricProperties:
    @SETTINGS
    @given(task_table())
    def test_pass_at_k_never_decreases_in_k(self, tn):
        df, n_runs = tn
        vals = [M.pass_at_k(df, k) for k in range(1, n_runs + 1)]
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))

    @SETTINGS
    @given(task_table())
    def test_pass_hat_k_never_increases_in_k(self, tn):
        df, n_runs = tn
        vals = [M.pass_hat_k(df, k) for k in range(1, n_runs + 1)]
        assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))

    @SETTINGS
    @given(task_table())
    def test_pass_hat_k_never_exceeds_pass_at_k(self, tn):
        df, n_runs = tn
        for k in range(1, n_runs + 1):
            assert M.pass_hat_k(df, k) <= M.pass_at_k(df, k) + 1e-9

    @SETTINGS
    @given(scores)
    def test_cvar_ordering_holds_for_any_sample(self, xs):
        x = np.asarray(xs, dtype=float)
        assert M.lower_cvar(x, 0.25) <= x.mean() + 1e-9
        assert M.upper_cvar(x, 0.25) >= x.mean() - 1e-9
        assert M.worst_case(x) <= M.lower_cvar(x, 0.25) + 1e-9

    @SETTINGS
    @given(scores)
    def test_degrading_the_worst_run_cannot_raise_lower_cvar(self, xs):
        x = np.asarray(xs, dtype=float)
        assume(x.size >= 4)
        worse = x.copy()
        worse[int(np.argmin(worse))] = 0.0
        assert M.lower_cvar(worse, 0.25) <= M.lower_cvar(x, 0.25) + 1e-9

    @SETTINGS
    @given(scores)
    def test_iqm_lies_within_the_sample_range(self, xs):
        x = np.asarray(xs, dtype=float)
        assert x.min() - 1e-9 <= M.iqm(x) <= x.max() + 1e-9

    @SETTINGS
    @given(scores, st.floats(min_value=0.0, max_value=0.5))
    def test_adding_a_constant_shifts_the_mean_and_cvar_equally(self, xs, c):
        x = np.asarray(xs, dtype=float)
        assume(x.size >= 3)
        shifted = x + c
        assert M.lower_cvar(shifted, 0.25) == pytest.approx(M.lower_cvar(x, 0.25) + c)


class TestClassificationProperties:
    @SETTINGS
    @given(success_counts())
    def test_label_is_always_one_of_the_declared_categories(self, sn):
        s, n = sn
        assert diagnose(binary_posterior(s, n)).label in CATEGORIES

    @SETTINGS
    @given(success_counts())
    def test_all_reported_probabilities_are_probabilities(self, sn):
        s, n = sn
        d = diagnose(binary_posterior(s, n))
        for f in ("capability", "reliability", "stuck_evidence",
                  "recoverability_gap", "recoverability_headroom",
                  "recoverability_evidence"):
            assert 0.0 <= getattr(d, f) <= 1.0

    @SETTINGS
    @given(success_counts())
    def test_capability_is_at_least_reliability(self, sn):
        """P(p > tau_cap) >= P(p > tau_rel) since tau_cap < tau_rel."""
        s, n = sn
        d = diagnose(binary_posterior(s, n))
        assert d.capability >= d.reliability - 1e-12

    @SETTINGS
    @given(success_counts())
    def test_stuck_and_capability_cannot_both_be_confident(self, sn):
        s, n = sn
        d = diagnose(binary_posterior(s, n))
        assert not (d.capability > 0.9 and d.stuck_evidence > 0.9)

    @SETTINGS
    @given(st.integers(min_value=1, max_value=40))
    def test_all_success_is_never_stuck_and_all_failure_is_never_solid(self, n):
        assert diagnose(binary_posterior(n, n)).label != "STUCK"
        assert diagnose(binary_posterior(0, n)).label != "SOLID"

    @SETTINGS
    @given(st.integers(min_value=1, max_value=30))
    def test_below_min_runs_is_always_uncertain(self, n):
        th = ReliabilityThresholds(min_runs=n + 1)
        for s in range(n + 1):
            assert diagnose(binary_posterior(s, n), thresholds=th).label == "UNCERTAIN"

    @SETTINGS
    @given(success_counts())
    def test_headroom_is_bounded(self, sn):
        s, n = sn
        assert 0.0 <= expected_headroom(binary_posterior(s, n),
                                        ReliabilityThresholds()) <= 1.0


class TestActiveProperties:
    @SETTINGS
    @given(success_counts(), st.integers(min_value=1, max_value=6))
    def test_flip_probability_is_a_probability_for_any_state(self, sn, horizon):
        s, n = sn
        v = flip_probability(binary_posterior(s, n), ReliabilityThresholds(),
                             horizon=horizon)
        assert 0.0 <= v <= 1.0

    @SETTINGS
    @given(
        st.lists(success_counts(), min_size=1, max_size=8),
        st.integers(min_value=0, max_value=40),
        st.sampled_from(["uniform", "greedy_voi", "thompson", "stratified_uniform"]),
    )
    def test_allocation_never_exceeds_the_budget(self, states, budget, policy):
        ts = [TaskState(f"t{i}", n, float(s)) for i, (s, n) in enumerate(states)]
        r = make_policy(policy).allocate(ts, budget, seed=0)
        assert r.spent <= budget
        assert all(v >= 0 for v in r.allocation.values())
        assert set(r.allocation) <= {t.task for t in ts}
