"""Tests for disteval.bootstrap – performance_profile and stratified_bootstrap_ci."""
import numpy as np
import pandas as pd
import pytest

from disteval.bootstrap import (
    analytical_ci,
    binomial_ci,
    performance_profile,
    stratified_bootstrap_ci,
)
from disteval.metrics import iqm


# ---------------------------------------------------------------------------
# performance_profile
# ---------------------------------------------------------------------------

class TestPerformanceProfile:
    def test_default_taus_length(self):
        """Default linspace should have 50 points."""
        scores = np.array([1.0, 2, 3, 4, 5])
        taus, fracs = performance_profile(scores)
        assert taus.shape == (50,)
        assert fracs.shape == (50,)

    def test_shapes_match(self):
        scores = np.linspace(0, 1, 20)
        taus, fracs = performance_profile(scores)
        assert taus.shape == fracs.shape

    def test_fraction_at_min_tau_is_one(self):
        """At the minimum score every episode scores >= tau, so fraction = 1.0."""
        scores = np.array([1.0, 2, 3, 4, 5])
        taus, fracs = performance_profile(scores)
        assert fracs[0] == pytest.approx(1.0)

    def test_fraction_at_max_tau(self):
        """At the maximum score, only episodes equal to the max score qualify.
        For [1,2,3,4,5] that is 1/5 = 0.2."""
        scores = np.array([1.0, 2, 3, 4, 5])
        taus, fracs = performance_profile(scores)
        assert fracs[-1] == pytest.approx(0.2)

    def test_fraction_is_monotone_decreasing(self):
        """As tau increases, the fraction of episodes scoring >= tau must not increase."""
        scores = np.random.default_rng(0).uniform(0, 10, 50)
        taus, fracs = performance_profile(scores)
        assert np.all(np.diff(fracs) <= 1e-12)

    def test_fraction_bounded(self):
        scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        taus, fracs = performance_profile(scores)
        assert np.all(fracs >= 0.0)
        assert np.all(fracs <= 1.0)

    def test_custom_taus(self):
        """Passing explicit taus should use them as-is and compute correct fractions."""
        scores = np.array([1.0, 2, 3, 4, 5])
        custom = np.array([0.0, 2.5, 5.0, 7.0])
        taus, fracs = performance_profile(scores, taus=custom)
        assert np.array_equal(taus, custom)
        # tau=0.0 → all 5 >= 0 → 1.0
        assert fracs[0] == pytest.approx(1.0)
        # tau=2.5 → [3,4,5] qualify → 3/5=0.6
        assert fracs[1] == pytest.approx(0.6)
        # tau=5.0 → only [5] → 1/5=0.2
        assert fracs[2] == pytest.approx(0.2)
        # tau=7.0 → none → 0.0
        assert fracs[3] == pytest.approx(0.0)

    def test_constant_scores(self):
        """All identical scores: fraction should be 1.0 everywhere (min == max)."""
        scores = np.full(10, 3.0)
        taus, fracs = performance_profile(scores)
        assert np.all(fracs == pytest.approx(1.0))

    def test_single_element(self):
        scores = np.array([7.0])
        taus, fracs = performance_profile(scores)
        assert fracs[0] == pytest.approx(1.0)
        assert fracs[-1] == pytest.approx(1.0)

    def test_returns_numpy_arrays(self):
        scores = np.array([1.0, 2, 3])
        taus, fracs = performance_profile(scores)
        assert isinstance(taus, np.ndarray)
        assert isinstance(fracs, np.ndarray)


# ---------------------------------------------------------------------------
# stratified_bootstrap_ci
# ---------------------------------------------------------------------------

class TestStratifiedBootstrapCI:
    def _make_df(self, scores=None, seed=0):
        rng = np.random.default_rng(seed)
        if scores is None:
            scores = rng.uniform(0, 1, 30)
        return pd.DataFrame({"score": scores})

    def test_returns_required_keys(self):
        df = self._make_df()
        result = stratified_bootstrap_ci(df, lambda d: float(d["score"].mean()))
        assert {"point", "lo", "hi", "width", "ci", "n_reps"}.issubset(result.keys())

    def test_lo_less_than_hi(self):
        df = self._make_df()
        result = stratified_bootstrap_ci(df, lambda d: float(d["score"].mean()),
                                         n_reps=500, seed=1)
        assert result["lo"] < result["hi"]

    def test_width_equals_hi_minus_lo(self):
        df = self._make_df()
        result = stratified_bootstrap_ci(df, lambda d: float(d["score"].mean()),
                                         n_reps=500, seed=2)
        assert result["width"] == pytest.approx(result["hi"] - result["lo"])

    def test_point_is_statistic_of_full_data(self):
        scores = np.arange(1, 11, dtype=float)
        df = pd.DataFrame({"score": scores})
        result = stratified_bootstrap_ci(df, lambda d: float(d["score"].mean()),
                                         n_reps=200, seed=3)
        assert result["point"] == pytest.approx(5.5)

    def test_ci_stored_correctly(self):
        df = self._make_df()
        result = stratified_bootstrap_ci(df, lambda d: float(d["score"].mean()),
                                         ci=0.90, n_reps=200, seed=4)
        assert result["ci"] == pytest.approx(0.90)

    def test_n_reps_stored_correctly(self):
        df = self._make_df()
        result = stratified_bootstrap_ci(df, lambda d: float(d["score"].mean()),
                                         n_reps=123, seed=5)
        assert result["n_reps"] == 123

    def test_point_inside_ci(self):
        """The point estimate should almost always sit within its own CI."""
        scores = np.arange(1, 21, dtype=float)
        df = pd.DataFrame({"score": scores})
        result = stratified_bootstrap_ci(df, lambda d: float(d["score"].mean()),
                                         n_reps=2000, seed=42)
        assert result["lo"] <= result["point"] <= result["hi"]

    def test_constant_data_narrow_ci(self):
        """Constant data has zero variance; bootstrap CI should be very narrow."""
        df = pd.DataFrame({"score": np.full(20, 5.0)})
        result = stratified_bootstrap_ci(df, lambda d: float(d["score"].mean()),
                                         n_reps=500, seed=6)
        assert result["width"] == pytest.approx(0.0, abs=1e-10)

    def test_wider_ci_with_more_variance(self):
        """High-variance data should produce a wider CI than low-variance data."""
        low_var  = pd.DataFrame({"score": np.full(50, 0.5)})
        high_var = pd.DataFrame({"score": np.random.default_rng(7).uniform(0, 1, 50)})
        ci_low  = stratified_bootstrap_ci(low_var,  lambda d: float(d["score"].mean()),
                                          n_reps=500, seed=8)
        ci_high = stratified_bootstrap_ci(high_var, lambda d: float(d["score"].mean()),
                                          n_reps=500, seed=8)
        assert ci_high["width"] > ci_low["width"]

    def test_stratified_resampling(self):
        """With strata_cols provided the result should still satisfy lo < hi."""
        df = pd.DataFrame({
            "score": np.arange(1, 11, dtype=float),
            "group": ["A"] * 5 + ["B"] * 5,
        })
        result = stratified_bootstrap_ci(
            df,
            stat_fn=lambda d: float(d["score"].mean()),
            strata_cols=["group"],
            n_reps=500,
            seed=9,
        )
        assert result["lo"] < result["hi"]

    def test_iqm_as_statistic(self):
        """IQM can be used as the stat_fn; CI should be valid."""
        scores = np.linspace(0, 1, 40)
        df = pd.DataFrame({"score": scores})
        result = stratified_bootstrap_ci(
            df,
            stat_fn=lambda d: iqm(d["score"].to_numpy()),
            n_reps=500,
            seed=10,
        )
        assert result["lo"] < result["hi"]
        assert result["point"] == pytest.approx(iqm(scores), rel=1e-6)


# ---------------------------------------------------------------------------
# analytical_ci
# ---------------------------------------------------------------------------

class TestAnalyticalCI:
    def test_ci_contains_mean(self):
        x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = analytical_ci(x)
        assert result["lo"] < result["point"] < result["hi"]

    def test_empty_returns_nan(self):
        result = analytical_ci(np.array([]))
        assert np.isnan(result["point"])

    def test_single_value_zero_width(self):
        result = analytical_ci(np.array([5.0]))
        assert result["width"] == 0.0


# ---------------------------------------------------------------------------
# binomial_ci
# ---------------------------------------------------------------------------

class TestBinomialCI:
    def test_perfect_interval_tight(self):
        result = binomial_ci(50, 50)
        assert result["lo"] > 0.9
        assert result["hi"] == pytest.approx(1.0, abs=1e-9)

    def test_half_interval_contains_point(self):
        result = binomial_ci(5, 10)
        assert result["point"] == pytest.approx(0.5)
        assert result["lo"] < 0.5 < result["hi"]

    def test_wilson_interval(self):
        result = binomial_ci(5, 10, method="wilson")
        assert result["point"] == pytest.approx(0.5)
        assert result["lo"] < result["hi"]

    def test_zero_n_returns_nan(self):
        result = binomial_ci(0, 0)
        assert np.isnan(result["point"])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown binomial CI method"):
            binomial_ci(5, 10, method="bad")
