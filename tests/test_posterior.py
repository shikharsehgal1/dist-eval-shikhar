"""Tests for latent-performance posteriors, including the sanity checks the
research brief calls for: 8/8 must be stronger evidence of reliability than 1/1,
the continuous estimator must reduce exactly to the binary one, and edge cases
(no runs, one run, all-success, all-failure) must not silently produce nonsense.
"""
import numpy as np
import pytest

from disteval.reliability.posterior import (
    JEFFREYS_PRIOR,
    UNIFORM_PRIOR,
    BetaPrior,
    binary_posterior,
    continuous_posterior,
    dispersion_ratio,
    posterior_from_scores,
)


class TestBetaPrior:
    def test_rejects_nonpositive_parameters(self):
        for a, b in [(0, 1), (1, 0), (-1, 1)]:
            with pytest.raises(ValueError):
                BetaPrior(a, b)

    def test_from_mean_strength_roundtrip(self):
        p = BetaPrior.from_mean_strength(0.3, 10)
        assert p.mean == pytest.approx(0.3)
        assert p.strength == pytest.approx(10)

    def test_from_mean_strength_rejects_boundary_mean(self):
        for m in (0.0, 1.0, -0.1, 1.1):
            with pytest.raises(ValueError):
                BetaPrior.from_mean_strength(m, 10)

    def test_jeffreys_is_weaker_than_uniform(self):
        assert JEFFREYS_PRIOR.strength < UNIFORM_PRIOR.strength


class TestBinaryPosterior:
    def test_conjugate_update(self):
        p = binary_posterior(3, 8, BetaPrior(1, 1))
        assert p.alpha == pytest.approx(4)
        assert p.beta == pytest.approx(6)
        assert p.mean == pytest.approx(0.4)

    def test_more_evidence_gives_higher_reliability_confidence(self):
        """8/8 successes must beat 1/1 under the same prior. The brief's example."""
        many = binary_posterior(8, 8)
        one = binary_posterior(1, 1)
        assert many.prob_above(0.9) > one.prob_above(0.9)
        assert many.sd < one.sd
        lo_m, hi_m = many.credible_interval()
        lo_o, hi_o = one.credible_interval()
        assert (hi_m - lo_m) < (hi_o - lo_o)
        assert lo_m > lo_o

    def test_uncertainty_shrinks_monotonically_with_more_iid_data(self):
        """At a fixed success proportion, more runs must tighten the posterior.

        The proportion is held fixed deliberately: comparing 0/1 with 1/2 varies
        the posterior mean as well as the sample size, and the sd of a Beta
        depends on both.
        """
        prev = float("inf")
        for n in (2, 4, 8, 16, 32, 64, 128):
            p = binary_posterior(n // 2, n)
            assert p.mean == pytest.approx(0.5)
            assert p.sd < prev
            prev = p.sd

    def test_uncertainty_shrinks_for_all_success_too(self):
        prev = float("inf")
        for n in (1, 2, 4, 8, 16, 32):
            p = binary_posterior(n, n)
            assert p.sd < prev
            prev = p.sd

    def test_all_success_never_reaches_certainty(self):
        p = binary_posterior(8, 8)
        assert p.mean < 1.0
        assert p.prob_above(0.99) < 1.0

    def test_all_failure_never_reaches_certainty(self):
        p = binary_posterior(0, 8)
        assert p.mean > 0.0
        assert p.prob_below(0.01) < 1.0

    def test_zero_trials_returns_prior(self):
        p = binary_posterior(0, 0, BetaPrior(2, 3))
        assert p.mean == pytest.approx(0.4)
        assert p.n_obs == 0

    def test_rejects_impossible_counts(self):
        for s, n in [(5, 3), (-1, 3), (1, -2)]:
            with pytest.raises(ValueError):
                binary_posterior(s, n)

    def test_prob_above_boundaries(self):
        p = binary_posterior(4, 8)
        assert p.prob_above(0.0) == 1.0
        assert p.prob_above(1.0) == 0.0
        assert p.prob_above(0.5) == pytest.approx(1 - p.prob_below(0.5))

    def test_credible_interval_contains_mean_and_widens_with_level(self):
        p = binary_posterior(3, 10)
        lo80, hi80 = p.credible_interval(0.80)
        lo99, hi99 = p.credible_interval(0.99)
        assert lo80 <= p.median <= hi80
        assert lo99 < lo80 and hi99 > hi80

    def test_credible_level_must_be_a_probability(self):
        p = binary_posterior(3, 10)
        for lvl in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError):
                p.credible_interval(lvl)

    def test_posterior_predictive_is_a_distribution(self):
        p = binary_posterior(3, 8)
        pmf = p.posterior_predictive(5)
        assert pmf.shape == (6,)
        assert pmf.sum() == pytest.approx(1.0)
        assert np.all(pmf >= 0)

    def test_posterior_predictive_mean_matches_posterior_mean(self):
        p = binary_posterior(3, 8)
        n = 10
        pmf = p.posterior_predictive(n)
        assert float(np.dot(np.arange(n + 1), pmf)) == pytest.approx(n * p.mean, rel=1e-9)

    def test_sampling_is_reproducible_and_in_range(self):
        p = binary_posterior(3, 8)
        a = p.sample(500, np.random.default_rng(0))
        b = p.sample(500, np.random.default_rng(0))
        assert np.array_equal(a, b)
        assert np.all((a >= 0) & (a <= 1))

    def test_mode_absent_when_posterior_is_j_shaped(self):
        assert binary_posterior(0, 0, BetaPrior(0.5, 0.5)).mode is None
        assert binary_posterior(5, 8, BetaPrior(1, 1)).mode is not None


class TestContinuousPosterior:
    def test_reduces_to_binary_for_binary_data(self):
        """The defining property: exact agreement on 0/1 data."""
        scores = [1, 0, 1, 1, 0, 1, 0, 0]
        cont = continuous_posterior(scores)
        exact = binary_posterior(int(sum(scores)), len(scores))
        assert cont.mean == pytest.approx(exact.mean, rel=1e-6)
        assert cont.n_eff == pytest.approx(len(scores), rel=1e-6)

    def test_deterministic_partial_credit_is_tightly_estimated(self):
        """Every run scores 0.5: the latent mean really is pinned down."""
        p = continuous_posterior([0.5] * 8)
        assert p.mean == pytest.approx(0.5, abs=0.02)
        lo, hi = p.credible_interval()
        assert (hi - lo) < 0.1
        assert p.n_eff > 8

    def test_dispersion_ratio_is_one_for_binary_and_small_for_constant(self):
        assert dispersion_ratio([0, 1, 0, 1, 1, 0]) == pytest.approx(1.0, abs=0.3)
        assert dispersion_ratio([0.5] * 8) < 0.01

    def test_dispersion_is_conservative_at_degenerate_means(self):
        assert dispersion_ratio([0.0] * 5) == 1.0
        assert dispersion_ratio([1.0] * 5) == 1.0
        assert dispersion_ratio([0.3]) == 1.0

    def test_rejects_out_of_range_scores(self):
        for bad in ([-0.1, 0.5], [0.5, 1.2]):
            with pytest.raises(ValueError):
                continuous_posterior(bad)

    def test_empty_returns_prior(self):
        p = continuous_posterior([], BetaPrior(2, 2))
        assert p.mean == pytest.approx(0.5)
        assert p.n_obs == 0

    def test_phi_floor_bounds_confidence(self):
        """A low-variance sample cannot manufacture unbounded confidence."""
        tight = continuous_posterior([0.5] * 4, phi_floor=1e-3)
        assert tight.n_eff <= 4 / 1e-3


class TestDispatch:
    def test_binary_data_takes_the_exact_path(self):
        assert posterior_from_scores([1, 0, 1, 1]).binary is True

    def test_continuous_data_takes_the_quasi_binomial_path(self):
        assert posterior_from_scores([0.3, 0.7, 0.5]).binary is False

    def test_threshold_dichotomises(self):
        p = posterior_from_scores([0.9, 0.4, 0.95, 0.2], success_threshold=0.8)
        assert p.binary is True
        assert p.alpha == pytest.approx(JEFFREYS_PRIOR.alpha + 2)

    def test_empty_input(self):
        assert posterior_from_scores([]).n_obs == 0

    def test_tied_scores_are_handled(self):
        p = posterior_from_scores([0.6] * 5)
        assert 0 < p.mean < 1
        assert np.isfinite(p.sd)

    def test_single_run_is_dominated_by_the_prior(self):
        p = posterior_from_scores([1.0])
        assert p.n_obs == 1
        lo, hi = p.credible_interval()
        assert (hi - lo) > 0.5, "one run should leave a very wide interval"

    def test_to_dict_is_serialisable(self):
        d = posterior_from_scores([1, 0, 1]).to_dict()
        assert {"posterior_mean", "ci95_lo", "ci95_hi", "n_obs"} <= set(d)
        assert all(isinstance(v, (int, float, bool)) for v in d.values())
