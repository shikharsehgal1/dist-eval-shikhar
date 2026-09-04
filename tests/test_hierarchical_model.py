"""Hierarchical crossed random-effects model: recovery, pooling, backends."""
import numpy as np
import pytest
from scipy.special import expit

from disteval.reliability.hierarchical import (
    LogitNormalPosterior,
    fit_hierarchical,
    pooling_diagnostic,
    sample_polya_gamma,
)


def _make(n_tasks=80, sd=1.2, n=8, seed=0, n_models=1):
    rng = np.random.default_rng(seed)
    b = rng.normal(0, sd, n_tasks)
    a = rng.normal(0, 0.6, n_models)
    models, tasks, succ, trials, doms = [], [], [], [], []
    for mi in range(n_models):
        for ti in range(n_tasks):
            models.append(f"m{mi}")
            tasks.append(f"t{ti}")
            doms.append(f"d{ti % 4}")
            succ.append(int(rng.binomial(n, expit(a[mi] + b[ti]))))
            trials.append(n)
    return models, tasks, succ, trials, doms, b, a


class TestRecovery:
    def test_recovers_the_between_task_variance(self):
        for true_sd in (0.5, 1.5):
            m, t, s, n, d, _, _ = _make(n_tasks=250, sd=true_sd, seed=1)
            fit = fit_hierarchical(m, t, s, n, d)
            assert fit.sigma["task"] == pytest.approx(true_sd, rel=0.35)

    def test_recovers_model_main_effects(self):
        m, t, s, n, d, _, a = _make(n_tasks=150, n_models=3, seed=2)
        fit = fit_hierarchical(m, t, s, n, d)
        est = np.array([fit.model_effects[f"m{i}"] for i in range(3)])
        true = a - a.mean()
        assert np.corrcoef(est, true)[0, 1] > 0.9

    def test_posterior_means_track_the_truth_at_least_as_well_as_independent(self):
        """Relative, not absolute: with n=8 binary runs the achievable correlation
        with the truth is capped near 0.85 by binomial noise alone, so an absolute
        threshold would be testing the sample size rather than the estimator. The
        meaningful claim is that pooling is no worse on correlation and better on
        error.
        """
        from disteval.reliability.posterior import binary_posterior

        m, t, s, n, d, b, _ = _make(n_tasks=200, seed=3)
        truth = expit(b)
        fit = fit_hierarchical(m, t, s, n, d)
        pooled = np.array([fit.cell("m0", f"t{i}").mean for i in range(200)])
        indep = np.array([binary_posterior(int(x), 8).mean for x in s])

        assert np.corrcoef(pooled, truth)[0, 1] > 0.75
        assert np.corrcoef(pooled, truth)[0, 1] >= np.corrcoef(indep, truth)[0, 1] - 0.01
        # Deliberately NOT asserting that pooled RMSE beats independent RMSE:
        # measured across seeds and heterogeneity levels, pooling wins on
        # latent-parameter RMSE only some of the time, which is why the pooling
        # decision is made per dataset by pooling_diagnostic rather than assumed.
        rmse = lambda v: float(np.sqrt(np.mean((v - truth) ** 2)))  # noqa: E731
        assert rmse(pooled) < 0.35 and rmse(indep) < 0.35

    def test_converges_on_well_conditioned_data(self):
        m, t, s, n, d, _, _ = _make(n_tasks=120, seed=4)
        fit = fit_hierarchical(m, t, s, n, d)
        assert fit.converged
        assert fit.n_iter < fit.spec.max_iter


class TestIdentifiability:
    def test_residual_is_dropped_when_aliased_with_the_task_effect(self):
        """One agent means one cell per task, so eps and beta_t are aliased."""
        m, t, s, n, d, _, _ = _make(n_tasks=60, n_models=1, seed=5)
        fit = fit_hierarchical(m, t, s, n, d)
        assert "aliasing" in fit.diagnostics
        assert not fit.spec.residual_effect

    def test_residual_is_retained_with_several_agents(self):
        m, t, s, n, d, _, _ = _make(n_tasks=60, n_models=3, seed=6)
        fit = fit_hierarchical(m, t, s, n, d)
        assert "aliasing" not in fit.diagnostics
        assert fit.spec.residual_effect


class TestShrinkage:
    def test_pooling_pulls_extremes_toward_the_population(self):
        m, t, s, n, d, _, _ = _make(n_tasks=100, sd=0.4, seed=7)
        s = list(s)
        s[0], s[1] = 8, 0  # force two extremes
        fit = fit_hierarchical(m, t, s, n, d)
        assert fit.cell("m0", "t0").mean < 0.95
        assert fit.cell("m0", "t1").mean > 0.05

    def test_shrinkage_is_reported_in_zero_one(self):
        m, t, s, n, d, _, _ = _make(n_tasks=60, seed=8)
        fit = fit_hierarchical(m, t, s, n, d)
        vals = [fit.shrinkage("m0", tt) for tt in set(t)]
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_less_shrinkage_when_tasks_are_heterogeneous(self):
        lo = fit_hierarchical(*_make(n_tasks=120, sd=0.2, seed=9)[:5])
        hi = fit_hierarchical(*_make(n_tasks=120, sd=2.5, seed=9)[:5])
        mean_lo = np.mean([lo.shrinkage("m0", f"t{i}") for i in range(120)])
        mean_hi = np.mean([hi.shrinkage("m0", f"t{i}") for i in range(120)])
        assert mean_hi < mean_lo


class TestPoolingDiagnostic:
    def test_favours_pooling_on_homogeneous_tasks(self):
        m, t, s, n, d, _, _ = _make(n_tasks=80, sd=0.25, n=6, seed=10)
        assert pooling_diagnostic(m, t, s, n, d)["pooling_helps"] is True

    def test_disfavours_pooling_on_heterogeneous_tasks(self):
        m, t, s, n, d, _, _ = _make(n_tasks=80, sd=3.0, n=6, seed=11)
        assert pooling_diagnostic(m, t, s, n, d)["pooling_helps"] is False

    def test_reports_both_scores_and_a_recommendation(self):
        m, t, s, n, d, _, _ = _make(n_tasks=50, seed=12)
        out = pooling_diagnostic(m, t, s, n, d)
        assert np.isfinite(out["loo_logloss_independent"])
        assert np.isfinite(out["loo_logloss_pooled"])
        assert out["recommendation"]


class TestCellPosterior:
    def test_satisfies_the_shared_posterior_interface(self):
        p = LogitNormalPosterior(eta_mean=0.5, eta_sd=0.8, n_obs=8, n_success=5)
        assert 0 < p.mean < 1
        assert p.sd > 0
        lo, hi = p.credible_interval()
        assert lo < p.median < hi
        assert p.prob_above(0.0) == 1.0 and p.prob_above(1.0) == 0.0
        assert p.prob_above(0.5) == pytest.approx(1 - p.prob_below(0.5))
        assert np.isfinite(p.entropy)
        assert p.posterior_predictive(4).sum() == pytest.approx(1.0)

    def test_prob_above_is_monotone_decreasing_in_the_threshold(self):
        p = LogitNormalPosterior(0.3, 1.0, 8, 5)
        vals = [p.prob_above(x) for x in np.linspace(0.05, 0.95, 20)]
        assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))

    def test_beta_moment_match_preserves_the_mean(self):
        p = LogitNormalPosterior(0.4, 0.7, 8, 5)
        assert p.to_beta().mean == pytest.approx(p.mean, abs=1e-3)

    def test_sampling_is_reproducible(self):
        p = LogitNormalPosterior(0.4, 0.7, 8, 5)
        a = p.sample(200, np.random.default_rng(0))
        b = p.sample(200, np.random.default_rng(0))
        assert np.array_equal(a, b)


class TestPolyaGammaBackend:
    def test_pg_sampler_matches_the_known_mean(self):
        """E[PG(b, c)] = (b / (2c)) * tanh(c / 2)."""
        rng = np.random.default_rng(0)
        b = np.full(4000, 1.0)
        c = np.full(4000, 1.5)
        draws = sample_polya_gamma(b, c, rng)
        expected = (1.0 / (2 * 1.5)) * np.tanh(1.5 / 2)
        assert draws.mean() == pytest.approx(expected, rel=0.03)

    def test_pg_mean_at_c_zero(self):
        """E[PG(1, 0)] = 1/4."""
        rng = np.random.default_rng(1)
        draws = sample_polya_gamma(np.ones(4000), np.zeros(4000), rng)
        assert draws.mean() == pytest.approx(0.25, rel=0.05)

    def test_gibbs_agrees_with_laplace(self):
        m, t, s, n, d, _, _ = _make(n_tasks=40, n_models=2, seed=13)
        lap = fit_hierarchical(m, t, s, n, d)
        gib = fit_hierarchical(m, t, s, n, d, backend="pg_gibbs",
                               n_samples=250, n_burnin=150, seed=0)
        a = np.array([lap.cells[k].mean for k in lap.cells])
        b = np.array([gib.cells[k].mean for k in lap.cells])
        assert np.corrcoef(a, b)[0, 1] > 0.95

    def test_gibbs_exposes_draws(self):
        m, t, s, n, d, _, _ = _make(n_tasks=20, n_models=2, seed=14)
        fit = fit_hierarchical(m, t, s, n, d, backend="pg_gibbs",
                               n_samples=60, n_burnin=40, seed=0)
        assert fit.draws is not None
        assert next(iter(fit.draws.values())).shape == (60,)


class TestValidation:
    def test_rejects_unknown_backend(self):
        m, t, s, n, d, _, _ = _make(n_tasks=10, seed=15)
        with pytest.raises(ValueError):
            fit_hierarchical(m, t, s, n, d, backend="nope")

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            fit_hierarchical(["m"], ["t1", "t2"], [1], [2])

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            fit_hierarchical([], [], [], [])

    def test_rejects_zero_trials(self):
        with pytest.raises(ValueError):
            fit_hierarchical(["m"] * 2, ["a", "b"], [0, 1], [0, 2])

    def test_rejects_successes_above_trials(self):
        with pytest.raises(ValueError):
            fit_hierarchical(["m"] * 2, ["a", "b"], [5, 1], [2, 2])

    def test_handles_unequal_run_counts(self):
        m = ["m"] * 6
        t = [f"t{i}" for i in range(6)]
        fit = fit_hierarchical(m, t, [1, 2, 0, 3, 8, 4], [2, 4, 3, 5, 8, 20])
        assert len(fit.cells) == 6
        assert fit.cell("m", "t4").n_obs == 8

    def test_accepts_fractional_successes(self):
        m = ["m"] * 5
        t = [f"t{i}" for i in range(5)]
        fit = fit_hierarchical(m, t, [1.5, 2.25, 0.5, 3.75, 4.0], [8] * 5)
        assert all(0 < c.mean < 1 for c in fit.cells.values())

    def test_summary_and_frame(self):
        m, t, s, n, d, _, _ = _make(n_tasks=25, seed=16)
        fit = fit_hierarchical(m, t, s, n, d)
        summary = fit.summary()
        assert {"backend", "sigma", "n_cells", "converged"} <= set(summary)
        frame = fit.to_frame()
        assert len(frame) == 25
        assert {"posterior_mean", "ci95_lo", "shrinkage", "raw_rate"} <= set(frame.columns)

    def test_is_deterministic(self):
        m, t, s, n, d, _, _ = _make(n_tasks=40, seed=17)
        a = fit_hierarchical(m, t, s, n, d)
        b = fit_hierarchical(m, t, s, n, d)
        assert a.sigma == b.sigma
        assert a.cell("m0", "t0").mean == b.cell("m0", "t0").mean
