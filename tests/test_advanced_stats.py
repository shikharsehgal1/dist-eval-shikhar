"""Tests for the advanced math/stats additions:

shrinkage (empirical Bayes / James-Stein), evt (extreme value theory),
bootstrap.betting_cs, compare (conformal / MMD / energy), metrics.divergences,
and best_arm (pure-exploration bandits).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from disteval import best_arm, compare, evt, metrics, shrinkage
from disteval.bootstrap import betting_cs, confidence_sequence


# ---------------------------------------------------------------------------
# shrinkage
# ---------------------------------------------------------------------------

class TestShrinkage:
    def test_eb_passrate_beats_mle_mse(self):
        rng = np.random.default_rng(0)
        n_tasks, trials = 300, 5
        true_p = rng.beta(2, 5, n_tasks)
        k = rng.binomial(trials, true_p)
        res = shrinkage.empirical_bayes_passrate(k, np.full(n_tasks, trials))
        mse_mle = np.mean((res["p_raw"] - true_p) ** 2)
        mse_eb = np.mean((res["p_shrunk"] - true_p) ** 2)
        assert mse_eb < mse_mle

    def test_mom_recovers_prior_mean(self):
        rng = np.random.default_rng(1)
        n_tasks, trials = 3000, 25
        true_p = rng.beta(2, 5, n_tasks)
        k = rng.binomial(trials, true_p)
        prior = shrinkage.fit_beta_binomial_mom(k, np.full(n_tasks, trials))
        assert prior["global_rate"] == pytest.approx(2 / 7, abs=0.03)
        implied = prior["alpha"] / (prior["alpha"] + prior["beta"])
        assert implied == pytest.approx(2 / 7, abs=0.05)

    def test_james_stein_dominates_mle_on_average(self):
        rng = np.random.default_rng(2)
        m, sigma2, sims = 10, 1.0, 400
        mse_js, mse_mle = 0.0, 0.0
        for _ in range(sims):
            true = rng.normal(0, 2, m)
            obs = true + rng.normal(0, 1, m)
            js = shrinkage.james_stein(obs, sigma2)
            mse_js += np.mean((js - true) ** 2)
            mse_mle += np.mean((obs - true) ** 2)
        assert mse_js < mse_mle

    def test_james_stein_positive_part_and_small_m(self):
        x = np.array([1.0, 5.0, 9.0])
        js = shrinkage.james_stein(x, 1.0)
        center = x.mean()
        # shrinks toward center, never crosses it
        assert np.all(np.sign(js - center) == np.sign(x - center))
        assert np.all(np.abs(js - center) <= np.abs(x - center) + 1e-9)
        # m < 3 returned unchanged
        np.testing.assert_array_equal(shrinkage.james_stein([1.0, 2.0], 1.0), [1.0, 2.0])

    def test_eb_credible_interval_brackets_point(self):
        res = shrinkage.eb_credible_interval([0, 2, 3], [3, 3, 3])
        assert np.all(res["lo"] <= res["point"])
        assert np.all(res["point"] <= res["hi"])
        assert np.all((res["lo"] >= 0) & (res["hi"] <= 1))


# ---------------------------------------------------------------------------
# evt
# ---------------------------------------------------------------------------

class TestEVT:
    def _gpd_lower(self, rng, xi, beta, n=6000):
        y = stats.genpareto.rvs(xi, scale=beta, size=n, random_state=rng)
        return -y  # lower-tail data with a GPD upper tail on -x

    def test_recovers_gpd_params(self):
        rng = np.random.default_rng(0)
        x = self._gpd_lower(rng, 0.3, 1.0)
        fit = evt.fit_gpd_pot(x, tail="lower", threshold=0.0)
        assert fit["xi"] == pytest.approx(0.3, abs=0.05)
        assert fit["beta"] == pytest.approx(1.0, abs=0.1)

    def test_var_matches_analytic(self):
        rng = np.random.default_rng(1)
        x = self._gpd_lower(rng, 0.2, 1.0)
        fit = evt.fit_gpd_pot(x, tail="lower", threshold=0.0)
        analytic = -stats.genpareto.ppf(0.9, 0.2, scale=1.0)
        assert evt.evt_var(fit, 0.1) == pytest.approx(analytic, rel=0.1)

    def test_cvar_infinite_mean_is_nan(self):
        rng = np.random.default_rng(2)
        x = self._gpd_lower(rng, 1.2, 1.0)
        fit = evt.fit_gpd_pot(x, tail="lower", threshold=0.0)
        fit["xi"] = 1.5  # force infinite-mean regime
        with pytest.warns(UserWarning):
            assert np.isnan(evt.evt_cvar(fit, 0.1))

    def test_low_confidence_flag(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 1, 20)
        fit = evt.fit_gpd_pot(x, tail="lower", tail_fraction=0.25)
        assert fit["low_confidence"] is True  # ~5 exceedances < 15

    def test_hill_recovers_pareto_index(self):
        rng = np.random.default_rng(4)
        x = stats.pareto.rvs(3.0, size=5000, random_state=rng)  # alpha=3 -> gamma=1/3
        gamma = evt.hill_estimator(x, k=500)
        assert gamma == pytest.approx(1 / 3, abs=0.08)

    def test_hill_nan_for_bounded_scores(self):
        assert np.isnan(evt.hill_estimator([-0.5, 0.2, 0.4, 0.9]))  # non-positive present

    def test_bootstrap_ci_brackets_and_flags(self):
        rng = np.random.default_rng(5)
        x = self._gpd_lower(rng, 0.2, 1.0, n=400)
        res = evt.evt_bootstrap_ci(x, "cvar", alpha=0.1, n_reps=200, tail="lower", threshold=0.0)
        assert res["lo"] <= res["point"] <= res["hi"]


# ---------------------------------------------------------------------------
# betting_cs
# ---------------------------------------------------------------------------

class TestBettingCS:
    def test_anytime_coverage_and_tighter_than_hoeffding(self):
        rng = np.random.default_rng(0)
        p, n_runs, n = 0.3, 300, 100
        covered = 0
        betting_w, hoeff_w = [], []
        for _ in range(n_runs):
            x = (rng.uniform(size=n) < p).astype(float)
            b = betting_cs(x, ci=0.9)
            lo = np.array(b["running_lo"])
            hi = np.array(b["running_hi"])
            ok = np.isfinite(lo) & np.isfinite(hi)
            if np.all((lo[ok] <= p) & (p <= hi[ok])):
                covered += 1
            betting_w.append(b["width"])
            hoeff_w.append(confidence_sequence(x, ci=0.9)["width"])
        assert covered / n_runs >= 0.85          # anytime-valid (nominal 0.9)
        assert np.median(betting_w) < np.median(hoeff_w)  # variance-adaptive => tighter

    def test_empty_and_degenerate(self):
        assert betting_cs([])["n"] == 0
        with pytest.raises(ValueError):
            betting_cs([0.5, 0.5], lo=1.0, hi=1.0)


# ---------------------------------------------------------------------------
# compare: conformal / MMD / energy
# ---------------------------------------------------------------------------

class TestConformal:
    def test_two_sided_marginal_coverage(self):
        rng = np.random.default_rng(0)
        alpha, trials, covered = 0.1, 500, 0
        for _ in range(trials):
            calib = rng.normal(size=60)
            nxt = rng.normal()
            iv = compare.conformal_interval(calib, alpha=alpha)
            if iv["lo"] <= nxt <= iv["hi"]:
                covered += 1
        assert covered / trials >= 0.88

    def test_one_sided_lower_bound(self):
        rng = np.random.default_rng(1)
        alpha, trials, covered = 0.1, 500, 0
        for _ in range(trials):
            calib = rng.normal(size=80)
            nxt = rng.normal()
            iv = compare.conformal_interval(calib, alpha=alpha, two_sided=False)
            assert iv["hi"] == float("inf")
            if iv["lo"] <= nxt:
                covered += 1
        assert covered / trials >= 0.88


class TestKernelTests:
    def test_mmd_same_distribution_not_significant(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=80), rng.normal(size=80)
        res = compare.mmd_test(a, b, n_perm=500, seed=0)
        assert res["p"] > 0.1

    def test_mmd_detects_location_shift(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(0, 1, 100), rng.normal(3, 1, 100)
        assert compare.mmd_test(a, b, n_perm=500, seed=0)["p"] < 0.05

    def test_mmd_detects_variance_difference(self):
        # Same mean, different variance: KS/Mann-Whitney are weak here.
        rng = np.random.default_rng(2)
        a, b = rng.normal(0, 1, 200), rng.normal(0, 3, 200)
        assert compare.mmd_test(a, b, n_perm=500, seed=0)["p"] < 0.05

    def test_energy_distance_nonneg_and_near_zero_for_same_dist(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(size=2000), rng.normal(size=2000)
        same = compare.energy_distance(a, b)   # independent, same distribution
        far = compare.energy_distance(a, rng.normal(3, 1, 2000))
        assert same >= 0.0
        assert same < 0.05          # off-diagonal estimator -> ~0 for equal dists
        assert far > same

    def test_energy_test_detects_shift(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(0, 1, 100), rng.normal(2, 1, 100)
        res = compare.energy_test(a, b, n_perm=500, seed=0)
        assert res["p"] < 0.05 and res["energy_distance"] > 0


# ---------------------------------------------------------------------------
# metrics.divergences
# ---------------------------------------------------------------------------

class TestDivergences:
    def test_identical_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=1000)
        d = metrics.divergences(x, x)
        assert d["tv"] == pytest.approx(0.0, abs=1e-6)
        assert d["hellinger"] == pytest.approx(0.0, abs=1e-6)
        assert d["kl"] == pytest.approx(0.0, abs=1e-6)

    def test_bounds_and_ordering(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, 2000)
        b = rng.normal(3, 1, 2000)
        near = rng.normal(0.2, 1, 2000)
        d_far = metrics.divergences(a, b)
        d_near = metrics.divergences(a, near)
        assert 0.0 <= d_far["tv"] <= 1.0
        assert 0.0 <= d_far["hellinger"] <= 1.0
        assert d_far["kl"] >= 0.0
        assert d_far["tv"] > d_near["tv"]  # farther distributions => larger TV


# ---------------------------------------------------------------------------
# best_arm
# ---------------------------------------------------------------------------

class TestBestArm:
    def _bernoulli_puller(self, means, rng):
        means = np.asarray(means, dtype=float)
        return lambda arm: float(rng.uniform() < means[arm])

    def test_sequential_halving_finds_best(self):
        rng = np.random.default_rng(0)
        means = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85])
        wins = 0
        for _ in range(100):
            pull = self._bernoulli_puller(means, rng)
            if best_arm.sequential_halving(pull, k=8, budget=600)["best"] == 7:
                wins += 1
        assert wins / 100 >= 0.8

    def test_successive_elimination_correct_selection(self):
        rng = np.random.default_rng(1)
        means = np.array([0.5, 0.5, 0.9, 0.5, 0.5])  # arm 2 clearly best
        wins, pulls = 0, []
        for _ in range(60):
            pull = self._bernoulli_puller(means, rng)
            res = best_arm.successive_elimination(pull, k=5, delta=0.1, max_pulls=20000)
            wins += int(res["best"] == 2)
            pulls.append(res["total_pulls"])
        assert wins / 60 >= 0.9
        # adaptivity: an easy arm gets far fewer pulls than the leader
        last = best_arm.successive_elimination(self._bernoulli_puller(means, rng), k=5, delta=0.1)
        assert last["pulls"][2] >= max(last["pulls"][a] for a in (0, 1, 3, 4))

    def test_single_arm(self):
        res = best_arm.sequential_halving(lambda a: 1.0, k=1, budget=10)
        assert res["best"] == 0
