"""Empirical-Bayes shrinkage / partial pooling for the few-runs-per-task regime.

disteval's defining situation is a handful of runs (3-8) on each of many tasks.
A raw per-task estimate — pass@1, mean score, or q̄/q* — is then extremely noisy,
and *ranking* tasks by a raw estimate invites the winner's curse (the top task is
disproportionately one whose gap is large by chance). Shrinkage fixes both: it
pulls each noisy per-task estimate toward the population, borrowing strength
across tasks, and provably lowers total error.

Three tools, all pure numpy/scipy:
  - empirical_bayes_passrate — Beta-Binomial partial pooling of pass rates.
  - james_stein — positive-part James-Stein shrinkage of per-task means.
  - eb_credible_interval — per-task intervals informed by the population prior.

Guard: when tasks are genuinely heterogeneous (large between-task variance) the
estimated shrinkage automatically goes to ~0, so pooling never overrides real
signal. Pool within difficulty strata when tiers differ a lot.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


__all__ = [
    "fit_beta_binomial_mom",
    "empirical_bayes_passrate",
    "james_stein",
    "eb_credible_interval",
]


def fit_beta_binomial_mom(successes, trials) -> dict:
    """Method-of-moments Beta(α,β) prior for a population of binomial pass rates.

    Crucially, this subtracts the *within-task* binomial sampling variance before
    matching moments, so the prior reflects genuine between-task spread rather
    than few-runs noise (a naive Beta fit to raw p_i = k_i/n_i over-estimates the
    prior variance and under-shrinks). Falls back to a weak Beta(1,1) prior when
    the corrected between-task variance is non-positive (tasks look homogeneous).

    Returns {alpha, beta, global_rate}.
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if k.size == 0 or np.any(n <= 0):
        raise ValueError("need non-empty successes and positive trials")
    p = k / n
    m = float(p.mean())
    if p.size < 2 or m <= 0.0 or m >= 1.0:
        return {"alpha": 1.0, "beta": 1.0, "global_rate": m if 0 < m < 1 else 0.5}
    total_var = float(p.var(ddof=1))
    within = float(np.mean(p * (1 - p) / n))  # mean binomial sampling variance
    tau2 = total_var - within                 # between-task variance estimate
    max_tau2 = m * (1 - m)                     # variance of a Beta is < mean*(1-mean)
    if tau2 <= 0 or tau2 >= max_tau2:
        # Homogeneous (or noise-dominated): a strong-ish prior at the global rate.
        if tau2 <= 0:
            return {"alpha": max(m * 50, 1e-3), "beta": max((1 - m) * 50, 1e-3),
                    "global_rate": m}
        tau2 = 0.99 * max_tau2
    common = m * (1 - m) / tau2 - 1.0
    common = max(common, 1e-6)
    return {"alpha": m * common, "beta": (1 - m) * common, "global_rate": m}


def empirical_bayes_passrate(successes, trials, prior: dict | None = None) -> dict:
    """Empirical-Bayes shrunk per-task pass rates via Beta-Binomial partial pooling.

    Each raw pass rate k_i/n_i is replaced by the posterior mean
    (k_i + α) / (n_i + α + β), which pulls low-n tasks strongly toward the global
    rate and leaves high-n tasks near their MLE. This is the "batting average"
    estimator; it lowers total error versus the raw rates in the few-runs regime.

    Returns {p_shrunk (array), p_raw (array), alpha, beta, global_rate,
    shrinkage (per-task weight toward the prior in [0,1])}.
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if prior is None:
        prior = fit_beta_binomial_mom(k, n)
    a, b = prior["alpha"], prior["beta"]
    p_shrunk = (k + a) / (n + a + b)
    shrinkage = (a + b) / (n + a + b)  # weight placed on the prior
    return {
        "p_shrunk": p_shrunk,
        "p_raw": k / n,
        "alpha": a,
        "beta": b,
        "global_rate": prior.get("global_rate", a / (a + b)),
        "shrinkage": shrinkage,
    }


def james_stein(means, sigma2, grand: float | None = None) -> np.ndarray:
    """Positive-part James-Stein shrinkage of per-task means toward a common center.

    For m >= 3 tasks the James-Stein estimator has strictly lower total MSE than
    the vector of raw means (Stein's paradox). Each mean is pulled toward the
    grand mean by a data-driven factor that vanishes when the spread between tasks
    is large (genuinely heterogeneous tasks are left alone) and is strong when the
    spread looks like noise. The positive-part variant never over-shrinks past the
    center.

    means: per-task mean estimates. sigma2: the sampling variance of each mean
    (a scalar common variance, e.g. score_var / n_runs). Returns shrunk means;
    for m < 3 the input is returned unchanged (JS gives no guarantee there).
    """
    x = np.asarray(means, dtype=float)
    m = x.size
    if m < 3:
        return x.copy()
    center = float(x.mean()) if grand is None else float(grand)
    ss = float(np.sum((x - center) ** 2))
    if ss <= 0 or sigma2 <= 0:
        return x.copy()
    shrink = 1.0 - (m - 2) * sigma2 / ss
    shrink = max(0.0, shrink)  # positive-part
    return center + shrink * (x - center)


def eb_credible_interval(successes, trials, ci: float = 0.95, prior: dict | None = None) -> dict:
    """Empirical-Bayes credible intervals for per-task pass rates.

    Unlike an independent Clopper-Pearson interval per task (which for 0/3 gives a
    useless [0, 0.71]), these use the fitted population prior: the per-task
    posterior is Beta(k+α, n-k+β) and the interval is its equal-tailed credible
    interval. They are correctly narrower at small n because they borrow strength
    across tasks.

    Returns {point (posterior mean array), lo (array), hi (array), alpha, beta}.
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if prior is None:
        prior = fit_beta_binomial_mom(k, n)
    a, b = prior["alpha"], prior["beta"]
    post_a = k + a
    post_b = n - k + b
    lo = stats.beta.ppf((1 - ci) / 2, post_a, post_b)
    hi = stats.beta.ppf(1 - (1 - ci) / 2, post_a, post_b)
    point = post_a / (post_a + post_b)
    return {"point": np.asarray(point, dtype=float),
            "lo": np.asarray(lo, dtype=float),
            "hi": np.asarray(hi, dtype=float),
            "alpha": a, "beta": b, "ci": ci}
