"""Task-level latent-performance posteriors.

Motivation
----------
``disteval.right_tail`` historically used ``Q*(t) = max_i q_i`` as the estimate of
"demonstrated capability". That is a biased estimator of anything latent: it is
monotonically non-decreasing in the number of runs, so an agent evaluated 20
times looks strictly more capable than the same agent evaluated 3 times, and a
single lucky success is enough to declare a task mastered. See
:mod:`disteval.reliability.classify` for the replacement and ``THEORY.md`` for
the argument.

This module provides the primitive that replaces it: a posterior over the latent
per-task performance parameter.

Binary outcomes
---------------
The standard conjugate model::

    p_t ~ Beta(alpha, beta)
    s_t | p_t ~ Binomial(n_t, p_t)
    p_t | s_t ~ Beta(alpha + s_t, beta + n_t - s_t)

``p_t`` is the latent probability that the agent completes task ``t``. All
downstream quantities (capability, reliability, recoverability, classification)
are functionals of this posterior, so they inherit its uncertainty for free.

Continuous rubric scores in [0, 1]
----------------------------------
A Bernoulli likelihood does not apply to fractional rubric scores, and pretending
it does (by rounding, or by feeding fractional "successes" into a Binomial)
either throws away information or produces an invalid likelihood. We instead
target the latent *mean score* ``mu_t = E[q | t]`` with a **quasi-binomial**
(dispersion-corrected) Beta posterior:

1. Compute the sample mean ``m`` and unbiased sample variance ``s2``.
2. The Bernoulli distribution has the maximum variance of any distribution on
   [0, 1] with mean ``m``, namely ``m(1-m)``. So the dispersion ratio
   ``phi = s2 / (m(1-m))`` lies in ``[0, 1]`` up to sampling noise.
3. Each run therefore carries *at least* as much information about ``mu_t`` as a
   Bernoulli draw would. Define the effective sample size ``n_eff = n / phi`` and
   split it into pseudo-counts ``n_eff*m`` and ``n_eff*(1-m)``.
4. The posterior is ``Beta(alpha + n_eff*m, beta + n_eff*(1-m))``.

Properties that make this defensible rather than decorative:

* **Exact reduction.** For binary data ``phi -> 1`` and the estimator collapses
  to the exact Beta-Binomial posterior above (checked in the tests).
* **Correct direction of the correction.** Deterministic partial credit
  (e.g. every run scores exactly 0.5) has ``phi -> 0``: the latent mean really is
  pinned down, and the posterior is correspondingly tight. Bimodal 0/1 behaviour
  has ``phi -> 1``: maximally uncertain, as it should be.
* **Bounded influence.** ``phi`` is clipped to ``[phi_floor, 1]`` so a lucky
  low-variance sample cannot manufacture unbounded confidence.

It is a quasi-likelihood, not a generative model: interpret the interval as a
calibrated-by-construction summary of the mean score, not as an exact Bayesian
posterior under a named data-generating process. When you need a fully
generative treatment of continuous scores, dichotomise at a documented rubric
threshold and use the binary path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "BetaPrior",
    "JEFFREYS_PRIOR",
    "UNIFORM_PRIOR",
    "TaskPosterior",
    "binary_posterior",
    "continuous_posterior",
    "posterior_from_scores",
    "dispersion_ratio",
]


@dataclass(frozen=True)
class BetaPrior:
    """A Beta(alpha, beta) prior over a latent [0, 1] performance parameter."""

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta prior parameters must be strictly positive")

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        """Prior sample size (alpha + beta): how many pseudo-runs it is worth."""
        return self.alpha + self.beta

    @classmethod
    def from_mean_strength(cls, mean: float, strength: float) -> "BetaPrior":
        """Build a prior from an interpretable (centre, pseudo-run count) pair."""
        if not 0.0 < mean < 1.0:
            raise ValueError("prior mean must be strictly inside (0, 1)")
        if strength <= 0:
            raise ValueError("prior strength must be positive")
        return cls(alpha=mean * strength, beta=(1.0 - mean) * strength)


#: Uniform prior. Maximum-entropy, but worth 2 pseudo-runs, which is a lot when
#: n_t is 3. Good default only when you have no cross-task population to pool.
UNIFORM_PRIOR = BetaPrior(1.0, 1.0)

#: Jeffreys prior for a binomial proportion. Weaker (worth 1 pseudo-run) and
#: invariant to reparameterisation; the recommended fixed fallback prior.
JEFFREYS_PRIOR = BetaPrior(0.5, 0.5)


def dispersion_ratio(scores: Sequence[float], phi_floor: float = 1e-3) -> float:
    """Sample variance divided by the maximum ([0,1]-Bernoulli) variance.

    Returns a value in ``[phi_floor, 1]``. Exactly 1.0 (before clipping) for
    binary data, small for near-deterministic partial credit. Returns 1.0 when
    the mean is degenerate (0 or 1) or fewer than two observations are available,
    which is the conservative choice.
    """
    x = np.asarray(scores, dtype=float)
    if x.size < 2:
        return 1.0
    m = float(x.mean())
    if m <= 0.0 or m >= 1.0:
        return 1.0
    s2 = float(x.var(ddof=1))
    phi = s2 / (m * (1.0 - m))
    return float(np.clip(phi, phi_floor, 1.0))


@dataclass(frozen=True)
class TaskPosterior:
    """Beta posterior over a task's latent performance parameter.

    Attributes
    ----------
    alpha, beta:
        Posterior Beta parameters.
    n_obs:
        Number of actual runs observed (not pseudo-counts).
    n_eff:
        Effective sample size contributed by the likelihood. Equals ``n_obs`` for
        binary data; larger for under-dispersed continuous scores.
    binary:
        Whether the likelihood was the exact Binomial one.
    prior:
        The prior that was applied.
    """

    alpha: float
    beta: float
    n_obs: int
    n_eff: float
    binary: bool
    prior: BetaPrior

    # -- point estimates ----------------------------------------------------
    @property
    def mean(self) -> float:
        """Posterior mean. The default point estimate of latent performance."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def median(self) -> float:
        """Posterior median; more robust than the mean for very skewed posteriors."""
        return float(stats.beta.median(self.alpha, self.beta))

    @property
    def mode(self) -> Optional[float]:
        """Posterior mode (MAP), or ``None`` when the posterior has no interior mode."""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        return None

    @property
    def sd(self) -> float:
        """Posterior standard deviation: the headline uncertainty summary."""
        a, b = self.alpha, self.beta
        return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0))))

    @property
    def entropy(self) -> float:
        """Differential entropy of the posterior, in nats. Used by active sampling."""
        return float(stats.beta.entropy(self.alpha, self.beta))

    # -- interval and probability queries -----------------------------------
    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval at ``level``."""
        if not 0.0 < level < 1.0:
            raise ValueError("credible level must be in (0, 1)")
        tail = (1.0 - level) / 2.0
        return (
            float(stats.beta.ppf(tail, self.alpha, self.beta)),
            float(stats.beta.ppf(1.0 - tail, self.alpha, self.beta)),
        )

    def prob_above(self, threshold: float) -> float:
        """``P(p_t > threshold | D_t)``, the primitive behind capability/reliability."""
        if threshold <= 0.0:
            return 1.0
        if threshold >= 1.0:
            return 0.0
        return float(stats.beta.sf(threshold, self.alpha, self.beta))

    def prob_below(self, threshold: float) -> float:
        """``P(p_t < threshold | D_t)``."""
        return 1.0 - self.prob_above(threshold)

    def quantile(self, q: float) -> float:
        """Posterior quantile; ``quantile(0.05)`` is a pessimistic performance bound."""
        return float(stats.beta.ppf(q, self.alpha, self.beta))

    def sample(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``size`` posterior samples of the latent parameter."""
        rng = rng or np.random.default_rng()
        return rng.beta(self.alpha, self.beta, size=size)

    def posterior_predictive(self, n: int) -> np.ndarray:
        """Beta-Binomial PMF over the number of successes in ``n`` future runs.

        Returns an array of length ``n + 1``. This is the quantity to use when
        asking "if I deploy this agent ``n`` more times, how many successes should
        I expect?" -- it integrates over parameter uncertainty rather than
        plugging in a point estimate.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        k = np.arange(n + 1)
        from scipy.special import betaln, gammaln

        log_choose = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        log_pmf = (
            log_choose
            + betaln(k + self.alpha, n - k + self.beta)
            - betaln(self.alpha, self.beta)
        )
        return np.exp(log_pmf)

    def to_dict(self) -> dict:
        lo, hi = self.credible_interval()
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "n_obs": self.n_obs,
            "n_eff": self.n_eff,
            "binary": self.binary,
            "posterior_mean": self.mean,
            "posterior_median": self.median,
            "posterior_sd": self.sd,
            "ci95_lo": lo,
            "ci95_hi": hi,
        }


def binary_posterior(
    successes: int,
    trials: int,
    prior: BetaPrior = JEFFREYS_PRIOR,
) -> TaskPosterior:
    """Exact Beta-Binomial posterior for ``successes`` out of ``trials``."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid counts: {successes} successes of {trials} trials")
    return TaskPosterior(
        alpha=prior.alpha + successes,
        beta=prior.beta + (trials - successes),
        n_obs=int(trials),
        n_eff=float(trials),
        binary=True,
        prior=prior,
    )


def continuous_posterior(
    scores: Sequence[float],
    prior: BetaPrior = JEFFREYS_PRIOR,
    phi_floor: float = 1e-3,
) -> TaskPosterior:
    """Dispersion-corrected Beta posterior over the latent mean score.

    See the module docstring for the derivation and its assumptions. ``scores``
    must lie in [0, 1].
    """
    x = np.asarray(list(scores), dtype=float)
    if x.size == 0:
        return TaskPosterior(prior.alpha, prior.beta, 0, 0.0, False, prior)
    if np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError("continuous scores must lie in [0, 1]")
    m = float(x.mean())
    phi = dispersion_ratio(x, phi_floor=phi_floor)
    n_eff = float(x.size) / phi
    return TaskPosterior(
        alpha=prior.alpha + n_eff * m,
        beta=prior.beta + n_eff * (1.0 - m),
        n_obs=int(x.size),
        n_eff=n_eff,
        binary=False,
        prior=prior,
    )


def posterior_from_scores(
    scores: Iterable[float],
    prior: BetaPrior = JEFFREYS_PRIOR,
    *,
    success_threshold: Optional[float] = None,
    phi_floor: float = 1e-3,
) -> TaskPosterior:
    """Dispatch to the exact binary model when the data allow it.

    * ``success_threshold`` set -> dichotomise at that threshold and use the exact
      Beta-Binomial model (recommended when a rubric defines task success).
    * otherwise, if every score is 0 or 1 -> exact Beta-Binomial.
    * otherwise -> the dispersion-corrected continuous estimator.
    """
    x = np.asarray(list(scores), dtype=float)
    if x.size == 0:
        return TaskPosterior(prior.alpha, prior.beta, 0, 0.0, True, prior)
    if success_threshold is not None:
        succ = int((x >= success_threshold).sum())
        return binary_posterior(succ, int(x.size), prior)
    if np.all(np.isin(x, (0.0, 1.0))):
        return binary_posterior(int(x.sum()), int(x.size), prior)
    return continuous_posterior(x, prior, phi_floor=phi_floor)
