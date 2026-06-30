"""Stratified bootstrap CIs and performance profiles.

This is the layer that, in production, you'd delegate to rliable
(google-research/rliable) -- see adapters/rliable_bridge.py for the matrix the
library expects. We reimplement the core here in numpy so the prototype runs
with zero heavy deps, and so the *stratified* resampling (resample within each
stratum, the missing primitive) is explicit.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    "stratified_bootstrap_ci",
    "performance_profile",
    "analytical_ci",
    "binomial_ci",
    "confidence_sequence",
]


def stratified_bootstrap_ci(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    strata_cols: list[str] | None = None,
    n_reps: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap a statistic of an outcome distribution, resampling *within* strata.

    Stratified resampling keeps every stratum (task / difficulty / domain)
    represented in every bootstrap replicate -- the right way to propagate
    finite-sample uncertainty when tasks are heterogeneous.

    NOTE: this resamples the *already-collected episodes*. It cannot resample the
    data-generating process (fresh task draws, new env seeds, LLM nondeterminism).
    That gap is exactly what repeat.py measures -- and why bootstrap CIs are
    typically narrower than the true run-to-run spread.
    """
    rng = np.random.default_rng(seed)
    strata_cols = strata_cols or []
    if strata_cols:
        groups = [g.index.to_numpy() for _, g in df.groupby(strata_cols)]
    else:
        groups = [df.index.to_numpy()]

    point = stat_fn(df)
    boot = np.empty(n_reps)
    for b in range(n_reps):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        boot[b] = stat_fn(df.loc[idx])
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "width": float(hi - lo), "ci": ci, "n_reps": n_reps}


def performance_profile(scores: np.ndarray, taus: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Empirical run-score distribution: fraction of episodes scoring >= tau.

    Returns (taus, fractions). Plotting fractions vs taus gives rliable's
    performance profile -- the whole distribution, tails included, in one curve.
    """
    scores = np.asarray(scores, dtype=float)
    if taus is None:
        taus = np.linspace(scores.min(), scores.max(), 50)
    frac = np.array([(scores >= t).mean() for t in taus])
    return taus, frac


def analytical_ci(x: np.ndarray, ci: float = 0.95) -> dict:
    """Normal-approximation confidence interval for the mean.

    Cheaper than bootstrap for large samples; useful when you just need a quick
    CI for the mean of a score array. Returns {point, lo, hi, width, ci}.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "width": float("nan"), "ci": ci}
    n = len(x)
    point = float(x.mean())
    if n < 2:
        return {"point": point, "lo": point, "hi": point, "width": 0.0, "ci": ci}
    se = float(x.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    lo, hi = point - z * se, point + z * se
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "width": float(hi - lo), "ci": ci}


def confidence_sequence(x: np.ndarray, ci: float = 0.95, lo: float = 0.0, hi: float = 1.0) -> dict:
    """Anytime-valid confidence sequence for the mean of bounded scores.

    A fixed-n CI (analytical_ci/binomial_ci) is only valid if you fix the sample
    size in advance; if you keep adding eval runs and stop as soon as the CI
    excludes a threshold, you p-hack and the stated coverage is a lie. A
    confidence sequence is valid *simultaneously at every sample size*, so you may
    peek after each run and stop early without inflating error.

    This is a Hoeffding confidence sequence built by a union bound over sample
    sizes with weights alpha_t = alpha * 6 / (pi^2 t^2) (which sum to alpha). It
    is conservative but provably valid for i.i.d. observations in [lo, hi]; for
    tighter (empirical-Bernstein) sequences see Howard et al. 2021.

    Returns the running interval at the final sample size plus the full per-step
    bounds, so callers can find the first t where the interval clears a bound.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "width": float("nan"), "ci": ci, "n": 0,
                "running_mean": [], "running_lo": [], "running_hi": []}
    rng = hi - lo
    if rng <= 0:
        raise ValueError("hi must be greater than lo")
    t = np.arange(1, n + 1)
    alpha = 1 - ci
    # Hoeffding: P(|mean_t - mu| >= eps) <= 2 exp(-2 t eps^2 / range^2). Spend
    # alpha_t = alpha * 6/(pi^2 t^2) at each t; union bound over t keeps the whole
    # sequence valid at level (1 - alpha).
    alpha_t = alpha * 6.0 / (np.pi**2 * t**2)
    radius = rng * np.sqrt(np.log(2.0 / alpha_t) / (2.0 * t))
    running_mean = np.cumsum(x) / t
    running_lo = np.clip(running_mean - radius, lo, hi)
    running_hi = np.clip(running_mean + radius, lo, hi)
    return {
        "point": float(running_mean[-1]),
        "lo": float(running_lo[-1]),
        "hi": float(running_hi[-1]),
        "width": float(running_hi[-1] - running_lo[-1]),
        "ci": ci,
        "n": int(n),
        "running_mean": running_mean.tolist(),
        "running_lo": running_lo.tolist(),
        "running_hi": running_hi.tolist(),
    }


def binomial_ci(k: int, n: int, ci: float = 0.95, method: str = "clopper-pearson") -> dict:
    """Exact binomial confidence interval for pass@k / pass^k estimates.

    Computes the Clopper-Pearson interval (default) or Wilson interval. Returns
    {point, lo, hi, width, ci, n}. The point estimate is k / n.
    """
    if n <= 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "width": float("nan"), "ci": ci, "n": n}
    point = k / n
    alpha = 1 - ci
    if method == "clopper-pearson":
        lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
        hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    elif method == "wilson":
        z = stats.norm.ppf(1 - alpha / 2)
        p = point
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        margin = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
        lo = max(0.0, center - margin)
        hi = min(1.0, center + margin)
    else:
        raise ValueError(f"Unknown binomial CI method: {method}")
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "width": float(hi - lo), "ci": ci, "n": n}
