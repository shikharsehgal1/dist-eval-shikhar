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
