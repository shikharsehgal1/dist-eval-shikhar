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
