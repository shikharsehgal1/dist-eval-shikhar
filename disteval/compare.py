"""Distribution-to-distribution comparison of two agents.

A scalar delta of means cannot tell you whether B is *better*, *more reliable*,
or *dominates*. These compare whole outcome distributions.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """Earth-mover distance: aggregate displacement between two outcome distributions.

    Use when you care about overall mismatch (the native distributional-RL metric).
    """
    return float(stats.wasserstein_distance(np.asarray(a, float), np.asarray(b, float)))


def ks(a: np.ndarray, b: np.ndarray) -> dict:
    """Kolmogorov-Smirnov: sup-norm gap between CDFs + non-parametric equality test.

    Use when you care about the single largest local discrepancy ("one big shift").
    """
    res = stats.ks_2samp(np.asarray(a, float), np.asarray(b, float))
    return {"D": float(res.statistic), "p": float(res.pvalue)}


def prob_improvement(a: np.ndarray, b: np.ndarray) -> float:
    """P(A > B) for a random A-episode vs a random B-episode (ties = 0.5).

    Mann-Whitney-style. ~0.5 means indistinguishable; pair with a center metric
    because it ignores magnitude.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    # vectorized via rank statistic
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u / (len(a) * len(b)))


def stochastic_dominance(a: np.ndarray, b: np.ndarray, grid: int = 200) -> dict:
    """First- and second-order stochastic dominance of A over B (higher = better).

    FSD: CDF_A(x) <= CDF_B(x) for all x  -> A is preferred by *every* increasing
         utility (A is unambiguously better).
    SSD: integral of CDF_A <= integral of CDF_B everywhere -> A preferred by every
         increasing *concave* (risk-averse) utility. SSD aggregates CVaR criteria.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    xs = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), grid)
    Fa = np.array([(a <= x).mean() for x in xs])
    Fb = np.array([(b <= x).mean() for x in xs])
    fsd_a_over_b = bool(np.all(Fa <= Fb + 1e-9))
    fsd_b_over_a = bool(np.all(Fb <= Fa + 1e-9))
    # second order: integrate CDFs
    Ia = np.cumsum(Fa) * (xs[1] - xs[0])
    Ib = np.cumsum(Fb) * (xs[1] - xs[0])
    ssd_a_over_b = bool(np.all(Ia <= Ib + 1e-9))
    ssd_b_over_a = bool(np.all(Ib <= Ia + 1e-9))
    return {
        "FSD_A_dominates_B": fsd_a_over_b,
        "FSD_B_dominates_A": fsd_b_over_a,
        "SSD_A_dominates_B": ssd_a_over_b,
        "SSD_B_dominates_A": ssd_b_over_a,
    }
