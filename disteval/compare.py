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

    Pairwise comparison. ~0.5 means indistinguishable; pair with a center metric
    because it ignores magnitude.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    # vectorized pairwise comparison
    greater = np.sum(a[:, None] > b[None, :])
    ties = np.sum(a[:, None] == b[None, :])
    return float((greater + 0.5 * ties) / (len(a) * len(b)))


def stochastic_dominance(a: np.ndarray, b: np.ndarray, grid: int = 200, tol: float = 1e-9) -> dict:
    """First- and second-order stochastic dominance of A over B (higher = better).

    FSD: CDF_A(x) <= CDF_B(x) for all x  -> A is preferred by *every* increasing
         utility (A is unambiguously better).
    SSD: integral of CDF_A <= integral of CDF_B everywhere -> A preferred by every
         increasing *concave* (risk-averse) utility. SSD aggregates CVaR criteria.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi:
        # Constant/degenerate distributions: both dominate each other trivially.
        return {
            "FSD_A_dominates_B": True,
            "FSD_B_dominates_A": True,
            "SSD_A_dominates_B": True,
            "SSD_B_dominates_A": True,
        }
    xs = np.linspace(lo, hi, grid)
    Fa = np.array([(a <= x).mean() for x in xs])
    Fb = np.array([(b <= x).mean() for x in xs])
    fsd_a_over_b = bool(np.all(Fa <= Fb + tol))
    fsd_b_over_a = bool(np.all(Fb <= Fa + tol))
    # second order: integrate CDFs
    dx = xs[1] - xs[0]
    Ia = np.cumsum(Fa) * dx
    Ib = np.cumsum(Fb) * dx
    ssd_a_over_b = bool(np.all(Ia <= Ib + tol))
    ssd_b_over_a = bool(np.all(Ib <= Ia + tol))
    return {
        "FSD_A_dominates_B": fsd_a_over_b,
        "FSD_B_dominates_A": fsd_b_over_a,
        "SSD_A_dominates_B": ssd_a_over_b,
        "SSD_B_dominates_A": ssd_b_over_a,
    }
