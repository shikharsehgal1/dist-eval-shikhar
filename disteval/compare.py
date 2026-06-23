"""Distribution-to-distribution comparison of two agents.

A scalar delta of means cannot tell you whether B is *better*, *more reliable*,
or *dominates*. These compare whole outcome distributions.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


__all__ = [
    "wasserstein",
    "ks",
    "prob_improvement",
    "stochastic_dominance",
    "effect_size",
    "mann_whitney_u",
    "compare_distributions",
]


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


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: standardized mean difference between two outcome distributions.

    A rule-of-thumb scale: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium,
    > 0.8 large. Useful alongside prob_improvement to quantify magnitude.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> dict:
    """Mann-Whitney U test: non-parametric test for stochastic ordering.

    Returns the U statistic and two-sided p-value. A small p-value means the
    distributions are unlikely to be identical. Unlike `prob_improvement`, this
    provides a significance level for P(A > B).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(res.statistic), "p": float(res.pvalue),
            "prob_A_greater_B": prob_improvement(a, b)}


def compare_distributions(a: np.ndarray, b: np.ndarray) -> dict:
    """All-in-one comparison of two outcome distributions.

    Returns a single dict with Wasserstein, KS, prob_improvement, stochastic
    dominance, effect size, and Mann-Whitney U. Useful for agent leaderboards.
    """
    return {
        "wasserstein": wasserstein(a, b),
        "ks": ks(a, b),
        "prob_improvement": prob_improvement(a, b),
        "stochastic_dominance": stochastic_dominance(a, b),
        "effect_size": effect_size(a, b),
        "mann_whitney_u": mann_whitney_u(a, b),
    }
