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
    "adjust_pvalues",
    "min_detectable_effect",
    "required_n",
    "score_length_bias",
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
    if a.size == 0 or b.size == 0:
        return float("nan")
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
    if not np.isfinite(pooled_std) or pooled_std == 0:
        # Undefined for n=1 (std is nan) or zero-variance inputs; report no effect.
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


def adjust_pvalues(pvals, method: str = "benjamini-hochberg") -> np.ndarray:
    """Correct a family of p-values for multiple comparisons.

    Comparing one baseline against many candidate agents (or running a test per
    rubric criterion) inflates false positives — with 20 independent tests at
    alpha=0.05 you expect ~1 spurious "winner". Returns adjusted p-values in the
    *same order* as the input; compare these against your original alpha.

    method:
      "benjamini-hochberg" / "bh" — controls the false-discovery rate (more
          powerful; the right default for leaderboard / many-model screening).
      "holm" — controls the family-wise error rate (more conservative).
    """
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return p.copy()
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    method = method.lower()
    if method in ("benjamini-hochberg", "bh", "fdr"):
        # adj_(i) = min over j>=i of (n/j) * p_(j), then clipped to <= 1.
        factors = n / np.arange(1, n + 1)
        adj_sorted = np.minimum.accumulate((factors * ranked)[::-1])[::-1]
    elif method == "holm":
        # adj_(i) = max over j<=i of (n-j+1) * p_(j), then clipped to <= 1.
        factors = n - np.arange(n)
        adj_sorted = np.maximum.accumulate(factors * ranked)
    else:
        raise ValueError(f"Unknown multiple-comparisons method: {method!r}")
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj_sorted
    return out


def min_detectable_effect(n_a: int, n_b: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """Smallest standardized effect (Cohen's d) a two-sample test could detect.

    Most agent comparisons run only a handful of seeds and are badly underpowered
    — so a "no significant difference" result is uninformative. This reports the
    floor: any true effect smaller than the returned d would be missed more often
    than not at the requested power. Multiply by your pooled std to get the
    minimum detectable raw mean gap.
    """
    if n_a < 1 or n_b < 1:
        return float("nan")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    return float((z_alpha + z_power) * np.sqrt(1.0 / n_a + 1.0 / n_b))


def required_n(effect: float, power: float = 0.8, alpha: float = 0.05, ratio: float = 1.0) -> int:
    """Per-group sample size needed to detect a standardized effect (Cohen's d).

    `ratio` = n_b / n_a (default 1 for equal groups). Returns n_a (round up);
    n_b = ceil(ratio * n_a). The inverse of `min_detectable_effect`.
    """
    if effect == 0 or not np.isfinite(effect):
        return 2**31 - 1  # an infinite effect-detection requirement, capped
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    n_a = (z_alpha + z_power) ** 2 * (1.0 + 1.0 / ratio) / effect**2
    return int(np.ceil(n_a))


def score_length_bias(scores: np.ndarray, lengths: np.ndarray, threshold: float = 0.3) -> dict:
    """Detect length / verbosity bias: does score correlate with trajectory length?

    Length bias is the best-documented reward-hacking mode — longer outputs get
    higher scores regardless of quality, and the bias compounds in iterative
    self-improvement loops. Returns the Spearman rank correlation, its p-value,
    and a `flagged` boolean when the correlation is both significant and at least
    `threshold` in magnitude (a signal to length-normalize before forming pairs).
    """
    s = np.asarray(scores, float)
    ell = np.asarray(lengths, float)
    mask = np.isfinite(s) & np.isfinite(ell)
    s, ell = s[mask], ell[mask]
    if s.size < 3 or np.all(ell == ell[0]) or np.all(s == s[0]):
        return {"rho": float("nan"), "p": float("nan"), "n": int(s.size), "flagged": False}
    res = stats.spearmanr(s, ell)
    rho, p = float(res.statistic), float(res.pvalue)
    flagged = bool(np.isfinite(p) and p < 0.05 and abs(rho) >= threshold)
    return {"rho": rho, "p": p, "n": int(s.size), "flagged": flagged}


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
