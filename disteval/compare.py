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
    "bradley_terry",
    "win_matrix_from_pairs",
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


def win_matrix_from_pairs(pairs, items: list | None = None) -> tuple[np.ndarray, list]:
    """Build a K×K win matrix from pairwise outcomes for Bradley-Terry ranking.

    ``pairs`` is an iterable of ``(a, b, s)`` where ``s`` is a's result against
    b: 1.0 = a won, 0.0 = b won, 0.5 = tie. ``items`` fixes the row/column order;
    if omitted it is inferred (sorted) from the pairs. Returns ``(win_matrix,
    items)`` where ``win_matrix[i][j]`` is the (possibly fractional, for ties)
    number of wins of item i over item j.
    """
    pairs = list(pairs)
    if items is None:
        seen = set()
        for a, b, _ in pairs:
            seen.add(a)
            seen.add(b)
        items = sorted(seen)
    idx = {name: i for i, name in enumerate(items)}
    k = len(items)
    w = np.zeros((k, k), dtype=float)
    for a, b, s in pairs:
        s = float(s)
        w[idx[a], idx[b]] += s
        w[idx[b], idx[a]] += 1.0 - s
    return w, list(items)


def bradley_terry(
    win_matrix: np.ndarray,
    ci: float | None = None,
    reg: float = 1e-3,
    n_boot: int = 1000,
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> dict:
    """Bradley-Terry MLE ranking of N systems from a pairwise win matrix.

    The pairwise ``prob_improvement`` only ranks two systems; Bradley-Terry
    jointly fits a strength per system so a full leaderboard is consistent (the
    de-facto standard since Chatbot Arena moved from Elo to a BT MLE). Fit by the
    MM algorithm (Hunter 2004). A small symmetric ``reg`` pseudo-count keeps
    strengths finite when a system wins or loses all its games.

    ``win_matrix[i][j]`` = wins of i over j (fractional allowed for ties). With
    ``ci`` set (e.g. 0.95) a parametric bootstrap resamples each pair's games
    from the fitted probabilities and returns per-system CIs on the log-strength
    plus a ``ranking_unstable`` flag when any CIs overlap the top system's.

    Returns {strengths (log scale, mean-centered), probs (sum to 1), ranking
    (indices best-first), n_items, iterations, and — if ci — strength_lo/hi,
    ranking_unstable, and rank_flip_prob (bootstrap P(runner-up >= top))}.
    """
    W = np.asarray(win_matrix, dtype=float).copy()
    k = W.shape[0]
    if W.ndim != 2 or W.shape[1] != k:
        raise ValueError("win_matrix must be square (K x K)")
    if k < 2:
        raise ValueError("need at least 2 systems to rank")
    np.fill_diagonal(W, 0.0)
    # Symmetric regularization: a tiny shared prior game between every pair.
    if reg > 0:
        W = W + reg * (1.0 - np.eye(k))

    games = W + W.T           # n_ij, symmetric
    wins = W.sum(axis=1)      # W_i, total wins per system

    def _fit(wins_vec, games_mat):
        p = np.ones(k)
        for it in range(max_iter):
            p_old = p.copy()
            for i in range(k):
                denom = 0.0
                for j in range(k):
                    if j == i:
                        continue
                    denom += games_mat[i, j] / (p[i] + p[j])
                p[i] = wins_vec[i] / denom if denom > 0 else p[i]
            p /= p.sum()
            if np.max(np.abs(p - p_old)) < tol:
                break
        return p, it + 1

    p, iters = _fit(wins, games)
    strengths = np.log(p)
    strengths -= strengths.mean()  # center for identifiability
    ranking = np.argsort(strengths)[::-1]

    out = {
        "strengths": strengths,
        "probs": p,
        "ranking": ranking,
        "n_items": k,
        "iterations": iters,
    }

    if ci is not None:
        rng = np.random.default_rng(seed)
        # Fitted win probability for each ordered pair.
        P = p[:, None] / (p[:, None] + p[None, :])
        boot = np.empty((n_boot, k))
        n_int = np.rint(games).astype(int)
        for b in range(n_boot):
            Wb = np.zeros((k, k))
            for i in range(k):
                for j in range(i + 1, k):
                    n_ij = n_int[i, j]
                    if n_ij <= 0:
                        continue
                    wins_i = rng.binomial(n_ij, P[i, j])
                    Wb[i, j] = wins_i
                    Wb[j, i] = n_ij - wins_i
            if reg > 0:
                Wb = Wb + reg * (1.0 - np.eye(k))
            pb, _ = _fit(Wb.sum(axis=1), Wb + Wb.T)
            sb = np.log(pb)
            boot[b] = sb - sb.mean()
        lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2], axis=0)
        out["strength_lo"] = lo
        out["strength_hi"] = hi
        out["ci"] = ci
        best = ranking[0]
        others = np.arange(k) != best
        # Unstable if ANY other system's CI reaches the top system's lower CI.
        out["ranking_unstable"] = bool(np.any(hi[others] >= lo[best]))
        # Bootstrap probability the top system is caught/overtaken by the runner-up.
        second = ranking[1]
        out["rank_flip_prob"] = float(np.mean(boot[:, second] >= boot[:, best]))

    return out


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
