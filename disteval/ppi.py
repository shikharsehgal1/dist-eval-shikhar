"""Prediction-powered inference (PPI++) for LLM-judge debiasing.

disteval takes the score column as ground truth. In agentic / open-ended evals
that "score" is often a *biased* LLM judge (position, verbosity and
self-preference bias are all well documented). Two bad options remain: trust the
judge (biased) or hand-label everything (expensive, tiny sample, wide CIs).

PPI++ is the third option: label a *small* gold/human calibration set, run the
cheap judge on everything, and combine them into an estimate that is unbiased
for the gold quantity *and* has a tighter CI than using the gold labels alone —
as long as the judge is at all correlated with the truth. The estimator is

    theta_hat(lambda) = lambda * mean(judge_unlabeled)
                        + mean(gold) - lambda * mean(judge_gold)

with lambda tuned to minimize variance (Angelopoulos et al. 2023, "PPI++").
lambda = 1 is classic PPI (rectified judge mean); lambda = 0 collapses to the
gold-only mean, so PPI++ never does worse than ignoring the judge.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


__all__ = ["ppi_mean", "optimal_lambda"]


def optimal_lambda(gold: np.ndarray, judge_gold: np.ndarray, judge_unlabeled: np.ndarray) -> float:
    """Variance-minimizing power-tuning weight lambda for the PPI++ mean.

    lambda* = Cov(gold, judge_gold) / (Var(judge_gold) + (n/N) * Var(judge_unlabeled)),
    clipped to [0, 1] (the range over which PPI++ is guaranteed no worse than the
    gold-only estimator). Returns 0.0 when the judge carries no usable variance.
    """
    gold = np.asarray(gold, dtype=float)
    judge_gold = np.asarray(judge_gold, dtype=float)
    judge_unlabeled = np.asarray(judge_unlabeled, dtype=float)
    n, N = gold.size, judge_unlabeled.size
    if n < 2 or N < 1:
        return 0.0
    cov = float(np.cov(gold, judge_gold, ddof=1)[0, 1])
    var_gl = float(np.var(judge_gold, ddof=1))
    var_un = float(np.var(judge_unlabeled, ddof=1)) if N > 1 else 0.0
    denom = var_gl + (n / N) * var_un
    if denom <= 0:
        return 0.0
    return float(np.clip(cov / denom, 0.0, 1.0))


def ppi_mean(
    gold: np.ndarray,
    judge_gold: np.ndarray,
    judge_unlabeled: np.ndarray,
    ci: float = 0.95,
    lam: float | None = None,
) -> dict:
    """Prediction-powered estimate of the mean gold score, with a CI.

    gold: gold/human labels on the calibration set (length n).
    judge_gold: judge scores on the SAME calibration items (length n).
    judge_unlabeled: judge scores on the remaining unlabeled items (length N).
    lam: power-tuning weight; if None, the variance-optimal value is used.

    Returns {point, lo, hi, se, lambda, ci, n_labeled, n_unlabeled,
    gold_only_mean, gold_only_se, judge_mean}. Compare ``se`` to ``gold_only_se``
    to see how much the judge tightened the estimate.
    """
    gold = np.asarray(gold, dtype=float)
    judge_gold = np.asarray(judge_gold, dtype=float)
    judge_unlabeled = np.asarray(judge_unlabeled, dtype=float)
    n, N = gold.size, judge_unlabeled.size
    if gold.shape != judge_gold.shape:
        raise ValueError("gold and judge_gold must have the same length")
    if n == 0:
        raise ValueError("need at least one gold label")

    gold_mean = float(gold.mean())
    gold_only_se = float(gold.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    # With no unlabeled data or a single gold point, fall back to the gold mean.
    if N == 0 or n < 2:
        z = stats.norm.ppf(1 - (1 - ci) / 2)
        se = gold_only_se
        half = z * se if np.isfinite(se) else float("nan")
        return {
            "point": gold_mean, "lo": gold_mean - half if np.isfinite(half) else gold_mean,
            "hi": gold_mean + half if np.isfinite(half) else gold_mean,
            "se": se, "lambda": 0.0, "ci": ci, "n_labeled": n, "n_unlabeled": N,
            "gold_only_mean": gold_mean, "gold_only_se": gold_only_se,
            "judge_mean": float(judge_unlabeled.mean()) if N else float("nan"),
        }

    if lam is None:
        lam = optimal_lambda(gold, judge_gold, judge_unlabeled)

    point = lam * float(judge_unlabeled.mean()) + gold_mean - lam * float(judge_gold.mean())
    # Independent labeled and unlabeled samples -> variances add.
    var_rectifier = float(np.var(gold - lam * judge_gold, ddof=1)) / n
    var_unlabeled = (lam**2) * float(np.var(judge_unlabeled, ddof=1)) / N if N > 1 else 0.0
    se = float(np.sqrt(var_rectifier + var_unlabeled))
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    return {
        "point": point,
        "lo": point - z * se,
        "hi": point + z * se,
        "se": se,
        "lambda": float(lam),
        "ci": ci,
        "n_labeled": n,
        "n_unlabeled": N,
        "gold_only_mean": gold_mean,
        "gold_only_se": gold_only_se,
        "judge_mean": float(judge_unlabeled.mean()),
    }
