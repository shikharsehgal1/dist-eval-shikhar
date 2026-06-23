"""Distribution-aware aggregates that a single mean throws away.

Everything here consumes raw per-episode arrays (or a RecordStore) and returns
either a scalar summary or, where it matters, a richer object. The point is to
expose: robust center (IQM), tail risk (VaR/CVaR), and reliability (pass@k vs
pass^k) -- the three things mean-collapse hides.
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Robust center                                                               #
# --------------------------------------------------------------------------- #
def iqm(x: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of values.

    Robust like the median, statistically more efficient (rliable's primary
    aggregate). This is the headline number to prefer over a raw mean.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    lo, hi = np.percentile(x, [25, 75])
    mid = x[(x >= lo) & (x <= hi)]
    return float(mid.mean()) if mid.size else float(np.median(x))


# --------------------------------------------------------------------------- #
# Tail risk                                                                   #
# --------------------------------------------------------------------------- #
def var_at(x: np.ndarray, alpha: float = 0.1, tail: str = "lower") -> float:
    """Value-at-Risk: the alpha-quantile threshold of the (lower) tail."""
    x = np.asarray(x, dtype=float)
    q = alpha if tail == "lower" else 1 - alpha
    return float(np.quantile(x, q))


def cvar(x: np.ndarray, alpha: float = 0.1, tail: str = "lower") -> float:
    """Conditional Value-at-Risk: mean of the worst-alpha tail.

    For *returns* the risk is the lower tail (catastrophic episodes), so default
    tail='lower'. CVaR is coherent (subadditive); VaR is not. This is the number
    that surfaces "sometimes the agent catastrophically fails" -- invisible to
    the mean.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    v = var_at(x, alpha, tail)
    tail_vals = x[x <= v] if tail == "lower" else x[x >= v]
    return float(tail_vals.mean()) if tail_vals.size else v


# --------------------------------------------------------------------------- #
# Reliability: peak capability vs consistency                                 #
# --------------------------------------------------------------------------- #
def _per_task_counts(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Return [(n_trials, n_success), ...] per task."""
    out = []
    for _task, g in df.groupby("task"):
        out.append((len(g), int(g["success"].sum())))
    return out


def pass_at_k(df: pd.DataFrame, k: int) -> float:
    """Unbiased pass@k (Chen et al.): P(at least one of k trials succeeds), averaged
    over tasks. Measures *peak capability* -- "can it ever do this".
    """
    if "task" not in df.columns or "success" not in df.columns:
        raise ValueError("DataFrame must have 'task' and 'success' columns")
    vals = []
    for n, c in _per_task_counts(df):
        if n < k:
            vals.append(float(c > 0))  # not enough trials; fall back to empirical
        elif c < 0 or c > n or k < 0:
            vals.append(float("nan"))  # corrupted / invalid data
        else:
            vals.append(1.0 - comb(n - c, k) / comb(n, k))
    return float(np.mean(vals)) if vals else float("nan")


def pass_hat_k(df: pd.DataFrame, k: int) -> float:
    """Unbiased pass^k: P(ALL k trials succeed), averaged over tasks.

    Measures *reliability/consistency* (tau-bench). The gap between pass@k and
    pass^k is the reliability gap -- the single most important thing a mean hides.
    """
    if "task" not in df.columns or "success" not in df.columns:
        raise ValueError("DataFrame must have 'task' and 'success' columns")
    vals = []
    for n, c in _per_task_counts(df):
        if n < k:
            vals.append(float(c == n))
        elif c < 0 or c > n or k < 0:
            vals.append(float("nan"))  # corrupted / invalid data
        else:
            vals.append(comb(c, k) / comb(n, k))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# One-shot summary of an outcome distribution                                 #
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame, alpha: float = 0.1, ks: tuple[int, ...] = (1, 4, 8)) -> dict:
    s = df["score"].to_numpy(dtype=float)
    out = {
        "n_episodes": int(len(s)),
        "mean": float(s.mean()),
        "iqm": iqm(s),
        "median": float(np.median(s)),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        f"VaR@{alpha}": var_at(s, alpha),
        f"CVaR@{alpha}": cvar(s, alpha),
        "success_rate": float(df["success"].mean()),
    }
    for k in ks:
        out[f"pass@{k}"] = pass_at_k(df, k)
        out[f"pass^{k}"] = pass_hat_k(df, k)
    return out
