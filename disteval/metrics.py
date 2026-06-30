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


__all__ = [
    "iqm",
    "var_at",
    "var",
    "cvar",
    "pass_at_k",
    "pass_hat_k",
    "reliability_decay",
    "variance_amplification_factor",
    "grpo_advantages",
    "summarize",
    "optimality_gap",
]


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


# Short alias matching the common RL/finance name and the README usage.
var = var_at


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


def reliability_decay(df: pd.DataFrame, ks: tuple[int, ...] = (1, 2, 4, 8)) -> dict:
    """Reliability decay curve: pass^k as a function of k, plus its slope.

    A single pass^8 number hides the *shape* of how reliability collapses with
    more required successes — pass^k can fall far faster than pass@1 (e.g. on
    tau-bench, 61% pass@1 -> 25% pass^8). The slope (per unit k, from a linear
    fit) summarizes how steeply consistency degrades: more negative = a sharper
    reliability cliff. Returns the curve, the slope, and the total drop.
    """
    ks = tuple(int(k) for k in ks)
    curve = [pass_hat_k(df, k) for k in ks]
    arr_k = np.asarray(ks, dtype=float)
    arr_v = np.asarray(curve, dtype=float)
    valid = np.isfinite(arr_v)
    if valid.sum() >= 2 and np.ptp(arr_k[valid]) > 0:
        slope = float(np.polyfit(arr_k[valid], arr_v[valid], 1)[0])
    else:
        slope = float("nan")
    finite = arr_v[valid]
    drop = float(finite[0] - finite[-1]) if finite.size >= 2 else float("nan")
    return {"ks": list(ks), "pass_hat_k": curve, "slope": slope, "total_drop": drop}


def variance_amplification_factor(short_scores: np.ndarray, long_scores: np.ndarray) -> float:
    """Var(long-horizon scores) / Var(short-horizon scores).

    Long-horizon agentic tasks amplify run-to-run variance: small per-step
    nondeterminism compounds over many steps. A VAF >> 1 means the agent is far
    less repeatable on long tasks than short ones — a reliability signature the
    mean cannot show. Returns nan if either sample has < 2 points or the
    short-horizon variance is zero.
    """
    s = np.asarray(short_scores, dtype=float)
    ell = np.asarray(long_scores, dtype=float)
    if s.size < 2 or ell.size < 2:
        return float("nan")
    var_short = float(s.var(ddof=1))
    var_long = float(ell.var(ddof=1))
    if var_short == 0:
        return float("nan")
    return var_long / var_short


def grpo_advantages(scores: np.ndarray, groups: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    """Group-relative advantages: (score - group_mean) / (group_std + eps).

    This is the critic-free, variance-normalized signal at the heart of GRPO-style
    RL — and disteval's multi-run-per-task structure is exactly the "group of
    samples per prompt" GRPO assumes. Positive advantage = above-average run for
    its task (reinforce); negative = below average (contrast). With `groups`
    (e.g. per-task labels) each task is normalized independently; without, a
    single global group is used. Returns an array aligned to the input order.
    """
    s = np.asarray(scores, dtype=float)
    adv = np.zeros_like(s)
    if s.size == 0:
        return adv
    if groups is None:
        groups = np.zeros(s.size, dtype=int)
    else:
        groups = np.asarray(groups)
    for g in np.unique(groups):
        idx = groups == g
        vals = s[idx]
        adv[idx] = (vals - vals.mean()) / (vals.std() + eps)
    return adv


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


def optimality_gap(scores: np.ndarray, optimal: float = 1.0) -> float:
    """Normalized distance from the best possible score.

    The optimality gap is a standard RL aggregate: (optimal - mean(scores)) / optimal.
    A value of 0 means the agent matches the optimal score; 1 means it scores zero.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return float("nan")
    if optimal == 0:
        raise ValueError("optimal score must be non-zero")
    return float((optimal - scores.mean()) / optimal)
