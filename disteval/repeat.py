"""Repeated-evaluation meta-distribution  --  the "run the whole eval over and over" idea.

WHY THIS IS A DISTINCT, FIRST-CLASS PRIMITIVE
---------------------------------------------
There are TWO levels of distribution in an eval, and they answer different
questions:

  Level 1  (episode distribution):  over episodes within ONE eval run.
            "How does the agent's behaviour spread out?"  -> CVaR, pass^k, profiles.

  Level 2  (META-distribution):     over the AGGREGATE SCORE across REPEATED whole
            evals, each with fresh task draws / env seeds / sampling.
            "If I re-ran this benchmark tomorrow, how much would the headline move?"

Level 2 is what this module adds. It is the *sampling distribution of the score
itself* -- i.e. eval reliability / test-retest noise.

WHY YOU CAN'T GET IT FROM A BOOTSTRAP
-------------------------------------
A bootstrap (bootstrap.py) resamples the episodes you ALREADY collected. It is
blind to the variance from re-drawing task instances, re-seeding stochastic
environments, and LLM nondeterminism, because those samples were frozen the
moment you collected them. So a single-run bootstrap CI is a *lower bound* on
true run-to-run variance. Repeating the whole eval captures all of it.

This is precisely the "is my 2% improvement real or eval noise?" question.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .bootstrap import stratified_bootstrap_ci
from .records import RecordStore


def run_repeated(eval_fn: Callable[[int], RecordStore], n_repeats: int) -> list[RecordStore]:
    """Run a full eval n_repeats times. eval_fn(rep_index) must produce a fresh
    RecordStore (fresh task draws / seeds / sampling each call)."""
    return [eval_fn(r) for r in range(n_repeats)]


def meta_distribution(
    stores: list[RecordStore],
    stat_fn: Callable[[pd.DataFrame], float],
    ci: float = 0.95,
) -> dict:
    """Distribution of an aggregate statistic across repeated whole evals."""
    vals = np.array([stat_fn(s.df()) for s in stores], dtype=float)
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "values": vals.tolist(),
        "n_repeats": len(stores),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "spread": float(vals.max() - vals.min()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_width": float(hi - lo),
    }


def bootstrap_vs_repeat(
    stores: list[RecordStore],
    stat_fn: Callable[[pd.DataFrame], float],
    strata_cols: list[str] | None = None,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """The headline diagnostic: how badly does a single-run bootstrap CI
    underestimate the true run-to-run spread?

    Returns the single-run bootstrap CI width (averaged over the repeats, so it's
    not cherry-picked) vs the empirical meta CI width. A ratio >> 1 means the
    bootstrap is overconfident and your published error bars are too tight.
    """
    meta = meta_distribution(stores, stat_fn, ci=ci)
    boot_widths = []
    for i, s in enumerate(stores):
        b = stratified_bootstrap_ci(s.df(), stat_fn, strata_cols=strata_cols,
                                    n_reps=2000, ci=ci, seed=seed + i)
        boot_widths.append(b["width"])
    boot_width = float(np.mean(boot_widths))
    return {
        "meta_ci_width": meta["ci_width"],
        "meta_std": meta["std"],
        "mean_single_run_bootstrap_ci_width": boot_width,
        "underconfidence_ratio": (meta["ci_width"] / boot_width) if boot_width > 0 else float("inf"),
        "verdict": (
            "bootstrap CI is a LOWER BOUND on true noise -- "
            f"true run-to-run spread is ~{meta['ci_width'] / boot_width:.1f}x wider"
            if boot_width > 0 else "n/a"
        ),
    }


def is_gap_real(
    stores_a: list[RecordStore],
    stores_b: list[RecordStore],
    stat_fn: Callable[[pd.DataFrame], float],
) -> dict:
    """Given repeated evals of A and B, is A's advantage real or within eval noise?

    Uses paired repeats (rep i of A vs rep i of B) to estimate P(A beats B on a
    fresh re-run) -- the decision-relevant quantity, not a point delta.
    """
    a = np.array([stat_fn(s.df()) for s in stores_a], dtype=float)
    b = np.array([stat_fn(s.df()) for s in stores_b], dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    p_a_wins = float((a > b).mean())
    return {
        "mean_gap": float(a.mean() - b.mean()),
        "P(A>B on a fresh re-run)": p_a_wins,
        "A_range": [float(a.min()), float(a.max())],
        "B_range": [float(b.min()), float(b.max())],
        "ranges_overlap": bool(a.min() <= b.max() and b.min() <= a.max()),
    }
