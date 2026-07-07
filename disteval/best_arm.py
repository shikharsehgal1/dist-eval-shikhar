"""Best-arm identification: spend an eval budget adaptively to find the best agent.

Comparing K agents (or K prompt/config variants) by running each the same number
of times wastes budget on obvious losers. Pure-exploration bandit algorithms
instead concentrate runs where the race is close and stop early, with a
correctness guarantee. Two complementary tools:

  - sequential_halving — FIXED total budget, parameter-free. Best default when
    eval runs cost a fixed amount of money/time (Karnin et al. 2013).
  - successive_elimination — FIXED confidence: keep running until the best arm is
    identified with probability >= 1-delta (Even-Dar et al. 2006).

Both take a ``pull_fn(arm_index) -> score`` callback returning a score in [0, 1]
(one fresh eval run of that arm), so they drive real evaluation loops.
"""
from __future__ import annotations

import math

import numpy as np


__all__ = ["sequential_halving", "successive_elimination"]


def sequential_halving(pull_fn, k: int, budget: int) -> dict:
    """Fixed-budget best-arm identification (Karnin, Koren & Somekh 2013).

    Runs ceil(log2 k) rounds; each round splits its share of the budget equally
    among the surviving arms, then discards the empirically worst half. Parameter-
    free and near-optimal — a good "smart budget splitter" for leaderboard
    screening. ``budget`` is the total number of ``pull_fn`` calls (approximately;
    at least one pull per surviving arm per round).

    Returns {best (arm index), means (per-arm empirical mean), pulls (per-arm
    count)}.
    """
    if k < 1:
        raise ValueError("need at least one arm")
    if k == 1:
        s = float(pull_fn(0))
        return {"best": 0, "means": {0: s}, "pulls": {0: 1}}
    sums = np.zeros(k)
    counts = np.zeros(k, dtype=int)
    active = list(range(k))
    n_rounds = max(1, math.ceil(math.log2(k)))
    for _ in range(n_rounds):
        per_arm = max(1, budget // (len(active) * n_rounds))
        for a in active:
            for _p in range(per_arm):
                sums[a] += float(pull_fn(a))
                counts[a] += 1
        means = {a: sums[a] / counts[a] for a in active}
        active.sort(key=lambda a: means[a], reverse=True)
        active = active[: max(1, len(active) // 2)]
    means_all = {a: float(sums[a] / counts[a]) for a in range(k) if counts[a] > 0}
    best = max(means_all, key=means_all.get)
    return {"best": best, "means": means_all, "pulls": {a: int(counts[a]) for a in range(k)}}


def successive_elimination(pull_fn, k: int, delta: float = 0.05, max_pulls: int = 100000) -> dict:
    """Fixed-confidence best-arm identification (Even-Dar, Mannor & Mansour 2006).

    Pulls every active arm once per round and eliminates an arm as soon as some
    other arm's confidence lower bound exceeds its upper bound. Returns the
    surviving arm, which is the best with probability >= 1-delta. The confidence
    radius sqrt(log(4 k n^2 / delta) / (2 n)) is time-uniform (union bound over
    arms and rounds), so continuous checking does not break the guarantee.

    ``max_pulls`` caps total ``pull_fn`` calls as a safety stop. Returns {best,
    means, pulls, total_pulls, stopped_early}.
    """
    if k < 1:
        raise ValueError("need at least one arm")
    if k == 1:
        s = float(pull_fn(0))
        return {"best": 0, "means": {0: s}, "pulls": {0: 1}, "total_pulls": 1, "stopped_early": True}
    sums = np.zeros(k)
    counts = np.zeros(k, dtype=int)
    active = set(range(k))
    total = 0
    stopped_early = False
    while len(active) > 1 and total < max_pulls:
        for a in list(active):
            sums[a] += float(pull_fn(a))
            counts[a] += 1
            total += 1
        means = sums / np.maximum(counts, 1)
        rad = np.array([
            math.sqrt(math.log(4 * k * counts[a] ** 2 / delta) / (2 * counts[a])) if counts[a] > 0 else float("inf")
            for a in range(k)
        ])
        best_lcb = max(means[a] - rad[a] for a in active)
        survivors = {a for a in active if means[a] + rad[a] >= best_lcb}
        if survivors and survivors != active:
            active = survivors
        if len(active) == 1:
            stopped_early = True
            break
    means_all = {a: float(sums[a] / counts[a]) for a in range(k) if counts[a] > 0}
    best = max(active, key=lambda a: means_all.get(a, -np.inf)) if active else max(means_all, key=means_all.get)
    return {"best": best, "means": means_all,
            "pulls": {a: int(counts[a]) for a in range(k)},
            "total_pulls": int(total), "stopped_early": stopped_early}
