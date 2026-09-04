"""Value of information: what is one more run on this task actually worth?

Repeated-run evaluation of a frontier agent is the dominant cost of this whole
methodology. Eight runs on 500 tasks is 4000 agent executions, and most of them
are spent confirming what is already obvious -- a task at 0/6 does not need a
seventh run to establish that it is stuck.

Every function here scores a *candidate next run* on one task, so an allocation
policy can spend the budget where it changes something. All are exact rather than
Monte-Carlo: with a Beta posterior the predictive distribution over the next
binary outcome is two-valued, so the expectation is a two-term sum and there is
no sampling noise to destabilise the ranking.

The three criteria answer different questions, and which one is right depends on
what the evaluation is *for*:

``expected_entropy_reduction``
    How much the posterior tightens. The right criterion when the goal is an
    accurate estimate of every task's reliability.

``bald``
    Mutual information between the next outcome and the latent parameter:
    ``H[E[p]] - E[H[p]]``. Prefers tasks where the *outcome* is uncertain because
    the *parameter* is uncertain, rather than because the parameter is genuinely
    near 0.5. This distinction matters: a task known to be a fair coin has a
    maximally uncertain outcome and nothing left to learn.

``flip_probability``
    Probability that the task's SOLID/RECOVERABLE/STUCK/UNCERTAIN label changes
    within the next ``horizon`` runs. The right criterion when the evaluation
    exists to *classify* tasks -- which is the case when the output feeds
    training-data selection. It is the criterion the default policy uses.

    ``horizon`` defaults to 2 rather than 1 for a concrete reason: greedy
    one-step lookahead is blind to tasks that need two more runs to resolve. A
    task at 0/6 is UNCERTAIN and *no* single additional run can change that --
    one-step flip probability is exactly 0 -- yet two more failures would settle
    it as STUCK. A one-step policy starves precisely the tasks nearest to
    becoming decidable. The multi-step version is still exact, not sampled: the
    label after k more runs depends on the outcomes only through their count, so
    the expectation is a k+1 term sum against the Beta-Binomial predictive.

``rank_instability``
    How uncertain a task's position in the recoverability ranking is, estimated
    by posterior resampling. The right criterion when only the top-k matters.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..reliability.classify import ReliabilityThresholds, _label
from ..reliability.posterior import BetaPrior, TaskPosterior, binary_posterior

__all__ = [
    "expected_entropy_reduction",
    "bald",
    "flip_probability",
    "expected_runs_to_decide",
    "rank_instability",
    "label_of",
]


def _bernoulli_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def _updated(post: TaskPosterior, success: bool) -> TaskPosterior:
    """The posterior after observing one more binary outcome."""
    return TaskPosterior(
        alpha=post.alpha + (1.0 if success else 0.0),
        beta=post.beta + (0.0 if success else 1.0),
        n_obs=post.n_obs + 1,
        n_eff=post.n_eff + 1.0,
        binary=post.binary,
        prior=post.prior,
    )


def expected_entropy_reduction(post: TaskPosterior) -> float:
    """Expected drop in the posterior's differential entropy from one more run.

    Always non-negative in expectation (observing data cannot increase expected
    entropy), and largest for tasks whose posterior is still wide.
    """
    p = post.mean
    h0 = post.entropy
    h1 = p * _updated(post, True).entropy + (1 - p) * _updated(post, False).entropy
    return float(h0 - h1)


def bald(post: TaskPosterior) -> float:
    """Mutual information ``I(y_next ; p) = H[E[p]] - E[H[p]]``, in nats.

    For a Beta posterior, ``E[H(Bernoulli(p))]`` is computed by quadrature. The
    result separates *epistemic* uncertainty (we don't know p) from *aleatoric*
    uncertainty (p really is near 0.5) and rewards only the former -- which is
    the whole point, since more runs cannot reduce aleatoric uncertainty.
    """
    grid = np.linspace(1e-6, 1 - 1e-6, 512)
    cdf = np.array([1.0 - post.prob_above(x) for x in grid])
    mass = np.diff(cdf)
    mid = 0.5 * (grid[:-1] + grid[1:])
    e_h = float(np.sum(mass * np.array([_bernoulli_entropy(m) for m in mid])))
    return float(max(_bernoulli_entropy(post.mean) - e_h, 0.0))


def label_of(post: TaskPosterior, th: ReliabilityThresholds) -> str:
    """The task label implied by a posterior, without building a full diagnosis."""
    return _label(
        post.prob_above(th.tau_cap),
        post.prob_above(th.tau_rel),
        1.0 - post.prob_above(th.tau_stuck),
        post.n_obs,
        th,
    )


def flip_probability(
    post: TaskPosterior,
    thresholds: Optional[ReliabilityThresholds] = None,
    horizon: int = 2,
) -> float:
    """``P(the task's label changes within the next ``horizon`` runs)``.

    Exact, not sampled. The label after ``k`` further runs is a function of the
    success count alone, so the expectation is a ``k+1`` term sum weighted by the
    Beta-Binomial posterior predictive.

    Returns 0 for a task whose label is robust across every reachable outcome --
    precisely the task not worth more runs when the goal is classification. See
    the module docstring for why ``horizon=1`` is the wrong default.
    """
    th = thresholds or ReliabilityThresholds()
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    # A task below min_runs is UNCERTAIN by rule, and no horizon shorter than the
    # gap can change that. Without this extension a brand-new task scores exactly
    # 0 and an allocation policy driven by this criterion would never run it --
    # starving precisely the tasks it knows least about.
    horizon = max(horizon, th.min_runs - post.n_obs)
    now = label_of(post, th)
    pmf = post.posterior_predictive(horizon)
    total = 0.0
    for j, w in enumerate(pmf):
        if w <= 0:
            continue
        nxt = TaskPosterior(
            alpha=post.alpha + j,
            beta=post.beta + (horizon - j),
            n_obs=post.n_obs + horizon,
            n_eff=post.n_eff + horizon,
            binary=post.binary,
            prior=post.prior,
        )
        if label_of(nxt, th) != now:
            total += float(w)
    return float(total)


def rank_instability(
    posteriors: Sequence[TaskPosterior],
    scores_fn=None,
    *,
    top_k: int = 10,
    n_samples: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """Per-task probability of moving in or out of the top-``k`` ranking.

    Resamples each task's latent value from its posterior, re-ranks, and measures
    how often each task's top-k membership differs from the point-estimate
    ranking. Tasks near the top-k boundary score high; tasks safely in or safely
    out score ~0. This is the criterion to optimise when the evaluation exists to
    pick a training set, because only the boundary is decision-relevant.
    """
    scores_fn = scores_fn or (lambda p: p.mean)
    rng = np.random.default_rng(seed)
    n = len(posteriors)
    if n == 0:
        return np.array([])
    k = int(min(top_k, n))
    point = np.array([scores_fn(p) for p in posteriors], dtype=float)
    base = set(np.argsort(-point)[:k].tolist())

    draws = np.vstack([p.sample(n_samples, rng) for p in posteriors])  # (n, S)
    flips = np.zeros(n)
    for s in range(n_samples):
        order = np.argsort(-draws[:, s])[:k]
        cur = set(order.tolist())
        for i in range(n):
            if (i in cur) != (i in base):
                flips[i] += 1
    return flips / n_samples


def expected_runs_to_decide(
    post: TaskPosterior,
    thresholds: Optional[ReliabilityThresholds] = None,
    max_runs: int = 32,
) -> float:
    """Smallest ``k`` such that the label is more likely than not to be settled.

    Concretely: the least ``k`` for which the posterior-predictive probability of
    leaving UNCERTAIN within ``k`` further runs exceeds 0.5, or ``inf`` if no
    ``k <= max_runs`` achieves it. Reported alongside the allocation so a reader
    can see which tasks are cheap to resolve and which are hopeless at any
    realistic budget -- a task needing 30 more runs to classify is usually better
    dropped than funded.
    """
    th = thresholds or ReliabilityThresholds()
    if label_of(post, th) != "UNCERTAIN":
        return 0.0
    for k in range(1, max_runs + 1):
        pmf = post.posterior_predictive(k)
        settled = 0.0
        for j, w in enumerate(pmf):
            nxt = TaskPosterior(
                post.alpha + j, post.beta + (k - j), post.n_obs + k,
                post.n_eff + k, post.binary, post.prior,
            )
            if label_of(nxt, th) != "UNCERTAIN":
                settled += float(w)
        if settled > 0.5:
            return float(k)
    return float("inf")
