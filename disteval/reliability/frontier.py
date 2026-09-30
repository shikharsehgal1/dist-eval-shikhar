"""The learnable frontier: is a task's success rate inside a target band?

The quantity
------------
A recurring idea in curriculum learning and synthetic-task generation is that the
useful tasks are neither trivial nor impossible, but sit in a band::

    U(t) = 1[ a <= p_t <= b ]

with, for example, ``a = 1/8, b = 3/8`` for math and code and ``a = 1/3, b = 2/3``
for software-engineering tasks (Wolf et al., 2026, "Breaking the Solver
Bottleneck: Training Task Generators at the Learnable Frontier"). The same
intuition drives competence-progress curricula and the ``E[p(1-p)]``
learnability proxy in :mod:`disteval.selection.selectors`.

**This module estimates band membership. It does not assert that the band is
where training value lives** -- that is the hypothesis this repository exists to
test, and the fact that the intuition is widely held and independently
operationalised is a reason to test it carefully, not a reason to assume it.

Why the estimate needs uncertainty
----------------------------------
Band membership is usually computed as a plug-in: run the task ``K`` times, take
``k/K``, check whether it lands in ``[a, b]``. At ``K = 8`` with a band of
``[0.125, 0.375]``, that is a decision about a Bernoulli parameter from eight
draws, and the plug-in is wrong often:

* a task at true ``p = 0.45`` (outside the band) produces 1, 2 or 3 successes in
  8 runs -- and is therefore *labelled* in-band -- roughly a third of the time;
* a task at true ``p = 0.25`` (comfortably inside) lands outside the band about
  a quarter of the time.

:func:`band_probability` reports ``P(a <= p_t <= b | D)`` instead, and
:func:`band_decision` refuses to label a task whose membership is not resolved,
which is the honest output at small ``K``.

Why this matters practically
----------------------------
Labelling tasks into the band is the expensive step in a generate-and-filter
pipeline: it costs ``K`` solver rollouts per candidate, and for long-horizon
agent tasks a rollout can take minutes. :func:`band_information` scores how much
one more run would resolve a task's membership, so
:mod:`disteval.active` can spend the rollout budget on candidates whose label is
still in doubt rather than re-confirming obvious ones. That is the same
value-of-information argument used for reliability classification, applied to a
different decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .classify import PosteriorLike
from .posterior import TaskPosterior

__all__ = [
    "FrontierBand",
    "MATH_CODE_BAND",
    "SWE_BAND",
    "BandDecision",
    "band_probability",
    "band_decision",
    "band_information",
    "expected_runs_to_resolve_band",
    "plugin_band_error",
]


@dataclass(frozen=True)
class FrontierBand:
    """A target success-rate interval ``[lo, hi]``."""

    lo: float
    hi: float
    name: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.lo < self.hi <= 1.0:
            raise ValueError(f"band must satisfy 0 <= lo < hi <= 1, got [{self.lo}, {self.hi}]")

    def contains(self, p: float) -> bool:
        return self.lo <= p <= self.hi

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @classmethod
    def from_counts(cls, lo_k: int, hi_k: int, n: int, name: str = "") -> "FrontierBand":
        """Band expressed as success counts out of ``n``, e.g. 1/8 to 3/8."""
        return cls(lo_k / n, hi_k / n, name or f"{lo_k}/{n}-{hi_k}/{n}")


#: Band used for math and code tasks in the PROPEL setup (1/8 to 3/8 of K=8).
MATH_CODE_BAND = FrontierBand(1 / 8, 3 / 8, "math_code")
#: Band used for software-engineering tasks there (1/3 to 2/3 of K=3).
SWE_BAND = FrontierBand(1 / 3, 2 / 3, "swe")


@dataclass(frozen=True)
class BandDecision:
    """Whether a task is inside the frontier band, and how sure we are."""

    task: str
    band: FrontierBand
    p_in: float            # P(lo <= p <= hi | D)
    p_below: float         # P(p < lo | D) -- too easy to be informative? no: too hard
    p_above: float         # P(p > hi | D)
    posterior_mean: float
    n_runs: int
    label: str             # "in_band" | "below_band" | "above_band" | "unresolved"
    plugin_label: str      # what the naive k/K rule would have said
    confidence: float

    @property
    def disagrees_with_plugin(self) -> bool:
        """Whether the posterior decision differs from the plug-in one.

        Reported because the plug-in rule is what generate-and-filter pipelines
        typically use, and the disagreement rate is the cost of using it.
        """
        return self.label != self.plugin_label

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "band": [self.band.lo, self.band.hi],
            "p_in_band": self.p_in,
            "p_below": self.p_below,
            "p_above": self.p_above,
            "posterior_mean": self.posterior_mean,
            "n_runs": self.n_runs,
            "label": self.label,
            "plugin_label": self.plugin_label,
            "disagrees_with_plugin": self.disagrees_with_plugin,
            "confidence": self.confidence,
        }


def band_probability(post: PosteriorLike, band: FrontierBand) -> float:
    """``P(band.lo <= p <= band.hi | D)``, exactly from the posterior CDF."""
    return float(max(post.prob_above(band.lo) - post.prob_above(band.hi), 0.0))


def _plugin_label(successes: float, trials: int, band: FrontierBand) -> str:
    if trials <= 0:
        return "unresolved"
    rate = successes / trials
    if rate < band.lo:
        return "below_band"
    if rate > band.hi:
        return "above_band"
    return "in_band"


def band_decision(
    post: PosteriorLike,
    band: FrontierBand,
    *,
    task: str = "",
    confidence: float = 0.80,
    successes: Optional[float] = None,
) -> BandDecision:
    """Classify a task against the band, declining when it is not resolved.

    ``unresolved`` is a common and correct outcome at small ``K``. A pipeline
    that must decide anyway should use ``p_in`` as a weight rather than forcing a
    label -- a task at ``p_in = 0.45`` is genuinely half a frontier task, and
    thresholding it discards that.
    """
    p_in = band_probability(post, band)
    p_above = float(post.prob_above(band.hi))
    p_below = float(max(1.0 - post.prob_above(band.lo), 0.0))

    label = "unresolved"
    conf = max(p_in, p_above, p_below)
    if p_in >= confidence:
        label = "in_band"
    elif p_above >= confidence:
        label = "above_band"
    elif p_below >= confidence:
        label = "below_band"

    n = int(getattr(post, "n_obs", 0))
    s = successes if successes is not None else float(getattr(post, "n_success", np.nan))
    if not np.isfinite(s):
        s = post.mean * n
    return BandDecision(
        task=task, band=band, p_in=p_in, p_below=p_below, p_above=p_above,
        posterior_mean=float(post.mean), n_runs=n, label=label,
        plugin_label=_plugin_label(s, n, band), confidence=float(conf),
    )


def _updated(post: TaskPosterior, success: bool) -> TaskPosterior:
    return TaskPosterior(
        alpha=post.alpha + (1.0 if success else 0.0),
        beta=post.beta + (0.0 if success else 1.0),
        n_obs=post.n_obs + 1, n_eff=post.n_eff + 1.0,
        binary=post.binary, prior=post.prior,
    )


def band_information(
    post: TaskPosterior, band: FrontierBand, horizon: int = 1
) -> float:
    """Expected reduction in band-membership entropy from ``horizon`` more runs.

    The quantity to maximise when allocating a rollout budget across candidate
    tasks whose band membership is still in doubt. Exact, not sampled: the
    posterior predictive over the next ``horizon`` outcomes is a Beta-Binomial,
    so the expectation is a ``horizon + 1`` term sum.

    Near zero for a task already clearly in or clearly out of the band -- which
    is precisely the candidate not worth another rollout.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    def h(p: float) -> float:
        p = float(np.clip(p, 1e-12, 1 - 1e-12))
        return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))

    before = h(band_probability(post, band))
    pmf = post.posterior_predictive(horizon)
    after = 0.0
    for j, w in enumerate(pmf):
        if w <= 0:
            continue
        nxt = TaskPosterior(
            alpha=post.alpha + j, beta=post.beta + (horizon - j),
            n_obs=post.n_obs + horizon, n_eff=post.n_eff + horizon,
            binary=post.binary, prior=post.prior,
        )
        after += float(w) * h(band_probability(nxt, band))
    return float(max(before - after, 0.0))


def expected_runs_to_resolve_band(
    post: TaskPosterior,
    band: FrontierBand,
    *,
    confidence: float = 0.80,
    max_runs: int = 64,
) -> float:
    """Smallest ``k`` making band membership more likely than not to be resolved.

    ``inf`` when no ``k <= max_runs`` suffices. Useful for triage: a candidate
    needing 40 more rollouts to classify is usually better discarded than
    funded, and that is a budget decision the plug-in rule cannot inform.
    """
    if band_decision(post, band, confidence=confidence).label != "unresolved":
        return 0.0
    for k in range(1, max_runs + 1):
        pmf = post.posterior_predictive(k)
        resolved = 0.0
        for j, w in enumerate(pmf):
            nxt = TaskPosterior(
                post.alpha + j, post.beta + (k - j), post.n_obs + k,
                post.n_eff + k, post.binary, post.prior,
            )
            if band_decision(nxt, band, confidence=confidence).label != "unresolved":
                resolved += float(w)
        if resolved > 0.5:
            return float(k)
    return float("inf")


def plugin_band_error(
    band: FrontierBand,
    k: int,
    *,
    true_ps: Optional[Sequence[float]] = None,
    n_grid: int = 201,
) -> "object":
    """How often the plug-in ``k/K`` rule mislabels band membership.

    For each true ``p``, the exact probability that ``k/K`` falls on the wrong
    side of the band, computed from the Binomial pmf -- no simulation. This is
    the quantity that justifies estimating band membership rather than reading it
    off a ratio, and at the sample sizes generate-and-filter pipelines use it is
    large.

    Returns a DataFrame over ``true_p`` with the misclassification probability
    and the direction of the error.
    """
    import pandas as pd
    from scipy import stats

    ps = list(true_ps) if true_ps is not None else list(np.linspace(0.01, 0.99, n_grid))
    rows = []
    counts = np.arange(k + 1)
    rate = counts / k
    is_in = (rate >= band.lo) & (rate <= band.hi)
    for p in ps:
        pmf = stats.binom.pmf(counts, k, p)
        truth = band.contains(p)
        p_label_in = float(pmf[is_in].sum())
        err = (1.0 - p_label_in) if truth else p_label_in
        rows.append({
            "true_p": float(p),
            "in_band_truth": bool(truth),
            "p_labelled_in_band": p_label_in,
            "p_mislabelled": float(err),
            "error_type": ("missed" if truth else "false_positive") if err > 0 else "none",
        })
    return pd.DataFrame(rows)
