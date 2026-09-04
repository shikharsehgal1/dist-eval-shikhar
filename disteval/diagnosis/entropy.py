"""Failure-mode entropy: does a task fail for one reason, or for many?

The hypothesis
--------------
A task whose failures all originate from the same localised defect ("it reads
the wrong quarter's file") plausibly has one thing to fix. A task that fails a
different way every time is diffuse incompetence, and matched preference pairs
are unlikely to teach anything coherent. So:

    Tasks with demonstrated capability AND low failure entropy may be especially
    recoverable.

This is an experimental feature and is stated as a hypothesis. It is one of the
candidate signals compared in :mod:`disteval.reliability.recoverability`, and
the simulation study in :mod:`disteval.sim` includes worlds where it is
predictive and worlds where it is not.

Estimation, honestly
--------------------
Entropy is estimated from very few failures -- often three to six. The plug-in
estimator is badly biased *downward* at those counts (it sees fewer distinct
modes than exist, so it reports more concentration than there is), which would
systematically flatter this hypothesis. Two corrections are therefore applied:

* **Miller-Madow bias correction**: adds ``(K_obs - 1) / (2N)`` nats, the
  first-order bias of the plug-in estimator, where ``K_obs`` is the number of
  modes actually observed.
* **Add-alpha smoothing** over the *registered* taxonomy rather than the
  observed support, so an unobserved mode is not assumed impossible. The default
  spreads a total pseudo-count of 1 across the support (``alpha = 1/K``, a Perks
  prior) rather than a fixed 0.5 per mode: with a ten-mode taxonomy and six
  observed failures, per-mode 0.5 would contribute five pseudo-counts against
  six real ones and swamp the data.

**Support choice matters and is not neutral.** Entropy is normalised by
``log(K)`` over whichever support is used, so a task scored against a 10-mode
taxonomy is not comparable to one scored against a 4-mode taxonomy. Hold the
support fixed across any set of tasks you intend to compare -- pass it
explicitly. ``dominant_share`` (the fraction of failures sharing the most common
mode) is support-free and is the more robust headline when N is small.

Both are configurable and both can be turned off. :func:`entropy_ci` gives a
bootstrap interval, which at N=4 is wide enough to make clear that a single
task's entropy is a weak measurement -- the signal only becomes usable when
aggregated or when N is larger.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "FailureDistribution",
    "failure_distribution",
    "plugin_entropy",
    "entropy_ci",
    "normalized_entropy",
    "dominant_mode_share",
]


def plugin_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in nats of a probability vector."""
    p = np.asarray([x for x in probs if x > 0], dtype=float)
    if p.size == 0:
        return float("nan")
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


@dataclass(frozen=True)
class FailureDistribution:
    """The estimated distribution of failure modes for one task."""

    task: str
    model: str
    counts: dict[str, int]
    probs: dict[str, float]
    n_failures: int
    entropy: float               # nats, corrected
    entropy_raw: float           # plug-in, uncorrected
    max_entropy: float           # log(K) over the smoothing support
    support_size: int

    @property
    def normalized(self) -> float:
        """Entropy scaled to [0, 1]. 0 = one repeated cause, 1 = uniform chaos."""
        if not np.isfinite(self.entropy) or self.max_entropy <= 0:
            return float("nan")
        return float(np.clip(self.entropy / self.max_entropy, 0.0, 1.0))

    @property
    def dominant_mode(self) -> Optional[str]:
        return max(self.probs, key=self.probs.get) if self.probs else None

    @property
    def dominant_share(self) -> float:
        """Fraction of failures attributed to the single most common mode."""
        if not self.counts or self.n_failures == 0:
            return float("nan")
        return max(self.counts.values()) / self.n_failures

    @property
    def concentration(self) -> float:
        """1 - normalized entropy: the "one repeated defect" score."""
        n = self.normalized
        return float(1.0 - n) if np.isfinite(n) else float("nan")

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "model": self.model,
            "n_failures": self.n_failures,
            "failure_entropy": self.entropy,
            "failure_entropy_raw": self.entropy_raw,
            "failure_entropy_normalized": self.normalized,
            "failure_concentration": self.concentration,
            "dominant_mode": self.dominant_mode,
            "dominant_share": self.dominant_share,
            "support_size": self.support_size,
        }


def failure_distribution(
    modes: Sequence[str],
    task: str = "",
    model: str = "",
    *,
    support: Optional[Sequence[str]] = None,
    alpha: Optional[float] = None,
    miller_madow: bool = True,
) -> FailureDistribution:
    """Estimate ``P(F = j | t)`` and its entropy from a list of failure labels.

    ``support`` defaults to the registered taxonomy, so unobserved modes get
    pseudo-counts rather than probability zero. ``alpha`` is the per-mode
    pseudo-count; ``None`` (the default) uses ``1/K``, spreading a total of one
    pseudo-observation across the support. Pass ``alpha=0`` and
    ``miller_madow=False`` for the raw plug-in estimate.

    Because the normalisation is ``log(K)``, results are only comparable across
    tasks scored against the same ``support``.
    """
    modes = [m for m in modes if m]
    n = len(modes)
    if support is None:
        from .taxonomy import TAXONOMY

        support = sorted(set(TAXONOMY) | set(modes))
    support = sorted(set(support) | set(modes))

    if alpha is None:
        alpha = 1.0 / max(len(support), 1)

    counts = Counter(modes)
    raw_probs = {m: counts.get(m, 0) / n for m in support} if n else {}
    entropy_raw = plugin_entropy(list(raw_probs.values())) if n else float("nan")

    denom = n + alpha * len(support)
    probs = (
        {m: (counts.get(m, 0) + alpha) / denom for m in support}
        if denom > 0
        else {}
    )
    ent = plugin_entropy(list(probs.values())) if probs else float("nan")
    if miller_madow and n > 0:
        k_obs = len([m for m in support if counts.get(m, 0) > 0])
        ent = ent + (k_obs - 1) / (2.0 * n)

    return FailureDistribution(
        task=task,
        model=model,
        counts=dict(counts),
        probs=probs,
        n_failures=n,
        entropy=float(ent),
        entropy_raw=float(entropy_raw),
        max_entropy=float(np.log(len(support))) if support else float("nan"),
        support_size=len(support),
    )


def normalized_entropy(modes: Sequence[str], **kwargs) -> float:
    """Convenience: the [0, 1] normalised failure entropy."""
    return failure_distribution(modes, **kwargs).normalized


def dominant_mode_share(modes: Sequence[str]) -> float:
    """Fraction of failures sharing the single most common mode."""
    modes = [m for m in modes if m]
    if not modes:
        return float("nan")
    return max(Counter(modes).values()) / len(modes)


def entropy_ci(
    modes: Sequence[str],
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    **kwargs,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the corrected entropy.

    At N=4 failures this interval spans most of the range. That is the correct
    message: a single task's failure entropy is a weak measurement, and the
    signal should be used in aggregate or with more runs.
    """
    modes = [m for m in modes if m]
    if len(modes) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(modes, dtype=object)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        vals.append(failure_distribution(list(sample), **kwargs).entropy)
    tail = (1 - level) / 2
    return (float(np.quantile(vals, tail)), float(np.quantile(vals, 1 - tail)))
