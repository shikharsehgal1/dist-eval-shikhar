"""Are repeated runs actually independent? Usually not, and it matters.

Every interval in this package -- Beta-Binomial credible intervals, pass^k,
bootstrap CIs -- assumes the runs of a task are i.i.d. draws. Real agent
evaluation violates this routinely:

* runs share an environment snapshot, so a broken fixture fails all of them;
* runs share a seed or a cached retrieval index;
* runs execute in the same provider window, so a degraded endpoint hits all of
  them together;
* runs are graded by the same LLM judge instance.

Under positive intra-cluster correlation, the effective number of independent
observations is smaller than the run count, and every interval is too narrow --
the classic overconfidence failure. This module measures how much.

What is computed
----------------
**Intra-cluster correlation (ICC).** For binary outcomes grouped by a cluster
key (seed, environment version, provider window, judge), the ANOVA estimator of
``rho``. Zero means the clustering carries no information; positive means runs
within a cluster resemble each other more than runs across clusters.

**Design effect.** ``deff = 1 + (m_bar - 1) * rho`` with ``m_bar`` the average
cluster size. This is the factor by which the variance is understated, so
``n_eff = n / deff`` is the honest sample size. A deff of 2 means your eight runs
are worth four.

**Inflated intervals.** :func:`adjusted_posterior` returns a Beta posterior built
on ``n_eff`` rather than ``n``, so every downstream quantity -- capability,
reliability, classification -- inherits the correction with no other change.

**A warning, not a silent fix.** :func:`iid_diagnostics` returns an explicit
``warn`` flag and message when the design effect exceeds a threshold. The default
pipeline surfaces it in the report. Silently applying a correction would hide a
data-collection problem that is usually better fixed at the source (vary the
seeds, re-snapshot the environment) than adjusted for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from .posterior import JEFFREYS_PRIOR, BetaPrior, TaskPosterior

__all__ = ["ClusterDiagnostic", "intraclass_correlation", "design_effect",
           "iid_diagnostics", "adjusted_posterior"]


@dataclass(frozen=True)
class ClusterDiagnostic:
    """How far from i.i.d. a set of runs is, under one clustering."""

    cluster_key: str
    n_runs: int
    n_clusters: int
    mean_cluster_size: float
    icc: float
    design_effect: float
    n_effective: float
    warn: bool
    message: str

    def to_dict(self) -> dict:
        return {
            "cluster_key": self.cluster_key,
            "n_runs": self.n_runs,
            "n_clusters": self.n_clusters,
            "mean_cluster_size": self.mean_cluster_size,
            "icc": self.icc,
            "design_effect": self.design_effect,
            "n_effective": self.n_effective,
            "warn": self.warn,
            "message": self.message,
        }


def intraclass_correlation(values: Sequence[float], clusters: Sequence) -> float:
    """One-way ANOVA estimator of the intra-cluster correlation.

    ``rho = (MSB - MSW) / (MSB + (m0 - 1) * MSW)`` with ``m0`` the
    balance-corrected average cluster size. Clipped to [0, 1]: a negative point
    estimate means the clustering carries no positive information, which is
    reported as 0 rather than as a negative correlation that would *narrow*
    intervals.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(list(clusters))
    if v.size < 3:
        return 0.0
    uniq = np.unique(c)
    k = uniq.size
    if k < 2 or k == v.size:
        return 0.0
    sizes = np.array([np.sum(c == u) for u in uniq], dtype=float)
    means = np.array([v[c == u].mean() for u in uniq], dtype=float)
    grand = float(v.mean())
    ssb = float(np.sum(sizes * (means - grand) ** 2))
    ssw = float(sum(float(np.sum((v[c == u] - m) ** 2)) for u, m in zip(uniq, means)))
    dfb, dfw = k - 1, v.size - k
    if dfb <= 0 or dfw <= 0:
        return 0.0
    msb, msw = ssb / dfb, ssw / dfw
    n = float(v.size)
    m0 = (n - float(np.sum(sizes**2)) / n) / (k - 1)
    denom = msb + (m0 - 1.0) * msw
    if denom <= 0:
        return 0.0
    return float(np.clip((msb - msw) / denom, 0.0, 1.0))


def design_effect(icc: float, mean_cluster_size: float) -> float:
    """``1 + (m_bar - 1) * icc``: the variance inflation from clustering."""
    return float(max(1.0 + (mean_cluster_size - 1.0) * icc, 1.0))


def iid_diagnostics(
    values: Sequence[float],
    clusters: Sequence,
    cluster_key: str = "cluster",
    warn_threshold: float = 1.25,
) -> ClusterDiagnostic:
    """Full clustering diagnostic with an explicit warning flag."""
    v = np.asarray(values, dtype=float)
    c = list(clusters)
    uniq = sorted(set(c))
    k = len(uniq)
    m_bar = (v.size / k) if k else 0.0
    icc = intraclass_correlation(v, c)
    deff = design_effect(icc, m_bar)
    n_eff = v.size / deff if deff > 0 else float(v.size)
    warn = deff > warn_threshold
    msg = ""
    if warn:
        msg = (
            f"runs are correlated within {cluster_key!r} (ICC={icc:.2f}, design "
            f"effect={deff:.2f}): {v.size} runs carry about {n_eff:.1f} runs' worth "
            f"of information, so naive Binomial intervals are too narrow. Prefer "
            f"varying {cluster_key} at collection time over adjusting after the fact."
        )
    elif k < 2:
        msg = f"only one distinct {cluster_key!r} value; correlation is not estimable"
    return ClusterDiagnostic(
        cluster_key=cluster_key,
        n_runs=int(v.size),
        n_clusters=k,
        mean_cluster_size=float(m_bar),
        icc=float(icc),
        design_effect=float(deff),
        n_effective=float(n_eff),
        warn=bool(warn),
        message=msg,
    )


def adjusted_posterior(
    successes: float,
    trials: int,
    diagnostic: ClusterDiagnostic,
    prior: BetaPrior = JEFFREYS_PRIOR,
) -> TaskPosterior:
    """Beta posterior built on the effective, not the nominal, sample size.

    Successes are scaled by the same factor as trials so the posterior mean is
    unchanged and only the width grows -- clustering costs you precision, not
    accuracy.
    """
    if trials <= 0:
        return TaskPosterior(prior.alpha, prior.beta, 0, 0.0, True, prior)
    factor = diagnostic.n_effective / trials
    s_eff = float(successes) * factor
    n_eff = float(trials) * factor
    return TaskPosterior(
        alpha=prior.alpha + s_eff,
        beta=prior.beta + (n_eff - s_eff),
        n_obs=int(trials),
        n_eff=n_eff,
        binary=True,
        prior=prior,
    )
