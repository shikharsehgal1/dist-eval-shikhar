"""Simulation studies that validate (or refute) the methodology's claims.

Each function answers one question that cannot be answered on real data, because
the truth is unavailable there. They are the evidence base for the claims made in
``THEORY.md``, and every one of them can come out negative -- several do.

Studies
-------
``max_bias_study``
    Is the observed maximum really a biased capability estimate, and by how much?
    Quantifies the growth of ``E[max]`` with ``k`` against the posterior mean.

``posterior_recovery_study``
    Does the Beta-Binomial posterior recover ``p_t``? Reports bias, RMSE, and --
    the one that matters -- **credible-interval coverage**. A 95% interval that
    covers 80% of the time is a broken estimator regardless of its RMSE.

``shrinkage_study``
    Does hierarchical pooling beat independent per-task estimates, and at what
    ``n``? Includes the regime where it should *not* help (highly heterogeneous
    tasks), because a shrinkage method that always helps is not being tested.

``classification_study``
    How often is the SOLID/RECOVERABLE/STUCK label wrong, as a function of ``n``?
    Reports the confusion matrix and the UNCERTAIN rate, which is the price paid
    for not guessing.

``runs_needed_study``
    How many runs per task are actually needed for a target classification
    accuracy? The practical planning number.

``adaptive_saving_study``
    Does adaptive allocation reach a given classification quality with fewer
    executions than uniform? Measured against ground truth, not against uniform.

``decomposition_study``
    Does the capability/execution split recover ``q`` and ``r``, and what happens
    when its necessity assumption is deliberately broken?

``selection_study``
    The central one. Across all four benefit-coupling regimes, does
    recoverability selection capture more true training benefit than random,
    hardest, and variance selection?
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..active.allocation import GreedyValuePolicy, UniformAllocation
from ..active.evaluator import compare_allocation
from ..reliability.classify import ReliabilityThresholds, diagnose
from ..reliability.hierarchical import fit_hierarchical
from ..reliability.posterior import binary_posterior
from ..selection.selectors import make_selector
from .world import BENEFIT_COUPLINGS, SimulatedWorld, WorldConfig

__all__ = [
    "max_bias_study",
    "posterior_recovery_study",
    "shrinkage_study",
    "classification_study",
    "runs_needed_study",
    "adaptive_saving_study",
    "decomposition_study",
    "selection_study",
    "band_label_noise_study",
    "run_all_studies",
]


def _frame(rows):
    import pandas as pd

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def max_bias_study(
    ks: Sequence[int] = (1, 2, 4, 8, 16, 32), n_tasks: int = 400, seed: int = 0
):
    """Quantify the upward bias of the observed maximum as an estimate of p_t.

    For each ``k``, compares ``E[max of k runs]`` and ``E[posterior mean]`` against
    the true ``p_t``. The max's bias grows monotonically with ``k`` by
    construction; this measures how large it actually is at realistic run counts.
    """
    w = SimulatedWorld(WorldConfig(n_tasks=n_tasks, seed=seed))
    rng = np.random.default_rng(seed + 1)
    rows = []
    for k in ks:
        max_err, post_err, mean_err = [], [], []
        for task, truth in w.truth.items():
            s = w.scores(task, k, rng)
            max_err.append(max(s) - truth.p)
            post_err.append(binary_posterior(int(sum(s)), k).mean - truth.p)
            mean_err.append(float(np.mean(s)) - truth.p)
        rows.append(
            {
                "k": k,
                "max_bias": float(np.mean(max_err)),
                "max_rmse": float(np.sqrt(np.mean(np.square(max_err)))),
                "posterior_bias": float(np.mean(post_err)),
                "posterior_rmse": float(np.sqrt(np.mean(np.square(post_err)))),
                "sample_mean_bias": float(np.mean(mean_err)),
                "sample_mean_rmse": float(np.sqrt(np.mean(np.square(mean_err)))),
            }
        )
    return _frame(rows)


def posterior_recovery_study(
    ks: Sequence[int] = (2, 3, 5, 8, 16, 32),
    n_tasks: int = 400,
    level: float = 0.95,
    seed: int = 0,
):
    """Bias, RMSE and credible-interval coverage of the Beta-Binomial posterior.

    Coverage is the decisive column. A 95% interval should contain the truth 95%
    of the time; systematic under-coverage means every downstream probability
    statement is overconfident. Note that exact nominal coverage is not expected:
    the prior is not the true task-generating distribution, so there is genuine
    prior-data conflict at small n. That is the honest reading, and the study
    reports it rather than tuning the prior until the number looks right.
    """
    w = SimulatedWorld(WorldConfig(n_tasks=n_tasks, seed=seed))
    rng = np.random.default_rng(seed + 2)
    rows = []
    for k in ks:
        err, cov, widths = [], [], []
        for task, truth in w.truth.items():
            s = w.scores(task, k, rng)
            post = binary_posterior(int(sum(s)), k)
            lo, hi = post.credible_interval(level)
            err.append(post.mean - truth.p)
            cov.append(lo <= truth.p <= hi)
            widths.append(hi - lo)
        rows.append(
            {
                "k": k,
                "bias": float(np.mean(err)),
                "rmse": float(np.sqrt(np.mean(np.square(err)))),
                "coverage": float(np.mean(cov)),
                "nominal": level,
                "mean_ci_width": float(np.mean(widths)),
            }
        )
    return _frame(rows)


def shrinkage_study(
    ks: Sequence[int] = (2, 3, 5, 8, 16),
    n_tasks: int = 150,
    domain_sds: Sequence[float] = (0.2, 1.5),
    seed: int = 0,
):
    """Does hierarchical pooling beat independent estimates, and when?

    Run at two levels of between-domain heterogeneity. Pooling should help when
    tasks are similar (low ``domain_effect_sd``) and help little or not at all
    when they are genuinely different -- a shrinkage method that always wins is
    not being tested against a case where it should not.
    """
    rows = []
    for sd in domain_sds:
        w = SimulatedWorld(
            WorldConfig(n_tasks=n_tasks, domain_effect_sd=sd, seed=seed)
        )
        rng = np.random.default_rng(seed + 3)
        tasks = w.tasks()
        for k in ks:
            counts = {t: int(sum(w.scores(t, k, rng))) for t in tasks}
            indep = np.array([binary_posterior(counts[t], k).mean for t in tasks])
            fit = fit_hierarchical(
                ["m"] * len(tasks), tasks, [counts[t] for t in tasks], [k] * len(tasks),
                [w.truth[t].domain for t in tasks],
            )
            pooled = np.array([fit.cell("m", t).mean for t in tasks])
            truth = np.array([w.truth[t].p for t in tasks])
            rows.append(
                {
                    "domain_effect_sd": sd,
                    "k": k,
                    "independent_rmse": float(np.sqrt(np.mean((indep - truth) ** 2))),
                    "hierarchical_rmse": float(np.sqrt(np.mean((pooled - truth) ** 2))),
                    "rmse_reduction": float(
                        1 - np.sqrt(np.mean((pooled - truth) ** 2))
                        / np.sqrt(np.mean((indep - truth) ** 2))
                    ),
                    "mean_shrinkage": float(
                        np.mean([fit.shrinkage("m", t) for t in tasks])
                    ),
                }
            )
    return _frame(rows)


def classification_study(
    ks: Sequence[int] = (2, 3, 5, 8, 16, 32),
    n_tasks: int = 400,
    thresholds: Optional[ReliabilityThresholds] = None,
    seed: int = 0,
):
    """Label accuracy vs run count, with the UNCERTAIN rate as the honest cost.

    ``accuracy_excl_uncertain`` is accuracy over tasks the classifier was willing
    to label; ``uncertain_rate`` is how often it declined. Both are needed:
    a classifier can trivially get 100% accuracy by labelling nothing.
    ``recoverable_recall`` and ``recoverable_precision`` are broken out because
    that class is the one that feeds selection.
    """
    th = thresholds or ReliabilityThresholds()
    w = SimulatedWorld(WorldConfig(n_tasks=n_tasks, seed=seed))
    rng = np.random.default_rng(seed + 4)
    rows = []
    for k in ks:
        got, true = [], []
        for task, truth in w.truth.items():
            s = w.scores(task, k, rng)
            d = diagnose(binary_posterior(int(sum(s)), k), task=task, thresholds=th, scores=s)
            got.append(d.label)
            true.append(truth.true_label)
        got_a, true_a = np.array(got), np.array(true)
        decided = got_a != "UNCERTAIN"
        # MARGINAL is a true-label bucket with no classifier counterpart; exclude
        # it from accuracy rather than scoring the classifier on an impossible call.
        scorable = decided & (true_a != "MARGINAL")
        tp = int(np.sum((got_a == "RECOVERABLE") & (true_a == "RECOVERABLE")))
        rows.append(
            {
                "k": k,
                "uncertain_rate": float(np.mean(~decided)),
                "accuracy_excl_uncertain": float(
                    np.mean(got_a[scorable] == true_a[scorable])
                ) if scorable.any() else float("nan"),
                "accuracy_incl_uncertain": float(np.mean(got_a == true_a)),
                "recoverable_recall": float(
                    tp / max(int(np.sum(true_a == "RECOVERABLE")), 1)
                ),
                "recoverable_precision": float(
                    tp / max(int(np.sum(got_a == "RECOVERABLE")), 1)
                ),
                "solid_misread_as_recoverable": int(
                    np.sum((got_a == "RECOVERABLE") & (true_a == "SOLID"))
                ),
                "stuck_misread_as_recoverable": int(
                    np.sum((got_a == "RECOVERABLE") & (true_a == "STUCK"))
                ),
            }
        )
    return _frame(rows)


def runs_needed_study(
    target_accuracy: float = 0.85,
    ks: Sequence[int] = (2, 3, 5, 8, 12, 16, 24, 32),
    n_tasks: int = 300,
    seed: int = 0,
) -> dict:
    """Smallest ``k`` reaching ``target_accuracy`` on decided tasks. The planning number."""
    df = classification_study(ks=ks, n_tasks=n_tasks, seed=seed)
    ok = df[df["accuracy_excl_uncertain"] >= target_accuracy]
    return {
        "target_accuracy": target_accuracy,
        "runs_needed": int(ok["k"].iloc[0]) if len(ok) else None,
        "uncertain_rate_there": float(ok["uncertain_rate"].iloc[0]) if len(ok) else float("nan"),
        "curve": df.to_dict("records"),
        "note": "" if len(ok) else f"no tested k reached {target_accuracy}",
    }


def adaptive_saving_study(
    n_tasks: int = 120,
    budgets_per_task: Sequence[int] = (4, 8, 16),
    seed: int = 0,
):
    """Executions used and label quality, adaptive vs uniform, against ground truth.

    Compared against the *true* labels, not against the uniform baseline's labels
    -- otherwise the study would only measure how well adaptive imitates uniform.
    """
    w = SimulatedWorld(WorldConfig(n_tasks=n_tasks, seed=seed))
    rng = np.random.default_rng(seed + 5)
    tasks = w.tasks()
    ref = {t: v.true_label for t, v in w.truth.items() if v.true_label != "MARGINAL"}

    def run_fn(task: str, i: int) -> float:
        return float(w.run(task, rng)["score"])

    th = ReliabilityThresholds()
    frames = []
    for b in budgets_per_task:
        df = compare_allocation(
            tasks, run_fn, ref,
            policies={"uniform": UniformAllocation(th), "greedy_voi": GreedyValuePolicy(thresholds=th)},
            budget=b * n_tasks, warmup=2, thresholds=th, seed=seed,
        )
        df["budget_per_task"] = b
        frames.append(df)
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def decomposition_study(
    ks: Sequence[int] = (8, 16, 32),
    n_tasks: int = 200,
    violation_rates: Sequence[float] = (0.0, 0.15),
    seed: int = 0,
):
    """Does the q/r decomposition recover the truth, and what breaks it?

    Run with the necessity assumption intact and deliberately violated. Under
    violation the estimate should degrade and ``valid`` should be False -- if the
    estimator looks fine when its assumption is broken, the assumption was not
    doing any work and the model is not what it claims to be.
    """
    from ..reliability.decomposition import decompose, milestone_from_rubric

    rows = []
    milestone = milestone_from_rubric(["milestone"])
    for vr in violation_rates:
        w = SimulatedWorld(
            WorldConfig(n_tasks=n_tasks, necessity_violation_rate=vr, seed=seed)
        )
        rng = np.random.default_rng(seed + 6)
        for k in ks:
            q_err, r_err, invalid = [], [], 0
            for task, truth in w.truth.items():
                trajs = [w.trajectory(task, i, rng) for i in range(k)]
                d = decompose(trajs, milestone, task=task)
                q_err.append(d.q_mean - truth.q)
                if d.r_estimable:
                    r_err.append(d.r_mean - truth.r)
                invalid += int(not d.valid)
            rows.append(
                {
                    "necessity_violation_rate": vr,
                    "k": k,
                    "q_bias": float(np.mean(q_err)),
                    "q_rmse": float(np.sqrt(np.mean(np.square(q_err)))),
                    "r_bias": float(np.mean(r_err)) if r_err else float("nan"),
                    "r_rmse": float(np.sqrt(np.mean(np.square(r_err)))) if r_err else float("nan"),
                    "pct_flagged_invalid": invalid / n_tasks,
                    "pct_r_estimable": len(r_err) / n_tasks,
                }
            )
    return _frame(rows)


def selection_study(
    n_tasks: int = 300,
    k: int = 8,
    n_select: int = 40,
    couplings: Sequence[str] = BENEFIT_COUPLINGS,
    strategies: Sequence[str] = (
        "random", "hardest", "lowest_mean", "highest_variance",
        "success_failure", "recoverability", "uncertainty_aware",
    ),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    seed: Optional[int] = None,
):
    """The central study: does recoverability selection capture more true benefit?

    For each benefit-coupling regime, every selector picks ``n_select`` tasks from
    the same evaluation data, and the study reports the mean *true* ``g_t`` of what
    each picked, normalised against the oracle's ceiling.

    Read the ``none`` and ``adversarial`` rows first. If recoverability selection
    beats random there, something is wrong -- benefit is by construction unrelated
    (or anti-related) to recoverability in those worlds, and no method should be
    able to exploit it.

    **Replicated across ``seeds`` and reported as mean +/- sd.** A single-seed run
    of this study is not interpretable: with 300 tasks the sampling noise on
    ``oracle_fraction`` is large enough that a method can appear to beat random in
    the ``none`` world by chance. The standard deviation column is what says
    whether a gap is real, and the study is deliberately structured so that
    reading a single seed is inconvenient.
    """
    if seed is not None:
        seeds = (seed,)
    rows = []
    for coupling in couplings:
      for sd_seed in seeds:
        w = SimulatedWorld(
            WorldConfig(n_tasks=n_tasks, benefit_coupling=coupling, seed=sd_seed)
        )
        rng = np.random.default_rng(sd_seed + 7)
        diags = []
        for task in w.tasks():
            s = w.scores(task, k, rng)
            diags.append(
                diagnose(binary_posterior(int(sum(s)), k), task=task, model="sim", scores=s)
            )
        benefit = w.true_benefit()
        oracle = make_selector("oracle", benefit=benefit).select(diags, n_select)
        ceiling = float(np.mean([benefit[t] for t in oracle.tasks]))
        floor = float(np.mean(list(benefit.values())))

        for name in strategies:
            res = make_selector(name, seed=sd_seed).select(diags, n_select)
            if not res.tasks:
                continue
            got = float(np.mean([benefit[t] for t in res.tasks]))
            rows.append(
                {
                    "coupling": coupling,
                    "seed": sd_seed,
                    "strategy": name,
                    "mean_true_benefit": got,
                    "vs_random_pct": 100.0 * (got - floor) / max(floor, 1e-9),
                    "oracle_fraction": (got - floor) / max(ceiling - floor, 1e-9),
                    "n_selected": res.delivered,
                    "shortfall": res.shortfall,
                }
            )
        rows.append(
            {
                "coupling": coupling, "seed": sd_seed, "strategy": "oracle",
                "mean_true_benefit": ceiling,
                "vs_random_pct": 100.0 * (ceiling - floor) / max(floor, 1e-9),
                "oracle_fraction": 1.0, "n_selected": oracle.delivered, "shortfall": 0,
            }
        )
    df = _frame(rows)
    agg = (
        df.groupby(["coupling", "strategy"])
        .agg(
            oracle_fraction=("oracle_fraction", "mean"),
            oracle_fraction_sd=("oracle_fraction", "std"),
            mean_true_benefit=("mean_true_benefit", "mean"),
            vs_random_pct=("vs_random_pct", "mean"),
            n_seeds=("seed", "count"),
            mean_shortfall=("shortfall", "mean"),
        )
        .reset_index()
    )
    # A gap is only reported as real when it exceeds the seed-to-seed noise.
    agg["se"] = agg["oracle_fraction_sd"] / np.sqrt(agg["n_seeds"].clip(lower=1))
    agg.attrs["per_seed"] = df
    return agg


def run_all_studies(seed: int = 0, quick: bool = False) -> dict:
    """Run the whole validation suite. ``quick`` shrinks it for CI."""
    n = 120 if quick else 400
    ks = (2, 4, 8) if quick else (2, 3, 5, 8, 16, 32)
    return {
        "max_bias": max_bias_study(ks=ks, n_tasks=n, seed=seed),
        "posterior_recovery": posterior_recovery_study(ks=ks, n_tasks=n, seed=seed),
        "shrinkage": shrinkage_study(
            ks=(2, 4, 8) if quick else (2, 3, 5, 8, 16), n_tasks=100 if quick else 150, seed=seed
        ),
        "classification": classification_study(ks=ks, n_tasks=n, seed=seed),
        "decomposition": decomposition_study(
            ks=(8,) if quick else (8, 16, 32), n_tasks=60 if quick else 200, seed=seed
        ),
        "selection": selection_study(
            n_tasks=120 if quick else 300,
            n_select=20 if quick else 40,
            seeds=(seed, seed + 1) if quick else tuple(range(seed, seed + 5)),
        ),
        "band_label_noise": band_label_noise_study(),
        "adaptive": adaptive_saving_study(
            n_tasks=60 if quick else 120,
            budgets_per_task=(4, 8) if quick else (4, 8, 16),
            seed=seed,
        ),
    }


def band_label_noise_study(
    bands: Optional[Sequence] = None,
    ks: Sequence[int] = (3, 8, 16, 24, 32, 48, 64),
    seed: int = 0,
):
    """How noisy is a plug-in frontier-band label, and what does it cost?

    Generate-and-filter pipelines for synthetic tasks label a candidate as
    "at the learnable frontier" by running it ``K`` times and checking whether
    ``k/K`` lands in a band. This study computes, exactly from the Binomial pmf
    (no simulation), how often that label is wrong, and the smallest ``K`` at
    which a task at the band's centre becomes resolvable.

    The result is not that such pipelines are broken. Label noise is symmetric
    across arms, so a *relative* comparison between two generators sharing one
    labeller remains meaningful -- though attenuated, in the regression-dilution
    sense, meaning reported improvements understate true ones. What the noise
    does bias is the *absolute* frontier rate, and it sets a floor on how small a
    difference the metric can resolve at a given ``K``.
    """
    from ..reliability.frontier import (
        MATH_CODE_BAND,
        SWE_BAND,
        band_decision,
        plugin_band_error,
    )
    from ..reliability.posterior import binary_posterior

    bands = list(bands) if bands is not None else [MATH_CODE_BAND, SWE_BAND]
    rows = []
    for band in bands:
        centre = 0.5 * (band.lo + band.hi)
        for k in ks:
            err = plugin_band_error(band, k)
            in_truth = err[err["in_band_truth"]]
            out_truth = err[~err["in_band_truth"]]
            # Can any observed count resolve "in band" at 80% confidence?
            resolvable = any(
                band_decision(binary_posterior(s, k), band).label == "in_band"
                for s in range(k + 1)
            )
            centre_dec = band_decision(binary_posterior(round(centre * k), k), band)
            rows.append({
                "band": band.name or f"[{band.lo:.2f},{band.hi:.2f}]",
                "band_width": band.width,
                "K": k,
                "mean_miss_rate": float(in_truth["p_mislabelled"].mean()),
                "mean_false_positive_rate": float(out_truth["p_mislabelled"].mean()),
                "worst_false_positive_rate": float(out_truth["p_mislabelled"].max()),
                "in_band_resolvable_at_80pct": bool(resolvable),
                "p_in_at_band_centre": float(centre_dec.p_in),
            })
    return _frame(rows)
