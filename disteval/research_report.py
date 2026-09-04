"""The end-to-end research report: evaluation, diagnosis, and what to train on.

One call takes runs (and optionally tasks and trajectories) and produces a
Markdown/HTML report a researcher can read in a few minutes: aggregate metrics,
per-task posteriors with intervals and classifications, the recoverability
ranking, the preference pairs available, and the figures.

Three rules the report follows, because a report is where overclaiming happens:

1. **Every section names what it assumes.** Metric definitions and assumptions
   come from :mod:`disteval.metrics_spec`, so a number cannot appear without them.
2. **Missing analyses are reported as missing.** ``describe_dataset`` says which
   analyses the data supports; the rest are listed as skipped with the field that
   was absent, so "no failure analysis" never reads as "no failures".
3. **Uncertainty and ties are surfaced, not smoothed.** The ranking carries
   credible intervals and the tie diagnostic, and the report states plainly when
   the top-k is partly arbitrary.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from . import metrics as M
from . import metrics_spec, plots
from .diagnosis.entropy import failure_distribution
from .diagnosis.survival import survival_summary
from .loaders import TaskSpec, describe_dataset
from .reliability.classify import (
    RECOVERABLE,
    ReliabilityThresholds,
    diagnose,
    rank_by_recoverability,
    tie_diagnostics,
)
from .reliability.hierarchical import fit_hierarchical, pooling_diagnostic
from .reliability.posterior import JEFFREYS_PRIOR, posterior_from_scores
from .trajectory.counterfactual import intervention_distance
from .trajectory.embed import embed_trajectories, neighbourhood_distance
from .trajectory.events import Trajectory, TrajectorySet

__all__ = ["ReportData", "build_report_data", "render_markdown", "render_html", "generate_report"]


@dataclass
class ReportData:
    """Everything the report renders. Computed once, rendered to any format."""

    dataset: dict
    aggregate: dict
    diagnoses: list
    ranking: list
    ties: dict
    per_task_frame: object
    failure: dict = field(default_factory=dict)
    rubric: dict = field(default_factory=dict)
    survival: dict = field(default_factory=dict)
    trajectory: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    pairs: dict = field(default_factory=dict)
    hierarchical: Optional[dict] = None
    pooling: Optional[dict] = None
    skipped: dict[str, str] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    thresholds: ReliabilityThresholds = field(default_factory=ReliabilityThresholds)
    title: str = "Agent reliability report"


def build_report_data(
    runs: Sequence[Mapping],
    *,
    tasks: Optional[Mapping[str, TaskSpec]] = None,
    trajectories: Optional[TrajectorySet] = None,
    thresholds: Optional[ReliabilityThresholds] = None,
    hierarchical: str | bool = "auto",
    title: str = "Agent reliability report",
) -> ReportData:
    """Compute every section. Sections without data are recorded in ``skipped``.

    ``hierarchical`` may be ``True``, ``False`` or ``"auto"`` (the default).
    ``"auto"`` runs :func:`~disteval.reliability.hierarchical.pooling_diagnostic`
    and pools only when partial pooling predicts held-out runs better on this
    dataset. Pooling is not universally better -- it shrinks extremes toward the
    population, which helps on a homogeneous suite and hurts on a suite of
    mostly-similar tasks plus a few outliers -- so the choice is made empirically
    and the decision is reported.
    """
    import pandas as pd

    th = thresholds or ReliabilityThresholds()
    desc = describe_dataset(runs, tasks, trajectories)
    enabled = desc["enabled_analyses"]
    skipped: dict[str, str] = {}

    scores_by_task: dict[str, list[float]] = {}
    domains: dict[str, str] = {}
    for r in runs:
        scores_by_task.setdefault(r["task"], []).append(float(r["score"]))
        if r.get("domain"):
            domains[r["task"]] = r["domain"]

    df = pd.DataFrame(
        [{"task": r["task"], "score": float(r["score"]), "success": bool(r["success"])}
         for r in runs]
    )
    flat = df["score"].to_numpy(float)
    min_runs = min((len(v) for v in scores_by_task.values()), default=1)
    ks = [k for k in (1, 2, 4, 8) if k <= max(len(v) for v in scores_by_task.values())]
    aggregate = {
        "n_runs": len(runs),
        "n_tasks": len(scores_by_task),
        "mean": float(flat.mean()),
        "iqm": M.iqm(flat),
        "median": float(np.median(flat)),
        "std": float(flat.std(ddof=1)) if flat.size > 1 else 0.0,
        "lower_cvar_0.2": M.lower_cvar(flat, 0.2),
        "lower_cvar_0.2_ci": M.bootstrap_ci(flat),
        "worst_case": M.worst_case(flat),
        **{f"pass@{k}": M.pass_at_k(df, k) for k in ks},
        **{f"pass^{k}": M.pass_hat_k(df, k) for k in ks},
        "reliability_gap@max_k": M.reliability_gap(df, max(ks)) if ks else float("nan"),
    }

    # -- posteriors ---------------------------------------------------------
    hier_summary = None
    posts = {}
    shrink = {}
    pooling: Optional[dict] = None
    use_hier = bool(hierarchical) and enabled["hierarchical_pooling"]
    if hierarchical == "auto" and enabled["hierarchical_pooling"]:
        names = list(scores_by_task)
        pooling = pooling_diagnostic(
            ["policy"] * len(names), names,
            [float(sum(scores_by_task[t])) for t in names],
            [len(scores_by_task[t]) for t in names],
            [domains.get(t, "_all") for t in names],
        )
        use_hier = bool(pooling["pooling_helps"])
        if not use_hier:
            skipped["hierarchical_pooling"] = (
                "not used: " + pooling["recommendation"]
                + f" (LOO log-likelihood per run {pooling['delta_per_run']:+.4f})"
            )
    if use_hier:
        names = list(scores_by_task)
        fit = fit_hierarchical(
            ["policy"] * len(names), names,
            [float(sum(scores_by_task[t])) for t in names],
            [len(scores_by_task[t]) for t in names],
            [domains.get(t, "_all") for t in names],
        )
        hier_summary = fit.summary()
        posts = {t: fit.cell("policy", t) for t in names}
        shrink = {t: fit.shrinkage("policy", t) for t in names}
    else:
        if hierarchical and not enabled["hierarchical_pooling"]:
            skipped["hierarchical_pooling"] = (
                "fewer than 4 tasks; partial pooling needs a population to pool across"
            )
        posts = {
            t: posterior_from_scores(v, JEFFREYS_PRIOR) for t, v in scores_by_task.items()
        }

    diagnoses = [
        diagnose(posts[t], task=t, model=runs[0]["model"], thresholds=th,
                 scores=scores_by_task[t], domain=domains.get(t),
                 shrinkage=shrink.get(t))
        for t in sorted(posts)
    ]
    ranking = rank_by_recoverability(diagnoses, labels=None)
    ties = tie_diagnostics(diagnoses)
    per_task = pd.DataFrame([d.to_dict() for d in diagnoses])

    # -- failure modes ------------------------------------------------------
    failure: dict = {}
    modes_by_task: dict[str, list[str]] = {}
    for r in runs:
        if r.get("failure_mode"):
            modes_by_task.setdefault(r["task"], []).append(r["failure_mode"])
    if modes_by_task:
        dists = {
            t: failure_distribution(m, task=t) for t, m in modes_by_task.items()
        }
        failure = {
            "per_task": {t: d.to_dict() for t, d in dists.items()},
            "distributions": list(dists.values()),
            "most_concentrated": sorted(
                (d for d in dists.values() if d.n_failures >= 3),
                key=lambda d: -(d.concentration if np.isfinite(d.concentration) else -1),
            )[:5],
        }
    else:
        skipped["failure_entropy"] = "no run carried a failure_mode label"

    # -- rubric -------------------------------------------------------------
    rubric: dict = {}
    if enabled["rubric_reliability"]:
        from .diagnosis.causality import CriterionGraph
        from .reliability.rubric import profile_rubric, root_criterion_profile, rubric_matrix

        by_task: dict[str, list[dict]] = {}
        for r in runs:
            if r.get("rubric_scores"):
                by_task.setdefault(r["task"], []).append(r["rubric_scores"])
        deps = []
        if tasks:
            for t in tasks.values():
                deps.extend(t.criterion_dependencies)
        cg = CriterionGraph.from_pairs(deps) if deps else None
        profiles = [
            profile_rubric(v, task=t, model=runs[0]["model"], thresholds=th,
                           criterion_graph=cg)
            for t, v in by_task.items()
        ]
        rubric = {
            "profiles": [p.to_dict() for p in profiles],
            "matrix": rubric_matrix(profiles),
            "root_causes": {p.task: root_criterion_profile(p) for p in profiles},
            "has_dependencies": cg is not None,
        }
    else:
        skipped["rubric_reliability"] = "no run carried per-criterion rubric_scores"

    # -- trajectory-derived -------------------------------------------------
    trajectory: dict = {}
    survival: dict = {}
    pairs: dict = {}
    if trajectories and len(trajectories):
        grouped = trajectories.group()
        neigh, interv = {}, {}
        for (_model, task), runs_t in grouped.items():
            succ = [t for t in runs_t if t.success]
            fail = [t for t in runs_t if not t.success]
            if len(runs_t) >= 4 and succ and fail:
                try:
                    emb = embed_trajectories(runs_t)
                    nd = neighbourhood_distance(emb)
                    if np.isfinite(nd.get("normalized_distance", np.nan)):
                        neigh[task] = float(nd["normalized_distance"])
                except Exception:
                    pass
            if succ and fail:
                iv = intervention_distance(fail[:4], succ[:4])
                if np.isfinite(iv.get("mean_normalized_cost", np.nan)):
                    interv[task] = iv
        trajectory = {
            "neighbourhood": neigh,
            "intervention": {k: v["mean_normalized_cost"] for k, v in interv.items()},
            "intervention_detail": interv,
            "n_tasks_with_pairs": sum(
                1 for v in grouped.values()
                if any(t.success for t in v) and any(not t.success for t in v)
            ),
        }
        survival = survival_summary(list(trajectories))
        pairs = {
            task: {
                "n_success": sum(1 for t in v if t.success),
                "n_failure": sum(1 for t in v if not t.success),
                "max_pairs": sum(1 for t in v if t.success) * sum(1 for t in v if not t.success),
            }
            for (_m, task), v in grouped.items()
        }
    else:
        for k in ("trajectory_divergence", "counterfactual_intervention",
                  "trajectory_embedding", "survival_analysis", "preference_pairs"):
            skipped[k] = "no trajectory event logs supplied"

    # -- cost ---------------------------------------------------------------
    cost: dict = {}
    if enabled["cost_analysis"]:
        from .reliability.cost import cost_profile

        c = [float((r.get("cost") or {}).get("usd", 0.0)) for r in runs]
        y = [float(r["success"]) for r in runs]
        if any(c):
            cost = cost_profile(c, y, "usd", runs[0]["model"]).to_dict()
    else:
        skipped["cost_analysis"] = "no run carried cost information"

    return ReportData(
        dataset=desc,
        aggregate=aggregate,
        diagnoses=diagnoses,
        ranking=ranking,
        ties=ties,
        per_task_frame=per_task,
        failure=failure,
        rubric=rubric,
        survival=survival,
        trajectory=trajectory,
        cost=cost,
        pairs=pairs,
        hierarchical=hier_summary,
        pooling=pooling,
        skipped=skipped,
        thresholds=th,
        title=title,
    )


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def build_figures(
    data: ReportData,
    runs: Sequence[Mapping],
    directory: str,
    fmt: str = "png",
    trajectories: Optional[TrajectorySet] = None,
) -> dict[str, str]:
    """Render every figure the data supports. Failures are skipped, not fatal."""
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    figs = {}
    df = pd.DataFrame(
        [{"task": r["task"], "score": float(r["score"]), "success": bool(r["success"])}
         for r in runs]
    )
    builders = {
        "capability_reliability": lambda: plots.capability_reliability_map(data.diagnoses),
        "recoverability_ranking": lambda: plots.recoverability_ranking(data.diagnoses),
        "pass_at_k_vs_pass_hat_k": lambda: plots.pass_at_k_vs_pass_hat_k(df),
        "score_distribution": lambda: plots.score_distributions(
            {runs[0]["model"]: df["score"].tolist()}
        ),
    }
    if data.failure.get("distributions"):
        builders["failure_entropy"] = lambda: plots.failure_entropy_plot(
            data.failure["distributions"]
        )
    if data.rubric.get("matrix") is not None and len(data.rubric["matrix"]):
        builders["rubric_reliability"] = lambda: plots.rubric_reliability_matrix(
            data.rubric["matrix"]
        )
    if trajectories and len(trajectories) >= 6:
        def _emb():
            return plots.trajectory_embedding_plot(embed_trajectories(list(trajectories)))

        builders["trajectory_embedding"] = _emb
    if data.survival:
        from .diagnosis.survival import kaplan_meier, run_survival_record

        def _surv():
            recs = [run_survival_record(t) for t in trajectories]
            return plots.survival_curves({runs[0]["model"]: kaplan_meier(recs)})

        builders["survival"] = _surv

    built = {}
    for name, fn in builders.items():
        try:
            built[name] = fn()
        except Exception as exc:  # a broken figure must not lose the report
            data.skipped[f"figure:{name}"] = f"{type(exc).__name__}: {exc}"
    figs = plots.save_all(built, directory, fmt)
    data.figures = figs
    return figs


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3) -> str:
    if isinstance(v, float):
        return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"
    if isinstance(v, tuple):
        return f"[{_fmt(v[0])}, {_fmt(v[1])}]"
    return str(v)


def _table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    return "\n".join(out)


def render_markdown(data: ReportData, figure_dir: str = "figures") -> str:
    """Render the report as Markdown."""
    th = data.thresholds
    d = data.dataset
    L = [f"# {data.title}", ""]

    L += [
        "> Generated by `disteval`. Every metric below is defined, with its "
        "assumptions and edge cases, in the metric reference at the end.",
        "",
        "## 1. Dataset",
        "",
        _table(
            ["property", "value"],
            [
                ["runs", d["n_runs"]], ["tasks", d["n_tasks"]],
                ["models", d["n_models"]], ["domains", d["n_domains"]],
                ["runs per task", f"{d['runs_per_task_min']}-{d['runs_per_task_max']} "
                                 f"(mean {d['runs_per_task_mean']:.1f})"],
                ["unequal run counts", d["unequal_run_counts"]],
                ["rubric scores", d["has_rubric_scores"]],
                ["trajectories", d["has_trajectories"]],
                ["cost data", d["has_cost"]],
            ],
        ),
        "",
    ]
    if data.skipped:
        L += ["**Analyses skipped for lack of data** (absent, not empty):", ""]
        L += [f"- `{k}` — {v}" for k, v in sorted(data.skipped.items())]
        L += [""]

    # -- aggregate ----------------------------------------------------------
    a = data.aggregate
    L += [
        "## 2. Aggregate evaluation",
        "",
        _table(
            ["metric", "value", "reading"],
            [
                ["mean", a["mean"], "average run score"],
                ["IQM", a["iqm"], "middle 50%, robust to outliers"],
                ["median", a["median"], ""],
                ["std", a["std"], "run-to-run spread"],
                ["lower-tail CVaR (α=0.2)", a["lower_cvar_0.2"],
                 f"worst-20% mean, 95% CI {_fmt(a['lower_cvar_0.2_ci'])}"],
                ["worst run", a["worst_case"], "maximally pessimistic"],
            ]
            + [[k, v, "at least one of k succeeds"] for k, v in a.items() if k.startswith("pass@")]
            + [[k, v, "all k succeed"] for k, v in a.items() if k.startswith("pass^")]
            + [["reliability gap", a["reliability_gap@max_k"],
                "apparent capability that does not reproduce"]],
        ),
        "",
    ]
    gap = a.get("reliability_gap@max_k", float("nan"))
    if np.isfinite(gap) and gap > 0.2:
        L += [
            f"The reliability gap of **{gap:.2f}** is the headline finding: most of "
            "what this agent can *sometimes* do, it cannot do *consistently*. That "
            "gap, not the mean, is what the rest of this report analyses.",
            "",
        ]

    if data.pooling:
        pl = data.pooling
        L += [
            "### Pooling decision",
            "",
            f"Partial pooling was **{'used' if pl['pooling_helps'] else 'not used'}**. "
            f"Leave-one-run-out predictive log-likelihood per run: "
            f"{-pl['loo_logloss_independent']:.4f} independent vs "
            f"{-pl['loo_logloss_pooled']:.4f} pooled "
            f"({pl['delta_per_run']:+.4f}). {pl['recommendation'].capitalize()}.",
            "",
            f"Fitted between-task sd {pl['sigma_task']:.3f} on the logit scale; "
            f"mean shrinkage {pl['mean_shrinkage']:.2f}. {pl['note'].capitalize()}.",
            "",
        ]
    if data.hierarchical:
        h = data.hierarchical
        L += [
            "### Hierarchical model",
            "",
            f"`logit(p) = mu + alpha_model + beta_task + gamma_domain + eps` fitted by "
            f"`{h['backend']}` ({'converged' if h['converged'] else 'hit the iteration cap'} "
            f"in {h['n_iter']} iterations). Global rate {h['global_rate']:.3f}. "
            f"Variance components (logit scale): "
            + ", ".join(f"`{k}` {v:.3f}" for k, v in h["sigma"].items())
            + ".",
            "",
            "A large `task` component means tasks genuinely differ and little pooling "
            "occurs; a large `resid` component means agent-by-task interaction beyond "
            "the main effects, which is where recoverable tasks live.",
            "",
        ]

    # -- task-level ---------------------------------------------------------
    counts: dict[str, int] = {}
    for x in data.diagnoses:
        counts[x.label] = counts.get(x.label, 0) + 1
    L += [
        "## 3. Task-level analysis",
        "",
        f"Thresholds: `tau_cap={th.tau_cap}`, `tau_rel={th.tau_rel}`, "
        f"`tau_stuck={th.tau_stuck}`, confidence `{th.confidence}`, "
        f"`min_runs={th.min_runs}`. These are choices, not constants; every label "
        "below is conditional on them.",
        "",
        _table(["classification", "tasks", "meaning"],
               [["SOLID", counts.get("SOLID", 0), "capable and reliable"],
                ["RECOVERABLE", counts.get("RECOVERABLE", 0), "capable, not reliable"],
                ["STUCK", counts.get("STUCK", 0), "no evidence of capability"],
                ["UNCERTAIN", counts.get("UNCERTAIN", 0), "too few runs to say"]]),
        "",
    ]
    rows = []
    for x in sorted(data.diagnoses, key=lambda z: (-z.recoverability, z.task))[:30]:
        rows.append([
            x.task, x.domain or "-", f"{x.n_success:.0f}/{x.n_runs}",
            x.posterior_mean, f"[{x.ci_lo:.2f}, {x.ci_hi:.2f}]",
            x.capability, x.reliability, x.label, x.recoverability,
        ])
    L += [
        _table(
            ["task", "domain", "successes", "post. mean", "95% CI", "C_t", "R_t",
             "label", "recoverability"],
            rows,
        ),
        "",
        "`C_t = P(p_t > tau_cap)`, `R_t = P(p_t > tau_rel)`. The observed maximum is "
        "deliberately absent: it is a descriptive statistic, not a capability "
        "estimate (see THEORY.md).",
        "",
    ]

    # -- ranking / ties -----------------------------------------------------
    t = data.ties
    L += ["## 4. Training-data recommendation", ""]
    top = [x for x in data.ranking if x.label == RECOVERABLE][:10]
    if top:
        rows = []
        for x in top:
            p = data.pairs.get(x.task, {})
            rows.append([
                x.task, x.domain or "-", x.recoverability,
                f"[{x.ci_lo:.2f}, {x.ci_hi:.2f}]",
                p.get("n_success", "?"), p.get("n_failure", "?"),
                _fmt(data.trajectory.get("intervention", {}).get(x.task, float("nan"))),
            ])
        L += [
            _table(
                ["task", "domain", "recoverability", "95% CI", "successes",
                 "failures", "intervention cost"],
                rows,
            ),
            "",
            "Each of these has demonstrated capability with posterior evidence, "
            "remains unreliable, and has both successful and failed trajectories "
            "available to pair. Lower intervention cost means the failed runs are "
            "closer to the successful ones in edit distance.",
            "",
        ]
    else:
        L += ["No task met the RECOVERABLE criteria at these thresholds.", ""]

    if t.get("cutoff_is_arbitrary"):
        L += [
            "> **The ranking is only partly resolved.** "
            f"{t['n_distinct_scores']} distinct scores across {t['n_tasks']} tasks "
            f"(largest tie: {t['largest_tier_size']} tasks). {t['note']}",
            "",
        ]

    # -- failure / rubric / survival / cost ---------------------------------
    if data.failure:
        L += ["## 5. Failure structure", ""]
        rows = [
            [d.task, d.n_failures, d.dominant_mode, d.dominant_share, d.concentration]
            for d in data.failure["most_concentrated"]
        ]
        L += [
            "Most concentrated failure profiles — one repeated defect rather than "
            "diffuse incompetence, and therefore the most plausible targets for a "
            "single corrective signal:",
            "",
            _table(["task", "failures", "dominant mode", "share", "concentration"], rows),
            "",
        ]
    if data.rubric:
        L += ["## 6. Rubric-level reliability", ""]
        amps = [
            (k, v["amplification"]) for k, v in data.rubric["root_causes"].items()
            if np.isfinite(v.get("amplification", np.nan)) and v["amplification"] > 1
        ]
        L += [
            f"Per-criterion posteriors computed for {len(data.rubric['profiles'])} tasks."
            + (
                f" Criterion dependencies were supplied, and {len(amps)} task(s) show "
                "downstream amplification — one root failure counted as several "
                "criterion failures."
                if data.rubric["has_dependencies"] else
                " No criterion dependency graph was supplied, so every failing "
                "criterion is treated as its own root cause and the amplification "
                "factor is 1.0 by construction."
            ),
            "",
        ]
        if amps:
            L += [_table(["task", "amplification"], sorted(amps, key=lambda x: -x[1])[:8]), ""]
    if data.survival:
        s = data.survival
        L += [
            "## 7. Sequential reliability",
            "",
            _table(
                ["quantity", "value"],
                [
                    ["runs", s["n_runs"]],
                    ["irrecoverable failures", s["n_irrecoverable"]],
                    ["median survival step", s["median_survival_step"]],
                    ["final survival", s["final_survival"]],
                    ["median time to first error", s["time_to_first_error"]["median"]],
                    ["P(success | an error occurred)", s["recovery"]["p_success_given_error"]],
                    ["P(success | no error)", s["recovery"]["p_success_given_no_error"]],
                    ["cost of an error", s["recovery"]["error_cost"]],
                ],
            ),
            "",
        ]
    if data.cost:
        c = data.cost
        L += [
            "## 8. Cost",
            "",
            _table(
                ["quantity", "value"],
                [
                    ["mean cost/run (usd)", c["mean_cost"]],
                    ["p90 cost/run", c["p90_cost"]],
                    ["cost | success", c["mean_cost_given_success"]],
                    ["cost | failure", c["mean_cost_given_failure"]],
                    ["failure/success cost ratio", c["cost_ratio_failure_to_success"]],
                    ["success per dollar", c["success_per_cost"]],
                ],
            ),
            "",
        ]

    # -- figures ------------------------------------------------------------
    if data.figures:
        L += ["## 9. Figures", ""]
        for name, path in sorted(data.figures.items()):
            rel = f"{figure_dir}/{Path(path).name}"
            L += [f"### {name.replace('_', ' ')}", "", f"![{name}]({rel})", ""]

    # -- metric reference ---------------------------------------------------
    used = [
        n for n in ("mean", "iqm", "pass@k", "pass^k", "reliability_gap", "lower_cvar",
                    "posterior_mean", "capability", "reliability",
                    "recoverability_headroom", "failure_entropy", "hazard")
        if n in metrics_spec.REGISTRY
    ]
    L += [
        "## 10. Metric reference",
        "",
        "Definitions, assumptions, edge cases and prior work for every metric used "
        "above.",
        "",
        metrics_spec.to_markdown(used),
        "",
        "---",
        "",
        "### Scope of these results",
        "",
        "This report characterises one agent's run distribution on this dataset. It "
        "does not establish that training on the recommended tasks improves the "
        "agent — that is the hypothesis this framework exists to test, and testing "
        "it requires the selection experiment (`disteval experiment`) plus an "
        "actual training run. Nothing here is a benchmark result.",
        "",
    ]
    return "\n".join(L)


def render_html(data: ReportData, markdown_text: Optional[str] = None,
                figure_dir: str = "figures") -> str:
    """Render a self-contained HTML page (minimal Markdown subset, no dependency)."""
    md = markdown_text or render_markdown(data, figure_dir)
    body: list[str] = []
    in_table = False
    for line in md.split("\n"):
        s = line.rstrip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                body.append("<table>")
                in_table = True
            body.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if s.startswith("!["):
            alt, _, rest = s[2:].partition("](")
            body.append(f'<img src="{html.escape(rest.rstrip(")"))}" alt="{html.escape(alt)}">')
        elif s.startswith("#"):
            lvl = len(s) - len(s.lstrip("#"))
            body.append(f"<h{lvl}>{html.escape(s.lstrip('# '))}</h{lvl}>")
        elif s.startswith("> "):
            body.append(f"<blockquote>{html.escape(s[2:])}</blockquote>")
        elif s.startswith("- "):
            body.append(f"<li>{html.escape(s[2:])}</li>")
        elif s == "---":
            body.append("<hr>")
        elif s:
            body.append(f"<p>{html.escape(s)}</p>")
    if in_table:
        body.append("</table>")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(data.title)}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        max-width: 60rem; margin: 2rem auto; padding: 0 1.25rem; }}
h1 {{ border-bottom: 2px solid currentColor; padding-bottom: .3rem; }}
h2 {{ margin-top: 2.2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9em;
         display: block; overflow-x: auto; }}
th, td {{ border: 1px solid #8884; padding: .35rem .6rem; text-align: left; }}
th {{ background: #8881; font-weight: 600; }}
img {{ max-width: 100%; height: auto; margin: 1rem 0; }}
blockquote {{ border-left: 3px solid #E69F00; margin: 1rem 0; padding: .5rem 1rem;
              background: #E69F0018; }}
code {{ background: #8881; padding: .1rem .3rem; border-radius: 3px; }}
</style></head><body>
{chr(10).join(body)}
</body></html>"""


def generate_report(
    runs: Sequence[Mapping],
    output_dir: str,
    *,
    tasks: Optional[Mapping[str, TaskSpec]] = None,
    trajectories: Optional[TrajectorySet] = None,
    thresholds: Optional[ReliabilityThresholds] = None,
    title: str = "Agent reliability report",
    figure_format: str = "png",
    html_output: bool = True,
    hierarchical: str | bool = "auto",
) -> dict[str, str]:
    """Compute, render and write the full report. Returns the paths written."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_report_data(
        runs, tasks=tasks, trajectories=trajectories, thresholds=thresholds,
        title=title, hierarchical=hierarchical,
    )
    build_figures(data, runs, str(out / "figures"), figure_format, trajectories)

    md = render_markdown(data)
    paths = {"markdown": str(out / "report.md")}
    (out / "report.md").write_text(md, encoding="utf-8")
    if html_output:
        (out / "report.html").write_text(render_html(data, md), encoding="utf-8")
        paths["html"] = str(out / "report.html")
    data.per_task_frame.to_csv(out / "task_metrics.csv", index=False)
    paths["task_metrics"] = str(out / "task_metrics.csv")
    try:
        data.per_task_frame.to_parquet(out / "task_metrics.parquet", index=False)
        paths["task_metrics_parquet"] = str(out / "task_metrics.parquet")
    except Exception:
        pass
    (out / "summary.json").write_text(
        json.dumps(
            {"dataset": data.dataset, "aggregate": data.aggregate,
             "ties": data.ties, "hierarchical": data.hierarchical,
             "pooling": data.pooling, "skipped": data.skipped},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    paths["summary"] = str(out / "summary.json")
    paths.update({f"figure:{k}": v for k, v in data.figures.items()})
    return paths
