"""The evaluate -> diagnose -> select -> train -> re-evaluate loop.

This is the experiment the repository exists to run:

**Phase A -- Evaluation.** Run each task ``k`` times (uniformly, or adaptively
under a budget). Estimate ``p_{m,t}`` with uncertainty, hierarchically pooled if
configured. Compute pass@k, pass^k, mean, lower-tail CVaR.

**Phase B -- Diagnosis and selection.** Classify tasks, score recoverability
(optionally with trajectory-derived signals), and select an equal-size training
set under each strategy being compared.

**Phase C -- Training.** Build matched preference pairs and hand them to a
backend. In simulation the backend applies a response model driven by ground
truth; in practice it exports a dataset.

**Phase D -- Re-evaluation.** Evaluate the resulting policy on **held-out**
tasks, and report mean, pass^k, lower-tail performance and reliability. Never on
the tasks the pairs came from.

What makes the comparison fair
------------------------------
* Every strategy sees the **same Phase A data**. Re-evaluating each one would
  confound selection with evaluation noise.
* Every strategy gets the **same pair budget**, not the same task budget.
* Every strategy is run over the **same seeds**, and results are reported as
  mean +/- se across seeds. A single-seed comparison of selection strategies is
  not interpretable; the simulation study found a spurious win that way.
* The held-out set is fixed before selection, so no strategy can select into it.

What this does not establish
----------------------------
With a simulated backend, this measures whether the *selection signal* finds
tasks with high true benefit under a stated response model. It does not measure
whether preference training on real trajectories improves a real agent. That
requires the export path and a real trainer, and the report labels simulated
results as such rather than presenting them as measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from ..metrics import lower_cvar, pass_at_k, pass_hat_k
from ..reliability.classify import TaskDiagnosis, diagnose
from ..reliability.hierarchical import fit_hierarchical
from ..reliability.posterior import JEFFREYS_PRIOR, UNIFORM_PRIOR, binary_posterior
from ..selection.pairs import PairConfig, build_dataset
from ..selection.selectors import make_selector
from ..trajectory.events import Trajectory
from .config import ExperimentConfig
from .splits import Split, make_split
from .training import TrainerBackend, make_backend

__all__ = ["PhaseAResult", "StrategyOutcome", "ExperimentResult", "run_experiment"]

RunFn = Callable[[str, int], float]


@dataclass
class PhaseAResult:
    """Evaluation output shared by every selection strategy."""

    scores: dict[str, list[float]]
    diagnoses: list[TaskDiagnosis]
    aggregate: dict
    n_executions: int
    domains: dict[str, str] = field(default_factory=dict)
    hierarchical_summary: Optional[dict] = None

    def by_task(self) -> dict[str, TaskDiagnosis]:
        return {d.task: d for d in self.diagnoses}

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([d.to_dict() for d in self.diagnoses])


@dataclass
class StrategyOutcome:
    """What one selection strategy achieved, on held-out tasks."""

    strategy: str
    seed: int
    n_tasks_selected: int
    n_pairs: int
    pair_shortfall: int
    heldout_mean_before: float
    heldout_mean_after: float
    heldout_delta: float
    heldout_pass_hat_k_before: float
    heldout_pass_hat_k_after: float
    heldout_cvar_before: float
    heldout_cvar_after: float
    train_delta: float
    benefit_captured: float
    is_simulated: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ExperimentResult:
    """Everything the experiment produced."""

    config: ExperimentConfig
    phase_a: PhaseAResult
    split: Split
    outcomes: list[StrategyOutcome]
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([o.to_dict() for o in self.outcomes])

    def summary(self):
        """Mean +/- standard error across seeds, per strategy. The headline table."""
        import pandas as pd

        df = self.to_frame()
        if df.empty:
            return df
        agg = (
            df.groupby("strategy")
            .agg(
                n_seeds=("seed", "count"),
                heldout_delta=("heldout_delta", "mean"),
                heldout_delta_sd=("heldout_delta", "std"),
                heldout_pass_hat_k_after=("heldout_pass_hat_k_after", "mean"),
                heldout_cvar_after=("heldout_cvar_after", "mean"),
                benefit_captured=("benefit_captured", "mean"),
                n_pairs=("n_pairs", "mean"),
                pair_shortfall=("pair_shortfall", "mean"),
            )
            .reset_index()
        )
        agg["heldout_delta_se"] = agg["heldout_delta_sd"] / np.sqrt(
            agg["n_seeds"].clip(lower=1)
        )
        return agg.sort_values("heldout_delta", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Phase A                                                                     #
# --------------------------------------------------------------------------- #
def _prior(name: str):
    return {"jeffreys": JEFFREYS_PRIOR, "uniform": UNIFORM_PRIOR}.get(name, JEFFREYS_PRIOR)


def run_phase_a(
    tasks: Sequence[str],
    run_fn: RunFn,
    config: ExperimentConfig,
    *,
    domains: Optional[Mapping[str, str]] = None,
    seed: int = 0,
) -> PhaseAResult:
    """Evaluate every task and estimate its latent reliability."""
    import pandas as pd

    ev = config.evaluation
    th = config.thresholds()
    domains = dict(domains or {})
    scores: dict[str, list[float]] = {}
    n_exec = 0

    if ev.adaptive_sampling:
        from ..active.allocation import make_policy
        from ..active.evaluator import AdaptiveEvaluator
        from ..active.stopping import default_stopping

        policy = make_policy(
            ev.allocation_policy,
            **({"thresholds": th} if ev.allocation_policy != "thompson" else {"thresholds": th}),
        )
        stopping = (
            default_stopping(th, min_runs=ev.min_runs_per_task, max_runs=ev.max_runs_per_task)
            if ev.stopping else None
        )
        evaluator = AdaptiveEvaluator(
            policy, stopping, th, _prior(config.reliability.prior),
            round_size=max(len(tasks), 8), success_threshold=ev.success_threshold,
        )
        budget = (ev.budget_per_task or ev.runs_per_task) * len(tasks)
        trace = evaluator.run(
            list(tasks), run_fn, budget, domains=domains,
            warmup=ev.warmup_runs, seed=seed,
        )
        scores = {t: list(s.scores) for t, s in trace.states.items()}
        n_exec = trace.n_executions
    else:
        for t in tasks:
            scores[t] = [float(run_fn(t, i)) for i in range(ev.runs_per_task)]
            n_exec += ev.runs_per_task

    prior = _prior(config.reliability.prior)
    succ = {t: float(sum(1 for x in v if x >= ev.success_threshold)) for t, v in scores.items()}

    hier_summary = None
    if config.reliability.model == "hierarchical" and len(tasks) >= 4:
        live = [t for t in tasks if scores.get(t)]
        fit = fit_hierarchical(
            ["policy"] * len(live), live,
            [succ[t] for t in live], [len(scores[t]) for t in live],
            [domains.get(t, "_all") for t in live],
            backend=config.reliability.backend,
            n_samples=config.reliability.n_mcmc_samples,
            seed=seed,
        )
        hier_summary = fit.summary()
        posteriors = {t: fit.cell("policy", t) for t in live}
        shrink = {t: fit.shrinkage("policy", t) for t in live}
    else:
        posteriors = {
            t: binary_posterior(int(succ[t]), len(scores[t]), prior)
            for t in tasks if scores.get(t)
        }
        shrink = {}

    diagnoses = [
        diagnose(
            posteriors[t], task=t, model="policy", thresholds=th, scores=scores[t],
            domain=domains.get(t),
            primary=config.reliability.recoverability_estimator,
            shrinkage=shrink.get(t),
        )
        for t in posteriors
    ]

    rows = [
        {"task": t, "score": s, "success": s >= ev.success_threshold}
        for t, v in scores.items() for s in v
    ]
    df = pd.DataFrame(rows)
    flat = np.array([s for v in scores.values() for s in v], dtype=float)
    ks = [k for k in (1, 2, 4, 8) if k <= min((len(v) for v in scores.values()), default=1)]
    aggregate = {
        "n_tasks": len(scores),
        "n_executions": n_exec,
        "mean": float(flat.mean()) if flat.size else float("nan"),
        "lower_cvar_0.2": float(lower_cvar(flat, 0.2)) if flat.size else float("nan"),
        **{f"pass@{k}": pass_at_k(df, k) for k in ks},
        **{f"pass^{k}": pass_hat_k(df, k) for k in ks},
    }
    return PhaseAResult(scores, diagnoses, aggregate, n_exec, domains, hier_summary)


# --------------------------------------------------------------------------- #
# Full experiment                                                             #
# --------------------------------------------------------------------------- #
def run_experiment(
    tasks: Sequence[str],
    run_fn: RunFn,
    config: ExperimentConfig,
    *,
    trajectories: Optional[Mapping[str, Sequence[Trajectory]]] = None,
    domains: Optional[Mapping[str, str]] = None,
    true_benefit: Optional[Mapping[str, float]] = None,
    strategies: Optional[Sequence[str]] = None,
    backend: Optional[TrainerBackend] = None,
    output_dir: str = "",
    split: Optional[Split] = None,
) -> ExperimentResult:
    """Run Phases A-D for each strategy, across ``config.experiment.n_seeds`` seeds.

    ``strategies`` defaults to whatever ``config.selection.method`` names:

    * a single strategy name -> only that strategy is run;
    * ``"all"`` -> every strategy is compared, which is the usual research run;
    * ``"baselines"`` -> the baseline set without recoverability.

    Honouring the config here is what makes ``selection.method`` a usable sweep
    axis. When it was ignored, every cell of a sweep over that axis produced
    identical results with different ids -- a silently useless ablation.
    """
    import tempfile

    ALL = (
        "random", "hardest", "lowest_mean", "highest_variance",
        "success_failure", "recoverability", "uncertainty_aware",
    )
    BASELINES = ("random", "hardest", "lowest_mean", "highest_variance", "success_failure")
    if strategies is None:
        method = config.selection.method
        if method == "all":
            strategies = ALL
        elif method == "baselines":
            strategies = BASELINES
        elif method in ALL:
            strategies = (method,)
        else:
            raise ValueError(
                f"selection.method={method!r} is not a known strategy; use one of "
                f"{sorted(ALL)}, or 'all' / 'baselines'"
            )

    warnings = list(config.validate())
    domains = dict(domains or {})
    output_dir = output_dir or tempfile.mkdtemp(prefix="disteval_")

    # -- fixed held-out split, decided BEFORE any selection ------------------
    if split is None:
        split = make_split(
            list(tasks),
            strategy=config.split.strategy,
            domains=domains or None,
            test_fraction=config.split.test_fraction,
            holdout_groups=config.split.holdout_groups,
            seed=config.experiment.seed,
        )
    if split.note:
        warnings.append(f"split: {split.note}")

    # -- Phase A: one evaluation, shared by every strategy -------------------
    phase_a = run_phase_a(
        tasks, run_fn, config, domains=domains, seed=config.experiment.seed
    )
    diag_by_task = phase_a.by_task()
    train_diags = [d for d in phase_a.diagnoses if d.task in set(split.train)]

    if true_benefit is None and config.training.backend == "simulated":
        raise ValueError(
            "the simulated training backend needs `true_benefit` (ground-truth "
            "per-task benefit). Without it there is nothing to measure a selection "
            "strategy against, and a response model driven by the selector's own "
            "score would make the experiment circular."
        )

    outcomes: list[StrategyOutcome] = []
    test_tasks = list(split.test)
    before_scores = {t: phase_a.scores.get(t, []) for t in test_tasks}

    def _stats(effects: Optional[dict], subset: Sequence[str]) -> dict:
        """Held-out statistics under a per-task logit shift."""
        import pandas as pd
        from scipy.special import expit, logit

        rows, means = [], []
        for t in subset:
            base = diag_by_task[t].posterior_mean if t in diag_by_task else float("nan")
            if not np.isfinite(base):
                continue
            if effects:
                shifted = float(expit(logit(np.clip(base, 1e-6, 1 - 1e-6)) + effects.get(t, 0.0)))
            else:
                shifted = base
            means.append(shifted)
            n = len(phase_a.scores.get(t, [])) or config.evaluation.runs_per_task
            # Expected counts under the shifted probability; the re-evaluation is
            # analytic rather than resampled so strategies are not compared
            # through independent Monte-Carlo noise.
            for i in range(n):
                rows.append({"task": t, "score": shifted, "success": False})
        arr = np.asarray(means, dtype=float)
        k = min(config.evaluation.runs_per_task, 4)
        # pass^k under independence within a task: p^k, averaged over tasks.
        return {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "pass_hat_k": float(np.mean(arr**k)) if arr.size else float("nan"),
            "cvar": float(lower_cvar(arr, 0.2)) if arr.size else float("nan"),
        }

    base_test = _stats(None, test_tasks)
    base_train = _stats(None, split.train)

    for seed_i in range(config.experiment.n_seeds):
        seed = config.experiment.seed + seed_i
        for name in strategies:
            kwargs = {"seed": seed}
            if name in ("recoverability", "uncertainty_aware"):
                kwargs["estimator"] = config.reliability.recoverability_estimator
                kwargs["restrict_to_recoverable"] = config.selection.restrict_to_recoverable
            sel = make_selector(name, **kwargs).select(train_diags, config.selection.n_tasks)

            pair_cfg = PairConfig(
                min_margin=config.selection.min_margin,
                max_pairs_per_task=config.selection.max_pairs_per_task,
                match_environment=config.selection.match_environment,
            )
            if trajectories:
                ds = build_dataset(
                    sel.tasks, trajectories, strategy=name,
                    target_pairs=config.selection.n_pairs,
                    recoverability={d.task: d.recoverability for d in train_diags},
                    config=pair_cfg, seed=seed,
                )
            else:
                # No trajectory logs: synthesise the pair *accounting* only, so the
                # selection comparison still runs. No fake trajectory content is
                # produced, and the note says so.
                from ..selection.pairs import PreferenceDataset

                per = config.selection.max_pairs_per_task
                counted, tlist = 0, []
                for t in sel.tasks:
                    d = diag_by_task[t]
                    avail = int(min(per, d.n_success, d.n_runs - d.n_success))
                    if avail > 0:
                        tlist.append(t)
                        counted += avail
                    if counted >= config.selection.n_pairs:
                        break
                ds = PreferenceDataset(
                    pairs=[], strategy=name,
                    requested_pairs=config.selection.n_pairs,
                    selected_tasks=sel.tasks, tasks_with_pairs=tlist,
                    note="no trajectory logs supplied; pair counts are derived from "
                         "run outcomes and no pair content was generated",
                )
                ds.pairs = []
                ds._counted = min(counted, config.selection.n_pairs)

            n_pairs = len(ds.pairs) or getattr(ds, "_counted", 0)
            pairs_per_task = {}
            for t in ds.tasks_with_pairs:
                d = diag_by_task[t]
                pairs_per_task[t] = int(
                    min(config.selection.max_pairs_per_task, d.n_success,
                        d.n_runs - d.n_success)
                )

            be = backend or make_backend(
                config.training.backend,
                **(
                    {"true_benefit": dict(true_benefit or {}), "domains": domains,
                     "seed": seed}
                    if config.training.backend == "simulated" else {}
                ),
            )
            if isinstance(be, type(be)) and hasattr(be, "true_benefit"):
                be.seed = seed
            if ds.pairs:
                result = be.train(ds, output_dir, all_tasks=list(tasks))
            else:
                from ..selection.pairs import PreferencePair

                synth = [
                    PreferencePair(t, {}, {}, 1.0, 0.0,
                                   diag_by_task[t].recoverability, name, 1.0)
                    for t, m in pairs_per_task.items() for _ in range(m)
                ][: config.selection.n_pairs]
                ds2 = PreferenceDataset(
                    synth, name, config.selection.n_pairs, ds.selected_tasks,
                    ds.tasks_with_pairs, note=ds.note,
                )
                result = be.train(ds2, output_dir, all_tasks=list(tasks))
                n_pairs = len(synth)

            after_test = _stats(result.task_effects, test_tasks)
            after_train = _stats(result.task_effects, split.train)
            captured = (
                float(np.mean([float((true_benefit or {}).get(t, 0.0)) for t in sel.tasks]))
                if sel.tasks else float("nan")
            )

            outcomes.append(
                StrategyOutcome(
                    strategy=name,
                    seed=seed,
                    n_tasks_selected=sel.delivered,
                    n_pairs=n_pairs,
                    pair_shortfall=max(config.selection.n_pairs - n_pairs, 0),
                    heldout_mean_before=base_test["mean"],
                    heldout_mean_after=after_test["mean"],
                    heldout_delta=after_test["mean"] - base_test["mean"],
                    heldout_pass_hat_k_before=base_test["pass_hat_k"],
                    heldout_pass_hat_k_after=after_test["pass_hat_k"],
                    heldout_cvar_before=base_test["cvar"],
                    heldout_cvar_after=after_test["cvar"],
                    train_delta=after_train["mean"] - base_train["mean"],
                    benefit_captured=captured,
                    is_simulated=result.is_simulated,
                    note=(sel.note + " " + ds.note).strip(),
                )
            )

    # Equal *pair* budgets are what makes the comparison fair, and strategies
    # can fail to reach the target for real reasons (they picked tasks with no
    # usable success/failure pairs). When they end up with materially different
    # amounts of data, the held-out comparison is confounded by data volume and
    # the result must say so rather than being read as a selection effect.
    by_strategy: dict[str, list[int]] = {}
    for o in outcomes:
        by_strategy.setdefault(o.strategy, []).append(o.n_pairs)
    if by_strategy:
        means = {k: float(np.mean(v)) for k, v in by_strategy.items()}
        lo, hi = min(means.values()), max(means.values())
        if hi > 0 and (hi - lo) / hi > 0.15:
            worst = min(means, key=means.get)
            warnings.append(
                f"strategies received materially different pair counts "
                f"({lo:.0f} to {hi:.0f}; lowest: {worst!r}). The held-out comparison "
                "is therefore confounded by training-data volume, not only by which "
                "tasks were chosen. Lower selection.n_pairs to the minimum achieved, "
                "or read the benefit_captured column instead, which is independent "
                "of pair yield."
            )

    return ExperimentResult(
        config=config,
        phase_a=phase_a,
        split=split,
        outcomes=outcomes,
        warnings=warnings,
        meta={
            "output_dir": output_dir,
            "n_strategies": len(strategies),
            "held_out_tasks": len(test_tasks),
            "training_pool_tasks": len(split.train),
        },
    )
