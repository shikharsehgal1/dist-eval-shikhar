"""The adaptive evaluation loop, and the honest accounting of what it saves.

Usage
-----
Supply a ``run_fn(task, run_index) -> score`` that executes the agent once, a
budget, an allocation policy and a stopping rule. The evaluator alternates
allocation and execution in rounds, applying the stopping rules after each round,
and returns per-task states plus a full trace.

``run_fn`` is the only thing that touches the agent, so the same loop drives a
simulator, a cached replay of an existing eval, or a live frontier model.

Measuring the saving, without cheating
--------------------------------------
The claim "adaptive sampling saves N% of executions" is easy to fake -- stop
early everywhere and declare victory. :func:`compare_allocation` therefore fixes
the *outcome quality* and compares the cost of reaching it, rather than fixing
cost and comparing quality:

* run the uniform baseline at ``k`` runs per task,
* run the adaptive policy under a budget,
* compare **classification agreement against a high-precision reference** and the
  number of executions each used.

The reference is the same tasks evaluated at a much larger ``k`` (in simulation,
the known ground truth). Without a reference, "agreement with the baseline" would
just measure how well adaptive imitates uniform, which is not the question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from ..reliability.classify import ReliabilityThresholds
from ..reliability.posterior import JEFFREYS_PRIOR, BetaPrior
from .allocation import AllocationPolicy, TaskState, UniformAllocation
from .stopping import StoppingLedger, StoppingRule, default_stopping
from .value import label_of

__all__ = ["EvaluationTrace", "AdaptiveEvaluator", "compare_allocation"]

RunFn = Callable[[str, int], float]


@dataclass
class EvaluationTrace:
    """Everything that happened during an adaptive evaluation."""

    states: dict[str, TaskState]
    ledger: StoppingLedger
    n_executions: int
    budget: int
    rounds: list[dict] = field(default_factory=list)
    policy: str = ""

    def labels(self, thresholds: Optional[ReliabilityThresholds] = None) -> dict[str, str]:
        th = thresholds or ReliabilityThresholds()
        return {t: label_of(s.posterior, th) for t, s in self.states.items()}

    def to_frame(self):
        import pandas as pd

        rows = []
        for t, s in self.states.items():
            lo, hi = s.posterior.credible_interval()
            rows.append(
                {
                    "task": t,
                    "domain": s.domain,
                    "n_runs": s.n_runs,
                    "n_success": s.n_success,
                    "posterior_mean": s.posterior.mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ci_width": hi - lo,
                    "stopped": s.stopped,
                    "stop_reason": s.stop_reason,
                }
            )
        return pd.DataFrame(rows).sort_values("task").reset_index(drop=True)

    def summary(self) -> dict:
        runs = [s.n_runs for s in self.states.values()]
        return {
            "policy": self.policy,
            "n_tasks": len(self.states),
            "n_executions": self.n_executions,
            "budget": self.budget,
            "mean_runs_per_task": float(np.mean(runs)) if runs else 0.0,
            "min_runs_per_task": int(min(runs)) if runs else 0,
            "max_runs_per_task": int(max(runs)) if runs else 0,
            "n_rounds": len(self.rounds),
            **{f"stopping_{k}": v for k, v in self.ledger.summary().items()},
        }


class AdaptiveEvaluator:
    """Runs the allocate -> execute -> update -> stop loop under a hard budget."""

    def __init__(
        self,
        policy: Optional[AllocationPolicy] = None,
        stopping: Optional[StoppingRule] = None,
        thresholds: Optional[ReliabilityThresholds] = None,
        prior: BetaPrior = JEFFREYS_PRIOR,
        *,
        round_size: int = 16,
        success_threshold: float = 1.0,
    ):
        self.policy = policy or UniformAllocation()
        self.thresholds = thresholds or ReliabilityThresholds()
        self.stopping = stopping or default_stopping(self.thresholds)
        self.prior = prior
        self.round_size = int(round_size)
        self.success_threshold = success_threshold

    def run(
        self,
        tasks: Sequence[str],
        run_fn: RunFn,
        budget: int,
        *,
        domains: Optional[Mapping[str, str]] = None,
        warmup: int = 0,
        seed: int = 0,
    ) -> EvaluationTrace:
        """Spend at most ``budget`` executions across ``tasks``.

        ``warmup`` runs per task are spent uniformly before any adaptation. This
        matters: with zero observations every posterior is the prior, so the
        first adaptive decision is made on no information and the policy is
        effectively arbitrary. A warmup of 2-3 is cheap and makes the adaptive
        phase meaningful. The warmup prefix is also the part of the data that
        remains a clean uniform sample, which is what suite-level aggregates
        should be computed from.
        """
        domains = domains or {}
        states = {
            t: TaskState(t, domain=domains.get(t), prior=self.prior) for t in tasks
        }
        ledger = StoppingLedger()
        trace = EvaluationTrace(states, ledger, 0, budget, policy=self.policy.name)
        remaining = int(budget)

        def execute(task: str) -> None:
            nonlocal remaining
            st = states[task]
            score = float(run_fn(task, st.n_runs))
            st.observe(score, success=score >= self.success_threshold)
            trace.n_executions += 1
            remaining -= 1

        # -- uniform warmup ---------------------------------------------------
        for _ in range(max(warmup, 0)):
            for t in tasks:
                if remaining <= 0:
                    break
                execute(t)
        if warmup:
            trace.rounds.append(
                {"kind": "warmup", "executions": trace.n_executions, "per_task": warmup}
            )

        # -- adaptive rounds --------------------------------------------------
        round_idx = 0
        while remaining > 0:
            for t, st in states.items():
                if st.stopped:
                    continue
                stop, why = self.stopping.should_stop(st)
                if stop:
                    st.stopped, st.stop_reason = True, why
                    ledger.record(st, why)
            live = [s for s in states.values() if not s.stopped]
            if not live:
                break
            size = min(self.round_size, remaining)
            alloc = self.policy.allocate(live, size, seed=seed + round_idx)
            if alloc.spent == 0:
                break
            for t in alloc.order:
                if remaining <= 0:
                    break
                execute(t)
            trace.rounds.append(
                {
                    "kind": "adaptive",
                    "round": round_idx,
                    "allocation": dict(alloc.allocation),
                    "remaining": remaining,
                }
            )
            round_idx += 1

        assert trace.n_executions <= budget, "evaluator exceeded its budget"
        return trace


def compare_allocation(
    tasks: Sequence[str],
    run_fn: RunFn,
    reference_labels: Mapping[str, str],
    *,
    policies: Optional[Mapping[str, AllocationPolicy]] = None,
    budget: int,
    warmup: int = 2,
    thresholds: Optional[ReliabilityThresholds] = None,
    stopping: Optional[StoppingRule] = None,
    seed: int = 0,
) -> "object":
    """Compare allocation policies at equal budget against a reference labelling.

    ``reference_labels`` must come from a *much* better-resourced evaluation (or,
    in simulation, from the known ground truth). Comparing an adaptive policy to
    the uniform baseline instead would only measure how well it imitates uniform.

    Returns a DataFrame with, per policy: executions used, agreement with the
    reference, agreement restricted to the RECOVERABLE class (the one that feeds
    selection, and the one that matters), and the share of tasks left UNCERTAIN.
    """
    import pandas as pd

    th = thresholds or ReliabilityThresholds()
    policies = policies or {
        "uniform": UniformAllocation(th),
        "greedy_voi": __import__(
            "disteval.active.allocation", fromlist=["GreedyValuePolicy"]
        ).GreedyValuePolicy(thresholds=th),
    }
    rows = []
    for name, pol in policies.items():
        ev = AdaptiveEvaluator(pol, stopping, th, round_size=max(len(tasks), 8))
        tr = ev.run(tasks, run_fn, budget, warmup=warmup, seed=seed)
        got = tr.labels(th)
        shared = [t for t in tasks if t in reference_labels]
        agree = [got[t] == reference_labels[t] for t in shared]
        rec = [t for t in shared if reference_labels[t] == "RECOVERABLE"]
        rec_agree = [got[t] == reference_labels[t] for t in rec]
        rows.append(
            {
                "policy": name,
                "executions": tr.n_executions,
                "budget": budget,
                "agreement": float(np.mean(agree)) if agree else float("nan"),
                "recoverable_recall": float(np.mean(rec_agree)) if rec_agree else float("nan"),
                "n_recoverable_reference": len(rec),
                "pct_uncertain": float(
                    np.mean([v == "UNCERTAIN" for v in got.values()])
                ),
                "mean_runs_per_task": tr.summary()["mean_runs_per_task"],
            }
        )
    return pd.DataFrame(rows).sort_values("agreement", ascending=False).reset_index(drop=True)
