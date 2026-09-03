"""A single registry declaring what every headline metric means.

Every metric in this repository is registered here with its mathematical
definition, assumptions, valid domain, edge-case behaviour, estimator, and
uncertainty method. The registry is the source of truth used by the research
report generator and by ``docs/METRICS.md``, so a metric cannot appear in a
report without its assumptions travelling with it.

This is deliberately data, not prose: :func:`as_frame` renders it, and the tests
assert that every metric named in a generated report has a registry entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

__all__ = ["MetricSpec", "REGISTRY", "get", "as_frame", "to_markdown", "register"]


@dataclass(frozen=True)
class MetricSpec:
    """Formal declaration of one metric."""

    name: str
    #: One-line plain-language reading.
    summary: str
    #: Mathematical definition, in plain text/LaTeX-ish notation.
    definition: str
    #: What must be true for the number to mean what it says.
    assumptions: tuple[str, ...]
    #: Valid range of the metric.
    domain: str
    #: What happens at n=0, n=1, all-success, all-failure, ties.
    edge_cases: tuple[str, ...]
    #: How it is estimated from a finite sample.
    estimator: str
    #: How uncertainty is quantified, or why it is not.
    uncertainty: str
    #: Direction in which larger is better.
    higher_is_better: bool = True
    #: Where the implementation lives.
    implementation: str = ""
    #: Prior work this is taken from. Empty means "standard/no single source".
    prior_work: tuple[str, ...] = ()
    references: tuple[str, ...] = ()


REGISTRY: dict[str, MetricSpec] = {}


def register(spec: MetricSpec) -> MetricSpec:
    if spec.name in REGISTRY:
        raise ValueError(f"metric {spec.name!r} already registered")
    REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> MetricSpec:
    return REGISTRY[name]


def _r(**kw) -> MetricSpec:
    return register(MetricSpec(**kw))


# --------------------------------------------------------------------------- #
# Aggregate outcome metrics                                                   #
# --------------------------------------------------------------------------- #
_r(
    name="mean",
    summary="Average score across all runs.",
    definition="mean(q) = (1/N) * sum_i q_i",
    assumptions=("Runs are exchangeable.", "Scores are on a common scale."),
    domain="[0, 1] for normalised scores",
    edge_cases=("nan for N=0.", "Identical for [0,0,1,1] and [0.5,0.5,0.5,0.5]."),
    estimator="Sample mean (unbiased).",
    uncertainty="Percentile bootstrap, or the normal interval when N is large.",
    implementation="numpy",
    prior_work=("Universal; not novel.",),
)

_r(
    name="iqm",
    summary="Interquartile mean: mean of the middle 50% of runs.",
    definition="IQM(q) = mean of q_i with Q1 <= q_i <= Q3",
    assumptions=("At least 4 runs for the trimming to do anything meaningful.",),
    domain="[0, 1] for normalised scores",
    edge_cases=(
        "Falls back to the median when the interquartile mask is empty.",
        "For N<=2 it equals the mean.",
    ),
    estimator="Trimmed mean; more efficient than the median, more robust than the mean.",
    uncertainty="Stratified percentile bootstrap (rliable's recommendation).",
    implementation="disteval.metrics.iqm",
    prior_work=("Agarwal et al. 2021 (rliable). Not novel here.",),
    references=("Agarwal et al., 'Deep RL at the Edge of the Statistical Precipice', NeurIPS 2021",),
)

_r(
    name="pass@k",
    summary="Probability at least one of k attempts succeeds. Peak capability.",
    definition="pass@k(t) = 1 - C(n-c, k) / C(n, k), averaged over tasks t",
    assumptions=(
        "The n observed runs are i.i.d. draws from the task's run distribution.",
        "Sampling k without replacement from the n observed runs estimates the "
        "k-attempt success probability -- this is the unbiased estimator, not "
        "1-(1-p_hat)^k.",
    ),
    domain="[0, 1]",
    edge_cases=(
        "Requires n >= k; falls back to the empirical indicator when n < k.",
        "Non-decreasing in k (property-tested).",
        "pass@1 equals the per-task success rate.",
    ),
    estimator="Unbiased combinatorial estimator (Chen et al. 2021).",
    uncertainty="Task-level bootstrap; the per-task estimator itself is unbiased.",
    implementation="disteval.metrics.pass_at_k",
    prior_work=("Chen et al. 2021 (Codex). Not novel here.",),
    references=("Chen et al., 'Evaluating Large Language Models Trained on Code', 2021",),
)

_r(
    name="pass^k",
    summary="Probability ALL k attempts succeed. Deployment reliability.",
    definition="pass^k(t) = C(c, k) / C(n, k), averaged over tasks t",
    assumptions=(
        "Same i.i.d. assumption as pass@k. Violated when runs share a seed, "
        "environment snapshot, or cached retrieval -- see "
        "disteval.reliability.correlated for the diagnostic.",
    ),
    domain="[0, 1]",
    edge_cases=(
        "Non-increasing in k (property-tested).",
        "Zero as soon as any observed run fails and k > c.",
        "pass^1 equals pass@1.",
    ),
    estimator="Unbiased combinatorial estimator.",
    uncertainty="Task-level bootstrap.",
    implementation="disteval.metrics.pass_hat_k",
    prior_work=("tau-bench (Yao et al. 2024) popularised it for agents. Not novel here.",),
    references=("Yao et al., 'tau-bench', 2024",),
)

_r(
    name="reliability_gap",
    summary="pass@k minus pass^k: apparent capability that does not reproduce.",
    definition="gap(k) = pass@k - pass^k",
    assumptions=("Both components share the i.i.d. run assumption.",),
    domain="[0, 1]",
    edge_cases=("Zero when every task is all-success or all-failure.",),
    estimator="Difference of the two unbiased estimators.",
    uncertainty="Paired task-level bootstrap of the difference.",
    implementation="disteval.metrics.reliability_gap",
    prior_work=("A restatement of the tau-bench observation, not a new metric.",),
)

_r(
    name="lower_cvar",
    summary="Expected shortfall: mean of the worst alpha fraction of runs.",
    definition="CVaR_alpha^lower(q) = E[q | q <= VaR_alpha(q)]",
    assumptions=(
        "alpha * N >= 1, otherwise the statistic is a single order statistic.",
        "Scores are comparable across runs.",
    ),
    domain="[0, 1] for normalised scores; <= mean always",
    edge_cases=(
        "nan for N=0.",
        "Equals min(q) when alpha*N < 1.",
        "Equals the mean when all runs are tied.",
        "Coherent (subadditive); VaR is not.",
    ),
    estimator="Empirical mean below the empirical alpha-quantile.",
    uncertainty="Percentile bootstrap; wide at small N by construction.",
    higher_is_better=True,
    implementation="disteval.metrics.lower_cvar",
    prior_work=("Rockafellar & Uryasev 2000. Standard risk measure; not novel here.",),
    references=("Rockafellar & Uryasev, 'Optimization of Conditional Value-at-Risk', 2000",),
)

_r(
    name="upper_cvar",
    summary="Mean of the BEST alpha fraction of runs. Risk-seeking, not reliability.",
    definition="CVaR_alpha^upper(q) = E[q | q >= VaR_{1-alpha}(q)]",
    assumptions=("Same as lower_cvar.",),
    domain="[0, 1]; >= mean always",
    edge_cases=(
        "Cannot distinguish [0,0,1] from [1,1,1] at alpha=1/3: both are 1.0. "
        "This is why it must not be used as a reliability objective.",
    ),
    estimator="Empirical mean above the empirical (1-alpha)-quantile.",
    uncertainty="Percentile bootstrap.",
    implementation="disteval.metrics.upper_cvar",
    prior_work=("Standard; included for the counterexample, not as a recommendation.",),
)

# --------------------------------------------------------------------------- #
# Latent reliability metrics                                                  #
# --------------------------------------------------------------------------- #
_r(
    name="posterior_mean",
    summary="Posterior mean of the latent per-task performance parameter.",
    definition="E[p_t | D_t] under Beta(alpha+s, beta+n-s), or the hierarchical fit",
    assumptions=(
        "Binary outcomes: exact Beta-Binomial conjugacy.",
        "Continuous scores: quasi-binomial dispersion correction; the interval is "
        "a calibrated summary of the mean score, not an exact posterior.",
        "Runs within a task are i.i.d. given p_t.",
    ),
    domain="[0, 1]",
    edge_cases=(
        "With n=0 it returns the prior mean.",
        "0/n never returns exactly 0 and n/n never exactly 1 -- deliberate.",
    ),
    estimator="Conjugate posterior, or the hierarchical Laplace/PG-Gibbs fit.",
    uncertainty="Equal-tailed credible interval from the same posterior.",
    implementation="disteval.reliability.posterior",
    prior_work=("Beta-Binomial conjugacy is textbook. Not novel here.",),
)

_r(
    name="capability",
    summary="C_t: posterior probability the agent can do the task at all.",
    definition="C_t = P(p_t > tau_cap | D_t)",
    assumptions=(
        "tau_cap is a choice, not a fact; results must be reported with it.",
        "Inherits the posterior's assumptions.",
    ),
    domain="[0, 1]",
    edge_cases=(
        "Near the prior value when n is small -- this is why min_runs exists.",
        "Monotone increasing in observed successes.",
    ),
    estimator="Survival function of the posterior at tau_cap.",
    uncertainty="It IS a probability; no second-order interval is reported.",
    implementation="disteval.reliability.classify.diagnose",
    prior_work=("A threshold query on a standard posterior. Not novel.",),
)

_r(
    name="reliability",
    summary="R_t: posterior probability the agent meets the deployment bar.",
    definition="R_t = P(p_t > tau_rel | D_t), with tau_rel > tau_cap",
    assumptions=("As for capability, with a higher threshold.",),
    domain="[0, 1]",
    edge_cases=(
        "8/8 successes under a Jeffreys prior gives only R ~ 0.81 at tau_rel=0.9. "
        "Certainty about high reliability genuinely requires many runs.",
    ),
    estimator="Survival function of the posterior at tau_rel.",
    uncertainty="It IS a probability.",
    implementation="disteval.reliability.classify.diagnose",
    prior_work=("Not novel.",),
)

_r(
    name="recoverability_headroom",
    summary="Posterior-expected reliability missing, given evidence of capability.",
    definition="rho_t = E[(tau_rel - p_t)^+ * 1{p_t > tau_cap}] / (tau_rel - tau_cap)",
    assumptions=(
        "That expected reliability headroom is a reasonable proxy for training "
        "value. THIS IS THE HYPOTHESIS UNDER TEST, not an established result.",
        "Inherits the posterior's assumptions.",
    ),
    domain="[0, 1]",
    edge_cases=(
        "~0 for a task already at the bar (nothing to gain).",
        "~0 for a task with no evidence of capability (the indicator kills it).",
        "Maximised for demonstrated capability far below the bar.",
    ),
    estimator="Quadrature against the posterior CDF.",
    uncertainty="Propagated from the posterior by construction; also reported via "
    "rank stability under posterior resampling.",
    implementation="disteval.reliability.classify.expected_headroom",
    prior_work=(
        "The *use* of a task-level reliability estimate to select training "
        "trajectories is what this repository is testing. Repeated-run "
        "evaluation, posterior estimation and preference learning are all prior "
        "work.",
    ),
)

_r(
    name="failure_entropy",
    summary="Entropy of a task's failure-mode distribution.",
    definition="H(F_t) = -sum_j P(F=j|t) log P(F=j|t)",
    assumptions=(
        "The failure taxonomy is exhaustive and mutually exclusive for the "
        "labelled runs.",
        "Estimated on few failures, so heavily biased downward without correction.",
    ),
    domain="[0, log J] nats for J failure modes",
    edge_cases=(
        "0 when every failure has the same cause.",
        "Undefined (nan) with no observed failures.",
        "Miller-Madow correction applied by default at small counts.",
    ),
    estimator="Plug-in entropy of the add-alpha smoothed failure counts, with a "
    "Miller-Madow bias correction.",
    uncertainty="Bootstrap over the failure labels.",
    implementation="disteval.diagnosis.entropy",
    prior_work=("Plug-in entropy estimation is standard.",),
)

_r(
    name="hazard",
    summary="Per-step probability of irrecoverable failure given survival so far.",
    definition="h_j = P(fail at step j | survived to step j)",
    assumptions=(
        "Steps are comparable across runs after alignment; see the alignment "
        "module. Raw step index is a weak clock when agents take different paths.",
        "Right-censoring (a run that ends successfully) is non-informative.",
    ),
    domain="[0, 1] per step",
    edge_cases=(
        "Undefined where the risk set is empty.",
        "Kaplan-Meier survival is a step function; late steps have tiny risk sets.",
    ),
    estimator="Nelson-Aalen / Kaplan-Meier on the discrete step clock.",
    uncertainty="Greenwood's formula for the survival variance.",
    implementation="disteval.diagnosis.survival",
    prior_work=("Standard survival analysis. Not novel.",),
)


def as_frame():
    """Render the registry as a DataFrame."""
    import pandas as pd

    rows = []
    for spec in REGISTRY.values():
        rows.append(
            {
                "metric": spec.name,
                "summary": spec.summary,
                "definition": spec.definition,
                "domain": spec.domain,
                "higher_is_better": spec.higher_is_better,
                "assumptions": " | ".join(spec.assumptions),
                "edge_cases": " | ".join(spec.edge_cases),
                "estimator": spec.estimator,
                "uncertainty": spec.uncertainty,
                "implementation": spec.implementation,
                "prior_work": " | ".join(spec.prior_work),
            }
        )
    return pd.DataFrame(rows)


def to_markdown(names: Optional[list[str]] = None) -> str:
    """Render selected metrics (default: all) as a Markdown reference section."""
    out = []
    for name in names or list(REGISTRY):
        s = REGISTRY[name]
        out.append(f"### `{s.name}`\n")
        out.append(f"{s.summary}\n")
        out.append(f"**Definition**  `{s.definition}`\n")
        out.append(f"**Domain**  {s.domain} (higher is "
                   f"{'better' if s.higher_is_better else 'worse'})\n")
        out.append("**Assumptions**\n")
        out.extend(f"- {a}" for a in s.assumptions)
        out.append("\n**Edge cases**\n")
        out.extend(f"- {e}" for e in s.edge_cases)
        out.append(f"\n**Estimator**  {s.estimator}\n")
        out.append(f"**Uncertainty**  {s.uncertainty}\n")
        if s.prior_work:
            out.append("**Prior work**  " + " ".join(s.prior_work) + "\n")
        if s.implementation:
            out.append(f"**Implementation**  `{s.implementation}`\n")
        out.append("")
    return "\n".join(out)
