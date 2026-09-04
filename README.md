# dist-eval-shikhar

> A reliability-focused evaluation framework for long-horizon agents: it estimates
> **per-criterion** latent reliability from repeated runs, measures the gap between
> what an agent can demonstrably do and what it does dependably, diagnoses *how*
> its failures are structured — and tests whether that structure helps pick
> training data better than difficulty or learning progress already do.

[![tests](https://img.shields.io/badge/tests-970%20passing-brightgreen)](#tests)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

```bash
pip install -e .
python -m disteval evaluate examples/runs.json \
    --trajectories examples/trajectories.jsonl \
    --tasks examples/tasks.json -o report/
```

No API key, no network, no GPU. Writes a Markdown/HTML report with per-task
posteriors, classifications, the capability–reliability gap ranking, and figures.

---

## Motivation

A benchmark that runs each task once reports the mean of a distribution it never
looked at. Two agents scoring 0.5 can be very different products:

```
Agent A:  0.5 0.5 0.5 0.5 0.5 0.5     consistently half-right
Agent B:  1.0 0.0 1.0 0.0 0.0 1.0     right half the time
```

For long-horizon professional work the second is usually the harder problem, and
the mean cannot see it. On the shipped demo data the agent scores **pass@8 = 0.92
but pass^8 = 0.17**: most of what it can *ever* do, it cannot do *reliably*.
That gap — not the mean — is what this framework measures and then tries to act on.

---

## Core idea

Consider three tasks, each run eight times:

```
Task A:  1 1 1 1 1 1 1 1
Task B:  1 0 1 0 0 1 0 0
Task C:  0 0 0 0 0 0 0 0
```

It is tempting to say A is solved, B is "recoverable", C is stuck. But those are
claims about a latent quantity — the probability $p_t$ that a run succeeds — and
eight coin flips do not pin it down. So the framework estimates $p_t$ with a
posterior and reports **probabilities, not verdicts**:

| task | successes | posterior mean | 95% CI | $C_t=P(p>0.15)$ | $R_t=P(p>0.9)$ | label |
|---|---|---|---|---|---|---|
| A | 8/8 | 0.94 | [0.74, 1.00] | 1.000 | 0.813 | SOLID |
| B | 3/8 | 0.39 | [0.12, 0.71] | 0.950 | 0.000 | RECOVERABLE |
| C | 0/8 | 0.06 | [0.00, 0.26] | 0.101 | 0.000 | STUCK |

Note what the numbers say. Even 8/8 gives only **0.81** confidence that this agent
clears a 90% reliability bar — high reliability is genuinely expensive to certify.
And B is not called recoverable because one run succeeded; it is called that
because the posterior puts 95% mass on "can do it at all" *and* essentially none
on "does it dependably". Run B only twice and it comes out **UNCERTAIN**, which is
the correct answer at that sample size.

**Where rubrics exist, this is done per criterion, not per task.** A task with
nine always-satisfied criteria and one coin flip has the same pass rate as one
where all ten are half-right, and they need completely different fixes.

---

## Methodology

### 1. Criterion-level reliability estimation

$$\mathrm{logit}(p_{t,j}) = \mu + \alpha_t + \beta_j + \gamma_{d(t)} + \varepsilon_{t,j}$$

Crossed random effects over (task, criterion) cells, fitted by Laplace/EM or by
Pólya-Gamma Gibbs. Partial pooling lets a criterion observed on eight runs borrow
strength from the same criterion elsewhere; the interaction term keeps "this
criterion is unstable *on this task*" from being shrunk into main effects. An
empirical-Bayes path and an unpooled baseline are available for the ablations.

Pooling is **not** assumed to help: `pooling_diagnostic()` decides per dataset by
leave-one-run-out predictive log-likelihood and the report prints the decision.

### 2. The capability–reliability gap

$$C_t = \tfrac1J\textstyle\sum_j P(p_{t,j} > \tau_{\text{cap}})
\qquad
R_t = \tfrac1J\textstyle\sum_j P(p_{t,j} > \tau_{\text{rel}})
\qquad
G_t = C_t - R_t$$

The fraction of the rubric the agent can demonstrably satisfy but does not satisfy
dependably. **No observed maximum appears in it**, so it does not inflate with more
runs. *This metric is not claimed to be novel* — it is a posterior restatement, at
criterion granularity, of the same comparison as pass@k vs pass^k.

Three things about it that are easy to get wrong are documented in
[THEORY.md §7](THEORY.md): it is **non-monotone** in performance, so shrinkage can
*reorder* a ranking rather than compress it; criterion-level and task-level
capability can diverge sharply; and a high gap says nothing on its own about
training value.

### 3. Adaptive rerun allocation

Repeated runs are the dominant cost. Rather than $k$ runs everywhere, spend the
budget where it changes an answer: exact value-of-information scoring (multi-step
label-flip probability, BALD, expected entropy reduction), greedy or Thompson
allocation under a hard budget, and sequential stopping once a task's label is
settled or another run is worthless.

### 4. Trajectory structure

Successful and failed runs of the same task are aligned (Needleman–Wunsch over a
pluggable event similarity, or DTW over state features) so that an inserted retry
becomes a gap rather than shifting every later step. From the alignment: the
**earliest consequential divergence**, whether the run truly recovered, an
extensible **failure-mode taxonomy** with causal DAGs, **failure-mode entropy**,
**trajectory distance** between the success and failure clouds, and the
**counterfactual intervention cost** of the cheapest edit turning a failure into
an observed success.

### 5. Curriculum selection

| strategy | ranks by |
|---|---|
| `uniform` | nothing — the control |
| `difficulty` | lowest estimated performance |
| `uncertainty` | largest posterior sd |
| `learning_progress` | PAC-style: observed progress if a history exists, else $\mathbb{E}[p(1-p)]$ |
| `capability_reliability_gap` | criterion-level $G_t$ |
| `gap_plus_structure` | $G_t$ weighted by within-task failure structure |

### 6. Optimiser-agnostic training

Selection is separate from optimisation. One selection is exported through four
views — `pairwise` (DPO/IPO/SLiC), `listwise`, `weighted_sft`, `scalar_reward`
(PPO/GRPO, reward models) — so comparing objectives is not confounded by a
different data pipeline. **DPO is one optional backend, not the framework's
objective.** `weighted_sft` is the control: if it matches a preference loss on the
same data, the preference machinery is not what is doing the work.

### 7. Held-out reliability gain per unit of training data

Evaluation is on held-out **domains**, not held-out tasks — an in-distribution
split cannot separate transferable improvement from domain-specific adaptation.
Results are reported per 100 examples and per 1000 tokens, because a strategy that
merely produced more pairs would otherwise look better for the wrong reason.

---

## The hypothesis under test

> **Does within-task failure structure add predictive value for training-data
> selection beyond difficulty and learning progress?**

This is an *incremental-validity* question, and it is the whole claim.

**What is deliberately not claimed.** Not that the gap metric is novel. Not that
tasks with intermediate success rates are inherently more trainable — a plausible
intuition with obvious failure modes: such a task may be nearly solved with little
headroom, its failures may be irreducibly stochastic and carry no learnable
signal, and any gain may not transfer. Not that repeated-run evaluation, Pass@k,
Pass^k, CVaR, perturbation testing or preference learning from success/failure
trajectories are new; all are prior work, cited in [THEORY.md](THEORY.md).

**How it is tested.** `incremental_validity()` regresses observed training benefit
on {difficulty, learning progress} and on {difficulty, learning progress,
structure}, comparing **out-of-fold** $R^2$ with a paired bootstrap interval —
cross-validated because adding features always raises in-sample $R^2$, which would
make the test vacuous. Verified to detect real signal (+0.336, CI [+0.302,
+0.369]) and to reject pure noise (−0.004, verdict "no evidence"). Alongside it,
the six strategies are compared at equal budget on held-out domains.

**Status: not yet answered on real agent data.** The machinery is implemented and
tested; the experiment is `configs/curriculum_baselines.yaml`. See
[Research status](#research-status--limitations).

---

## Quickstart

```bash
pip install -e ".[dev]"

# Full report from the shipped synthetic demo
python -m disteval evaluate examples/runs.json \
    --trajectories examples/trajectories.jsonl --tasks examples/tasks.json -o report/

# Per-task table, no files written
python -m disteval diagnose examples/runs.json --top 15

# Build preference pairs from a chosen strategy
python -m disteval select examples/runs.json \
    --trajectories examples/trajectories.jsonl \
    --strategy gap_plus_structure --n-pairs 100 -o pairs.jsonl

# The six-strategy curriculum comparison (built-in simulator)
python -m disteval experiment configs/curriculum_baselines.yaml --coupling weak
python -m disteval experiment configs/curriculum_baselines.yaml --coupling none  # sanity check

# Ablation sweep: method x shrinkage x evaluation budget
python -m disteval sweep configs/structure_ablation.yaml

# Validate the estimators against known ground truth
python -m disteval simulate --study all -o docs/validation

# Any metric's definition, assumptions, edge cases and prior work
python -m disteval metrics recoverability_headroom
```

## Example output

```
$ python -m disteval diagnose examples/runs.json --top 6

24 tasks: RECOVERABLE=13, SOLID=3, STUCK=2, UNCERTAIN=6
thresholds: tau_cap=0.15 tau_rel=0.9 tau_stuck=0.1 confidence=0.8 min_runs=3

task                         n  succ    post          95% CI    C_t    R_t  label         recov
-----------------------------------------------------------------------------------------------
access_review                8     3   0.389     [0.12,0.71]  0.950  0.000  RECOVERABLE   0.628
citation_audit               8     3   0.389     [0.12,0.71]  0.950  0.000  RECOVERABLE   0.628
incident_triage              8     3   0.389     [0.12,0.71]  0.950  0.000  RECOVERABLE   0.628
literature_sweep             8     3   0.389     [0.12,0.71]  0.950  0.000  RECOVERABLE   0.628
figure_reproduce             8     2   0.278     [0.06,0.59]  0.800  0.000  UNCERTAIN     0.615
sla_report                   8     2   0.278     [0.06,0.59]  0.800  0.000  UNCERTAIN     0.615

Note: only 10 distinct recoverability scores across 24 tasks (largest tie 5). Ranking within a tier is arbitrary.
```

That last line is the point: the framework reports when its own ranking is not
resolved rather than presenting an ordering the data cannot support.

## Repository structure

```
disteval/
  reliability/     posteriors, hierarchical/EB fits, criterion-level gap,
                   classification, rubric profiles, cost, scaling, comparison
  trajectory/      canonical events, alignment, divergence, embeddings,
                   counterfactual intervention distance
  diagnosis/       failure taxonomy, causality graphs, entropy, survival
  active/          value of information, adaptive allocation, stopping rules
  selection/       six curriculum selectors, pair construction, view exports
  experiments/     config, tracking, splits, A-D pipeline, sweeps
  sim/             ground-truth simulator and the estimator validation suite
  metrics.py       IQM, lower-tail CVaR, pass@k / pass^k
  metrics_spec.py  registry: definition, assumptions, edge cases, prior work
  plots.py         publication figures
  research_report.py
  right_tail.py    LEGACY max-based taxonomy, retained and marked deprecated
configs/           reproducible experiment and sweep configs
examples/          synthetic demo dataset + its generator
docs/validation/   committed outputs of the estimator validation suite
tests/             970 tests
```

## Backwards compatibility

Nothing has been removed. `right_tail.py` (`Q* = max`, `Δ = Q* − mean`, `κ`, the
max-based SOLID/RECOVERABLE/STUCK taxonomy), `self_engine`, `training_sim` and the
`report`/`compare`/`sim`/`engine`/`train` CLI subcommands all still work and still
pass their original tests.

`right_tail` is **marked legacy**: its docstring explains why the observed maximum
is not a capability estimate, and `TaskOutcomeProfile` now also carries the
posterior, the posterior label, $C_t$, $R_t$ and the gap, so migration is
incremental. The old selector names (`random`, `hardest`, `highest_variance`, …)
remain as working aliases and extra ablation arms.

## Tests

```bash
pytest -q     # 970 passed
```

Covering: posterior estimation and its edge cases (no runs, one run, all-success,
all-failure, unequal counts, tied and continuous scores); the mathematical sanity
checks (8/8 must beat 1/1; the continuous estimator must reduce exactly to the
Beta-Binomial); property-based invariants via Hypothesis (pass@k non-decreasing in
k, pass^k non-increasing, CVaR ordering, allocation never exceeding budget);
criterion-level gap behaviour including its non-monotonicity; all six selectors;
the four export views; incremental validity detecting signal and rejecting noise;
and the demo dataset and CLI end-to-end as a subprocess.

## Research status / limitations

**Validated** — against known ground truth in simulation, with committed outputs
in [`docs/validation/`](docs/validation):

- the observed maximum's bias grows from +0.20 at $k$=2 to +0.55 at $k$=32 and its
  RMSE *increases* with more data, while the posterior mean stays unbiased;
- 95% credible intervals cover 93–97%;
- RECOVERABLE precision is 0.94–1.00, with 20–43% of tasks left UNCERTAIN;
- hierarchical pooling helps on homogeneous suites (+0.029 nats/run) and hurts on
  heterogeneous ones (−0.046), which is why the choice is made per dataset;
- the incremental-validity test detects genuine signal and rejects pure noise.

**Not established.**

- **The central hypothesis has not been tested on real agent data.** The
  simulation results in THEORY.md §10 use a *modelled* training response and the
  legacy selector set; they show a gap-style signal can find high-benefit tasks
  where such tasks exist, not that failure structure adds value over difficulty
  and learning progress.
- No claim that preference training on real trajectories improves a real agent,
  nor that any improvement transfers out of domain.
- Honest negatives found along the way and left in: adaptive allocation reaches
  perfect RECOVERABLE recall but does **not** beat uniform on overall label
  agreement at equal budget; the uncertainty-corrected variant scored *worse* than
  the plain one under strong coupling; and with independent estimation at $n$ runs
  a gap ranking has at most $n{+}1$ distinct values, so a "top 40" list is partly
  arbitrary.
- The demo dataset is **synthetic** and labelled as such in its own metadata. No
  benchmark numbers are implied anywhere in this repository.
- The leave-one-out figure in `trajectory_monitor` is demo-scale (tens of
  correlated trials, no held-out set) and is labelled as a sanity check, not a
  result.

## Related work

This repository builds on, and does not claim, the following:

- **rliable** — Agarwal et al., *Deep RL at the Edge of the Statistical
  Precipice*, NeurIPS 2021. IQM, performance profiles, stratified bootstrap.
- **pass@k** — Chen et al., 2021. The unbiased estimator used here.
- **pass^k / τ-bench** — Yao et al., 2024. Reliability under repetition for
  tool-using agents.
- **ETO** — Song et al., *Trial and Error: Exploration-Based Trajectory
  Optimization for LLM Agents*, ACL 2024. Preference learning from success/failure
  agent trajectories.
- **CVaR** — Rockafellar & Uryasev, 2000.
- **Pólya-Gamma augmentation** — Polson, Scott & Windle, JASA 2013.
- **PQL variance bias** — Breslow & Clayton, JASA 1993.
- **Shrinkage** — Efron & Morris, JASA 1975.
- **Automatic curricula / learning progress** — Oudeyer & Kaplan; Graves et al.,
  *Automated Curriculum Learning for Neural Networks*, 2017.

The methodology is designed to be applicable to long-horizon professional-agent
benchmarks such as APEX. This is an independent research project, not affiliated
with or endorsed by Mercor, and it uses no proprietary data or infrastructure —
`load_runs` / `load_tasks` / `load_trajectories` define the generic shape such
data takes so an adapter is all that is needed.

## License

MIT.
