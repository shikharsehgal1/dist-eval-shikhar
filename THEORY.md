# Theory

This document states the statistical basis of the framework, the claims it
makes, and — importantly — the claims it does not make. Where an earlier version
of this document was wrong, the error is stated explicitly rather than quietly
edited out.

**Nothing here about repeated-run evaluation, Pass@k, Pass^k, CVaR, perturbation
testing, or preference learning from success/failure trajectories is novel.** All
of it is prior work, cited below. Nor is the **capability--reliability gap**
claimed as a novel metric: a gap between demonstrated and dependable performance
is the same idea as pass@k versus pass^k, as best-of-n versus single-sample, and
as "capability overhang" in ordinary evaluation practice. What is offered here is
a specific posterior definition of it at *criterion* granularity, and one
falsifiable question about it, stated in §7 and stated as a **hypothesis to be
tested**, not as a result.

---

## 1. Why a mean benchmark score is insufficient

An agent evaluated once per task produces one number per task, and a benchmark
reports their mean. That statistic is a summary of a distribution the evaluation
never looked at.

Two agents with an identical mean of 0.5 on a task:

```
Agent A:  0.5, 0.5, 0.5, 0.5, 0.5, 0.5     consistently half-right
Agent B:  1.0, 0.0, 1.0, 0.0, 0.0, 1.0     right half the time
```

These are different products. A is predictable and can be built on: you know what
you are getting. B cannot be deployed without a retry-and-verify wrapper, and
whether that wrapper is affordable depends on the cost of a wrong answer. The
mean is identical.

The distributional view is not new — `rliable` (Agarwal et al., 2021) made this
argument for deep RL, and it is why this repository reports IQM, quantiles and
tail statistics alongside the mean. What follows builds on that rather than
restating it.

---

## 2. Capability and reliability are different latent quantities

Write $p_t$ for the latent probability that the agent completes task $t$ on a
single attempt. Every run is a draw from $\mathrm{Bernoulli}(p_t)$; $p_t$ itself
is never observed.

Two questions about $p_t$ are usually conflated:

* **Capability** — is $p_t$ meaningfully above zero? Can the agent do this at all?
* **Reliability** — is $p_t$ above a deployment bar? Can it be relied on?

Both are probability statements about a quantity we can only estimate, so both
should be reported *as* probabilities, with their uncertainty:

$$
C_t = P(p_t > \tau_{\text{cap}} \mid D_t)
\qquad
R_t = P(p_t > \tau_{\text{rel}} \mid D_t)
\qquad
\tau_{\text{cap}} < \tau_{\text{rel}}
$$

$\tau_{\text{cap}}$ is deliberately low (default 0.15): it asks whether there is
real evidence the agent can do the task, not whether it is good at it.
$\tau_{\text{rel}}$ is a deployment bar (default 0.90). Neither is a universal
constant; both are configuration, and every reported label is conditional on them.

### The estimator

For binary outcomes, the conjugate model:

$$
p_t \sim \mathrm{Beta}(\alpha, \beta),
\qquad
s_t \mid p_t \sim \mathrm{Binomial}(n_t, p_t),
\qquad
p_t \mid s_t \sim \mathrm{Beta}(\alpha + s_t,\; \beta + n_t - s_t)
$$

The default prior is Jeffreys' $\mathrm{Beta}(\tfrac12,\tfrac12)$: weaker than
uniform (worth one pseudo-run rather than two, which matters a great deal at
$n_t = 3$) and invariant to reparameterisation.

### Continuous rubric scores

A Bernoulli likelihood does not apply to fractional rubric scores, and feeding
fractional "successes" into a Binomial is not a valid likelihood. The target is
instead the latent *mean score* $\mu_t$, estimated by a quasi-binomial
dispersion correction.

The Bernoulli distribution has the maximum variance of any distribution on
$[0,1]$ with a given mean, so the dispersion ratio

$$
\phi = \frac{s^2}{m(1-m)} \in [0, 1]
$$

says how much *less* variable the observed scores are than a Bernoulli with the
same mean. Each run therefore carries at least as much information about $\mu_t$
as a Bernoulli draw would, and the effective sample size is
$n_{\text{eff}} = n / \phi$, split into pseudo-counts $n_{\text{eff}} m$ and
$n_{\text{eff}}(1-m)$.

Three properties make this defensible rather than decorative:

1. **It reduces exactly to the Beta-Binomial on binary data** ($\phi \to 1$),
   verified in the tests.
2. **The correction points the right way.** Deterministic partial credit — every
   run scores exactly 0.5 — has $\phi \to 0$: the latent mean really is pinned
   down, and the posterior is correspondingly tight. Bimodal 0/1 behaviour has
   $\phi \to 1$: maximally uncertain, as it should be.
3. **Influence is bounded.** $\phi$ is clipped below, so a low-variance sample
   cannot manufacture unbounded confidence.

It is a quasi-likelihood, not a generative model. Read the interval as a
calibrated summary of the mean score, not as an exact posterior under a named
data-generating process. When you need a fully generative treatment, dichotomise
at a documented rubric threshold and use the binary path.

### Criterion level is the primary path

Where a rubric grades individual criteria, collapsing it to one number destroys
information before any analysis begins. A task with ten criteria where nine are
always satisfied and one is a coin flip has the same pass rate as one where all
ten are half-right; the first has a single localised execution defect and the
second suggests the agent does not know the task.

So the primary model is per criterion ``j`` of task ``t``:

$$
\mathrm{logit}(p_{t,j}) = \mu + \alpha_t + \beta_j + \gamma_{d(t)} + \varepsilon_{t,j}
$$

fitted by the crossed random-effects machinery of §6, with an empirical-Bayes
alternative (a moment-fitted Beta prior pooled per criterion across tasks, so
"did it verify its work" borrows strength wherever that criterion appears) and an
unpooled baseline for the ablations. Partial pooling matters more here than at
task level, and the interaction term $\varepsilon_{t,j}$ is what preserves "this
criterion is specifically unstable *on this task*".

The task-level estimators remain available and are what a benchmark reporting
only a scalar gets. The pipeline warns when it has to fall back to them.

---

## 3. Why the observed maximum is not a capability estimate

An earlier version of this repository used

$$
Q^*(t) = \max_i q_i(t)
$$

as "demonstrated capability". This is wrong, for two reasons.

**The maximum is biased upward and grows with the number of runs.** For $k$ i.i.d.
draws, $\mathbb{E}[\max]$ is strictly increasing in $k$, so the *same agent*
evaluated 20 times appears strictly more capable than evaluated 3 times. This is
the multiple-comparisons problem in its simplest form: with enough attempts,
something will look good by chance.

The bias is not a technicality. Measured in simulation against known ground truth
(`disteval simulate --study max_bias`, 400 tasks):

| runs $k$ | bias of $\max$ | RMSE of $\max$ | bias of posterior mean | RMSE of posterior mean |
|---|---|---|---|---|
| 2 | +0.204 | 0.479 | +0.020 | 0.228 |
| 5 | +0.429 | 0.533 | +0.014 | 0.168 |
| 8 | +0.486 | 0.554 | +0.006 | 0.140 |
| 16 | +0.521 | 0.573 | -0.000 | 0.104 |
| 32 | +0.551 | 0.595 | +0.002 | 0.074 |

<sub>Reproduce: `python -m disteval simulate --study max_bias -o docs/validation`.
The committed table is `docs/validation/max_bias.csv`.</sub>

The maximum's bias grows monotonically with $k$ and its error *increases* with
more data. The posterior mean's bias stays at zero and its error falls as
$O(1/\sqrt{k})$. More evaluation makes one estimator better and the other worse.

**A single lucky success declares mastery.** Under a max-based rule, runs
`[0, 0, 0, 0, 0, 0, 0, 1]` and `[1, 1, 1, 1, 1, 1, 1, 1]` both have
$Q^* = 1$. The posterior distinguishes them completely: $P(p_t > 0.9)$ is $0.000$
for the first and $0.813$ for the second.

The observed maximum is retained throughout the codebase as a **descriptive
statistic** and is labelled as such. It is never the estimate.

---

## 4. Pass@k and Pass^k measure different things

Both are prior work; both are computed here with their standard unbiased
estimators.

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
\qquad
\text{pass}^k = \frac{\binom{c}{k}}{\binom{n}{k}}
$$

for $c$ successes in $n$ runs (Chen et al., 2021 for pass@k; pass^k popularised
for agents by tau-bench, Yao et al., 2024).

* **Pass@k** — at least one of $k$ attempts succeeds. This is *peak capability
  under retry*: the right metric when failures are cheap and detectable, and you
  can afford to try again.
* **Pass^k** — *all* $k$ attempts succeed. This is *reliability*: the right metric
  when a single failure is costly, or when the task is one step of a longer
  pipeline that must complete.

Pass@k is non-decreasing in $k$; pass^k is non-increasing. Both are property-tested.
Their difference, the **reliability gap**, is the fraction of apparent capability
that does not reproduce, and on the demo dataset it is 0.87 at $k=8$ — pass@8 of
0.96 against pass^8 of 0.08.

Neither metric is novel and neither is claimed here.

---

## 5. Tail risk: a correction

**An earlier version of this document was wrong about this and the error is worth
stating plainly.**

It claimed that maximising *upper-tail* CVaR — $\mathbb{E}[q \mid q \ge
\mathrm{VaR}_{1-\alpha}]$, the expected score in the best $\alpha$ fraction of
runs — penalises unreliable low-scoring runs. It does not. Upper-tail CVaR
ignores the lower tail by construction.

Concretely, at $\alpha = 1/3$ on three runs:

```
scores [0, 0, 1]   upper-tail CVaR = 1.0
scores [1, 1, 1]   upper-tail CVaR = 1.0
```

An agent that fails two runs in three and one that never fails are
**indistinguishable** under that objective. That is the opposite of what a
reliability metric must do. This counterexample is now an executable test.

Reliability is a statement about the **bad** tail:

$$
\mathrm{CVaR}^{\text{lower}}_\alpha(q) = \mathbb{E}\!\left[q \mid q \le \mathrm{VaR}_\alpha(q)\right]
$$

the mean of the worst $\alpha$ fraction of runs (Rockafellar & Uryasev, 2000).
It is coherent, where VaR is not, and it responds monotonically to degradation in
the worst runs. On the two samples above it gives 0.0 and 1.0, correctly
separating them.

Optimising an upper-tail functional is a *risk-seeking* objective. It is a
reasonable thing to want when the goal is "reach the frontier at least
sometimes". It is not a reliability objective, and this repository no longer
describes it as one.

---

## 6. Partial pooling, and when it hurts

With three to eight runs per task, a per-task estimate is dominated by noise, and
*ranking* tasks by a noisy estimate invites the winner's curse: the top-ranked
task is disproportionately one that got lucky. Partial pooling addresses this by
shrinking each task toward the population:

$$
\mathrm{logit}(p_{m,t}) = \mu + \alpha_m + \beta_t + \gamma_{d(t)} + \varepsilon_{m,t}
$$

with independent Gaussian random effects for agent, task, domain, and the
agent-by-task interaction. The interaction term is what makes the model usable
here: without it, a task where one specific agent is unreliable would be pulled to
the population mean, destroying exactly the signal of interest.

Two backends: a fast Laplace/EM fit, and Pólya-Gamma Gibbs (Polson, Scott &
Windle, 2013) when the approximation needs checking. They agree at $r > 0.99$ on
simulated data.

Three implementation facts worth recording, because each was a bug:

**Identifiability.** With a single agent there is one cell per task, so $\beta_t$
and $\varepsilon_{m,t}$ index the same units and are perfectly aliased. Their sum
is identified; the split is not. Left in, the fit divides the true between-task
variance arbitrarily between them and over-shrinks everything. The interaction
term is dropped when it is not identified.

**Variance-component bias.** The moment-based EM update is the standard
penalized-quasi-likelihood M-step, and it is known to under-estimate variance
components for binary data with small clusters (Breslow & Clayton, 1993). It
recovered $\sigma = 1.82$ when the truth was 3.0. The M-step now maximises the
exact Gauss-Hermite quadrature marginal likelihood, which recovers 1.501 at a
true 1.5.

**Pooling is not free.** It helps when tasks are genuinely similar and *hurts*
when the task-effect distribution is far from Gaussian — a cluster of
mostly-similar tasks plus a few extremes is exactly the bad case, since the fitted
between-task variance comes out small and the extremes get dragged inward.
Measured by leave-one-run-out predictive log-likelihood: $+0.029$ per run on
homogeneous tasks, $-0.046$ per run on heterogeneous ones. The framework therefore
**decides per dataset** rather than asserting a default, and reports the decision.

Note the honest caveat: that diagnostic optimises predictive accuracy on the next
run, which is the only thing measurable without ground truth. It agrees with
accuracy against the latent $p_t$ about two thirds of the time.

---

## 7. The capability--reliability gap, and the hypothesis

### The metric

At a capability threshold $\tau_{\text{cap}}$ and a deployment threshold
$\tau_{\text{rel}}$, per criterion:

$$
c_{t,j} = P(p_{t,j} > \tau_{\text{cap}} \mid D)
\qquad
r_{t,j} = P(p_{t,j} > \tau_{\text{rel}} \mid D)
$$

and per task:

$$
C_t = \frac{1}{J}\sum_j c_{t,j}
\qquad
R_t = \frac{1}{J}\sum_j r_{t,j}
\qquad
G_t = C_t - R_t
$$

$G_t \in [0,1]$ is the fraction of the rubric the agent can demonstrably satisfy
but does not satisfy dependably. No observed maximum appears anywhere in it, so
it does not grow with the number of runs, and one lucky success on one criterion
moves it only as far as one run of evidence warrants.

**This is not claimed to be a novel metric.** It is a posterior restatement, at
criterion granularity, of a comparison the field already makes.

### Three properties that are easy to get wrong

**$G_t$ is not monotone in performance.** It is an inverted U: a criterion never
satisfied has $c \approx 0, r \approx 0$ and hence near-zero gap; one always
satisfied has $c \approx 1, r \approx 1$ and also near-zero gap; the gap peaks in
between. Consequently ranking by $G_t$ is *not* ranking by difficulty in either
direction, and — less obviously — **shrinkage can reorder a gap ranking rather
than merely compress it**, since compressing $p$ does not compress a non-monotone
function of $p$ proportionally. Measured on a 20-task synthetic corpus, the
unpooled and empirical-Bayes rankings correlate at $\rho = 0.73$ while the full
hierarchical fit can reorder them substantially. Neither is wrong. The estimator
choice is a substantive modelling decision and must be reported with the ranking.

**Criterion-level and task-level capability differ.** A task can score high on
$C_t$ — every criterion individually within reach — while the agent has never once
satisfied all of them together. On the demo data, `precedent_search` has the
*highest* criterion gap (0.75) and a joint capability of 0.10. That is a real and
useful finding, not an artefact; a task-level score reports it only as "always
fails". `joint_reliability` and `joint_capability` are reported alongside, from
the observed all-satisfied count rather than by multiplying per-criterion
posteriors, since criteria are not independent and the product is badly biased.

**A high $G_t$ says nothing on its own about training value.** That is the
hypothesis, not a property of the metric.

### The hypothesis

> **Does within-task failure structure add predictive value for training-data
> selection beyond difficulty and learning progress?**

Note what is deliberately *not* claimed.

It is **not** claimed that tasks with intermediate success rates are inherently
more trainable. That is a plausible intuition with obvious failure modes: such a
task may be nearly solved with little headroom left; its failures may be
irreducibly stochastic (sampling temperature, environment flakiness) and carry no
learnable signal; and any gain may not transfer beyond the tasks trained on.
Treating the intuition as established would be precisely the error this framework
exists to avoid.

The claim under test is narrower and falsifiable: that knowing *how* a task fails
— which criterion, how consistently, how far the failed runs sit from the
successful ones, how small an intervention separates them — predicts training
value **over and above** knowing how hard the task is and how fast it is
currently improving.

### How the hypothesis is tested

Two ways, because they answer it differently.

**Incremental validity.** Regress observed training benefit on the base features
(difficulty, learning progress) and on base + structure, comparing **out-of-fold**
$R^2$ with a paired bootstrap interval on the difference. Cross-validated, not
in-sample, for a load-bearing reason: adding features can only increase in-sample
$R^2$, so an in-sample comparison would be guaranteed to "support" the hypothesis
and would mean nothing. Repeated k-fold, because a single split over a few
hundred tasks is noisy enough to flip the sign. The verdict is positive only when
the interval excludes zero. Verified to detect genuine signal ($+0.336$,
CI $[+0.302, +0.369]$) and to reject pure noise ($-0.004$, verdict "no evidence").

**A curriculum comparison at equal budget.** Six strategies, evaluated on
held-out domains, at a fixed number of training pairs — and reported **per unit of
training data**, since the claim is about sample efficiency rather than final
score. A strategy that merely yielded more pairs would otherwise look better for
the wrong reason.

### A structural limitation

With independent per-task estimation and $n$ binary runs, every posterior quantity
is a function of the success count alone, so a task-level gap takes **at most
$n+1$ distinct values**. At $n = 8$, a 200-task suite yields 9 tiers with 38 tasks
in the largest, and a "top 40" list draws 13 of its members arbitrarily from a
33-way tie. `tie_diagnostics()` reports this and the generated report prints it.
Criterion-level estimation and hierarchical pooling both break ties with
information rather than noise: on the same data, pooling raises distinct scores
from 9 to 34 and cuts the tie at the cutoff from 33 tasks to 9.

---

## 8. Classification

The SOLID / RECOVERABLE / STUCK taxonomy is retained because it is intuitive. The
*definitions* are replaced. The legacy rules keyed off the observed maximum:

```
SOLID        max(q) > 0 and max(q) == mean(q)
RECOVERABLE  max(q) > 0 and max(q) >  mean(q)
STUCK        max(q) == 0
```

so a task's label depended on how many times it happened to be run, and a task run
*once* was always SOLID or STUCK — never uncertain.

The posterior definitions, with $\delta$ the required confidence:

| label | condition | reading |
|---|---|---|
| SOLID | $C_t \ge \delta$ and $R_t \ge \delta$ | capable and reliable |
| STUCK | $P(p_t < \tau_{\text{stuck}}) \ge \delta$ | no evidence of capability |
| RECOVERABLE | $C_t \ge \delta$ and $R_t \le 1 - \delta$ | capable, demonstrably not reliable |
| UNCERTAIN | otherwise, or $n_t <$ `min_runs` | too few runs to say |

UNCERTAIN is a common and correct outcome at $n = 2$–$3$, and that is the point.
The discrete label never travels without $C_t$, $R_t$, the posterior mean and the
credible interval.

Measured against ground truth (`--study classification`, 400 tasks;
`docs/validation/classification.csv`):

| runs $k$ | UNCERTAIN rate | RECOVERABLE recall | RECOVERABLE precision | SOLID misread as RECOVERABLE |
|---|---|---|---|---|
| 2 | 1.000 | 0.000 | 0.000 | 0 |
| 3 | 0.425 | 0.594 | 0.935 | 1 |
| 5 | 0.385 | 0.666 | 0.980 | 3 |
| 8 | 0.325 | 0.616 | 0.991 | 0 |
| 16 | 0.250 | 0.790 | 0.997 | 0 |
| 32 | 0.205 | 0.848 | 0.997 | 0 |

Precision on the RECOVERABLE class — the one that feeds selection — is high
(0.94–1.00), which is what matters: the cost of a false positive is training on a
task that had nothing to teach. Recall is much lower (0.59–0.85), so the method
misses genuinely recoverable tasks, and the UNCERTAIN rate stays at 20–43% even at
$k=32$. That last number is the price of not guessing, and it is reported rather
than hidden. At $k=2$ *everything* is UNCERTAIN by construction, since the default
`min_runs` is 3.

Thresholds interact with sample size in ways worth knowing before choosing them:
asserting $P(p < 0.10) \ge 0.80$ takes about 8 consecutive failures under a
Jeffreys prior, and $P(p > 0.90) \ge 0.95$ takes about 20 consecutive successes.
**High reliability is expensive to certify, and no estimator makes it cheap.**

---

## 9. Curriculum selection, and the optimiser

Preference learning from paired successful and failed trajectories is prior work
(ETO, Song et al., 2024, and the broader trajectory-preference literature). This
repository does not claim it, and **DPO is one optional backend rather than the
framework's objective.**

### Six strategies

| strategy | ranks by |
|---|---|
| `uniform` | nothing — the control |
| `difficulty` | lowest estimated performance |
| `uncertainty` | largest posterior sd |
| `learning_progress` | observed progress where a training history exists, else the learnability proxy $\mathbb{E}[p(1-p)]$ |
| `capability_reliability_gap` | criterion-level $G_t$ |
| `gap_plus_structure` | $G_t$ weighted by within-task failure structure |

`learning_progress` is PAC-flavoured in the standard automatic-curriculum sense:
with a history it is absolute learning progress, the change between an earlier and
a recent window. Without one — the usual case for a one-shot evaluation — progress
cannot be *observed*, so it falls back to $\mathbb{E}[p(1-p)]$, the
posterior-expected outcome variance, which is the Bernoulli Fisher information up
to a constant and the standard zone-of-proximal-development quantity. **The
fallback measures potential, not progress**, and every result records which was
computed.

The comparison that settles the hypothesis is `gap_plus_structure` against
`difficulty` and `learning_progress` — **not** against `uniform`, which is too
weak a baseline to support the claim.

### Structure

"Within-task failure structure" is operationalised as three signals, each computed
only where the data supports it and left absent rather than imputed otherwise:
failure-mode concentration (does it fail the same way every time?), embedding
neighbourhood (how close failed runs sit to successful ones, normalised by the
spread of the successes), and counterfactual intervention cost (how small an edit
turns a failed run into an observed successful one).

### Pairs, and objectives

Pairs are constructed *within* a task and, where the data supports it, within the
same environment variant, so instruction, environment and rubric are held fixed
and the contrast is behavioural. A "chosen" from an easy task against a
"rejected" from a hard one teaches task difficulty, not task competence.

The selected data is then exported through **views** — `pairwise` (DPO, IPO,
SLiC), `listwise`, `weighted_sft`, `scalar_reward` (PPO/GRPO, reward models) — all
rendered from the same selection, so a comparison across objectives is not
confounded by a different data pipeline. `weighted_sft` is included as a control:
if plain supervised fine-tuning on the same selection matches a preference loss,
the preference machinery is not what is doing the work, and that is worth knowing.

Every export carries the dataset's training cost, so held-out gain is reported per
100 examples and per 1000 tokens as well as absolutely.

---

## 10. What the simulation shows, and what it does not

The estimators can be validated exactly, because in simulation the truth is known.
The selection *hypothesis* can only be probed, because the response to training is
modelled rather than measured.

The simulation results below predate the current selector set: they were produced
with the legacy task-level `recoverability` selector, which the
`capability_reliability_gap` selector supersedes. They are retained because the
*methodological* points they establish are unchanged and load-bearing -- that the
simulator must include worlds where the method fails, that single-seed runs of
this study are not interpretable, and that the uncertainty correction is not free.
The six-strategy comparison under the current selectors is the experiment
`configs/curriculum_baselines.yaml` runs; it has not been run at scale here.

The simulator generates tasks with known $q_t$ (capability), $r_t$ (execution
reliability), $p_t = q_t r_t$, failure-mode distributions, and a true training
benefit $g_t$. The critical control is `benefit_coupling`, which sets how $g_t$
relates to true recoverability: `strong`, `weak`, `none`, or `adversarial` (where
$g_t$ is *anti*-correlated and the method should lose). **A simulator implementing
only the favourable case would make the method win by construction and prove
nothing.**

Fraction of the oracle's achievable benefit captured, 300 tasks, $k=8$, top 40
selected, mean ± se over 8 seeds:

| strategy (legacy selector set) | strong | weak | none | adversarial |
|---|---|---|---|---|
| recoverability | **+0.322 ± 0.028** | **+0.193 ± 0.039** | +0.019 ± 0.064 | -0.282 ± 0.036 |
| uncertainty-aware | +0.239 ± 0.066 | +0.149 ± 0.055 | +0.035 ± 0.036 | -0.279 ± 0.033 |
| highest variance | +0.127 ± 0.054 | +0.069 ± 0.037 | +0.002 ± 0.025 | -0.189 ± 0.060 |
| success/failure | +0.115 ± 0.025 | +0.085 ± 0.013 | +0.030 ± 0.018 | -0.093 ± 0.033 |
| random | -0.063 ± 0.023 | -0.041 ± 0.038 | -0.006 ± 0.039 | +0.082 ± 0.022 |
| hardest | -0.227 ± 0.063 | -0.151 ± 0.031 | -0.025 ± 0.022 | **+0.247 ± 0.080** |

<sub>`docs/validation/selection.csv`. Reproduce:
`python -m disteval simulate --study selection -o docs/validation`.</sub>

Read the `none` column first. Recoverability selection is **indistinguishable from
random** there, well within one standard error of zero, which is the correct result: benefit is
unrelated to recoverability in that world by construction, and a method that
appeared to win would be broken. Under `adversarial` it correctly loses, and
hardest-first correctly wins.

This matters procedurally: the *first single-seed run* of this study showed
recoverability apparently beating random in the `none` world at $+0.18$. Eight
seeds revealed it as sampling noise. Replication is now built into the function
signature rather than left to the caller.

Two honest negatives from the same suite:

* **Uncertainty correction is not free.** Uncertainty-aware recoverability is
  *worse* than plain recoverability under strong coupling (0.226 vs 0.322).
* **Adaptive allocation does not dominate.** It reaches perfect RECOVERABLE recall
  where uniform reaches 0.85–0.96, but does not beat uniform on overall label
  agreement at equal budget.

### The limits of this evidence

The simulated training backend applies an analytic response model driven by the
*true* $g_t$ — never by the selector's own score, which would be circular. So
these results establish that **the recoverability signal identifies tasks with
high true benefit, when such tasks exist and are identifiable from run outcomes**.

They do **not** establish that preference training on real trajectories improves a
real agent, nor that the improvement transfers out of domain. Those require real
runs and a real trainer. The framework exports the datasets and provides the
splits to run that experiment; it has not been run here. See §12.

---

## 11. Sequential and structural diagnosis

Two policies with identical pass rates can fail very differently. Each run is
therefore also treated as a survival process over its step clock, with hazard
$h_j = P(\text{irrecoverable failure at } j \mid \text{survived to } j)$,
estimated by Kaplan-Meier with Greenwood standard errors.

Right-censoring is the reason for using KM rather than a histogram of failure
steps: the histogram silently conditions on the run having failed, and will report
that late steps are safe merely because few runs reach them.

Failures are attributed to a stage-ordered taxonomy and assembled into per-run
causal DAGs. **No causal discovery is performed and none is claimed.** An edge is
proposed only where temporal order, pipeline-stage order, *and* an observable
dependence all hold, and each edge carries the reason it was proposed. These are
necessary conditions for causation, not sufficient ones. The output should be read
as "these are the chains consistent with the evidence".

The practical payoff is rubric attribution: if one wrong retrieval causes five
criteria to fail, counting five capability deficits overstates the problem
fivefold, and the amplification factor is reported.

---

## 12. The experiment this framework is built to run

On a benchmark with repeated runs and trajectory logs:

1. **Evaluate.** Run each task $k$ times, allocating reruns adaptively toward the
   tasks whose posteriors are least settled. Estimate $p_{t,j}$ per criterion with
   uncertainty.
2. **Diagnose.** Compute $C_t$, $R_t$, $G_t$ and the gap's concentration; align
   successful against failed trajectories; locate the earliest consequential
   divergence; attribute failure modes and measure their entropy and the
   trajectory distance between success and failure.
3. **Select.** Build equal-sized datasets under all six strategies, from a
   *training* split only.
4. **Train.** Fine-tune an open model on each dataset, identically, through one
   fixed objective — and ideally repeat with a second objective from a different
   view, since the selection claim should not depend on the optimiser.
5. **Re-evaluate.** On **held-out** tasks — ideally held-out *domains* — measuring
   mean, pass^k, lower-tail CVaR and criterion-level reliability, reported **per
   unit of training data**. Repeat across dataset sizes for a learning curve,
   since the claim is about sample efficiency, not final score.
6. **Test incremental validity.** Regress observed benefit on difficulty and
   learning progress, then on those plus the structure features, and compare
   out-of-fold $R^2$.

The result that would support the hypothesis: at a fixed pair budget and
normalised per unit of training data, `gap_plus_structure` produces a larger
held-out reliability gain than `difficulty` and `learning_progress` — with the
difference exceeding seed-to-seed variation, corroborated by a positive
incremental-validity interval, and surviving the `none`-coupling sanity check.

**This experiment has not been run on real agent data in this repository.** The
machinery to run it is here and tested; the result is not claimed.

---

## References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G.
  (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice.
  *NeurIPS*. (rliable; IQM, performance profiles, stratified bootstrap)
- Breslow, N. E., & Clayton, D. G. (1993). Approximate Inference in Generalized
  Linear Mixed Models. *JASA*. (the PQL variance-component bias)
- Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code.
  (the unbiased pass@k estimator)
- Efron, B., & Morris, C. (1975). Data Analysis Using Stein's Estimator and Its
  Generalizations. *JASA*. (shrinkage; why partial pooling lowers total error)
- Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*. Cambridge University Press.
- Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian Inference for
  Logistic Models Using Pólya-Gamma Latent Variables. *JASA*.
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional
  Value-at-Risk. *Journal of Risk*.
- Song, Y., et al. (2024). Trial and Error: Exploration-Based Trajectory
  Optimization for LLM Agents (ETO). *ACL*. (preference learning from
  success/failure agent trajectories — prior work this repository builds on)
- Yao, S., et al. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in
  Real-World Domains. (pass^k for agent reliability)
