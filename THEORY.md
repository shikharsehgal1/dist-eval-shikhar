# Theory

This document states the statistical basis of the framework, the claims it
makes, and — importantly — the claims it does not make. Where an earlier version
of this document was wrong, the error is stated explicitly rather than quietly
edited out.

**Nothing here about repeated-run evaluation, Pass@k, Pass^k, CVaR, perturbation
testing, or preference learning from success/failure trajectories is novel.** All
of it is prior work, cited below. The one thing this repository puts forward as
its own contribution is stated in §7, and it is stated as a **hypothesis to be
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

## 7. Recoverability — the hypothesis

Everything above is estimation. This section is the claim under test.

> **Hypothesis.** Tasks where an agent has demonstrated meaningful capability but
> remains unreliable may provide more sample-efficient post-training data than
> randomly selected tasks or than simply the hardest tasks.

The intuition: on such a task, successful behaviour *already exists in the model's
trajectory distribution*. The model has produced a correct run; it just does not
do so consistently. Training then has a target that the policy can already reach,
and matched successful and failed trajectories on the same task differ in ways
attributable to the agent rather than to task difficulty. On a task the model has
never solved, there is no successful trajectory to contrast against and no
evidence the behaviour is within reach at all.

That is an argument, not a result. It could be wrong in at least three ways: the
recoverable tasks might be recoverable precisely because they are nearly solved
and have little headroom left; the failures might be irreducibly stochastic
(sampling temperature, environment flakiness) and carry no learnable signal; or
the improvement might not transfer beyond the tasks trained on.

### The estimator

Ranking tasks needs a scalar. Three are implemented and none is asserted to be
best:

**Posterior gap** — $C_t\,(1 - R_t)$. The literal reading. Simple and monotone in
the right directions, but it saturates: any task with clear capability and clear
unreliability scores near 1 regardless of how much reliability is missing.

**Expected headroom** (default) —

$$
\rho_t = \frac{\mathbb{E}\!\left[(\tau_{\text{rel}} - p_t)^{+} \cdot \mathbb{1}\{p_t > \tau_{\text{cap}}\}\right]}{\tau_{\text{rel}} - \tau_{\text{cap}}}
$$

the posterior-expected amount of reliability that is missing *given* evidence of
capability. It is largest for tasks demonstrably capable and far from the bar,
and small both for tasks already near the bar (little to gain) and for tasks with
no evidence of capability (the indicator removes them). It inherits the posterior's
uncertainty by construction.

**Evidence-weighted gap** — the posterior gap discounted by $n/(n + n_0)$, for
when you want ranking to be more conservative about low-$n$ tasks than the
posterior already is.

Trajectory-derived signals — embedding neighbourhood, failure-mode concentration,
counterfactual intervention distance — are combined with these in
`disteval.reliability.recoverability`. The combination is **not assumed to be
better**: `compare_signals` rank-correlates every signal against observed benefit
and reports which one actually predicts it. On synthetic data where the extra
signals are noise, the combined score correctly scores *worse* than expected
headroom alone.

### A structural limitation

With independent per-task estimation and $n$ binary runs, every posterior quantity
is a function of the success count alone, so recoverability takes **at most $n+1$
distinct values**. At $n = 8$, a 200-task suite yields 9 tiers with 38 tasks in
the largest, and a "top 40" list draws 13 of its members arbitrarily from a
33-way tie. `tie_diagnostics()` reports this and the generated report prints it.

Hierarchical estimation breaks ties with information rather than noise: on the
same data it raises distinct scores from 9 to 34 and cuts the tie at the cutoff
from 33 tasks to 9.

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

## 9. Training-data selection

Preference learning from paired successful and failed trajectories is prior work
(ETO, Song et al., 2024, and the broader trajectory-preference literature). This
repository does not claim it.

What is under test is **which tasks to build pairs from**. Pairs are constructed
*within* a task and, where the data supports it, within the same environment
variant, so instruction, environment and rubric are held fixed and the contrast is
behavioural. A "chosen" from an easy task against a "rejected" from a hard one
teaches task difficulty, not task competence.

The baselines are deliberately not strawmen:

* **random** — hardest to beat by accident.
* **hardest** — what most curricula actually do. Its weakness is the one the
  hypothesis targets: the hardest tasks are disproportionately ones never solved,
  so there is no successful trajectory to contrast against.
* **highest variance** — captures "inconsistent" directly from observed scores
  with no Bayesian apparatus at all. **If recoverability selection cannot beat
  this, the posterior machinery is not earning its place.**
* **generic success/failure** — the closest baseline to existing work, so it
  isolates exactly the contribution claimed: that ranking *within* that pool by
  estimated recoverability beats sampling from it uniformly.

---

## 10. What the simulation shows, and what it does not

The estimators can be validated exactly, because in simulation the truth is known.
The selection *hypothesis* can only be probed, because the response to training is
modelled rather than measured.

The simulator generates tasks with known $q_t$ (capability), $r_t$ (execution
reliability), $p_t = q_t r_t$, failure-mode distributions, and a true training
benefit $g_t$. The critical control is `benefit_coupling`, which sets how $g_t$
relates to true recoverability: `strong`, `weak`, `none`, or `adversarial` (where
$g_t$ is *anti*-correlated and the method should lose). **A simulator implementing
only the favourable case would make the method win by construction and prove
nothing.**

Fraction of the oracle's achievable benefit captured, 300 tasks, $k=8$, top 40
selected, mean ± se over 8 seeds:

| strategy | strong | weak | none | adversarial |
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

1. **Evaluate.** Run each task $k$ times. Estimate $p_{m,t}$ with uncertainty.
2. **Diagnose.** Classify, rank by recoverability, locate divergences.
3. **Select.** Build equal-sized preference datasets under each strategy,
   from a *training* split only.
4. **Train.** Preference-tune an open model on each dataset, identically.
5. **Re-evaluate.** On **held-out** tasks — ideally held-out *domains* — measuring
   mean, pass^k, lower-tail CVaR, and reliability. Repeat across dataset sizes to
   get a learning curve, since the claim is about *sample efficiency*, not final
   score.

The result that would support the hypothesis: at a fixed pair budget, the
recoverability-selected dataset produces a larger held-out reliability gain than
random, hardest, variance, and generic success/failure selection — with the
difference exceeding seed-to-seed variation, and surviving the `none`-coupling
sanity check.

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
