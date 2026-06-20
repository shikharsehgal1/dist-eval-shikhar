# The Right-Tail Training Signal

## The problem with mean reward in agentic evals

Standard RL maximizes expected reward: J(θ) = E[R]. When an agent runs the
same task k times, every attempt is averaged equally. A run that scores 1.0
and a run that scores 0.0 produce the same gradient signal as two runs that
score 0.5. The agent learns nothing about *why* it failed on the zero run
even though it clearly knows how to solve the task — it just demonstrated
that on the 1.0 run.

This is the **consistency problem**: the agent has the capability but can't
reliably access it. Mean reward treats this as equivalent to a genuine
capability gap, and the gradient signal points in the wrong direction.

---

## Formal setup

Let π_θ be the agent policy. On task t, it produces outcomes drawn from
distribution F_t(q; θ). With k attempts we observe q_1, ..., q_k ~ F_t.

Define:

```
Q*(t)   = max_i q_i         right tail: demonstrated best on task t
Q̄(t)    = (1/k) Σ q_i      mean: what standard RL optimizes
δ_i(t)  = Q*(t) - q_i       right-tail residual for attempt i  ≥ 0
Δ(t)    = Q*(t) - Q̄(t)      right-tail gap for task t          ≥ 0
κ(t)    = Q̄(t) / Q*(t)      consistency index                  ∈ [0,1]
```

**Standard RL objective:**

```
J_mean(θ) = E_t [ E_{q ~ F_t} [q] ]  =  E_t [ Q̄(t) ]
```

**Right-tail objective:**

```
J_rt(θ)   = E_t [ CVaR_{1-α}(F_t) ]
           = E_t [ E[q | q ≥ VaR_{1-α}(F_t)] ]
```

where CVaR_{1-α} is the *upper*-tail conditional value at risk — the
expected score in the top-α fraction of outcomes.

---

## Why J_rt is better for agentic tasks

### 1. It specifically penalizes variance

J_mean can be improved by a single lucky run. J_rt requires *consistently*
high scores because the upper tail expectation rises only when low outcomes
are lifted toward the high ones — a lucky outlier doesn't help if the rest
are zero.

For an agent with outcomes [0, 0, 1] on a task:
- J_mean contribution: 0.333
- J_rt (top-33%) contribution: 1.0 (only the 1.0 run contributes)

J_rt rewards the 1.0 run but penalizes the agent for not replicating it.
The gradient of J_rt points specifically at the failed runs.

### 2. The recoverable-gap gradient

For a RECOVERABLE task (Q*(t) > 0, Δ(t) > 0), the gradient of J_rt
with respect to θ is approximately:

```
∂J_rt/∂θ ≈ E[ ∂ log π(τ_i)/∂θ · (q_i - VaR_{1-α}) | q_i < VaR_{1-α} ] · (-1)
```

This is a negative signal on the *below-threshold trajectories*. Combined
with the positive signal on above-threshold trajectories, this is exactly
a **contrastive training objective**: reinforce the high runs, contrast the
low runs of the *same task*.

The contrastive pair is derived entirely from the agent's own eval data.
No human labels. No separate scoring model. The agent's own right tail
on each task is the positive example.

### 3. Natural task taxonomy

Tasks sort into three categories under this framework:

```
SOLID        Q*(t) > 0,  Δ(t) = 0   consistency κ = 1.0
             → no signal needed; agent is already consistent

RECOVERABLE  Q*(t) > 0,  Δ(t) > 0   consistency κ < 1.0
             → right-tail signal is maximally informative
             → agent CAN do this; train it to do it reliably

STUCK        Q*(t) = 0               consistency undefined
             → no demonstrated upper bound to pull toward
             → needs new capabilities, not consistency training
```

The right-tail signal is zero for SOLID (nothing to improve), undefined for
STUCK (no target to aim at), and **maximally informative for RECOVERABLE**.
This gives a natural curriculum: address recoverable tasks first, then
invest in the stuck ones.

### 4. Connection to distributional RL

This framework is a finite-sample application of distributional RL
(Bellemare et al., 2017; Dabney et al., 2018). Instead of representing the
full return distribution during training, we:

1. Collect k attempts per task in eval
2. Estimate the agent's per-task return distribution empirically
3. Use the right tail of that distribution as the training target

The key insight from distributional RL that applies here: **the full
distribution of returns carries more information than the mean**, and
optimizing a risk-seeking functional (upper CVaR) of that distribution
produces policies with better tail performance than optimizing the mean.

disteval makes this concrete and *measurable*: we show exactly which tasks
are recoverable, exactly how large the gap is, and exactly which
trajectories to use as positive vs negative examples.

---

## The consistency index κ as a diagnostic

κ(t) = Q̄(t) / Q*(t) measures how consistently the agent achieves its
demonstrated best on task t. Across tasks:

```
κ_total = Σ_t Q̄(t) / Σ_t Q*(t)   (aggregate consistency)
```

This is the fraction of *achievable performance* that the agent actually
captures on average. The gap (1 - κ_total) is the fraction of performance
left on the table due to inconsistency alone — assuming the agent is never
trained on new tasks.

From the real benchmark data (3 attempts per task, 6 tasks):

```
Claude Code:  κ = 1.000  (0.00 score left as inconsistency)
Gemini CLI:   κ = 0.885  (0.50 score points recoverable)
Codex CLI:    κ = 0.630  (1.33 score points recoverable = 37% of peak)
```

Codex's gap is large: 37% of its theoretically achievable score is left on
the table not because it can't do the tasks (it solved medium-rest-client
and easy-fizzbuzz perfectly at least once each) but because it can't do
them consistently.

---

## What this looks like in practice

For Codex on `medium-rest-client`:
- Attempts: [0.0, 0.0, 1.0]
- Q* = 1.0, Q̄ = 0.333, Δ = 0.667, κ = 0.333

The agent solved this task perfectly on attempt 3. Attempts 1 and 2
scored zero. Under right-tail training:

- **Reinforce**: trajectory #2 (score=1.0) — this is what correct behavior
  looks like
- **Contrast**: trajectories #0 and #1 (score=0.0) — these are the failure
  modes to move away from

If the trajectories are inspected, the difference between attempt 3 and
attempts 1-2 is the *actual* information needed to improve the agent. The
mean reward gradient would have averaged these three runs into a 0.333
signal pointing weakly upward, diluting the strong positive signal from
attempt 3 with noise from attempts 1 and 2.

---

## Why this solves "getting the same thing wrong repeatedly"

The standard RL loop with mean reward has no mechanism to distinguish:

(a) Agent scored 0.5 on every attempt → genuine partial capability
(b) Agent scored [1, 0, 0] → full capability, access is random
(c) Agent scored [0.5, 0.5, 0.5] → stuck in local maximum

All three produce the same mean reward. The right-tail framework
distinguishes them precisely:

(a) Q* = 0.5, Δ = 0   → SOLID (at its best consistently, best = 0.5)
(b) Q* = 1.0, Δ = 0.67 → RECOVERABLE (train it to access attempt #0's behavior)
(c) Q* = 0.5, Δ = 0   → SOLID (same as (a) — improving this requires new skill)

Case (b) — which is exactly "gets the same thing wrong repeatedly despite
knowing how" — gets a targeted training signal under J_rt. It doesn't under
J_mean.

---

## Summary

| Property | J_mean | J_rt |
|---|---|---|
| Uses full outcome distribution | No | Yes |
| Penalizes inconsistency | No | Yes |
| Distinguishes recoverable vs stuck | No | Yes |
| Provides per-trajectory training signal | No | Yes (reinforce/contrast) |
| Natural training curriculum | No | Yes (solid → recoverable → stuck) |
| Requires human labels | No | No |
| Requires model changes | No | No (post-hoc eval signal) |

The right-tail gap Δ(t) and consistency index κ are computable from any
benchmark that runs k > 1 attempts per task. They require no changes to the
agent, the training loop, or the benchmark. They are a pure function of the
eval data that standard leaderboards discard.

---

## References

- Bellemare, M. G., Dabney, W., & Munos, R. (2017). A distributional perspective
  on reinforcement learning. ICML.
- Dabney, W., et al. (2018). Distributional reinforcement learning with quantile
  regression. AAAI.
- Agarwal, R., et al. (2021). Deep reinforcement learning at the edge of the
  statistical precipice. NeurIPS. (rliable; IQM and performance profiles)
- Cobbe, K., et al. (2021). Training verifiers to solve math word problems.
  (pass@k formulation for LLMs)
