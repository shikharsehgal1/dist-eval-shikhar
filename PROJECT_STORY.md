# disteval: Distribution-First Evaluation and Self-Improvement for AI Agents

## Inspiration

Every AI agent benchmark we looked at reported a single number: mean reward. You'd see a leaderboard like:

| Agent | Score |
|-------|-------|
| Claude Code | 0.836 |
| Gemini CLI | 0.754 |
| Codex CLI | 0.300 |

And that's it. That's supposed to tell you something useful.

The problem hit us when we looked at the actual run data underneath one of those means. Gemini's 0.754 was hiding something: on easy tasks — the kind a junior developer handles in five minutes — Gemini's CVaR@0.1 was **0.000**. Not low. Zero. In its worst 10% of runs on easy tasks, Gemini scored nothing. The mean had completely obscured a tail collapse that would be catastrophic in any real deployment.

But the deeper insight came from looking at Codex. Its mean was 0.300 — looked like a weak agent. But when we ran the same tasks three times each, we found that Codex had solved `medium-rest-client` perfectly. Once. It also solved `easy-fizzbuzz` perfectly. Once. The agent demonstrably *knew* how to do these things. It just couldn't do them consistently.

That's a completely different problem from "the agent can't do this." And mean reward treats them identically.

That's what disteval is about.

---

## The Core Insight

Standard RL maximizes $J(\theta) = \mathbb{E}[R]$. When an agent runs the same task $k$ times, every attempt is averaged equally. A run that scores 1.0 and a run that scores 0.0 produce the same mean as two runs that score 0.5. The agent learns nothing about *why* it failed on the zero run, even though it clearly demonstrated the solution on the 1.0 run.

We define three quantities for each (agent, task) cell:

$$Q^*(t) = \max_i\, q_i \qquad \text{(demonstrated peak capability)}$$

$$\bar{Q}(t) = \frac{1}{k}\sum_i q_i \qquad \text{(what standard RL optimizes)}$$

$$\Delta(t) = Q^*(t) - \bar{Q}(t) \qquad \text{(recoverable gap)}$$

$$\kappa(t) = \frac{\bar{Q}(t)}{Q^*(t)} \qquad \text{(consistency index, } \kappa \in [0, 1]\text{)}$$

$\kappa = 1$ means the agent achieves its best performance every time. $\kappa = 0.33$ means the agent is leaving two-thirds of its demonstrated capability unrealized on every attempt.

From this, every task falls into exactly one of three classes:

| Class | Condition | Meaning |
|-------|-----------|---------|
| **SOLID** | $Q^* > 0,\ \Delta = 0$ | Agent is already consistent — nothing to do |
| **RECOVERABLE** | $Q^* > 0,\ \Delta > 0$ | Agent can do it but doesn't always — **train here** |
| **STUCK** | $Q^* = 0$ | Agent has never solved it — needs new capability |

The critical observation: for every **RECOVERABLE** task, a passing trajectory and a failing trajectory for the same task already exist in your eval data. That is a **DPO training pair**. No human labels. No synthetic data. The agent's own right tail *is* the training signal.

---

## What We Built

**disteval** is a Python library and CLI that implements this entire loop:

1. **Measure the full distribution** — IQM, CVaR@0.1, pass@$k$, pass$^k$ (deployment consistency), stratified bootstrap CIs, Wasserstein distance between agents.

2. **Classify every task** — compute $Q^*(t)$, $\bar{Q}(t)$, $\Delta(t)$, $\kappa(t)$ for every (agent, task) cell and assign SOLID / RECOVERABLE / STUCK.

3. **Rank by training leverage** — priority score $= \Delta(t) \times (1 - \kappa(t))$. Tasks with a large gap *and* low consistency come first.

4. **Extract training pairs** — for each RECOVERABLE task, find the trajectory files for the passing and failing runs. Identify the **structural divergence step** — the exact step where the two trajectories first made a different tool-call choice.

5. **Output a curriculum** — a JSON file with `reinforce_traj_path`, `contrast_traj_path`, and `structural_divergence_step` for each task, ready to feed into any DPO trainer.

The full pipeline runs in one command:

```bash
disteval engine jobs/run_1/ --agent my-agent --output plan.json
```

We also built:

- **Real-time trajectory monitoring** — the structural signature of an agent's tool-call sequence (write early vs. search forever) predicts final outcome with **89% leave-one-out accuracy** before the run finishes. An agent doing `web_search × 8` at step 3 already has $p(\text{success}) = 0.07$.

- **Cross-session trajectory memory** — instead of chronological memory, we index by task type × outcome. Before a new run, the agent retrieves the trajectories where it succeeded on similar tasks it normally fails — specifically surfacing RECOVERABLE-high runs.

- **Adapters for Harbor, Inspect (UK AISI), rliable**, and a generic JSONL format so any eval framework can feed disteval.

---

## What the Data Shows

On our benchmark (3 agents × 6 tasks × 3 attempts each):

| Agent | $\kappa$ | Recoverable gap | SOLID | RECOVERABLE | STUCK |
|-------|----------|-----------------|-------|-------------|-------|
| Claude Code | 1.000 | 0.00 | 6 | 0 | 0 |
| Gemini CLI | 0.885 | 0.50 | 2 | 3 | 1 |
| Codex CLI | 0.630 | 1.33 | 1 | 4 | 1 |

**Codex's consistency index is 0.63.** That means 37% of its theoretically achievable score is left on the table not because it lacks capability — it demonstrated those capabilities in specific runs — but because it can't reliably access them.

The right-tail objective is a finite-sample application of distributional RL. Instead of optimizing $\mathbb{E}[R]$, we optimize the upper-tail CVaR:

$$J_{rt}(\theta) = \mathbb{E}_t\left[\mathbb{E}\left[q \mid q \geq \text{VaR}_{1-\alpha}(F_t)\right]\right]$$

For a RECOVERABLE task, the gradient of $J_{rt}$ with respect to $\theta$ is approximately:

$$\frac{\partial J_{rt}}{\partial \theta} \approx -\mathbb{E}\left[\frac{\partial \log \pi(\tau_i)}{\partial \theta} \cdot \left(q_i - \text{VaR}_{1-\alpha}\right) \,\middle|\, q_i < \text{VaR}_{1-\alpha}\right]$$

This is a negative signal on below-threshold trajectories combined with a positive signal on above-threshold ones — which is exactly a contrastive training objective. DPO implements exactly this, and disteval produces the pairs automatically.

---

## How We Built It

The architecture has four layers:

**Records layer** — `EpisodeRecord` and `RecordStore`. The first principle was *never collapse early*. Every adapter preserves per-episode scores so that IQM, CVaR, and pass$^k$ can be computed on demand. Most eval frameworks aggregate before you can stop them; disteval lifts the raw scores back out.

**Metrics layer** — `metrics.py`, `bootstrap.py`, `compare.py`. Stratified bootstrap CIs resample *within* difficulty strata to preserve task heterogeneity. The rliable bridge converts to `{algo: ndarray[n_runs, n_tasks]}` format for compatibility.

**Right-tail layer** — `right_tail.py`. The core $Q^*$, $\bar{Q}$, $\Delta$, $\kappa$ computation. The SOLID/RECOVERABLE/STUCK classification. Priority ranking by $\Delta \times (1 - \kappa)$.

**Self-engine** — `self_engine.py`. Assembles everything: loads trajectories, runs right-tail analysis, finds the structural divergence step via tool-call sequence comparison, queries trajectory memory for similar past successes, and outputs the ranked JSON curriculum.

We tested against three real agents (Claude Code, Gemini CLI, Codex CLI) running six Harbor tasks across three attempts each — 54 episodes total, all real data, no synthetic inflation.

---

## Challenges

**The "failing run" problem.** Our first instinct for the demo was to show a live replay of a passing run vs. a failing run side by side. We spent a lot of time looking for a failure case where the agent had written *wrong code* (not just crashed or hit a network error). It turned out that in our dataset, most "failing" runs either never produced code at all or failed for infrastructure reasons. The cases where an agent wrote plausible-but-incorrect code and a passing run for the same task also existed were rare. We eventually removed the replay panel entirely — the data didn't support the story we wanted to tell with it.

**Mean reward's stickiness.** The hardest conceptual challenge was explaining why the right-tail signal is strictly better, not just "more information." The key is the task taxonomy: mean reward cannot distinguish RECOVERABLE from STUCK, and the intervention for each is completely different. RECOVERABLE tasks have a solution sitting in your `jobs/` directory right now. STUCK tasks need new capability. Treating them the same — which mean reward forces you to do — is the actual problem.

**Closing the loop without a training run.** We can generate the DPO pairs and rank them by leverage. What we can't yet show is $\kappa$ going from 0.63 to 0.85 after a training round, because we haven't run the full fine-tuning experiment end-to-end. The Monte Carlo simulation (showing +249% improvement vs. random trajectory selection) is the current evidence, but it's a bootstrap simulation over existing score distributions, not a live training result. That's the honest limitation.

---

## What We Learned

The distribution *is* the signal. Mean reward isn't a lossy compression of performance — it's a lossy compression that specifically loses the information you need most. The gap between pass@$k$ and pass$^k$ tells you more about an agent's deployment readiness than any single aggregate number. And the trajectories that sit in your eval data, discarded after the leaderboard is updated, contain the training pairs that could close that gap.

The right-tail framework connects three things that are usually treated separately: evaluation (how good is this agent?), diagnosis (where is it inconsistent?), and training (what data do we use to improve it?). disteval is the pipeline that connects all three.
