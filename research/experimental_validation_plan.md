# Experimental Validation Plan for disteval Innovations

**Goal:** Define a reproducible experimental programme that demonstrates whether
each disteval innovation is *better* than the relevant baseline, and by how much,
under realistic conditions. The plan is methodology-first: simulations and
proxy experiments that can be run without expensive re-training of frontier
models, but with clear hooks for real-agent follow-ups.

**Source:** This plan was produced by dispatching five parallel research subagents
to design detailed protocols for each innovation area, then synthesizing their
output into a single coherent programme. The subagents read the current disteval
codebase and produced the hypotheses, baselines, variables, protocols, and code
sketches that appear below.

## Principles

- **No new runtime dependencies.** Use only numpy, pandas, scipy, matplotlib and
the existing disteval code base.
- **Simulation-first, real-agent-ready.** Most experiments start with
controlled synthetic data or the existing `training_sim.py`; each protocol states
how to swap in real Harbor/Inspect runs.
- **Statistical rigor.** Report confidence intervals, repeat runs, and false
positive/negative rates where possible.
- **Reproducibility.** Every experiment is a script under `research/experiments/`
with a deterministic seed and a written log of expected outputs.

## Innovation map

| Exp | Innovation | Claim | Baseline | Key metric |
|-----|------------|-------|----------|------------|
| 1 | Distribution-first metrics | Detect tail risk / inconsistency mean hides | Mean-only reporting | Spearman ρ vs mean, κ, CVaR, pass^k |
| 2 | RECOVERABLE training | Higher gain per example than random/hardest | Random, top-K, all-task, SOLID-only | Score gain per example, rounds to threshold |
| 3 | SelfEngine curriculum | Automated ranking near oracle | Random, human, top-K-by-gap | Kendall τ vs oracle, Δκ, rounds to threshold |
| 4 | Trajectory monitor | Predict final outcome mid-episode | Majority, first-step, random, heuristic | LOO accuracy, AUC-ROC, early-stop regret |
| 5 | Trajectory memory | Retrieval boosts success rate | No memory, random, chronological | Success rate boost, especially on RECOVERABLE |
| 6 | Recursion engine | Decompose STUCK tasks into solvable sub-tasks | Flat retry, random decomposition | Sub-task solve rate, parent graduation |
| 7 | Distributed evaluation | Reduce eval noise and false positives | Single-agent single-run | Meta CI width, underconfidence ratio, P(A>B) |
| 8 | Agent harness | Lower integration cost, complete records | Manual JSONL, ad-hoc adapter | LOC, error rate, iteration count, completeness |
| 9 | Training simulation | Predict rank-order of gains | No-op, mean-only, random, oracle | Spearman ρ predicted vs actual, MAE |

## Common experimental substrate

All experiments reuse the same canonical data sets where possible:

- **Real data:** Existing `jobs/` and `tasks/` used in the repo's tests and
README examples.
- **Synthetic data:** `disteval.training_sim` can generate agents with known
gap, consistency, and learning curves.
- **Benchmark tasks:** The `tasks/` directory contains tasks of varying
difficulty that already have Harbor trajectories.

The preferred evaluation flow is:

```
synthetic proof-of-concept  →  real Harbor/Inspect validation  →  real training follow-up
```

## Meta-analysis plan

After individual experiments, aggregate:

- A **scorecard** of effect sizes and confidence intervals for each innovation.
- A **decision tree** telling a practitioner which innovation to use when.
- A list of **real-agent follow-up experiments** needed to validate simulation
findings.

The scorecard threshold for "validated" is:

- **Simulation:** p < 0.05, Cohen's d > 0.5 (medium effect), and at least one
baseline beaten.
- **Real data:** same statistical threshold, plus qualitative agreement with the
simulation direction.

---

## Experiment 1 — Distribution-first metrics vs mean-only reporting

**Hypothesis:** Mean-only reporting ranks agents identically even when their
tail behaviour differs. IQM, CVaR@0.1, pass^k and the consistency index κ
separate agents with identical mean but different reliability.

**Baselines:** mean-only ranking, median-only ranking, std-only ranking.

**Independent variables:** agent profile (consistent, inconsistent, tail-heavy),
number of tasks (6, 12), number of attempts per task (3, 5, 10), mean score
level (0.30, 0.50, 0.70).

**Dependent variables:** Spearman ρ between mean ranking and each distribution
metric; Cohen's d between consistent and inconsistent profiles; κ; CVaR@0.1;
pass@k and pass^k for k ∈ {1, 4, 8}; bootstrap 95% CI width.

**Protocol:**
1. Generate three synthetic agent profiles with mean ≈ 0.50 but different
tail shapes (truncated normal, Bernoulli 0/1, shifted Beta).
2. Build `RecordStore` with `EpisodeRecord`s per task and attempt.
3. Compute all metrics using `disteval.metrics` and `disteval.right_tail`.
4. Rank agents by each metric and compute Spearman/Kendall agreement with the
mean-based ranking.
5. Run sensitivity analysis on attempts, tasks, and mean level.
6. Permutation test on Spearman ρ (1000 permutations).

**Decision threshold:** Distribution metrics show ρ < 0.80 vs. mean, and Cohen's
d > 0.8 when comparing consistent vs. inconsistent profiles.

**Real-data swap:** Load multiple agents with `disteval.adapters.harbor_jobs` and
show that their κ distribution matches the synthetic profile categories.

---

## Experiment 2 — RECOVERABLE training vs random/top-K/all-task training

**Hypothesis:** Training on RECOVERABLE tasks (where Q* > 0 but Q* > Q̄) yields
higher score gain per example than random sampling, top-K hardest, all-task, or
SOLID-only training.

**Baselines:** random sampling, top-K lowest score, all-task, SOLID-only.

**Independent variables:** curriculum strategy (5 levels), training round (1–5),
selection size k, learning rate α, task difficulty.

**Dependent variables:** score gain per example, rounds to reach 0.80 mean
score, Δκ per task, RECOVERABLE → SOLID graduation rate, STUCK emergence rate.

**Protocol:**
1. Load real Harbor stores or generate synthetic data with known gap structure.
2. Run `right_tail_analysis` to classify tasks.
3. For each bootstrap iteration (n=5000), resample within tasks.
4. For each strategy, select k trajectories and apply `training_sim.apply_training_effect`.
5. Compute gain and κ improvement.
6. Run data-efficiency simulation to count rounds to threshold.
7. Pairwise bootstrap p-values and Cohen's d vs. each baseline.

**Decision threshold:** disteval RECOVERABLE strategy beats ≥3/4 baselines at
p < 0.05 and Cohen's d > 0.3, and achieves ≥20% fewer rounds than top-K.

**Real-data swap:** Replace `training_sim` with real DPO using
`disteval.training_harness.TRLReferenceTrainer` and re-evaluate.

---

## Experiment 3 — SelfEngine curriculum vs oracle/human/random curricula

**Hypothesis:** `SelfEngine.run_cycle()` ranks RECOVERABLE tasks within
Kendall's τ = 0.80 of an oracle that knows the true gap structure and learning
rates, and produces κ improvement within 0.05 of the oracle.

**Baselines:** oracle (true gap × learning rate), random curriculum, human-
curated (gap-only), top-K-by-gap.

**Independent variables:** task gap structure (homogeneous, heterogeneous,
adversarial), number of RECOVERABLE tasks (3, 6, 12), learning curve shape
(linear, diminishing, threshold), score noise level (0.0, 0.05, 0.10).

**Dependent variables:** Kendall τ and Spearman ρ vs. oracle, final κ
improvement, rounds to κ = 0.80, pair quality (reinforce > 0.75, contrast <
0.25), curriculum diversity.

**Protocol:**
1. Define an oracle curriculum with true gaps, consistencies, and learning
rates; compute oracle priority = Δ × α × (1 − κ).
2. Generate synthetic agent data consistent with the oracle.
3. Run `SelfEngine` on the synthetic data to obtain its ranking.
4. Generate baseline curricula (random, gap-only, human).
5. Compute ranking agreement (Kendall τ, Spearman ρ, rank inversions).
6. Simulate training for each curriculum and compare Δκ and rounds to threshold.
7. Sensitivity analysis on noise level and learning curve shape.

**Decision threshold:** SelfEngine τ > 0.80 vs. oracle; |Δκ_SelfEngine −
Δκ_oracle| < 0.05; outperforms random and human baselines at p < 0.05.

**Real-data swap:** Run `SelfEngine` on real Harbor jobs and compare its
ranking to a human expert ranking.

---

## Experiment 4 — Trajectory monitor predictive accuracy

**Hypothesis:** The structural tool-call signature predicts final outcome with
≥70% LOO accuracy and AUC-ROC ≥ 0.75, outperforming majority-class, first-step,
random, and constant-heuristic baselines.

**Baselines:** majority-class predictor, first-step-only predictor, random
coin flip, constant heuristic (n_exec ≥ 2).

**Independent variables:** trajectory prefix length (1, 2, 5, 10, 20, full),
data source (real vs. synthetic), score threshold for "high" (0.5, 0.7, 0.9),
feature set (full vs. ablated).

**Dependent variables:** LOO accuracy, AUC-ROC, precision/recall for low
outcomes, early-stop regret, expected calibration error (ECE), feature
importance.

**Protocol:**
1. Load real trajectories from Harbor jobs and generate synthetic trajectories.
2. Featurize at multiple prefix lengths using `TrajectoryFeaturizer`.
3. Train `OutcomePredictor` via leave-one-out cross-validation.
4. Compute baselines at the same prefix lengths.
5. Compute early-stop regret by simulating a stop when p_high < 0.35.
6. Ablate each feature and measure accuracy drop.
7. Paired t-tests on per-sample accuracy differences.

**Decision threshold:** Monitor accuracy > baselines at p < 0.05, AUC-ROC ≥ 0.75,
ECE < 0.10.

**Real-data swap:** Point `REAL_JOB_DIRS` to the user's Harbor directories; the
loader already handles the format.

---

## Experiment 5 — Trajectory memory retrieval

**Hypothesis:** Retrieving structurally similar high-outcome trajectories before
a task boosts success rate by ≥15% over no-memory, random-memory, and
chronological-memory baselines, with the largest gain on RECOVERABLE tasks.

**Baselines:** no memory, random memory, chronological (most recent) memory.

**Independent variables:** memory condition (4 levels), task kind
(SOLID/RECOVERABLE/STUCK), memory pool size (10, 50, 100, 500), retrieval k
(1, 3, 5), similarity threshold (0.3, 0.5, 0.7).

**Dependent variables:** task success rate, average score, early success rate,
memory utilization (tool-sequence overlap), boost magnitude.

**Protocol:**
1. Load all past trajectories into `TrajectoryMemory`.
2. Define a success-probability model for each task from historical data.
3. For each memory condition, retrieve k memories:
   - none: empty list.
   - random: sample k entries uniformly.
   - chronological: k most recent same-task entries.
   - structural: `TrajectoryMemory.retrieve_for_new_task(task, k=k)`.
4. Apply a calibrated boost: P_success = P_base + α × avg_similarity ×
avg_outcome_quality, with α = 0.15 as a starting point.
5. Monte Carlo simulate N=1000 trials per task and condition.
6. Aggregate success rates with 95% bootstrap CIs; paired t-tests against no
memory.
7. Sensitivity analysis on pool size, k, and threshold.

**Decision threshold:** Structural memory success rate ≥ no memory + 15% at p <
0.05; boost is largest for RECOVERABLE tasks.

**Real-data swap:** Run the agent with and without the memory prompt on the same
tasks and compare empirical success rates.

---

## Experiment 6 — Recursion engine

**Hypothesis:** Decomposing STUCK and high-gap RECOVERABLE tasks into sub-tasks
with entry/exit conditions improves sub-task solve rate and parent-task
graduation compared to flat retry and random decomposition.

**Baselines:** flat retry, random segmentation, no decomposition,
checkpoint-aligned decomposition without monitor divergence.

**Independent variables:** decomposition strategy (4 levels), task kind
(stuck/recoverable), recursion depth (1, 2, 3), sub-task count (2–5), checkpoint
alignment (yes/no).

**Dependent variables:** sub-task solve rate, parent task graduation rate, gap
reduction per example, recursive gap, solve latency (cycles to q_star > 0.5),
sub-task success variance, checkpoint alignment score.

**Protocol:**
1. Create 10 synthetic tasks with known sub-task structure and ground-truth
boundaries (write-then-verify, multi-checkpoint, search-then-execute templates).
2. Generate high/low/zero outcome trajectories for each task.
3. Run `RecursionEngine.decompose()` with `TrajectoryMonitor`.
4. Measure alignment of inferred boundaries to ground truth (within ±2 steps).
5. Simulate training on sub-tasks and aggregate improvements back to the parent.
6. Compare to baselines on solve rate, graduation, and gap reduction.
7. Real-data validation: load Harbor runs and compare inferred boundaries to
`test_suite_parser` checkpoints.

**Decision threshold:** Recursion engine > flat retry at p < 0.05 with Cohen's d
> 0.5 on sub-task solve rate; > random decomposition on parent graduation at p
< 0.05 (Fisher's exact test).

**Real-data swap:** Use `load_harbor_job` and `TrajectoryMonitor.from_store` to
decompose real STUCK tasks; validate against `test_suite_parser` checkpoints.

---

## Experiment 7 — Distributed evaluation

**Hypothesis:** Aggregating multiple agents and multiple runs narrows the true
meta-distribution CI and reduces false positives when comparing agents, vs.
single-agent single-run evaluation.

**Baselines:** single-agent single-run, single-agent multi-run, multi-agent
single-run, multi-agent multi-run.

**Independent variables:** number of agents (1, 2, 3, 5), runs per agent (1, 2,
5, 10), aggregate statistic (mean, IQM, CVaR, pass^k), task suite (synthetic vs.
real), agent similarity (similar vs. diverse).

**Dependent variables:** meta CI width, underconfidence ratio, P(A > B on fresh
re-run), false positive rate, false negative rate, CI coverage.

**Protocol:**
1. Generate synthetic agents with known true distributions (e.g., A: μ=0.75,
B: μ=0.70, C: μ=0.75 identical to A for FPR testing).
2. Create M=10 independent runs per agent.
3. Compute single-run bootstrap CIs with `stratified_bootstrap_ci`.
4. Compute meta-distribution CIs with `repeat.meta_distribution` for:
   - single-agent multi-run,
   - multi-agent single-run,
   - multi-agent multi-run.
5. Compute underconfidence ratio = meta CI width / mean bootstrap CI width.
6. Estimate false positive rate by comparing A vs. C (true gap = 0) and false
negative rate by comparing A vs. B (true gap = 0.05).
7. Compute `is_gap_real` P(A > B) for single-run vs. multi-run.

**Decision threshold:** Mean underconfidence ratio > 1.5 at p < 0.05; multi-run
P(A > B) is closer to the true gap than single-run; lower FPR at p < 0.05.

**Real-data swap:** Load real Harbor jobs and simulate pseudo-runs by resampling
tasks with different seeds; or use multiple available real runs.

---

## Experiment 8 — Agent harness

**Hypothesis:** The `AgentHarness` reduces integration cost (≤50% LOC vs.
manual JSONL, ≤70% vs. ad-hoc adapter) and produces complete, correct records
and trajectories with zero first-run errors.

**Baselines:** manual JSONL construction, ad-hoc adapter script.

**Independent variables:** implementation method (3 levels), task set (6 tasks
easy/medium/hard), episodes per task (3), agent complexity (simple, medium,
complex).

**Dependent variables:** lines of code, time to first valid run, record
completeness (%), trajectory completeness (%), first-run error rate, iteration
count to correctness, field validation errors, trajectory reference correctness.

**Protocol:**
1. Prepare six tasks from `tasks/` and a deterministic verifier (e.g., file
exists).
2. Implement the same simple agent logic in three ways:
   - `baselines/manual_jsonl.py`: hand-built dicts and JSONL.
   - `baselines/adhoc_adapter.py`: custom adapter + `GenericAdapter`.
   - `treatment/harness_based.py`: `AgentHarness` with `Agent`/`ToolExecutor`/`Verifier`.
3. Run each method on 6 tasks × 3 agents × 3 episodes = 54 runs.
4. Load outputs with `GenericAdapter` and count validation errors.
5. Compare LOC, error rate, and iteration count with 95% CIs and χ² / t-tests.
6. Optionally repeat with a real LLM agent.

**Decision threshold:** Harness LOC ≤ 50% of manual and ≤ 70% of adapter; error
rate = 0%; iteration count ≤ 3; p < 0.05 on LOC and iteration differences.

**Real-data swap:** Replace the synthetic agent with a real LLM client (Claude,
OpenAI, etc.) and run through the same harness.

---

## Experiment 9 — Training simulation fidelity

**Hypothesis:** `training_sim.py` predicts the rank-order of training gains
correctly (Spearman ρ > 0.7) and is calibrated to within 15% normalized MAE, even
if absolute gains are approximate.

**Baselines:** no-op model (zero gain), mean-only model, random gain model,
oracle model (true learning curve).

**Independent variables:** ground-truth type (synthetic, real, hybrid), training
strategy (disteval_right_tail, mean_reward, random), training round (1–5),
model parameters (α, DPO_BONUS), bootstrap size (10, 100, 1000).

**Dependent variables:** Spearman ρ between predicted and actual gain
rankings, normalized MAE, correlation between predicted and actual Δκ, learning
curve monotonicity, prediction coefficient of variation, strategy ranking
stability.

**Protocol:**
1. Create synthetic agents with known true learning curves (e.g., disteval
+0.05/round, mean +0.03/round, random +0.01/round).
2. Generate predictions with `training_sim.apply_training_effect` for each
strategy and round.
3. Compare predicted gains to ground truth via Spearman ρ and normalized MAE.
4. Check monotonicity of predicted learning curves.
5. Correlate predicted Δκ with actual Δκ per task.
6. Run sensitivity analysis on α and DPO_BONUS.
7. On real data, compare predicted gain to measured gain before/after training.

**Decision threshold:** Spearman ρ > 0.7 at p < 0.05; normalized MAE < 0.15;
≥95% of predicted learning curves are monotonic.

**Real-data swap:** Measure actual training gain from a real DPO run and compare
to `training_sim` predictions.

---

## Repository structure for experiments

```
research/experiments/
  experiment_01_distribution_metrics/
    README.md   — short protocol and expected outputs
    run.py      — executable script
    results/    — generated CSVs and plots
  experiment_02_recoverable_training/
  experiment_03_self_engine_oracle/
  experiment_04_trajectory_monitor/
  experiment_05_trajectory_memory/
  experiment_06_recursion_engine/
  experiment_07_distributed_eval/
  experiment_08_agent_harness/
  experiment_09_training_sim_fidelity/
  run_all.py    — convenience wrapper to run all experiments
```

Each `run.py` must:

1. Set `np.random.default_rng(SEED)` with a fixed seed.
2. Use only the existing disteval package and standard scientific Python.
3. Write `results/results.csv` and `results/results.json`.
4. Print a concise summary including effect sizes and p-values.
5. Document in its `README.md` how to swap in real Harbor/Inspect data.

## Suggested execution order

Run experiments in this order because later ones consume insights from earlier
ones:

1. Distribution metrics (1) — establishes that the evaluation signal itself is
richer than mean-only.
2. Trajectory monitor (4) — shows we can predict the signal mid-episode.
3. Trajectory memory (5) — shows we can reuse high-outcome episodes.
4. RECOVERABLE training (2) — shows the right-tail signal is trainable.
5. SelfEngine oracle (3) — shows the curriculum is automated well.
6. Training simulation fidelity (9) — shows we can predict gains cheaply.
7. Recursion engine (6) — handles STUCK tasks that right-tail training cannot.
8. Distributed evaluation (7) — shows aggregation improves reliability.
9. Agent harness (8) — provides the uniform substrate for all of the above.

## Deliverables

- This methodology document (`research/experimental_validation_plan.md`).
- One `run.py` and `README.md` per experiment under `research/experiments/`.
- A `run_all.py` aggregator for batch execution.
- A scorecard (to be populated after runs) in
`research/experiments/scorecard.md`.

## Open questions for the next research pass

- Should we define a unified synthetic task suite that all experiments share,
rather than each experiment generating its own?
- What is the minimum real-agent budget (API calls / GPU hours) required to
validate the simulation findings credibly?
- Should we publish a public benchmark release based on these experiments?
