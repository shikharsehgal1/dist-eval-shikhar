# Phase 4B — End-to-End Validation Plan for the Recursive Self-Improvement Prototype

**Date:** 2026-06-24  
**Input:** `.devin/skills/research-recursive-self-improvement/SKILL.md`, `research/phase3_master_report.md`, `research/phase3{a,b,c}_*.md`, `disteval/self_engine.py`, `disteval/training_sim.py`, `disteval/__main__.py`, `disteval/right_tail.py`, `disteval/trajectory_monitor.py`, `disteval/trajectory_memory.py`, `demo.py`, `wow_demo.py`, and existing tests.  
**Output:** This design document.  
**Constraint:** No existing disteval code is modified. Only the design document is produced.

---

## 1. Complete end-to-end flow of the recursive self-improvement prototype

The prototype is a **seven-stage, multi-cycle pipeline** that turns distributed agent evaluations into a recursively shrinking curriculum. Each stage consumes a concrete artifact from the previous stage and produces a named artifact for the next one. The file paths below are the actual paths in the repository; commands that do not yet exist are marked **(proposed)** and are grounded in the existing `disteval` CLI shape (`disteval/__main__.py`, lines 52–64).

### Stage 0 — Baseline (flat) cycle (optional, for comparison)

Run the existing `SelfEngine` on the current Harbor job directories to establish a baseline plan.

```bash
# Existing CLI command, defined in disteval/__main__.py:170-256
python -m disteval engine \
  jobs/run_A/disteval-run-A \
  --agent "Claude Code" \
  --model "claude-sonnet-4-5" \
  --output research/phase4b/baseline_claude_plan.json \
  --cycle 0

python -m disteval engine \
  jobs/run_B/disteval-run-B \
  --agent "Gemini CLI" \
  --model "gemini-2.5-flash" \
  --output research/phase4b/baseline_gemini_plan.json \
  --cycle 0

python -m disteval engine \
  jobs/run_C/disteval-run-C \
  --agent "Codex CLI" \
  --model "openai/o4-mini" \
  --output research/phase4b/baseline_codex_plan.json \
  --cycle 0
```

**Produces:** three `SelfImprovementPlan` JSONs (`research/phase4b/baseline_*_plan.json`) using the flat `SelfEngine.run_cycle()` implementation (`disteval/self_engine.py`, lines 375–435).

### Stage 1 — EVAL: collect distributed agent trajectories

Run each agent on the Harbor task suite. This stage is external to disteval; the only contract is that the runner writes `trajectory.json` and score records into a job directory that `disteval.adapters.harbor_jobs.load_harbor_job()` can read.

```bash
# Placeholder for the external agent runner (Harbor or equivalent).
# The output directory must match the structure already expected by
# load_harbor_job in disteval/adapters/harbor_jobs.py.
harbor run-suite \
  --tasks tasks/ \
  --agent claude-code \
  --out jobs/run_A_cycle0/disteval-run-A

harbor run-suite \
  --tasks tasks/ \
  --agent gemini-cli \
  --out jobs/run_B_cycle0/disteval-run-B

harbor run-suite \
  --tasks tasks/ \
  --agent codex-cli \
  --out jobs/run_C_cycle0/disteval-run-C
```

**Produces:** per-agent job directories at `jobs/run_{A,B,C}_cycle0/disteval-run-{A,B,C}` containing `trajectory.json`, `reward.txt`, and task-level metadata.

### Stage 2 — TAXONOMY: classify every task as SOLID / RECOVERABLE / STUCK

Ingest the three job directories into a shared `DistributedEvalPool` (proposed in `research/phase3c_distributed_evals.md`, section 5). This runs `right_tail_analysis()` (`disteval/right_tail.py`, lines 207–263) once per agent.

```bash
# (proposed) new subcommand / module: disteval.distributed_eval
python -m disteval.distributed_eval ingest \
  --agent "Claude Code" --model "claude-sonnet-4-5" --job-dir jobs/run_A_cycle0/disteval-run-A \
  --agent "Gemini CLI"  --model "gemini-2.5-flash" --job-dir jobs/run_B_cycle0/disteval-run-B \
  --agent "Codex CLI"   --model "openai/o4-mini"  --job-dir jobs/run_C_cycle0/disteval-run-C \
  --output research/phase4b/distributed_pool_cycle0.json
```

**Produces:** `research/phase4b/distributed_pool_cycle0.json` containing per-agent `TaskOutcomeProfile`s, per-agent `SubTaskGraph`s, and consensus sub-task environments (`research/phase3c_distributed_evals.md`, section 5.2).

### Stage 3 — DECOMPOSITION: build the `SubTaskGraph` with entry/exit boundaries

The `RecursionEngine` (designed in `research/phase2_master_report.md` and consumed in `research/phase3_master_report.md`, section 4) decomposes STUCK and high-gap RECOVERABLE tasks using the `TrajectoryMonitor` divergence signal.

```bash
# (proposed)
python -m disteval.recursion_engine decompose \
  --pool research/phase4b/distributed_pool_cycle0.json \
  --max-depth 3 \
  --output research/phase4b/subtask_graph_cycle0.json
```

Key internals:
- `TrajectoryMonitor.check()` (`disteval/trajectory_monitor.py`, lines 488–524) returns `p_high` for a prefix, which is used as the soft boundary confidence.
- `SelfEngine._find_divergence_step()` (`disteval/self_engine.py`, lines 553–575) already finds the first step where a high-scoring and low-scoring trajectory diverge; the recursive engine generalizes this to sub-task entry/exit detection.

**Produces:** `research/phase4b/subtask_graph_cycle0.json` with `SubTaskDefinition` nodes (e.g., `medium-2::phase-2`, `medium-2::phase-3`) annotated with `entry_step`, `exit_step`, `phase_tag`, and `kind`.

### Stage 4 — ENV GENERATION: produce `GenEnv` JSONs and update the `EnvironmentRegistry`

The `EnvironmentGenerator` (proposed in `research/phase3a_environment_schema.md`, section 2 and `research/phase3_master_report.md`, section 6) maps each `SubTaskDefinition` to a `GenEnv` JSON. It also parses `tasks/{task}/tests/test.sh` to extract checkpoint weights (`research/phase3a_environment_schema.md`, section 3.1).

```bash
# (proposed)
python -m disteval.environment_generator generate \
  --subtask-graph research/phase4b/subtask_graph_cycle0.json \
  --tasks-dir tasks/ \
  --registry research/phase4b/environment_registry.jsonl \
  --output-dir research/phase4b/envs/cycle0/
```

Concrete outputs for `medium-2`:
- `research/phase4b/envs/cycle0/medium-2__phase-2.json` — Engineering groupby sub-task, reward `0.25`, derived from `tasks/medium-2/tests/test.sh` lines 37–44.
- `research/phase4b/envs/cycle0/medium-2__phase-3.json` — Sales groupby sub-task, reward `0.20`, derived from `tests/test.sh` lines 47–54.
- `research/phase4b/envs/cycle0/medium-2__phase-4.json` — HR groupby sub-task, reward `0.20`, derived from `tests/test.sh` lines 57–64.

The `EnvironmentRegistry` (proposed in `research/phase3b_self_improvement_loop.md`, section 6) is updated with statuses:
- `active` for RECOVERABLE sub-tasks,
- `pending_decomposition` for freshly STUCK sub-tasks,
- `retired` for SOLID sub-tasks,
- `recursively_stuck` or `depth_cap` for decomposition failures.

**Produces:** `research/phase4b/environment_registry.jsonl` and one `GenEnv` JSON per active sub-task.

### Stage 5 — TRAINING: turn the plan into DPO pairs and fine-tune

This stage is intentionally outside disteval. The prototype provides a JSON curriculum that a trainer consumes.

```bash
# (proposed) export a merged curriculum for the DPO trainer
python -m disteval.curriculum export \
  --pool research/phase4b/distributed_pool_cycle0.json \
  --registry research/phase4b/environment_registry.jsonl \
  --agent "Codex CLI" \
  --output research/phase4b/curriculum_codex_cycle0.json
```

The curriculum JSON contains:
- Self-pairs from `SelfEngine._build_training_pairs()` (`disteval/self_engine.py`, lines 505–551),
- Cross-agent pairs from `DistributedEvalPool.build_cross_agent_pairs()` (`research/phase3c_distributed_evals.md`, section 3.2),
- `reinforce_traj_path` / `contrast_traj_path` file paths,
- `structural_divergence_step` from `SelfEngine._find_divergence_step()` (`disteval/self_engine.py`, lines 553–575),
- per-sub-task `priority_score = gap × (1 - consistency)` (`disteval/self_engine.py`, lines 451 and `research/phase3b_self_improvement_loop.md`, section 4.4).

The external trainer then produces an updated model checkpoint, e.g., `checkpoints/codex_cycle1/`.

### Stage 6 — RE-EVAL: run the updated agent on the active environment distribution

The updated agent is evaluated on the **active** task set:
- Retired SOLID sub-tasks are skipped (or run only for regression auditing).
- Active RECOVERABLE sub-tasks are run as generated sub-task environments.
- Pending STUCK sub-tasks are run as the parent task to collect new trajectories for deeper decomposition.

```bash
# (proposed) external runner uses the active distribution produced by the registry
harbor run-suite \
  --tasks research/phase4b/envs/cycle0/ \
  --agent checkpoints/codex_cycle1/ \
  --out jobs/run_C_cycle1/disteval-run-C
```

**Produces:** `jobs/run_C_cycle1/disteval-run-C` with new trajectories and scores.

### Stage 7 — DIST UPDATE: compare cycles, propagate rewards, and update the registry

Load the new job directory back into disteval and run the loop again.

```bash
# (proposed) single-cycle update command that re-runs taxonomy + decomposition + registry merge
python -m disteval.recursive_loop step \
  --previous-registry research/phase4b/environment_registry.jsonl \
  --previous-pool research/phase4b/distributed_pool_cycle0.json \
  --job-dir jobs/run_C_cycle1/disteval-run-C \
  --agent "Codex CLI" \
  --output-dir research/phase4b/cycle1/
```

This compares:
- `kind_n` vs `kind_{n+1}` using the taxonomy transition table from `research/phase3b_self_improvement_loop.md`, section 2.
- Parent scores via the weighted-sum rule: `parent_score = Σ_i checkpoint_weight_i × I(checkpoint_i passed)` (`research/phase3a_environment_schema.md`, section 3.3).

**Produces:** `research/phase4b/cycle1/distributed_pool_cycle1.json`, `research/phase4b/cycle1/subtask_graph_cycle1.json`, and an appended `research/phase4b/environment_registry.jsonl`.

---

## 2. Multi-cycle loop orchestration

### 2.1 Option A: manual CLI steps (recommended for the first prototype)

For the first 2–3 cycles, run the commands from Section 1 manually. This lets a human inspect each artifact before the next stage. The manual path is the safest because the recursive engine has no prior production validation.

### 2.2 Option B: single driver script (recommended for validation runs)

A new top-level script `research/phase4b/run_recursive_loop.py` (or a future `disteval recursive-loop` CLI) can chain the stages. It is not an existing file; it is the proposed validation harness.

```bash
python research/phase4b/run_recursive_loop.py \
  --agents-config research/phase4b/agents.toml \
  --max-cycles 5 \
  --tasks-dir tasks/ \
  --output-dir research/phase4b/
```

The driver would internally call:

1. `disteval.distributed_eval ingest` for all agents.
2. `disteval.recursion_engine decompose` on the resulting pool.
3. `disteval.environment_generator generate` to build the registry and GenEnv JSONs.
4. `disteval.curriculum export` to build per-agent curricula.
5. *(external trainer)*.
6. *(external re-eval runner)*.
7. `disteval.recursive_loop step` to merge the new cycle into the registry.
8. Repeat until a termination criterion is met.

### 2.3 Termination criteria

Taken from `research/phase3b_self_improvement_loop.md`, section 4.4:

1. `κ >= 0.95` global consistency index, where `κ = sum_q_bar / sum_q_star` (`disteval/right_tail.py`, lines 278 and `disteval/self_engine.py`, lines 397–400).
2. Plateau: `|Δκ| < 0.005` for 3 consecutive cycles.
3. Stuck saturation: all registry entries are `recursively_stuck` or `retired`.
4. Safety cap: `MAX_CYCLES = 20`.

### 2.4 CLI surface compatibility

The recursive prototype must be **default-disabled** so that existing `disteval engine` users are unaffected. Two backward-compatible designs are possible:

- **New subcommand:** `python -m disteval recursive-loop ...` (parallel to `engine`, `sim`, `report`, `compare` in `disteval/__main__.py`, lines 53–60).
- **Flag on existing command:** `python -m disteval engine ... --recursive --max-depth 3`. The flag is ignored by default, and the existing `SelfEngine.run_cycle()` path (`disteval/self_engine.py`, lines 375–435) remains unchanged.

Either approach keeps the existing flat path intact.

---

## 3. Validation plan

### 3.1 Metrics to measure at every cycle

| Metric | How to compute | Where it lives | What it tells us |
|---|---|---|---|
| Global consistency index `κ` | `sum_q_bar / sum_q_star` | `disteval/right_tail.py:278`, `disteval/self_engine.py:397-400` | Overall consistency across all tasks. Should increase monotonically if the loop is working. |
| Total recoverable gap `Δ_total` | `Σ_t (q_star - q_bar)` | `disteval/right_tail.py:239` | How much score is still available from fixing inconsistency. Should shrink. |
| `n_recoverable` / `n_solid` / `n_stuck` | counts from `right_tail_analysis()` | `disteval/right_tail.py:236-238` | Track tasks moving from RECOVERABLE → SOLID and STUCK → RECOVERABLE. |
| Per-sub-task `priority_score` | `gap × (1 - consistency)` | `disteval/self_engine.py:451` | Shows which sub-tasks are driving training each cycle. |
| Predicted gain per cycle | `training_sim.apply_training_effect()` / `_fast_apply_improvement()` | `disteval/training_sim.py:199-295`, `351-396` | Expected score gain before expensive fine-tuning. |
| Rounds to threshold | `training_sim.simulate_rounds_to_threshold()` / `_fast_rounds_to_threshold()` | `disteval/training_sim.py:399-521`, `523-607` | How many cycles the simulator predicts are needed to reach `κ = 0.80` (or another threshold). |
| Boundary confidence | `TrajectoryMonitor.check().p_high` | `disteval/trajectory_monitor.py:488-524` | Confidence that the sub-task boundary is correct. Should stabilize (`σ < 0.10` across cycles). |
| Registry status distribution | counts of `active`, `retired`, `recursively_stuck`, `depth_cap` | `research/phase3b_self_improvement_loop.md`, section 6.1 | Shows whether the environment distribution is shrinking or expanding. |
| Cross-agent pair yield | `n_cross_agent_pairs` / total pairs | `research/phase3c_distributed_evals.md`, section 5.2 | Whether distributed evals are producing useful transfer signals. |

### 3.2 How to detect that the loop is working

The loop is working if **all** of the following hold for at least two consecutive cycles:

1. `κ` increases by at least `0.01` per cycle (or the slope of a linear fit over 3 cycles is positive and `p < 0.10` under a bootstrap test).
2. The number of RECOVERABLE tasks that transition to SOLID is greater than the number that regress from SOLID/RECOVERABLE to STUCK.
3. The total recoverable gap `Δ_total` decreases, even if `κ` is noisy.
4. The `EnvironmentRegistry` has a growing fraction of `retired` entries and a shrinking or stable fraction of `active` entries.
5. The simulator (`training_sim`) predicts a positive gain for the recursive strategy, and that predicted gain is larger than the flat `SelfEngine` gain for the same data (see Section 5).
6. Sub-task boundaries do not drift by more than ±2 tool-call indices between cycles (measured by `boundary_confidence` variance in the registry).

### 3.3 How to detect failure modes

| Failure mode | Detection signal | Diagnostic command / file |
|---|---|---|
| **Plateau** | `|Δκ| < 0.005` for 3 consecutive cycles AND `Δ_total` is unchanged. | Inspect `research/phase4b/environment_registry.jsonl`; run `python -m disteval sim` to see if predicted gain ≈ 0. |
| **Recursively stuck** | Registry shows a task with `status = "recursively_stuck"` or all children of a STUCK task are also STUCK. | Check `subtask_graph_cycleN.json` for leaf nodes with `kind = "stuck"` at `max_depth`. |
| **Boundary drift** | Variance of `entry_step` / `exit_step` across cycles > 3 tool calls. | Compute diff of `boundary_variants` in `distributed_pool_cycle*.json`. |
| **Regression** | A SOLID task in cycle `n` becomes RECOVERABLE or STUCK in cycle `n+1`. | Compare `kind_history` arrays in the registry. |
| **False cross-agent transfer** | Cross-agent pairs have low `structural_similarity` (< 0.50) and the consumer agent does not improve on those sub-tasks. | Inspect `cross_agent_pairs` in `distributed_pool_cycle*.json` and subsequent `κ` for those sub-tasks. |
| **Over-decomposition** | Registry grows faster than it retires (`n_active + n_pending` increases by > 20% per cycle for 2 cycles). | Count entries per cycle from `environment_registry.jsonl`. |
| **Reward leakage** | Parent score changes without any child `kind` change (indicates checkpoint weight parsing bug). | Compare `parent_score` in registry with weighted sum of child scores. |

### 3.4 Automated smoke tests

The existing tests can be run as a pre-flight check before any prototype cycle:

```bash
pytest tests/test_right_tail.py -q          # taxonomy logic
pytest tests/test_training_sim.py -q         # gain prediction & simulation
pytest tests/test_trajectory_monitor.py -q   # boundary / divergence detection
pytest tests/test_trajectory_memory.py -q    # memory retrieval & similarity
```

A new test file `tests/test_recursive_loop.py` (proposed for Phase 4 implementation) should add:
- Registry merge correctness.
- GenEnv JSON schema validation.
- Cross-agent pair structural-similarity filtering.
- Cycle-to-cycle `κ` monotonicity on synthetic data.

---

## 4. Data needed for a meaningful signal

### 4.1 Per-agent data

| Quantity | Minimum | Recommended | Rationale |
|---|---|---|---|
| Attempts per task | 3 | 5–10 | `right_tail.task_outcome_profile()` (`disteval/right_tail.py:163-204`) needs enough samples to distinguish SOLID from RECOVERABLE reliably. With 3 attempts, a single lucky 1.0 can misclassify a STUCK task as RECOVERABLE. With 5 attempts the false-positive rate drops sharply. |
| Tasks per agent | 6 (full suite) | 6 + generated sub-tasks | The existing suite has 6 tasks (`disteval/self_engine.py:471-478`). Sub-tasks expand this to ~15–30 active environments. |
| Agents | 1 | 3 | One agent can run the recursive loop, but the distributed/cross-agent benefits in `research/phase3c_distributed_evals.md` require at least 2 agents. Three agents match the existing `AGENT_CONFIGS` in `disteval/training_sim.py:59-75`. |
| Cycles | 3 | 5–8 | Need enough cycles to observe transitions (RECOVERABLE → SOLID) and to detect plateaus. |
| Runs per cycle (for statistical power) | 1 full eval per agent | 2 full evals per agent, aggregated | Reduces run-level shock, which `demo.py` (lines 36–68) shows can dominate the headline mean. |

### 4.2 Total data footprint estimate

For 3 agents, 6 tasks, 5 attempts per task, 5 cycles, and 2 evals per cycle:

- `3 × 6 × 5 × 5 × 2 = 900` episode records.
- Sub-tasks multiply this by ~3–5, giving ~3,000–5,000 sub-task episodes after decomposition.
- Trajectory JSON files: ~1,000 files across cycles.

This is large but not unmanageable; the existing `wow_output/` directory already contains real trajectory data from 37 episodes (`wow_demo.py`, line 712).

### 4.3 Synthetic fallback

If real agent runs are expensive, the prototype can first be validated with `disteval/training_sim.py` on the existing `jobs/run_*` data. The simulator provides a meaningful signal with the existing 37 trajectories and can be augmented with synthetic score arrays for sub-task decomposition (see Section 5).

---

## 5. Comparison plan: recursive vs flat `SelfEngine`

The central research question is whether the recursive loop (sub-task decomposition + environment generation + per-sub-task training) improves faster than the flat `SelfEngine` that trains on whole-task trajectories.

### 5.1 Simulation-based comparison (cheap, first)

Use the existing `training_sim.py` as the comparison engine. It already compares three strategies (`disteval_right_tail`, `mean_reward`, `random`) and exposes the fast numpy path `_fast_apply_improvement()` (`disteval/training_sim.py`, lines 351–396) and `_fast_rounds_to_threshold()` (`disteval/training_sim.py`, lines 523–607).

Proposed augmented simulation:

```bash
# (proposed) new simulation mode that adds a "recursive" strategy
python -m disteval.training_sim \
  --mode recursive_vs_flat \
  --agents-config disteval/training_sim.py:AGENT_CONFIGS \
  --subtask-graph research/phase4b/subtask_graph_cycle0.json \
  --registry research/phase4b/environment_registry.jsonl \
  --output research/phase4b/sim_recursive_vs_flat.json
```

The simulator would add a fourth strategy:
- **Flat:** current `select_disteval_right_tail()` on whole-task records (`disteval/training_sim.py`, lines 132–172).
- **Recursive:** apply `_fast_apply_improvement()` per sub-task with the DPO bonus when both reinforce and contrast sub-task trajectories are present, then propagate scores upward via the weighted-sum rule (`research/phase3a_environment_schema.md`, section 3.3).

Success criterion for the simulation: the recursive strategy reaches `κ = 0.80` in fewer rounds than the flat strategy, with a bootstrap p-value `p < 0.05` computed the same way as `p_value_vs_mean_reward` in `disteval/training_sim.py`, lines 830–831.

### 5.2 Real-eval A/B comparison (expensive, definitive)

1. **Baseline arm:** run the flat `SelfEngine` for 3 cycles. Train only on whole-task reinforce/contrast pairs from `SelfImprovementPlan.curriculum` (`disteval/self_engine.py`, lines 125–129). Save plans to `research/phase4b/flat_cycle{0,1,2,3}/`.
2. **Recursive arm:** run the full seven-stage loop for 3 cycles. Train on sub-task GenEnv curricula. Save plans to `research/phase4b/recursive_cycle{0,1,2,3}/`.
3. **Controlled variables:** same agent, same initial checkpoint, same number of training steps, same external trainer.
4. **Evaluation metric:** final `κ` and `Δ_total` on a held-out fresh eval after the last cycle.
5. **Statistical test:** bootstrap within tasks (same as `training_sim.bootstrap_resample_within_tasks()`, `disteval/training_sim.py`, lines 300–334) to get a 95% CI on the `κ` difference between arms.

### 5.3 Comparison checklist

- [ ] Flat arm achieves a `κ` improvement consistent with `training_sim` predictions (`predicted_total_gain` in `SelfImprovementPlan`, `disteval/self_engine.py`, lines 128 and 164).
- [ ] Recursive arm achieves strictly larger `κ` improvement than flat arm after the same number of cycles.
- [ ] Recursive arm retires at least one sub-task (i.e., produces a `retired` registry entry) that was RECOVERABLE in the flat arm.
- [ ] The simulator predicted the ranking correctly before either real training run was executed.
- [ ] No regression: the recursive arm does not lose whole-task score on tasks that were SOLID in the flat arm.

---

## 6. Risks and mitigation strategies

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Entry-state replay is non-deterministic** (Tier 1 replay in `research/phase3a_environment_schema.md`, section 2) | Medium | High | Fallback to Tier 2 file snapshots or Tier 3 synthetic state. Validate replay determinism by running the same prefix twice and diffing `fs_state`. |
| **Test script parsing is brittle** | Medium | High | `TestSuiteParser` (proposed in `research/phase3a_environment_schema.md`, section 3.1) should have unit tests for each of the six `tasks/*/tests/test.sh` files. If a script fails to parse, fall back to the monitor-based soft reward (`p_high >= 0.70`). |
| **Boundary detection overfits to small sample sizes** | Medium | Medium | Require at least 5 trajectories per task before decomposition. Use bootstrap confidence intervals on boundary positions. Do not decompose a sub-task whose boundary confidence variance exceeds 2 steps. |
| **Cross-agent transfer introduces negative transfer** | Medium | High | Filter cross-agent pairs by `TrajectoryMemory` structural similarity (≥ 0.50) and require `privacy_approved = True` (`research/phase3c_distributed_evals.md`, section 3.2). Monitor the consumer agent's sub-task score after using a cross-agent pair; drop the pair if the score drops. |
| **Recursion depth explodes** | Low | Medium | Hard cap `max_depth = 3` in `RecursionEngineConfig` (`research/phase2_master_report.md`, section 2.4). Stop decomposing when all children are STUCK (`recursively_stuck`). |
| **Registry grows stale** | Medium | Medium | Append-only registry with per-entry `cycle_last_updated`; only the most recent entry per `sub_task_id` is authoritative (`research/phase3b_self_improvement_loop.md`, section 6). |
| **External trainer ignores the curriculum** | Medium | High | The curriculum JSON follows `CURRICULUM_FORMAT.md` and `SelfImprovementPlan.to_dict()` (`disteval/self_engine.py`, lines 169–213), so existing trainers can consume it. Add a lightweight mock trainer in the prototype to verify the file is parseable. |
| **Backward compatibility breakage** | Low | High | Keep recursive features behind a flag or new subcommand (Section 2.4). Ensure `python -m disteval engine jobs/run_C/disteval-run-C` still works unchanged. |
| **Compute cost exceeds budget** | Medium | Medium | Run the simulator first (Section 5.1). Use synthetic data and the existing 37 trajectories before paying for 5 real cycles. |
| **Human inspection bottleneck** | Medium | Low | Emit a Markdown summary per cycle (`plan.summary()` is already human-readable, `disteval/self_engine.py`, lines 135–167) and a JSON registry that can be diffed programmatically. |

---

## 7. Go / no-go checklist for proceeding to full implementation

Before implementing `disteval/recursion_engine.py`, `disteval/environment_generator.py`, `disteval/environment_registry.py`, `disteval/distributed_eval.py`, and the recursive CLI, the prototype must satisfy the following conditions.

### 7.1 Must-have (all must be true to proceed)

- [ ] **End-to-end artifact chain is reproducible:** running the commands in Section 1 twice on the same input produces identical `distributed_pool*.json` and `subtask_graph*.json` (deterministic up to the random seed in the trainer).
- [ ] **Simulation shows recursive advantage:** `training_sim` (or its recursive variant) reports that the recursive strategy reaches `κ = 0.80` in fewer rounds than the flat `SelfEngine` strategy, with `p < 0.05`.
- [ ] **At least one real sub-task decomposition is validated:** a human or automated checker confirms that a generated `GenEnv` JSON for `medium-2::phase-2` correctly maps to the Engineering groupby assertion in `tasks/medium-2/tests/test.sh` lines 37–44 and that the reward weight is `0.25`.
- [ ] **Taxonomy transitions are observable:** at least one task transitions from RECOVERABLE to SOLID, or from STUCK to RECOVERABLE, within the first 3 prototype cycles.
- [ ] **No regression on existing CLI:** `python -m disteval engine jobs/run_C/disteval-run-C` and `python -m disteval sim` still pass and produce the same outputs as before the prototype (because recursive features are default-disabled).
- [ ] **All new files have test coverage:** proposed `tests/test_recursion_engine.py`, `tests/test_environment_generator.py`, `tests/test_environment_registry.py`, and `tests/test_distributed_eval.py` run and pass on synthetic data.

### 7.2 Should-have (strongly preferred)

- [ ] Cross-agent pairs improve the consumer agent on at least one sub-task where the consumer was STUCK and a donor agent was SOLID.
- [ ] The `EnvironmentRegistry` retires more sub-tasks than it adds after cycle 2 (the active distribution shrinks).
- [ ] The simulator's predicted `κ` trajectory matches the real-eval `κ` trajectory within ±0.05 across 3 cycles.
- [ ] Boundary positions are stable: no sub-task boundary shifts by more than 2 tool-call indices between consecutive cycles.

### 7.3 No-go triggers (stop and reassess if any occur)

- [ ] The recursive arm is worse than the flat arm after 3 cycles (real eval `κ` lower or `Δ_total` higher).
- [ ] A majority of decomposed sub-tasks are classified as `recursively_stuck` or `depth_cap` with no new learning signal.
- [ ] The prototype introduces a new runtime dependency that is not justified by the `pyproject.toml` dependency stack.
- [ ] The external trainer cannot consume the exported curriculum JSON without modification.
- [ ] The registry grows unbounded (> 100 active/pending entries per parent task) indicating over-decomposition.

---

## 8. Open questions for future work

These questions are inherited from `research/phase3_master_report.md`, section 7, and refined by the prototype plan above.

1. **Entry-state replay determinism:** Does Harbor expose enough hooks to deterministically replay `steps[0:entry_step]` against the same Docker container image? If not, which fallback tier (snapshot vs. synthetic) is most reliable for each task?
2. **Test script parser robustness:** How well does `TestSuiteParser` handle the less-structured `tests/test.sh` files for `easy-1`, `easy-2`, `hard-1`, and `hard-2` compared to `medium-2`?
3. **Minimal GenEnv fields:** Which of the many proposed JSON fields in `research/phase3a_environment_schema.md` are strictly necessary for the first end-to-end prototype, and which can be deferred?
4. **Cross-agent similarity threshold:** What cosine-similarity threshold on tool sequences (from `TrajectoryMemory._task_match()`, `disteval/trajectory_memory.py`, lines 164–181) produces useful cross-agent pairs without false positives?
5. **Registry merge semantics:** How exactly should `EnvironmentRegistry` merge entries when sub-task boundaries shift slightly between cycles? Is last-write-wins acceptable, or do we need a weighted average of boundaries?
6. **Recursive simulation API:** What concrete changes to `disteval/training_sim.py` are needed to add a `recursive` strategy alongside the existing `disteval_right_tail`, `mean_reward`, and `random` strategies (`disteval/training_sim.py`, lines 132–195)?
7. **Plateau vs. optimal:** How do we distinguish a true plateau (no more recoverable score) from a situation where the agent needs more exploration or a different decomposition depth?
8. **External trainer contract:** Should disteval ship a reference DPO trainer that reads the curriculum JSON, or should the contract remain "disteval produces the plan, an external trainer consumes it"?
9. **Privacy default:** Should `shareable` default to `False` (as in `research/phase3c_distributed_evals.md`, section 6) or to `True` for internal multi-agent experiments? What is the opt-in UX?
10. **Scalability:** How does the runtime of `RecursionEngine.decompose()` scale with the number of trajectories and sub-tasks? Is a naive `O(n²)` boundary search acceptable for the prototype, or do we need an early-exit approximation?

---

## 9. Summary of commands and file paths

| Purpose | File path / command |
|---|---|
| Research skill | `.devin/skills/research-recursive-self-improvement/SKILL.md` |
| Phase 3 master report | `research/phase3_master_report.md` |
| Flat engine | `disteval/self_engine.py` |
| Training simulator | `disteval/training_sim.py` |
| CLI dispatcher | `disteval/__main__.py` |
| Right-tail taxonomy | `disteval/right_tail.py` |
| Trajectory monitor / divergence | `disteval/trajectory_monitor.py` |
| Trajectory memory | `disteval/trajectory_memory.py` |
| Existing demo | `wow_demo.py` |
| Existing tests | `tests/test_training_sim.py`, `tests/test_right_tail.py`, `tests/test_trajectory_monitor.py`, `tests/test_trajectory_memory.py` |
| Prototype pool | `research/phase4b/distributed_pool_cycleN.json` |
| Prototype sub-task graph | `research/phase4b/subtask_graph_cycleN.json` |
| Prototype environment registry | `research/phase4b/environment_registry.jsonl` |
| Prototype generated environments | `research/phase4b/envs/cycleN/*.json` |
| Prototype per-agent curriculum | `research/phase4b/curriculum_*_cycleN.json` |
| Prototype validation driver | `research/phase4b/run_recursive_loop.py` (proposed) |
| Proposed modules | `disteval/recursion_engine.py`, `disteval/environment_generator.py`, `disteval/environment_registry.py`, `disteval/distributed_eval.py` |

This document is the design input for the Phase 4 implementation. It does not modify any existing code; it only defines the expected end-to-end behavior, validation criteria, and decision gates for the recursive self-improvement prototype.
