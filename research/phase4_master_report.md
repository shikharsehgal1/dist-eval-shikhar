# Phase 4 Master Report: Prototype Integration Plan

**Date:** 2026-06-24
**Input:** Phase 3 master report + two Phase 4 design documents
**Output:** Consolidated prototype implementation plan and validation strategy
**Status:** Research complete; ready for implementation approval

---

## 1. What Phase 4 designed

Phase 4 produced two design documents:

- `research/phase4a_implementation_plan.md` — detailed, milestone-based implementation plan with file-by-file API specifications and code skeletons.
- `research/phase4b_end_to_end_validation.md` — complete seven-stage end-to-end flow and validation plan for the recursive self-improvement prototype.

This master report consolidates the agreed plan and defines the go/no-go criteria for starting implementation.

---

## 2. Implementation roadmap (4 milestones)

### Milestone 1 — Test-suite checkpoint parser (no recursion, no containers)

**New file:** `disteval/test_suite_parser.py`

Parses `tasks/{task}/tests/test.sh` for every disteval task and extracts ordered `(checkpoint index, description, reward weight)` triples. For `medium-2` this yields weights `[0.10, 0.25, 0.25, 0.20, 0.20]`; for `easy-1` it yields `[0.34, 0.33, 0.33]`.

**Validation:** `pytest tests/test_test_suite_parser.py` confirms all six tasks parse and weights sum to 1.0.

### Milestone 2 — RecursionEngine and sub-task decomposition

**New file:** `disteval/recursion_engine.py`
**Extended files:** `disteval/self_engine.py`, `disteval/right_tail.py`, `disteval/trajectory_monitor.py`, `disteval/trajectory_memory.py`, `disteval/__main__.py`

Adds `RecursionEngine`, `PhaseBoundary`, `SubTaskDefinition`, `RMDPNode`, `SubTaskGraph`, and `RecursionEngineConfig`. Decomposes STUCK and high-gap RECOVERABLE tasks into 1-exit sub-task RMDPs with `entry_step`/`exit_step` boundaries derived from `TrajectoryMonitor` divergence signals.

Integration is opt-in via `--enable-recursion` on the existing `disteval engine` CLI. All new dataclass fields are optional; default behavior is unchanged.

**Validation:** `pytest tests/test_recursion_engine.py` confirms depth cap, monotone-complexity check, and that `medium-2` decomposes into five sub-tasks with correct checkpoint weights.

### Milestone 3 — Environment generation and self-improvement loop

**New files:** `disteval/environment_generator.py`, `disteval/environment_registry.py`
**Extended file:** `disteval/training_sim.py`

Maps each `SubTaskDefinition` to a `GenEnv` JSON and persists it across cycles in an `EnvironmentRegistry`. The registry decides which sub-tasks to keep (`recoverable`), drop (`solid`), decompose further (`stuck`, depth < max), or escalate (`recursively_stuck` / `depth_cap`).

`training_sim.py` is extended with a recursive improvement propagation rule so that simulated gains can be compared between the flat and recursive strategies.

**Validation:** `pytest tests/test_environment_generator.py` and `pytest tests/test_environment_registry.py`; manual 2–3 cycle loop on `medium-2` with held-out eval.

### Milestone 4 — Distributed eval pooling

**New file:** `disteval/distributed_eval.py`

Maintains a `DistributedEvalPool` that ingests per-agent `RightTailReport`s and `SubTaskGraph`s, builds a consensus sub-task graph with boundary variants, classifies environments as `stable` / `contrastive` / `exploration_target` / `cross_agent_gap`, and generates cross-agent `TrainingPair`s when the source agent has opted in.

**Validation:** `pytest tests/test_distributed_eval.py`; cross-agent pair yield and structural similarity metrics on the three-agent `medium-2` dataset.

---

## 3. Key API additions

| File | New / changed | Summary |
|---|---|---|
| `disteval/test_suite_parser.py` | new | `CheckpointSpec`, `parse_test_suite()`, `parse_all_tasks()` |
| `disteval/recursion_engine.py` | new | `RecursionEngine`, `SubTaskDefinition`, `SubTaskGraph`, `RecursionEngineConfig` |
| `disteval/environment_generator.py` | new | `GenEnv`, `EnvironmentGenerator.generate()` |
| `disteval/environment_registry.py` | new | `RegistryEntry`, `EnvironmentRegistry.update()` |
| `disteval/distributed_eval.py` | new | `DistributedEvalPool`, `CrossAgentTrainingPair` |
| `disteval/right_tail.py` | extend | `parent_task`, `sub_task_depth`, `sub_task_profiles`, `recursive_gap` on `TaskOutcomeProfile` |
| `disteval/trajectory_monitor.py` | extend | `find_phase_boundaries()` method; optional `entry_step`/`exit_step` fields |
| `disteval/trajectory_memory.py` | extend | `sub_task_slices`; `retrieve_for_sub_task()` |
| `disteval/self_engine.py` | extend | `sub_tasks` on `TaskImprovement`; recursion context on `SelfImprovementPlan`; `run_cycle_with_recursion()` |
| `disteval/training_sim.py` | extend | recursive improvement propagation; new "recursive" strategy in simulation |
| `disteval/__main__.py` | extend | `--enable-recursion`, `--registry-path`, `--max-depth` flags |

Full API signatures and skeleton code are in `research/phase4a_implementation_plan.md`.

---

## 4. End-to-end flow (one cycle)

```
Stage 1  EVAL         harbor run-suite → jobs/run_{A,B,C}_cycle0/
Stage 2  TAXONOMY     python -m disteval.distributed_eval ingest → distributed_pool.json
Stage 3  DECOMPOSITION python -m disteval.recursion_engine decompose → subtask_graph.json
Stage 4  ENV GENERATION python -m disteval.environment_generator generate → envs/cycle0/ + registry.jsonl
Stage 5  TRAINING      python -m disteval.curriculum export → curriculum.json → external DPO trainer
Stage 6  RE-EVAL       harbor run-suite on active env distribution → jobs/run_*_cycle1/
Stage 7  DIST UPDATE   python -m disteval.recursive_loop step → updated pool + registry
```

This loop is repeated until one of the convergence criteria is met:

1. Global consistency index `κ >= 0.95`.
2. Plateau: `|Δκ| < 0.005` for 3 consecutive cycles.
3. Stuck saturation: all registry entries are `recursively_stuck` or `retired`.
4. Safety cap: `MAX_CYCLES = 20`.

---

## 5. Validation plan

### 5.1 Metrics to measure each cycle

| Metric | Source | Target |
|---|---|---|
| Global consistency index `κ` | `right_tail_analysis()` | Increase cycle over cycle |
| Total recoverable gap `Δ_total` | `RightTailReport` | Decrease cycle over cycle |
| `n_solid` / `n_recoverable` / `n_stuck` | `SelfImprovementPlan` | More solid, fewer recoverable |
| Sub-task priority score | `gap × (1 - consistency)` | Stable or rising for active sub-tasks |
| Predicted recursive gain vs flat gain | `training_sim.py` | Recursive > flat for STUCK tasks |
| Boundary confidence | `TrajectoryMonitor.p_high` | Stable or increasing |
| Registry status counts | `EnvironmentRegistry` | Active → retired over time |
| Cross-agent pair yield | `DistributedEvalPool` | Non-zero when agents differ |

### 5.2 Comparison against flat baseline

1. **Simulation-first:** Add a "recursive" strategy to `training_sim.py` and compare against the existing flat right-tail strategy on the same score distributions. Expect recursive > flat for STUCK tasks and comparable for already-recoverable tasks.
2. **Real-eval A/B:** Run 3 cycles with recursion enabled and 3 cycles with the current flat `SelfEngine` on the same task suite. Compare final `κ`, `Δ_total`, and per-task improvement using bootstrap CIs.

### 5.3 Failure-mode detection

| Failure mode | Detection | Mitigation |
|---|---|---|
| Plateau | `|Δκ| < 0.005` for 3 cycles | Stop; human review; increase exploration |
| Recursively stuck | All children STUCK after `max_depth` | Escalate to capability expansion |
| Boundary drift | Entry/exit confidence drops > 0.10 | Recompute boundary; invalidate old pairs |
| Regression | Task transitions SOLID → RECOVERABLE | Re-activate; investigate distribution shift |
| False cross-agent transfer | Similarity below threshold | Filter pairs; require structural compatibility |
| Over-decomposition | Sub-task < 2 tool calls | Merge with adjacent segment; raise min segment length |
| Reward leakage | Parent score ≠ weighted sum of sub-task scores | Re-parse test script; validate checkpoint weights |

---

## 6. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Entry-state replay is non-deterministic | Generated sub-task environments may not reproduce parent state | Tiered state synthesis; prefer snapshots when available; validate by re-running parent prefix |
| `test.sh` parsing is brittle | Wrong checkpoint weights | Extensive tests on all six tasks; graceful fallback to structural proxy |
| Boundary overfitting | Sub-task windows trained on too few trajectories | Require ≥ 2 trajectories per boundary; use confidence-weighted consensus |
| Cross-agent negative transfer | One agent's solution misleads another | Structural similarity filter; opt-in sharing; small held-out eval |
| Recursion depth explosion | Infinite decomposition of STUCK tasks | Hard depth cap; monotone complexity check; budget limit |
| Registry staleness | Old sub-task entries persist when agent behavior changes | Re-run full decomposition on `reload()`; cache keyed by trajectory hash |
| External trainer integration | Curriculum JSON not consumed correctly | Backward-compatible schema; clear docs; example DPO loader |
| Compute cost | Sub-task runs multiply eval cost | Retire SOLID sub-tasks; skip retired tasks; batch sub-task generation |
| Backward compatibility | Existing `disteval engine` users affected | Recursion default-disabled; all new fields optional; existing CLI unchanged |

---

## 7. Go / no-go checklist

### Go signals (must have at least 4 of 6)

- [ ] Milestone 1 test-suite parser passes for all six tasks.
- [ ] Milestone 2 `RecursionEngine` decomposes `medium-2` into five checkpoint-aligned sub-tasks.
- [ ] `GenEnv` JSON for `medium-2::phase-2` validates against a hand-check of `test.sh` lines 37–44.
- [ ] `EnvironmentRegistry` correctly retires a SOLID sub-task and keeps a RECOVERABLE one active.
- [ ] `training_sim.py` recursive strategy shows higher simulated gain than flat strategy on a STUCK-task scenario.
- [ ] Cross-agent pair generation produces at least one plausible pair from the existing three-agent `medium-2` data.

### No-go signals (any one is a stop)

- Entry-state replay fails to reproduce the parent task score on a re-run.
- `test.sh` parsing yields incorrect weights for ≥ 2 tasks.
- Recursion decomposes a SOLID task (false positive decomposition).
- Cross-agent pairs consistently hurt the consumer agent's held-out score.
- The recursive loop makes `κ` decrease over two consecutive cycles without an obvious distribution shift.

---

## 8. Open questions for future work

1. How to implement deterministic entry-state replay against the Harbor Docker runner.
2. Whether to extend the `test.sh` format to emit per-checkpoint reward files (`reward_c{i}.txt`) as a cleaner alternative to parsing.
3. How to canonicalize cross-cycle sub-task identity when boundaries shift slightly.
4. What cosine-similarity threshold on tool sequences is best for cross-agent pairs.
5. How to integrate the generated environments with a real DPO trainer (TRL, Axolotl, OpenRLHF).
6. Whether to support true multi-exit environments or stick to chains of 1-exit sub-RMDPs.
7. How to handle tasks that are non-deterministic or have multiple valid solution paths.
8. How to scale the distributed pool to 100+ tasks and 10+ agents without performance degradation.
9. Whether to expose recursion as a new top-level CLI subcommand (`disteval recursive-loop`) rather than flags on `disteval engine`.
10. How to publish the recursive consistency metric alongside `κ` and `CVaR` in rliable-compatible tooling.

---

## 9. Recommendation

The research is complete and the design is ready for implementation. The recommended path is:

1. **Approve the plan** and confirm the four-milestone scope.
2. **Implement Milestone 1** (test-suite parser) immediately — it is a pure addition with no risk to existing code.
3. **Implement Milestone 2** (RecursionEngine) behind the `--enable-recursion` flag.
4. **Run the simulation comparison** from Milestone 3 before any live training to validate that recursive gains exceed flat gains.
5. **Proceed to Milestones 3 and 4** only after the simulation comparison is positive.

All research deliverables are in `research/` and the task specification is in `.devin/skills/research-recursive-self-improvement/SKILL.md`.
