# Phase 3 Master Report: RL Environment Generation and Self-Improvement Loop

**Date:** 2026-06-23
**Input:** Phase 2 master report + three Phase 3 design documents
**Output:** Consolidated design for generated RL environments and the multi-cycle recursive self-improvement loop
**Next step:** Phase 4 — prototype integration plan for disteval

---

## 1. What Phase 3 designed

Phase 3 produced three design documents:

- `research/phase3a_environment_schema.md` — schema and generation of RL environments from `SubTaskDefinition`.
- `research/phase3b_self_improvement_loop.md` — multi-cycle self-improvement loop where cycle `n` affects cycle `n+1`.
- `research/phase3c_distributed_evals.md` — distributed agent evals feeding into a shared environment pool.

This master report consolidates the agreed design and identifies the decisions that Phase 4 must consume.

---

## 2. Generated RL environment schema (`GenEnv`)

A generated environment is the atomic unit produced by the environment generation layer. It is defined as a six-tuple:

```
GenEnv = (S, A, O, R, T, Z)
```

### 2.1 State space `S`

```json
{
  "fs_state": "file-system snapshot at sub-task entry",
  "tty_state": "terminal / process state (e.g., mock server running)",
  "context_prefix": "tool-call steps before entry_step",
  "context_outputs": "accumulated stdout/stderr from prefix",
  "sub_task_id": "e.g. medium-2::phase-2",
  "step_index": 0,
  "phase_tag": "write | exec | verify",
  "cycle": 1
}
```

### 2.2 Action space `A`

Same canonical tool vocabulary as `TrajectoryFeaturizer._TOOL_ALIASES` (`disteval/trajectory_monitor.py` lines 99–121): `write_file`, `run_shell_command`, `exec_command`, `read_file`, `list_directory`, `search_tool`, etc.

### 2.3 Observation space `O`

- `instruction` — derived sub-task instruction.
- `context_summary` — condensed prefix (≤300 tokens).
- `fs_listing` — files visible at the working directory.
- `prev_tool_output` — stdout/stderr from the most recent action.
- `phase_hint` — optional hint from `phase_tag`.
- `memory_prompt` — optional retrieval from `TrajectoryMemory`.

### 2.4 Reward function `R`

Sparse at checkpoints. For a single-checkpoint sub-task:

```
R = checkpoint_weight  if exit_condition satisfied
R = 0.0                  otherwise (default)
R = -checkpoint_weight optional failure penalty
```

Checkpoint weights are read directly from `tests/test.sh` (e.g., `medium-2` weights: 0.10, 0.25, 0.25, 0.20, 0.20).

### 2.5 Transition function `T`

Implicit in the Harbor / Docker container runtime. The environment re-runs tool calls against the container image and observes the resulting file-system and process state.

### 2.6 Termination conditions `Z`

- `success` — exit condition satisfied (test checkpoint passes or monitor `p_high >= 0.70`).
- `failure` — hard failure observed.
- `truncation` — step budget exhausted.
- `timeout` — wall-clock time exceeds `task.toml` timeout.
- `stack_limit` — sub-task depth >= `max_depth`.

### 2.7 Format choice

**Chosen format:** disteval-specific `GenEnv` JSON. Rejected Gymnasium (new dependency, LLM actions are not numeric vectors) and pure Harbor tasks (no step/reset API, rewards only at end). The `GenEnv` JSON has zero new runtime dependencies, extends `CURRICULUM_FORMAT.md` naturally, and can be materialized into Harbor task files by a thin `EnvironmentGenerator.materialise()` method in Phase 4.

---

## 3. Environment generation from `SubTaskDefinition`

A new module `disteval/environment_generator.py` (Phase 4) will map a `SubTaskDefinition` to a `GenEnv` via these steps:

1. **Instruction derivation** — combine `phase_tag` + parent `instruction.md` + checkpoint description.
2. **Context prefix extraction** — take `steps[0:entry_step]` from the high-scoring reinforce trajectory.
3. **Entry state synthesis** — replay the context prefix against the parent Docker container image to produce file-system and process state at `entry_step`. Tier 1 = replay; Tier 2 = file snapshot; Tier 3 = synthetic state.
4. **Reward wiring** — parse `tests/test.sh` to extract the checkpoint weight for the sub-task exit condition.
5. **Sub-task test snippet generation** — write a minimal test snippet that emits the checkpoint reward.
6. **Final JSON assembly** — produce the `GenEnv` JSON.

A concrete example for `medium-2::phase-2` (Engineering groupby, reward 0.25) is in `research/phase3a_environment_schema.md` section 4.

---

## 4. Multi-cycle self-improvement loop

### 4.1 Seven stages per cycle

```
CYCLE n
─────────────────────────────────────────────────────
Stage 1  EVAL         agent runs benchmark → job_dir_n
Stage 2  TAXONOMY     right_tail_analysis → SOLID/RECOVERABLE/STUCK
Stage 3  DECOMPOSITION RecursionEngine → SubTaskGraph
Stage 4  ENV GENERATION EnvironmentRegistry updated
Stage 5  TRAINING      DPO pairs from plan_n → fine-tuning
Stage 6  RE-EVAL       updated agent → job_dir_{n+1}
Stage 7  DIST UPDATE   compare plans; update registry
```

### 4.2 Environment keep / drop / modify rules

| Sub-task status | Action |
|---|---|
| SOLID | Retire from active distribution; archive pair for cross-agent use. |
| RECOVERABLE | Keep active; update DPO pair and boundary if confidence improved. |
| STUCK, depth < max, not decomposed | Decompose further; add children to registry. |
| STUCK, all children STUCK | Mark `recursively_stuck`; escalate to capability expansion. |
| STUCK, depth == max | Mark `depth_cap`; log for human review. |

### 4.3 How phase-0 affects phase-1 (medium-2 example)

```
medium-2 (RECOVERABLE, Δ=0.667, κ=0.333)
├── M_2a  phase-0: HTTP client runs → reward 0.10
├── M_2b  phase-1: filter → reward 0.25
├── M_2c  phase-2: Engineering groupby → reward 0.25
├── M_2d  phase-3: Sales groupby → reward 0.20
└── M_2e  phase-4: HR groupby → reward 0.20
```

If M_2a becomes SOLID in cycle `n`, it is retired from the active curriculum. M_2b's entry state is now stable (the HTTP client always runs), so M_2b can focus training purely on the filtering logic. The weighted-sum reward propagation means improving M_2b adds 0.25 to the parent score without requiring M_2a to be retrained.

### 4.4 Convergence / termination criteria

1. Global consistency index `κ >= 0.95`.
2. Plateau: `|Δκ| < 0.005` for 3 consecutive cycles.
3. Stuck saturation: all registry entries are `recursively_stuck` or `retired`.
4. Safety cap: `MAX_CYCLES = 20`.

---

## 5. Distributed agent evals

### 5.1 Shared pool

A new module `disteval/distributed_eval.py` (Phase 4) maintains a `DistributedEvalPool` that ingests per-agent `RightTailReport`s and `SubTaskGraph`s. It builds a **consensus sub-task graph with boundary variants**:

- Consensus sub-task identities are semantic (aligned to test checkpoints, e.g., `medium-2::phase-2`).
- Each consensus node stores per-agent boundary variants `(entry_step, exit_step, confidence)`.
- A preferred boundary is computed by confidence-weighted voting.
- If agent spread exceeds a threshold, multiple environment variants are kept.

### 5.2 Environment classification from distributed signals

| Status | Criterion |
|---|---|
| Stable | ≥ 1 agent SOLID, no agent STUCK. |
| Contrastive | ≥ 1 agent RECOVERABLE and ≥ 1 agent SOLID (possibly same). |
| Exploration target | All agents STUCK. |
| Cross-agent gap | Agent A SOLID and Agent B STUCK/RECOVERABLE. |

### 5.3 Cross-agent training pairs

When a consensus sub-task shows a cross-agent gap, the pool generates a `CrossAgentTrainingPair`:

- `reinforce_traj_path` from the SOLID agent's sliced trajectory.
- `contrast_traj_path` from the consumer agent's sliced trajectory.
- Matched by `TrajectoryMemory` structural similarity on tool sequences.
- Only used if the SOLID agent opted in to sharing (`shareable=True`).

This is the key mechanism for using distributed evals to close capability gaps without human-written solutions.

### 5.4 Privacy / attribution

- Per-agent `shareable` opt-in flag.
- `share_level` modes: `full`, `embedding`, `none`.
- Provenance logging (`cross_agent_attribution`) records which agent contributed which trajectory.
- Default is `shareable=False`; cross-agent transfer is explicit.

---

## 6. New modules required in Phase 4

Based on the Phase 3 design, the following new modules are needed:

- `disteval/environment_generator.py` — generate `GenEnv` JSON from `SubTaskDefinition`.
- `disteval/environment_registry.py` — persist and update environment distribution across cycles.
- `disteval/test_suite_parser.py` — parse `test.sh` checkpoints and weights.
- `disteval/distributed_eval.py` — shared pool for multi-agent evals and cross-agent pairs.

---

## 7. Open questions for Phase 4

1. **Prototype scope:** Which files should be implemented first? Recommended: `recursion_engine.py`, `environment_generator.py`, `test_suite_parser.py`, then `environment_registry.py`, then `distributed_eval.py`.
2. **Entry-state replay:** How to implement Tier 1 context prefix replay deterministically against a Docker container? Does Harbor expose enough hooks?
3. **Test script parsing:** How robust is `TestSuiteParser` across all six disteval tasks? Some tasks may have less structured `test.sh` files than `medium-2`.
4. **Curriculum JSON extensions:** Which of the many proposed new fields are strictly necessary for a minimal end-to-end prototype?
5. **Cross-agent similarity threshold:** What cosine-similarity threshold on tool sequences produces useful cross-agent pairs without false positives?
6. **Cycle-to-cycle registry merge:** How exactly does `EnvironmentRegistry` merge entries when sub-task boundaries shift slightly between cycles?
7. **Recursive simulation API:** What changes to `training_sim.py` are needed to validate that recursive gains exceed flat gains?
8. **Backward compatibility:** How to ensure all new recursion features are default-disabled and do not break existing `disteval engine` CLI users?

These questions are the input for Phase 4: the prototype integration plan.
