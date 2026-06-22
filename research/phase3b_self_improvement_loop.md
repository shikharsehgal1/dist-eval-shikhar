# Phase 3B — Self-Improvement Loop Design

**Date:** 2026-06-23
**Input:** Phase 2 Master Report (`research/phase2_master_report.md`), Phase 2A/2B/2C designs, `disteval/self_engine.py`, `disteval/training_sim.py`, `disteval/right_tail.py`, `disteval/trajectory_monitor.py`, `disteval/trajectory_memory.py`, `CURRICULUM_FORMAT.md`
**Output:** Design document for the multi-cycle recursive self-improvement loop where cycle `n`'s solution affects the task/environment distribution for cycle `n+1`.
**Constraint:** No existing code is modified. This is a research-only design document.

---

## 1. Overview

The current `SelfEngine.run_cycle()` (`disteval/self_engine.py`, lines 375–435) executes one flat improvement cycle: observe → localize → retrieve → schedule → simulate → output. Each call is independent; the plan from cycle `n` does not alter what tasks are run or how they are scored in cycle `n+1`.

This document designs the **recursive self-improvement loop** that closes that gap. The loop has two levels:

1. **Within-task recursion** (designed in Phase 2): a single parent task is decomposed into sub-task RMDPs at recursion depth 1–3. The sub-task scores propagate upward to the parent.
2. **Cross-cycle recursion** (this document): the environment distribution itself changes each cycle. Tasks that become SOLID are removed from the active curriculum; tasks that remain STUCK are decomposed further; new sub-tasks spawn from partial-credit signals; the training pairs generated in cycle `n` may be reused, invalidated, or superseded in cycle `n+1`.

The design is grounded in the existing primitives:

- `right_tail_analysis()` / `TaskOutcomeProfile` (`disteval/right_tail.py`, lines 163–263) for taxonomy.
- `RecursionEngine` (Phase 2 design, `disteval/recursion_engine.py`) for sub-task decomposition.
- `SelfImprovementPlan.to_dict()` (`disteval/self_engine.py`, lines 169–213) for JSON persistence.
- `training_sim._fast_apply_improvement()` (`disteval/training_sim.py`, lines 351–396) for gain prediction.
- `CURRICULUM_FORMAT.md` for the JSON output schema.

---

## 2. Step-by-step flow of a multi-cycle recursive self-improvement loop

The loop has seven stages per cycle. Each stage produces a named artifact that the next stage consumes.

```
CYCLE n
─────────────────────────────────────────────────────────────────────
Stage 1  EVAL                agent runs benchmark; scores written to job_dir_n
Stage 2  TAXONOMY            right_tail_analysis → SOLID / RECOVERABLE / STUCK
Stage 3  DECOMPOSITION       RecursionEngine → SubTaskGraph (sub-task RMDPs)
Stage 4  ENV GENERATION      EnvironmentRegistry updated; stale envs retired
Stage 5  TRAINING            DPO pairs from plan_n fed into fine-tuning
Stage 6  RE-EVAL             updated agent runs benchmark → job_dir_{n+1}
Stage 7  DIST UPDATE         compare plan_n with plan_{n+1}; update env registry

↓
CYCLE n+1  (same seven stages, but env registry carries forward)
```

### Stage 1 — EVAL

The agent (e.g., Codex CLI) is run on the Harbor task suite, writing `trajectory.json` and score records into a new job directory `jobs/run_{n}/`. This stage is external to disteval but its output feeds `load_harbor_job()` (`disteval/adapters/harbor_jobs.py`).

**Cycle-0 special case:** The first cycle has no prior environment registry. The full benchmark suite (`tasks/easy-1/`, `tasks/medium-2/`, etc.) is the initial environment distribution. `SelfEngine.from_job_dirs()` initializes `EnvironmentRegistry` with all tasks active.

### Stage 2 — TAXONOMY

```python
report_n = right_tail_analysis(store_n, model_name=None)
```

`right_tail_analysis()` (`disteval/right_tail.py`, lines 207–263) produces per-task `TaskOutcomeProfile` with `kind ∈ {solid, recoverable, stuck}`, `q_star`, `q_bar`, `gap`, and `consistency`.

The taxonomy is compared against the **previous cycle's taxonomy** (loaded from `plan_{n-1}.json`). Three transitions are possible:

| Previous kind | Current kind | Transition |
|---|---|---|
| recoverable | solid | Graduated: task removed from active curriculum |
| recoverable | recoverable | Persisted: training pair updated |
| stuck | recoverable | Unblocked: new training pairs become available |
| recoverable | stuck | Regressed: training pairs from cycle `n-1` invalidated |
| stuck | stuck | Stalled: sub-task decomposition deepened one level |
| solid | any | Should not occur unless tasks are non-deterministic |

### Stage 3 — DECOMPOSITION

```python
sub_task_graph_n = recursion_engine.decompose(report_n, traj_records_n)
```

`RecursionEngine.decompose()` (Phase 2 design, `disteval/recursion_engine.py`) uses `TrajectoryMonitor.find_phase_boundaries()` to segment STUCK and high-gap RECOVERABLE tasks into sub-task RMDPs. The graph carries `SubTaskDefinition` objects with `entry_step`, `exit_step`, `phase_tag`, `kind`, and `estimated_q_star`.

**Cross-cycle carryover:** If a sub-task with the same `sub_task_id` existed in `sub_task_graph_{n-1}`, its historical scores are merged into the current profile before re-running `task_outcome_profile()`. This accumulates data across cycles rather than treating each cycle in isolation (see Section 6 for the data structure and Section 6 for merge rules).

### Stage 4 — ENVIRONMENT GENERATION

The `EnvironmentRegistry` (new module `disteval/environment_registry.py`, Section 5) is updated:

- **SOLID sub-tasks:** marked `status = "retired"` in the registry; removed from the next cycle's eval distribution.
- **RECOVERABLE sub-tasks:** `status = "active"`, training pairs re-generated from updated slices.
- **STUCK sub-tasks:** `status = "pending_decomposition"` if this is the first cycle they appear; `status = "recursively_stuck"` if all children are STUCK after `max_depth` attempts.
- **Newly unblocked sub-tasks** (previously STUCK, now RECOVERABLE): `status` transitions from `"pending_decomposition"` to `"active"`.

Environment parameters for each active sub-task are also updated from the new trajectories. Concretely, the `entry_step` and `exit_step` may shift if the monitor finds a new, more confident boundary in the latest eval data.

### Stage 5 — TRAINING

The `SelfImprovementPlan` (plan_n) is serialized to JSON via `SelfImprovementPlan.to_dict()` and consumed by a DPO trainer. The trainer reads `training_pairs` in priority order (`priority_score = gap × (1 - consistency)`), fine-tunes the model on reinforce/contrast pairs, and produces an updated checkpoint.

This step is **external to disteval**. From disteval's perspective, training is a black box: the engine produces a curriculum and the trained model runs again in Stage 6.

### Stage 6 — RE-EVAL

The fine-tuned agent runs on the same Harbor task suite (possibly restricted to the active environment distribution — see Section 4) and writes results into `jobs/run_{n+1}/`. If the environment distribution was updated in Stage 4 (e.g., SOLID tasks removed), this cycle evaluates the reduced task set plus any newly generated sub-task environments.

### Stage 7 — DISTRIBUTION UPDATE

After Stage 6, `SelfEngine` compares `plan_n` with `plan_{n+1}`:

1. For each task, compare `kind_{n}` with `kind_{n+1}` (taxonomy transition table, Stage 2).
2. For each sub-task, compare `kind_{n}` with `kind_{n+1}` and propagate changes to parent scores using the weighted-sum rule (`disteval/training_sim.py`, `_fast_apply_improvement()`, lines 351–396).
3. Update the `EnvironmentRegistry` with new `q_star`, `q_bar`, and `gap` values.
4. Log the cycle delta to `environment_registry.jsonl` (Section 5).

The updated `EnvironmentRegistry` is the input to Stage 2 of cycle `n+1`.

---

## 3. How the system decides which environments to keep, drop, or modify

The decision is based on sub-task taxonomy status after each cycle's right-tail analysis. The rules below are grounded in the existing `task_outcome_profile()` logic (`disteval/right_tail.py`, lines 163–204) and the `RecursionEngineConfig` (Phase 2 Master Report, Section 2.4).

### 3.1 SOLID sub-tasks → retire

A sub-task `t_i` is retired from the active environment distribution when:

```
kind(t_i) == "solid"
```

i.e., `q_star(t_i) > 0` and `gap(t_i) < 1e-9` (same threshold used in `task_outcome_profile()`, `right_tail.py` line 184).

**Action:** Set `EnvironmentRegistry[t_i].status = "retired"`. The sub-task is excluded from the next Harbor run. Its last good training pair is archived in `archived_pairs[t_i]` for potential retrieval as a reinforce demonstration for agents that are STUCK on this sub-task.

**Rationale:** Training on an already-SOLID sub-task wastes compute and risks regression by over-specifying behavior that is already consistent.

### 3.2 RECOVERABLE sub-tasks → keep and update

A sub-task remains in the active distribution when:

```
kind(t_i) == "recoverable"
```

i.e., `q_star(t_i) > 0` and `gap(t_i) >= 1e-9`.

**Action:** Keep `status = "active"`. Update the DPO pair using the newest high/low trajectory pair. Update `entry_step` / `exit_step` if the monitor found a more confident boundary in the new data (higher `p_high` confidence at the boundary; see `TrajectoryMonitor.check()`, `disteval/trajectory_monitor.py`, lines 488–524). Recompute `priority_score = gap × (1 - consistency)`.

**Modification rule for entry/exit shift:** if the new boundary confidence exceeds the old boundary confidence by more than 0.10, replace the stored `entry_step` / `exit_step` pair. This ensures the sub-task window sharpens as more data accumulates.

### 3.3 STUCK sub-tasks → decompose further or escalate

A sub-task is STUCK when:

```
kind(t_i) == "stuck"  (q_star(t_i) == 0)
```

**Sub-case A: depth < max_depth and sub-task has not been decomposed yet.**
Action: attempt `RecursionEngine._decompose_task(profile, traj_records, depth+1)`. If the decomposition produces at least one RECOVERABLE child, add those children to the registry as `status = "active"`. Update parent to `status = "pending_decomposition"`.

**Sub-case B: all children are also STUCK (recursively stuck).**
Action: Set `status = "recursively_stuck"`. Do not add to DPO curriculum. Surface in plan summary: "Task `{t_i}` has no demonstrated solution at depth ≤ {max_depth}; capability expansion required." This mirrors the `recursively_stuck` flag proposed in Phase 2B (`research/phase2b_decomposition_algorithm.md`, lines 320–328).

**Sub-case C: depth == max_depth.**
Action: Set `status = "depth_cap"`. Same treatment as recursively stuck: excluded from DPO curriculum, logged for human review.

### 3.4 Newly unblocked tasks (STUCK → RECOVERABLE)

When a task that was STUCK in cycle `n-1` becomes RECOVERABLE in cycle `n` (the agent solved it at least once), the `EnvironmentRegistry` transitions it:

```
status: "pending_decomposition" → "active"
```

The previous cycle's decomposition tree (which may have been based on memory-retrieved demonstrations from other agents) is discarded and replaced with boundaries derived from the agent's own first successful trajectory.

---

## 4. How phase-0 of medium-2 affects phase-1: state transitions, reward propagation, and curriculum updates

This section uses `medium-2` (REST client, `tasks/medium-2/`) as a concrete example because:

- It has an explicit 5-checkpoint scoring structure (`tasks/medium-2/tests/test.sh`, lines 25–64).
- Phase 2 already decomposed it into sub-tasks `M_2a`–`M_2e` (Phase 2 Master Report, Section 6).
- The Codex CLI agent's observed score vector `[0.0, 0.0, 1.0]` (RECOVERABLE, `q_star=1.0`, `q_bar=0.333`) provides a realistic starting point.

### 4.1 Sub-task chain structure

```
medium-2 (depth 0, RECOVERABLE, Δ=0.667, κ=0.333)
├── M_2a  phase-0: HTTP client runs → reward_delta=0.10
├── M_2b  phase-1: filter (eligible_users=7) → reward_delta=0.25
├── M_2c  phase-2: Engineering groupby → reward_delta=0.25
├── M_2d  phase-3: Sales groupby → reward_delta=0.20
└── M_2e  phase-4: HR groupby → reward_delta=0.20
```

Each `reward_delta` matches the checkpoint weights in `tasks/medium-2/tests/test.sh` (lines 26, 34, 44, 54, 64), as documented in Phase 2B (`research/phase2b_decomposition_algorithm.md`, lines 269–278).

### 4.2 State transition: how M_2a (phase-0) affects M_2b (phase-1)

**Entry state of M_2b** is defined as the file-system + terminal state that exists at `exit_step` of M_2a. Concretely:

- `client.py` has been written and produces valid JSON output (M_2a exit condition).
- The mock server is running at `localhost:{PORT}` (from the environment setup in `tasks/medium-2/environment/`).
- No groupby logic is yet present in `client.py`.

The `entry_context` for M_2b is the tool-call prefix `steps[0:exit_step_M_2a]` stored in `TrainingPair.entry_context` (Phase 2B proposed field, `disteval/self_engine.py` extension at lines 69–79). The DPO trainer receives this context so M_2b training does not hallucinate a different client structure.

**State transition rule:**

```
If M_2a.kind == "solid" in cycle n:
    → M_2b's entry state is stable and trusted.
    → M_2b's entry_step is fixed; no boundary re-detection needed.
    → M_2b is the new training focus.
    → M_2a is retired from the active distribution.

If M_2a.kind == "recoverable" in cycle n:
    → M_2a's exit varies across attempts (sometimes client.py is valid, sometimes not).
    → M_2b's entry state is unstable; training M_2b in isolation risks context mismatch.
    → Train M_2a first (higher in priority queue due to gateway nature).
    → M_2b is deferred until M_2a reaches solid status.

If M_2a.kind == "stuck" in cycle n:
    → No successful M_2a trajectory exists.
    → M_2b cannot be reached from within the task; treat as downstream-blocked.
    → Decompose M_2a further OR mark the whole chain as pending.
```

This **dependency propagation** is the core cross-phase interaction: a sub-task's entry state validity depends on the upstream sub-task's exit reliability.

### 4.3 Reward propagation from M_2a → parent and M_2b

Using the weighted-sum rule from `disteval/training_sim.py` (lines 351–396):

```
Q̄(medium-2) = Σ_i w_i × Q̄(M_2i)
            = 0.10×Q̄(M_2a) + 0.25×Q̄(M_2b) + 0.25×Q̄(M_2c) + 0.20×Q̄(M_2d) + 0.20×Q̄(M_2e)
```

When M_2a transitions from RECOVERABLE (say `Q̄(M_2a)=0.667`) to SOLID (`Q̄(M_2a)=1.0`) after cycle `n`:

```
ΔQ̄(medium-2) = 0.10 × (1.0 - 0.667) = +0.033
```

This is a small but compounding effect: closing the M_2a gap lifts the parent score and simultaneously stabilizes M_2b's entry state, enabling higher-quality M_2b training pairs in cycle `n+1`.

**The DPO improvement formula** (from `training_sim.apply_training_effect()`, lines 199–295) applied to M_2b in cycle `n+1`:

```
improvement(M_2b) = α × DPO_BONUS × q_star(M_2b) × (1 - q_bar(M_2b))
                  = 0.4 × 1.5 × 1.0 × (1 - 0.667)
                  = 0.4 × 1.5 × 0.333
                  ≈ 0.20
```

where `α = 0.4` (`training_sim.py`, line 46) and `DPO_BONUS = 1.5` (line 52) apply because M_2b has a paired reinforce/contrast trajectory once M_2a's entry state is stable.

### 4.4 Curriculum update after M_2a reaches SOLID

Cycle `n` curriculum:

```
Priority 1: M_2a  (gateway sub-task, kind=recoverable, priority_score=0.10×0.333×0.667=0.022)
Priority 2: M_2c  (large gap, kind=recoverable, priority_score=0.25×0.667×0.667=0.111)
Priority 3: M_2d  (medium gap, kind=recoverable, priority_score=0.20×0.667×0.667=0.089)
Priority 4: M_2e  (medium gap, kind=recoverable, priority_score=0.20×0.667×0.667=0.089)
Priority 5: M_2b  (deferred pending M_2a stability)
```

After training on cycle `n` (M_2a becomes SOLID):

Cycle `n+1` curriculum:

```
M_2a  → RETIRED (removed)
M_2b  → now RECOVERABLE with clean entry state → priority_score = 0.25×0.333×0.667 = 0.056
M_2c  → still high gap → priority_score = 0.25×(updated gap)×(updated 1-κ)
M_2d  → similar
M_2e  → similar
```

The priority scores in `n+1` are recomputed from the updated `q_bar` values, not inherited from `n`. The `EnvironmentRegistry` stores the last known `q_star`, `q_bar`, and `gap` for each sub-task to support this recomputation.

---

## 5. How the environment distribution evolves over cycles

### 5.1 New sub-tasks added

New sub-tasks are added to the active distribution when any of the following occur:

1. **First decomposition of a STUCK task:** `RecursionEngine.decompose()` produces children with `sub_task_depth = 1` (or deeper). Each child is `INSERT`ed into the registry with `status = "active"` or `"pending_decomposition"` depending on its kind.

2. **Boundary refinement:** An existing RECOVERABLE sub-task's monitor confidence crosses a secondary threshold, suggesting a finer-grained boundary exists. The sub-task is split into two sub-sub-tasks (`sub_task_depth` incremented by 1). This is the depth-2 recursion path.

3. **Previously STUCK becomes RECOVERABLE:** The agent solves a STUCK task at least once. `RecursionEngine` re-decomposes it using the new successful trajectory as a basis for boundary detection, potentially finding more informative boundaries than the memory-based fallback used in the prior cycle.

4. **New partial-credit signal:** An agent that previously scored `0.0` on all attempts scores `0.1` (partial credit) on one attempt. This transitions the task from pure STUCK to partial-credit RECOVERABLE and enables divergence-point cleavage (Phase 2B, Section 3, criterion in lines 38–46).

### 5.2 Solved sub-tasks removed

A sub-task is removed (retired) when `kind == "solid"` for two consecutive cycles. The two-cycle requirement prevents over-eager retirement due to sampling variance (a sub-task can appear SOLID if the agent happens to get lucky on all `k` attempts in a small-k run).

Formally:
```
retire_condition(t_i, n) = (kind_n(t_i) == "solid") AND (kind_{n-1}(t_i) in {"solid", "recoverable"})
```

When retired, the sub-task's best training pair is archived rather than deleted, for two reasons:
- It can serve as a reinforce demonstration for a different agent that is STUCK on the same sub-task (cross-agent sharing, addressed in Phase 3C).
- If the parent task regresses (the agent degrades on a previously SOLID sub-task), the archived pair can be quickly reactivated.

### 5.3 Partially solved sub-tasks modified

When `kind` stays `"recoverable"` across two cycles but `gap` decreases, the sub-task is modified:

1. **Training pair replaced:** The new cycle's highest-scoring trajectory (with a score at least `0.05` above the previous reinforce score) replaces the old reinforce trajectory. This ensures the reinforce target is always the most recent best.

2. **Entry/exit boundary re-evaluated:** The monitor re-runs `find_phase_boundaries()` on the updated trajectory set. If a new boundary has confidence `>= 0.10` above the current boundary, the boundary is updated.

3. **Priority score recomputed:** `priority_score = gap_{n+1} × (1 - consistency_{n+1})`. As `gap` shrinks and `consistency` rises, the sub-task naturally falls in the curriculum ranking, making room for higher-gap sub-tasks.

4. **Predicted rounds to threshold updated:** `training_sim.simulate_rounds_to_threshold()` is called with the updated score arrays (`disteval/training_sim.py`, lines 399–521), replacing the stale estimate from cycle `n`.

---

## 6. Data structure persisting the cross-cycle environment state

The `EnvironmentRegistry` is a new JSON-Lines file (`environment_registry.jsonl`) maintained in the output directory alongside the curriculum JSON. Each line is one registry entry (one sub-task or top-level task). The file is append-only: each cycle appends a new entry; the most recent entry for a given `sub_task_id` is the authoritative state.

### 6.1 Per-entry schema

```json
{
  "sub_task_id": "medium-2::phase-1",
  "parent_task": "disteval/medium-rest-client",
  "sub_task_depth": 1,
  "phase_tag": "filter",
  "entry_step": 4,
  "exit_step": 9,
  "reward_delta": 0.25,
  "weight": 0.25,
  "status": "active",
  "cycle_introduced": 0,
  "cycle_last_updated": 2,
  "kind_history": [
    {"cycle": 0, "kind": "stuck"},
    {"cycle": 1, "kind": "recoverable"},
    {"cycle": 2, "kind": "recoverable"}
  ],
  "score_history": [
    {"cycle": 0, "scores": [], "q_star": 0.0, "q_bar": 0.0, "gap": 0.0},
    {"cycle": 1, "scores": [0.25, 0.0, 1.0], "q_star": 1.0, "q_bar": 0.417, "gap": 0.583},
    {"cycle": 2, "scores": [1.0, 0.25, 1.0], "q_star": 1.0, "q_bar": 0.750, "gap": 0.250}
  ],
  "training_pair": {
    "reinforce_traj_path": "jobs/run_2/.../medium-2__abc/agent/trajectory.json",
    "contrast_traj_path": "jobs/run_2/.../medium-2__def/agent/trajectory.json",
    "reinforce_score": 1.0,
    "contrast_score": 0.25,
    "gap": 0.75,
    "structural_divergence_step": 2,
    "entry_step": 4,
    "exit_step": 9,
    "cycle": 2
  },
  "archived_pairs": [
    {
      "reinforce_traj_path": "jobs/run_1/.../medium-2__xyz/agent/trajectory.json",
      "contrast_traj_path": "jobs/run_1/.../medium-2__uvw/agent/trajectory.json",
      "cycle": 1
    }
  ],
  "boundary_confidence": 0.84,
  "boundary_source": "structural_divergence",
  "recursion_context": {
    "decomposed_reason": "recoverable",
    "depth_cap_hit": false,
    "recursively_stuck": false
  },
  "call_stack": [
    ["disteval/medium-rest-client", "medium-2::phase-1"]
  ]
}
```

### 6.2 Field semantics

| Field | Type | Description |
|---|---|---|
| `sub_task_id` | string | Unique identifier; format `"{parent_task_name}::phase-{i}"`. Matches `SubTaskDefinition.sub_task_id` from Phase 2 Master Report, Section 2.2. |
| `parent_task` | string | Parent task identifier in disteval record store format (e.g., `"disteval/medium-rest-client"`). |
| `sub_task_depth` | int | 0 = top-level task, 1 = first decomposition, etc. Capped at `RecursionEngineConfig.max_depth = 3`. |
| `phase_tag` | string | Coarse semantic tag (e.g., `"setup"`, `"filter"`, `"groupby"`, `"verify"`). Derived from `TrajectoryMonitor` tool-category transitions. |
| `entry_step` | int | First tool-call index (0-based) within the parent trajectory for this sub-task. |
| `exit_step` | int | Last tool-call index (inclusive) for this sub-task. `-1` = end of trajectory. |
| `reward_delta` | float | The incremental parent-task score contributed by this sub-task. Matches checkpoint weight from `test.sh`. |
| `weight` | float | Same as `reward_delta` normalized so all sibling weights sum to 1.0. Used in the weighted-sum reward propagation formula. |
| `status` | string | `"active"` / `"retired"` / `"pending_decomposition"` / `"recursively_stuck"` / `"depth_cap"`. |
| `cycle_introduced` | int | Cycle number when this sub-task was first added to the registry. |
| `cycle_last_updated` | int | Last cycle that produced new scores for this sub-task. |
| `kind_history` | array | Per-cycle `kind` classification (SOLID/RECOVERABLE/STUCK) for audit and trend analysis. |
| `score_history` | array | Per-cycle raw scores, `q_star`, `q_bar`, and `gap`. Cumulative; not cleared between cycles. |
| `training_pair` | object | Current best (reinforce, contrast) DPO pair. Updated each cycle when a better pair is found. |
| `archived_pairs` | array | Previous training pairs (one per past cycle). Kept for cross-agent sharing and regression detection. |
| `boundary_confidence` | float | `p_high` confidence of the boundary from `TrajectoryMonitor.check()` (`trajectory_monitor.py`, line 81). |
| `boundary_source` | string | `"structural_divergence"` / `"checkpoint"` / `"memory"` — matches `SubTask.source` from Phase 2A. |
| `recursion_context` | object | Termination metadata: reason for decomposition, whether depth cap was hit, whether all children are STUCK. |
| `call_stack` | array | Flat stack of `[parent, child]` pairs representing the RMDP call hierarchy. |

### 6.3 Registry access patterns

The registry is read by `SelfEngine.run_cycle()` at the start of each cycle (Stage 2) and written at the end (Stage 7). The access patterns are:

- **Read by `sub_task_id`:** look up current `status`, `entry_step`, `exit_step`, `boundary_confidence`.
- **Read by `parent_task`:** retrieve all children of a parent to recompute the weighted-sum parent score.
- **Write append:** each cycle appends updated entries for changed sub-tasks only (not all entries every cycle).
- **Read for cross-agent sharing:** the `archived_pairs` list is queried when an agent is STUCK on a sub-task that another agent has SOLID archived pairs for (Phase 3C pattern).

The append-only design is important for reproducibility: the complete history of every sub-task's evolution is preserved and can be replayed to reconstruct any past cycle's curriculum.

---

## 7. How training pairs from cycle n are reused or invalidated in cycle n+1

### 7.1 Validity conditions for a training pair

A training pair from cycle `n` is valid in cycle `n+1` if and only if all three conditions hold:

1. **The sub-task is still active:** `status ∈ {"active", "pending_decomposition"}`. Retired sub-tasks' pairs are archived, not used for training.

2. **The entry/exit boundary has not shifted significantly:** If the new cycle's boundary confidence is more than 0.10 higher than the stored `boundary_confidence` and the new `entry_step` differs by more than 2 steps from the stored one, the old pair's context alignment is likely stale. The pair is invalidated and re-generated from the new boundary.

3. **The reinforce trajectory is still the best available:** The reinforce score from cycle `n` must be `>= 0.9 × q_star_{n+1}` (using the updated `q_star`). If a higher-scoring trajectory exists in cycle `n+1`, the reinforce trajectory is replaced. The contrast trajectory is always replaced with the lowest-scoring cycle-`n+1` attempt.

Formally:
```python
def pair_is_valid(pair_n, registry_entry_n1):
    if registry_entry_n1.status not in ("active", "pending_decomposition"):
        return False  # rule 1
    boundary_shift = abs(pair_n.entry_step - registry_entry_n1.entry_step) > 2
    confidence_gain = registry_entry_n1.boundary_confidence - pair_n.boundary_confidence > 0.10
    if boundary_shift and confidence_gain:
        return False  # rule 2
    q_star_n1 = registry_entry_n1.score_history[-1]["q_star"]
    if pair_n.reinforce_score < 0.9 * q_star_n1:
        return False  # rule 3
    return True
```

### 7.2 Reuse (pair is valid)

When a pair from cycle `n` passes all validity checks, it is **reused in cycle `n+1`** with the following adjustments:

- The contrast trajectory is **always replaced** with the lowest-scoring attempt from cycle `n+1`. Even if the old contrast trajectory is technically still valid, using the newest low-scoring attempt keeps the DPO signal grounded in the model's current failure mode rather than a stale one.
- The `gap` and `priority_score` are recomputed from cycle `n+1` scores.
- The `structural_divergence_step` is re-computed using `SelfEngine._find_divergence_step()` (`disteval/self_engine.py`, lines 553–575) on the current reinforce/contrast pair, since the model may have changed its structural behavior.

### 7.3 Invalidation and regeneration (pair fails a validity check)

When a pair from cycle `n` fails any validity check:

1. The old pair is moved to `archived_pairs` in the registry entry.
2. A new pair is generated from cycle `n+1` trajectories using `SelfEngine._build_training_pairs()` (`disteval/self_engine.py`, lines 505–551).
3. The `structural_divergence_step` is re-detected on the new pair.
4. If the sub-task was STUCK in cycle `n` and is now RECOVERABLE in cycle `n+1`, the `boundary_source` is updated from `"memory"` to `"structural_divergence"` (more reliable).

### 7.4 Sub-task graduation: pair becomes a reinforce demonstration

When a sub-task transitions to SOLID, its best training pair transitions from "active DPO pair" to "reinforce demonstration":

- The archived pair becomes queryable by `TrajectoryMemory.retrieve_for_sub_task()` (Phase 2C, Section 5.2).
- Other agents that are STUCK on the same `sub_task_id` can use this pair as a bootstrap reinforce target (Phase 3C cross-agent sharing design).

This is the key mechanism by which a cycle `n` solution "affects" cycle `n+1` even for a different agent: the SOLID agent's archived pair enters the shared trajectory memory and can unblock other agents.

---

## 8. Convergence and termination criteria for the loop

The loop terminates when one of the following conditions is met. Multiple criteria are evaluated at the end of each cycle's Stage 7.

### 8.1 Full consistency convergence (ideal termination)

```
κ_{global}(n) = sum_q_bar / sum_q_star ≥ THRESHOLD_KAPPA  (default: 0.95)
```

`κ_{global}` is the consistency index computed over **all active (non-retired) sub-tasks** in the environment registry, using the aggregated score histories. When every active sub-task has `consistency ≥ 0.95`, the agent is achieving at least 95% of its demonstrated capability on every sub-task consistently.

At this point, `SelfImprovementPlan.cycle_complete = True` (which currently checks `n_recoverable == 0` in `disteval/self_engine.py`, line 431). Under the recursive loop, the condition is extended:

```python
cycle_complete = (
    n_recoverable == 0
    and len([e for e in registry if e.status == "active"]) == 0
)
```

### 8.2 No improvement convergence (plateau detection)

```
|κ_{global}(n) - κ_{global}(n-1)| < THRESHOLD_DELTA  for P consecutive cycles
```

Default: `THRESHOLD_DELTA = 0.005`, `P = 3`. If the consistency index has not improved by more than 0.5% for three consecutive cycles, the loop is considered converged even if `κ < 0.95`. This detects the case where DPO training can no longer close the gap (a capability ceiling has been reached).

The loop then transitions to "capability expansion mode": RECOVERABLE tasks with persistent gaps are re-classified as exploration targets for Stages 3–4 (deeper decomposition, new tools, human examples).

### 8.3 STUCK saturation (all remaining tasks are recursively stuck)

```
all(e.status in {"recursively_stuck", "depth_cap", "retired"} for e in registry)
```

When every non-retired entry in the registry has no demonstrated solution at any depth, DPO training is exhausted. No training pairs exist. The loop terminates with the report: "All remaining tasks require capability expansion."

This condition is equivalent to the `SelfEngine` state where `n_recoverable == 0` and `n_stuck == N` (the agent has never solved the remaining tasks). The existing `SelfEngine.run_cycle()` already detects this implicitly through the empty `curriculum` check (`disteval/self_engine.py`, line 403 — `report.priority_tasks` is empty when no RECOVERABLE tasks exist).

### 8.4 Maximum cycle cap (safety)

```
n >= MAX_CYCLES  (default: 20)
```

A hard cycle cap prevents unbounded resource consumption in pathological cases (e.g., a task that oscillates between SOLID and RECOVERABLE due to environment non-determinism).

### 8.5 Convergence metrics tracked per cycle

The following metrics are written to `convergence_log.jsonl` each cycle for monitoring:

```json
{
  "cycle": 3,
  "kappa_global": 0.81,
  "n_active": 8,
  "n_retired": 4,
  "n_recoverable": 6,
  "n_stuck": 2,
  "n_recursively_stuck": 0,
  "recoverable_score_left": 0.45,
  "predicted_total_gain": 0.09,
  "kappa_delta_from_prev": 0.07,
  "termination_reason": null
}
```

When `termination_reason` is set (to `"kappa_threshold"`, `"plateau"`, `"stuck_saturation"`, or `"max_cycles"`), the loop stops.

---

## 9. Open questions for Phase 3C and Phase 4

### 9.1 Open questions for Phase 3C (Distributed Evaluations)

These questions arise directly from the single-agent loop described here and require multi-agent aggregation to answer.

1. **Cross-cycle sub-task identity:** When two agents decompose the same parent task independently, they may produce boundaries at slightly different steps. How do we canonicalize `sub_task_id` across agents? Option A: use the checkpoint-aligned boundary (from `test.sh`) as the canonical one. Option B: use a content hash of the tool-sequence pattern at the boundary. Option C: allow agent-specific sub-task IDs and align them post-hoc by similarity.

2. **Shared environment registry:** Should the `environment_registry.jsonl` be a per-agent file or a shared, multi-agent file? A shared file enables cross-agent pair sharing but requires a merge strategy when two agents disagree on whether a sub-task is SOLID vs RECOVERABLE.

3. **Registry merge on disagreement:** If Agent A classifies `medium-2::phase-1` as SOLID but Agent B classifies it as RECOVERABLE, the shared registry must resolve the conflict. Candidate rules: (a) take the pessimistic classification (RECOVERABLE wins over SOLID); (b) take the majority vote across agents; (c) keep per-agent entries and only compute global consensus lazily.

4. **Cross-agent reinforce targets:** The Phase 3B design archives a SOLID sub-task's training pair for potential use by other agents. When should a different agent's archived pair be used as a reinforce target? Threshold on structural similarity (`TrajectoryMemory` cosine similarity `>= 0.5`)? Or always?

5. **Evaluation frequency vs. training frequency:** In the current loop, every cycle runs the full benchmark. With a distributed multi-agent setting, should different agents run evaluations at different cadences (e.g., Agent A re-evals every 2 cycles, Agent B every 3)? How does the environment distribution stay in sync across agents on different schedules?

### 9.2 Open questions for Phase 4 (Prototype Integration)

1. **EnvironmentRegistry implementation:** Should `EnvironmentRegistry` be a standalone module (`disteval/environment_registry.py`) that wraps the JSONL file, or should it extend the existing `RecordStore` (`disteval/records.py`)? The key difference is that `RecordStore` is a flat table of `EpisodeRecord` objects, while the registry is a hierarchical, append-only log with rich per-sub-task history.

2. **Harbor task generation:** Sub-tasks as defined in the registry have `entry_step`, `exit_step`, and `entry_context`, but Harbor tasks require a `task.toml` and a `tests/test.sh`. Should Phase 4 design a sub-task runner that injects `entry_context` into a stripped-down copy of the parent task, or should it generate new task directories in a `sub_tasks/` folder?

3. **Entry-state replay:** The `entry_context` (tool-call prefix `steps[0:entry_step]`) tells the DPO trainer what happened before the sub-task, but it does not capture the file-system state. For tasks like `medium-2` where `entry_context` includes `write_file("client.py", ...)`, the DPO trainer needs the file to exist. How is this replayed? Options: (a) snapshot the Harbor container state at `entry_step`; (b) include a synthetic `write_file` step at the beginning of the sub-task training pair; (c) pre-populate the sub-task environment with the output of replaying the context prefix.

4. **Pair validity under model version change:** When the agent is fine-tuned between cycle `n` and `n+1`, its structural behavior changes. A training pair from cycle `n` (pre-fine-tune) may have a different `structural_divergence_step` after fine-tuning. Should validity rule 2 (boundary shift check) use the old model's divergence step or re-detect it on the new model's trajectories?

5. **Reward shaping at sub-task level:** The current `training_sim` uses a flat DPO improvement formula (`α × DPO_BONUS × q_star × (1 - q_bar)`). For sub-tasks with `reward_delta < 0.15` (e.g., M_2a at 0.10), the raw improvement is small. Should Phase 4 normalize sub-task reward deltas to `[0, 1]` before applying the DPO formula, then rescale back to the parent's coordinate system? Or does the current formula naturally handle small deltas?

6. **Cycle_complete semantics extension:** The current `SelfImprovementPlan.cycle_complete` flag (`disteval/self_engine.py`, line 131) is set to `True` when `n_recoverable == 0`. Under the recursive loop, there can be zero RECOVERABLE top-level tasks but many active RECOVERABLE sub-tasks. Should `cycle_complete` be renamed to `flat_cycle_complete` and a new `recursive_cycle_complete` flag added that checks the full registry?

7. **`training_sim` extension for recursive propagation:** `_fast_apply_improvement()` (`disteval/training_sim.py`, lines 351–396) currently applies improvement per top-level task. The recursive version needs to apply improvement per sub-task and then aggregate upward via the weighted-sum rule. The `sub_task_weights` parameter proposed in Phase 2C (`research/phase2c_integration_design.md`, Section 5.3) needs a concrete API and integration into `simulate_rounds_to_threshold()`.

8. **Convergence calibration:** The threshold `THRESHOLD_KAPPA = 0.95` and plateau detection parameters (`THRESHOLD_DELTA = 0.005`, `P = 3`) are heuristic. Phase 4 should calibrate these using the existing 37-trajectory Harbor dataset by simulating the loop and checking whether the thresholds produce sensible termination on the observed score progressions for Claude, Gemini, and Codex CLI.

---

## Appendix: Quick-reference table — cycle-n artifacts and their role in cycle n+1

| Artifact | Produced by | Consumed by cycle n+1 |
|---|---|---|
| `plan_n.json` | `SelfImprovementPlan.to_dict()` | Stage 2: taxonomy comparison; Stage 7: delta computation |
| `environment_registry.jsonl` (updated by cycle n) | Stage 7 registry update | Stage 2: boundary lookup; Stage 4: env generation |
| `convergence_log.jsonl` | Stage 7 convergence check | Stage 8 (loop controller): terminate or continue |
| Trajectory files in `jobs/run_n/` | Harbor eval (Stage 1) | Stage 3: `RecursionEngine.decompose()` reads traj_records_n |
| Archived training pairs (in registry) | Stage 4: pair invalidation path | Stage 3: memory retrieval for STUCK sub-tasks; cross-agent sharing |
| Sub-task graph (`sub_task_graph` in plan JSON) | Stage 3: `RecursionEngine.decompose()` | Stage 3 of cycle n+1: carryover scores merged into sub-task profiles |
| `TrajectoryMemory` (in-memory, rebuilt from job_dirs) | `SelfEngine.__init__()` | Stage 3: `retrieve_for_sub_task()` bootstrap for STUCK sub-tasks |

---

## Appendix: Worked example — two-cycle trace for medium-2 (Codex CLI)

**Cycle 0 starting state:**
- Codex CLI scores on `medium-2`: `[0.0, 0.0, 1.0]`
- Root profile: RECOVERABLE, `q_star=1.0`, `q_bar=0.333`, `gap=0.667`, `κ=0.333`
- No prior registry entries.

**Cycle 0 execution:**
1. EVAL: 3 Harbor runs produce scores `[0.0, 0.0, 1.0]`.
2. TAXONOMY: `medium-2` → RECOVERABLE; `M_2a`–`M_2e` all → RECOVERABLE (one successful attempt each).
3. DECOMPOSITION: Boundaries detected at `test.sh` checkpoints; 5 sub-tasks registered.
4. ENV GENERATION: All 5 sub-tasks added as `status="active"`.
5. TRAINING: DPO pairs generated for `M_2c`, `M_2d`, `M_2e` (highest gaps); fine-tuning runs.
6. RE-EVAL: 3 new Harbor runs.

**Cycle 1 starting state (hypothetical):**
- New scores: `[0.35, 0.35, 1.0]` (groupby phases slightly improved)
- `M_2a` per-attempt: `[1, 1, 1]` → SOLID
- `M_2b` per-attempt: `[1, 1, 1]` → SOLID
- `M_2c` per-attempt: `[0, 0, 1]` → still RECOVERABLE
- `M_2d` per-attempt: `[0, 0, 1]` → still RECOVERABLE
- `M_2e` per-attempt: `[0, 0, 1]` → still RECOVERABLE

**Cycle 1 execution:**
1. TAXONOMY: `M_2a`, `M_2b` → `kind_history` gains a `solid` entry.
2. DISTRIBUTION UPDATE: `M_2a`, `M_2b` → `status="retired"` (two consecutive cycles both `solid`; though here it is first time; with the two-cycle rule, they would be retired after cycle 2 confirms). Under the single-cycle rule (for demonstration): retired immediately.
3. ENV GENERATION: Active set = `{M_2c, M_2d, M_2e}`.
4. TRAINING: New DPO pairs for `M_2c`, `M_2d`, `M_2e` with fresh trajectories from cycle 1.
5. CONVERGENCE CHECK: `κ_{global}(1) = (1×0.333 + 1×0.333) + (3×0.333) / 5 ≈ 0.55` (rising from 0.333 in cycle 0 — not yet converged).

**Cycle 2 (projected):**
- If `M_2c`, `M_2d`, `M_2e` training succeeds: all three reach `SOLID`.
- `κ_{global}(2) → 1.0` (all active sub-tasks retired).
- `termination_reason = "kappa_threshold"` triggered.
- Final `medium-2` mean score: `1.0` (every attempt achieves all checkpoints).
