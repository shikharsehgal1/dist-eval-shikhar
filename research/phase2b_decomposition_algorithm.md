# Phase 2B — Decomposition Algorithm for the `RecursionEngine`

**Date:** 2026-06-22  
**Input:** `research/phase1_master_report.md`, `research/phase1b_disteval_mapping.md`, `research/phase1c_integration_questions.md`, and the disteval source files `disteval/self_engine.py`, `disteval/trajectory_monitor.py`, `disteval/trajectory_memory.py`, `disteval/right_tail.py`.  
**Output:** Design document for the decomposition algorithm that `RecursionEngine` will use to turn STUCK or RECOVERABLE tasks into sub-task RMDPs.  
**Constraint:** No existing code is modified; this document is research-only.

---

## 1. Overview

The `RecursionEngine` is a new module that sits alongside `SelfEngine` (`disteval/self_engine.py`). It takes the same SOLID / RECOVERABLE / STUCK classification produced by `right_tail_analysis()` (`disteval/right_tail.py`) and, for tasks that are not yet SOLID, produces a tree of **sub-task RMDPs** with explicit entry/exit conditions. Each sub-task is itself a `TaskOutcomeProfile` and can be trained with the same right-tail machinery as a full task.

The design below is deliberately close to the existing code: it reuses `TrajectoryMonitor` to find structural boundaries, `TrajectoryMemory` to retrieve demonstration context, and `right_tail.task_outcome_profile()` to score sub-tasks. Where new data is required (e.g., per-checkpoint rewards), the document proposes extensions that are backward-compatible.

---

## 2. Identifying candidate tasks for decomposition

### 2.1 Inputs

The engine consumes a `RightTailReport` from `right_tail_analysis()` (`disteval/right_tail.py`, lines 207–263). The report contains `TaskOutcomeProfile` objects with:

- `kind`: `"solid"`, `"recoverable"`, or `"stuck"`
- `q_star`: best score on the task
- `q_bar`: mean score
- `gap`: `q_star - q_bar`
- `scores`: all attempt scores, in order
- `reinforce_idx` / `contrast_idx`: indices above/below the `0.9 * q_star` threshold

These fields are defined at `disteval/right_tail.py` lines 116–132 and computed at lines 163–204.

### 2.2 Candidate categories

A task is a decomposition candidate if and only if it is **not SOLID**. The engine treats two sub-cases differently because the available signals are different.

| Candidate type | Criterion | Why it is decomposable | Decomposition strategy |
|---|---|---|---|
| **Partial-credit RECOVERABLE** | `kind == "recoverable"` **and** there exists at least one low-scoring attempt with `0 < score < 0.9 * q_star` | The agent has demonstrated some sub-tasks (the partial-credit milestones) but not others. The low-scoring trajectories contain a successful prefix that can be sliced. | **Divergence-point cleavage**: split at the monitor divergence step between a high-scoring and a low-scoring trajectory. |
| **Pure STUCK** | `kind == "stuck"` (i.e., `q_star == 0`) | No successful trajectory exists on this task. The only signal comes from structurally similar tasks in memory. | **Memory-based fallback**: retrieve high-scoring trajectories from similar tasks via `TrajectoryMemory` and use them as synthetic reinforce targets. |

The `reinforce_threshold = 0.9` is the same default used by `task_outcome_profile()` (`disteval/right_tail.py`, line 168) and `SelfEngine._build_training_pairs()` (`disteval/self_engine.py`, line 524).

### 2.3 Why the distinction matters

For a partial-credit RECOVERABLE task, the engine can create training pairs **entirely from the task's own trajectories**: the high-scoring run provides the reinforce slice, and the low-scoring partial-credit run provides the contrast slice for the same sub-task. For a pure STUCK task, no within-task high slice exists, so the engine must either import a demonstration from a similar task in `TrajectoryMemory` or mark the task as requiring new capability.

---

## 3. Entry and exit conditions for a sub-task

### 3.1 The boundary signal

The engine uses the `TrajectoryMonitor` as a boundary detector. The relevant existing methods are:

- `TrajectoryMonitor.check(steps, prefix_n)` — returns a `PatternMatch` with `p_high`, `prediction`, and `confidence` for a trajectory prefix (`disteval/trajectory_monitor.py`, lines 488–524).
- `SelfEngine._find_divergence_step(high_path, low_path)` — returns the first step where a high run is predicted `high` and a low run is predicted `low` or `uncertain` (`disteval/self_engine.py`, lines 553–575).

The engine proposes to expose this logic as a first-class public method:

```python
class TrajectoryMonitor:
    def find_phase_boundaries(
        self,
        high_path: str,
        low_path: str,
        max_check: int = 20,
    ) -> list[PhaseBoundary]:
        ...
```

where each `PhaseBoundary` records:

- `step_index`: the tool-call index where the boundary occurs
- `tool_name`: the canonical tool at that step
- `p_high_before`: `p_high` at `step_index - 1`
- `p_high_after`: `p_high` at `step_index`
- `phase_tag`: a coarse tag derived from the tool (`read`, `write`, `exec`, `search`, `test`, `unknown`)

### 3.2 Entry condition

The entry of sub-task `i` is defined as the step index immediately after the previous boundary, or `0` for the first sub-task. Formally:

```
entry_0 = 0
entry_i = boundary_{i-1} + 1   for i > 0
```

The **entry condition** is the conjunction of:

1. The tool-sequence prefix `steps[0:entry_i]` is present as conditioning context.
2. The environment state at step `entry_i` (file system, terminal output, any partial JSON produced) is captured as the initial state of the sub-task.
3. A memory retrieval from `TrajectoryMemory.retrieve_for_new_task(sub_task_description, k=3)` (`disteval/trajectory_memory.py`, lines 290–305) provides successful structural templates from similar past tasks.

### 3.3 Exit condition

The exit of sub-task `i` is the boundary step `boundary_i`. The **exit condition** is:

1. The sub-task slice reaches `step_index = boundary_i`.
2. The monitor's `p_high` at that prefix is above the high-confidence threshold (`>= 0.65`) for a successful sub-task, or below the low-confidence threshold (`< 0.35`) for a failed one.
3. When per-checkpoint test data is available, the exit is also validated by the corresponding test checkpoint (e.g., `summary.json` is valid JSON, or `total_eligible_users == 7`).

The last sub-task's exit is the final step of the trajectory, and its exit condition is the full `test.sh` reward scalar.

### 3.4 Tool-call boundary vs. semantic checkpoint

The `structural_divergence_step` is a **symptom** of a boundary, not necessarily a semantic one. For example, in `medium-2` the divergence between a perfect run and a failed run may occur at step 3 (first `write_file`) while the semantic checkpoint is the execution of `client.py`. The engine therefore **cross-validates** boundaries against two other signals when possible:

- **Test-script checkpoints:** `tasks/medium-2/tests/test.sh` (lines 25–64) defines five checkpoints. A `TestSuiteParser` will read these checkpoints and their weights to produce a semantic segmentation.
- **Memory phase tags:** `TrajectoryMemory` already stores `tool_sequence` and `first_write_pos` (`disteval/trajectory_memory.py`, lines 36–50). The engine prefers boundaries that align with a tag transition (`read → write`, `write → exec`, `exec → search`).

If the signals disagree, the engine keeps the test-script checkpoint boundary when available and falls back to the structural boundary otherwise. This is the conservative, data-grounded choice because the test script is the ground-truth reward function.

---

## 4. Splitting a parent trajectory into sub-trajectory slices

### 4.1 Boundary detection algorithm

```text
FUNCTION FindSubTaskBoundaries(task_profile, trajectories, monitor, memory, test_parser):
    # task_profile: TaskOutcomeProfile
    # trajectories: list of (score, steps, traj_path) for this task

    IF task_profile.kind == "recoverable" AND has_partial_credit(task_profile):
        high_runs  = runs with score >= 0.9 * q_star
        low_runs   = runs with score < 0.9 * q_star
        boundary_set = {}
        FOR each high_run IN high_runs:
            FOR each low_run IN low_runs:
                d = monitor.find_divergence_step(high_run.path, low_run.path)
                IF d > 0: boundary_set.add(d)
        boundaries = [0] + sorted(boundary_set) + [max_steps]

    ELSE IF task_profile.kind == "stuck":
        mem_results = memory.retrieve_for_new_task(task_profile.task, k=5)
        boundaries = boundaries_from_memory_phase_tags(mem_results, trajectories)

    ELSE:
        RETURN []   # solid task, no decomposition

    # Merge adjacent boundaries that are too close (fewer than MIN_SEGMENT_LEN tool calls)
    boundaries = merge_short_segments(boundaries, trajectories, min_len=2)
    RETURN boundaries
```

This reuses the monitor's existing `check` method (`disteval/trajectory_monitor.py`, lines 488–524) and the divergence logic from `SelfEngine._find_divergence_step` (`disteval/self_engine.py`, lines 553–575).

### 4.2 Slice extraction and context handling

Given boundaries `b_0 = 0 < b_1 < ... < b_m = N`, sub-task `i` corresponds to the slice `steps[b_i : b_{i+1}]`. The engine extracts this slice for **every** parent trajectory of the task, producing a set of sub-trajectories that all share the same entry/exit semantics.

**Context at the entry boundary** is handled by keeping two pieces of information:

1. **Context prefix:** `steps[0 : b_i]` is stored as `entry_context` in the sub-task. It is not part of the training episode but is provided to the DPO trainer or environment generator as conditioning so the model understands how the environment reached this state.
2. **Entry state snapshot:** If the runner supports it, the file-system / terminal state at step `b_i` is captured. If not, the engine synthesizes the entry state from the context prefix by replaying the file writes and shell commands in the prefix (this is a Phase 3 environment-generation concern).

The proposed `TrainingPair` extension is:

```python
@dataclass
class TrainingPair:
    task: str
    reinforce_traj_path: str
    contrast_traj_path: str
    reinforce_score: float
    contrast_score: float
    gap: float
    structural_divergence_step: int
    # New fields for recursion
    parent_task: str | None = None
    sub_task_depth: int = 0
    entry_step: int = 0
    exit_step: int = -1
    entry_context: list[dict] = field(default_factory=list)
```

This aligns with the `TrainingPair` definition at `disteval/self_engine.py` lines 70–78 and the `RecursionContext` recommendation in `research/phase1c_integration_questions.md` (lines 184–190).

### 4.3 Scoring each slice

The engine needs a scalar score for each sub-task slice on each parent attempt. The preferred method is to extend the test harness to emit per-checkpoint scores; in the meantime, the engine uses the following hierarchy:

1. **Per-checkpoint test scores** (target): parse `test.sh` to know each checkpoint and its weight, run the parent trajectory up to the exit step, and record which checkpoints are satisfied. This is the cleanest method and directly reflects the reward function.
2. **Structural proxy**: if per-checkpoint scores are unavailable, the score of a slice is `1.0` if the monitor predicts `high` at the slice's final step and `0.0` otherwise. This is noisy but requires no new instrumentation.
3. **Full-score inheritance**: for the last segment only, the slice score equals the full-task score. This is a fallback that preserves the parent reward but does not disentangle sub-tasks.

Method 1 is recommended because `tasks/medium-2/tests/test.sh` already has a natural 5-checkpoint scoring structure (lines 25–64). The test script can be extended to write `/logs/verifier/reward_c{i}.txt` without changing the total reward.

---

## 5. Recursive sub-task analysis

### 5.1 Re-running right-tail analysis on sub-tasks

After scoring each sub-task slice across all parent attempts, the engine creates a `TaskOutcomeProfile` for the sub-task by calling `task_outcome_profile()` from `disteval/right_tail.py` (lines 163–204) with the sub-task score array.

The sub-task profile has the same semantics as a full-task profile:

- `SOLID`: the sub-task is already consistently solved; no further action.
- `RECOVERABLE`: the sub-task is sometimes solved but not always; generate DPO pairs from the sub-task slices.
- `STUCK`: the sub-task is never solved; attempt further decomposition or mark it as recursively stuck.

### 5.2 Recursive decomposition procedure

```text
FUNCTION Decompose(task_profile, depth, trajectories, monitor, memory, test_parser):
    IF task_profile.kind == "solid":
        RETURN []

    IF depth > MAX_DEPTH:
        RETURN [LeafSubTask(task_profile, depth, reason="depth_cap")]

    boundaries = FindSubTaskBoundaries(task_profile, trajectories, monitor, memory, test_parser)
    IF len(boundaries) < 2:
        RETURN [LeafSubTask(task_profile, depth, reason="no_boundary")]

    weights = test_parser.checkpoint_weights(task_profile.task) OR uniform_weights(boundaries)

    subtasks = []
    FOR i IN 0 .. len(boundaries)-2:
        entry = boundaries[i]
        exit  = boundaries[i+1]
        sub_id = f"{task_profile.task}#{i}"

        sub_scores = ScoreSegment(i, trajectories, weights[i], monitor, test_parser)
        sub_profile = task_outcome_profile(sub_id, sub_scores)

        subtask = SubTaskRMDP(
            task_id=sub_id,
            parent_task=task_profile.task,
            depth=depth,
            weight=weights[i],
            entry_step=entry,
            exit_step=exit,
            profile=sub_profile,
        )

        IF sub_profile.kind == "stuck":
            subtask.children = Decompose(sub_profile, depth+1, slice_trajs, ...)
        ELSE IF sub_profile.kind == "recoverable":
            subtask.training_pairs = BuildSubTaskPairs(subtask, trajectories, monitor)

        subtasks.append(subtask)

    # Recursively-stuck detection
    IF all(s.profile.kind == "stuck" AND not s.children for s in subtasks):
        FOR s IN subtasks: s.recursively_stuck = True

    RETURN subtasks
```

`MAX_DEPTH` is recommended to be `3` (root depth = 0), matching the Phase 1C recommendation (`research/phase1c_integration_questions.md`, line 283).

### 5.3 Sub-task training-pair construction

For a RECOVERABLE sub-task, the engine builds reinforce/contrast pairs from the **same sub-task segment** across different parent trajectories. The logic mirrors `SelfEngine._build_training_pairs()` (`disteval/self_engine.py`, lines 505–551):

1. Identify high sub-task scores (`>= 0.9 * sub_q_star`) and low sub-task scores (`< 0.9 * sub_q_star`).
2. For each low-scoring slice, pair it with the highest-scoring slice of the same segment.
3. Compute a sub-task divergence step by calling `monitor.find_divergence_step` on the slice files (or on the slice step arrays).
4. Store the pair with `entry_step`, `exit_step`, and `entry_context` from the parent trajectory.

This keeps the DPO signal local to the sub-task instead of using the whole parent trajectory, which is the central benefit of the decomposition.

### 5.4 Reward propagation to the parent task

When sub-task scores are available, the parent task's score is reconstructed as a **weighted sum** of sub-task scores. For `medium-2` the weights are derived directly from `tests/test.sh`:

| Sub-task | Weight | Source in `tests/test.sh` |
|---|---|---|
| C0: valid JSON / client runs | 0.10 | line 26 (`SCORE += 10`) |
| C1: `total_eligible_users == 7` | 0.25 | line 34 (`SCORE += 25`) |
| C2: Engineering groupby | 0.25 | line 44 (`SCORE += 25`) |
| C3: Sales groupby | 0.20 | line 54 (`SCORE += 20`) |
| C4: HR groupby | 0.20 | line 64 (`SCORE += 20`) |

For a parent task `T` decomposed into sub-tasks `T_1, ..., T_m` with weights `w_i`:

```
Q*_T = Σ_i w_i * Q*_{T_i}
Q̄_T  = Σ_i w_i * Q̄_{T_i}
Δ_T  = Σ_i w_i * Δ_{T_i}
```

This is the **recursive right-tail** aggregation recommended in `research/phase1c_integration_questions.md` (lines 66–72). It has the important property that improving a single sub-task `T_i` lifts `Q̄_T` by `w_i * Δ_{T_i}`, which is exactly the targeted training signal the engine wants.

Sub-task priority is computed with the same formula as full tasks:

```
priority_score(T_i) = Δ_{T_i} * (1 - κ_{T_i}) * w_i
```

The extra `w_i` factor ensures that a sub-task contributing more to the parent score is ranked higher.

---

## 6. Termination rules

### 6.1 Hard depth cap

`MAX_DEPTH = 3` by default. Depth 0 is the original task, so the engine can create sub-tasks (depth 1), sub-sub-tasks (depth 2), and sub-sub-sub-tasks (depth 3), but never decomposes a node at depth 3. This is a safety guard, not a convergence guarantee, and is logged in the decomposition tree as `reason="depth_cap"`.

### 6.2 Monotone difficulty check

Before decomposing a sub-task, the engine verifies that the proposed children are **strictly simpler** than the parent. Difficulty is estimated from the trajectory slice:

```
complexity(slice) = len(slice_steps) + tool_diversity(slice)
```

where `tool_diversity` is `len(set(tool_sequence)) / len(tool_sequence)` (already computed by `TrajectoryFeaturizer` at `disteval/trajectory_monitor.py`, lines 149–220). A decomposition is accepted only if:

```
complexity(child_slice) < complexity(parent_slice) - MIN_DIFFICULTY_DELTA
```

with a default `MIN_DIFFICULTY_DELTA = 1`. If this check fails, the sub-task is kept as a leaf.

### 6.3 Recursively stuck detection

A sub-task is **recursively stuck** if:

- It is classified as `STUCK`, and
- All of its children (if any) are also classified as `STUCK` or are leaves stopped by the depth cap / monotone check, and
- No RECOVERABLE or SOLID sub-task was produced below it.

Recursively stuck tasks are not added to the DPO curriculum. Instead, they are flagged in the decomposition report with `recursively_stuck = True` and routed to a capability-expansion path (e.g., more episodes, new tools, or human-designed examples). This prevents the engine from wasting training cycles on sub-tasks that have no demonstrated solution at any depth.

---

## 7. Worked example: `disteval/medium-rest-client` (`tasks/medium-2/`)

### 7.1 Task structure

The task is defined in `tasks/medium-2/instruction.md` (lines 8–25) and scored by `tasks/medium-2/tests/test.sh` (lines 25–64). The scoring is a 5-checkpoint chain:

| Checkpoint | Reward | Criterion | RMDP sub-task |
|---|---|---|---|
| C0 | 0.10 | `client.py` exists and produces valid JSON (`tasks/medium-2/tests/test.sh`, line 26) | `M_2a` — HTTP client / JSON |
| C1 | 0.25 | `total_eligible_users == 7` (`tasks/medium-2/tests/test.sh`, line 34) | `M_2b` — filtering |
| C2 | 0.25 | Engineering groupby correct (`tasks/medium-2/tests/test.sh`, line 44) | `M_2c` — Engineering groupby |
| C3 | 0.20 | Sales groupby correct (`tasks/medium-2/tests/test.sh`, line 54) | `M_2d` — Sales groupby |
| C4 | 0.20 | HR groupby correct (`tasks/medium-2/tests/test.sh`, line 64) | `M_2e` — HR groupby |

The mock data is in `tasks/medium-2/environment/app/mock_server.py` (lines 7–18). The expected eligible users are Alice, Carol, Eve, Frank, Henry, Iris, Jack (7 users), as noted in `tests/test.sh` line 28.

### 7.2 Scenario A: partial-credit RECOVERABLE agent

Suppose an agent has the following per-checkpoint scores on three attempts (per-checkpoint scores are inferred from the test script; the final scores match the known Codex pattern from `THEORY.md` and `research/phase1_master_report.md`):

| Attempt | Final score | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| A0 | 0.35 | 1 | 1 | 0 | 0 | 0 |
| A1 | 0.00 | 0 | 0 | 0 | 0 | 0 |
| A2 | 1.00 | 1 | 1 | 1 | 1 | 1 |

Root profile: `Q* = 1.00`, `Q̄ = 0.45`, `Δ = 0.55`, `κ = 0.45`, `kind = recoverable`. It is a partial-credit RECOVERABLE candidate because A0 has a non-zero low score.

The engine decomposes by the five test checkpoints. The resulting sub-task profiles are:

| Sub-task | Weight | Scores | `Q*` | `Q̄` | `Δ` | `κ` | Kind |
|---|---|---|---|---|---|---|---|
| `M_2a` | 0.10 | [1, 0, 1] | 1.0 | 0.667 | 0.333 | 0.667 | RECOVERABLE |
| `M_2b` | 0.25 | [1, 0, 1] | 1.0 | 0.667 | 0.333 | 0.667 | RECOVERABLE |
| `M_2c` | 0.25 | [0, 0, 1] | 1.0 | 0.333 | 0.667 | 0.333 | RECOVERABLE |
| `M_2d` | 0.20 | [0, 0, 1] | 1.0 | 0.333 | 0.667 | 0.333 | RECOVERABLE |
| `M_2e` | 0.20 | [0, 0, 1] | 1.0 | 0.333 | 0.667 | 0.333 | RECOVERABLE |

Reconstructed parent score:

```
Q̄_T = 0.10*0.667 + 0.25*0.667 + 0.25*0.333 + 0.20*0.333 + 0.20*0.333
    ≈ 0.45          ← matches the observed parent Q̄
```

The engine would generate training pairs for `M_2c`, `M_2d`, and `M_2e` first because their gaps (`0.667`) are larger, even though `M_2a` and `M_2b` also have non-zero gaps. This is more targeted than the current whole-task DPO pair, which trains on the entire trajectory.

### 7.3 Scenario B: pure STUCK agent

Suppose a different agent has scores `[0.0, 0.0, 0.0]` on `medium-2`. The root is STUCK, so the engine cannot use internal divergence-point cleavage. It instead calls `TrajectoryMemory.retrieve_for_new_task("medium-rest-client", k=3)` (`disteval/trajectory_memory.py`, lines 290–305). The memory returns, for example, a successful trajectory from the agent in Scenario A on `medium-2` or a high-scoring trajectory from a similar JSON-processing task.

From the retrieved memory, the engine extracts the tool-sequence phase boundaries (`read → write → exec → read`) and creates candidate sub-tasks:

```
M_2a  → write client.py
M_2b  → run client.py and validate JSON
M_2c  → verify total_eligible_users
M_2d  → verify Engineering groupby
M_2e  → verify Sales groupby
M_2f  → verify HR groupby
```

Because the current agent has no successful slices on this task, every sub-task score is `0.0` → every sub-task is STUCK. The engine recurses once more, but the structural slices are already single-tool actions (e.g., one `write_file` or one `run_shell_command`). The monotone difficulty check fails because the children are not simpler than the parents, so the recursion stops. The engine marks all sub-tasks as **recursively stuck** and reports that the task requires new capability, not DPO training.

### 7.4 Decomposition tree

The combined tree for Scenario A (RECOVERABLE) is:

```
medium-rest-client (depth 0, RECOVERABLE, Δ=0.55)
├── M_2a  (depth 1, weight 0.10, RECOVERABLE, Δ=0.33)
├── M_2b  (depth 1, weight 0.25, RECOVERABLE, Δ=0.33)
├── M_2c  (depth 1, weight 0.25, RECOVERABLE, Δ=0.67)
├── M_2d  (depth 1, weight 0.20, RECOVERABLE, Δ=0.67)
└── M_2e  (depth 1, weight 0.20, RECOVERABLE, Δ=0.67)
```

The sub-tasks are leaves because:

- Each corresponds to a single semantic checkpoint in `tests/test.sh`.
- The structural slices contain only a few tool calls (mostly one `write_file` and one `run_shell_command` for the groupby checks).
- The monitor cannot find a second divergence inside a single checkpoint, so `FindSubTaskBoundaries` returns fewer than two boundaries for each child.

For Scenario B (STUCK), the tree would have the same shape but every node would be marked `recursively_stuck = True` after the first decomposition attempt, with the reason `no_demonstrated_solution`.

---

## 8. Failure modes and handling

| Failure mode | Cause | Detection | Handling |
|---|---|---|---|
| **No structural divergence found** | The task has only one trajectory, or all trajectories have the same structural signature, or the monitor's predictor has not trained. | `FindSubTaskBoundaries` returns fewer than two boundaries. | Return a single leaf with `reason="no_boundary"`. For RECOVERABLE tasks, fall back to the whole-task DPO pair. For STUCK tasks, mark as recursively stuck. |
| **Boundary does not align with semantic checkpoint** | The tool-call divergence occurs inside a single test checkpoint (e.g., mid-edit). | The sub-task score array does not match the expected checkpoint structure; `κ` is inconsistent across attempts. | Prefer test-script checkpoint boundaries when available. If unavailable, accept the boundary but flag it as `boundary_confidence = "structural_only"`. |
| **Context loss at slice boundary** | The agent in a later slice relies on variables, files, or reasoning from earlier steps. | DPO trainer or environment sees a truncated trajectory. | Store `entry_context` with each slice. In Phase 3, the environment generator will replay the context prefix or load a snapshot state. |
| **Recursively stuck** | A STUCK task decomposes into sub-tasks that are also all STUCK. | All children have `kind == "stuck"` and no further children. | Mark the parent and children as `recursively_stuck = True`. Do not add to DPO curriculum. Surface in the report for human/capability-expansion intervention. |
| **Depth cap truncation** | A genuine multi-level decomposition would require depth > 3. | Node depth equals `MAX_DEPTH`. | Return a leaf with `reason="depth_cap"`. Log the unresolved path so the cap can be raised in future cycles if evidence supports it. |
| **Score inference noise** | Without per-checkpoint scores, structural proxies misclassify sub-tasks. | Sub-task profiles disagree with observed parent scores. | Extend the test harness to emit per-checkpoint rewards (`reward_c{i}.txt`). Until then, use the structural proxy and report `score_source = "structural_proxy"`. |
| **Cross-task memory mismatch** | A STUCK task retrieves a memory from a task that solves the problem in a structurally different way. | Retrieved memory has low cosine similarity or a different tool-sequence pattern. | Require a similarity threshold (`similarity >= 0.5`) and a task-keyword overlap check (`task_match >= 0.3`) from `TrajectoryMemory.retrieve()` (`disteval/trajectory_memory.py`, lines 214–288). If no memory passes, mark as recursively stuck. |

---

## 9. Open questions for Phase 3

Phase 3 is responsible for turning the decomposition tree into runnable RL environments. The following questions must be answered there:

1. **RL environment schema:** Should the generated environment be a single JSON file per sub-task, or a chain environment per parent task? Should it extend `CURRICULUM_FORMAT.md` (`/Users/shikharsehgal/rl-dist-eval/CURRICULUM_FORMAT.md`) or define a new `environments/` schema?
2. **Entry-state serialization:** How is the environment state at a sub-task entry (file system, terminal, mock server state) captured and replayed? Does the Harbor runner support stateful task initialization, or must the engine synthesize the state from the `entry_context`?
3. **Multi-exit vs. 1-exit:** `medium-2` has five checkpoints, which is a multi-exit RMDP. The paper's convergence guarantee applies cleanly to 1-exit RMDPs. Should Phase 3 generate one 1-exit environment per checkpoint, or one multi-exit environment per parent task with binarized exits?
4. **Reward shaping:** Should sub-task rewards be the checkpoint weights (e.g., 0.25 for `M_2c`) or a sparse `{0, 1}` pass/fail signal? The former preserves the parent score aggregation; the latter is more standard for RL.
5. **Cross-agent sub-task sharing:** If one agent is SOLID on `M_2c` while another is STUCK, should the successful agent's sub-task trajectories be used as reinforce targets for the stuck agent? This is not part of the RMDP formalism but could be the highest-leverage path for stuck tasks.
6. **Stochastic transitions:** LLM agents produce stochastic trajectories. How should the environment generator model transition probabilities so that Recursive Q-learning's convergence assumptions are reasonably approximated?
7. **Distributed eval feedback:** How do evaluations from multiple agents and multiple cycles update the sub-task RMDP parameters (e.g., shifting the task distribution or the entry conditions based on the previous cycle's solution)?

---

## 10. Summary of changes proposed (but not implemented)

The decomposition algorithm proposes these **additive, backward-compatible** extensions to the existing disteval primitives:

- `TrajectoryMonitor.find_phase_boundaries()` — a public boundary detector built on the existing `check()` and `find_divergence_step` logic (`disteval/trajectory_monitor.py`, lines 488–524; `disteval/self_engine.py`, lines 553–575).
- `TestSuiteParser` — a new helper that reads `tasks/{task}/tests/test.sh` to extract checkpoint weights and criteria, grounded in files like `tasks/medium-2/tests/test.sh` (lines 25–64).
- `SubTaskRMDP` dataclass — a new data structure carrying `entry_step`, `exit_step`, `weight`, `profile`, `children`, and `recursively_stuck`.
- `TrainingPair` extension — add `parent_task`, `sub_task_depth`, `entry_step`, `exit_step`, and `entry_context` (building on the existing `TrainingPair` at `disteval/self_engine.py`, lines 70–78).
- `RecursionEngine.decompose()` — the core recursive procedure described in Sections 4–5.
- `TaskImprovement.sub_tasks` — a recursive list field inside `TaskImprovement` (proposed in `research/phase1c_integration_questions.md`, lines 168–174) so the curriculum JSON can carry the full decomposition tree.

No existing disteval code is modified. The deliverable is this design document, which Phase 3 will consume to generate the actual RL environments.
