# Phase 1C: Integration Questions for a RecursionEngine in disteval

**Phase**: 1C — Pre-Design Integration Analysis  
**Input consumed**: Phase 1 RMDP mapping (SKILL.md), disteval codebase, external literature survey  
**Deliverable**: Integration questions, proposed answers, leverage ranking, library survey, open questions for Phase 2  
**Date**: 2025-06-22

---

## 0. Executive Summary

The disteval codebase already embodies the *spirit* of RMDP-style recursion in its SOLID/RECOVERABLE/STUCK taxonomy, its `structural_divergence_step`, and the call-stack analogy in `SelfEngine`'s docstring (lines 38–45 of `self_engine.py`). The gap is **operational**: no code actually decomposes a task into sub-tasks, assigns entry/exit conditions to trajectory waypoints, or propagates rewards across a simulated call-stack. Before designing a `RecursionEngine`, the following integration questions must be answered in precise, implementable terms.

---

## 1. Critical Integration Questions

### Q1 — How do we define Entry and Exit States for a disteval task?

In the formal RMDP (arXiv:2206.11430, §2), each component MDP `M_i` has a designated entry node `En_i` (the context when the component is *called*) and exit nodes `Ex_i` (the states at which execution *returns* to the parent). In disteval, a "task" is currently a flat trajectory from prompt to final score—there is no explicit structural decomposition.

**Why it matters**: Without entry/exit identification, we cannot construct sub-MDPs from trajectory segments, cannot define where a recursive call begins and ends, and cannot propagate the sub-task Q-value back to the parent task.

**Proposed answers**

| Option | Definition | Trade-offs |
|--------|-----------|------------|
| **A. Tool-call boundary** | Entry = first tool call of a structural phase (read → explore; write → implement; exec → verify). Exit = the last tool call before the phase changes, or the final score event. Operationally: the `structural_divergence_step` from `TrajectoryMonitor._find_divergence_step()` (lines 553–575, `self_engine.py`) identifies the boundary where two trajectories diverge, which is a natural candidate for the exit of a sub-MDP. | Pro: grounded in existing data; no new annotations. Con: phases are fuzzy—the divergence step is a symptom, not a guaranteed phase boundary. |
| **B. Score milestone** | Entry = trajectory start; exit = any step at which an intermediate verifiable milestone is achieved (e.g., first successful `exec_command` with exit code 0, or first write that passes a partial test). This mirrors the RMDP "1-exit" case. | Pro: clean termination semantics. Con: requires either test harness access or a separate outcome predictor at intermediate steps—not currently in disteval. |
| **C. LLM-annotated waypoints** | Use an LLM (or the existing `OutcomePredictor`) to label the trajectory steps with phase tags (PLAN / IMPLEMENT / TEST / DEBUG). Entry = first step of a phase; exit = last step. | Pro: flexible and rich. Con: introduces a dependency on a labeling model; may not generalize; not pure-numpy. |

**Recommendation**: Start with **Option A** (tool-call boundary). The divergence step is already computed by `SelfEngine._find_divergence_step()` (`self_engine.py`, line 539) and maps directly to "the step where trajectories structurally separate"—which is precisely the exit of the pre-divergence sub-MDP. Option B is the longer-term target once the test harness is accessible.

---

### Q2 — How do we represent the Call Stack?

In an RMDP, the call stack `hκ, bi` (paper §3) tracks the sequence of box invocations that led to the current execution point. The key insight from the paper (§3, "exit value abstraction") is that the stack can be abstracted to its *exit value*—the expected reward from each pending sub-task at the current stack level—without tracking the full history.

In disteval, the analogy is: when we recursively decompose a STUCK task `T` into sub-tasks `T1, T2, T3`, the "stack" records which parent task is waiting for each sub-task's result, so that when `T1` resolves, its score can propagate to `T`.

**Why it matters**: The call stack representation determines (a) how deep recursion can go before hitting a computational bound, (b) how scores are aggregated across levels, and (c) whether the system can detect circular dependencies or stack overflow.

**Proposed answers**

| Option | Representation | Trade-offs |
|--------|---------------|------------|
| **A. Flat list of (parent_task, sub_task, expected_return) tuples** | `stack: list[tuple[str, str, float]]` where `expected_return` is the current Q-estimate for that sub-task. Consistent with the paper's "exit value abstraction." Stored in `SelfImprovementPlan` as a new field. | Pro: simple, JSON-serializable, directly usable by existing `SelfImprovementPlan.to_dict()` (`self_engine.py`, line 169). Con: no way to track multiple simultaneous branches (parallel sub-tasks). |
| **B. Recursive dataclass tree** | A tree of `SubTaskNode(task, parent, depth, entry_step, exit_step, q_estimate, children)`. Enables multi-exit and parallel sub-tasks. | Pro: naturally models multi-exit RMDPs; enables visualization. Con: harder to serialize; requires new `recursion_engine.py` module. |
| **C. External stack file (JSONL)** | Each push/pop is logged as a JSONL event: `{"op": "push", "task": ..., "subtask": ..., "step": ..., "timestamp": ...}`. The `RecursionEngine` replays this log to reconstruct the call stack state. | Pro: traceable, auditable, supports asynchronous/distributed execution. Con: I/O overhead; requires reader; complexity for a first prototype. |

**Recommendation**: **Option A** for Phase 2 prototype (aligns with Phase 1's mapping where "call-stack value backup ↔ memory retrieval at matching structural depth," SKILL.md line 43). **Option B** for Phase 3 when multi-exit and parallel decomposition are needed.

---

### Q3 — How do we propagate rewards from sub-tasks to their parent task?

This is the core algorithmic question. In Recursive Q-learning (paper §3, Algorithm 1), the Q-value update at a state `q` in component `M_i` incorporates the *exit value* `v(ex)` of any box `b` encountered on the path—the expected reward upon returning from the recursive call. The paper proves convergence for 1-exit RMDPs and deterministic multi-exit RMDPs.

In disteval, "reward" is the task score (0–1). If task `T` is decomposed into sub-tasks `T1, T2`, the parent score must be a function of the sub-task scores.

**Why it matters**: The wrong aggregation function produces incorrect gradients for the DPO curriculum. For example, if `T1` and `T2` must both succeed for `T` to succeed, a product aggregation is appropriate; if either succeeding is sufficient, a max is appropriate. The aggregation also determines whether sub-task training pairs are correctly ranked by their contribution to parent score.

**Proposed answers**

| Option | Aggregation | Trade-offs |
|--------|------------|------------|
| **A. Sequential product** | `score(T) = score(T1) × score(T2) × ... × score(Tn)`. Models "all sub-tasks must succeed." Natural for coding tasks where each step is a prerequisite for the next. | Pro: correct for strictly sequential pipelines; differentiable. Con: a single zero-score sub-task collapses the entire parent score—harsh; sensitive to sub-task boundaries. |
| **B. Weighted sum** | `score(T) = Σ_i w_i × score(T_i)` where `w_i` is the estimated contribution of sub-task `i` to the parent (e.g., based on how many test cases it covers). | Pro: graceful degradation; aligns with how many benchmarks assign partial credit. Con: weights are unknown unless the test harness exposes per-sub-task coverage; estimation adds bias. |
| **C. Right-tail sub-task Q-value** | Apply the same right-tail logic recursively: `Q*(T) = max_k score_k(T)` and `Q*(T_i) = max_k score_k(T_i)`. The gap `Δ(T_i)` at the sub-task level tells us *which sub-task is the bottleneck* and should receive DPO training. The parent `Q*(T)` is only updated after the sub-task gaps are closed. | Pro: fully consistent with disteval's existing framework; reuses `right_tail_analysis()` (`right_tail.py`, line 207) at each level of recursion; no new math needed. Con: requires multiple runs at the sub-task level to estimate sub-task Q*—adds evaluation cost. |

**Recommendation**: **Option C** is most coherent with disteval's existing theory. It recursively applies `right_tail_analysis()` at each level of the decomposition, letting the SOLID/RECOVERABLE/STUCK taxonomy propagate down the hierarchy. Sub-tasks that are STUCK at level `L` become the new exploration targets; sub-tasks that are RECOVERABLE at level `L` get DPO pairs generated.

---

### Q4 — How do we decompose a STUCK task into sub-tasks?

STUCK tasks (`Q*(T) = 0`; never solved) cannot be addressed by consistency training alone. The RMDP framework suggests treating them as composite MDPs where the agent *does* have the individual component skills but lacks the ability to compose them. The `SelfEngine` docstring (SKILL.md, line 15) explicitly says: "decompose STUCK tasks into sub-tasks the agent can solve."

**Why it matters**: This is the novel capability the `RecursionEngine` must add. Without a principled decomposition strategy, we cannot generate sub-task RMDPs or sub-task training environments.

**Proposed answers**

| Option | Decomposition strategy | Trade-offs |
|--------|----------------------|------------|
| **A. Trajectory-based decomposition** | Use `TrajectoryMemory.retrieve()` (`trajectory_memory.py`, line 214) to find the highest-scoring trajectories on *similar* tasks. Identify tool-sequence segments that correspond to solved sub-problems (write file, test execution, etc.). Define entry/exit at the boundaries of those segments. New sub-tasks are created by truncating the original task instruction to correspond to each segment's scope. | Pro: no new data collection; uses existing memory store. Con: sub-task boundaries depend on similarity of past tasks, which may be poor for genuinely novel STUCK tasks. |
| **B. Divergence-point cleavage** | For STUCK tasks with *partial* credit (score > 0 but < Q*_threshold), use `TrajectoryMonitor.check()` at each step to find the first step where `p_high` drops from ≥ 0.5 to < 0.5. Split the task at that step: sub-task 1 covers steps 1..divergence; sub-task 2 covers steps divergence+1..end. | Pro: directly grounded in the monitor's predictive signal; the divergence point is already computed in `_find_divergence_step()`. Con: only works when *some* score exists; pure STUCK (score=0) tasks have no divergence signal. |
| **C. LLM-guided decomposition with validation** | Call an LLM to propose a decomposition of the task instruction into 2–4 sub-goals. Each sub-goal becomes a new task with: (a) an entry condition = the state after the previous sub-goal completed, (b) an exit condition = a testable criterion for sub-goal completion, (c) an initial score = the result of running the sub-task independently. | Pro: works even for pure STUCK tasks; can generate novel sub-tasks not seen in training data. Con: introduces an LLM call outside disteval's pure-eval pipeline; sub-task test criteria require manual specification or harness access; hardest to keep "no external labels." |

**Recommendation**: Use **Option B** for tasks with partial credit (0 < score < Q*_threshold) and **Option A** as a fallback for pure STUCK (score = 0). Reserve **Option C** for Phase 4 when the full environment generation pipeline is in place.

---

### Q5 — How do we handle recursion depth and termination safety?

The paper (§3, "proper RMDPs") requires the RMDP to be *proper*—every execution eventually terminates. Without this guarantee, recursive decomposition can loop indefinitely or produce sub-tasks that are themselves STUCK, triggering infinite recursion.

**Why it matters**: disteval's evaluation loop is designed to be safe and deterministic. A `RecursionEngine` that recurses without bound would make the `SelfImprovementPlan` non-reproducible and could hang the improvement cycle.

**Proposed answers**

| Option | Termination strategy | Trade-offs |
|--------|---------------------|------------|
| **A. Hard depth cap** | `MAX_RECURSION_DEPTH = 3` (configurable). Any sub-task at depth `d == MAX_RECURSION_DEPTH` is classified as STUCK at that level and not further decomposed. | Pro: trivially safe; predictable runtime. Con: may miss improvements if the right decomposition requires depth 4+; arbitrary cap. |
| **B. Monotone difficulty check** | Before decomposing a sub-task further, verify that the sub-task has *strictly lower* structural complexity (fewer required tool calls, shorter trajectory length) than its parent. If not, decline decomposition. | Pro: theoretically sound; guarantees convergence if difficulty is well-ordered. Con: requires a difficulty metric that may not be well-calibrated for LLM agent tasks. |
| **C. Budget-based termination** | Assign a "recursion budget" `B` (e.g., max total sub-tasks = 2 × n_stuck). Each decomposition step consumes budget. When budget is exhausted, remaining STUCK sub-tasks are flagged for human review. | Pro: flexible; integrates with the existing `priority_score` ranking. Con: budget parameter is another hyperparameter; does not prevent individual chains from being too deep. |

**Recommendation**: Combine **Option A** (hard depth cap = 3) with **Option B** (monotone check). Depth cap prevents worst-case runaway; monotone check prevents trivially non-terminating decompositions. Log all termination decisions in the `SelfImprovementPlan` for auditability.

---

### Q6 — How do we handle multi-cycle identity: which task state carries across cycles?

The `SelfEngine.run_cycle()` method is stateless between cycles except for what is reloaded from `job_dirs` (line 363, `self_engine.py`). If a `RecursionEngine` creates sub-tasks in cycle `n`, those sub-tasks must be *runnable* tasks (with their own job directories, trajectories, and scores) in cycle `n+1`. This requires a persistence layer for sub-tasks that does not currently exist.

**Why it matters**: Without sub-task persistence, the recursive decomposition is re-run from scratch every cycle, losing the Q-value estimates built up from past evaluations.

**Proposed answers**

| Option | Persistence strategy | Trade-offs |
|--------|---------------------|------------|
| **A. Extend the curriculum JSON format** | Add `sub_tasks` field to each `TaskImprovement` in `SelfImprovementPlan.to_dict()` (`self_engine.py`, line 184). Each sub-task entry carries its own mini-curriculum with `q_star`, `q_bar`, `gap`, and trajectory paths. The `SelfEngine.reload()` method reads sub-task entries from the saved JSON on the next cycle. | Pro: zero new files; backward-compatible with existing `CURRICULUM_FORMAT.md`. Con: the curriculum JSON grows large for deep recursion; mixing parent-task and sub-task curriculum in one JSON may confuse downstream consumers. |
| **B. Separate sub-task registry file** | Write a `sub_task_registry.jsonl` file (one line per sub-task) that records: `{parent_task, sub_task_id, depth, entry_condition, exit_condition, scores: [...], trajectory_paths: [...]}`. The `RecursionEngine` loads and updates this file on each cycle. | Pro: clean separation; sub-task data is independently queryable. Con: new file format to maintain; requires a loader/writer in `recursion_engine.py`. |
| **C. In-memory graph with cycle checkpointing** | During a cycle, the `RecursionEngine` builds a live directed acyclic graph of tasks and sub-tasks. At end of cycle, serialize the full graph to a checkpoint JSON. On reload, the graph is restored from checkpoint. | Pro: most expressive; supports multi-parent sub-tasks (shared sub-task referenced by multiple parent tasks). Con: most complex to implement; graph serialization/deserialization is non-trivial. |

**Recommendation**: **Option A** in Phase 2 (minimal change to existing serialization), **Option B** in Phase 3 when sub-task management becomes the primary concern.

---

### Q7 — How do we generate training pairs for sub-tasks that have never been run?

The current `SelfEngine._build_training_pairs()` (lines 505–551) requires at least one high-scoring and one low-scoring trajectory for a task. For a newly decomposed sub-task, no trajectories exist yet—the agent has never run it. This is the bootstrapping problem for recursive self-improvement.

**Why it matters**: The entire DPO curriculum generation pipeline depends on trajectory pairs. Without pairs for sub-tasks, the `RecursionEngine` can identify what needs fixing but cannot produce training data for it.

**Proposed answers**

| Option | Bootstrapping strategy | Trade-offs |
|--------|----------------------|------------|
| **A. Parent-trajectory slicing** | For sub-task `T_i` decomposed from parent `T`, extract the trajectory *slice* corresponding to sub-task steps from the best parent trajectory (the `reinforce_traj_path`). Use this slice as the sub-task's `reinforce` trajectory. Construct a `contrast` trajectory by using the same slice from a low-scoring parent run. | Pro: no new agent runs needed; directly reuses existing trajectory data. Con: sliced trajectories may be invalid (the agent in the sub-task slice relied on context built in earlier steps); requires careful context-injection at slice entry. |
| **B. Prompted sub-task runs** | Output the newly decomposed sub-tasks in a format that the Harbor benchmark runner can execute (reusing `CURRICULUM_FORMAT.md`'s training pair format). The evaluation loop then runs the agent on sub-tasks independently and produces real trajectories. | Pro: produces authentic trajectories; cleanest solution. Con: requires Harbor benchmark runner integration; adds evaluation cost; delays the next cycle. |
| **C. Memory-based synthetic pairs** | Use `TrajectoryMemory.retrieve()` (`trajectory_memory.py`, line 214) to find the most similar past trajectory. Use the retrieved trajectory as a synthetic `reinforce` pair and pair it with the current STUCK trajectory as the `contrast`. | Pro: immediate; uses existing memory store; no new runs needed. Con: the retrieved trajectory is from a *different* task; the structural similarity may be low; adds noise to the DPO curriculum. |

**Recommendation**: **Option B** is the cleanest but requires Phase 4 integration with the Harbor runner. In Phase 2–3, use **Option A** (parent-trajectory slicing) with a validity check (ensure the slice starts from a well-defined tool call boundary), falling back to **Option C** when slices are too short or context-dependent.

---

## 2. Ranked List of Highest-Leverage Changes to disteval

The following changes are ranked by (impact × feasibility), from highest to lowest. All changes should be backward-compatible with the existing codebase.

### Rank 1 — Expose `structural_divergence_step` as a first-class entry/exit signal

**Files**: `disteval/trajectory_monitor.py`, `disteval/self_engine.py`  
**Current state**: `_find_divergence_step()` (lines 553–575, `self_engine.py`) returns a single integer but does not expose *which structural phase* the step belongs to, nor the feature vector at that step.  
**Change**: Add a method `TrajectoryMonitor.find_phase_boundaries(traj_path: str) -> list[PhaseBoundary]` where `PhaseBoundary` is a dataclass containing `{step_index, tool_name, p_high_before, p_high_after, phase_tag}`. This gives `RecursionEngine` the raw material to define entry/exit conditions.  
**Impact**: Directly enables Q3 (reward propagation), Q4 (decomposition), and Q1 (entry/exit definition) without modifying any core logic.  
**Effort**: ~50 lines in `trajectory_monitor.py`; no dependency changes.

### Rank 2 — Add sub-task taxonomy to `right_tail_analysis()`

**Files**: `disteval/right_tail.py`  
**Current state**: `right_tail_analysis()` (line 207) analyzes a flat `RecordStore`—it does not know about task hierarchy.  
**Change**: Add an optional `parent_task: str | None` parameter and a `sub_task_level: int = 0` parameter. When `sub_task_level > 0`, the analysis treats tasks as sub-tasks and returns a `RightTailReport` with a `parent_q_contribution: float` field (the sub-task's estimated contribution to parent task score).  
**Impact**: Enables Q3 (recursive reward propagation) by making `right_tail_analysis()` hierarchy-aware. The consistency index `κ` at the sub-task level directly maps to the RMDP's sub-MDP Q-value.  
**Effort**: ~30 lines; backward-compatible (default values preserve existing behavior).

### Rank 3 — Add `sub_tasks` field to `TaskImprovement` and `SelfImprovementPlan`

**Files**: `disteval/self_engine.py`  
**Current state**: `TaskImprovement` (line 82) has no field for sub-tasks. `SelfImprovementPlan` (line 104) has no recursive structure.  
**Change**: Add `sub_tasks: list[TaskImprovement] = field(default_factory=list)` to `TaskImprovement`, and add `recursion_depth: int = 0`. Update `to_dict()` recursively. This creates the data container that `RecursionEngine` will fill.  
**Impact**: Enables Q6 (cross-cycle persistence) and makes the curriculum JSON a complete recursive structure. Downstream consumers (DPO trainers, Phase 3 environment generators) can traverse the tree.  
**Effort**: ~15 lines; non-breaking addition.

### Rank 4 — Expose `TrajectoryMemory` retrieval by structural depth

**Files**: `disteval/trajectory_memory.py`  
**Current state**: `retrieve()` (line 214) matches by tool-sequence similarity and task-description keyword overlap. No notion of "depth" in the task hierarchy.  
**Change**: Add `depth_filter: int | None = None` parameter. When provided, only return memories from tasks at the same decomposition depth. Add a `depth: int = 0` field to `TrajectoryRecord` (`trajectory_memory.py`, line 37). Populate this when sub-tasks are added via `RecursionEngine`.  
**Impact**: Enables Q7 (training pair bootstrapping via memory) and aligns with the SKILL.md mapping: "call-stack value backup ↔ memory retrieval at matching structural depth."  
**Effort**: ~20 lines; backward-compatible.

### Rank 5 — Add a `RecursionContext` to `TrainingPair`

**Files**: `disteval/self_engine.py`  
**Current state**: `TrainingPair` (line 70) records `structural_divergence_step` but not which parent task it belongs to, what depth it's at, or what the entry/exit conditions are.  
**Change**: Add optional fields `parent_task: str | None = None`, `sub_task_depth: int = 0`, `entry_step: int = 0`, `exit_step: int = -1` to `TrainingPair`. These fields are populated by `RecursionEngine` and can be used by Phase 3's environment generator to define state/action boundaries.  
**Impact**: Enables Phase 3's RL environment schema (SKILL.md, Phase 3 deliverable) by embedding recursion context directly in training pairs.  
**Effort**: ~10 lines; additive-only change.

### Rank 6 — Add `simulate_recursive_training_gains()` to `training_sim.py`

**Files**: `disteval/training_sim.py`  
**Current state**: `apply_training_effect()` (line 199) models improvement at the flat task level. It does not propagate sub-task improvement to parent task scores.  
**Change**: Add `simulate_recursive_training_gains(task_graph: dict, ...)` that takes a DAG of tasks and sub-tasks and propagates improvement bottom-up using the aggregation rule from Q3.  
**Impact**: Enables Phase 4's end-to-end simulation (SKILL.md Phase 4) and validates that recursive training gains are actually larger than flat DPO gains for STUCK tasks.  
**Effort**: ~80 lines; requires the `TaskImprovement.sub_tasks` field (Rank 3) to exist first.

---

## 3. Note on Existing Libraries and Papers

### Directly Applicable

**ArCHer (arXiv:2402.19446, ICML 2024)** — "Training Language Model Agents via Hierarchical Multi-Turn RL"  
- Proposes exactly the two-level hierarchy needed: a high-level value function aggregates reward over utterances; a low-level token policy is trained using this value function as a reward signal.  
- The high-level value function in ArCHer plays the role of the sub-MDP Q-value in the RMDP framework.  
- **Reusability**: The ArCHer codebase (`github.com/YifeiZhou02/ArCHer`) is open-source but requires PyTorch and HuggingFace Transformers—outside disteval's numpy/pandas/scipy dependency scope. However, the *credit assignment algorithm* (high-level TD learning over utterances) can be re-implemented in numpy for disteval's tabular setting. Key adaptation: replace "utterances" with "trajectory phases" as defined by `TrajectoryMonitor.find_phase_boundaries()`.

**ADAPT (ACL Findings 2024)** — "As-Needed Decomposition and Planning with Language Models"  
- Proposes recursive decomposition *triggered by failure*: the executor detects a failing sub-task and the planner further decomposes it.  
- **Reusability**: The failure-triggered decomposition logic maps cleanly onto disteval's STUCK/RECOVERABLE taxonomy. When a sub-task is STUCK at depth `d`, it triggers further decomposition—exactly ADAPT's "decompose when executor fails" rule. The key difference: ADAPT uses an LLM as both planner and executor at runtime; disteval's `RecursionEngine` would apply this offline, from eval data, without LLM calls.

**HiPER (arXiv:2602.16165)** — "Hierarchical Plan-Execute RL with Explicit Credit Assignment"  
- Introduces "hierarchical advantage estimation" (HAE): credit is assigned separately at the planning level (which subgoal to pursue) and execution level (how to execute a tool call).  
- **Reusability**: HAE's per-segment reward aggregation is directly applicable to disteval's trajectory phases. The segment boundaries are defined by `structural_divergence_step`; the segment-level reward is `score(T_i)` for sub-task `T_i`. HAE can be approximated without TD learning using the existing bootstrap simulation in `training_sim.py`.

### Partially Applicable

**Options Framework (Sutton, Precup, Singh 1999)** — "Between MDPs and Semi-MDPs"  
- The call-and-return model of option execution is formally equivalent to the RMDP's box invocation. Intra-option Q-learning (Sutton, Precup, Singh 1998) enables learning about option value before the option terminates.  
- **Reusability**: The options framework's termination condition `β: S → [0,1]` maps to disteval's binary exit condition (did the agent achieve sub-task exit criterion?). The initiation set `I_ω ⊆ S` maps to the entry condition. The existing `OutcomePredictor.predict_proba()` (`trajectory_monitor.py`, line 318) can serve as a soft version of the option's *initiation set indicator*.

**CRAAM (github.com/marekpetrik/CRAAM)** — C++ RMDP solver  
- A header-only C++ library for solving MDPs/RMDPs with robustness to transition uncertainty.  
- **Reusability**: Not directly usable (C++ only; disteval is Python). The RMDP data structure definitions and value iteration algorithms are useful as reference implementations for the tabular case.

### Inspirational but Non-Reusable

**Gödel Agent (arXiv:2410.04444)** — recursive self-improvement via monkey patching  
- Implements truly recursive self-improvement by having the agent modify its own code at runtime.  
- Not applicable to disteval's *evaluation-driven* (offline) approach; however, the concept of using a recursive call structure to represent the self-improvement loop is aligned with the RMDP framing.

**STEP-HRL (arXiv:2604.05808)** — step-level hierarchical RL for LLM agents  
- Addresses "context explosion" in long-horizon LLM trajectories by conditioning the low-level policy on only single-step transitions (not full history).  
- **Partial applicability**: The "local progress module" (compact summary of sub-task progress) is analogous to the RMDP's exit value abstraction. disteval could use the existing `TrajectoryMemory._make_summary()` (`trajectory_memory.py`, line 511) as a lightweight version.

---

## 4. Open Questions for Phase 2

The following questions cannot be answered from the existing codebase and literature alone. They should be resolved in Phase 2's `RecursionEngine` design document.

### OQ1 — What is the minimum viable decomposition depth?

The SKILL.md Phase 2 deliverable asks for a concrete example using `medium-2` or `hard-1`. These tasks are 1–3 levels decomposable based on their trajectory structure. But the paper's convergence proofs (§4) assume finite, proper RMDPs. Empirically: what is the maximum depth at which disteval's tasks are decomposable without the sub-tasks becoming trivially simple or requiring context that is unavailable at sub-task entry?

### OQ2 — Can `structural_divergence_step` be used bidirectionally?

Currently, `_find_divergence_step()` (`self_engine.py`, line 553) finds the *first* step where a high trajectory diverges from a low trajectory. Can the same signal be used to find the *last* step of a sub-task (i.e., the exit condition), not just the entry? Or does finding exits require a different signal (e.g., the first step where the high trajectory *re-converges* with the expected completion pattern)?

### OQ3 — How are sub-task entry conditions specified for independent execution?

If a sub-task is defined as "steps 5–12 of the parent trajectory," the agent at entry needs the *state* after step 4. For a coding task, this means: the file system state, the terminal history, and any partial outputs from steps 1–4. How is this state captured, serialized, and provided as the initial observation for an independent sub-task run? Does the Harbor runner support stateful task initialization?

### OQ4 — What is the correct way to detect when sub-task decomposition is *not* useful?

If a STUCK task is STUCK because the agent lacks a fundamental capability (not because it has the capability but fails to compose it), recursive decomposition will generate sub-tasks that are also STUCK, and the recursion will only terminate at the depth cap. How do we detect this situation early (before wasting evaluation cycles) and redirect to the "capability expansion" path?

### OQ5 — How does multi-agent disteval interact with RMDP recursion?

The current `SelfEngine` is per-agent (`agent_name` field, line 254). When multiple agents (Claude, Gemini, Codex) are evaluated, their `SelfImprovementPlan`s are independent. If a `RecursionEngine` decomposes `hard-1` for Codex, should the sub-tasks also be evaluated for Claude (who is SOLID on `hard-1`)? If so, do Claude's successful sub-task trajectories serve as the `reinforce` targets for Codex's sub-task DPO curriculum? This cross-agent sub-task sharing is not addressed by the RMDP framework directly but may be the highest-leverage path for STUCK task improvement.

### OQ6 — Is the 1-exit assumption appropriate for disteval tasks?

The paper proves convergence for 1-exit RMDPs (§4, Theorem 5). Most coding tasks have a single "done" state (test passes / test fails), making them 1-exit. But some tasks may have multiple valid completion paths (e.g., `medium-2` can be solved with different HTTP client libraries). In the multi-exit case, Recursive Q-learning converges only under the determinism assumption (Theorem 4). Are disteval's tasks deterministic (same input → same output) or stochastic (different approaches → different intermediate scores)? The answer determines which convergence guarantee applies.

### OQ7 — Should sub-task priority scoring use the same `gap × (1 - consistency)` formula?

The current priority score in `SelfEngine._build_task_improvement()` (`self_engine.py`, line 451) uses `gap × (1 - consistency)`. For sub-tasks, the appropriate priority metric might be different: a sub-task that blocks multiple parent tasks should receive higher priority than one that blocks only a single parent. How should parent-task "blocking weight" be incorporated into the sub-task priority score?

---

## 5. Summary Table

| Question | Recommended Answer | Key disteval Change |
|----------|--------------------|---------------------|
| Q1: Entry/exit definition | Tool-call boundaries at divergence step | `TrajectoryMonitor.find_phase_boundaries()` |
| Q2: Call stack representation | Flat list of (parent, sub_task, expected_return) | New field in `SelfImprovementPlan` |
| Q3: Reward propagation | Recursive right-tail analysis | Hierarchy-aware `right_tail_analysis()` |
| Q4: Sub-task decomposition | Divergence-point cleavage + memory retrieval | `RecursionEngine.decompose_stuck()` |
| Q5: Recursion termination | Hard depth cap (3) + monotone difficulty | `RecursionEngine.MAX_DEPTH = 3` |
| Q6: Cross-cycle persistence | Extend curriculum JSON with `sub_tasks` | `TaskImprovement.sub_tasks` field |
| Q7: Sub-task training pairs | Parent-trajectory slicing → prompted runs | `RecursionEngine.slice_parent_trajectory()` |

---

*This document is Phase 1C of the recursive self-improvement research task. Phase 2 should consume this document and the existing codebase to produce a concrete `RecursionEngine` design with API signatures and a worked example using `medium-2` or `hard-1`.*
