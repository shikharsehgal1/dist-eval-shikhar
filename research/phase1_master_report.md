# Phase 1 Master Report: Recursive Self-Improvement for disteval

**Date:** 2026-06-22
**Input:** arXiv:2206.11430 (Hahn et al., NeurIPS 2022) + disteval codebase
**Output:** Phase 1A, 1B, 1C reports
**Next step:** Phase 2 — design a concrete `RecursionEngine` for disteval

---

## 1. The core opportunity

disteval already measures the full outcome distribution of agentic tasks, classifies tasks as SOLID / RECOVERABLE / STUCK, and generates DPO training pairs from RECOVERABLE gaps. The `SelfEngine` docstring (`disteval/self_engine.py` lines 31–49) explicitly frames this as an RMDP-style decomposition, but the formal RMDP machinery is not implemented.

The opportunity is to build a **RecursionEngine** that:

1. Decomposes STUCK tasks into sub-task RMDPs with explicit entry/exit points.
2. Propagates sub-task scores back to parent tasks using disteval's existing right-tail metrics.
3. Generates a recursive curriculum where closing a sub-task gap lifts the parent task score.
4. Persists sub-task definitions across cycles so the first cycle's decomposition shapes the next cycle's evaluation.

This directly addresses the user request: a recursion engine that produces RL environments that self-improve based on distributed agent evals, where the first step of a task affects the next based on the solution.

---

## 2. Key findings from arXiv:2206.11430

A Recursive MDP (RMDP) is a finite collection of component MDPs that can recursively invoke one another. Each component has entry/exit nodes and boxes (recursive calls). The semantics is an infinite-state MDP whose state is `(call_stack, current_vertex)`.

Convergence guarantees:

- **General multi-exit RMDPs:** undecidable.
- **1-exit RMDPs:** Recursive Q-learning converges to optimal values under properness and sufficient exploration.
- **Deterministic proper multi-exit:** Recursive Q-learning converges with learning rate 1.

The key algorithmic idea is **exit value abstraction**: the infinite call-stack is abstracted by the vector of expected rewards at each exit of the current component.

See `research/phase1a_rmdp_formalism.md` for full formal definitions and convergence theorems.

---

## 3. disteval ↔ RMDP mapping (validated)

| RMDP concept | disteval equivalent | Where |
|---|---|---|
| RMDP / component MDP | Benchmark task / sub-task | `right_tail.py` `TaskOutcomeProfile` |
| Entry point | Task instruction + initial state | `trajectory_monitor.py` `TrajectoryRecord.task_path` |
| Exit point | `test.sh` reward scalar | `tasks/medium-2/tests/test.sh` |
| Box / recursive call | Sub-task invocation (tool call) | `trajectory_monitor.py` `n_exec` |
| Call stack | Tool sequence prefix | `trajectory_monitor.py` `extract_tool_sequence` |
| Exit value `V(ex)` | Right-tail peak `Q*(t)` | `right_tail.py` L26 |
| Sub-MDP Q-value | Consistency `κ(t) = Q̄(t)/Q*(t)` | `right_tail.py` L30–33 |
| Convergence to `V*` | `κ → 1.0` for RECOVERABLE tasks | `self_engine.py` L45–49 |
| Call-stack value backup | `TrajectoryMemory` retrieval | `trajectory_memory.py` L290–305 |
| Structural divergence | `structural_divergence_step` | `self_engine.py` L553–575 |

See `research/phase1b_disteval_mapping.md` for the full 14-row mapping and a worked example decomposing `medium-2` (REST client) into 5 chained 1-exit sub-RMDPs.

---

## 4. Critical design decisions (recommended answers)

From `research/phase1c_integration_questions.md`:

| Question | Recommended answer | Key change |
|---|---|---|
| Q1: Entry/exit definition | Tool-call boundaries at the structural divergence step | `TrajectoryMonitor.find_phase_boundaries()` |
| Q2: Call stack representation | Flat list of `(parent_task, sub_task, expected_return)` tuples | New field in `SelfImprovementPlan` |
| Q3: Reward propagation | Recursive right-tail analysis on sub-task scores | Hierarchy-aware `right_tail_analysis()` |
| Q4: Decomposition | Divergence-point cleavage for partial-credit tasks; memory retrieval for pure STUCK | `RecursionEngine.decompose_stuck()` |
| Q5: Termination safety | Hard depth cap (default 3) + monotone difficulty check | `RecursionEngine.MAX_DEPTH = 3` |
| Q6: Cross-cycle persistence | Extend curriculum JSON with `sub_tasks` | `TaskImprovement.sub_tasks` field |
| Q7: Bootstrapping sub-task pairs | Parent-trajectory slicing (Phase 2–3); prompted sub-task runs (Phase 4) | `RecursionEngine.slice_parent_trajectory()` |

Highest-leverage changes, ranked:

1. Expose `structural_divergence_step` as a first-class entry/exit signal (`trajectory_monitor.py`).
2. Add sub-task taxonomy to `right_tail_analysis()` (`right_tail.py`).
3. Add `sub_tasks` to `TaskImprovement` / `SelfImprovementPlan` (`self_engine.py`).
4. Expose `TrajectoryMemory` retrieval by structural depth (`trajectory_memory.py`).
5. Add recursion context to `TrainingPair` (`self_engine.py`).
6. Add recursive training simulation to `training_sim.py`.

---

## 5. Concrete worked example: `medium-2` (REST client)

The task has a 5-checkpoint scoring structure. The test suite can be decomposed into a chain of 1-exit sub-RMDPs:

```
M_2 = M_2a ∘ M_2b ∘ M_2c ∘ M_2d ∘ M_2e
```

- `M_2a`: HTTP client runs without error → 0.10 reward
- `M_2b`: Correct filter (`total_eligible_users == 7`) → +0.25 reward
- `M_2c`–`M_2e`: Correct per-department groupby → +0.20 reward each

Under current disteval, Codex CLI's `[0.0, 0.0, 1.0]` profile is RECOVERABLE with a whole-trajectory DPO pair. Under RMDP decomposition, the engine could identify that `M_2a` and `M_2b` are SOLID on attempt 1 (score 0.35), while `M_2c`–`M_2e` are RECOVERABLE/STUCK. The curriculum would then train only on the groupby phase, not the HTTP client phase.

This is the central value proposition: **more targeted training signals from the same eval data.**

---

## 6. Risks and constraints for Phase 2

1. **Sub-task boundary detection:** The divergence step is a symptom, not a guaranteed semantic boundary. Need to validate against test-suite checkpoints or accept some fuzziness.
2. **Recursion depth:** Must cap depth and enforce monotone difficulty to prevent infinite decomposition of STUCK tasks.
3. **Multi-exit vs 1-exit:** Convergence guarantees apply cleanly to 1-exit sub-tasks. Decompose multi-checkpoint tasks into chains of 1-exit sub-RMDPs.
4. **Reward credit assignment:** Parent scores must be a function of sub-task scores. Recommended: weighted sum or right-tail sub-task Q-values.
5. **DPO pair validity:** Sliced parent trajectories may lack context. Need `entry_step` / `exit_step` fields in `TrainingPair`.
6. **Cross-agent sharing:** If one agent is SOLID on a sub-task, its trajectories could serve as reinforce targets for another agent. This is not yet addressed.
7. **No new runtime dependencies:** Keep implementation within Python 3.10 + numpy/pandas/scipy/matplotlib.

---

## 7. Phase 2 deliverable target

Phase 2 should produce `research/phase2_recursion_engine.md` containing:

- A `RecursionEngine` class/module with API signatures.
- How it decomposes STUCK tasks using `TrajectoryMonitor` and `TrajectoryMemory`.
- How it defines sub-task entry/exit conditions and call-stack representation.
- How it integrates with `SelfEngine` and `right_tail_analysis`.
- A worked example using `medium-2` or `hard-1`.
- A list of open questions for Phase 3 (RL environment generation).

The design must be implementable as a backward-compatible extension to the existing disteval codebase.
