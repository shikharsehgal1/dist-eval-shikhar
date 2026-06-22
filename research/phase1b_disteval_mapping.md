# Phase 1B — disteval ↔ RMDP Detailed Mapping Report

**Phase:** 1B (follows Phase 1 literature review)
**Date:** 2025
**Source paper:** arXiv:2206.11430 — "Recursive Reinforcement Learning" (Hahn et al., NeurIPS 2022)
**Codebase snapshot:** `/Users/shikharsehgal/rl-dist-eval/` (disteval)

---

## 1. Summary of arXiv:2206.11430 Formal Objects

The paper defines a **Recursive MDP (RMDP)** as a finite collection of _component MDPs_
M₁, M₂, …, Mₖ. Each component Mᵢ has:

| Formal object | Definition |
|---|---|
| **Component MDP (Mᵢ)** | A standard MDP with designated entry and exit nodes. |
| **Entry node (en_i)** | The unique start state when component Mᵢ is "called". |
| **Exit node(s) (Ex_i)** | Terminal states of Mᵢ that return control to the caller. |
| **Box (b)** | A state inside Mᵢ that invokes another component as a sub-routine (recursive call). |
| **Call-stack (κ)** | The stack of pending (component, box) pairs waiting for control to return. |
| **Exit value V(ex)** | Expected total reward from reaching exit ex in component Mᵢ under the current call-stack context. |
| **Recursive Q-value Q(q, a; κ, b)** | Q-value of taking action a in state q under stack context ⟨κ, b⟩. |
| **OPTrecur / OPTcont** | Optimality equations for the full recursive structure; the "cont" variant abstracts stacks to their exit values. |
| **1-exit RMDP** | Special case where every component has exactly one exit; Recursive Q-learning converges here (Theorem 4). |
| **Proper RMDP** | Every execution almost surely reaches an exit; ensures well-defined Q-values. |
| **Recursive Q-learning** | Model-free algorithm that maintains a Q-table per component and updates using Bellman-style backups across recursive boundaries. Proven to converge for finite, single-exit, and deterministic multi-exit RMDPs. |

The key convergence insight (Theorem 4 of the paper): for proper 1-exit RMDPs, Recursive Q-learning
converges to the optimal strategy when all state-action pairs are visited infinitely often. The proof
relies on the exit value abstraction that collapses the countably-infinite call-stack to a finite set
of "entry × exit value" pairs.

---

## 2. RMDP Concept → disteval Equivalent Mapping Table

The mapping below is grounded in file paths and line numbers wherever possible.

| RMDP Concept (arXiv:2206.11430) | disteval Equivalent | File(s) & Lines |
|---|---|---|
| **Recursive MDP (RMDP)** — a collection of component MDPs that can invoke one another | The full 6-task evaluation suite, where each task is a component environment. Implicitly referenced as the "RECOVERABLE taxonomy (nested gap localization)" hierarchy. | `self_engine.py` L38-44 (RMDP hierarchy docstring) |
| **Component MDP (Mᵢ)** — one callable sub-routine with its own state/action space | One disteval task (e.g., `medium-rest-client`, `hard-bugfix`). Represented as a `TaskOutcomeProfile` holding all attempt scores for that component. | `right_tail.py` L116-132 (`TaskOutcomeProfile`) |
| **Entry point (en_i)** — the state where execution of Mᵢ begins upon being called | The task instruction + initial environment state at episode start. The first step of a `trajectory.json` before any tool calls. In disteval, it is implicitly the `task_path` + container state at time 0. | `self_engine.py` L42 ("Entry/exit points"); `trajectory_monitor.py` L63-73 (`TrajectoryRecord.task_path`) |
| **Exit point (ex ∈ Ex_i)** — state where Mᵢ terminates and returns a reward | Test-suite pass/fail verdict emitted by `test.sh`; the scalar score written to `/logs/verifier/reward.txt`. A single test suite = 1-exit RMDP per task. | `tasks/medium-2/tests/test.sh` L66; `tasks/hard-1/tests/test.sh` |
| **Recursive call (box b)** — a state in Mᵢ that invokes component Mⱼ | Not yet first-class in disteval. Analogous to the agent deciding to call a sub-tool (e.g., run a helper script, make an HTTP call to a sub-service). Implicitly captured by `n_exec` in `TrajectoryFeatures`. | `trajectory_monitor.py` L53 (`n_exec`); `trajectory_memory.py` L48 (`n_exec`) |
| **Call-stack (κ)** — stack of pending (component, box) pairs | The ordered sequence of tool calls in a trajectory up to the current step. The "structural signature" the monitor tracks. Also captured as `memory retrieval at matching structural depth` in the docstring. | `self_engine.py` L43 ("Call-stack value backup"); `trajectory_monitor.py` L139-147 (`extract_tool_sequence`) |
| **Exit value V(ex)** — expected total reward from an exit node under stack context | The per-task right-tail metric Q*(t): the maximum score the agent has ever achieved on task t. It is the demonstrated "value at exit" — what the agent gets when it does reach the exit correctly. | `right_tail.py` L26 (`Q*(t) = max_i q_i`), L122 (`q_star`) |
| **Sub-MDP Q-value Q(q,a; κ,b)** | Per-task consistency score κ(t) = Q̄(t)/Q*(t). This is the ratio of actual mean return to the right-tail (maximum) return — directly analogous to how far the current policy is from the optimal value at the exit. | `right_tail.py` L30-33 (κ definition), L124 (`consistency`); `self_engine.py` L41 ("Sub-MDP Q-value") |
| **Right-tail gap Δ(t) = Q*(t) − Q̄(t)** | Same symbol in disteval. Gap between demonstrated best and current mean. The Q-value "shortfall" across the component boundary. | `right_tail.py` L26-28, L123 (`gap`); `THEORY.md` L29 |
| **Convergence toward V*** — Recursive Q-learning converges to optimal Q-values | As RECOVERABLE gaps close (κ → 1.0 task by task), the priority queue empties and all tasks graduate to SOLID. The SelfEngine explicitly calls this the convergence analog. | `self_engine.py` L45-49; `right_tail.py` L39-43 (SOLID definition) |
| **Proper RMDP** — every execution almost surely reaches an exit | Disteval tasks with `verifier.timeout_sec` ensure every episode terminates with a reward. Tasks that *always* time out and score 0 are the "STUCK" category — an improper component. | `tasks/medium-2/task.toml` L14; `right_tail.py` L41 ("STUCK") |
| **1-exit RMDP** — each component has exactly one exit state | Each disteval task has exactly one terminal reward value (the `reward.txt` scalar). This is structurally a 1-exit component — the convergence guarantee applies. | `tasks/medium-2/tests/test.sh` L66 (single reward write) |
| **Optimality equations OPTcont(M)** — abstracted Bellman equations using exit values | `training_sim.py` training-effect model: `improvement = α × DPO_BONUS × q_star × (1 − current_mean)`. This is a one-step Bellman update toward q_star (the exit value) scaled by a behavioral cloning rate α. | `training_sim.py` L267-271 (DPO formula); L46-52 (constants) |
| **Recursive Q-learning update** — backprop of Q-values across component boundaries | The multi-cycle SelfEngine loop: each `run_cycle()` observes outcome distribution, updates the priority queue, applies one round of DPO, and re-evaluates. The loop IS the Recursive Q-learning outer loop, approximated by behavioral cloning. | `self_engine.py` L375-435 (`run_cycle()`); L244-249 (multi-cycle example) |
| **Call-stack value backup** — propagating value from a sub-MDP exit back to the calling state | `trajectory_memory.py` retrieval at matching structural depth: when a new task is started, the memory returns the highest-scoring past trajectory from a structurally similar depth. | `self_engine.py` L43 (docstring); `trajectory_memory.py` L290-305 (`retrieve_for_new_task`) |
| **Structural divergence** — the step where the call-stack context causes two policies to differ | `structural_divergence_step`: the first tool-call index where the high-scoring trajectory is predicted HIGH and the low-scoring trajectory is predicted LOW by the monitor. This is the "fork" in the RMDP tree. | `self_engine.py` L553-575 (`_find_divergence_step`); `CURRICULUM_FORMAT.md` L120-121 |

---

## 3. How SelfEngine Already References RMDP Concepts

The module docstring of `self_engine.py` at lines 31–50 contains an explicit mapping table:

```python
# self_engine.py, lines 37–44
RMDP concept              SelfEngine equivalent
─────────────────────     ─────────────────────────────────────────────────
Recursive MDP hierarchy   RECOVERABLE taxonomy (nested gap localization)
Sub-MDP Q-value           per-task consistency score κ(t)
Entry/exit points         trajectory step where monitor prediction diverges
Convergence toward V*     κ → 1.0 as RECOVERABLE gaps close across cycles
Call-stack value backup   memory retrieval at matching structural depth
```

This mapping is accurate but *metaphorical*: the SelfEngine uses RMDP intuition without
implementing any of the formal RMDP machinery. Here is how each line maps to actual code:

### 3.1 "Recursive MDP hierarchy ↔ RECOVERABLE taxonomy"

The RECOVERABLE classification in `right_tail.py` (lines 186–189) is the disteval equivalent of
identifying which component MDPs still have a gap between their current policy and their optimal
exit value. Tasks classified RECOVERABLE are exactly those component MDPs where the agent has
*demonstrated* the capability to reach the exit optimally (Q*(t) > 0) but does not do so
consistently (Δ(t) > 0). This is the structural criterion that in an RMDP would determine which
sub-routine to re-optimize.

The "nested gap localization" phrase alludes to the fact that closing the gap on a harder task
(e.g., `hard-1`) may expose gaps in sub-steps that were previously masked — the recursion unfolds
naturally as training cycles proceed.

### 3.2 "Sub-MDP Q-value ↔ κ(t)"

κ(t) = Q̄(t)/Q*(t) measures how far the agent's mean policy value is from its optimal value at the
exit of component t. In RMDP terms, Q*(t) is the optimal exit value V*(en_t) achievable from the
entry point, and Q̄(t) is the value achieved by the current policy. The ratio is a normalized
shortfall in Q-value.

Implemented at `right_tail.py` lines 187–189:
```python
consistency = q_bar / q_star if q_star > 0 else 0.0
```

And exposed in `TaskOutcomeProfile.consistency` (line 124) and used as a priority signal in
`self_engine.py` line 451:
```python
priority_score = profile.gap * (1.0 - profile.consistency)
```

### 3.3 "Entry/exit points ↔ trajectory step where monitor prediction diverges"

This is the most precise RMDP analogy in the current code. The `TrajectoryMonitor.check()` method
(`trajectory_monitor.py` lines 488–524) computes a binary prediction (HIGH / LOW / uncertain) at
each step prefix. The `_find_divergence_step` in `self_engine.py` (lines 553–575) finds the
smallest step index where:
- The high-scoring trajectory is predicted HIGH
- The low-scoring trajectory is predicted LOW or uncertain

This divergence step corresponds to the "box" node in the RMDP at which the agent invokes a
sub-routine (makes a structural choice) that determines whether the component will exit successfully.
It is where Q-value differs between the optimal and suboptimal policies.

### 3.4 "Convergence toward V* ↔ κ → 1.0"

The convergence criterion is made explicit in `self_engine.py` lines 45–49:

```python
# The key convergence result: as RECOVERABLE gaps close (κ → 1.0 for each task),
# the engine's priority queue naturally empties — RECOVERABLE tasks graduate to SOLID.
# What remains are STUCK tasks, which require capability expansion, not consistency
# training. The engine correctly identifies when it has exhausted consistency training
# and signals that new exploration is needed.
```

And it manifests in the `SelfImprovementPlan.cycle_complete` flag (`self_engine.py` line 431):
```python
cycle_complete=(report.n_recoverable == 0),
```

The multi-cycle loop in `self_engine.py` lines 244–249 is the annealing procedure — each cycle
moves tasks toward SOLID (κ=1), mimicking the convergence of Recursive Q-learning.

### 3.5 "Call-stack value backup ↔ memory retrieval at matching structural depth"

When `TrajectoryMemory.retrieve_for_new_task()` is called (`trajectory_memory.py` lines 290–305),
it retrieves high-outcome trajectories from similar past tasks, giving preference to
`is_recoverable_high` entries (line 266). These entries represent trajectories where the agent DID
reach the exit optimally — they are the empirical demonstration of V*(en_i). Retrieving them for a
new similar task is equivalent to initializing the Q-table for the new component using the backed-up
value from a previously solved component.

---

## 4. Components That Would Need to Change for Explicit RMDP-Style Recursion

To go from the current *implicit* RMDP analogy to an *explicit* RMDP recursion engine, the
following components would need changes:

### 4.1 `right_tail.py` — Expose sub-task gap decomposition

**Current state:** `TaskOutcomeProfile` (lines 116–132) holds per-task Q*, Q̄, gap, and
consistency. It treats each task as a monolithic atomic component.

**Required change:** Add sub-task decomposition. A STUCK task with Q*(t) = 0 needs to be
decomposed into sub-tasks {t₁, t₂, …} where the agent might have Q*(tᵢ) > 0. The function
`task_outcome_profile()` (lines 163–204) would need to accept a list of sub-task score arrays and
compute a recursive gap: Δ_total = Q*(t) − Σᵢ λᵢ · Q̄(tᵢ), where λᵢ is the sub-task weight.

Specifically, `RightTailReport` (lines 136–158) would need a new field:
```python
sub_task_profiles: dict[str, list[TaskOutcomeProfile]]  # task → sub-task profiles
```

**Why it matters:** Without sub-task profiles, there is no signal for STUCK tasks. RMDP recursion
enables generating sub-tasks whose profiles are RECOVERABLE, providing training signal where none
currently exists.

### 4.2 `trajectory_monitor.py` — Expose divergence step as RMDP entry/exit signal

**Current state:** `TrajectoryMonitor.check()` (lines 488–524) outputs a PatternMatch with a
binary HIGH/LOW prediction and a single `prefix_len`. The `_find_divergence_step` call in
`self_engine.py` uses it post-hoc.

**Required change:** `PatternMatch` (lines 75–87) needs two additions:
1. `entry_step: int` — the step index that defines the entry point of the current sub-task
   (currently implicit as 0 or the previous divergence step).
2. `exit_step: int | None` — the predicted step at which the current sub-task will terminate
   (reach an exit node), estimated from historical completion lengths.

A new method `find_entry_exit_boundaries(steps, min_confidence=0.7)` would segment a trajectory
into a sequence of `(entry_step, exit_step, sub_task_label)` tuples — the RMDP call stack trace
made explicit.

### 4.3 `trajectory_memory.py` — Retrieve sub-task demonstrations

**Current state:** `TrajectoryMemory.retrieve_for_new_task()` (lines 290–305) retrieves whole-task
memories and ranks by cosine similarity of tool-frequency vectors.

**Required change:** The memory needs to support sub-trajectory retrieval:
1. `MemoryEntry` (lines 52–60) needs a `sub_trajectories: list[SubTrajectoryEntry]` field where
   each `SubTrajectoryEntry` captures a bounded slice `[entry_step:exit_step]` with its own score.
2. `TrajectoryMemory.retrieve_for_sub_task(sub_task_description, entry_features, k)` — retrieves
   memories that match not just the whole task but a specific structural phase within it.

This enables the call-stack value backup: when the recursion engine decides to call a sub-RMDP, it
can initialize it with the best demonstration of that structural phase from memory.

### 4.4 `training_sim.py` — Simulate recursive training gains

**Current state:** The training effect model (`apply_training_effect`, lines 199–295) applies a
flat improvement formula per task. It does not model recursion depth or sub-task structure.

**Required change:** The simulation needs a recursive improvement model:
- For a STUCK task with sub-tasks {t₁, t₂}:
  - If t₁ is SOLID and t₂ is RECOVERABLE: improvement comes from closing t₂'s gap
  - If both are STUCK: mark parent as "recursively stuck", trigger expansion
- The `_fast_apply_improvement` function (lines 351–396) would need a depth parameter and a
  propagation rule: closing a sub-task gap updates the parent task's simulated score.

### 4.5 `self_engine.py` — Integrate a RecursionEngine

**Current state:** `SelfEngine.run_cycle()` (lines 375–435) treats each task independently. The
RMDP concepts are in the docstring but not in the logic. STUCK tasks are identified but produce no
training signal (no entries in `curriculum`).

**Required change:** A new `RecursionEngine` class (see SKILL.md Phase 2 deliverable) would sit
alongside SelfEngine and:
1. After `right_tail_analysis`, identify STUCK tasks.
2. Use `trajectory_monitor` to segment the best-available trajectory (even a partial one) at the
   structural divergence step.
3. Produce sub-task definitions `{t_stuck → [t₁, t₂]}` with their own entry/exit conditions.
4. Re-run `right_tail_analysis` on the sub-task score arrays.
5. Return sub-task RMDP structures to the curriculum as a new task kind: `"decomposed"`.

The `SelfImprovementPlan.curriculum` list would gain entries with `kind="decomposed"`, and the
curriculum format (`CURRICULUM_FORMAT.md`) would need a new section for decomposed items.

---

## 5. Concrete Example: medium-2 (REST Client) as a Sub-Task RMDP

### 5.1 Task description

**Task:** `disteval/medium-rest-client` (`tasks/medium-2/`)
**Instruction:** Write a Python script `/app/client.py` that (1) fetches all users from a mock HTTP
API, (2) filters to users aged ≥ 30, (3) groups by department, (4) computes average salary per
department, (5) writes results to `/app/summary.json`.

**Observed performance (from `THEORY.md` and `CURRICULUM_FORMAT.md`):**
- Codex CLI: attempts = [0.0, 0.0, 1.0] → Q* = 1.0, Q̄ = 0.333, Δ = 0.667, κ = 0.333
- This is the canonical RECOVERABLE example: the agent solved it perfectly on attempt 3.

### 5.2 Test-suite scoring as RMDP exit structure

The test script `tasks/medium-2/tests/test.sh` reveals a **5-checkpoint scoring structure**:

| Checkpoint | Points | Test criterion | RMDP sub-exit |
|---|---|---|---|
| **C0**: JSON validity | 10/100 | `json.load()` succeeds | exit_0: file exists and is valid JSON |
| **C1**: total_eligible_users | 25/100 | `total_eligible_users == 7` | exit_1: filter logic correct |
| **C2**: Engineering dept | 25/100 | `count==3, avg_salary==111666.67` | exit_2: Engineering groupby correct |
| **C3**: Sales dept | 20/100 | `count==1, avg_salary==85000.0` | exit_3: Sales groupby correct |
| **C4**: HR dept | 20/100 | `count==2, avg_salary==71500.0` | exit_4: HR groupby correct |

The test file writes `$SCORE / 100.0` as the final reward. A score of 0.35 (= 35/100) means the
agent passed JSON validity + total_eligible_users (10+25=35) but failed all department checks.
A score of 0.0 means no file at all or a Python error before any JSON was written.

### 5.3 Decomposing into sub-task RMDPs

Under the RMDP framework, the full task `medium-rest-client` becomes a **multi-exit component** M_2
with 5 exit nodes corresponding to the checkpoints. For a 1-exit formulation (required for
Recursive Q-learning convergence), we decompose M_2 into a **chain of 1-exit sub-MDPs**:

```
M_2 = M_2a ∘ M_2b ∘ M_2c ∘ M_2d ∘ M_2e
```

where:

**M_2a — "HTTP client sub-task"**
- Entry: empty `/app/` directory, mock server running at `localhost:5000`
- Exit: `/app/client.py` exists AND `python3 /app/client.py` produces output without error
- Q* from data: any run that gets past the Python error = score ≥ 0.10
- RMDP exit value: 0.10 (the C0 checkpoint reward, verifiable independently)
- disteval equivalent: if we could measure "score ≥ 0.10" as a separate label, this sub-task
  would have its own SOLID/RECOVERABLE/STUCK classification.

**M_2b — "Filtering sub-task"**
- Entry: `/app/client.py` produces valid JSON with any structure
- Exit: `total_eligible_users == 7` in the output JSON
- RMDP exit value: 0.25 additional reward (the C1 checkpoint)
- Current barrier: agents that write `client.py` but use `age > 30` instead of `age >= 30`
  pass M_2a but fail M_2b.

**M_2c–M_2e — "Groupby sub-tasks"**
- Entry: correct filter result
- Exit: per-department count and avg_salary correct for each of Engineering, Sales, HR
- Each exit adds 20-25 additional reward points

**Chain structure in RMDP notation:**

```
Entry(M_2) → box_a [calls M_2a] → box_b [calls M_2b] → ... → Exit(M_2)
```

The call stack at the deepest checkpoint is:
```
κ = [(M_2, box_a), (M_2a, box_b_of_a), ...]
```

### 5.4 What disteval currently captures vs what it misses

| Observable | Currently captured | RMDP decomposition adds |
|---|---|---|
| Final score [0.0, 0.0, 1.0] | ✅ by `right_tail_analysis` | Per-checkpoint scores [0.35, 0.0, 1.0] would reveal whether M_2a is SOLID while M_2b–M_2e are STUCK |
| Divergence step | ✅ by `_find_divergence_step` (reports step ~3-5 for this task) | Divergence step *within each sub-task* M_2a, M_2b, etc. |
| Training pair | ✅ (attempt 3 vs attempt 1/2) | Separate pairs per sub-task — reinforce the sub-task success trajectory slices |
| Memory retrieval | ✅ whole-task similarity | Sub-trajectory retrieval: "find me an M_2b success even from a different whole-task" |
| Predicted gain | ✅ analytic: 0.4 × 0.667 × 0.333 = 0.089 | Per-sub-task gain: closing M_2b alone might add 0.25 without requiring full solution |

### 5.5 How an agent failure decomposition would work

Hypothetical: Codex CLI attempt 1 scores 0.35 (JSON valid + filter correct, but all department
checks fail). Under current disteval, this produces `kind="recoverable"` with the same training
pair as a 0.0-score attempt. Under RMDP decomposition:

- M_2a: SOLID (κ=1.0, always passes, no training signal needed)
- M_2b: SOLID (κ=1.0 on this agent, it always gets the count right when it writes code)
- M_2c–M_2e: STUCK on some attempts, RECOVERABLE on others

The recursion engine would then generate a sub-task curriculum that trains only on the department
groupby logic (sub-MDPs M_2c–M_2e), not on the HTTP client code. The DPO pair would be:
- Reinforce: the slice of attempt 3's trajectory covering the groupby logic (steps 8–14)
- Contrast: the slice of attempt 1's trajectory covering the same phase

This is a more targeted training signal than the current whole-trajectory DPO pair.

---

## 6. Open Questions and Risks for Phase 2

### 6.1 Sub-task boundary detection

**Question:** How do we identify the structural boundaries between sub-tasks
(entry/exit of each sub-RMDP) from trajectory data, without access to the test-suite internals?

**Current state:** `TrajectoryMonitor` can detect a binary high/low divergence step but does not
segment the trajectory into sub-task phases. The divergence step (e.g., step 5) tells us *where*
the policy differs, not which sub-RMDP boundary it corresponds to.

**Risk:** If sub-task boundaries are derived solely from trajectory structure (tool-call patterns),
they may not align with the semantic checkpoints in the test suite. A DPO pair that is correctly
segmented at the tool-call level might still mix multiple test checkpoints in one slice.

**Mitigation path:** Expose sub-test scores by modifying `test.sh` to emit intermediate rewards
(e.g., `/logs/verifier/reward_c0.txt`, `reward_c1.txt`). This is a data-collection change, not a
code change.

### 6.2 Recursion depth and stack overflow safety

**Question:** How deep should the sub-task recursion go? An unconstrained decomposition of a
STUCK task into sub-sub-tasks that are themselves STUCK could recurse indefinitely.

**Risk:** The RMDP convergence guarantee (Theorem 4) requires a *proper* 1-exit RMDP — every
execution almost surely reaches an exit. If a STUCK task is decomposed into sub-tasks that are also
all STUCK, the recursion terminates only by reaching a maximum depth limit, not by any natural
mathematical criterion.

**Mitigation path:**
- Set a maximum recursion depth parameter (default 2–3 levels).
- Apply a budget constraint: only decompose a sub-task if its estimated complexity (trajectory
  length, tool diversity) is strictly less than the parent's.
- Mark "recursively stuck" tasks explicitly and skip them for DPO training.

### 6.3 Reward credit assignment across sub-task boundaries

**Question:** How does the RMDP framework assign rewards to sub-tasks when the only observable is
the final whole-task score?

**Current state:** `right_tail.py` works entirely on whole-task scores. `task_outcome_profile()`
receives a flat list of attempt scores with no sub-task breakdown (lines 163–204).

**Risk:** If sub-task rewards must be estimated (not directly observed), the estimates may be noisy.
For example, inferring "sub-task M_2c failed" from "total score = 0.35" requires knowing what 0.35
corresponds to — which requires parsing the test script.

**Mitigation path:** Build a `TestSuiteParser` that reads `test.sh` files and extracts checkpoint
labels and point values. This is a one-time parsing step that produces a sub-reward schema for each
task, enabling proper RMDP reward decomposition.

### 6.4 Multi-exit vs 1-exit

**Question:** The test suites award partial credit (multi-exit: multiple checkpoint-passing states
yield different rewards). Recursive Q-learning convergence is only proven for 1-exit RMDPs.

**Risk:** medium-2 has 5 distinct score levels (0, 0.10, 0.35, 0.60, 0.80, 1.0). Treating this as
a multi-exit RMDP means the convergence guarantee of Theorem 4 does NOT directly apply. The paper
states multi-exit convergence holds only for *deterministic* proper RMDPs (Theorem 5).

**Mitigation path:**
- Binarize exits: for each checkpoint, define a 1-exit sub-RMDP with reward = {0, 1} for
  pass/fail. The chain of 1-exit sub-RMDPs preserves the convergence guarantee.
- Alternatively, use the 1-exit approximation with a single aggregate exit (score ≥ 0.8 threshold
  as used in `training_sim.py` line 55: `THRESHOLD = 0.8`).

### 6.5 Distribution shift across cycles

**Question:** As the agent improves across cycles, the task distribution changes (RECOVERABLE →
SOLID; STUCK → RECOVERABLE). How does the recursion engine adapt its sub-task decompositions when
previously STUCK sub-tasks become RECOVERABLE?

**Current state:** `SelfEngine.reload()` (lines 363–371) reloads all data after an external
training step. The recursion engine would need to re-derive sub-task boundaries from the updated
trajectory data.

**Risk:** Sub-task decompositions from cycle N may be stale in cycle N+1 if the agent's behavior
changes enough that the old divergence steps are no longer informative.

**Mitigation path:** Re-run the full RMDP decomposition (sub-task boundary detection + profile
computation) at every `reload()` call. Cache decompositions keyed by the trajectory hash so only
changed tasks are re-analyzed.

### 6.6 Compatibility with DPO training format

**Question:** Can sub-trajectory slices (a contiguous prefix of a whole trajectory) be used as
DPO training pairs in the existing `CURRICULUM_FORMAT.md` schema?

**Current state:** `TrainingPair` in `self_engine.py` (lines 70–78) references whole trajectory
files (`reinforce_traj_path`, `contrast_traj_path`). The DPO trainer reads the full `steps` array.

**Risk:** A DPO trainer that receives a truncated trajectory (steps 0–8 of a 20-step file) may not
have enough context to learn the correct behavior, or may learn to terminate early as a side effect.

**Mitigation path:** Add `entry_step: int` and `exit_step: int` fields to `TrainingPair`. The DPO
trainer would slice `steps[entry_step:exit_step]` and prepend the full context up to `entry_step`
as conditioning. This is a backward-compatible schema extension.

### 6.7 Computational cost of sub-task analysis

**Question:** For k tasks each decomposed into m sub-tasks, the analysis cost grows as O(k·m).
With the current 6-task benchmark this is trivial, but at 100+ tasks it could be slow.

**Risk:** `TrajectoryMemory._rebuild_index()` (lines 182–194) rebuilds the full embedding matrix on
every `add()` call. At sub-trajectory granularity, the number of entries could grow by 5–10×.

**Mitigation path:** Defer index rebuilding until a `flush()` call (batch mode). The current single-
call rebuild is already noted in the code as a performance concern.

---

## 7. Summary and Readiness for Phase 2

The current disteval codebase has all the primitive components needed for RMDP-style recursion:

1. **Component MDPs** — each task is already a component with entry/exit structure.
2. **Exit values** — Q*(t) is already computed for every task.
3. **Sub-MDP Q-values** — κ(t) already measures the gap from V*.
4. **Divergence detection** — `TrajectoryMonitor._find_divergence_step` already identifies the
   structural fork point corresponding to a recursive call boundary.
5. **Value backup** — `TrajectoryMemory.retrieve_for_new_task` already does similarity-based
   retrieval that serves as an empirical call-stack value backup.
6. **Convergence criterion** — `cycle_complete` already implements the natural termination
   condition (κ → 1.0 for all RECOVERABLE tasks).

**What is missing:**

- A `RecursionEngine` class that explicitly decomposes STUCK tasks into sub-task RMDPs.
- Sub-task boundary detection (entry/exit step annotation on trajectories).
- Per-checkpoint reward decomposition (requires parsing test scripts or modifying them).
- Sub-trajectory retrieval and DPO pair slicing.
- Depth/budget constraints on the recursion.

These gaps map directly to the Phase 2 design deliverable
(`research/phase2_recursion_engine.md`). The formal RMDP convergence guarantees (Theorem 4)
apply cleanly to the binarized 1-exit formulation of each disteval sub-task, providing a
principled mathematical foundation for the recursion engine design.

---

## References

- Hahn, Perez, Schewe, Somenzi, Trivedi, Wojtczak (2022). "Recursive Reinforcement Learning."
  NeurIPS 2022. arXiv:2206.11430.
- disteval codebase: `/Users/shikharsehgal/rl-dist-eval/`
- Bellemare, Dabney, Munos (2017). "A Distributional Perspective on Reinforcement Learning." ICML.
- Rafailov et al. (2023). "Direct Preference Optimization." NeurIPS.
