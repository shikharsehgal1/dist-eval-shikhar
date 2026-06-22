# Research: Recursive Self-Improvement for disteval

## Goal

Investigate how disteval's agent improvement engine can be enhanced by combining:

1. **Recursive Reinforcement Learning** (arXiv:2206.11430) — environments described as recursive MDPs (RMDPs) with entry/exit points and a call-stack.
2. **A recursion engine** that produces RL environments from agent eval data.
3. **A self-improvement loop** where those environments evolve based on distributed agent evaluations, with each cycle's solution affecting the next cycle's tasks.

## Why this matters for disteval

disteval already measures the full outcome distribution, classifies tasks as SOLID / RECOVERABLE / STUCK, and generates a DPO curriculum from the agent's own eval data. The current `SelfEngine` (see `disteval/self_engine.py`) explicitly references RMDP concepts in its docstring but does not implement them. This research should determine whether implementing RMDP-style recursion can:

- Decompose STUCK tasks into sub-tasks the agent can solve.
- Close RECOVERABLE gaps by treating each task as a recursive call with entry/exit conditions.
- Generate new training environments automatically from the gaps found in distributed evals.
- Make the first cycle's output (e.g., a curriculum) change the task distribution for the next cycle.

## Multi-step research plan

Each phase produces a deliverable that the next phase consumes. A phase should not start until the previous phase's deliverable is available.

### Phase 1 — Literature mapping (input: arXiv paper + current codebase)

**Deliverable:** A written report (`research/phase1_rmdp_mapping.md`) containing:

- A concise summary of arXiv:2206.11430 (Recursive Reinforcement Learning / RMDPs).
- The key formal objects: RMDP, entry/exit points, call-stack, Recursive Q-learning, convergence guarantees.
- A mapping table from RMDP concepts to disteval concepts:
  - RMDP ↔ task / environment
  - Entry point ↔ task instruction + initial state
  - Exit point ↔ task solution / test success
  - Recursive call ↔ sub-task invocation
  - Call-stack ↔ trajectory step history / dependency chain
  - Q-value at entry/exit ↔ per-task consistency score `κ(t)` or gap `Δ(t)`
- Identification of which parts of disteval (`self_engine.py`, `right_tail.py`, `trajectory_monitor.py`, `trajectory_memory.py`, `training_sim.py`) would need to change.
- A list of open questions for Phase 2.

### Phase 2 — Recursion engine design (input: Phase 1 report)

**Deliverable:** A design document (`research/phase2_recursion_engine.md`) containing:

- A proposed `RecursionEngine` class/module that sits alongside or inside `SelfEngine`.
- How it decomposes a task into sub-tasks using the trajectory monitor's divergence step and trajectory memory.
- How it builds sub-task RMDPs with entry/exit conditions derived from eval data.
- How it handles recursion depth, termination, and stack overflow safety.
- A concrete example using one of the existing disteval tasks (e.g., `medium-2` or `hard-1`).
- Open questions for Phase 3.

### Phase 3 — RL environment generation (input: Phase 2 design)

**Deliverable:** A design document (`research/phase3_environment_generation.md`) containing:

- A schema for generated RL environments (states, actions, rewards, transitions).
- How environments are generated from the SOLID/RECOVERABLE/STUCK taxonomy:
  - SOLID sub-tasks become stable sub-environments.
  - RECOVERABLE sub-tasks become contrastive environments with reinforce/contrast trajectories.
  - STUCK tasks become exploration targets that trigger environment expansion.
- How distributed agent evaluations (multiple agents, multiple runs) feed back into environment generation.
- How the solution from cycle `n` changes the task distribution / environment parameters for cycle `n+1`.
- A data format for the generated environments (could extend `CURRICULUM_FORMAT.md` or define a new JSON schema).
- Open questions for Phase 4.

### Phase 4 — Prototype integration plan (input: Phase 3 design)

**Deliverable:** A prototype plan (`research/phase4_integration_plan.md`) containing:

- Concrete changes to disteval files:
  - `disteval/self_engine.py` — integrate `RecursionEngine`.
  - `disteval/right_tail.py` — expose sub-task gaps.
  - `disteval/trajectory_monitor.py` — expose divergence step as entry/exit signal.
  - `disteval/trajectory_memory.py` — retrieve sub-task demonstrations.
  - `disteval/training_sim.py` — simulate recursive training gains.
- Proposed new files (e.g., `disteval/recursion_engine.py`, `disteval/environment_generator.py`).
- API signatures with type hints.
- A minimal end-to-end flow: eval → recursion analysis → environment generation → training → re-eval.
- A risk/uncertainty section and next steps for implementation.

## Running the research

Use `run_subagent` with the `subagent_explore` profile for independent research phases. Each phase should read the previous phase's deliverable and the current codebase. Phase 1 can run in parallel with multiple subagents investigating different aspects; later phases should wait for the prior deliverable.

When a phase completes, update the project todo list and append the deliverable to this skill file or link it from the `research/` directory.

## Constraints

- Do not modify existing disteval code during the research phase; produce design documents only.
- Keep deliverables grounded in the actual repository (cite file paths and line numbers where possible).
- All generated code should be compatible with Python 3.10+ and the existing numpy/pandas/scipy dependency stack.
- Do not introduce new runtime dependencies unless justified.
