# Phase 2 Master Report: RecursionEngine Design for disteval

**Date:** 2026-06-23
**Input:** Phase 1 master report + three Phase 2 design documents
**Output:** Consolidated `RecursionEngine` design
**Next step:** Phase 3 — RL environment generation that self-improves from distributed evals

---

## 1. What Phase 2 designed

Phase 2 produced three design documents:

- `research/phase2a_recursion_engine_api.md` — API and data classes for `RecursionEngine`.
- `research/phase2b_decomposition_algorithm.md` — algorithm for decomposing tasks into sub-task RMDPs.
- `research/phase2c_integration_design.md` — how to integrate the engine into existing disteval files.

This master report consolidates the agreed design and identifies the decisions that Phase 3 must consume.

---

## 2. Core data classes

New module: `disteval/recursion_engine.py`.

### 2.1 `PhaseBoundary`

```python
@dataclass
class PhaseBoundary:
    entry_step: int
    exit_step: Optional[int] = None
    label: str = ""
    confidence: float = 0.0
    tool_signature: tuple[str, ...] = field(default_factory=tuple)
    entry_features: Optional[TrajectoryFeatures] = None
    exit_features: Optional[TrajectoryFeatures] = None
    is_predicted_exit: bool = False
```

Represents one contiguous phase in a parent trajectory, used as an RMDP entry/exit pair.

### 2.2 `SubTask` / `SubTaskDefinition`

```python
@dataclass
class SubTaskDefinition:
    sub_task_id: str
    parent_task: str
    sub_task_depth: int
    entry_step: int
    exit_step: int
    phase_tag: str
    instruction: str
    estimated_q_star: float
    estimated_q_bar: float
    kind: str  # "solid" | "recoverable" | "stuck"
```

A callable sub-RMDP component with a unique ID, parent reference, and structural boundary.

### 2.3 `RMDPNode` / `SubTaskGraph`

```python
@dataclass
class SubTaskGraph:
    parent_tasks: list[str]
    sub_tasks: list[SubTaskDefinition]
    edges: list[tuple[str, str]]
    profiles: dict[str, TaskOutcomeProfile]
```

JSON-serializable graph of parent/sub-task relationships returned by `RecursionEngine.decompose()`.

### 2.4 `RecursionEngineConfig`

```python
@dataclass
class RecursionEngineConfig:
    max_depth: int = 3
    divergence_confidence: float = 0.70
    max_phase_boundaries: int = 5
    min_monotone_difficulty: float = 0.0
    reward_propagation: str = "weighted_sum"
    enable_decompose_stuck: bool = True
    enable_decompose_recoverable: bool = True
    min_sub_task_score: float = 0.0
    require_checkpoint_alignment: bool = False
    max_stack_nodes: int = 100
    memory_retrieval_k: int = 3
```

Defaults chosen to prevent infinite decomposition of STUCK tasks and to match the five-checkpoint structure of `medium-2`.

---

## 3. `RecursionEngine` public API

```python
class RecursionEngine:
    def __init__(
        self,
        monitor: TrajectoryMonitor,
        memory: Optional[TrajectoryMemory] = None,
        config: Optional[RecursionEngineConfig] = None,
        agent_name: str = "agent",
        model_name: str = "unknown",
    ) -> None: ...

    def decompose(self, report: RightTailReport, traj_records: list) -> SubTaskGraph: ...
    def decompose_task(self, profile, traj_paths, depth=0, parent=None) -> RMDPNode: ...
    def decompose_stuck_tasks(self, report: RightTailReport) -> dict[str, RMDPNode]: ...
    def decompose_recoverable_tasks(self, report: RightTailReport, min_gap=0.30) -> dict[str, RMDPNode]: ...
    def find_phase_boundaries(self, steps, start_step=0, end_step=None) -> list[PhaseBoundary]: ...
    def compute_recursive_gap(self, root: RMDPNode) -> float: ...
    def to_sub_task_graph(self, roots: list[RMDPNode]) -> SubTaskGraph: ...
```

Instantiated from `SelfEngine.from_job_dirs()` and called inside `SelfEngine.run_cycle()` when recursion is enabled.

---

## 4. Decomposition algorithm

### 4.1 Candidate identification

| Type | Criterion | Strategy |
|---|---|---|
| Partial-credit RECOVERABLE | `kind == "recoverable"` and a low-scoring attempt has `0 < score < 0.9 * q_star` | Divergence-point cleavage using `TrajectoryMonitor` |
| Pure STUCK | `kind == "stuck"` (`q_star == 0`) | `TrajectoryMemory` retrieval fallback |
| SOLID | `kind == "solid"` | No decomposition |

### 4.2 Boundary detection

For each high-scoring vs low-scoring trajectory pair, find the first divergence step where the monitor predicts HIGH for the high run and LOW/uncertain for the low run. Collect all unique divergence steps, add `0` and `max_steps`, and merge adjacent boundaries that are too close.

Pure STUCK tasks use `TrajectoryMemory.retrieve_for_new_task()` to find similar successful tasks and derive boundaries from their phase tags.

### 4.3 Entry and exit conditions

- Entry of sub-task `i` = boundary `i-1 + 1` (or `0` for the first sub-task).
- Exit of sub-task `i` = boundary `i`.
- Entry condition = the tool-sequence prefix `steps[0:entry]` as conditioning context + captured environment state.
- Exit condition = reaching `exit_step` with monitor `p_high` confident, optionally validated by a test-script checkpoint.

### 4.4 Scoring sub-task slices

1. **Preferred:** per-checkpoint test scores (extend `test.sh` to write `/logs/verifier/reward_c{i}.txt`).
2. **Fallback:** structural proxy — `1.0` if monitor predicts HIGH at the slice's final step, else `0.0`.
3. **Last resort:** inherit the full-task score for the last segment only.

After scoring, re-run `right_tail.task_outcome_profile()` on the sub-task score array to classify it as SOLID/RECOVERABLE/STUCK.

### 4.5 Recursive termination

- Hard depth cap (`max_depth = 3`).
- Monotone difficulty check: each sub-task must be strictly easier than its parent (e.g., fewer required tool calls).
- "Recursively stuck" detection: if all sub-tasks of a STUCK parent remain STUCK, mark them and stop decomposition.

---

## 5. Integration with existing disteval components

### 5.1 Files requiring changes

- `disteval/self_engine.py` — add recursion fields to `TrainingPair`, `TaskImprovement`, `SelfImprovementPlan`; call `RecursionEngine` inside `run_cycle()`; slice trajectories using `entry_step`/`exit_step`.
- `disteval/right_tail.py` — add `parent_task`, `sub_task_depth`, `sub_task_profiles`, `recursive_gap` to `TaskOutcomeProfile` and `RightTailReport`.
- `disteval/trajectory_monitor.py` — add `PhaseBoundary` dataclass and `find_phase_boundaries()` method; add `entry_step`/`exit_step` to `TrajectoryRecord` and `PatternMatch`.
- `disteval/trajectory_memory.py` — add `sub_task_slices` to `TrajectoryRecord`; add `retrieve_for_sub_task()` method.
- `disteval/training_sim.py` — propagate sub-task improvements to parent task scores using weighted-sum rule.
- `disteval/__main__.py` — add `--enable-recursion` flag to `disteval engine`.
- `CURRICULUM_FORMAT.md` — extend JSON schema with sub-task fields.

### 5.2 New files

- `disteval/recursion_engine.py` — the main `RecursionEngine` class and data classes.
- `tests/test_recursion_engine.py` — unit tests using `medium-2` fixtures.

### 5.3 Backward compatibility

Recursion is default-disabled. All new fields are optional. Existing `disteval engine` CLI usage is unchanged unless `--enable-recursion` is passed.

---

## 6. Worked example: `medium-2` (REST client)

The task has a 5-checkpoint scoring chain (0.10 + 0.25 + 0.25 + 0.20 + 0.20). Under recursion:

```
medium-2
├── phase-0: HTTP client runs without error (reward 0.10)
├── phase-1: Correct filter (reward +0.25)
├── phase-2: Engineering groupby correct (reward +0.25)
├── phase-3: Sales groupby correct (reward +0.20)
└── phase-4: HR groupby correct (reward +0.20)
```

For Codex CLI's `[0.0, 0.0, 1.0]` profile, the current disteval curriculum would train on the whole-trajectory DPO pair. Under the `RecursionEngine`, if the partial-credit attempt shows that phases 0–1 are already SOLID and phases 2–4 are RECOVERABLE/STUCK, the curriculum would target only the groupby logic.

---

## 7. Open questions for Phase 3

1. **RL environment schema:** What is the JSON/structural format of a generated RL environment? Should it be a Gymnasium env, a JSON config for Harbor, or a new disteval-specific format?
2. **Entry-state serialization:** How is the environment state at a sub-task entry captured and replayed for independent sub-task runs?
3. **Reward shaping:** Should generated environments use sparse per-checkpoint rewards or shaped intermediate rewards?
4. **Self-improvement loop:** How does the solution from cycle `n` change the environment distribution for cycle `n+1`? Specifically, how do newly SOLID sub-tasks get removed from the environment pool and newly STUCK parent tasks get added?
5. **Distributed evals:** How do multiple agents' evals aggregate into a shared sub-task graph and environment generation pool?
6. **Multi-exit vs 1-exit:** Decompose multi-checkpoint tasks into chains of 1-exit sub-RMDPs to preserve convergence guarantees, or support true multi-exit environments?
7. **Cross-agent sharing:** If Agent A is SOLID on a sub-task, can its trajectories serve as reinforce targets for Agent B's sub-task curriculum?

These questions are the input for Phase 3: RL environment generation and the recursive self-improvement loop.
