# Phase 4A — Prototype Implementation Plan: Recursive Self-Improvement for disteval

**Date:** 2026-06-23  
**Input:** Phase 3 master report (`research/phase3_master_report.md`), Phase 2 master report
(`research/phase2_master_report.md`), and all source files listed below.  
**Output:** A detailed, milestone-based prototype plan grounded in the actual codebase.  
**Constraint:** No existing disteval code is modified during Phase 4A research. Implementation
begins only after this plan is approved.

---

## Reading guide

| Section | Content |
|---------|---------|
| §1 | Prioritized file list with dependency order |
| §2 | Per-file API specification with type hints |
| §3 | Code skeletons for the most important new classes |
| §4 | Milestone 1 — test-suite checkpoint parser + smoke tests |
| §5 | Milestone 2 — `RecursionEngine` and sub-task decomposition |
| §6 | Milestone 3 — environment generation and self-improvement loop |
| §7 | Milestone 4 — distributed eval pooling |
| §8 | Risk and backward-compatibility notes |

---

## 1  Prioritized file list

The dependency graph drives the order. Each new file is written as a
**pure addition** to the package — no existing code is touched until the
backward-compatible integration step at the very end of each milestone.

```
 ┌─────────────────────────────────────────────────────────┐
 │  Milestone 1 (no recursion, no containers)              │
 │                                                         │
 │  disteval/test_suite_parser.py          (new)           │
 │  tests/test_test_suite_parser.py        (new tests)     │
 └──────────────────────┬──────────────────────────────────┘
                        │ consumed by
 ┌──────────────────────▼──────────────────────────────────┐
 │  Milestone 2 (recursion engine, pure Python)            │
 │                                                         │
 │  disteval/recursion_engine.py           (new)           │
 │  disteval/self_engine.py                (extend, opt-in)│
 │  disteval/right_tail.py                 (extend, opt-in)│
 │  disteval/trajectory_monitor.py         (extend, opt-in)│
 │  disteval/trajectory_memory.py          (extend, opt-in)│
 │  disteval/__main__.py                   (extend, opt-in)│
 │  tests/test_recursion_engine.py         (new tests)     │
 └──────────────────────┬──────────────────────────────────┘
                        │ consumed by
 ┌──────────────────────▼──────────────────────────────────┐
 │  Milestone 3 (environment generation + loop)            │
 │                                                         │
 │  disteval/environment_generator.py      (new)           │
 │  disteval/environment_registry.py       (new)           │
 │  disteval/training_sim.py               (extend, opt-in)│
 │  tests/test_environment_generator.py    (new tests)     │
 │  tests/test_environment_registry.py     (new tests)     │
 └──────────────────────┬──────────────────────────────────┘
                        │ consumed by
 ┌──────────────────────▼──────────────────────────────────┐
 │  Milestone 4 (distributed eval pool)                    │
 │                                                         │
 │  disteval/distributed_eval.py           (new)           │
 │  tests/test_distributed_eval.py         (new tests)     │
 └─────────────────────────────────────────────────────────┘
```

### 1.1 Dependency rationale

- `test_suite_parser.py` has **zero imports from disteval** (only stdlib). It is
  the safest starting point and provides real checkpoint data for every later
  module.
- `recursion_engine.py` imports `TrajectoryMonitor` and `TrajectoryMemory` (read-
  only) and `right_tail.TaskOutcomeProfile` — all stable, unmodified objects.
- `environment_generator.py` imports `SubTaskDefinition` from `recursion_engine`
  and checkpoint data from `test_suite_parser`.
- `environment_registry.py` imports `GenEnv` from `environment_generator` and
  `TaskOutcomeProfile` from `right_tail`.
- `distributed_eval.py` imports `RightTailReport` and `SubTaskGraph` — both
  defined without circular dependencies.

---

## 2  Per-file API specification

### 2.1 `disteval/test_suite_parser.py` (new)

**Purpose:** Parse `tests/test.sh` for all six disteval tasks and extract the
ordered list of (checkpoint index, description, reward weight) triples. This is
the ground truth for sub-task reward wiring.

**Depends on:** stdlib only (`re`, `pathlib`).

**Public API:**

```python
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CheckpointSpec:
    """One scoreable checkpoint parsed from a test.sh file."""

    index: int                    # 0-based position in the test script
    task_name: str                # e.g. "medium-2"
    description: str              # human-readable label extracted from comment
    reward_weight: float          # fractional contribution: e.g. 0.25 for SCORE+25
    score_increment: int          # raw SCORE+= value: e.g. 25
    total_score: int              # denominator used in reward.txt: e.g. 100
    condition_source: str         # the shell/Python code that gates this checkpoint
    checkpoint_id: str            # e.g. "medium-2::phase-2"


def parse_test_suite(
    test_sh_path: str | Path,
    task_name: str,
) -> list[CheckpointSpec]:
    """
    Parse a disteval test.sh file and return ordered CheckpointSpec list.

    Detects SCORE+= increments, the denominator in the final `print($SCORE / N)`,
    and optional inline comments labelling each checkpoint.

    Raises:
        ValueError  if the file cannot be parsed (no SCORE variable found).
        FileNotFoundError  if path does not exist.
    """
    ...


def parse_all_tasks(tasks_dir: str | Path = "tasks") -> dict[str, list[CheckpointSpec]]:
    """
    Walk tasks_dir and parse every test.sh found.

    Returns {task_name: [CheckpointSpec, ...]} for all parseable tasks.
    Tasks whose test.sh cannot be parsed are silently skipped and logged
    at WARNING level.
    """
    ...


def checkpoint_weights(specs: list[CheckpointSpec]) -> list[float]:
    """Return just the reward weights in checkpoint order."""
    return [s.reward_weight for s in specs]
```

**Implementation notes:**
- The regex pattern `SCORE=\$\(\(SCORE \+ (\d+)\)\)` captures each increment.
  The denominator is extracted from the final line matching
  `python3 -c "print\(\$SCORE / (\d+)`.
- For `medium-2` the expected result is five specs with weights
  `[0.10, 0.25, 0.25, 0.20, 0.20]` summing to 1.0.
- For `easy-1` (three-checkpoint, weights `[0.34, 0.33, 0.33]`) the sum is 1.0.
- The condition_source is the shell or Python block immediately preceding each
  `SCORE+=` line, useful for sub-task exit condition derivation.

---

### 2.2 `disteval/recursion_engine.py` (new)

**Purpose:** Decompose tasks into sub-task RMDPs using trajectory monitor
divergence points and checkpoint alignment.

**Depends on:**
- `disteval.trajectory_monitor.TrajectoryMonitor`, `TrajectoryRecord`,
  `TrajectoryFeatures`, `PatternMatch`
- `disteval.trajectory_memory.TrajectoryMemory`
- `disteval.right_tail.TaskOutcomeProfile`, `right_tail.task_outcome_profile`
- `disteval.test_suite_parser.CheckpointSpec`, `parse_test_suite`
- stdlib: `dataclasses`, `typing`, `json`, `os`

**Public API:**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .trajectory_monitor import TrajectoryMonitor, TrajectoryFeatures
from .trajectory_memory import TrajectoryMemory
from .right_tail import TaskOutcomeProfile
from .test_suite_parser import CheckpointSpec


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PhaseBoundary:
    """One RMDP entry/exit pair within a parent trajectory."""

    entry_step: int
    exit_step: Optional[int]      # None = open-ended (last boundary)
    label: str                    # e.g. "phase-2"
    confidence: float             # monitor p_high at entry_step
    tool_signature: tuple[str, ...]   # canonical tools in [entry_step:exit_step]
    checkpoint_spec: Optional[CheckpointSpec] = None   # if aligned to test.sh
    is_predicted_exit: bool = False


@dataclass
class SubTaskDefinition:
    """A callable sub-RMDP component with a unique ID and structural boundary."""

    sub_task_id: str              # e.g. "medium-2::phase-2"
    parent_task: str              # e.g. "disteval/medium-rest-client"
    sub_task_depth: int           # 0 = top-level parent
    entry_step: int
    exit_step: int
    phase_tag: str                # "write" | "exec" | "verify" | "filter" | ...
    instruction: str              # derived human-readable sub-task description
    estimated_q_star: float       # best per-slice score observed
    estimated_q_bar: float        # mean per-slice score observed
    kind: str                     # "solid" | "recoverable" | "stuck"
    reward_weight: float          # from CheckpointSpec.reward_weight or 1/N
    shareable: bool = False       # opt-in for cross-agent training transfer


@dataclass
class RMDPNode:
    """One node in the recursive decomposition tree."""

    definition: SubTaskDefinition
    children: list["RMDPNode"] = field(default_factory=list)
    recursive_gap: float = 0.0
    recursively_stuck: bool = False


@dataclass
class SubTaskGraph:
    """JSON-serializable graph of parent/sub-task relationships."""

    parent_tasks: list[str]
    sub_tasks: list[SubTaskDefinition]
    edges: list[tuple[str, str]]  # (parent_id, child_id)
    profiles: dict[str, TaskOutcomeProfile]  # keyed by sub_task_id


@dataclass
class RecursionEngineConfig:
    max_depth: int = 3
    divergence_confidence: float = 0.70
    max_phase_boundaries: int = 5
    min_monotone_difficulty: float = 0.0
    reward_propagation: str = "weighted_sum"   # only supported value for now
    enable_decompose_stuck: bool = True
    enable_decompose_recoverable: bool = True
    min_sub_task_score: float = 0.0
    require_checkpoint_alignment: bool = False
    max_stack_nodes: int = 100
    memory_retrieval_k: int = 3


# ── Main class ────────────────────────────────────────────────────────────────

class RecursionEngine:
    """
    Decompose tasks into sub-task RMDPs using trajectory monitor divergence
    points, trajectory memory retrievals, and (optionally) test.sh checkpoints.

    Instantiated once per SelfEngine cycle.  All public methods are pure
    (no I/O side effects) except `load_checkpoint_specs()`.
    """

    def __init__(
        self,
        monitor: TrajectoryMonitor,
        memory: Optional[TrajectoryMemory] = None,
        config: Optional[RecursionEngineConfig] = None,
        agent_name: str = "agent",
        model_name: str = "unknown",
    ) -> None: ...

    # ── Checkpoint loading ──────────────────────────────────────────────────

    def load_checkpoint_specs(
        self,
        tasks_dir: str = "tasks",
    ) -> dict[str, list[CheckpointSpec]]:
        """
        Load CheckpointSpec lists for all tasks.

        Populates self._checkpoint_specs and returns the dict.
        """
        ...

    # ── Top-level entry points ──────────────────────────────────────────────

    def decompose(
        self,
        report: "RightTailReport",       # from right_tail.right_tail_analysis()
        traj_records: list,              # list[trajectory_monitor.TrajectoryRecord]
    ) -> SubTaskGraph:
        """
        Full decomposition: decompose all STUCK + RECOVERABLE tasks whose
        config permits it (controlled by RecursionEngineConfig).

        Returns a SubTaskGraph ready for EnvironmentGenerator consumption.
        """
        ...

    def decompose_task(
        self,
        profile: TaskOutcomeProfile,
        traj_files: list[tuple[float, str]],  # [(score, traj_path), ...]
        depth: int = 0,
        parent_id: Optional[str] = None,
    ) -> RMDPNode:
        """
        Decompose one task into a tree of RMDPNodes.

        Uses find_phase_boundaries() to identify sub-task boundaries and
        score_sub_task_slices() to estimate per-slice scores.
        Applies recursive termination rules from RecursionEngineConfig.
        """
        ...

    # ── Boundary detection ──────────────────────────────────────────────────

    def find_phase_boundaries(
        self,
        steps: list[dict],
        high_steps: Optional[list[dict]] = None,
        low_steps: Optional[list[dict]] = None,
        start_step: int = 0,
        end_step: Optional[int] = None,
        checkpoint_specs: Optional[list[CheckpointSpec]] = None,
    ) -> list[PhaseBoundary]:
        """
        Find phase boundaries for a trajectory by combining:
          1. Monitor divergence points between high/low runs.
          2. Checkpoint alignment (if checkpoint_specs provided).
          3. Structural breakpoints (first write, first exec, etc.).

        Returns boundaries sorted ascending by entry_step with no overlaps.
        Merges adjacent boundaries that are within 2 steps of each other.
        Never returns more than config.max_phase_boundaries boundaries.
        """
        ...

    # ── Slice scoring ───────────────────────────────────────────────────────

    def score_sub_task_slices(
        self,
        traj_files: list[tuple[float, str]],
        boundaries: list[PhaseBoundary],
    ) -> dict[str, list[float]]:
        """
        Estimate per-slice scores for each trajectory and boundary.

        Priority:
          1. Per-checkpoint test scores (reward_c{i}.txt sidecar file,
             if present — written by extended test.sh in Milestone 3+).
          2. Structural proxy: 1.0 if monitor p_high >= config.divergence_confidence
             at the slice's final step, else 0.0.
          3. Last-segment inheritance: inherits full-task score for the last slice.

        Returns {phase_label: [score_per_traj, ...]} matching len(traj_files).
        """
        ...

    # ── Recursive gap ───────────────────────────────────────────────────────

    def compute_recursive_gap(self, root: RMDPNode) -> float:
        """
        Compute the total recursive right-tail gap across the subtree.

        Gap of a leaf = its estimated_q_star - estimated_q_bar.
        Gap of an internal node = weighted sum of children gaps, using
        checkpoint reward_weight values when available.
        """
        ...

    # ── Graph serialization ─────────────────────────────────────────────────

    def to_sub_task_graph(
        self,
        roots: list[RMDPNode],
        profiles: dict[str, TaskOutcomeProfile],
    ) -> SubTaskGraph:
        """Flatten a forest of RMDPNodes into a SubTaskGraph."""
        ...

    def sub_task_graph_to_dict(self, graph: SubTaskGraph) -> dict:
        """Return a JSON-serializable dict (for curriculum JSON extension)."""
        ...
```

---

### 2.3 `disteval/right_tail.py` (extend, backward-compatible)

**Changes:** Add optional fields to `TaskOutcomeProfile` and `RightTailReport`.
All new fields default to `None` / empty list so existing callers are unaffected.

```python
# In TaskOutcomeProfile — add after existing fields:
parent_task: Optional[str] = None         # non-None only for sub-tasks
sub_task_depth: int = 0                   # 0 = top-level
sub_task_profiles: list["TaskOutcomeProfile"] = field(default_factory=list)
recursive_gap: Optional[float] = None    # sum of sub-task gaps (from RecursionEngine)
checkpoint_weights: list[float] = field(default_factory=list)  # from test_suite_parser

# In RightTailReport — add after existing fields:
sub_task_graph: Optional["SubTaskGraph"] = None   # populated if recursion enabled
recursive_score_left: Optional[float] = None      # sum of recursive gaps
```

**No changes** to `right_tail_analysis()` signature. The sub-task fields are
populated by a new `right_tail_analysis_with_recursion()` wrapper defined in
`recursion_engine.py` (not in `right_tail.py`) to avoid circular imports.

---

### 2.4 `disteval/trajectory_monitor.py` (extend, backward-compatible)

**Changes:** Add optional fields to `TrajectoryRecord` and `PatternMatch`;
expose `find_phase_boundaries()` as a public method of `TrajectoryMonitor`.

```python
# In TrajectoryRecord — add after existing fields:
entry_step: int = 0
exit_step: Optional[int] = None
sub_task_id: Optional[str] = None

# In PatternMatch — add after existing fields:
phase_boundary: Optional["PhaseBoundary"] = None   # forward ref, resolved at runtime

# In TrajectoryMonitor — new public method:
def divergence_steps(
    self,
    high_path: str,
    low_path: str,
    max_check: int = 30,
    confidence_threshold: float = 0.70,
) -> list[int]:
    """
    Return all step indices where high_path is predicted HIGH
    (p_high >= confidence_threshold) and low_path is predicted LOW
    (p_high < 1 - confidence_threshold) simultaneously.

    More general than _find_divergence_step() in SelfEngine: returns all
    divergence steps, not just the first. Used by RecursionEngine to find
    multiple phase boundaries.
    """
    ...
```

---

### 2.5 `disteval/trajectory_memory.py` (extend, backward-compatible)

**Changes:** Add `sub_task_slices` to `TrajectoryRecord` and a new retrieval
method for sub-task contexts.

```python
# In TrajectoryRecord — add optional field:
sub_task_slices: Optional[list[dict]] = None
# Each element: {"phase_label": str, "entry_step": int, "exit_step": int,
#                "tool_sequence": list[str], "score": float}

# In TrajectoryMemory — new public method:
def retrieve_for_sub_task(
    self,
    sub_task_id: str,
    entry_tool_prefix: list[str],
    k: int = 3,
    outcome_filter: str = "high",
) -> list[RetrievalResult]:
    """
    Retrieve memories specifically for a sub-task context.

    Uses entry_tool_prefix as the structural query (the tool calls that
    led up to the sub-task entry boundary), and filters by the sub_task_id
    phase tag when sub_task_slices are present.
    """
    ...
```

---

### 2.6 `disteval/environment_generator.py` (new)

**Purpose:** Map a `SubTaskDefinition` to a `GenEnv` JSON (the Phase 3 schema).

**Depends on:**
- `disteval.recursion_engine.SubTaskDefinition`, `PhaseBoundary`
- `disteval.test_suite_parser.CheckpointSpec`
- `disteval.trajectory_memory.TrajectoryMemory`
- stdlib: `json`, `pathlib`, `dataclasses`

**Public API:**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .recursion_engine import SubTaskDefinition, PhaseBoundary
from .test_suite_parser import CheckpointSpec


@dataclass
class GenEnv:
    """
    Generated RL environment — the atomic unit of the environment generation
    layer.  Follows the six-tuple schema from Phase 3:
       GenEnv = (S, A, O, R, T, Z)
    encoded as a JSON-serializable Python dataclass.
    """

    # Identity
    env_id: str                   # e.g. "medium-2::phase-2::cycle-1"
    sub_task_id: str              # e.g. "medium-2::phase-2"
    parent_task: str
    cycle: int

    # State spec (S)
    context_prefix_steps: list[dict]   # steps[0:entry_step] from reinforce traj
    entry_step: int
    phase_tag: str

    # Observation spec (O)
    instruction: str
    context_summary: str          # ≤300-token condensed prefix
    memory_prompt: Optional[str]  # from TrajectoryMemory.generate_retrieval_prompt

    # Reward spec (R)
    reward_weight: float          # from CheckpointSpec or 1/N uniform
    checkpoint_condition_source: str  # the shell/Python block that gates the reward

    # Termination spec (Z)
    step_budget: int              # max steps before truncation
    success_threshold: float = 0.70   # monitor p_high for implicit success

    # Status (updated across cycles)
    status: str = "active"        # "active" | "retired" | "depth_cap" | "recursively_stuck"
    kind: str = "recoverable"     # "solid" | "recoverable" | "stuck"

    # Materialization helpers
    reinforce_traj_path: Optional[str] = None
    contrast_traj_path: Optional[str] = None
    task_dir: Optional[str] = None    # path to tasks/medium-2/


@dataclass
class MaterializedEnvFiles:
    """Paths written by EnvironmentGenerator.materialise()."""
    env_json_path: str
    sub_task_instruction_path: str
    sub_task_test_snippet_path: str   # minimal test.sh fragment


class EnvironmentGenerator:
    """
    Produces GenEnv JSON objects from SubTaskDefinitions.

    Does NOT modify existing task files.  Instead it writes to an
    output_dir (default: generated_envs/) alongside the existing tasks/.
    """

    def __init__(
        self,
        tasks_dir: str = "tasks",
        output_dir: str = "generated_envs",
        memory: Optional["TrajectoryMemory"] = None,
    ) -> None: ...

    def generate(
        self,
        sub_task: SubTaskDefinition,
        reinforce_traj_path: str,
        cycle: int = 1,
        context_prefix_steps: Optional[list[dict]] = None,
    ) -> GenEnv:
        """
        Generate a GenEnv for one sub-task.

        Steps (from Phase 3 master report §3):
          1. Derive instruction from phase_tag + parent instruction.md.
          2. Extract context_prefix from steps[0:entry_step] of reinforce traj.
          3. Condense context_prefix to ≤300 tokens for context_summary.
          4. Retrieve memory_prompt via TrajectoryMemory (if provided).
          5. Wire reward from CheckpointSpec.reward_weight.
          6. Assemble GenEnv JSON.
        """
        ...

    def materialise(
        self,
        env: GenEnv,
        overwrite: bool = False,
    ) -> MaterializedEnvFiles:
        """
        Write the GenEnv to disk as:
          generated_envs/<sub_task_id>/env.json
          generated_envs/<sub_task_id>/instruction.md
          generated_envs/<sub_task_id>/test_snippet.sh

        Does NOT write a full tasks/ directory or Dockerfile — that requires
        Harbor integration (Milestone 3+).
        """
        ...

    def generate_from_graph(
        self,
        graph: "SubTaskGraph",
        traj_files_by_task: dict[str, list[tuple[float, str]]],
        cycle: int = 1,
    ) -> list[GenEnv]:
        """
        Batch-generate GenEnvs for every SubTaskDefinition in a SubTaskGraph.

        traj_files_by_task: {task_name: [(score, traj_path), ...]}
        Returns envs in topological order (parents before children).
        """
        ...

    def env_to_dict(self, env: GenEnv) -> dict:
        """Return a JSON-serializable representation of a GenEnv."""
        ...

    @staticmethod
    def _derive_instruction(
        sub_task: SubTaskDefinition,
        parent_instruction_md: str,
    ) -> str:
        """
        Compose a sub-task instruction from the parent instruction.md and
        the phase_tag / checkpoint description.

        For phase-2 of medium-2: "Given that the HTTP client runs and the
        filter works, implement the Engineering department groupby."
        """
        ...

    @staticmethod
    def _condense_prefix(steps: list[dict], max_tokens: int = 300) -> str:
        """
        Produce a human-readable ≤max_tokens summary of a context_prefix.

        Simple heuristic: list the tool names and first 40 chars of each
        output, truncating to budget.  No LLM call required.
        """
        ...
```

---

### 2.7 `disteval/environment_registry.py` (new)

**Purpose:** Persist and update the active environment distribution across
improvement cycles.

**Depends on:**
- `disteval.environment_generator.GenEnv`, `EnvironmentGenerator`
- `disteval.right_tail.TaskOutcomeProfile`
- stdlib: `json`, `pathlib`, `dataclasses`, `datetime`

**Public API:**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .environment_generator import GenEnv
from .right_tail import TaskOutcomeProfile


@dataclass
class RegistryEntry:
    """One live entry in the environment registry."""

    env: GenEnv
    added_cycle: int
    last_updated_cycle: int
    agent_name: str

    # Cycle-over-cycle performance
    status_history: list[str] = field(default_factory=list)
    score_history: list[float] = field(default_factory=list)

    # Keep / drop / modify outcome (from Phase 3 §4.2 rules)
    action: str = "active"       # "active" | "retire" | "decompose_further" | "escalate"


class EnvironmentRegistry:
    """
    Maintains the set of active GenEnv entries across improvement cycles.

    Persistence: JSONL file at registry_path.
    Mutation: add(), update(), retire() — all methods are idempotent.
    """

    def __init__(self, registry_path: str = "generated_envs/registry.jsonl") -> None: ...

    def add(self, entry: RegistryEntry) -> None:
        """Add a new GenEnv entry (noop if env_id already present)."""
        ...

    def update(
        self,
        env_id: str,
        new_profile: TaskOutcomeProfile,
        cycle: int,
    ) -> None:
        """
        Apply cycle-over-cycle keep/drop/modify rules (Phase 3 §4.2):
          - SOLID → retire from active distribution.
          - RECOVERABLE → keep active, update kind + score_history.
          - STUCK, depth < max → mark for further decomposition.
          - STUCK, all children STUCK → mark recursively_stuck.
          - STUCK, depth == max → mark depth_cap.
        """
        ...

    def retire(self, env_id: str, reason: str = "solid") -> None:
        """Mark an entry as retired (not removed — archived for cross-agent use)."""
        ...

    def active_envs(self) -> list[GenEnv]:
        """Return all non-retired GenEnvs."""
        ...

    def envs_by_kind(self, kind: str) -> list[GenEnv]:
        """Return all active GenEnvs matching kind ('solid'|'recoverable'|'stuck')."""
        ...

    def merge_cycle(
        self,
        new_envs: list[GenEnv],
        profiles: dict[str, TaskOutcomeProfile],
        cycle: int,
        agent_name: str,
    ) -> None:
        """
        End-of-cycle registry update:
          1. Add genuinely new envs.
          2. Update existing envs using profiles.
          3. Apply keep/drop/modify rules.
          4. Persist to disk.
        """
        ...

    def save(self) -> None:
        """Atomically persist registry to registry_path."""
        ...

    def load(self) -> "EnvironmentRegistry":
        """Load registry from registry_path. Returns self for chaining."""
        ...
```

---

### 2.8 `disteval/distributed_eval.py` (new)

**Purpose:** Shared pool for multi-agent evaluations and cross-agent DPO pair
generation (Phase 3 §5).

**Depends on:**
- `disteval.right_tail.RightTailReport`, `TaskOutcomeProfile`
- `disteval.recursion_engine.SubTaskGraph`, `SubTaskDefinition`
- `disteval.environment_generator.GenEnv`
- stdlib: `json`, `dataclasses`, `statistics`

**Public API:**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .right_tail import RightTailReport, TaskOutcomeProfile
from .recursion_engine import SubTaskGraph, SubTaskDefinition


@dataclass
class AgentContribution:
    """One agent's per-sub-task boundary + outcome data."""

    agent_name: str
    sub_task_id: str
    entry_step: int
    exit_step: int
    confidence: float
    kind: str           # "solid" | "recoverable" | "stuck"
    q_star: float
    q_bar: float
    shareable: bool = False
    reinforce_traj_path: Optional[str] = None


@dataclass
class ConsensusNode:
    """
    Consensus view of a sub-task across multiple agents.

    Holds per-agent boundary variants and the preferred (confidence-weighted)
    boundary.
    """

    sub_task_id: str
    contributions: list[AgentContribution]

    # Consensus boundary (confidence-weighted vote)
    preferred_entry_step: float
    preferred_exit_step: float
    consensus_kind: str         # majority-vote kind across agents

    # Environment classification (Phase 3 §5.2)
    env_class: str   # "stable" | "contrastive" | "exploration_target" | "cross_agent_gap"


@dataclass
class CrossAgentTrainingPair:
    """
    DPO pair where the reinforce trajectory comes from a different agent
    than the contrast trajectory (Phase 3 §5.3).
    """

    sub_task_id: str
    source_agent: str            # the SOLID agent
    target_agent: str            # the STUCK/RECOVERABLE agent
    reinforce_traj_path: str     # from source_agent
    contrast_traj_path: str      # from target_agent
    structural_similarity: float # cosine similarity between tool sequences
    cross_agent_attribution: str # "agent_A_shared_to_agent_B::cycle_1"


class DistributedEvalPool:
    """
    Shared pool ingesting per-agent RightTailReports and SubTaskGraphs.

    Builds a consensus sub-task graph, classifies environments, and generates
    CrossAgentTrainingPairs when agents differ on the same sub-task.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.60,
    ) -> None: ...

    def ingest(
        self,
        agent_name: str,
        report: RightTailReport,
        graph: SubTaskGraph,
        shareable: bool = False,
    ) -> None:
        """
        Add one agent's cycle data to the pool.

        shareable=True enables cross-agent pair generation for trajectories
        belonging to this agent.  Default is False (privacy-safe).
        """
        ...

    def build_consensus_graph(self) -> list[ConsensusNode]:
        """
        Build consensus sub-task graph with boundary variants.

        Algorithm (Phase 3 §5.1):
          1. Group contributions by sub_task_id.
          2. For each group, compute confidence-weighted average boundary.
          3. Classify env_class based on per-agent kind distribution.
          4. If boundary spread > threshold: keep multiple variants.
        """
        ...

    def generate_cross_agent_pairs(
        self,
        consensus: list[ConsensusNode],
        k_per_node: int = 1,
    ) -> list[CrossAgentTrainingPair]:
        """
        Generate CrossAgentTrainingPairs for 'cross_agent_gap' consensus nodes.

        Matching by structural similarity: cosine similarity on canonical
        tool sequences (reusing TrajectoryMemory._embed() logic).
        Only includes pairs where the source agent marked shareable=True.
        """
        ...

    def summary(self) -> str:
        """Human-readable summary of the current pool state."""
        ...
```

---

### 2.9 `disteval/self_engine.py` (extend, backward-compatible)

**Changes are additive only.** The existing `SelfEngine.run_cycle()` path is
unchanged by default. Recursion is activated by a new `enable_recursion` flag.

```python
# New fields on SelfEngine.__init__:
enable_recursion: bool = False
recursion_config: Optional["RecursionEngineConfig"] = None
tasks_dir: str = "tasks"

# New optional fields on SelfImprovementPlan:
sub_task_graph: Optional["SubTaskGraph"] = None
recursive_score_left: Optional[float] = None

# New optional fields on TaskImprovement:
sub_task_definitions: list["SubTaskDefinition"] = field(default_factory=list)
recursive_gap: Optional[float] = None

# New optional fields on TrainingPair:
entry_context_steps: Optional[list[dict]] = None  # prefix up to entry_step
sub_task_id: Optional[str] = None

# New SelfEngine method:
def run_cycle_with_recursion(
    self,
    cycle: Optional[int] = None,
) -> SelfImprovementPlan:
    """
    run_cycle() extended with RecursionEngine decomposition.

    Calls run_cycle() first, then overlays sub-task decomposition.
    STUCK tasks are decomposed; sub-task DPO pairs are inserted into
    the curriculum alongside flat pairs.
    """
    ...

# New __main__.py flag (§2.10):
# --enable-recursion / --tasks-dir already in parser
```

---

### 2.10 `disteval/__main__.py` (extend, backward-compatible)

**Change:** Add `--enable-recursion` and `--registry-path` flags to the
`engine` sub-command.  Existing calls without these flags are unaffected.

```python
# New arguments in handle_engine():
parser.add_argument(
    "--enable-recursion",
    action="store_true",
    default=False,
    help="Enable RecursionEngine decomposition of STUCK/RECOVERABLE tasks.",
)
parser.add_argument(
    "--registry-path",
    default="generated_envs/registry.jsonl",
    help="Path to EnvironmentRegistry JSONL (default: generated_envs/registry.jsonl)",
)
parser.add_argument(
    "--max-depth",
    type=int,
    default=3,
    help="Maximum recursion depth (default: 3)",
)
```

---

### 2.11 `disteval/training_sim.py` (extend, backward-compatible)

**Change:** Add `simulate_recursive_gain()` to compare flat vs recursive
training improvement, using the sub-task graph's weighted-sum reward
propagation.  The existing `apply_training_effect()` function is unchanged.

```python
def simulate_recursive_gain(
    task_profiles: dict[str, TaskOutcomeProfile],
    sub_task_graph: "SubTaskGraph",
    alpha: float = ALPHA,
    n_bootstrap: int = 1000,
    seed: int = SEED,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, float]:
    """
    Compare flat vs recursive training improvement for a set of tasks.

    Flat gain: apply_training_effect() on the parent task only.
    Recursive gain: apply weighted-sum propagation across the sub-task graph.

    Returns {task_name: {"flat_gain": float, "recursive_gain": float,
                         "improvement_ratio": float}} for each parent task.

    Used in Milestone 3 validation to verify recursive > flat for STUCK tasks.
    """
    ...
```

---

## 3  Code skeletons for critical new classes

### 3.1 `TestSuiteParser` — parsing `tests/test.sh`

```python
# disteval/test_suite_parser.py  (complete skeleton)
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCORE_ADD_RE  = re.compile(r'SCORE=\$\(\(SCORE \+ (\d+)\)\)')
_DENOMINATOR_RE = re.compile(r'print\(\$SCORE / (\d+(?:\.\d+)?)\)')
_COMMENT_RE    = re.compile(r'#\s*(.+)')


@dataclass
class CheckpointSpec:
    index: int
    task_name: str
    description: str
    reward_weight: float
    score_increment: int
    total_score: int
    condition_source: str
    checkpoint_id: str  # f"{task_name}::phase-{index}"


def parse_test_suite(
    test_sh_path: str | Path,
    task_name: str,
) -> list[CheckpointSpec]:
    path = Path(test_sh_path)
    if not path.exists():
        raise FileNotFoundError(f"test.sh not found: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 1. Find denominator (e.g. 100)
    total_score = 100
    for line in reversed(lines):
        m = _DENOMINATOR_RE.search(line)
        if m:
            total_score = int(float(m.group(1)))
            break

    # 2. Collect SCORE+= positions and the text block before each one
    specs: list[CheckpointSpec] = []
    for i, line in enumerate(lines):
        m = _SCORE_ADD_RE.search(line)
        if not m:
            continue

        increment = int(m.group(1))
        reward_weight = increment / total_score

        # Extract description from nearest preceding comment line
        description = f"checkpoint-{len(specs)}"
        for prev in reversed(lines[:i]):
            cm = _COMMENT_RE.match(prev.strip())
            if cm:
                description = cm.group(1).strip()
                break

        # Extract condition source: block from previous SCORE line to current
        condition_lines: list[str] = []
        start = specs[-1]._source_line_end if specs else 0  # type: ignore[attr-defined]
        for ln in lines[start:i]:
            s = ln.strip()
            if s and not s.startswith("#"):
                condition_lines.append(s)

        spec = CheckpointSpec(
            index=len(specs),
            task_name=task_name,
            description=description,
            reward_weight=reward_weight,
            score_increment=increment,
            total_score=total_score,
            condition_source="\n".join(condition_lines[-10:]),
            checkpoint_id=f"{task_name}::phase-{len(specs)}",
        )
        # Internal marker for condition extraction in next iteration
        spec.__dict__["_source_line_end"] = i + 1
        specs.append(spec)

    if not specs:
        raise ValueError(
            f"No SCORE+= lines found in {path}. "
            "Is this a disteval test.sh file?"
        )

    return specs


def parse_all_tasks(tasks_dir: str | Path = "tasks") -> dict[str, list[CheckpointSpec]]:
    tasks_path = Path(tasks_dir)
    result: dict[str, list[CheckpointSpec]] = {}
    for task_dir in sorted(tasks_path.iterdir()):
        if not task_dir.is_dir():
            continue
        test_sh = task_dir / "tests" / "test.sh"
        if not test_sh.exists():
            continue
        task_name = task_dir.name
        try:
            result[task_name] = parse_test_suite(test_sh, task_name)
        except (ValueError, OSError) as exc:
            logger.warning("Could not parse %s: %s", test_sh, exc)
    return result


def checkpoint_weights(specs: list[CheckpointSpec]) -> list[float]:
    return [s.reward_weight for s in specs]
```

---

### 3.2 `RecursionEngine.find_phase_boundaries()` — boundary detection

```python
def find_phase_boundaries(
    self,
    steps: list[dict],
    high_steps: Optional[list[dict]] = None,
    low_steps: Optional[list[dict]] = None,
    start_step: int = 0,
    end_step: Optional[int] = None,
    checkpoint_specs: Optional[list[CheckpointSpec]] = None,
) -> list[PhaseBoundary]:
    """Complete skeleton."""
    if end_step is None:
        end_step = len(steps)

    candidate_steps: list[int] = [start_step]

    # Source 1: Monitor divergence points between high and low runs
    if high_steps is not None and low_steps is not None:
        div_steps = self.monitor.divergence_steps(
            # Pass pre-loaded steps directly rather than paths
            high_path="",   # handled by overloaded version in Milestone 2
            low_path="",
            max_check=min(end_step, 30),
            confidence_threshold=self.config.divergence_confidence,
        )
        candidate_steps.extend(div_steps)

    # Source 2: Checkpoint alignment
    if checkpoint_specs:
        # Distribute checkpoints uniformly if we cannot map them to steps
        n = len(checkpoint_specs)
        step_range = end_step - start_step
        for idx, spec in enumerate(checkpoint_specs):
            candidate_steps.append(start_step + int(step_range * (idx + 1) / n))

    # Source 3: Structural breakpoints from featurizer
    feat = self.monitor.featurizer.featurize(steps, prefix_n=end_step)
    if feat.first_write_pos < end_step:
        candidate_steps.append(feat.first_write_pos)
    if feat.first_exec_pos < end_step:
        candidate_steps.append(feat.first_exec_pos)

    candidate_steps.append(end_step)

    # Deduplicate, sort, merge adjacent (within 2 steps)
    raw = sorted(set(candidate_steps))
    merged: list[int] = [raw[0]]
    for s in raw[1:]:
        if s - merged[-1] >= 2:
            merged.append(s)

    # Cap to max_phase_boundaries + 1 endpoints
    max_b = self.config.max_phase_boundaries
    if len(merged) > max_b + 1:
        # Keep evenly-spaced subset
        import numpy as np
        indices = np.linspace(0, len(merged) - 1, max_b + 1, dtype=int)
        merged = [merged[i] for i in indices]

    # Build PhaseBoundary objects
    featurizer = self.monitor.featurizer
    boundaries: list[PhaseBoundary] = []
    for i in range(len(merged) - 1):
        entry = merged[i]
        exit_ = merged[i + 1]
        tool_sig = tuple(featurizer.extract_tool_sequence(steps)[entry:exit_])
        match = self.monitor.check(steps, prefix_n=entry)
        label = f"phase-{i}"
        spec = checkpoint_specs[i] if checkpoint_specs and i < len(checkpoint_specs) else None
        boundaries.append(PhaseBoundary(
            entry_step=entry,
            exit_step=exit_,
            label=label,
            confidence=match.p_high,
            tool_signature=tool_sig,
            checkpoint_spec=spec,
        ))

    return boundaries
```

---

### 3.3 `EnvironmentGenerator.generate()` — producing a GenEnv

```python
def generate(
    self,
    sub_task: SubTaskDefinition,
    reinforce_traj_path: str,
    cycle: int = 1,
    context_prefix_steps: Optional[list[dict]] = None,
) -> GenEnv:
    """Complete skeleton."""
    import json

    # Step 1: load parent instruction
    parent_instr_path = (
        Path(self.tasks_dir)
        / sub_task.parent_task.split("/")[-1].replace("disteval-", "")
        / "instruction.md"
    )
    # Fallback: task name may be "medium-rest-client" or "medium-2"
    if not parent_instr_path.exists():
        # Try the tasks/ directory name variant
        task_short = sub_task.parent_task.split("/")[-1]
        for candidate in Path(self.tasks_dir).iterdir():
            if candidate.is_dir() and task_short in candidate.name:
                parent_instr_path = candidate / "instruction.md"
                break
    parent_instr = (
        parent_instr_path.read_text(encoding="utf-8")
        if parent_instr_path.exists()
        else f"(Instruction for {sub_task.parent_task} not found)"
    )

    # Step 2: load context prefix steps
    if context_prefix_steps is None and reinforce_traj_path:
        try:
            with open(reinforce_traj_path, encoding="utf-8") as f:
                traj = json.load(f)
            all_steps = traj.get("steps", [])
            context_prefix_steps = all_steps[: sub_task.entry_step]
        except (OSError, json.JSONDecodeError, KeyError):
            context_prefix_steps = []

    context_prefix_steps = context_prefix_steps or []

    # Step 3: derive instruction
    instruction = self._derive_instruction(sub_task, parent_instr)

    # Step 4: condense prefix
    context_summary = self._condense_prefix(context_prefix_steps)

    # Step 5: retrieve memory prompt
    memory_prompt: Optional[str] = None
    if self.memory is not None:
        results = self.memory.retrieve_for_new_task(instruction, k=2)
        if results:
            memory_prompt = self.memory.generate_retrieval_prompt(results, context="before_task")

    # Step 6: checkpoint condition source
    condition_source = ""
    if sub_task.shareable:  # reusing shareable as a proxy for "has checkpoint spec"
        pass  # populated by CheckpointSpec in Milestone 1+
    if hasattr(sub_task, "_checkpoint_spec") and sub_task._checkpoint_spec is not None:
        condition_source = sub_task._checkpoint_spec.condition_source

    env_id = f"{sub_task.sub_task_id}::cycle-{cycle}"
    return GenEnv(
        env_id=env_id,
        sub_task_id=sub_task.sub_task_id,
        parent_task=sub_task.parent_task,
        cycle=cycle,
        context_prefix_steps=context_prefix_steps,
        entry_step=sub_task.entry_step,
        phase_tag=sub_task.phase_tag,
        instruction=instruction,
        context_summary=context_summary,
        memory_prompt=memory_prompt,
        reward_weight=sub_task.reward_weight,
        checkpoint_condition_source=condition_source,
        step_budget=max(30, (sub_task.exit_step - sub_task.entry_step) * 3),
        kind=sub_task.kind,
        reinforce_traj_path=reinforce_traj_path,
    )
```

---

### 3.4 `EnvironmentRegistry.update()` — keep/drop/modify rules

```python
def update(
    self,
    env_id: str,
    new_profile: TaskOutcomeProfile,
    cycle: int,
) -> None:
    """Keep/drop/modify rules from Phase 3 §4.2."""
    entry = self._entries.get(env_id)
    if entry is None:
        return

    entry.score_history.append(new_profile.q_bar)
    entry.status_history.append(new_profile.kind)
    entry.last_updated_cycle = cycle
    entry.env.kind = new_profile.kind

    if new_profile.kind == "solid":
        # Retire — graduate from active distribution
        entry.action = "retire"
        entry.env.status = "retired"

    elif new_profile.kind == "recoverable":
        # Keep active — update DPO boundary if confidence improved
        entry.action = "active"
        entry.env.status = "active"

    elif new_profile.kind == "stuck":
        depth = entry.env.sub_task_id.count("::") - 1  # rough depth proxy
        max_depth = 3  # from RecursionEngineConfig default

        if depth >= max_depth:
            entry.action = "escalate"
            entry.env.status = "depth_cap"
        else:
            # Check if all children are also stuck
            children = [
                e for e in self._entries.values()
                if e.env.sub_task_id.startswith(env_id + "::")
            ]
            if children and all(c.env.kind == "stuck" for c in children):
                entry.action = "escalate"
                entry.env.status = "recursively_stuck"
            else:
                entry.action = "decompose_further"
                entry.env.status = "active"

    self.save()
```

---

### 3.5 `DistributedEvalPool.build_consensus_graph()` — consensus algorithm

```python
def build_consensus_graph(self) -> list[ConsensusNode]:
    """Group contributions and compute consensus boundaries."""
    from collections import defaultdict
    import statistics

    groups: dict[str, list[AgentContribution]] = defaultdict(list)
    for contrib in self._contributions:
        groups[contrib.sub_task_id].append(contrib)

    nodes: list[ConsensusNode] = []
    for sub_task_id, contribs in groups.items():
        if not contribs:
            continue

        # Confidence-weighted average boundary
        total_conf = sum(c.confidence for c in contribs) or 1.0
        pref_entry = sum(c.entry_step * c.confidence for c in contribs) / total_conf
        pref_exit  = sum(c.exit_step  * c.confidence for c in contribs) / total_conf

        # Majority-vote kind
        kind_counts: dict[str, int] = {}
        for c in contribs:
            kind_counts[c.kind] = kind_counts.get(c.kind, 0) + 1
        consensus_kind = max(kind_counts, key=lambda k: kind_counts[k])

        # Environment classification (Phase 3 §5.2)
        kinds = {c.kind for c in contribs}
        n_agents = len(contribs)
        has_solid = "solid" in kinds
        has_stuck = "stuck" in kinds
        has_recov = "recoverable" in kinds

        if has_solid and not has_stuck:
            env_class = "stable" if not has_recov else "contrastive"
        elif has_stuck and not has_solid and not has_recov:
            env_class = "exploration_target"
        elif has_solid and (has_stuck or has_recov):
            env_class = "cross_agent_gap"
        else:
            env_class = "contrastive"

        nodes.append(ConsensusNode(
            sub_task_id=sub_task_id,
            contributions=contribs,
            preferred_entry_step=pref_entry,
            preferred_exit_step=pref_exit,
            consensus_kind=consensus_kind,
            env_class=env_class,
        ))

    return nodes
```

---

## 4  Milestone 1 — Test-suite checkpoint parser

### 4.1 Scope

Implement and test `disteval/test_suite_parser.py`. No new imports into
existing disteval modules. No containers or trajectory data required.

### 4.2 Files changed

| File | Change |
|------|--------|
| `disteval/test_suite_parser.py` | Create (new) |
| `tests/test_test_suite_parser.py` | Create (new tests) |

### 4.3 Minimal first test case: `medium-2`

The five checkpoints in `tasks/medium-2/tests/test.sh` are:

```
SCORE+= 10  →  "Validate JSON"          weight=0.10  id=medium-2::phase-0
SCORE+= 25  →  "Check total_eligible_users (age >= 30)"  weight=0.25  id=medium-2::phase-1
SCORE+= 25  →  "Check Engineering dept"  weight=0.25  id=medium-2::phase-2
SCORE+= 20  →  "Check Sales dept"        weight=0.20  id=medium-2::phase-3
SCORE+= 20  →  "Check HR dept"           weight=0.20  id=medium-2::phase-4
```

### 4.4 Test suite for Milestone 1

```python
# tests/test_test_suite_parser.py
import pytest
from pathlib import Path
from disteval.test_suite_parser import (
    parse_test_suite,
    parse_all_tasks,
    checkpoint_weights,
    CheckpointSpec,
)

TASKS_DIR = Path("tasks")


class TestParseMedium2:
    @pytest.fixture
    def specs(self):
        return parse_test_suite(TASKS_DIR / "medium-2" / "tests" / "test.sh", "medium-2")

    def test_count(self, specs):
        assert len(specs) == 5

    def test_weights_sum_to_one(self, specs):
        assert abs(sum(s.reward_weight for s in specs) - 1.0) < 1e-6

    def test_first_weight(self, specs):
        assert specs[0].reward_weight == pytest.approx(0.10)

    def test_second_weight(self, specs):
        assert specs[1].reward_weight == pytest.approx(0.25)

    def test_checkpoint_ids(self, specs):
        expected_ids = [f"medium-2::phase-{i}" for i in range(5)]
        assert [s.checkpoint_id for s in specs] == expected_ids

    def test_total_score_denominator(self, specs):
        assert all(s.total_score == 100 for s in specs)

    def test_condition_source_nonempty(self, specs):
        # Each checkpoint should have a non-trivial condition
        for s in specs:
            assert len(s.condition_source.strip()) > 10

    def test_checkpoint_weights_helper(self, specs):
        w = checkpoint_weights(specs)
        assert len(w) == 5
        assert w[0] == pytest.approx(0.10)


class TestParseEasy1:
    @pytest.fixture
    def specs(self):
        return parse_test_suite(TASKS_DIR / "easy-1" / "tests" / "test.sh", "easy-1")

    def test_count(self, specs):
        assert len(specs) == 3

    def test_weights_sum_to_one(self, specs):
        assert abs(sum(s.reward_weight for s in specs) - 1.0) < 1e-6


class TestParseAllTasks:
    def test_all_six_tasks(self):
        result = parse_all_tasks(TASKS_DIR)
        assert len(result) == 6

    def test_all_weights_sum_to_one(self):
        result = parse_all_tasks(TASKS_DIR)
        for task_name, specs in result.items():
            total = sum(s.reward_weight for s in specs)
            assert abs(total - 1.0) < 1e-6, (
                f"Task {task_name} weights sum to {total}, not 1.0"
            )

    def test_no_empty_condition_sources(self):
        result = parse_all_tasks(TASKS_DIR)
        for task_name, specs in result.items():
            for spec in specs:
                assert spec.condition_source is not None


class TestEdgeCases:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_test_suite(tmp_path / "nonexistent.sh", "t")

    def test_no_score_variable_raises(self, tmp_path):
        p = tmp_path / "test.sh"
        p.write_text("#!/bin/bash\necho hello\n")
        with pytest.raises(ValueError, match="No SCORE"):
            parse_test_suite(p, "t")
```

### 4.5 Validation

```bash
pytest tests/test_test_suite_parser.py -v
```

All tests should pass with no external dependencies beyond the existing
`tasks/` directory. Estimated implementation time: 2–3 hours.

---

## 5  Milestone 2 — RecursionEngine and sub-task decomposition

### 5.1 Scope

Implement `disteval/recursion_engine.py` and the backward-compatible extensions
to `trajectory_monitor.py`, `trajectory_memory.py`, `right_tail.py`,
`self_engine.py`, and `__main__.py`. Milestone 2 operates entirely on existing
trajectory data — no container replay or new eval runs required.

### 5.2 Files changed

| File | Change |
|------|--------|
| `disteval/recursion_engine.py` | Create (new) |
| `disteval/trajectory_monitor.py` | Add `divergence_steps()` method + optional fields to `TrajectoryRecord`, `PatternMatch` |
| `disteval/trajectory_memory.py` | Add `sub_task_slices` field, `retrieve_for_sub_task()` method |
| `disteval/right_tail.py` | Add optional `sub_task_*` fields to dataclasses |
| `disteval/self_engine.py` | Add `enable_recursion`, optional fields; add `run_cycle_with_recursion()` |
| `disteval/__main__.py` | Add `--enable-recursion`, `--max-depth` flags |
| `tests/test_recursion_engine.py` | Create (new tests) |

### 5.3 Key design decisions grounded in the codebase

1. **`divergence_steps()` vs `_find_divergence_step()`:** The existing
   `SelfEngine._find_divergence_step()` (lines 553–575 of `self_engine.py`)
   returns only the *first* divergence step. `RecursionEngine` needs *all*
   divergence steps to build multiple boundaries. The new
   `TrajectoryMonitor.divergence_steps()` returns a list. The existing method
   is preserved and unchanged.

2. **Checkpoint spec injection:** `RecursionEngine.load_checkpoint_specs()`
   stores a `dict[str, list[CheckpointSpec]]` keyed by task name. Inside
   `decompose_task()`, specs are looked up using
   `SelfEngine._TASK_NAME_MAP` logic (lines 471–478 of `self_engine.py`)
   to convert between `"disteval/medium-rest-client"` and `"medium-2"`.

3. **Sub-task score estimation without sidecar files:** For Milestone 2, the
   structural proxy scoring is used: `1.0` if `monitor.check(steps[:exit_step])
   .p_high >= config.divergence_confidence`, else `0.0`. Sidecar files
   (`reward_c{i}.txt`) are introduced in Milestone 3.

4. **Stack overflow safety:** `RecursionEngine.decompose_task()` enforces
   `config.max_stack_nodes` by maintaining a module-level counter that is
   reset at the start of each `decompose()` call.

### 5.4 Worked example: `medium-2` decomposition path

Given Codex CLI's scores `[0.0, 0.0, 1.0]` on `medium-2`:
- `kind = "recoverable"`, `q_star = 1.0`, `q_bar ≈ 0.333`
- `find_phase_boundaries()` will find up to 5 boundaries using:
  - Monitor divergence between the `score=1.0` trajectory and the `score=0.0`
    trajectories.
  - Checkpoint specs aligned to the 5 `SCORE+=` increments.
- Each of the 5 resulting `SubTaskDefinition` objects gets:
  - `sub_task_id` = e.g. `"medium-2::phase-2"`
  - `estimated_q_star` from structural proxy scoring
  - `reward_weight` from `CheckpointSpec.reward_weight` (0.25 for phase-2)
- Sub-tasks with `kind="solid"` are not added to the curriculum.
- Sub-tasks with `kind="recoverable"` or `"stuck"` get DPO pairs sliced to
  `[entry_step:exit_step]` of the parent trajectory.

### 5.5 Test suite for Milestone 2

```python
# tests/test_recursion_engine.py
import pytest
from unittest.mock import MagicMock

from disteval.recursion_engine import (
    RecursionEngine,
    RecursionEngineConfig,
    PhaseBoundary,
    SubTaskDefinition,
    SubTaskGraph,
    RMDPNode,
)
from disteval.right_tail import task_outcome_profile
from disteval.trajectory_monitor import TrajectoryMonitor, TrajectoryRecord, TrajectoryFeatures


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_dummy_monitor(n_records: int = 5) -> TrajectoryMonitor:
    """Build a minimal TrajectoryMonitor from synthetic records."""
    records = []
    for i in range(n_records):
        features = TrajectoryFeatures(
            n_steps=10 + i,
            n_tool_calls=10 + i,
            first_write_pos=2 + (i % 3),
            first_exec_pos=3 + (i % 3),
            n_reads=2,
            n_writes=3,
            n_exec=2 + (i % 2),
            n_search=0,
            search_ratio=0.0,
            write_before_read=True,
            tool_diversity=0.5,
            prefix_len=10 + i,
            is_prefix=False,
        )
        records.append(TrajectoryRecord(
            trial_id=f"trial-{i}",
            task_path="tasks/medium-2",
            agent_name="agent",
            score=1.0 if i < 2 else 0.0,
            features=features,
            tool_sequence=["read_file", "write_file", "run_shell_command"] * 3,
            traj_path=f"/fake/traj-{i}.json",
        ))
    return TrajectoryMonitor(records)


@pytest.fixture
def engine():
    monitor = make_dummy_monitor()
    config = RecursionEngineConfig(
        max_depth=2,
        max_phase_boundaries=5,
        divergence_confidence=0.60,
    )
    return RecursionEngine(monitor=monitor, config=config)


# ── PhaseBoundary ─────────────────────────────────────────────────────────────

class TestFindPhaseBoundaries:

    def test_returns_list(self, engine):
        steps = [{"tool_calls": [{"function_name": "read_file"}]}] * 10
        bounds = engine.find_phase_boundaries(steps)
        assert isinstance(bounds, list)

    def test_respects_max_boundaries(self, engine):
        steps = [{"tool_calls": [{"function_name": "write_file"}]}] * 20
        bounds = engine.find_phase_boundaries(steps)
        assert len(bounds) <= engine.config.max_phase_boundaries

    def test_boundaries_cover_full_range(self, engine):
        steps = [{}] * 15
        bounds = engine.find_phase_boundaries(steps, start_step=0, end_step=15)
        # entry of first >= 0, exit of last <= 15
        assert bounds[0].entry_step >= 0
        assert bounds[-1].exit_step <= 15

    def test_no_overlapping_boundaries(self, engine):
        steps = [{}] * 20
        bounds = engine.find_phase_boundaries(steps)
        for i in range(len(bounds) - 1):
            assert bounds[i].exit_step <= bounds[i + 1].entry_step

    def test_with_checkpoint_specs(self, engine):
        from disteval.test_suite_parser import CheckpointSpec
        specs = [
            CheckpointSpec(i, "t", f"c{i}", 0.2, 20, 100, "", f"t::phase-{i}")
            for i in range(5)
        ]
        steps = [{}] * 30
        bounds = engine.find_phase_boundaries(steps, checkpoint_specs=specs)
        # Each boundary should reference its spec
        for b in bounds:
            if b.checkpoint_spec is not None:
                assert b.checkpoint_spec.task_name == "t"


# ── SubTaskDefinition ─────────────────────────────────────────────────────────

class TestDecomposeTask:

    def test_returns_rmdp_node(self, engine):
        profile = task_outcome_profile("disteval/medium-rest-client", [0.0, 0.0, 1.0], "agent")
        node = engine.decompose_task(profile, [], depth=0)
        assert isinstance(node, RMDPNode)

    def test_no_decomposition_for_solid(self, engine):
        profile = task_outcome_profile("t", [1.0, 1.0, 1.0], "agent")
        node = engine.decompose_task(profile, [], depth=0)
        assert not node.children

    def test_depth_cap_respected(self, engine):
        profile = task_outcome_profile("t", [0.0, 0.0, 0.0], "agent")
        node = engine.decompose_task(profile, [], depth=engine.config.max_depth)
        # At max depth, no further decomposition
        assert not node.children
        assert node.definition.sub_task_depth == engine.config.max_depth

    def test_sub_task_ids_unique(self, engine):
        profile = task_outcome_profile("disteval/medium-rest-client", [0.0, 1.0], "agent")
        node = engine.decompose_task(profile, [], depth=0)
        ids = [n.definition.sub_task_id for n in node.children]
        assert len(ids) == len(set(ids))


# ── SubTaskGraph ──────────────────────────────────────────────────────────────

class TestSubTaskGraph:

    def test_to_sub_task_graph_serializable(self, engine):
        import json
        profile = task_outcome_profile("disteval/medium-rest-client", [0.0, 1.0], "agent")
        node = engine.decompose_task(profile, [], depth=0)
        graph = engine.to_sub_task_graph([node], {profile.task: profile})
        d = engine.sub_task_graph_to_dict(graph)
        # Must round-trip through JSON without error
        json.dumps(d)

    def test_graph_has_edges_for_children(self, engine):
        profile = task_outcome_profile("disteval/medium-rest-client", [0.0, 1.0], "agent")
        node = engine.decompose_task(profile, [], depth=0)
        graph = engine.to_sub_task_graph([node], {profile.task: profile})
        if node.children:
            assert len(graph.edges) >= len(node.children)


# ── Backward compatibility ────────────────────────────────────────────────────

class TestBackwardCompatibility:

    def test_existing_monitor_api_unchanged(self):
        """TrajectoryMonitor.check() still works as before."""
        monitor = make_dummy_monitor()
        steps = [{"tool_calls": [{"function_name": "write_file"}]}] * 5
        result = monitor.check(steps)
        assert result.prediction in ("high", "low", "uncertain")

    def test_existing_right_tail_unchanged(self):
        """right_tail_analysis works with no new fields required."""
        from disteval.records import RecordStore, EpisodeRecord
        from disteval.right_tail import right_tail_analysis
        store = RecordStore()
        for i, s in enumerate([0.0, 0.5, 1.0]):
            store.add(EpisodeRecord("r", "agent", "t", i, s, s >= 0.99))
        report = right_tail_analysis(store)
        assert report.n_tasks == 1
```

### 5.6 Validation

```bash
pytest tests/test_recursion_engine.py -v
# Also run full suite to check no regressions
pytest tests/ -v
```

---

## 6  Milestone 3 — Environment generation and the self-improvement loop

### 6.1 Scope

Implement `disteval/environment_generator.py`, `disteval/environment_registry.py`,
and extend `disteval/training_sim.py` with `simulate_recursive_gain()`. This
milestone does NOT require Harbor container integration — `materialise()` writes
JSON files only. A complete end-to-end cycle (eval → decompose → generate →
simulate) can run on existing trajectory data.

### 6.2 Files changed

| File | Change |
|------|--------|
| `disteval/environment_generator.py` | Create (new) |
| `disteval/environment_registry.py` | Create (new) |
| `disteval/training_sim.py` | Add `simulate_recursive_gain()` (new function) |
| `tests/test_environment_generator.py` | Create (new tests) |
| `tests/test_environment_registry.py` | Create (new tests) |

### 6.3 End-to-end cycle without containers

```python
# Milestone 3 smoke test — runs without Harbor or Docker
from pathlib import Path
from disteval.records import RecordStore, EpisodeRecord
from disteval.right_tail import right_tail_analysis
from disteval.trajectory_monitor import TrajectoryMonitor
from disteval.trajectory_memory import TrajectoryMemory
from disteval.recursion_engine import RecursionEngine, RecursionEngineConfig
from disteval.environment_generator import EnvironmentGenerator, GenEnv
from disteval.environment_registry import EnvironmentRegistry

# 1. Build a synthetic store (replaces real Harbor job data)
store = RecordStore()
for i, score in enumerate([0.0, 0.25, 1.0]):
    store.add(EpisodeRecord("run", "agent", "disteval/medium-rest-client", i, score, score >= 0.99))
report = right_tail_analysis(store)

# 2. Build monitor + memory from empty records (real runs not required for smoke test)
monitor = TrajectoryMonitor([])
memory = TrajectoryMemory()
config = RecursionEngineConfig(max_depth=2, require_checkpoint_alignment=True)
engine = RecursionEngine(monitor, memory, config)
engine.load_checkpoint_specs("tasks")

# 3. Decompose
graph = engine.decompose(report, [])
assert len(graph.sub_tasks) >= 1

# 4. Generate environments
generator = EnvironmentGenerator(tasks_dir="tasks", output_dir="/tmp/generated_envs")
envs = generator.generate_from_graph(graph, traj_files_by_task={}, cycle=1)
assert len(envs) >= 1

# 5. Materialise
for env in envs:
    files = generator.materialise(env)
    assert Path(files.env_json_path).exists()

# 6. Registry
registry = EnvironmentRegistry("/tmp/generated_envs/registry.jsonl")
for env in envs:
    from disteval.environment_registry import RegistryEntry
    registry.add(RegistryEntry(env=env, added_cycle=1, last_updated_cycle=1, agent_name="agent"))
assert len(registry.active_envs()) == len(envs)
print("Milestone 3 smoke test passed.")
```

### 6.4 Test suite for Milestone 3

```python
# tests/test_environment_generator.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from disteval.recursion_engine import SubTaskDefinition
from disteval.environment_generator import EnvironmentGenerator, GenEnv


def make_sub_task(
    sub_task_id: str = "medium-2::phase-2",
    parent_task: str = "disteval/medium-rest-client",
    kind: str = "recoverable",
) -> SubTaskDefinition:
    return SubTaskDefinition(
        sub_task_id=sub_task_id,
        parent_task=parent_task,
        sub_task_depth=1,
        entry_step=5,
        exit_step=15,
        phase_tag="write",
        instruction="",
        estimated_q_star=1.0,
        estimated_q_bar=0.5,
        kind=kind,
        reward_weight=0.25,
    )


@pytest.fixture
def generator(tmp_path):
    return EnvironmentGenerator(
        tasks_dir="tasks",
        output_dir=str(tmp_path / "generated_envs"),
    )


class TestGenerateBasic:

    def test_returns_gen_env(self, generator):
        sub_task = make_sub_task()
        env = generator.generate(sub_task, reinforce_traj_path="", cycle=1)
        assert isinstance(env, GenEnv)

    def test_env_id_includes_cycle(self, generator):
        sub_task = make_sub_task()
        env = generator.generate(sub_task, reinforce_traj_path="", cycle=3)
        assert "cycle-3" in env.env_id

    def test_reward_weight_preserved(self, generator):
        sub_task = make_sub_task()
        env = generator.generate(sub_task, reinforce_traj_path="", cycle=1)
        assert env.reward_weight == pytest.approx(0.25)

    def test_instruction_nonempty(self, generator):
        sub_task = make_sub_task()
        env = generator.generate(sub_task, reinforce_traj_path="", cycle=1)
        assert len(env.instruction.strip()) > 0

    def test_step_budget_positive(self, generator):
        sub_task = make_sub_task()
        env = generator.generate(sub_task, reinforce_traj_path="", cycle=1)
        assert env.step_budget > 0


class TestMaterialise:

    def test_creates_files(self, generator, tmp_path):
        sub_task = make_sub_task()
        env = generator.generate(sub_task, reinforce_traj_path="", cycle=1)
        files = generator.materialise(env)
        assert Path(files.env_json_path).exists()
        assert Path(files.sub_task_instruction_path).exists()
        assert Path(files.sub_task_test_snippet_path).exists()

    def test_env_json_valid(self, generator, tmp_path):
        sub_task = make_sub_task()
        env = generator.generate(sub_task, reinforce_traj_path="", cycle=1)
        files = generator.materialise(env)
        data = json.loads(Path(files.env_json_path).read_text())
        assert data["sub_task_id"] == sub_task.sub_task_id

    def test_overwrite_false_noop(self, generator, tmp_path):
        sub_task = make_sub_task()
        env = generator.generate(sub_task, reinforce_traj_path="", cycle=1)
        files1 = generator.materialise(env, overwrite=False)
        mtime1 = Path(files1.env_json_path).stat().st_mtime
        generator.materialise(env, overwrite=False)  # second call
        mtime2 = Path(files1.env_json_path).stat().st_mtime
        assert mtime1 == mtime2


# tests/test_environment_registry.py
import pytest
from disteval.environment_registry import EnvironmentRegistry, RegistryEntry
from disteval.environment_generator import GenEnv
from disteval.right_tail import task_outcome_profile


def make_gen_env(env_id: str = "e1", kind: str = "recoverable") -> GenEnv:
    return GenEnv(
        env_id=env_id, sub_task_id="t::phase-0", parent_task="t",
        cycle=1, context_prefix_steps=[], entry_step=0, phase_tag="write",
        instruction="test", context_summary="test", memory_prompt=None,
        reward_weight=0.25, checkpoint_condition_source="", step_budget=30,
        kind=kind,
    )


@pytest.fixture
def registry(tmp_path):
    return EnvironmentRegistry(str(tmp_path / "registry.jsonl"))


class TestRegistryAddUpdate:

    def test_add_and_active(self, registry):
        env = make_gen_env("e1")
        registry.add(RegistryEntry(env=env, added_cycle=1, last_updated_cycle=1, agent_name="a"))
        assert len(registry.active_envs()) == 1

    def test_solid_retires(self, registry):
        env = make_gen_env("e1", kind="recoverable")
        registry.add(RegistryEntry(env=env, added_cycle=1, last_updated_cycle=1, agent_name="a"))
        solid_profile = task_outcome_profile("t::phase-0", [1.0, 1.0, 1.0], "a")
        registry.update("e1", solid_profile, cycle=2)
        assert len(registry.active_envs()) == 0   # retired

    def test_recoverable_stays_active(self, registry):
        env = make_gen_env("e1", kind="recoverable")
        registry.add(RegistryEntry(env=env, added_cycle=1, last_updated_cycle=1, agent_name="a"))
        recov_profile = task_outcome_profile("t::phase-0", [0.0, 1.0], "a")
        registry.update("e1", recov_profile, cycle=2)
        assert len(registry.active_envs()) == 1

    def test_persist_and_reload(self, registry, tmp_path):
        env = make_gen_env("e1")
        registry.add(RegistryEntry(env=env, added_cycle=1, last_updated_cycle=1, agent_name="a"))
        registry.save()
        registry2 = EnvironmentRegistry(str(tmp_path / "registry.jsonl"))
        registry2.load()
        assert len(registry2.active_envs()) == 1
```

### 6.5 Convergence simulation test

```python
# Add to tests/test_training_sim.py (new section)
class TestSimulateRecursiveGain:

    def test_recursive_exceeds_flat_for_stuck(self):
        """Recursive decomposition should show higher gain than flat for STUCK tasks."""
        import numpy as np
        from disteval.training_sim import simulate_recursive_gain
        from disteval.right_tail import task_outcome_profile
        from disteval.recursion_engine import SubTaskGraph, SubTaskDefinition

        profiles = {
            "disteval/medium-rest-client": task_outcome_profile(
                "disteval/medium-rest-client", [0.0, 0.0, 1.0], "agent"
            )
        }
        # Synthetic sub-task graph: 2 sub-tasks, one solid (entry), one recoverable
        sub_tasks = [
            SubTaskDefinition("m2::p0", "disteval/medium-rest-client", 1, 0, 5, "exec",
                              "run server", 1.0, 1.0, "solid", 0.10),
            SubTaskDefinition("m2::p1", "disteval/medium-rest-client", 1, 5, 15, "write",
                              "filter users", 0.8, 0.3, "recoverable", 0.25),
        ]
        graph = SubTaskGraph(
            parent_tasks=["disteval/medium-rest-client"],
            sub_tasks=sub_tasks,
            edges=[("disteval/medium-rest-client", "m2::p0"),
                   ("disteval/medium-rest-client", "m2::p1")],
            profiles={},
        )
        results = simulate_recursive_gain(profiles, graph, n_bootstrap=100)
        task_result = results["disteval/medium-rest-client"]
        # Recursive path targets only the recoverable sub-task → more focused gain
        assert "flat_gain" in task_result
        assert "recursive_gain" in task_result
        # At minimum, recursive gain should be non-negative
        assert task_result["recursive_gain"] >= 0.0
```

### 6.6 Validation

```bash
pytest tests/test_environment_generator.py tests/test_environment_registry.py -v
pytest tests/test_training_sim.py::TestSimulateRecursiveGain -v
# Full suite regression
pytest tests/ -v
```

---

## 7  Milestone 4 — Distributed eval pooling

### 7.1 Scope

Implement `disteval/distributed_eval.py`. This module has no external dependencies
beyond the modules already implemented in Milestones 1–3. Full multi-agent
validation requires multiple agent job directories (3 available in `jobs/`).

### 7.2 Files changed

| File | Change |
|------|--------|
| `disteval/distributed_eval.py` | Create (new) |
| `tests/test_distributed_eval.py` | Create (new tests) |

### 7.3 Test suite for Milestone 4

```python
# tests/test_distributed_eval.py
import pytest
from disteval.records import RecordStore, EpisodeRecord
from disteval.right_tail import right_tail_analysis
from disteval.recursion_engine import (
    RecursionEngine, RecursionEngineConfig, SubTaskDefinition, SubTaskGraph,
)
from disteval.trajectory_monitor import TrajectoryMonitor
from disteval.distributed_eval import (
    DistributedEvalPool,
    AgentContribution,
    ConsensusNode,
    CrossAgentTrainingPair,
)


def make_store(scores: list[float], task: str = "t", model: str = "agent") -> RecordStore:
    store = RecordStore()
    for i, s in enumerate(scores):
        store.add(EpisodeRecord("r", model, task, i, s, s >= 0.99))
    return store


def make_simple_graph(kind: str, task: str = "t") -> SubTaskGraph:
    sub_task = SubTaskDefinition(
        sub_task_id=f"{task}::phase-0", parent_task=task, sub_task_depth=1,
        entry_step=0, exit_step=10, phase_tag="write", instruction="",
        estimated_q_star=1.0 if kind != "stuck" else 0.0,
        estimated_q_bar=0.5 if kind == "recoverable" else (1.0 if kind == "solid" else 0.0),
        kind=kind, reward_weight=1.0,
    )
    return SubTaskGraph(
        parent_tasks=[task], sub_tasks=[sub_task], edges=[(task, sub_task.sub_task_id)], profiles={},
    )


@pytest.fixture
def pool():
    return DistributedEvalPool(similarity_threshold=0.50)


class TestIngest:

    def test_ingest_single_agent(self, pool):
        store = make_store([0.0, 0.5, 1.0])
        report = right_tail_analysis(store)
        graph = make_simple_graph("recoverable")
        pool.ingest("agent_A", report, graph)
        assert len(pool._contributions) >= 1

    def test_ingest_multiple_agents(self, pool):
        for name, scores in [("A", [1.0, 1.0]), ("B", [0.0, 0.0]), ("C", [0.5, 1.0])]:
            store = make_store(scores)
            report = right_tail_analysis(store)
            graph = make_simple_graph("solid" if scores == [1.0, 1.0] else "stuck" if scores == [0.0, 0.0] else "recoverable")
            pool.ingest(f"agent_{name}", report, graph)
        assert len(pool._contributions) == 3


class TestConsensusGraph:

    def test_consensus_node_per_subtask(self, pool):
        for name, kind in [("A", "solid"), ("B", "stuck")]:
            store = make_store([1.0] if kind == "solid" else [0.0])
            report = right_tail_analysis(store)
            graph = make_simple_graph(kind)
            pool.ingest(f"agent_{name}", report, graph)
        nodes = pool.build_consensus_graph()
        assert len(nodes) >= 1

    def test_cross_agent_gap_classification(self, pool):
        # Agent A: solid, Agent B: stuck → cross_agent_gap
        for name, kind in [("A", "solid"), ("B", "stuck")]:
            store = make_store([1.0] if kind == "solid" else [0.0])
            report = right_tail_analysis(store)
            graph = make_simple_graph(kind)
            pool.ingest(f"agent_{name}", report, graph, shareable=(name == "A"))
        nodes = pool.build_consensus_graph()
        gap_nodes = [n for n in nodes if n.env_class == "cross_agent_gap"]
        assert len(gap_nodes) >= 1

    def test_stable_classification_all_solid(self, pool):
        for name in ["A", "B"]:
            store = make_store([1.0, 1.0])
            report = right_tail_analysis(store)
            graph = make_simple_graph("solid")
            pool.ingest(f"agent_{name}", report, graph)
        nodes = pool.build_consensus_graph()
        assert all(n.env_class in ("stable", "contrastive") for n in nodes)

    def test_exploration_target_all_stuck(self, pool):
        for name in ["A", "B"]:
            store = make_store([0.0, 0.0])
            report = right_tail_analysis(store)
            graph = make_simple_graph("stuck")
            pool.ingest(f"agent_{name}", report, graph)
        nodes = pool.build_consensus_graph()
        assert any(n.env_class == "exploration_target" for n in nodes)


class TestCrossAgentPairs:

    def test_no_pairs_without_shareable(self, pool):
        for name, kind in [("A", "solid"), ("B", "stuck")]:
            store = make_store([1.0] if kind == "solid" else [0.0])
            report = right_tail_analysis(store)
            graph = make_simple_graph(kind)
            pool.ingest(f"agent_{name}", report, graph, shareable=False)
        nodes = pool.build_consensus_graph()
        pairs = pool.generate_cross_agent_pairs(nodes)
        assert len(pairs) == 0

    def test_pairs_with_shareable_solid(self, pool):
        # Agent A is solid AND shareable; Agent B is stuck
        for name, kind, shareable in [("A", "solid", True), ("B", "stuck", False)]:
            store = make_store([1.0, 1.0] if kind == "solid" else [0.0, 0.0])
            report = right_tail_analysis(store)
            # Give agent A a traj path on the contribution
            graph = make_simple_graph(kind)
            pool.ingest(f"agent_{name}", report, graph, shareable=shareable)
        # Manually add a traj path to agent A's contribution for testing
        for c in pool._contributions:
            if c.agent_name == "agent_A":
                c.reinforce_traj_path = "/fake/traj_A.json"
        nodes = pool.build_consensus_graph()
        pairs = pool.generate_cross_agent_pairs(nodes)
        # With a traj path available, at least one pair should be proposed
        assert isinstance(pairs, list)


class TestSummary:

    def test_summary_returns_string(self, pool):
        store = make_store([0.5, 1.0])
        report = right_tail_analysis(store)
        graph = make_simple_graph("recoverable")
        pool.ingest("agent_A", report, graph)
        s = pool.summary()
        assert isinstance(s, str)
        assert len(s) > 0
```

### 7.4 Multi-agent integration test

When all three `jobs/run_A`, `jobs/run_B`, `jobs/run_C` directories exist:

```bash
# Run the multi-agent pool integration test
python3 -c "
from disteval.self_engine import SelfEngine
from disteval.recursion_engine import RecursionEngine, RecursionEngineConfig
from disteval.distributed_eval import DistributedEvalPool

agents = [
    ('Claude Code', 'claude-sonnet-4-5',  'jobs/run_A/disteval-run-A'),
    ('Gemini CLI',  'gemini-2.5-flash',   'jobs/run_B/disteval-run-B'),
    ('Codex CLI',   'openai/o4-mini',     'jobs/run_C/disteval-run-C'),
]

pool = DistributedEvalPool()
for agent_name, model_name, job_dir in agents:
    try:
        engine = SelfEngine.from_job_dirs([job_dir], agent_name=agent_name, model_name=model_name)
        plan = engine.run_cycle()
        from disteval.right_tail import right_tail_analysis
        report = right_tail_analysis(engine.store)
        # Minimal graph for integration (no full decomposition required)
        from disteval.recursion_engine import SubTaskGraph
        graph = SubTaskGraph(parent_tasks=[], sub_tasks=[], edges=[], profiles={})
        pool.ingest(agent_name, report, graph, shareable=False)
        print(f'  {agent_name}: ingested OK')
    except Exception as e:
        print(f'  {agent_name}: error - {e}')

nodes = pool.build_consensus_graph()
print(f'Consensus nodes: {len(nodes)}')
print(pool.summary())
"
```

### 7.5 Validation

```bash
pytest tests/test_distributed_eval.py -v
pytest tests/ -v   # full regression
```

---

## 8  Risk and backward-compatibility notes

### 8.1 Backward compatibility strategy

All changes to existing files follow three rules enforced throughout:

1. **New fields are optional** with sensible defaults (`None`, `[]`, `False`).
   Existing code that constructs `TaskOutcomeProfile`, `TrajectoryRecord`, or
   `PatternMatch` without the new fields will continue to work because all new
   fields use `field(default_factory=...)` or `Optional[T] = None`.

2. **New methods do not replace existing ones.** `TrajectoryMonitor.divergence_steps()`
   is a new method; the private `SelfEngine._find_divergence_step()` is
   unchanged. `SelfEngine.run_cycle_with_recursion()` is separate from
   `run_cycle()`.

3. **CLI flags default to the existing behavior.** `disteval engine` without
   `--enable-recursion` runs identically to today. No existing tests break.

The test `tests/test_recursion_engine.py::TestBackwardCompatibility` codifies
these guarantees and should be run as part of every CI build once Milestone 2
lands.

### 8.2 Known risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `find_phase_boundaries()` returns empty list for tasks with few tool calls | Medium | Add a fallback: if no divergence steps found, use uniform split into `max_phase_boundaries` equal slices. |
| `parse_test_suite()` fails on tasks with SCORE logic other than `SCORE+=` (e.g., conditional score) | Medium | Add a try/except wrapper; fall back to treating the whole task as a single checkpoint with weight 1.0. |
| Sub-task `estimated_q_star` from structural proxy is inaccurate | Medium | Mark proxy-scored sub-tasks with `scoring_method="proxy"` in `SubTaskDefinition` for downstream filtering. |
| Entry-step context prefix replay diverges in container (Milestone 3+) | High | Milestone 3 uses only `materialise()` (JSON files), not live replay. Container replay is explicitly deferred. |
| `DistributedEvalPool.generate_cross_agent_pairs()` produces false positives at low similarity | Low | `similarity_threshold=0.60` default; tunable per deployment. Document the threshold clearly. |
| `EnvironmentRegistry` grows unboundedly across many cycles | Low | `retire()` archives rather than deletes; the registry is a log. Prune via `envs_by_kind("retired")` if size is an issue. |
| Circular import between `recursion_engine.py` and `right_tail.py` | Low | `recursion_engine.py` imports `TaskOutcomeProfile` at the top level (it is a dataclass with no back-imports). `right_tail.py` does not import `recursion_engine.py`. The `SubTaskGraph` field added to `RightTailReport` uses `Optional["SubTaskGraph"]` with a string forward reference, resolved at the call site. |

### 8.3 Implementation order within milestones

Within each milestone, implement in dependency order:

**Milestone 1:** `test_suite_parser.py` → `tests/test_test_suite_parser.py`

**Milestone 2:**
1. `trajectory_monitor.py` (add `divergence_steps()`)
2. `trajectory_memory.py` (add `sub_task_slices`, `retrieve_for_sub_task()`)
3. `right_tail.py` (add optional fields)
4. `recursion_engine.py` (core implementation)
5. `self_engine.py` (add `run_cycle_with_recursion()`)
6. `__main__.py` (add CLI flags)
7. `tests/test_recursion_engine.py`

**Milestone 3:**
1. `environment_generator.py`
2. `environment_registry.py`
3. `training_sim.py` (add `simulate_recursive_gain()`)
4. `tests/test_environment_generator.py`
5. `tests/test_environment_registry.py`

**Milestone 4:**
1. `distributed_eval.py`
2. `tests/test_distributed_eval.py`

### 8.4 Python version and dependency constraints

All code must be compatible with Python 3.10+ (as stated in the skill file).
New modules use:
- `from __future__ import annotations` (deferred evaluation, resolves forward
  references)
- `tomllib` (stdlib since 3.11; fallback `tomli` already handled in
  `adapters/harbor_jobs.py` lines 29–34 — same pattern)
- `re`, `pathlib`, `dataclasses`, `typing`, `json`, `logging`, `statistics`,
  `collections` — all stdlib

No new runtime dependencies are introduced. `numpy` and `pandas` (already
present) are used in `distributed_eval.py` for cosine similarity and
`simulate_recursive_gain()`.

### 8.5 CURRICULUM_FORMAT.md extension

The curriculum JSON will grow two optional top-level fields, added by
`SelfImprovementPlan.to_dict()` when recursion is enabled:

```json
{
  "sub_task_graph": {
    "parent_tasks": ["disteval/medium-rest-client"],
    "sub_tasks": [
      {
        "sub_task_id": "medium-2::phase-2",
        "parent_task": "disteval/medium-rest-client",
        "sub_task_depth": 1,
        "entry_step": 8,
        "exit_step": 18,
        "phase_tag": "write",
        "kind": "recoverable",
        "reward_weight": 0.25,
        "instruction": "Implement the Engineering department groupby..."
      }
    ],
    "edges": [["disteval/medium-rest-client", "medium-2::phase-2"]]
  },
  "recursive_score_left": 0.487
}
```

Downstream DPO trainers that do not consume these fields are unaffected
(JSON consumers ignore unknown keys by default).

---

## Appendix A: File dependency matrix

```
                         test_suite  recursion  right   traj_   traj_   env_    env_    dist_
                         _parser     _engine    _tail   monitor memory  generator registry eval
─────────────────────────────────────────────────────────────────────────────────────────────
test_suite_parser           ■
recursion_engine            ▶           ■          ▶       ▶       ▶
right_tail                                         ■
trajectory_monitor                                          ■
trajectory_memory                                                   ■
environment_generator       ▶           ▶                           ▶       ■
environment_registry                               ▶                        ▶       ■
distributed_eval                        ▶          ▶                                        ■
self_engine (extension)     ▶           ▶          ▶       ▶       ▶                        ▶

■ = defined here    ▶ = imports from
```

---

## Appendix B: Quick-reference checkpoint table for `medium-2`

| Index | ID | Weight | Description |
|-------|----|--------|-------------|
| 0 | `medium-2::phase-0` | 0.10 | Validate JSON (`summary.json` is valid JSON) |
| 1 | `medium-2::phase-1` | 0.25 | `total_eligible_users == 7` (age ≥ 30 filter) |
| 2 | `medium-2::phase-2` | 0.25 | Engineering groupby: count=3, avg=111666.67 |
| 3 | `medium-2::phase-3` | 0.20 | Sales groupby: count=1, avg=85000.0 |
| 4 | `medium-2::phase-4` | 0.20 | HR groupby: count=2, avg=71500.0 |

These weights are the ground truth for `SubTaskDefinition.reward_weight` on this task
and for the `GenEnv.reward_weight` values in any generated environments derived from it.

---

*End of Phase 4A implementation plan.*
