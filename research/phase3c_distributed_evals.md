# Phase 3C — Distributed Evaluations and the Recursive Self-Improvement Loop

**Date:** 2026-06-23
**Input:** Phase 2 Master Report (`research/phase2_master_report.md`), Phase 2A/2B/2C designs, and the current disteval source (`disteval/self_engine.py`, `disteval/compare.py`, `disteval/compare_report.py`, `disteval/trajectory_memory.py`, `disteval/right_tail.py`, `disteval/trajectory_monitor.py`, `disteval/training_sim.py`, `CURRICULUM_FORMAT.md`).
**Output:** Design document for how distributed agent evaluations (multiple agents, multiple runs) feed into the recursive self-improvement loop and environment generation.
**Constraint:** No existing code is modified. This is a research-only design document.

---

## 1. Goal and scope

This document answers the Phase 2 open questions on distributed evaluations (Phase 2 Master Report, section 7, questions 5 and 7):

> 5. **Distributed evals:** How do multiple agents' evals aggregate into a shared sub-task graph and environment generation pool?
> 7. **Cross-agent sharing:** If Agent A is SOLID on a sub-task, can its trajectories serve as reinforce targets for Agent B's sub-task curriculum?

The current `disteval` pipeline is per-agent:

- `SelfEngine` (`disteval/self_engine.py`, lines 224–435) builds a `SelfImprovementPlan` for one agent from that agent's own Harbor job directories.
- `compare_report.py` compares agents at the whole-task level using mean, IQM, CVaR, and pass metrics (lines 52–181), but it does not compare agents on sub-tasks or generate training environments from cross-agent gaps.
- `RecursionEngine` (designed in Phase 2) decomposes a *single* agent's STUCK/RECOVERABLE tasks into sub-task RMDPs. It does not yet pool decompositions across agents.

This design extends that pipeline with a **distributed evaluation pool**: a shared repository of sub-task boundaries, trajectories, and derived environments that is populated by multiple agents' evals and consumed by each agent's `SelfEngine` / `RecursionEngine` to improve the next cycle.

The deliverable is a concrete design for:

1. Aggregation of multi-agent evals into a shared sub-task graph / environment pool.
2. Cross-agent reinforce/contrast trajectory generation.
3. Disagreement handling between agents on sub-task boundaries or solutions.
4. A shared data structure for distributed eval results and derived environments.
5. Interaction between the distributed eval loop and the per-agent `SelfEngine` / `RecursionEngine`.
6. Privacy / attribution considerations when one agent's trajectories train another.
7. Open questions for Phase 4.

---

## 2. Aggregation of multi-agent evals into a shared sub-task graph / environment pool

### 2.1 What is being aggregated

Each agent run produces a `RightTailReport` (`disteval/right_tail.py`, lines 135–158) and, under recursion, a `SubTaskGraph` (Phase 2A/2C). The distributed pool aggregates the following for every task in the shared task suite:

- Per-agent `TaskOutcomeProfile` (scores, `q_star`, `q_bar`, `gap`, `kind`, `reinforce_idx`, `contrast_idx`).
- Per-agent `SubTaskGraph` (entry/exit boundaries, sub-task IDs, parent/child edges, per-sub-task profiles).
- Per-agent trajectory slices for each sub-task boundary (paths into the Harbor job directories, plus the sliced `steps` indices).
- Per-agent `TrajectoryMemory` embeddings (so the pool can retrieve similar cross-agent demonstrations structurally).

The pool is **not** a new training loop; it is a data structure that downstream loops query for environments and cross-agent training pairs.

### 2.2 Aggregation pipeline

```text
FOR each agent A in {Claude, Gemini, Codex, ...}:
    run_A = Harbor job directory for A
    store_A = load_harbor_job(run_A)        # disteval/adapters/harbor_jobs.py
    report_A = right_tail_analysis(store_A)  # disteval/right_tail.py:207
    graph_A = RecursionEngine.decompose(report_A, traj_records_A)  # Phase 2C

    pool.ingest(agent=A, report=report_A, graph=graph_A, job_dir=run_A)

pool.aggregate()   # builds consensus sub-task graph + environment pool
```

The `pool.ingest()` step normalizes task names using the existing `_TASK_NAME_MAP` in `SelfEngine` (`disteval/self_engine.py`, lines 471–478) and uses the same trajectory loader (`disteval/trajectory_loader.py`) already used by `TrajectoryMonitor` and `TrajectoryMemory`.

### 2.3 Consensus sub-task graph

A key design choice is whether the pool forces all agents to share the *same* sub-task boundaries or keeps per-agent boundary variants. We propose a **consensus graph with boundary variants**:

- **Consensus nodes:** Sub-task identities are derived from the *semantic* task structure (e.g., `medium-2::phase-2` = "Engineering groupby correct"), not from any one agent's tool-call index. This is possible because the Phase 2 design supports test-script checkpoint alignment (`require_checkpoint_alignment`, Phase 2A, section 2.4; Phase 2B, section 3.4) and the `medium-2` test script already exposes five semantic checkpoints (`tasks/medium-2/tests/test.sh`, lines 25–64, referenced in Phase 2B section 7.1).
- **Boundary variants:** Each consensus node stores a set of `(agent, entry_step, exit_step, confidence)` boundary records. If agents disagree on the exact tool-call window, all variants are retained, and the pool exposes a **preferred boundary** computed by confidence-weighted voting (see section 4).
- **Edges:** Parent/child edges are taken from the union of all agent graphs. If Agent A decomposes a task and Agent B does not, the edge is kept if at least one agent with sufficient data (e.g., ≥ 3 trajectories) produced it.

### 2.4 Environment pool derivation

From the consensus graph, the pool derives a set of **sub-task environments**. These are not full Gymnasium environments yet; they are environment *definitions* that can be turned into training tasks. Each environment definition contains:

- `env_id`: the consensus sub-task ID (e.g., `medium-2::phase-2`).
- `parent_task`: the original task (e.g., `disteval/medium-rest-client`).
- `entry_condition`: the context prefix and, when available, the file-system state at the entry boundary.
- `exit_condition`: the test checkpoint or monitor confidence threshold that marks success.
- `reward`: the incremental reward weight derived from the parent test script (e.g., 0.25 for `medium-2` C1; see Phase 2B section 5.4).
- `source_agents`: which agents contributed trajectories and boundary variants.
- `status`: whether the environment is `stable`, `contrastive`, or `exploration_target` (see section 2.5).

The environment pool is the bridge between the aggregated sub-task graph and per-agent training. It is analogous to the way `SelfImprovementPlan.curriculum` is the bridge between `right_tail_analysis` and a DPO trainer in `CURRICULUM_FORMAT.md` (section 4).

### 2.5 Environment classification from distributed SOLID/RECOVERABLE/STUCK signals

The pool classifies each consensus sub-task environment based on the *union* of agent classifications, not just one agent's classification:

| Status | Criterion | Source in codebase |
|---|---|---|
| **Stable** | ≥ 1 agent is SOLID on the sub-task and no agent is STUCK | Mirrors `kind == "solid"` in `right_tail.task_outcome_profile()` (line 184) |
| **Contrastive** | At least one agent is RECOVERABLE on the sub-task, and at least one agent (possibly the same) is SOLID | Mirrors `kind == "recoverable"` (line 187) |
| **Exploration target** | All agents are STUCK on the sub-task | Mirrors `kind == "stuck"` (line 181) |
| **Cross-agent gap** | Agent A is SOLID and Agent B is STUCK/RECOVERABLE | New distributed signal derived from per-agent `TaskOutcomeProfile` comparisons |

The "cross-agent gap" status is the most valuable new signal: it identifies a sub-task where one agent has demonstrated a reliable solution and another has not. That is exactly the situation where cross-agent training pairs can close a capability gap without requiring a human-written solution.

---

## 3. Cross-agent reinforce/contrast trajectories

### 3.1 When cross-agent pairs are valid

The current `SelfEngine._build_training_pairs()` (`disteval/self_engine.py`, lines 505–551) only pairs trajectories from the *same* agent on the *same* task. The distributed pool extends this to **cross-agent pairs** under the following conditions:

1. **Sub-task identity match:** The reinforce and contrast trajectories must belong to the same consensus sub-task environment (same `env_id`, aligned boundaries). The pool's preferred boundary is used to slice both trajectories.
2. **Outcome contrast:** The reinforce trajectory is from an agent that is SOLID on the sub-task (or a high-scoring run). The contrast trajectory is from an agent that is RECOVERABLE or STUCK on the same sub-task (or a low-scoring run).
3. **Structural compatibility:** The tool sequences of the two slices must be structurally similar enough that the contrast agent can plausibly learn from the reinforce agent. This is measured using `TrajectoryMemory._task_match()` and cosine similarity on the bag-of-tools embeddings (`disteval/trajectory_memory.py`, lines 164–181, 312–323), the same primitives already used for memory retrieval.
4. **Privacy / attribution gate:** The reinforce agent must have opted in to cross-agent sharing (see section 6).

### 3.2 Cross-agent pair generation algorithm

```text
FUNCTION build_cross_agent_pairs(pool, sub_task_env_id, consumer_agent):
    env = pool.environments[sub_task_env_id]
    reinforce_candidates = []
    contrast_candidates = []

    FOR each agent A in env.source_agents:
        profile_A = pool.agent_profile(A, env.parent_task, sub_task_env_id)
        slices_A = pool.agent_slices(A, sub_task_env_id)

        IF profile_A.kind == "solid" OR high_score_slice_exists(slices_A):
            reinforce_candidates.extend(slices_A)
        ELSE IF A == consumer_agent AND profile_A.kind == "recoverable":
            contrast_candidates.extend(slices_A)

    pairs = []
    FOR each contrast_slice IN contrast_candidates:
        # Pick the reinforce slice from the *most similar* solid agent,
        # using the same structural similarity that TrajectoryMemory uses.
        best_reinforce = argmax_{r in reinforce_candidates} sim(r.tool_sequence, contrast_slice.tool_sequence)
        pairs.append(CrossAgentPair(
            reinforce_agent=best_reinforce.agent,
            reinforce_traj_path=best_reinforce.traj_path,
            reinforce_entry=best_reinforce.entry,
            reinforce_exit=best_reinforce.exit,
            contrast_agent=consumer_agent,
            contrast_traj_path=contrast_slice.traj_path,
            contrast_entry=contrast_slice.entry,
            contrast_exit=contrast_slice.exit,
            gap=best_reinforce.sub_score - contrast_slice.sub_score,
        ))

    RETURN pairs
```

This mirrors the existing `SelfEngine._build_training_pairs()` logic (pair every low slice with the highest-scoring high slice) but operates across agents and sub-task windows instead of whole-task scores.

### 3.3 Example: Claude SOLID on a sub-task that Codex is STUCK on

Consider the `medium-2` task from Phase 2B (section 7.1):

- Sub-task `medium-2::phase-2` = "Engineering groupby correct" (reward 0.25).
- Suppose Claude Code is SOLID on this sub-task (consistently scores 0.25/0.25).
- Suppose Codex CLI is STUCK on this sub-task (never scores the 0.25 checkpoint).

The pool generates a cross-agent `TrainingPair`:

- `reinforce_traj_path`: Claude's successful trajectory on `medium-2`, sliced to the phase-2 boundary.
- `contrast_traj_path`: Codex's failing trajectory on `medium-2`, sliced to the same phase-2 boundary.
- `reinforce_agent`: `Claude Code`.
- `contrast_agent`: `Codex CLI`.
- `gap`: 0.25.

This pair is inserted into Codex's `SelfImprovementPlan` for `medium-2` with a `source` field indicating it is a cross-agent reinforce target. The pair is *not* inserted into Claude's plan (Claude is already SOLID on the sub-task).

### 3.4 Cross-agent contrastive signal vs. self-contrastive signal

There are two kinds of contrastive signal now:

- **Self-contrastive:** Same agent, same task, high vs. low score. Used by `SelfEngine._build_training_pairs()` today.
- **Cross-agent contrastive:** Different agents, same consensus sub-task, solid vs. stuck/recoverable score.

The cross-agent signal is stronger when it comes from a *structurally similar* agent (e.g., two LLM coding agents) and weaker when the tool sequences differ dramatically (e.g., Claude writes Python early while the consumer searches extensively). The pool uses `TrajectoryMemory` structural similarity to filter out implausible pairs.

---

## 4. Handling disagreement between agents on sub-task boundaries or solutions

### 4.1 Sources of disagreement

Agents can disagree on sub-task decomposition in two ways:

1. **Boundary disagreement:** Agent A places the entry/exit of `medium-2::phase-2` at tool-call indices (4, 12), while Agent B places it at (3, 14). This is common because different agents have different tool-usage patterns (e.g., extra `read_file` calls before writing code).
2. **Solution disagreement:** Agent A solves `medium-2::phase-2` with a `write_file` + `run_shell_command` pattern, while Agent B solves it with a different code structure or even a different interpretation of the groupby requirement. The test script is the ground truth for whether the solution is correct, but the *process* can differ.

### 4.2 Boundary disagreement resolution

The pool uses a **confidence-weighted consensus** rule:

1. For each consensus sub-task, collect all `(agent, entry_step, exit_step, confidence)` boundary records.
2. The confidence of a boundary is derived from the `TrajectoryMonitor` prediction probabilities at the boundary steps (`PatternMatch.p_high`, `disteval/trajectory_monitor.py`, lines 75–87) and the number of trajectories supporting it.
3. Compute the median boundary and the inter-agent spread. If the spread is below a threshold (e.g., ≤ 3 tool calls), accept the median as the **preferred boundary**.
4. If the spread exceeds the threshold, keep the agent-specific boundary variants and create **multiple environment variants** for the same sub-task. The consumer agent trains on the variant produced by the agent most structurally similar to it (using `TrajectoryMemory` similarity).

This rule is conservative: it does not force a single boundary when agents genuinely use different tool strategies, but it still enables cross-agent transfer when boundaries are close enough.

### 4.3 Solution disagreement resolution

When agents reach the same test checkpoint via different solution paths, the pool retains *multiple valid solution paths* for the same sub-task environment. This is analogous to the way `TrajectoryMemory` already stores multiple high-outcome memories for the same task (`disteval/trajectory_memory.py`, lines 106–128). The consumer agent can retrieve the solution path most similar to its own trajectory prefix.

If the solutions disagree on the *test outcome* (one agent passes the checkpoint, another does not), the test script is the final arbiter. The pool only marks a sub-task as SOLID for an agent if the agent's trajectories actually satisfy the checkpoint reward condition, mirroring `right_tail.task_outcome_profile()` (lines 176–189).

### 4.4 Explicit disagreement log

The pool records a `disagreement` object for each consensus sub-task with high boundary spread or conflicting solution outcomes. This object is exposed to the per-agent `SelfImprovementPlan` as a `boundary_confidence` field and a `solution_variants` list, so a downstream trainer can decide whether to use cross-agent pairs or fall back to self-pairs.

---

## 5. Shared data structure: the `DistributedEvalPool`

### 5.1 New module and class

We propose a new module `disteval/distributed_eval.py` containing a `DistributedEvalPool` class. It is a pure-Python / numpy data structure that builds on existing dataclasses.

### 5.2 Core dataclasses

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .right_tail import TaskOutcomeProfile
from .recursion_engine import SubTaskDefinition, SubTaskGraph


@dataclass
class AgentSubTaskContribution:
    """One agent's observed boundaries and score profile for a consensus sub-task."""
    agent_name: str
    model_name: str
    sub_task_id: str
    entry_step: int
    exit_step: int
    boundary_confidence: float
    profile: TaskOutcomeProfile
    reinforce_slice_paths: list[str] = field(default_factory=list)
    contrast_slice_paths: list[str] = field(default_factory=list)
    shareable: bool = True   # privacy gate (see section 6)


@dataclass
class SubTaskEnvironment:
    """A derived training environment for a consensus sub-task."""
    env_id: str
    parent_task: str
    phase_tag: str
    instruction: str
    reward_delta: float
    preferred_boundary: tuple[int, int]   # (entry_step, exit_step) chosen by consensus
    boundary_variants: list[AgentSubTaskContribution]
    status: str   # "stable" | "contrastive" | "exploration_target" | "cross_agent_gap"
    source_agents: list[str]
    entry_condition: dict = field(default_factory=dict)
    exit_condition: dict = field(default_factory=dict)
    solution_variants: list[dict] = field(default_factory=list)
    disagreement: Optional[dict] = None


@dataclass
class CrossAgentTrainingPair:
    """A cross-agent reinforce/contrast pair derived from the pool."""
    env_id: str
    parent_task: str
    reinforce_agent: str
    reinforce_model: str
    reinforce_traj_path: str
    reinforce_entry_step: int
    reinforce_exit_step: int
    contrast_agent: str
    contrast_model: str
    contrast_traj_path: str
    contrast_entry_step: int
    contrast_exit_step: int
    gap: float
    structural_similarity: float
    privacy_approved: bool = False


@dataclass
class DistributedEvalPool:
    """Shared pool of distributed eval results and derived sub-task environments."""
    task_suite: str
    agents: list[str]
    # agent -> task -> TaskOutcomeProfile
    profiles: dict[str, dict[str, TaskOutcomeProfile]] = field(default_factory=dict)
    # agent -> task -> SubTaskGraph
    graphs: dict[str, dict[str, SubTaskGraph]] = field(default_factory=dict)
    # consensus sub-task env_id -> SubTaskEnvironment
    environments: dict[str, SubTaskEnvironment] = field(default_factory=dict)
    # env_id -> list of cross-agent pairs
    cross_agent_pairs: dict[str, list[CrossAgentTrainingPair]] = field(default_factory=dict)
    # audit trail
    ingest_log: list[dict] = field(default_factory=list)
```

### 5.3 Public API

```python
class DistributedEvalPool:
    def ingest(
        self,
        agent_name: str,
        model_name: str,
        job_dir: str,
        report: RightTailReport,
        graph: SubTaskGraph,
        shareable: bool = True,
    ) -> None: ...

    def build_consensus_graph(self) -> SubTaskGraph: ...

    def derive_environments(self) -> dict[str, SubTaskEnvironment]: ...

    def build_cross_agent_pairs(
        self,
        consumer_agent: str,
        min_similarity: float = 0.50,
    ) -> dict[str, list[CrossAgentTrainingPair]]: ...

    def get_environment_for_agent(
        self,
        env_id: str,
        agent_name: str,
    ) -> SubTaskEnvironment: ...

    def to_dict(self) -> dict: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "DistributedEvalPool": ...
```

### 5.4 JSON serialization format

The pool is serialized to a single JSON file, e.g., `distributed_eval_pool.json`. The format is intentionally close to the existing curriculum JSON in `CURRICULUM_FORMAT.md` so that downstream trainers can reuse parsing logic.

Top-level structure:

```json
{
  "task_suite": "tasks/",
  "agents": ["Claude Code", "Gemini CLI", "Codex CLI"],
  "n_agent_reports": 3,
  "n_environments": 12,
  "n_cross_agent_pairs": 7,
  "profiles": {
    "Claude Code": {
      "disteval/medium-rest-client": { ...TaskOutcomeProfile... }
    }
  },
  "graphs": {
    "Claude Code": {
      "disteval/medium-rest-client": { ...SubTaskGraph... }
    }
  },
  "environments": {
    "medium-2::phase-2": {
      "env_id": "medium-2::phase-2",
      "parent_task": "disteval/medium-rest-client",
      "phase_tag": "engineering_groupby",
      "instruction": "Implement the Engineering department groupby correctly in the REST client.",
      "reward_delta": 0.25,
      "preferred_boundary": [4, 12],
      "status": "cross_agent_gap",
      "source_agents": ["Claude Code", "Codex CLI"],
      "boundary_variants": [
        {
          "agent_name": "Claude Code",
          "model_name": "claude-sonnet-4-5",
          "entry_step": 4,
          "exit_step": 12,
          "boundary_confidence": 0.84,
          "kind": "solid",
          "shareable": true
        },
        {
          "agent_name": "Codex CLI",
          "model_name": "openai/o4-mini",
          "entry_step": 3,
          "exit_step": 14,
          "boundary_confidence": 0.61,
          "kind": "stuck",
          "shareable": true
        }
      ],
      "disagreement": {
        "boundary_spread": 2.0,
        "resolution": "preferred_boundary_is_median",
        "solution_variants": 2
      }
    }
  },
  "cross_agent_pairs": {
    "medium-2::phase-2": [
      {
        "env_id": "medium-2::phase-2",
        "reinforce_agent": "Claude Code",
        "reinforce_traj_path": "jobs/run_A/.../medium-2__abc/agent/trajectory.json",
        "reinforce_entry_step": 4,
        "reinforce_exit_step": 12,
        "contrast_agent": "Codex CLI",
        "contrast_traj_path": "jobs/run_C/.../medium-2__def/agent/trajectory.json",
        "contrast_entry_step": 3,
        "contrast_exit_step": 14,
        "gap": 0.25,
        "structural_similarity": 0.72,
        "privacy_approved": true
      }
    ]
  },
  "ingest_log": [
    {
      "agent": "Claude Code",
      "job_dir": "jobs/run_A/disteval-run-A",
      "timestamp": "2026-06-23T10:00:00Z",
      "n_tasks": 6,
      "shareable": true
    }
  ]
}
```

### 5.5 Reuse of existing primitives

The pool reuses the following existing components rather than inventing new data formats:

- `TaskOutcomeProfile` from `disteval/right_tail.py` (lines 115–132) for per-agent, per-sub-task scores.
- `SubTaskGraph` / `SubTaskDefinition` from the Phase 2 RecursionEngine design for the hierarchical structure.
- `TrajectoryMemory` embeddings and `_task_match()` (`disteval/trajectory_memory.py`, lines 164–181, 312–323) for cross-agent structural similarity.
- `TrajectoryMonitor.check()` (`disteval/trajectory_monitor.py`, lines 488–524) for boundary confidence.
- `compare.wasserstein()` / `prob_improvement()` (`disteval/compare.py`, lines 12–39) for quantifying cross-agent distribution differences at the sub-task level.

---

## 6. Interaction between the distributed eval loop and per-agent `SelfEngine` / `RecursionEngine`

### 6.1 The new outer loop

The current per-agent loop is:

```text
eval → SelfEngine.run_cycle() → SelfImprovementPlan → train → re-eval
```

The distributed loop wraps this:

```text
FOR each agent A:
    eval_A → SelfEngine_A.run_cycle() → plan_A → RecursionEngine_A.decompose() → graph_A

pool = DistributedEvalPool()
FOR each agent A:
    pool.ingest(A, report_A, graph_A, job_dir_A)

pool.build_consensus_graph()
pool.derive_environments()
pool.build_cross_agent_pairs(consumer_agent=A) for each A

FOR each agent A:
    augmented_plan_A = augment_plan_with_pool(plan_A, pool)
    train on augmented_plan_A → re-eval_A
```

### 6.2 `SelfEngine` integration points

We propose two new optional integration points in `SelfEngine` (no breaking changes to existing APIs):

1. **`SelfEngine.from_job_dirs()`** accepts a new optional `distributed_pool: Optional[DistributedEvalPool] = None` argument (analogous to the `enable_recursion` extension in Phase 2C, section 1.1).
2. **`SelfEngine.run_cycle()`** consults the pool after building the flat curriculum:
   - For each RECOVERABLE or STUCK task in the agent's curriculum, look up the corresponding consensus sub-task environments in the pool.
   - If cross-agent pairs exist for those environments and the agent is the consumer, append them to the `TaskImprovement.training_pairs` list.
   - If the pool provides a preferred boundary that differs from the agent's own `RecursionEngine` boundary, use the pool's preferred boundary (higher confidence from more data) unless the agent's own boundary is more structurally similar to its own trajectories.

### 6.3 `RecursionEngine` integration points

`RecursionEngine` also benefits from the pool in two ways:

1. **Initial boundary hints:** When decomposing a task for a new agent, `RecursionEngine` can query the pool for the consensus sub-task boundaries of that task. This is a form of warm-start: the new agent does not have to rediscover boundaries from scratch if other agents have already produced them. The hint is advisory; the agent's own monitor confidence still controls whether the boundary is accepted (Phase 2A `RecursionEngineConfig.divergence_confidence = 0.70`).
2. **Memory augmentation:** `RecursionEngine._retrieve_sub_task_memory()` (Phase 2A, section 3.3) can retrieve not only the agent's own memory but also the pool's cross-agent reinforce slices, filtered by structural similarity and privacy approval.

### 6.4 Cycle-to-cycle feedback

The pool persists across cycles. After each cycle:

- New evals are ingested.
- The consensus graph is recomputed.
- Environments that were previously `cross_agent_gap` may become `contrastive` or `stable` as the consumer agent improves.
- Environments that were previously `exploration_target` (all agents stuck) may remain stuck, signaling that a new capability (e.g., a new tool or a human example) is needed rather than more cross-agent training.

This is the recursive self-improvement loop: the distributed eval output of cycle `n` changes the environment distribution and training pairs available for cycle `n+1`.

### 6.5 CLI integration

A new subcommand `disteval pool` is proposed (integrated in `disteval/__main__.py`, lines 52–63, 170–256). It accepts multiple agent job directories, builds the pool, and emits `distributed_eval_pool.json`:

```bash
disteval pool \
  jobs/run_A/disteval-run-A \
  jobs/run_B/disteval-run-B \
  jobs/run_C/disteval-run-C \
  --agents "Claude Code" "Gemini CLI" "Codex CLI" \
  --output distributed_eval_pool.json
```

The existing `disteval engine` subcommand can then consume the pool:

```bash
disteval engine jobs/run_C/disteval-run-C \
  --agent "Codex CLI" \
  --pool distributed_eval_pool.json \
  --output codex_plan_with_cross_agent.json
```

---

## 7. Privacy / attribution considerations

### 7.1 Why this matters

The distributed pool makes one agent's trajectories available as training data for another agent. This raises three concerns:

1. **Attribution:** A consumer agent's improvement may be partly due to another agent's trajectory. The pool must record provenance so that results can be audited and credited.
2. **Opt-in / opt-out:** Not all agents or their operators may want their trajectories used to train competing agents. The pool must respect per-agent sharing preferences.
3. **Data leakage:** Trajectory files may contain task inputs, intermediate outputs, or environment states that should not be shared across organizational boundaries. The pool should only share what is necessary.

### 7.2 Design: provenance and attribution

Every `CrossAgentTrainingPair` records:

- `reinforce_agent` and `reinforce_model` (e.g., `Claude Code`, `claude-sonnet-4-5`).
- `reinforce_traj_path` (path to the original Harbor trajectory).
- `contrast_agent` and `contrast_model`.
- `privacy_approved` flag.

The `SelfImprovementPlan.to_dict()` output (extended in Phase 2C, section 3.2) is augmented with a `cross_agent_attribution` object:

```json
{
  "cross_agent_attribution": {
    "total_pairs_used": 7,
    "agents_contributed": ["Claude Code", "Gemini CLI"],
    "pairs": [
      {
        "env_id": "medium-2::phase-2",
        "reinforce_agent": "Claude Code",
        "contrast_agent": "Codex CLI",
        "gap": 0.25,
        "shareable": true
      }
    ]
  }
}
```

This mirrors the existing `SelfImprovementPlan` audit fields (`job_dirs`, `n_trajectories_loaded`, lines 131–134) and the provenance fields in `EpisodeRecord` (`run_id`, `model`, `task`, `trajectory_ref`, `disteval/records.py`, lines 19–42).

### 7.3 Design: opt-in / opt-out

The `DistributedEvalPool.ingest()` method accepts a `shareable: bool = True` parameter per agent. If `shareable=False`, that agent's trajectories are used only for its own `SelfEngine` and are excluded from `cross_agent_pairs`. This is the default-safe setting: agents must explicitly opt in to cross-agent sharing.

The CLI exposes this as `--shareable` per agent directory:

```bash
disteval pool \
  jobs/run_A/disteval-run-A \
  jobs/run_B/disteval-run-B \
  --agents "Claude Code" "Gemini CLI" \
  --shareable true false \
  --output pool.json
```

### 7.4 Design: minimal shared data

The pool does not need to copy full trajectory JSON files. It stores only:

- Trajectory file paths (which point into the original Harbor job directories).
- Sliced step indices (`entry_step`, `exit_step`).
- Bag-of-tools embeddings (derived by `TrajectoryMemory`, no raw text).
- Sub-task score profiles.

If a deployment requires stronger privacy, the pool can store **only the embeddings and score profiles** and omit the trajectory paths entirely. The consumer agent would then use the embeddings to retrieve similar trajectories from its *own* memory, rather than loading the source agent's trajectory file. This is a privacy/utility trade-off controlled by a `share_level` parameter:

- `share_level="full"`: share paths + embeddings + scores (default research mode).
- `share_level="embedding"`: share only embeddings + scores.
- `share_level="none"`: no cross-agent sharing; agent only trains on itself.

### 7.5 License and model attribution

The `AgentSubTaskContribution` dataclass stores `agent_name` and `model_name` (matching `SelfImprovementPlan.agent_name` and `model_name`, `disteval/self_engine.py`, lines 112–114). The `ingest_log` records which job directory contributed which trajectories. This is sufficient for academic attribution and for detecting whether a model's improvement came from cross-agent data or from its own data.

---

## 8. Open questions for Phase 4

The following questions should be answered before implementation in Phase 4:

1. **Environment runtime format:** Should the derived `SubTaskEnvironment` be emitted as a JSON config for Harbor, a Gymnasium `Env` class, or a new disteval-specific format? The current `CURRICULUM_FORMAT.md` (section 4) only describes training pairs, not replayable environments.

2. **Entry-state capture:** How is the file-system / terminal state at a sub-task entry boundary captured and replayed? The current codebase does not persist intermediate states (`trajectory_loader.py` only reads `trajectory.json` and `reward.txt`). Does Harbor support state snapshots, or must we replay the context prefix?

3. **Cross-agent similarity threshold:** What is the right `min_similarity` cutoff for cross-agent pairs? Too high excludes useful transfer; too low includes misleading trajectories. A calibration experiment on the existing `jobs/run_A`, `run_B`, `run_C` data is needed.

4. **Weighting cross-agent vs. self-pairs:** In the DPO curriculum, should cross-agent pairs be weighted equally with self-pairs, down-weighted because the agent identity differs, or up-weighted because they provide a stronger signal? This interacts with `training_sim.apply_training_effect()` (`disteval/training_sim.py`, lines 199–295).

5. **Pool update policy:** Should the pool be recomputed from scratch every cycle, or incrementally updated? Incremental updates are faster but require careful handling of stale boundaries and trajectories that no longer exist.

6. **Scalability:** How does the pool behave with 10+ agents? The consensus graph and boundary voting algorithms are O(n_agents × n_sub_tasks), but memory similarity computations across all pairs could become expensive. Should we precompute a similarity index?

7. **Security and sandboxing:** When one agent's trajectory is used to guide another, are there risks of prompt injection or malicious intermediate files? The pool slices only tool-call sequences and file-system states, but a full security review is needed before cross-organizational sharing.

8. **Human oversight:** For `exploration_target` environments where all agents are STUCK, should the system automatically request a human demonstration, or should it flag the task and wait? This is the boundary between automated self-improvement and human-in-the-loop capability expansion.

9. **Reward propagation across agents:** The current `training_sim` propagates sub-task improvements to parent scores using a weighted sum (Phase 2B, section 5.4). How should this be extended when a sub-task's improvement comes from a *different* agent's trajectory? Does the training-effect model need an additional cross-agent transfer coefficient?

10. **Backward compatibility:** The `CURRICULUM_FORMAT.md` spec is already extended by Phase 2C. The distributed pool adds new top-level fields. Should these be added to the same curriculum JSON, or should the pool be a separate output file that the curriculum references by path?

---

## 9. Summary of proposed files and changes (no code edits yet)

- **New file:** `disteval/distributed_eval.py` — `DistributedEvalPool`, `AgentSubTaskContribution`, `SubTaskEnvironment`, `CrossAgentTrainingPair`.
- **New file:** `research/phase3c_distributed_evals.md` — this document.
- **New CLI subcommand:** `disteval pool` in `disteval/__main__.py` (lines 52–63, 170–256) — optional, default-disabled.
- **Optional extensions:** `SelfEngine.from_job_dirs()` and `SelfEngine.run_cycle()` in `disteval/self_engine.py` (lines 311–361, 375–435) to consume a `DistributedEvalPool`; `RecursionEngine` in `disteval/recursion_engine.py` (Phase 2 design) to query pool boundaries as warm-start hints.
- **No breaking changes:** all new fields are optional; existing `disteval engine` and `disteval compare` behavior is unchanged unless the pool is explicitly provided.
