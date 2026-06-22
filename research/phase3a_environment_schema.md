# Phase 3A — RL Environment Generation Layer: Schema and Design

**Date:** 2026-06-23  
**Input:** Phase 2 Master Report (`research/phase2_master_report.md`), Phase 2A/2B/2C designs, Phase 3B (`research/phase3b_self_improvement_loop.md`), Phase 3C (`research/phase3c_distributed_evals.md`), `disteval/self_engine.py`, `disteval/right_tail.py`, `disteval/trajectory_monitor.py`, `disteval/trajectory_memory.py`, `disteval/training_sim.py`, `CURRICULUM_FORMAT.md`, `tasks/medium-2/task.toml`, `tasks/medium-2/tests/test.sh`, `tasks/medium-2/instruction.md`, `tasks/medium-2/environment/`.  
**Output:** Schema and design document for the RL environment generation layer that the `RecursionEngine` (Phase 2) and `EnvironmentGenerator` (Phase 4 implementation) will produce.  
**Constraint:** No existing disteval code is modified. Design only; grounded in actual file paths and line numbers.

---

## 1. The RL environment schema

A **generated RL environment** (abbreviated `GenEnv`) is the atomic unit produced by the environment generation layer. It encapsulates everything needed to replay a sub-task, score it, and use it for training — while remaining consistent with disteval's existing task file structure (`task.toml`, `instruction.md`, `environment/`, `tests/test.sh`).

### 1.1 Formal components

Each `GenEnv` is defined by a six-tuple:

```
GenEnv = (S, A, O, R, T, Z)

S  — state space
A  — action space
O  — observation space
R  — reward function (maps S × A × S → ℝ)
T  — transition function (implicit: the Harbor/container runtime)
Z  — termination conditions
```

These are described in detail below.

### 1.2 State space `S`

The state of a sub-task environment at time-step `t` is:

```
s_t = {
  "fs_state":      file-system snapshot at the sub-task entry boundary,
  "tty_state":     terminal / process state (mock server live or not, env vars, etc.),
  "context_prefix": the tool-call step sequence steps[0 : entry_step],
  "context_outputs": accumulated stdout / stderr produced by context_prefix,
  "sub_task_id":   e.g. "medium-2::phase-2",
  "step_index":    current step within the sub-task (0 at entry),
  "phase_tag":     e.g. "write" | "exec" | "verify",
  "cycle":         which improvement cycle generated this state
}
```

**Grounding:** `context_prefix` corresponds to `entry_context` in the proposed `TrainingPair` extension (Phase 2B, `research/phase2b_decomposition_algorithm.md`, lines 158–177). `fs_state` captures what the agent has already written to disk before the sub-task entry point. `tty_state` captures long-running services (e.g., the Flask mock server started in `tasks/medium-2/tests/test.sh` lines 5–8) that must still be alive at the sub-task boundary.

**State transition at entry:** When the environment is initialized, it replays `context_prefix` deterministically against the container image to produce the exact file-system and process state at `entry_step`. This is the **state synthesis** approach described in Phase 2B (section 4.2) as an alternative to snapshotting, chosen because Harbor currently has no snapshot API.

### 1.3 Action space `A`

The action space mirrors the agent's existing tool call vocabulary, with no new tools introduced:

```
A = {write_file, run_shell_command, exec_command, read_file,
     list_directory, search_tool, ... (any canonical tool from TrajectoryFeaturizer)}
```

**Grounding:** The canonical tool names are enumerated in `disteval/trajectory_monitor.py` `TrajectoryFeaturizer._TOOL_ALIASES` (lines 99–121). Each action is a structured call `{tool_name, args}`. Actions outside the sub-task's `exit_step` window produce a timeout signal that maps to the `truncation` termination condition.

### 1.4 Observation space `O`

The agent receives:

```
o_t = {
  "instruction":     the sub-task instruction string (derived from SubTaskDefinition),
  "context_summary": a condensed summary of context_prefix steps (≤ 300 tokens),
  "fs_listing":      listing of files visible at /app (or whatever WORKDIR),
  "prev_tool_output": stdout / stderr from the most recent action,
  "phase_hint":      optional hint derived from phase_tag (e.g., "You are in the 'exec' phase"),
  "memory_prompt":   optional retrieved trajectory prompt from TrajectoryMemory
}
```

**Grounding:** `memory_prompt` uses the existing `TrajectoryMemory.generate_retrieval_prompt()` output (`disteval/trajectory_memory.py`, lines 332–390) — no new code required. `context_summary` is a condensed prefix kept below a token budget to avoid blowing LLM context limits.

### 1.5 Reward function `R`

The reward at each step is **sparse at checkpoints**. For a sub-task that corresponds to a single test-script checkpoint:

```
R(s_t, a_t, s_{t+1}) = {
  checkpoint_weight   if exit_condition is satisfied at s_{t+1},
  small_step_penalty  otherwise (default: 0.0, optional: -0.001 per step to encourage efficiency),
  -checkpoint_weight  if a failure condition is triggered (optional, default: 0.0)
}
```

**Where checkpoint weights come from:** They are read directly from `tasks/medium-2/tests/test.sh`. The five checkpoints and their weights are (see `tests/test.sh` lines 26, 34, 44, 54, 64):

| Checkpoint | Lines in test.sh | Bash increment | Weight |
|---|---|---|---|
| C0: valid JSON produced | 26 | `SCORE += 10` | 0.10 |
| C1: `total_eligible_users == 7` | 34 | `SCORE += 25` | 0.25 |
| C2: Engineering groupby correct | 44 | `SCORE += 25` | 0.25 |
| C3: Sales groupby correct | 54 | `SCORE += 20` | 0.20 |
| C4: HR groupby correct | 64 | `SCORE += 20` | 0.20 |

For a sub-task environment targeting checkpoint C2 specifically, `R` returns `0.25` when the Engineering assertion passes and `0.0` otherwise.

**Reward normalisation:** `checkpoint_weight` is kept in `[0, 1]` (directly from `SCORE / 100.0`, line 66 of `test.sh`). When a `GenEnv` covers multiple checkpoints (a rarer multi-phase environment), the reward is the sum of all satisfied checkpoint weights up to the exit step.

### 1.6 Transition function `T`

The transition function is **implicit in the Harbor container runtime**. disteval does not model environment transitions analytically; instead, it re-runs the agent's tool calls against the same Docker container image (`tasks/medium-2/environment/Dockerfile`) and observes the resulting file-system and process state. This matches the existing Harbor task execution model precisely.

From the RecursionEngine's perspective:

```
T(s_t, a_t) → s_{t+1}  =  execute a_t in the running container,
                             capture stdout/stderr and file-system diffs,
                             increment step_index by 1
```

The container image is fixed per parent task. Sub-task environments re-use the same image, initialised at the entry boundary state via context prefix replay.

### 1.7 Termination conditions `Z`

A sub-task episode terminates when any of the following is true:

```
Z = {
  "success":     exit_condition satisfied (test checkpoint passes OR monitor p_high >= 0.70),
  "failure":     a hard failure observed (e.g., Python exception that prevents further progress),
  "truncation":  step_index >= exit_step - entry_step + MAX_EXTRA_STEPS (step budget exhausted),
  "timeout":     wall-clock time exceeds verifier timeout_sec (from task.toml line 14),
  "stack_limit": sub-task depth >= RecursionEngineConfig.max_depth (safety cap)
}
```

**Grounding:** `timeout_sec = 60.0` for `medium-2` (from `tasks/medium-2/task.toml` line 14). `exit_step` is the boundary step from the parent task decomposition (`SubTaskDefinition.exit_step`, Phase 2 master report section 2.2). `MAX_EXTRA_STEPS` is a configurable buffer (default: 5 steps) to avoid penalising the agent for a slightly longer but still successful path.

`"success"` has a dual criterion: the hard criterion is the test-script checkpoint (preferred when available); the soft criterion is the monitor's `p_high >= 0.70` threshold (fallback, using `TrajectoryMonitor.check()` at `disteval/trajectory_monitor.py` lines 488–524). This matches the `divergence_confidence = 0.70` default in `RecursionEngineConfig` (Phase 2 master report, section 2.4).

---

## 2. From `SubTaskDefinition` to a generated environment

The `RecursionEngine` (Phase 2 master report, section 2.2) produces `SubTaskDefinition` objects with these fields:

```python
@dataclass
class SubTaskDefinition:
    sub_task_id:    str     # e.g. "medium-2::phase-2"
    parent_task:    str     # e.g. "disteval/medium-rest-client"
    sub_task_depth: int     # recursion depth (0 = root)
    entry_step:     int     # first tool-call index in the parent trajectory
    exit_step:      int     # last tool-call index (inclusive)
    phase_tag:      str     # "write" | "exec" | "verify" | "unknown"
    instruction:    str     # human-readable sub-task description
    estimated_q_star: float # best sub-task score observed or estimated
    estimated_q_bar:  float # mean sub-task score
    kind:           str     # "solid" | "recoverable" | "stuck"
```

The **environment generator** (proposed new module `disteval/environment_generator.py`, to be defined in Phase 4) maps a `SubTaskDefinition` to a `GenEnv` via the following procedure:

### Step 1 — Instruction derivation

The sub-task instruction string is built from:

1. **Primary:** the `instruction` field of `SubTaskDefinition`, which is set by `RecursionEngine.find_phase_boundaries()` using either the test-script checkpoint description or a template derived from the phase tag.
2. **Fallback:** the parent `instruction.md` content, prepended with a scoping sentence such as: "Your task is to complete **phase 2** of the REST client problem. The first two steps (fetching users and filtering by age) have already been completed. Focus only on: computing `avg_salary` per department for the **Engineering** department and writing the correct `departments.Engineering` key to `/app/summary.json`."

**Grounding:** The parent instruction is at `tasks/medium-2/instruction.md`. The checkpoint descriptions are in `tests/test.sh` as inline comments and assertion messages (lines 28, 36, 46, 50, 59).

### Step 2 — Context prefix extraction

The generator extracts `steps[0 : entry_step]` from one of the high-scoring parent trajectories (the `reinforce` trajectory from the `TrainingPair`). This prefix is stored in the `GenEnv.context_prefix` field and is **not** part of the training episode — it is used only for state synthesis.

**Grounding:** This is the `entry_context` concept from Phase 2B, `research/phase2b_decomposition_algorithm.md` lines 153–156.

### Step 3 — Entry state synthesis

The generator replays the context prefix against the parent task's Docker container image (`tasks/medium-2/environment/Dockerfile`) to produce the file-system and terminal state at `entry_step`. For `medium-2`, this means:
- Starting the mock Flask server (`python3 /app/mock_server.py &` from `tests/test.sh` lines 5–8).
- Replaying any `write_file` and `run_shell_command` tool calls in `steps[0:entry_step]`.
- Capturing which files now exist under `/app/` and what their contents are.

The synthesized state is serialised as a JSON snapshot:

```json
{
  "files": {
    "/app/mock_server.py": "<content>",
    "/app/client.py": "<partial content if already created>"
  },
  "processes": [
    {"name": "mock_server.py", "port": 5000, "status": "running"}
  ],
  "env_vars": {}
}
```

### Step 4 — Reward wiring

The generator parses `tests/test.sh` to extract the checkpoint weight for the sub-task's exit condition. For the groupby sub-task (`phase-2`, C2), the weight is `0.25` (line 44 of `test.sh`, `SCORE += 25`).

The `GenEnv.reward_fn` is stored as a description (not executable code) of which bash assertion block to run and what the weight is. The actual evaluation is still performed by `test.sh` — the generator produces metadata that tells the runner which lines of `test.sh` to execute for this sub-task.

### Step 5 — Test snippet generation

For each sub-task, the generator produces a **sub-task test script** that:
1. Runs only the checkpoint(s) covered by the sub-task.
2. Writes `reward_c{i}.txt` to `/logs/verifier/` instead of the combined `reward.txt`.
3. Does not require the full environment to be set up from scratch (relies on the entry state synthesis from Step 3).

For `medium-2::phase-2` (Engineering groupby), the sub-task test snippet would be:

```bash
#!/bin/bash
# Sub-task test: medium-2::phase-2 (Engineering groupby)
# Entry state: mock server running, /app/client.py already exists (from context prefix)
python3 /app/client.py 2>/dev/null

SCORE=0
if [ -f /app/summary.json ]; then
    python3 -c "
import json
d = json.load(open('/app/summary.json'))
eng = d['departments']['Engineering']
assert eng['count'] == 3, f'Engineering count wrong: {eng[\"count\"]}'
assert eng['avg_salary'] == 111666.67, f'Engineering avg wrong: {eng[\"avg_salary\"]}'
print('ok')
" 2>/dev/null && SCORE=100
fi
python3 -c "print($SCORE / 100.0)" > /logs/verifier/reward_c2.txt
```

**Grounding:** The checkpoint assertion is taken verbatim from `tasks/medium-2/tests/test.sh` lines 37–44. The `reward_c{i}.txt` naming convention is proposed in Phase 2B section 4.3.

### Step 6 — Metadata assembly

The `GenEnv` is assembled from all of the above and serialised to JSON (see Section 4 for the full example).

---

## 3. Mapping test.sh checkpoints to reward signals

`tests/test.sh` currently writes a single scalar to `/logs/verifier/reward.txt` (line 66). The environment generation layer introduces a **per-checkpoint reward mapping** that exposes the intermediate reward structure.

### 3.1 TestSuiteParser

A new helper class `TestSuiteParser` (proposed in Phase 2B, section 3.4) reads a `test.sh` file and extracts:

```python
@dataclass
class CheckpointDef:
    checkpoint_index: int     # 0-based
    weight: float             # extracted from SCORE += N; normalised by total
    assertion_lines: list[str]  # the bash block that performs the check
    description: str          # comment or assertion message in the script
    reward_file: str          # "/logs/verifier/reward_c{i}.txt"
```

For `tasks/medium-2/tests/test.sh`, `TestSuiteParser` would return:

```python
[
  CheckpointDef(0, 0.10, lines 25–26,  "valid JSON produced",       "reward_c0.txt"),
  CheckpointDef(1, 0.25, lines 29–34,  "total_eligible_users == 7", "reward_c1.txt"),
  CheckpointDef(2, 0.25, lines 37–44,  "Engineering groupby",       "reward_c2.txt"),
  CheckpointDef(3, 0.20, lines 47–54,  "Sales groupby",             "reward_c3.txt"),
  CheckpointDef(4, 0.20, lines 57–64,  "HR groupby",                "reward_c4.txt"),
]
```

**Parsing heuristic:** `TestSuiteParser` identifies checkpoint blocks by the pattern `SCORE=$((SCORE + N))` or `SCORE=$((SCORE + N))` and reads the preceding `python3 -c "..."` block. The `weight = N / 100` is derived from the test-script convention that `reward.txt = SCORE / 100.0` (line 66). This heuristic covers the `medium-2` format exactly; other tasks may require the parser to be extended in Phase 4.

### 3.2 Reward signal mapping

The environment generator maps `CheckpointDef` objects to sub-task reward signals according to the following rules:

| Sub-task phase | Covers checkpoints | Reward signal |
|---|---|---|
| Single-checkpoint sub-task | C_i | Sparse: `checkpoint_weight_i` on pass, `0.0` on fail |
| Multi-checkpoint sub-task | C_i through C_j | Cumulative: sum of weights for all checkpoints passed up to exit |
| Partial-credit sub-task | Entry at C_i, exit before C_{i+1} | Monitor proxy: `p_high` as a soft reward `∈ [0, 1]` (requires no test instrumentation) |

The preferred mapping is **single-checkpoint** per sub-task (one `GenEnv` per checkpoint), because:
1. It preserves the RMDP convergence guarantee for 1-exit MDPs (Phase 2 master report, section 7, question 6).
2. The reward signal is unambiguous: the agent either passes the checkpoint or it does not.
3. Training pairs align exactly with the checkpoint structure, making the DPO signal maximally focused.

### 3.3 Reward propagation to parent

When sub-task rewards are aggregated back to the parent task score, the weighted-sum rule (Phase 2B, `research/phase2b_decomposition_algorithm.md` section 5.4) applies:

```
parent_score = Σ_i  checkpoint_weight_i × I(checkpoint_i passed)
```

For `medium-2`, this is `0.10 × C0 + 0.25 × C1 + 0.25 × C2 + 0.20 × C3 + 0.20 × C4`, which recovers the original `test.sh` total exactly (final line: `python3 -c "print($SCORE / 100.0)"`, `test.sh` line 66).

### 3.4 Backward compatibility with reward.txt

The existing `reward.txt` is written last by `test.sh` and is read by `RecordStore` / `harbor_jobs.py` to populate `EpisodeRecord.score`. To stay backward-compatible, the sub-task test snippets write `reward_c{i}.txt` **in addition to**, not instead of, `reward.txt`. The parent `test.sh` continues to produce the combined score. `reward_c{i}.txt` files are optional outputs that the environment generator looks for when computing per-checkpoint sub-task scores; if they are absent, the structural proxy is used instead.

---

## 4. Concrete JSON example: `medium-2::phase-2` (Engineering groupby sub-task)

The following is a concrete JSON representation of the generated environment for the Engineering groupby sub-task of `disteval/medium-rest-client`. This is the phase-2 sub-task identified in the Phase 2 master report (section 6) and Phase 2B (section 7.1).

```json
{
  "schema_version": "genenv/1.0",
  "env_id": "medium-2::phase-2",
  "parent_task": "disteval/medium-rest-client",
  "parent_task_dir": "tasks/medium-2",
  "sub_task_depth": 1,
  "phase_tag": "verify",
  "cycle": 1,
  "generated_by": "RecursionEngine v0.1 / disteval",

  "origin": {
    "entry_step": 5,
    "exit_step": 9,
    "boundary_confidence": 0.81,
    "boundary_source": "checkpoint_aligned",
    "divergence_from_trajectory": "jobs/run_C/disteval-run-C/medium-2__XYZ/agent/trajectory.json",
    "divergence_step_in_parent": 7
  },

  "instruction": {
    "text": "A mock API server is running at http://localhost:5000. A Python script at /app/client.py already exists and successfully fetches all users and filters to those aged 30 or older (7 users total).\n\nYour task: extend /app/client.py so that it groups the eligible users by department and, for each department, computes the average salary (rounded to 2 decimal places). Write the result to /app/summary.json.\n\nSpecifically, ensure that for the Engineering department the output contains:\n  {\"count\": 3, \"avg_salary\": 111666.67}\n\n(Engineering employees aged >= 30: Alice Johnson $95,000, Carol White $110,000, Frank Miller $130,000.)",
    "source": "checkpoint_description + parent_instruction.md",
    "parent_instruction_path": "tasks/medium-2/instruction.md"
  },

  "state": {
    "initial_fs": {
      "files": {
        "/app/mock_server.py": "tasks/medium-2/environment/app/mock_server.py",
        "/app/client.py": {
          "note": "partially_written_by_context_prefix",
          "content_summary": "Fetches /users, filters age>=30, stores eligible list. Does not yet group by department.",
          "synthesis_source": "context_prefix_replay"
        }
      },
      "processes": [
        {"name": "mock_server.py", "host": "localhost", "port": 5000, "status": "running"}
      ],
      "env_vars": {}
    },
    "context_prefix": {
      "entry_step": 5,
      "step_count": 5,
      "tool_sequence": ["read_file", "write_file", "run_shell_command", "read_file", "write_file"],
      "synthesis_method": "replay",
      "source_trajectory": "jobs/run_C/disteval-run-C/medium-2__high_scoring/agent/trajectory.json"
    }
  },

  "actions": {
    "space": "disteval_tool_vocab",
    "allowed_tools": [
      "write_file", "run_shell_command", "exec_command",
      "read_file", "list_directory"
    ],
    "step_budget": 10,
    "max_extra_steps": 5
  },

  "observation": {
    "instruction_field": "instruction.text",
    "context_summary_max_tokens": 300,
    "fs_listing_path": "/app",
    "memory_prompt": {
      "enabled": true,
      "retrieval_k": 3,
      "query_description": "group by department compute average salary json python",
      "source": "TrajectoryMemory.retrieve_for_new_task"
    }
  },

  "reward": {
    "type": "sparse_checkpoint",
    "checkpoints": [
      {
        "checkpoint_index": 2,
        "weight": 0.25,
        "description": "Engineering groupby correct: count==3 and avg_salary==111666.67",
        "assertion_source": "tasks/medium-2/tests/test.sh",
        "assertion_lines": "37-44",
        "reward_file": "/logs/verifier/reward_c2.txt",
        "test_snippet_inline": "python3 -c \"\nimport json\nd = json.load(open('/app/summary.json'))\neng = d['departments']['Engineering']\nassert eng['count'] == 3\nassert eng['avg_salary'] == 111666.67\nprint('ok')\n\" 2>/dev/null"
      }
    ],
    "step_penalty": 0.0,
    "max_reward": 0.25,
    "reward_normalised": true
  },

  "termination": {
    "success_condition": {
      "type": "checkpoint_pass",
      "checkpoint_index": 2,
      "soft_fallback": {
        "type": "monitor_p_high",
        "threshold": 0.70,
        "source": "TrajectoryMonitor.check"
      }
    },
    "failure_conditions": [
      {"type": "python_exception", "description": "Unhandled exception in client.py"},
      {"type": "missing_output_file", "path": "/app/summary.json"}
    ],
    "truncation": {
      "type": "step_budget",
      "max_steps": 15,
      "timeout_sec": 60.0,
      "source": "tasks/medium-2/task.toml line 14"
    }
  },

  "taxonomy": {
    "kind": "recoverable",
    "estimated_q_star": 1.0,
    "estimated_q_bar": 0.333,
    "gap": 0.667,
    "consistency": 0.333,
    "priority_score": 0.278,
    "sub_task_weight_in_parent": 0.25,
    "environment_status": "active",
    "scores_observed": [0.0, 0.0, 1.0],
    "reinforce_indices": [2],
    "contrast_indices": [0, 1]
  },

  "training_pairs": [
    {
      "reinforce_traj_path": "jobs/run_C/disteval-run-C/medium-2__high/agent/trajectory.json",
      "contrast_traj_path":  "jobs/run_C/disteval-run-C/medium-2__low_A/agent/trajectory.json",
      "reinforce_score": 1.0,
      "contrast_score":  0.0,
      "gap": 1.0,
      "entry_step": 5,
      "exit_step": 9,
      "structural_divergence_step": 7,
      "entry_context_steps": [0, 5]
    }
  ],

  "environment_file_layout": {
    "note": "Files below are written to a sub-task directory under the parent task.",
    "task_toml":     "tasks/medium-2/generated/phase-2/task.toml",
    "instruction_md":"tasks/medium-2/generated/phase-2/instruction.md",
    "test_sh":       "tasks/medium-2/generated/phase-2/tests/test.sh",
    "dockerfile":    "tasks/medium-2/environment/Dockerfile",
    "entry_state_json": "tasks/medium-2/generated/phase-2/environment/entry_state.json"
  }
}
```

**Notes on the example:**

- `entry_step = 5`: after 5 tool calls in the high-scoring parent trajectory, the agent has fetched users and written a partial `client.py` but has not yet added the groupby logic. The sub-task starts from this state.
- `exit_step = 9`: by tool call 9 in the high-scoring trajectory, the groupby logic was complete and the Engineering assertion passes.
- `kind = "recoverable"`: based on the three-attempt score vector `[0, 0, 1]` from Phase 2B section 7.2. The agent solved it once but not consistently.
- `estimated_q_star = 1.0`, `estimated_q_bar = 0.333`: sub-task scores derived from whether checkpoint C2 was satisfied, not from the parent's full score. The parent's full score of `[0.35, 0.0, 1.0]` is aggregated from all five checkpoints.
- `boundary_source = "checkpoint_aligned"`: the boundary was found by both structural divergence (monitor) and test-script checkpoint alignment, giving high confidence.
- The `environment_file_layout` shows that generated sub-task environments are placed under `tasks/<parent>/generated/<phase>/`. They reuse the parent Dockerfile and add only the sub-task-specific `instruction.md`, `tests/test.sh` snippet, and `entry_state.json`.

---

## 5. Format choice: Gymnasium, Harbor, or disteval-specific?

### 5.1 Options considered

| Format | Description | Pros | Cons |
|---|---|---|---|
| **Gymnasium-compatible** | Implement the Gymnasium `Env` interface (`reset()`, `step()`, `observation_space`, `action_space`) | Standard; works with existing RL training code (PPO, SAC, etc.) | Gymnasium assumes numeric/discrete spaces; LLM tool calls are structured text objects. `observation_space` and `action_space` would need to be `gym.spaces.Text` or custom, reducing compatibility with standard algorithms. Also adds a new runtime dependency (`gymnasium`). |
| **Harbor-task-compatible** | Generate new Harbor task directories (`task.toml` + `instruction.md` + `environment/` + `tests/test.sh`) | Zero new dependencies; reuses the exact infra that already runs disteval tasks; any Harbor-compatible runner can execute the sub-task. | Harbor is a harness, not an RL env; no `step()`/`reset()` API; reward is only observed at episode end, not mid-episode. |
| **disteval-specific `GenEnv` JSON** | A new JSON schema (defined in this document) that extends `CURRICULUM_FORMAT.md` and can be rendered into Harbor task files *and* queried by the training loop. | No new dependencies; grounded in the existing task format; serialisable; can express all fields (entry state, reward wiring, taxonomy, training pairs) that the RL loop needs; extensible in Phase 4 to a thin wrapper that materialises Harbor task dirs. | Not directly executable by a generic RL framework; requires Phase 4 implementation of the `EnvironmentGenerator.materialise()` method to produce actual files. |

### 5.2 Recommended choice: disteval-specific `GenEnv` JSON (this document's format)

**Justification:**

1. **No new runtime dependencies.** disteval's stated constraint (SKILL.md lines 86–91) is that no new dependencies are introduced unless justified. Gymnasium would introduce one. The `GenEnv` JSON format has none.

2. **Directly extends the existing CURRICULUM_FORMAT.md.** The `GenEnv` JSON is structurally similar to a curriculum item (it has `task`, `kind`, `gap`, `training_pairs`) and can be appended to `SelfImprovementPlan.curriculum` with a `"kind": "generated_env"` marker. This keeps all serialisation in the existing `SelfImprovementPlan.to_dict()` path (`disteval/self_engine.py`, lines 169–213).

3. **Compatible with Harbor task execution.** The `environment_file_layout` section of the `GenEnv` JSON maps directly to the Harbor task directory structure. A thin `EnvironmentGenerator.materialise()` method in Phase 4 can render any `GenEnv` into a proper Harbor task directory that Harbor can run without modification.

4. **LLM agent actions are not numeric vectors.** Gymnasium's design assumes box or discrete action spaces. An LLM's action is a structured tool call; the natural representation is a JSON dict, not a vector. Wrapping this in Gymnasium would add indirection without benefit.

5. **Reward at exit is already the model.** Harbor tasks report reward by writing to `/logs/verifier/reward.txt` at the end of the episode. The `GenEnv` format embraces this model (checkpoint reward is observed at the end of the sub-task) rather than trying to retrofit continuous reward shaping. This also matches the sparse-reward convention used in the Phase 1 paper (arXiv:2206.11430) for RMDP exit rewards.

6. **Gymnasium compatibility is optional later.** If a Phase 4 implementer wants Gymnasium compatibility for a specific sub-task (e.g., to use PPO with a small action vocabulary), a thin `GenEnvGym` adapter can wrap `GenEnv.materialise()` without changing the core schema.

**The `GenEnv` format is therefore the primary representation.** Harbor task files are the materialised form. Gymnasium wrappers are deferred to Phase 4 if needed.

---

## 6. Entry-state handling at the sub-task boundary

The central practical challenge of sub-task environments is that an LLM agent's trajectory produces **side effects** (files written, servers started, shell state accumulated) that must be reproduced exactly for the sub-task to be valid. This section specifies how the environment generator captures and replays that state.

### 6.1 Three tiers of entry-state fidelity

| Tier | Method | When used | Cost |
|---|---|---|---|
| **Tier 1: Context prefix replay** | Re-execute all `write_file` and `run_shell_command` calls in `steps[0:entry_step]` against a fresh container | Always available from the trajectory JSON; no new container infrastructure needed | O(entry_step) wall time; non-deterministic if tool calls have random outputs |
| **Tier 2: File snapshot** | Capture `/app/` file-system state at `entry_step` from a real run and store it in `entry_state.json` | When a high-scoring run exists and the runner can be instrumented | Requires runner instrumentation; produces exact state but may be large |
| **Tier 3: Synthetic state** | Generator synthesises the file system based on `phase_tag` and the parent instruction | Fallback when no trajectory for the context prefix exists (e.g., STUCK parent) | Potentially inaccurate; must be validated before use |

**Default: Tier 1 (context prefix replay).**

The replay procedure for `medium-2::phase-2` (entry_step = 5):

```
1. Start fresh container from tasks/medium-2/environment/Dockerfile
2. Start mock server: python3 /app/mock_server.py &
3. Replay steps[0:5] from the high-scoring parent trajectory:
   - read_file("/app/mock_server.py")           # step 0
   - write_file("/app/client.py", partial_impl)  # step 1
   - run_shell_command("python3 /app/client.py") # step 2
   - read_file("/app/client.py")                 # step 3
   - write_file("/app/client.py", updated_impl)  # step 4
4. Container is now in the state where /app/client.py exists and fetches users,
   but does not yet group by department.
5. This is the entry state for the sub-task.
```

**Non-determinism handling:** If the context prefix includes tool calls whose output is non-deterministic (e.g., a shell command that writes a timestamp), the replayed state may differ from the original. The environment generator handles this by:
1. Preferring to replay **only write_file calls** (deterministic) and ignoring read/exec calls that do not produce files.
2. For `run_shell_command` calls that produce output files, using the captured output from the original trajectory's `prev_tool_output` field as a mock stdin/stdout fixture.

### 6.2 Mock server state

For `medium-2`, the mock server (`tasks/medium-2/environment/app/mock_server.py`) is a Flask app that serves static data (lines 7–18). It is stateless — every call to `/users` returns the same JSON array. Therefore, the mock server does not need state serialisation: it only needs to be running, which the entry state synthesis handles by including the `python3 /app/mock_server.py &` invocation in the replay sequence.

For tasks with stateful services (e.g., a database that is modified by context prefix operations), Tier 2 (file snapshot) would be required. This is deferred to Phase 4.

### 6.3 Context prefix as conditioning, not training target

A key design decision is that the context prefix steps are **not part of the training episode**. They are provided to the agent as a read-only observation (summarised in `context_summary`) so the agent understands what state the environment is in, but the DPO loss is not computed over them. This mirrors the standard instruction-following setup where a task description is given but not optimised over.

**Grounding:** `entry_context: list[dict]` in the `TrainingPair` extension (Phase 2B, lines 173–176) already distinguishes context from episode steps.

### 6.4 Partial output files

When the context prefix includes a partial implementation (e.g., `client.py` fetches users but does not group them), the partial file is present in the entry state. The agent's first action in the sub-task episode may be to read this file and extend it. The environment generator ensures this file is included in `initial_fs.files` with its partial content.

### 6.5 Tool-call state at boundary

For `medium-2`, the tool-call sequence up to `entry_step = 5` produced a `client.py` that:
- Fetches from `http://localhost:5000/users`
- Filters `age >= 30`
- Does **not** yet group by department

The agent receives `client.py` at its partial state and must complete the groupby logic to satisfy the C2 checkpoint. The observation includes a `context_summary` that describes what was already done, and a `fs_listing` that shows `/app/client.py` exists, so the agent knows to build on it rather than starting from scratch.

---

## 7. Open questions for Phase 3B/3C and Phase 4

### 7.1 For Phase 3B (self-improvement loop)

**Q1: Environment stability threshold.** When does a `GenEnv` become stable enough to include in the active Harbor evaluation suite? The current proposal is `kind == "recoverable"` → `status = "active"`, but sub-tasks with `q_star < 0.5` (estimated from the monitor proxy, not a real run) may be too noisy. Phase 3B should define a minimum `q_star_confidence` threshold.

**Q2: Reward shaping vs. sparse reward.** The current schema uses sparse checkpoint rewards (`0.0` or `checkpoint_weight`). Phase 3B should evaluate whether adding a small step penalty (e.g., `-0.001` per step) or a shaped intermediate reward (e.g., `monitor.p_high` as a proxy) improves training efficiency without introducing reward hacking. The `step_penalty: 0.0` default in Section 1.5 of this document is conservative; Phase 3B should experiment with non-zero values.

**Q3: Distribution evolution.** When SOLID sub-tasks are retired from the active environment distribution, the remaining sub-tasks represent a harder distribution. Phase 3B should specify how the curriculum re-ranks after retirement, and whether SOLID sub-tasks should occasionally be re-evaluated (spot-checked) to detect regression.

**Q4: Entry-state quality assurance.** Context prefix replay (Tier 1) is non-deterministic for tool calls with external side effects. Phase 3B should specify a validation step that confirms the replayed state is equivalent to the original state (e.g., by running the C0/C1 checkpoints against the replayed state before starting the sub-task episode).

### 7.2 For Phase 3C (distributed evals)

**Q5: Boundary voting for cross-agent environments.** When multiple agents produce different `entry_step` / `exit_step` values for the same sub-task (because their trajectories differ in length), the consensus graph uses confidence-weighted voting (Phase 3C, section 2.3). Phase 3C should specify the voting rule precisely: is it a weighted median, a max-confidence pick, or a union (widest window)?

**Q6: Cross-agent training pairs and entry-state compatibility.** If Agent A (Claude) and Agent B (Codex) both run `medium-2`, their `context_prefix` steps for the same sub-task may differ (different tool sequences, different partial `client.py` implementations). A cross-agent training pair (Claude reinforce, Codex contrast) requires that both trajectories' entry states are compatible — the agent being trained can be placed in either state. Phase 3C should define a compatibility check and a fallback (use only self-agent pairs when cross-agent entry states are incompatible).

**Q7: Cross-agent sharing consent model.** Phase 3C flags this as a privacy/attribution concern (section 6). For the design to be complete, Phase 4 needs a minimal consent model: at minimum, a `sharing_permitted: true/false` flag per trajectory in the `EnvironmentPool`, with `false` as the default.

### 7.3 For Phase 4 (prototype integration)

**Q8: `EnvironmentGenerator.materialise()` API.** The `GenEnv` JSON defined in this document must be rendered into actual Harbor task directories (`task.toml`, `instruction.md`, `tests/test.sh`, `environment/entry_state.json`). Phase 4 must define the exact method signature and the file naming convention for generated sub-tasks. The placeholder layout in Section 2 (`tasks/medium-2/generated/phase-2/`) is a proposal; Phase 4 may choose a different location (e.g., `generated_envs/medium-2::phase-2/`).

**Q9: `task.toml` for generated environments.** The generated `task.toml` for a sub-task environment must set appropriate `timeout_sec` values (the sub-task is shorter than the parent; `timeout_sec = 60.0` in `tasks/medium-2/task.toml` line 14 may be excessive for a single-checkpoint sub-task) and must carry the `sub_task_id`, `parent_task`, and `cycle` as metadata. Phase 4 should define the `task.toml` schema extension for generated sub-tasks.

**Q10: TestSuiteParser robustness.** The `TestSuiteParser` described in Section 3.1 uses a pattern-matching heuristic for `SCORE += N` increments. Phase 4 should assess whether this heuristic works for all existing tasks in the `tasks/` directory (e.g., `tasks/easy-1/`, `tasks/hard-1/`) and document which task structures require a more general parser.

**Q11: Sub-task environment versioning.** As the agent improves across cycles, the `entry_step` / `exit_step` boundaries of a sub-task may shift (Phase 3B, section 3.2). The `GenEnv` JSON includes a `cycle` field to track this. Phase 4 should define a versioning scheme (e.g., `env_id = "medium-2::phase-2::cycle-1"`) and a merge rule for when the same logical sub-task is regenerated in a later cycle with a shifted boundary.

**Q12: STUCK sub-task environments — what do we generate?** For STUCK sub-tasks (where `q_star = 0`), the environment generator has no successful trajectory to use as a reinforce demonstration. The Phase 2B fallback is to use a `TrajectoryMemory`-retrieved demonstration from a similar task. Phase 4 must specify the exact format of a STUCK sub-task `GenEnv` — specifically, whether the `context_prefix` can come from a *different agent's* successful trajectory on the same sub-task (cross-agent replay), and what the entry-state compatibility requirements are for that case.

---

## 8. Summary: how this design fits into the disteval pipeline

```
disteval pipeline (with environment generation layer)

EVAL CYCLE n
────────────────────────────────────────────────────────────────────────
1. Harbor runs agent on parent task suite → job_dir_n
2. right_tail_analysis(store_n) → RightTailReport (SOLID / RECOVERABLE / STUCK)
3. RecursionEngine.decompose(report_n) → SubTaskGraph
   ↓
   For each SubTaskDefinition in SubTaskGraph:
4. EnvironmentGenerator.generate(sub_task_def, parent_traj_paths) → GenEnv (JSON)
   - TestSuiteParser reads tasks/medium-2/tests/test.sh → CheckpointDef list
   - Selects checkpoint matching sub_task_def.phase_tag
   - Synthesizes entry state via context prefix replay (Tier 1)
   - Assembles GenEnv JSON (schema: this document, Section 4)
5. EnvironmentRegistry.update(gen_envs) → retire SOLID, keep RECOVERABLE, escalate STUCK
6. EnvironmentGenerator.materialise(gen_env) → tasks/medium-2/generated/phase-2/ directory
7. SelfImprovementPlan includes GenEnv training_pairs in curriculum
8. DPO trainer fine-tunes on sub-task training pairs
9. go to EVAL CYCLE n+1
```

**New modules required (Phase 4 implementation targets):**
- `disteval/environment_generator.py` — `EnvironmentGenerator` class with `generate()` and `materialise()` methods.
- `disteval/test_suite_parser.py` — `TestSuiteParser` and `CheckpointDef` dataclass.
- `disteval/environment_registry.py` — `EnvironmentRegistry` with `update()`, `retire()`, `status` tracking, and `environment_registry.jsonl` persistence.

**Existing modules that remain unchanged:**
- `disteval/self_engine.py` — consumes `GenEnv.training_pairs` via the existing `TaskImprovement.training_pairs` field (with the Phase 2C extension for `sub_task_depth` and `entry_step`).
- `disteval/right_tail.py` — no changes; its output feeds the generation pipeline unchanged.
- `disteval/trajectory_monitor.py` — no changes; its `check()` and `find_phase_boundaries()` methods are called by `RecursionEngine` before `EnvironmentGenerator`.
- `disteval/training_sim.py` — no changes; the per-sub-task reward weights become the `w_i` in `_fast_apply_improvement()` (the existing weighted-sum rule at lines 351–396).
- `tasks/medium-2/tests/test.sh` — no changes to the existing file; `TestSuiteParser` reads it read-only. The generated sub-task test snippets are placed in `tasks/medium-2/generated/phase-2/tests/test.sh`, not in the original.
