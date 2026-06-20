# disteval Trajectory Format

disteval reads two kinds of data from agent runs: **episode records** (scalar
outcomes per attempt) and **trajectory files** (the sequence of tool calls the
agent made). This document specifies both.

---

## 1. Episode Records — what disteval requires to compute metrics

Episode records are the minimum unit disteval needs. A record is one attempt by
one agent on one task.

### JSONL format (one object per line)

```jsonl
{"run_id":"run_001","model":"my-agent","task":"word-count","episode":0,"score":1.0,"success":true,"difficulty":"easy","trajectory":"/runs/run_001/traj_0.json"}
{"run_id":"run_001","model":"my-agent","task":"word-count","episode":1,"score":0.0,"success":false,"difficulty":"easy","trajectory":"/runs/run_001/traj_1.json"}
{"run_id":"run_001","model":"my-agent","task":"log-parser","episode":0,"score":0.5,"success":false,"difficulty":"medium","trajectory":"/runs/run_001/traj_2.json"}
```

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Identifies which eval repetition this belongs to. Use the same value across all attempts in one eval run. |
| `model` | string | Agent/model name. Used to group runs by agent. |
| `task` | string | Task identifier. The same string across all attempts at the same task. |
| `score` | float | Scalar outcome, 0.0–1.0. disteval never collapses this — it keeps every individual score. |

### Optional fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `episode` | int | 0 | Index of this attempt within (run_id, model, task). |
| `success` | bool | `score >= 0.99` | Binary success flag for pass@k / pass^k. |
| `difficulty` | string | `null` | Task difficulty tier. Enables stratified analysis. Any string works; "easy"/"medium"/"hard" are conventional. |
| `trajectory` | string | `null` | Path to the trajectory JSON file for this attempt. Required for training pair extraction. |
| `metadata` | object | `{}` | Any extra data (duration, step count, cost, etc.). Stored but not used by disteval metrics. |

Any other top-level string or number key is automatically promoted to a
stratification dimension (e.g. `"domain": "finance"` lets you call
`store.slice(domain="finance")`).

### Loading episode records

```python
from disteval.adapters.generic import load_records

store = load_records("my_runs.jsonl")
```

Or from Harbor's jobs/ directory:

```python
from disteval.adapters.harbor_jobs import load_harbor_job

store = load_harbor_job("jobs/run_A/", tasks_dir="tasks/")
```

---

## 2. Trajectory Files — required for training pair extraction

disteval uses trajectory files to:
1. Extract the reinforce/contrast trajectory pair for each RECOVERABLE task
2. Find the structural divergence step (where the successful run diverged from the failing run)
3. Feed trajectory memory for future-run retrieval

### Format

A trajectory file is a JSON object with a `steps` array. Each step represents
one agent turn and contains the tool calls made in that turn.

```json
{
  "steps": [
    {
      "message": "I need to read the task first.",
      "tool_calls": [
        {
          "function_name": "read_file",
          "arguments": {"file_path": "/app/task.md"},
          "tool_call_id": "tc_001"
        }
      ],
      "observation": {
        "results": [
          {
            "source_call_id": "tc_001",
            "output": "# Word Count\nWrite a script..."
          }
        ]
      }
    },
    {
      "message": "I will write the solution.",
      "tool_calls": [
        {
          "function_name": "write_file",
          "arguments": {
            "file_path": "/app/solution.py",
            "content": "import sys\n..."
          },
          "tool_call_id": "tc_002"
        }
      ]
    },
    {
      "message": "Now test it.",
      "tool_calls": [
        {
          "function_name": "run_shell_command",
          "arguments": {"command": "python3 /app/solution.py < test_input.txt"},
          "tool_call_id": "tc_003"
        }
      ],
      "observation": {
        "results": [
          {
            "source_call_id": "tc_003",
            "output": "words: 4\nlines: 2\nunique: 3"
          }
        ]
      }
    }
  ]
}
```

### Tool name aliases

disteval's `TrajectoryFeaturizer` normalizes tool names to these canonical
categories. Any tool name that contains the listed substring is mapped:

| Category | Matches |
|----------|---------|
| `write_file` | `write_file`, `str_replace_based_edit_tool`, `create_file` |
| `run_shell_command` | `run_shell_command`, `exec_command`, `bash`, `shell` |
| `read_file` | `read_file`, `view_file`, `cat` |
| `list_directory` | `list_directory`, `ls` |
| `search_tool` | `search_files`, `grep`, `ripgrep`, `find` |

Tool names outside these categories are still recorded but classified as
"other". If your agent uses different tool names, the structural analysis still
works — only the category labels change.

### Minimal trajectory (no observations)

If your agent does not capture tool outputs, the `observation` field can be
omitted entirely. disteval's training pair extraction uses tool sequence and
score; it does not require output content.

```json
{
  "steps": [
    {"tool_calls": [{"function_name": "write_file", "arguments": {"file_path": "/app/sol.py", "content": "..."}}]},
    {"tool_calls": [{"function_name": "run_shell_command", "arguments": {"command": "python3 /app/sol.py"}}]}
  ]
}
```

---

## 3. Harbor compatibility

If you run agents through [Harbor](https://github.com/av/harbor), disteval
reads its output directly without any conversion:

```
jobs/<run>/<trial_id>/
  result.json              # contains task_name, agent model, verifier_result.rewards.reward
  verifier/reward.txt      # scalar reward (optional, used if result.json reward missing)
  agent/trajectory.json    # trajectory in Harbor's ATIF format
```

```python
from disteval.adapters.harbor_jobs import load_harbor_job

store = load_harbor_job("jobs/run_A/", tasks_dir="tasks/")
```

---

## 4. Other eval frameworks

| Framework | Adapter |
|-----------|---------|
| Harbor | `disteval.adapters.harbor_jobs.load_harbor_job` |
| Inspect (UK AISI) | `disteval.adapters.inspect_log.load_inspect_json` |
| rliable | `disteval.adapters.rliable_bridge.to_rliable_dict` |
| Custom / any | `disteval.adapters.generic.load_records` (JSONL) |

---

## 5. Minimum viable setup (5 minutes)

```bash
pip install disteval
```

```python
# your_eval.py — whatever you use to run your agent
import json

results = []
for task in tasks:
    for attempt in range(3):
        score, traj = run_agent(agent, task)   # your function
        save_trajectory(traj, f"runs/traj_{task}_{attempt}.json")
        results.append({
            "run_id": "run_001",
            "model": "my-agent",
            "task": task,
            "episode": attempt,
            "score": score,
            "difficulty": task_difficulty[task],
            "trajectory": f"runs/traj_{task}_{attempt}.json",
        })

with open("runs.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
```

```bash
disteval report runs.jsonl --agent my-agent
disteval engine runs.jsonl --agent my-agent --output plan.json
```

The report shows your full score distribution. The engine outputs a ranked
list of tasks with training pairs ready to feed into DPO.
