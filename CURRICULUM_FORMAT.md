# disteval Curriculum Format

This document specifies the JSON output produced by `disteval engine` /
`SelfEngine.run_cycle()`. A DPO trainer, fine-tuning pipeline, or any
downstream tooling can consume this file directly.

---

## Overview

The curriculum file is a single JSON object. The top-level fields describe the
agent's current state; the `curriculum` array is an ordered list of tasks to
train on, ranked by training leverage (highest first).

---

## Top-level fields

```json
{
  "agent_name": "Gemini CLI",
  "model_name": "gemini-cli",
  "cycle": 1,
  "n_tasks_total": 6,
  "n_solid": 2,
  "n_recoverable": 3,
  "n_stuck": 1,
  "consistency_index": 0.808,
  "recoverable_score_left": 0.833,
  "predicted_total_gain": 0.117,
  "cycle_complete": false,
  "n_trajectories_loaded": 14,
  "curriculum": [ ... ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | string | Agent identifier passed to `--agent`. |
| `model_name` | string | Model identifier passed to `--model`. |
| `cycle` | int | Which improvement cycle this plan belongs to. Increment each time you run, train, and re-eval. |
| `n_tasks_total` | int | Total distinct tasks evaluated. |
| `n_solid` | int | Tasks where the agent is already consistent (Q\* = Q̄). No training needed. |
| `n_recoverable` | int | Tasks where Q\* > 0 and Q\* > Q̄. DPO pairs exist. |
| `n_stuck` | int | Tasks where Q\* = 0. Agent never solved them — no training pairs exist. |
| `consistency_index` | float | κ = Q̄ / Q\* averaged across all tasks (0–1). 1.0 = perfectly consistent. |
| `recoverable_score_left` | float | Sum of Δ(t) = Q\*(t) − Q̄(t) across all RECOVERABLE tasks. The total score improvement available without any new capability. |
| `predicted_total_gain` | float | Estimated mean score gain if all curriculum training pairs are used. |
| `cycle_complete` | bool | True once `run_cycle()` has been called and the plan is populated. |
| `n_trajectories_loaded` | int | Number of trajectory files loaded from the job directories. |

---

## Curriculum item

Each entry in `curriculum` is one RECOVERABLE task, sorted descending by
`priority_score = gap × (1 − consistency)`.

```json
{
  "task": "disteval/easy-word-count",
  "difficulty": "easy",
  "kind": "recoverable",
  "current_q_star": 1.0,
  "current_q_bar": 0.667,
  "consistency": 0.667,
  "gap": 0.333,
  "priority_score": 0.111,
  "predicted_gain": 0.039,
  "predicted_gain_ci": [0.0, 0.089],
  "predicted_rounds_to_threshold": 28.0,
  "recommendation": "Use 1 reinforce/contrast pair(s). ...",
  "n_training_pairs": 1,
  "training_pairs": [ ... ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `task` | string | Task identifier. |
| `difficulty` | string | "easy" / "medium" / "hard" (from task.toml, or null if not set). |
| `kind` | string | Always "recoverable" in the curriculum (SOLID/STUCK tasks are omitted). |
| `current_q_star` | float | Agent's best score on this task across all runs. |
| `current_q_bar` | float | Agent's mean score on this task. |
| `consistency` | float | κ(t) = Q̄ / Q\*. How reliably the agent achieves its best. |
| `gap` | float | Δ(t) = Q\* − Q̄. Score left on table due to inconsistency. |
| `priority_score` | float | gap × (1 − consistency). Ranking key: tasks with high gap and low consistency come first. |
| `predicted_gain` | float | Estimated mean score gain on this task from one DPO round (analytic: 0.4 × gap × consistency × 1.5 DPO bonus). |
| `predicted_gain_ci` | [float, float] | 70% CI on predicted gain: [gain × 0.7, gain × 1.3]. |
| `predicted_rounds_to_threshold` | float | Estimated training rounds to reach mean score 0.80 on this task. |
| `recommendation` | string | Human-readable action summary including pair count, scores, and divergence step. |
| `n_training_pairs` | int | Number of reinforce/contrast pairs found. 0 means not enough runs yet. |
| `training_pairs` | array | The actual training pairs (see below). |

---

## Training pair

Each entry in `training_pairs` is one DPO-ready pair: the trajectory file
to reinforce and the trajectory file to use as the contrast.

```json
{
  "reinforce_traj_path": "jobs/run_B/.../easy-1__HsggMk6/agent/trajectory.json",
  "contrast_traj_path":  "jobs/run_B/.../easy-1__2bNtUEa/agent/trajectory.json",
  "reinforce_score": 1.0,
  "contrast_score":  0.0,
  "gap": 1.0,
  "structural_divergence_step": 5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `reinforce_traj_path` | string | Path to the trajectory file for the **winning** run. The agent's behavior in this file should be reinforced. |
| `contrast_traj_path` | string | Path to the trajectory file for the **losing** run. The agent's behavior in this file should be contrasted against. |
| `reinforce_score` | float | Score of the winning run. |
| `contrast_score` | float | Score of the losing run. |
| `gap` | float | reinforce_score − contrast_score. |
| `structural_divergence_step` | int | The step index at which the two trajectories first made a different structural choice (write vs. search vs. exec). The DPO loss is most informative around this step. |

---

## How to use this in a DPO trainer

The training pairs point to trajectory JSON files in disteval's standard
format (see [TRAJECTORY_FORMAT.md](TRAJECTORY_FORMAT.md)).

```python
import json

plan = json.load(open("improvement_plan.json"))

dpo_pairs = []
for task in plan["curriculum"]:
    for pair in task["training_pairs"]:
        chosen  = json.load(open(pair["reinforce_traj_path"]))
        rejected = json.load(open(pair["contrast_traj_path"]))
        dpo_pairs.append({
            "task": task["task"],
            "chosen":   chosen["steps"],
            "rejected": rejected["steps"],
            "divergence_step": pair["structural_divergence_step"],
            "score_gap": pair["gap"],
        })

# Feed dpo_pairs into your DPO training loop.
# Priority: process tasks in curriculum order (already sorted by leverage).
```

---

## Re-running after training

After training on the curriculum, run your agent on the same tasks again and
regenerate the plan. The `cycle` counter tracks progress. A well-working loop:

```bash
# Cycle 1
harbor run --agents my-agent --tasks tasks/ --episodes 3 --output jobs/run_1/
disteval engine jobs/run_1/ --agent my-agent --output plan_1.json

# → train your agent on plan_1.json's training_pairs

# Cycle 2
harbor run --agents my-agent-v2 --tasks tasks/ --episodes 3 --output jobs/run_2/
disteval engine jobs/run_2/ --agent my-agent-v2 --cycle 2 --output plan_2.json
```

Watch `consistency_index` increase and `recoverable_score_left` decrease toward
zero as the agent improves. When `n_recoverable` = 0, the agent is consistent
on everything it can solve — further improvement requires new capability
(addressing the STUCK tasks by other means).

---

## Interpreting the numbers

**consistency_index = 0.81** means on average the agent achieves 81% of its
own best score on each task. There is 19% of its demonstrated capability that
it leaves unrealized due to inconsistency — that is the DPO training budget.

**recoverable_score_left = 0.83** means if the agent solved every RECOVERABLE
task at its Q\* rate consistently, total mean score across tasks would increase
by 0.83 points. That is an upper bound on what DPO training alone can achieve
without teaching the agent any new skills.

**n_stuck = 1** means one task was never solved in any run. No training pair
exists. You need to either: (a) run more episodes to give the agent more
chances, (b) provide additional tools or context, or (c) decompose the task
into sub-tasks the agent can already solve.
