# disteval

**Distribution-first evaluation and self-improvement for long-horizon AI agents.**

disteval does two things that no other eval framework does together:

1. **Measures the full outcome distribution** of agent runs — not just the mean,
   but tail risk, consistency, and stochastic dominance.
2. **Automatically generates training data** from those runs — no human labels,
   no synthetic data. If an agent sometimes solves a task and sometimes fails,
   those two trajectories are a ready-made DPO training pair.

---

## The problem in one picture

```
Harbor leaderboard:          disteval adds:

Claude Code  0.836 ████████  Mean    IQM    CVaR@0.1   pass^3
Gemini CLI   0.754 ███████   0.836   0.970   0.500     0.600   ← reliable
Codex CLI    0.300 ███       0.754   0.955   0.000 ←!  0.400   ← tail collapses
                             0.300   0.067   0.000     0.167   ← flaky
```

Gemini's CVaR@0.1 = 0.000 on easy tasks. Harbor's mean showed nothing about
this. And for Gemini's inconsistent tasks, disteval has found the DPO pairs —
the passing run and the failing run are already in `jobs/`.

---

## Install

```bash
pip install disteval
```

Requirements: Python ≥ 3.10, numpy, pandas, scipy, matplotlib.

Optional: `pip install disteval[inspect]` for Inspect (UK AISI) log support,
`pip install disteval[rliable]` for rliable matrix export.

---

## Quickstart: 5-minute loop from eval to training curriculum

```bash
# 1. Run your agents on tasks with Harbor
harbor run --agents my-agent --tasks tasks/ --episodes 3

# 2. Get the full distribution report
disteval report jobs/run_1/ --agent my-agent --tasks-dir tasks/

# 3. Generate a ranked training curriculum with DPO pairs
disteval engine jobs/run_1/ --agent my-agent --tasks-dir tasks/ --output plan.json

# 4. Train on the pairs disteval found (see CURRICULUM_FORMAT.md)
#    your_dpo_trainer.py --curriculum plan.json

# 5. Re-run and watch consistency_index rise each cycle
harbor run --agents my-agent-v2 --tasks tasks/ --episodes 3
disteval engine jobs/run_2/ --agent my-agent-v2 --cycle 2 --output plan_2.json
```

---

## CLI commands

```bash
disteval report   jobs/<run>/               # single-agent distribution report + charts
disteval compare  jobs/run_A/ jobs/run_B/   # head-to-head leaderboard comparison
disteval engine   jobs/<run>/               # generate training curriculum
disteval sim      jobs/<run>/               # Monte Carlo training simulation
```

Or invoke by module: `python -m disteval <subcommand>`.

---

## How it works

### Step 1 — measure the distribution, not just the mean

Five metrics that harbor reports only one of:

| Metric | What it tells you | Harbor? |
|--------|-------------------|---------|
| **Mean** | Average score | ✓ |
| **IQM** | Mean with top/bottom 25% stripped — outlier-resistant | ✗ |
| **CVaR@0.1** | Expected score in worst 10% of runs — tail risk | ✗ |
| **pass@k** | P(≥1 success in k tries) — peak capability | ✗ |
| **pass^k** | P(all k tries succeed) — deployment consistency | ✗ |

A large gap between `pass@k` and `pass^k` is the signature of inconsistency.

```python
from disteval.adapters.harbor_jobs import load_harbor_job
from disteval import metrics

store = load_harbor_job("jobs/run_1/", tasks_dir="tasks/")
df = store.df()

metrics.iqm(df["score"].values)           # 0.955
metrics.cvar(df["score"].values, 0.1)     # 0.000 — tail collapses
metrics.pass_at_k(df, k=3)               # 0.889
metrics.pass_hat_k(df, k=3)              # 0.400 — only 40% fully consistent
```

### Step 2 — classify every task as SOLID / RECOVERABLE / STUCK

For each task, disteval computes:

- **Q\*(t)** = best score across all runs (demonstrated capability)
- **Q̄(t)** = mean score (what standard RL optimizes)
- **Δ(t)** = Q\* − Q̄ (recoverable gap — score left on the table)
- **κ(t)** = Q̄ / Q\* (consistency index, 0–1)

| Class | Condition | What it means | Action |
|-------|-----------|---------------|--------|
| **SOLID** | Q\* > 0, Δ = 0 | Consistently achieves best | Skip — nothing to recover |
| **RECOVERABLE** | Q\* > 0, Δ > 0 | Can solve it but doesn't always | **Train here — DPO pair exists** |
| **STUCK** | Q\* = 0 | Never solved it | No pair possible — needs new capability |

```python
from disteval.right_tail import right_tail_analysis

report = right_tail_analysis(store, model_name="my-agent")
print(f"κ = {report.consistency_index:.2f}")       # 0.81
print(f"recoverable gap = {report.total_gap:.2f}") # 0.83 — score available to recover

for task in report.priority_tasks:   # sorted by Δ × (1 − κ), highest leverage first
    print(task.task, task.kind)
    print("  reinforce:", task.reinforce_idx)  # indices of passing runs
    print("  contrast:", task.contrast_idx)    # indices of failing runs
```

### Step 3 — generate the training curriculum

`SelfEngine` assembles the full pipeline in one call: reads trajectories, runs
right-tail analysis, finds the divergence step where the passing and failing
runs first diverge, queries trajectory memory for similar past successes, ranks
tasks by **Δ(t) × (1 − κ(t))**, and writes a JSON curriculum with file paths
ready to feed into DPO training.

```python
from disteval.self_engine import SelfEngine

engine = SelfEngine.from_job_dirs(
    ["jobs/run_1/"],
    agent_name="my-agent",
    model_name="my-model",
    tasks_dir="tasks/",
)
plan = engine.run_cycle(cycle=1)

print(plan.summary())
# Cycle 1 | 6 tasks: 2 SOLID · 3 RECOVERABLE · 1 STUCK
# consistency_index κ = 0.81 | recoverable_score_left = 0.83
# predicted_gain = +0.12

plan.save("plan.json")
```

The output `plan.json` contains the ranked curriculum with `reinforce_traj_path`
and `contrast_traj_path` for each RECOVERABLE task.
See [CURRICULUM_FORMAT.md](CURRICULUM_FORMAT.md) for the full spec.

---

## Bring your own agent — no Harbor required

disteval works with any agent that produces a score per attempt and a trajectory
file. Use the generic adapter:

```python
# your_eval.py
import json
from disteval.adapters.generic import load_records

# Build a JSONL file from your eval results:
results = []
for task in tasks:
    for attempt in range(3):
        score, traj_path = run_my_agent(task, attempt)
        results.append({
            "run_id": "run_001",
            "model": "my-agent",
            "task": task,
            "episode": attempt,
            "score": score,
            "difficulty": task_difficulty[task],   # optional
            "trajectory": traj_path,               # optional but needed for DPO pairs
        })

with open("runs.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# Load into disteval:
store = load_records("runs.jsonl")
```

Then run the same CLI:

```bash
disteval report runs.jsonl --agent my-agent
disteval engine runs.jsonl --agent my-agent --output plan.json
```

See [TRAJECTORY_FORMAT.md](TRAJECTORY_FORMAT.md) for the full record and
trajectory file specifications.

---

## Supported eval frameworks

| Framework | How to load |
|-----------|-------------|
| [Harbor](https://github.com/av/harbor) | `disteval.adapters.harbor_jobs.load_harbor_job` |
| [Inspect](https://inspect.ai) (UK AISI) | `disteval.adapters.inspect_log.load_inspect_json` |
| [rliable](https://github.com/google-research/rliable) | `disteval.adapters.rliable_bridge.to_rliable_dict` |
| Any custom eval | `disteval.adapters.generic.load_records` (JSONL) |

---

## Advanced features

### Real-time trajectory monitoring

The structural signature of an agent's tool-call sequence predicts final outcome
with **89% leave-one-out accuracy** before the run completes.

```python
from disteval.trajectory_monitor import TrajectoryMonitor

monitor = TrajectoryMonitor.from_job_dirs(["jobs/run_1/"])

# Check after each agent step:
match = monitor.check(current_steps, prefix_n=len(current_steps))
print(match.prediction)    # "high" | "low" | "uncertain"
print(match.p_high)        # 0.07 — heading for failure
print(match.warning)       # "Searching extensively without writing code..."
print(match.recommendation)# "Stop searching. Write a minimal implementation now."
```

### Cross-session trajectory memory

Retrieve the trajectories where the agent succeeded on tasks it normally fails
— before starting a new run.

```python
from disteval.trajectory_memory import TrajectoryMemory

mem = TrajectoryMemory()
mem.load_from_job_dirs(["jobs/run_1/", "jobs/run_2/"])

results = mem.retrieve_for_new_task("log file parser python", k=3)
prompt  = mem.generate_retrieval_prompt(results, context="before_task")
# Feed prompt to agent before it starts the task
```

### Distribution comparison between agents

```python
from disteval import compare

a = store_A.df()["score"].values
b = store_B.df()["score"].values

compare.wasserstein(a, b)             # 0.082
compare.prob_improvement(a, b)        # 0.546 — P(A > B)
compare.stochastic_dominance(a, b)    # {"FSD_A_dominates_B": True, ...}
```

---

## File layout

```
disteval/
  __main__.py             — unified CLI dispatcher (disteval <subcommand>)
  records.py              — EpisodeRecord, RecordStore
  metrics.py              — IQM, CVaR, VaR, pass@k, pass^k
  bootstrap.py            — stratified bootstrap CI, performance profile
  compare.py              — Wasserstein, KS, prob_improvement, stochastic dominance
  failure.py              — failure-mode distribution per stratum
  repeat.py               — repeated-eval meta-distribution
  right_tail.py           — right-tail gap Δ, consistency κ, RECOVERABLE taxonomy
  self_engine.py          — SelfEngine: full eval → training loop
  trajectory_monitor.py   — real-time outcome prediction from tool-call sequence
  trajectory_memory.py    — outcome-indexed retrieval across sessions
  training_sim.py         — Monte Carlo simulation: disteval vs random vs top-K
  report.py               — CLI: single-agent report
  compare_report.py       — CLI: multi-agent leaderboard comparison
  viz.py                  — matplotlib charts
  adapters/
    harbor_jobs.py        — Harbor jobs/ → RecordStore
    inspect_log.py        — Inspect .eval log → RecordStore
    rliable_bridge.py     — RecordStore → rliable matrix
    generic.py            — any (score, trajectory) source → RecordStore

TRAJECTORY_FORMAT.md      — spec: what disteval reads
CURRICULUM_FORMAT.md      — spec: what disteval engine outputs
THEORY.md                 — mathematical argument for right-tail training
```

---

## Running the tests

```bash
pip install disteval[dev]
pytest tests/ -v
```

---

## Interactive demo

```bash
python disteval_gui.py     # web UI at http://localhost:9173
```

Seven-step guided story: from mean-only leaderboard → full distribution →
inconsistency taxonomy → DPO pair extraction → training curriculum → live
self-engine run → Monte Carlo proof (+249% vs random selection).
Real data, three real agents, six real tasks.
