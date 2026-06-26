# Experiment 5: Trajectory memory retrieval

**Question:** Does retrieving structurally similar high-outcome trajectories
boost task success rate compared to no memory or random memory?

**Hypothesis:** Structural memory improves success rate by ≥15%, with the
largest gain on RECOVERABLE tasks.

**Baselines:** no memory, random memory, chronological memory.

**Key metrics:** simulated success rate, boost magnitude, sensitivity to pool
size / retrieval k / similarity threshold.

## How to run

```bash
cd /Users/shikharsehgal/rl-dist-eval
python3 research/experiments/experiment_05_trajectory_memory/run.py
```

## Outputs

- `results/results.csv` — per-task, per-condition simulation results.
- `results/summary.csv` — aggregated success rates by condition.

## How to swap in real data

Load real trajectories into `TrajectoryMemory` and run a real agent with / without
the generated prompt:

```python
from disteval.trajectory_memory import TrajectoryMemory
from disteval.trajectory_loader import load_trajectory_records

mem = TrajectoryMemory()
for rec in load_trajectory_records("jobs/run_A/"):
    mem.add(rec)
prompt = mem.generate_retrieval_prompt(mem.retrieve_for_new_task("task", k=3), context="before_task")
```
