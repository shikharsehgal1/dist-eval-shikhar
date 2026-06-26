# Experiment 6: Recursion engine

**Question:** Does decomposing STUCK/RECOVERABLE tasks into sub-tasks improve
solve rate compared to flat retry or random decomposition?

**Hypothesis:** Recursive decomposition achieves higher sub-task solve rate and
parent-task graduation than flat retry.

**Baselines:** flat retry, random decomposition, no decomposition,
checkpoint-aligned decomposition.

**Key metrics:** sub-task solve rate, parent graduation rate, gap reduction per
example, solve latency, checkpoint alignment score.

## How to run

```bash
cd /Users/shikharsehgal/rl-dist-eval
python3 research/experiments/experiment_06_recursion_engine/run.py
```

## Outputs

- `results/results.csv` — raw simulation results per task/run/strategy.
- `results/summary.csv` — mean parent and sub-task solve rates by strategy.

## How to swap in real data

Use the real `RecursionEngine` once tasks have checkpoint annotations:

```python
from disteval.recursion_engine import RecursionEngine, RecursionEngineConfig
from disteval.trajectory_monitor import TrajectoryMonitor
from disteval.right_tail import right_tail_analysis

report = right_tail_analysis(store, model_name="my-agent")
monitor = TrajectoryMonitor.from_store(store)
engine = RecursionEngine(monitor=monitor, config=RecursionEngineConfig(), agent_name="my-agent", model_name="my-model")
graph = engine.decompose(report)
```

This simulation stub can be replaced with that real pipeline when the
appropriate task fixtures are in place.
