# Experiment 3: SelfEngine curriculum vs oracle

**Question:** Does the automated `SelfEngine` curriculum rank RECOVERABLE tasks as
well as an oracle that knows the true gap structure?

**Hypothesis:** `SelfEngine.run_cycle()` ranks tasks within Kendall's τ = 0.80
of the oracle and produces κ improvement within 0.05 of the oracle.

**Baselines:** oracle, random, gap-only, human-curated.

**Key metrics:** Kendall τ and Spearman ρ vs. oracle, Δκ, rounds to κ = 0.80,
pair quality, curriculum diversity.

## How to run

```bash
cd /Users/shikharsehgal/rl-dist-eval
python3 research/experiments/experiment_03_self_engine_oracle/run.py
```

## Outputs

- `results/rankings.csv` — per-task oracle and SelfEngine priorities.
- `results/summary.json` — agreement metrics and simulated κ improvement.

## How to swap in real data

Run `SelfEngine` on a real Harbor job and compare its curriculum to a human
expert ranking:

```python
from disteval.adapters.harbor_jobs import load_harbor_job
from disteval.self_engine import SelfEngine

store = load_harbor_job("jobs/run_A/", tasks_dir="tasks/")
engine = SelfEngine(store=store, job_dirs=["jobs/run_A/"], agent_name="my-agent", model_name="my-model")
plan = engine.run_cycle(cycle=1)
print(plan.summary())
```
