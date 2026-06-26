# Experiment 2: RECOVERABLE training vs random/top-K/all-task/SOLID-only

**Question:** Does training on RECOVERABLE tasks yield higher score gain per
example than other curriculum strategies?

**Hypothesis:** RECOVERABLE tasks (where the agent already succeeds sometimes)
produce the highest leverage because they have both a demonstrated upper bound
and a contrast trajectory.

**Baselines:** random sampling, top-K by score, all-task training, SOLID-only.

**Key metrics:** score gain per example, rounds to reach 0.80 mean score,
Δκ, RECOVERABLE → SOLID graduation rate.

## How to run

```bash
cd /Users/shikharsehgal/rl-dist-eval
python3 research/experiments/experiment_02_recoverable_training/run.py
```

## Outputs

- `results/results.csv` — per-strategy gains and κ changes.
- `results/comparisons.csv` — pairwise tests vs. disteval right-tail.
- `results/summary.json` — configuration and comparisons.

## How to swap in real data

Load real Harbor jobs and pass the records to the same strategy loop:

```python
from disteval.adapters.harbor_jobs import load_harbor_job
store = load_harbor_job("jobs/run_A/", tasks_dir="tasks/")
records = store._records
report = right_tail_analysis(store, model_name="my-agent")
```

For real training, replace `apply_training_effect` with a DPO trainer from
`disteval.training_harness`.
