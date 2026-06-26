# Experiment 7: Distributed evaluation

**Question:** Does aggregating multiple agents and runs reduce the meta-
distribution CI and false positives compared to single-run evaluation?

**Hypothesis:** The multi-agent multi-run meta CI is narrower than the single-run
bootstrap CI suggests (underconfidence ratio > 1.5), and P(A > B) is more accurate.

**Baselines:** single-agent single-run, single-agent multi-run, multi-agent
single-run, multi-agent multi-run.

**Key metrics:** meta CI width, underconfidence ratio, P(A > B on fresh re-run),
false positive rate, false negative rate.

## How to run

```bash
cd /Users/shikharsehgal/rl-dist-eval
python3 research/experiments/experiment_07_distributed_eval/run.py
```

## Outputs

- `results/results.csv` — CI widths and underconfidence ratios by baseline.
- `results/summary.json` — P(A > B) single-run vs multi-run.

## How to swap in real data

Load multiple real Harbor runs and use `repeat.meta_distribution` / `is_gap_real`:

```python
from disteval.adapters.harbor_jobs import load_harbor_job
from disteval.repeat import meta_distribution, is_gap_real

stores_A = [load_harbor_job(f"jobs/run_A_{i}/", tasks_dir="tasks/") for i in range(3)]
stores_B = [load_harbor_job(f"jobs/run_B_{i}/", tasks_dir="tasks/") for i in range(3)]
result = is_gap_real(stores_A, stores_B, stat_fn=lambda df: df["score"].mean())
```
