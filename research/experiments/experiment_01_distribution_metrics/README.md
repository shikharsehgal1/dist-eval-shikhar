# Experiment 1: Distribution-first metrics vs mean-only reporting

**Question:** Do IQM, CVaR, pass^k and the consistency index κ detect
inconsistency that the mean score hides?

**Hypothesis:** Agents with identical mean but different tail shapes will be
ranked identically by mean, but differently by distribution-aware metrics.

**Baselines:** mean-only ranking, median-only ranking, std-only ranking.

**Key metrics:** Spearman ρ between mean and distribution-aware rankings,
Cohen's d between consistent and inconsistent profiles, κ, CVaR@0.1, pass^k,
bootstrap CI width.

## How to run

```bash
cd /Users/shikharsehgal/rl-dist-eval
python3 research/experiments/experiment_01_distribution_metrics/run.py
```

## Outputs

- `results/metrics.csv` — per-agent distribution metrics.
- `results/agreement.csv` — Spearman ranking agreement with mean-based ranking.
- `results/summary.json` — mean equality check and effect size.

## How to swap in real data

Replace the `generate_profile` calls with `disteval.adapters.harbor_jobs.load_harbor_job`:

```python
from disteval.adapters.harbor_jobs import load_harbor_job
store = load_harbor_job("jobs/run_A/", tasks_dir="tasks/")
```

Then run the same metric computation and ranking analysis on the real stores.
