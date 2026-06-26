# Experiment 4: Trajectory monitor predictive accuracy

**Question:** Does the structural tool-call signature predict final outcome better
than naive baselines?

**Hypothesis:** A logistic-regression structural predictor achieves ≥70% LOO
accuracy and AUC-ROC ≥ 0.75.

**Baselines:** majority-class, first-step-only, random, constant heuristic.

**Key metrics:** LOO accuracy, AUC-ROC, precision/recall, early-stop regret,
expected calibration error, feature ablation.

## How to run

```bash
cd /Users/shikharsehgal/rl-dist-eval
python3 research/experiments/experiment_04_trajectory_monitor/run.py
```

## Outputs

- `results/results.csv` — per-method LOO accuracy.
- `results/summary.json` — method accuracies and sample size.

## How to swap in real data

Load real Harbor trajectories and use the same featurization and LOO loop:

```python
from disteval.trajectory_loader import load_trajectory_records
records = load_trajectory_records("jobs/run_A/")
```
