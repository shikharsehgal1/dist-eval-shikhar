# Experiment 10: Bayesian Optimization for DPO Hyperparameters

## Purpose

Validate that `disteval.bayesian_optimization.optimize_dpo_hyperparameters` can
find `(alpha, dpo_bonus, k)` settings that outperform the default hyperparameters
on the training simulation objective.

## Method

1. Generate a synthetic agent with 5 RECOVERABLE, 3 STUCK, and 1 SOLID task.
2. Compute the default gain with `(alpha=0.4, dpo_bonus=1.5, k=5)`.
3. Run GP-based Bayesian optimization for 15 iterations over the same space.
4. Compare the BO result to a dense grid search.

## Validation threshold

- BO gain > default gain (strictly better than hand-tuned defaults)
- BO gain ≥ 90% of grid-best gain (well-calibrated surrogate)

## Run

```bash
python3 research/experiments/experiment_10_bayesian_optimization/run.py
```

## Interpretation

If BO finds a higher gain than the default with only 15 objective evaluations,
the GP surrogate is effectively capturing the hyperparameter landscape and can
replace manual tuning for DPO curriculum generation.
