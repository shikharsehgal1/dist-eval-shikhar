# Experiment 9: Training simulation fidelity

**Question:** Does `training_sim.py` predict the rank-order of training gains
correctly and stay calibrated to actual learning curves?

**Hypothesis:** Spearman ρ between predicted and true gain rankings > 0.7, and
normalized MAE < 0.15.

**Baselines:** no-op, mean-only, random, oracle.

**Key metrics:** Spearman ρ, normalized MAE, Δκ correlation, learning curve
monotonicity, prediction variance.

## How to run

```bash
cd /Users/shikharsehgal/rl-dist-eval
python3 research/experiments/experiment_09_training_sim_fidelity/run.py
```

## Outputs

- `results/predictions.csv` — predicted vs. true gain per strategy/round.
- `results/rank_correlation.csv` — Spearman ρ per round.
- `results/summary.json` — mean ρ and normalized MAE.

## How to swap in real data

Run a real DPO training round and compare the measured gain to the prediction:

```python
from disteval.training_sim import apply_training_effect
predicted_scores = apply_training_effect(records, selected, report)
actual_gain = store_after.df()["score"].mean() - store_before.df()["score"].mean()
```
