# Bayesian Optimization Opportunity for disteval

## Summary

Bayesian optimization (BO) is a promising addition to disteval, especially in places where the framework currently relies on fixed heuristics or scalar hand-tuned parameters. Literature from 2024–2025 shows BO consistently outperforms heuristic data selection and curriculum methods for LLM training, often with 10–25% convergence gains.

## Three high-value use cases

| Use case | Current disteval approach | What BO could do | Risk | Priority |
|----------|--------------------------|------------------|------|----------|
| **DPO hyperparameter tuning** | Fixed `ALPHA=0.4`, `DPO_BONUS=1.5`, selection size `k` | Optimize `(α, β, k)` per agent/task distribution using `training_sim.py` as the cheap objective | Low | **1** |
| **Curriculum scheduling** | Ranks RECOVERABLE tasks by `gap × (1 − κ)` | Learn an acquisition function over task profiles and historical κ improvement to choose the next task to train on | Medium | **2** |
| **Recursion boundary inference** | Uses structural divergence + test checkpoints | Search over candidate entry/exit boundaries to maximize predicted sub-task solvability | Medium–High | **3** |

## Key papers

- **DUET** (2025): BO for training data mixtures with noisy feedback — https://arxiv.org/html/2502.00270
- **ADMIRE-BayesOpt** (2024): Multi-fidelity BO for LLM data mixtures — https://openreview.net/pdf?id=0Euvm9zDpu
- **Bayesian Manifold Curriculum** (2024): Manifold-structured bandit for curriculum learning — https://arxiv.org/html/2606.19750v1
- **JoBS** (2025): Joint BO with scaling-law predictor for LLM training — https://arxiv.org/pdf/2602.08351
- **Long-Horizon Data Selection** (2025): Avoid rank reversal from short-term selection — https://arxiv.org/html/2605.30537

## Implementation notes

- **Dependencies:** scikit-learn (`GaussianProcessRegressor`) and scipy are already available; no new heavy dependencies needed.
- **Sample efficiency:** The DPO tuning case is the safest starting point because the objective (`training_sim.apply_training_effect`) is fast and deterministic, so 10–20 BO iterations cost seconds.
- **Cold-start risk:** Curriculum scheduling needs historical κ curves across 2–3 cycles before the GP surrogate is reliable; first cycle should fall back to the existing heuristic.
- **Surrogate choice:** Start with a GP on 3–5 normalized features. Add random-forest or tree-structured surrogates later if the task manifold becomes high-dimensional.

## Recommended next step

Prototype **Use case 2 (DPO hyperparameter tuning)** in `disteval/training_sim.py` or a new `disteval/bayesian_optimization.py` module. It is the lowest-risk, fastest-payoff entry point and can be validated immediately with `experiment_09_training_sim_fidelity`.
