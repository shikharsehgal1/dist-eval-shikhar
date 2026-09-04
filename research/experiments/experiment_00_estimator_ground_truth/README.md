# Experiment 0: Estimators vs closed-form ground truth

**Question:** Do disteval's estimators recover values that are known
analytically — or do they only satisfy the expectations baked into their own
unit tests?

**Hypothesis:** Each primitive matches its closed form to within Monte Carlo
error, and the 95% confidence intervals attain nominal coverage.

**Why this experiment exists:** every other experiment in this programme
assumes the estimators underneath are correct. A unit test proves the code does
what its author expected; it cannot catch an author who expected the wrong
thing. This experiment compares against mathematics instead.

## What is checked

| Primitive | Closed form it is checked against |
|---|---|
| `pass_at_k` | E[pass@k] = 1 − (1 − p)^k over 20,000 simulated 8-trial tasks |
| `pass_hat_k` | E[pass^k] = p^k, same design |
| `var_at` | VaR@a of Uniform[0,1] = a |
| `cvar` | CVaR@a of Uniform[0,1] = a/2 |
| `iqm` | interquartile mean, clean (0.5) and under 10% wild contamination (0.5556) |
| `stratified_bootstrap_ci` | empirical coverage over 1,000 independent experiments ≈ 0.95 |
| `binomial_ci` | Clopper-Pearson coverage ≥ nominal (conservative by construction) |
| `grpo_advantages` | per-group mean 0, std 1 |
| `divergences` | KL(N(0,1) ‖ N(1,1)) = 0.5 |

The unbiasedness checks use a 4-standard-error tolerance on the Monte Carlo
mean. This is the discriminating part of the experiment: the common shortcut
estimator (`success if any trial passed`) is biased upward and fails it, while
the Chen et al. combinatorial estimator disteval uses passes.

The IQM contamination case is worth reading closely. Replacing 10% of the
sample with a wild outlier does not leave the truth at 0.5 — IQM trims to the
middle 50% of the *sorted* sample, which now maps to uniform quantiles
[0.25/0.9, 0.75/0.9], so the analytic answer moves to 0.5556. The check asserts
that shifted value, and the mean is reported alongside as a contrast: it blows
up to 100,000.

## How to run

```bash
cd /Users/shikharsehgal/rl-dist-eval
python3 research/experiments/experiment_00_estimator_ground_truth/run.py
```

Runs in about a minute, no API keys, seeded and deterministic.

## Result

20/20 checks pass. Bootstrap coverage 0.955, Clopper-Pearson 0.958.

## Outputs

- `results/checks.csv` — every check with its truth, estimate, and tolerance.
- `results/summary.json` — pass count and headline finding.
