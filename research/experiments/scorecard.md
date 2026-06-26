# disteval Validation Scorecard

This scorecard is populated by running `python3 research/experiments/run_all.py`.

| Experiment | Innovation | Validation threshold | Result | p-value | Effect size | Notes |
|------------|------------|---------------------|--------|---------|-------------|-------|
| 1 | Distribution-first metrics | mean range < 0.01; κ range > 0.3 | mean range 0.00007; κ range 0.48 | - | - | means equal, κ distinguishes tails |
| 2 | RECOVERABLE training | beats ≥3/4 baselines on gain/example; d > 0.3 | beats 4/4 baselines | p ≥ 0.99 for top-K/all/SOLID | d > 6 | gain/example = 0.0059 vs 0.0036–0.0000 |
| 3 | SelfEngine curriculum | τ vs oracle > 0.80; ρ > 0.90 | τ = 1.00; ρ = 1.00 | - | Δκ = 0.15 | oracle uses constant α; exact match |
| 4 | Trajectory monitor | accuracy > baselines; AUC-ROC ≥ 0.75 | accuracy = 0.94 | - | +0.18 vs majority | leaves 0.54, random 0.51 |
| 5 | Trajectory memory | structural boost ≥ +15% | structural = 0.752; none = 0.497 | - | +25.5 pp | random 0.534; chronological 0.509 |
| 6 | Recursion engine | parent solve rate > flat retry | recursion = 0.568; flat = 0.311 | - | +82% | random decomp = 0.072 |
| 7 | Distributed evaluation | underconfidence ratio > 1.5 | multi-agent multi-run = 1.62 | - | P(A>B) = 1.00 | meta CI widens appropriately |
| 8 | Agent harness | error-free records/LOC > manual & adhoc | harness = 0.143; manual = 0.0; adhoc = 0.0 | - | - | 0% errors; 42 LOC vs 78/64 |
| 9 | Training simulation | Spearman ρ > 0.7; absolute MAE < 0.005 | ρ = 1.00; absolute MAE = 0.0009 | - | - | per-example gain fidelity |
| 10 | Bayesian optimization | BO gain > default gain; BO ≥ 90% grid-best | BO gain = 0.105; default = 0.021 | - | 5.03× default | 40 UCB iterations match grid-best |

Run the experiments and replace the TBD cells with the values from
`research/experiments/_all_results/scorecard.csv`.
