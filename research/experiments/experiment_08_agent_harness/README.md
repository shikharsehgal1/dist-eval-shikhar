# Experiment 8: Agent harness integration cost

**Question:** Does the `AgentHarness` reduce integration cost and produce more
correct records than manual JSONL or ad-hoc adapters?

**Hypothesis:** Harness requires ≤50% LOC vs. manual JSONL and ≤70% vs. ad-hoc
adapter, with 0% first-run validation errors.

**Baselines:** manual JSONL construction, ad-hoc adapter script.

**Key metrics:** lines of code, record completeness, trajectory completeness,
first-run error rate, iteration count to correctness.

## How to run

```bash
cd /Users/shikharsehgal/rl-dist-eval
python3 research/experiments/experiment_08_agent_harness/run.py
```

## Outputs

- `results/results.csv` — LOC, error count, and error rate per method.
- `results/summary.json` — comparison summary.

## How to swap in real data

Replace `SimpleAgent` with a real LLM client and `SimpleExecutor` with filesystem
/ API tool execution. The harness remains the same.
