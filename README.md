# disteval

**Distribution-first evaluation and self-improvement for long-horizon AI agents.**

disteval does two things that no other eval framework does together:

1. **Measures the full outcome distribution** of agent runs — not just the mean,
   but tail risk, consistency, stochastic dominance, and multi-run confidence
   intervals that expose whether a reported improvement is real or eval noise.
2. **Automatically generates training data** from those runs — no human labels,
   no synthetic data. If an agent sometimes solves a task and sometimes fails,
   those two trajectories are a ready-made DPO training pair.

Three design principles drive every decision:

- **Rigorous multi-run evaluation**: running an agent 8× on each task (standard
  practice) is worthless if you only report the mean. disteval reports CIs,
  per-run repeatability, and whether a two-point gap is within eval noise.
- **Criterion-level failure analysis**: aggregate pass/fail per rubric criterion
  across all episodes to surface *which specific requirement* an agent fails
  most — actionable at the task-design and training level.
- **Data efficiency**: the DPO curriculum disteval generates is proof that
  hundreds (not tens of thousands) of targeted trajectory pairs produce
  measurable capability gain — because they come from the exact tasks where
  the agent's knowledge is incomplete, not from random sampling.

---

## The problem in one picture

```
Harbor leaderboard:          disteval adds:

Claude Code  0.836 ████████  Mean    IQM    CVaR@0.1   pass^3
Gemini CLI   0.754 ███████   0.836   0.970   0.500     0.600   ← reliable
Codex CLI    0.300 ███       0.754   0.955   0.000 ←!  0.400   ← tail collapses
                             0.300   0.067   0.000     0.167   ← flaky
```

Gemini's CVaR@0.1 = 0.000 on easy tasks. Harbor's mean showed nothing about
this. And for Gemini's inconsistent tasks, disteval has found the DPO pairs —
the passing run and the failing run are already in `jobs/`.

---

## Install

```bash
pip install disteval
```

Requirements: Python ≥ 3.10, numpy, pandas, scipy, matplotlib.

Optional: `pip install disteval[inspect]` for Inspect (UK AISI) log support,
`pip install disteval[rliable]` for rliable matrix export.

---

## Quickstart: 5-minute loop from eval to training curriculum

```bash
# 1. Run your agents on tasks with Harbor
harbor run --agents my-agent --tasks tasks/ --episodes 3

# 2. Get the full distribution report
disteval report jobs/run_1/ --agent my-agent --tasks-dir tasks/

# 3. Generate a ranked training curriculum with DPO pairs
disteval engine jobs/run_1/ --agent my-agent --tasks-dir tasks/ --output plan.json

# 4. Train on the pairs disteval found (see CURRICULUM_FORMAT.md)
#    your_dpo_trainer.py --curriculum plan.json

# 5. Re-run and watch consistency_index rise each cycle
harbor run --agents my-agent-v2 --tasks tasks/ --episodes 3
disteval engine jobs/run_2/ --agent my-agent-v2 --cycle 2 --output plan_2.json
```

---

## CLI commands

```bash
disteval report   jobs/<run>/               # single-agent distribution report + charts
disteval compare  jobs/run_A/ jobs/run_B/   # head-to-head leaderboard comparison
disteval engine   jobs/<run>/               # generate training curriculum
disteval sim      jobs/<run>/               # Monte Carlo training simulation
```

Or invoke by module: `python -m disteval <subcommand>`.

---

## How it works

### Step 1 — measure the distribution, not just the mean

Five metrics that harbor reports only one of:

| Metric | What it tells you | Harbor? |
|--------|-------------------|---------|
| **Mean** | Average score | ✓ |
| **IQM** | Mean with top/bottom 25% stripped — outlier-resistant | ✗ |
| **CVaR@0.1** | Expected score in worst 10% of runs — tail risk | ✗ |
| **pass@k** | P(≥1 success in k tries) — peak capability | ✗ |
| **pass^k** | P(all k tries succeed) — deployment consistency | ✗ |

A large gap between `pass@k` and `pass^k` is the signature of inconsistency.

```python
from disteval.adapters.harbor_jobs import load_harbor_job
from disteval import metrics

store = load_harbor_job("jobs/run_1/", tasks_dir="tasks/")
df = store.df()

metrics.iqm(df["score"].values)           # 0.955
metrics.cvar(df["score"].values, 0.1)     # 0.000 — tail collapses
metrics.pass_at_k(df, k=3)               # 0.889
metrics.pass_hat_k(df, k=3)              # 0.400 — only 40% fully consistent
```

### Step 2 — classify every task as SOLID / RECOVERABLE / STUCK

For each task, disteval computes:

- **Q\*(t)** = best score across all runs (demonstrated capability)
- **Q̄(t)** = mean score (what standard RL optimizes)
- **Δ(t)** = Q\* − Q̄ (recoverable gap — score left on the table)
- **κ(t)** = Q̄ / Q\* (consistency index, 0–1)

| Class | Condition | What it means | Action |
|-------|-----------|---------------|--------|
| **SOLID** | Q\* > 0, Δ = 0 | Consistently achieves best | Skip — nothing to recover |
| **RECOVERABLE** | Q\* > 0, Δ > 0 | Can solve it but doesn't always | **Train here — DPO pair exists** |
| **STUCK** | Q\* = 0 | Never solved it | No pair possible — needs new capability |

```python
from disteval.right_tail import right_tail_analysis

report = right_tail_analysis(store, model_name="my-agent")
print(f"κ = {report.consistency_index:.2f}")       # 0.81
print(f"recoverable gap = {report.total_gap:.2f}") # 0.83 — score available to recover

for task in report.priority_tasks:   # sorted by Δ × (1 − κ), highest leverage first
    print(task.task, task.kind)
    print("  reinforce:", task.reinforce_idx)  # indices of passing runs
    print("  contrast:", task.contrast_idx)    # indices of failing runs
```

### Step 3 — generate the training curriculum

`SelfEngine` assembles the full pipeline in one call: reads trajectories, runs
right-tail analysis, finds the divergence step where the passing and failing
runs first diverge, queries trajectory memory for similar past successes, ranks
tasks by **Δ(t) × (1 − κ(t))**, and writes a JSON curriculum with file paths
ready to feed into DPO training.

```python
from disteval.self_engine import SelfEngine

engine = SelfEngine.from_job_dirs(
    ["jobs/run_1/"],
    agent_name="my-agent",
    model_name="my-model",
    tasks_dir="tasks/",
)
plan = engine.run_cycle(cycle=1)

print(plan.summary())
# Cycle 1 | 6 tasks: 2 SOLID · 3 RECOVERABLE · 1 STUCK
# consistency_index κ = 0.81 | recoverable_score_left = 0.83
# predicted_gain = +0.12

plan.save("plan.json")
```

The output `plan.json` contains the ranked curriculum with `reinforce_traj_path`
and `contrast_traj_path` for each RECOVERABLE task.
See [CURRICULUM_FORMAT.md](CURRICULUM_FORMAT.md) for the full spec.

---

## Mathematical foundations

disteval's choices are not ad-hoc heuristics. Each primitive is backed by a
specific statistical or decision-theoretic model.

### Distribution metrics

For a task with scores `x_1, ..., x_n`:

- **IQM** (interquartile mean): mean after removing the lowest and highest 25%.
  Robust to outliers while retaining more data than the median.
- **CVaR@α** (conditional value at risk): average of the worst `α` fraction of
  outcomes. In disteval, `α = 0.1` measures tail risk — how bad the agent can
  get on a bad run.
- **pass^k**: probability that all `k` independent runs succeed. This is the
  deployment-relevant consistency metric.

### Right-tail taxonomy

For each task `t`:

```
Q*(t) = max_i score_i(t)        # demonstrated capability
Q̄(t) = mean_i score_i(t)         # what standard RL optimizes
Δ(t) = Q*(t) - Q̄(t)              # recoverable gap
κ(t) = Q̄(t) / Q*(t)              # consistency index (0-1)
```

A task is **RECOVERABLE** when `Q*(t) > 0` and `Δ(t) > 0`. The training pair
is automatically `(reinforce, contrast) = (argmax_i score_i(t), argmin_i score_i(t))`.

### Curriculum ranking

The default heuristic ranks by leverage:

```
priority(t) = Δ(t) · (1 - κ(t))
```

An information-theoretic alternative is also available:

```
priority_eig(t) = H[score(t)] · (1 - κ(t))
```

where `H[score(t)]` is the empirical Shannon entropy of the per-task score
distribution. It prioritizes tasks whose outcomes are both uncertain and
recoverable.

### Optimal control formulation

Curriculum scheduling can be modeled as a finite-horizon MDP with state
` s = (κ_1, ..., κ_n, t) `, action `a ∈ {1, ..., n, STOP}`, deterministic
transition

```
κ_i' = min(1, κ_i + α · Δ(i) · (1 - κ_i))
```

and reward `R(s, a=i) = α · Δ(i) · (1 - κ_i)`. The Bellman optimality equation
is

```
V*(s) = max_a [ R(s, a) + γ · V*(s') ]
```

`disteval.curriculum_optimizer` provides value iteration and rolling-horizon MPC
solvers for this MDP.

### Bayesian optimization

Training hyperparameters such as the DPO learning rate `α` and the right-tail
bonus `β` are tuned via Gaussian Process Bayesian optimization. The surrogate
models the objective `f(x) = mean_score_after_training(x)` and the acquisition
function balances posterior mean (exploitation) and posterior variance
(exploration). `disteval.bayesian_optimization.optimize_dpo_hyperparameters`
exposes this for `(α, β, k)`.

### Robust distributed aggregation

When multiple agents evaluate the same task with different reliability, the
minimum-variance unbiased aggregate is the inverse-variance weighted mean:

```
μ̂_t = Σ_i w_i · x_i / Σ_i w_i,    w_i = 1 / σ_i²
```

For outlier agents, `aggregate_by_task_robust` uses M-estimation (Huber loss)
via iterative reweighted least squares. Consensus boundaries use confidence-
weighted medians instead of plain medians.

### Thompson Sampling for online task selection

`disteval.bayesian_optimization.ThompsonSamplingScheduler` maintains a Gaussian
posterior over feature weights `θ` and samples `θ̃ ~ N(μ, Σ)` at each cycle to
select the task with highest predicted reward `x_i^T θ̃`. This is a contextual
bandit with linear payoffs (LinTS; Agrawal & Goyal 2013) and provides
principled exploration-exploitation trade-offs.

---

## Bring your own agent — no Harbor required

disteval works with any agent that produces a score per attempt and a trajectory
file. Use the generic adapter:

```python
# your_eval.py
import json
from disteval.adapters.generic import load_records

# Build a JSONL file from your eval results:
results = []
for task in tasks:
    for attempt in range(3):
        score, traj_path = run_my_agent(task, attempt)
        results.append({
            "run_id": "run_001",
            "model": "my-agent",
            "task": task,
            "episode": attempt,
            "score": score,
            "difficulty": task_difficulty[task],   # optional
            "trajectory": traj_path,               # optional but needed for DPO pairs
        })

with open("runs.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# Load into disteval:
store = load_records("runs.jsonl")
```

Then run the same CLI:

```bash
disteval report runs.jsonl --agent my-agent
disteval engine runs.jsonl --agent my-agent --output plan.json
```

See [TRAJECTORY_FORMAT.md](TRAJECTORY_FORMAT.md) for the full record and
trajectory file specifications.

---

## Supported eval frameworks

| Framework | How to load |
|-----------|-------------|
| [Harbor](https://github.com/av/harbor) | `disteval.adapters.harbor_jobs.load_harbor_job` |
| [Inspect](https://inspect.ai) (UK AISI) | `disteval.adapters.inspect_log.load_inspect_json` |
| [rliable](https://github.com/google-research/rliable) | `disteval.adapters.rliable_bridge.to_rliable_dict` |
| Any custom eval | `disteval.adapters.generic.load_records` (JSONL) |

---

## Advanced features

### Real-time trajectory monitoring

The structural signature of an agent's tool-call sequence predicts final outcome
with **89% leave-one-out accuracy** before the run completes.

```python
from disteval.trajectory_monitor import TrajectoryMonitor

monitor = TrajectoryMonitor.from_job_dirs(["jobs/run_1/"])

# Check after each agent step:
match = monitor.check(current_steps, prefix_n=len(current_steps))
print(match.prediction)    # "high" | "low" | "uncertain"
print(match.p_high)        # 0.07 — heading for failure
print(match.warning)       # "Searching extensively without writing code..."
print(match.recommendation)# "Stop searching. Write a minimal implementation now."
```

### Cross-session trajectory memory

Retrieve the trajectories where the agent succeeded on tasks it normally fails
— before starting a new run.

```python
from disteval.trajectory_memory import TrajectoryMemory

mem = TrajectoryMemory()
mem.load_from_job_dirs(["jobs/run_1/", "jobs/run_2/"])

results = mem.retrieve_for_new_task("log file parser python", k=3)
prompt  = mem.generate_retrieval_prompt(results, context="before_task")
# Feed prompt to agent before it starts the task
```

### Distribution comparison between agents

```python
from disteval import compare

a = store_A.df()["score"].values
b = store_B.df()["score"].values

compare.wasserstein(a, b)             # 0.082
compare.prob_improvement(a, b)        # 0.546 — P(A > B)
compare.stochastic_dominance(a, b)    # {"FSD_A_dominates_B": True, ...}
```

### Criterion-level failure analysis (rubric grading)

Real evaluation rubrics score agents against multiple pass/fail criteria. Two
agents with identical aggregate success rates can fail on entirely different
requirements. `criterion_failure_rates` pinpoints which rubric items are broken:

```python
from disteval.failure import criterion_failure_rates, top_failing_criteria

# episodes: list of dicts, each with a "criteria" key mapping criterion → bool
episodes = [
    {"criteria": {"output_format": True, "cost_within_budget": False, "no_data_loss": True}, "difficulty": "hard"},
    {"criteria": {"output_format": False, "cost_within_budget": False, "no_data_loss": True}, "difficulty": "hard"},
    {"criteria": {"output_format": True,  "cost_within_budget": True,  "no_data_loss": True}, "difficulty": "easy"},
]

df = criterion_failure_rates(episodes)
# Returns: criterion | n_episodes | n_failed | failure_rate (sorted by failure_rate desc)
# → cost_within_budget: 2/3 failed (0.667) — most actionable rubric weakness

top3 = top_failing_criteria(episodes, n=3, by=["difficulty"])
# Stratified: which criteria fail most on "hard" vs "easy" tasks?
```

### Multi-run evaluation reliability

A standard single-run bootstrap CI is a *lower bound* on true run-to-run
variance — it can't capture env seed variance or LLM nondeterminism across
runs. `repeat.py` measures the actual meta-distribution:

```python
from disteval.repeat import meta_distribution, bootstrap_vs_repeat, is_gap_real

# Run your eval n times, collect a list of RecordStores
stores = [run_eval(seed=i) for i in range(8)]

meta = meta_distribution(stores, stat_fn=lambda df: df["score"].mean())
print(meta["ci_width"])    # true run-to-run CI width

diag = bootstrap_vs_repeat(stores, stat_fn=lambda df: df["score"].mean())
print(diag["underconfidence_ratio"])
# If >> 1, your single-run bootstrap CI is overconfident — the reported
# error bars are too tight and a 2-point improvement may be noise.

verdict = is_gap_real(stores_A, stores_B, stat_fn=lambda df: df["score"].mean())
print(verdict["P(A>B on a fresh re-run)"])   # decision-relevant probability
```

### Agent harness for running and recording episodes

If you want disteval to capture the agent lifecycle itself instead of only
reading logs from Harbor or Inspect, use the harness:

```python
from disteval.agent_harness import AgentHarness, Agent, TaskSpec

class MyAgent(Agent):
    def run_step(self, context):
        # ... call LLM, return tool calls ...
        return Step(tool_calls=[ToolCall("read_file", {"file_path": "task.md"})])

harness = AgentHarness(
    agent=MyAgent(),
    executor=MyToolExecutor(),
    verifier=MyVerifier(),
    agent_name="my-agent",
)

result = harness.run_episode(TaskSpec(id="task-1", instruction="..."), output_dir="runs/")
result.store.to_jsonl("runs/records.jsonl")
```

The harness manages the agent lifecycle (intent, tool execution, memory,
verification, and persistence) and writes records and trajectories in the
exact format the rest of disteval consumes. See
[`research/agent_harness.md`](research/agent_harness.md) for the design mapping.

---

## File layout

```
disteval/
  __main__.py             — unified CLI dispatcher (disteval <subcommand>)
  records.py              — EpisodeRecord, RecordStore
  metrics.py              — IQM, CVaR, VaR, pass@k, pass^k
  bootstrap.py            — stratified bootstrap CI, performance profile
  compare.py              — Wasserstein, KS, prob_improvement, stochastic dominance
  failure.py              — failure-mode distribution + criterion-level rubric analysis
  repeat.py               — repeated-eval meta-distribution, bootstrap underconfidence check
  right_tail.py           — right-tail gap Δ, consistency κ, RECOVERABLE taxonomy
  self_engine.py          — SelfEngine: full eval → training loop
  trajectory_monitor.py   — real-time outcome prediction from tool-call sequence
  trajectory_memory.py    — outcome-indexed retrieval across sessions
  training_sim.py         — Monte Carlo simulation: disteval vs random vs top-K
  report.py               — CLI: single-agent report
  compare_report.py       — CLI: multi-agent leaderboard comparison
  viz.py                  — matplotlib charts
  adapters/
    harbor_jobs.py        — Harbor jobs/ → RecordStore
    inspect_log.py        — Inspect .eval log → RecordStore
    rliable_bridge.py     — RecordStore → rliable matrix
    generic.py            — any (score, trajectory) source → RecordStore
    swebench_adapter.py   — SWE-bench predictions + SWE-agent trajectories → RecordStore
  logging.py              — CycleLogger: per-cycle κ tracking, plateau detection, JSON/CSV export
  training_harness.py     — DPOTrainerBase, NoOpTrainer, SimulatedTrainer, TRL/Axolotl stubs
  agent_harness.py        — lifecycle wrapper for running agents and capturing trajectories

TRAJECTORY_FORMAT.md      — spec: what disteval reads
CURRICULUM_FORMAT.md      — spec: what disteval engine outputs
THEORY.md                 — mathematical argument for right-tail training
research/agent_harness.md — mapping the "agent harness" concept to disteval
```

---

## Running the tests

```bash
pip install disteval[dev]
pytest tests/ -v
```
