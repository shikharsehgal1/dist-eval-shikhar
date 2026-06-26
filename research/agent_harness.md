# Agent Harness

**Source article:** https://parallel.ai/articles/what-is-an-agent-harness

## What the article says

An **agent harness** is the software infrastructure that wraps around a large
language model (LLM) or AI agent, handling everything *except* the model itself.
It is the complete architectural system surrounding the LLM that manages the
lifecycle of context: from intent capture through specification, compilation,
execution, verification, and persistence.

The article identifies five main responsibilities of a harness:

1. **Intent capture & orchestration** — translate the user's high-level goal
   into a structured plan and sequence of steps.
2. **Tool call execution** — intercept tool calls from the model, run them in
   the outside world, and return observations to the model.
3. **Context management & memory** — curate what the model sees on each turn,
   including persistent logs and cross-session memory.
4. **Result verification & iteration** — check outputs, run tests, and loop back
   to fix failures.
5. **Completion and handoff** — save artifacts so the next session can resume.

## Why this matters for disteval

disteval already measures the full outcome distribution of agent runs and turns
the inconsistency into DPO training data. But until now it has been a
*post-hoc* analysis framework: it reads records produced by Harbor, Inspect, or
custom adapters. The agent harness concept lets disteval become a *runtime*
partner in the agent lifecycle without changing its core design:

- The harness runs the agent and captures the exact trajectory disteval needs.
- It wires cross-session memory (via `TrajectoryMemory`) into the agent's
  context before each task.
- It feeds the final verification result into the same `RecordStore` that the
  rest of disteval already understands.
- The existing `SelfEngine` can then consume the records and produce the next
  training curriculum.

This closes the loop: **harness → records → engine → curriculum → training →
re-eval with harness**.

## Mapping to disteval

| Harness concept | disteval equivalent | Where it lives |
|-----------------|---------------------|----------------|
| Intent capture | `TaskSpec` with `instruction` and `initial_state` | `disteval/agent_harness.py` |
| Tool call execution | `ToolExecutor` + `Observation` | `disteval/agent_harness.py` |
| Context management | `AgentContext` + `TrajectoryMemory` retrieval | `disteval/agent_harness.py`, `disteval/trajectory_memory.py` |
| Result verification | `Verifier` returning `VerificationResult` | `disteval/agent_harness.py` |
| Completion / handoff | `Trajectory.save()` + `RecordStore.to_jsonl()` | `disteval/agent_harness.py`, `disteval/records.py` |
| Records consumed by analysis | `EpisodeRecord` / `RecordStore` | `disteval/records.py` |
| Curriculum generation | `SelfEngine.run_cycle()` | `disteval/self_engine.py` |
| Training execution | `DPOTrainerBase` implementations | `disteval/training_harness.py` |

## What was built

`disteval/agent_harness.py` introduces:

- `Agent` — abstract agent that the harness wraps.
- `ToolExecutor` — executes tool calls and returns observations.
- `Verifier` — scores the final state of a task.
- `AgentHarness` — orchestrates one episode or a batch of episodes.
- `EpisodeResult` — pairs a disteval `EpisodeRecord` with a `Trajectory`.
- `run_harness_episode()` and `run_harness_batch()` — convenience entrypoints.

The harness writes trajectories in the format already documented in
`TRAJECTORY_FORMAT.md` and writes records in the format already consumed by
`disteval report`, `disteval engine`, and `disteval compare`.

## Next steps / open questions

- Add a `RetryHarness` subclass that implements the article's
  "write code → run tests → fix errors" iteration loop.
- Connect the harness to a real LLM client (e.g. OpenAI/Anthropic) behind a
  thin `LLMAgent` implementation.
- Add a `MemoryToolExecutor` that can read/write from the memory store during
  a run, not just before it.
- Define how verification criteria should be exposed to `SelfEngine` so the
  engine can produce sub-curricula per criterion.
