"""disteval.agent_harness — lifecycle wrapper for running and evaluating agents.

This module implements the "agent harness" concept described in the recursive
self-improvement research track: the software infrastructure that surrounds an
LLM or agent, handling everything except the model itself. The harness
connects the agent to the outside world, captures its behaviour in the
disteval format, and feeds it into the self-improvement engine.

Design principles:
- disteval stays an evaluation framework; the harness does not train models.
- The harness is a thin, replaceable wrapper around an agent execution loop.
- It produces records and trajectories that are compatible with the existing
  adapters, so the report / engine / compare commands work unchanged.
- Heavy dependencies (e.g. trajectory memory) are optional imports.

A minimal harness episode:

    from disteval.agent_harness import AgentHarness, Agent, TaskSpec

    class MyAgent(Agent):
        def run_step(self, context):
            # ... call LLM, decide next tool, etc.
            return ToolCall(function_name="read_file", arguments={"path": "task.md"})

    harness = AgentHarness(agent=MyAgent(), executor=MyToolExecutor(), verifier=MyVerifier())
    result = harness.run_episode(TaskSpec(id="task-1", instruction="..."))
    result.store.to_jsonl("records.jsonl")

The article that motivated this module:
https://parallel.ai/articles/what-is-an-agent-harness
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .records import EpisodeRecord, RecordStore


try:
    from .trajectory_memory import TrajectoryMemory
except Exception:  # pragma: no cover - guard against import failures
    TrajectoryMemory = None


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """One tool call produced by an agent."""

    function_name: str
    arguments: dict = field(default_factory=dict)
    tool_call_id: str = ""


@dataclass
class Observation:
    """Result of executing a tool call."""

    source_call_id: str
    output: Any
    error: Optional[str] = None


@dataclass
class Step:
    """One turn in an agent trajectory."""

    message: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskSpec:
    """Specification for a task the agent should attempt."""

    id: str
    instruction: str
    initial_state: dict = field(default_factory=dict)
    difficulty: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Outcome of verifying an agent's final state."""

    score: float  # 0.0–1.0
    success: bool
    failure_mode: Optional[str] = None
    criteria: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentContext:
    """Mutable context passed to the agent at each step."""

    task: TaskSpec
    steps: list[Step] = field(default_factory=list)
    memory_prompt: str = ""
    state: dict = field(default_factory=dict)
    max_steps: int = 100
    done: bool = False


@dataclass
class Trajectory:
    """Agent trajectory in the disteval format."""

    task_id: str
    agent_name: str
    run_id: str
    episode: int
    steps: list[Step]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to the JSON format described in TRAJECTORY_FORMAT.md."""
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "run_id": self.run_id,
            "episode": self.episode,
            "steps": [
                {
                    "message": s.message,
                    "tool_calls": [
                        {
                            "function_name": tc.function_name,
                            "arguments": tc.arguments,
                            "tool_call_id": tc.tool_call_id,
                        }
                        for tc in s.tool_calls
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": o.source_call_id,
                                "output": o.output,
                                "error": o.error,
                            }
                            for o in s.observations
                        ]
                    }
                    if s.observations
                    else None,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "metadata": self.metadata,
        }

    def save(self, path: str) -> None:
        """Write the trajectory to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class EpisodeResult:
    """Result of running one episode through the harness."""

    record: EpisodeRecord
    trajectory: Trajectory
    trajectory_path: Optional[str] = None
    store: RecordStore = field(default_factory=RecordStore)

    def __post_init__(self):
        if not self.store._records:
            self.store.add(self.record)


# ── Abstract base classes ─────────────────────────────────────────────────────

class Agent(ABC):
    """Abstract agent: the thing the harness wraps around."""

    @abstractmethod
    def run_step(self, context: AgentContext) -> Step:
        """Return the next step given the current context."""
        ...

    def is_done(self, context: AgentContext) -> bool:
        """Return True when the agent considers the task complete."""
        return context.done


class ToolExecutor(ABC):
    """Abstract tool executor: runs tool calls in the outside world."""

    @abstractmethod
    def execute(self, tool_call: ToolCall, context: AgentContext) -> Observation:
        """Execute one tool call and return the observation."""
        ...


class Verifier(ABC):
    """Abstract verifier: scores the agent's final state against a task."""

    @abstractmethod
    def verify(self, task: TaskSpec, context: AgentContext) -> VerificationResult:
        """Verify the final state and return a score."""
        ...


# ── Convenience no-op implementations for testing and scaffolding ────────────

class NoOpAgent(Agent):
    """Agent that does nothing and immediately reports done."""

    def run_step(self, context: AgentContext) -> Step:
        context.done = True
        return Step(message="No-op agent: done.")


class NoOpToolExecutor(ToolExecutor):
    """Tool executor that returns the arguments as the output."""

    def execute(self, tool_call: ToolCall, context: AgentContext) -> Observation:
        return Observation(
            source_call_id=tool_call.tool_call_id or "noop",
            output=tool_call.arguments,
        )


class BinaryVerifier(Verifier):
    """Verifier that returns a fixed score, useful for tests and baselines."""

    def __init__(self, score: float = 1.0, success: bool = True):
        self.score = float(score)
        self.success = bool(success)

    def verify(self, task: TaskSpec, context: AgentContext) -> VerificationResult:
        return VerificationResult(score=self.score, success=self.success)


# ── Harness orchestrator ─────────────────────────────────────────────────────

class AgentHarness:
    """Run an agent on a task and produce disteval-compatible records.

    The harness is the lifecycle wrapper around the agent. It handles:

    1. Intent capture: read the task spec.
    2. Memory retrieval: optionally retrieve similar past trajectories.
    3. Step loop: repeatedly call the agent, execute tools, and capture steps.
    4. Verification: score the final state.
    5. Persistence: write episode records and trajectory files.
    """

    def __init__(
        self,
        agent: Agent,
        executor: ToolExecutor,
        verifier: Verifier,
        memory: Optional[Any] = None,
        max_steps: int = 100,
        agent_name: str = "agent",
        run_id: str = "run_001",
        success_threshold: float = 0.99,
    ):
        self.agent = agent
        self.executor = executor
        self.verifier = verifier
        self.memory = memory
        self.max_steps = max_steps
        self.agent_name = agent_name
        self.run_id = run_id
        self.success_threshold = success_threshold

    def run_episode(
        self,
        task: TaskSpec,
        episode: int = 0,
        output_dir: Optional[str] = None,
    ) -> EpisodeResult:
        """Run one episode and return a disteval-compatible result."""
        context = AgentContext(
            task=task,
            max_steps=self.max_steps,
            memory_prompt=self._memory_prompt(task),
        )

        for _ in range(self.max_steps):
            step = self.agent.run_step(context)
            context.steps.append(step)

            for tool_call in step.tool_calls:
                observation = self.executor.execute(tool_call, context)
                step.observations.append(observation)

            if self.agent.is_done(context):
                break

        verification = self.verifier.verify(task, context)
        trajectory = Trajectory(
            task_id=task.id,
            agent_name=self.agent_name,
            run_id=self.run_id,
            episode=episode,
            steps=context.steps,
            metadata={
                "n_steps": len(context.steps),
                "memory_prompt": context.memory_prompt,
                **verification.metadata,
            },
        )

        trajectory_path: Optional[str] = None
        if output_dir is not None:
            trajectory_path = os.path.join(output_dir, f"traj_{task.id}_{episode}.json")
            trajectory.save(trajectory_path)

        record = EpisodeRecord(
            run_id=self.run_id,
            model=self.agent_name,
            task=task.id,
            episode=episode,
            score=verification.score,
            success=verification.success,
            strata={"difficulty": task.difficulty} if task.difficulty else {},
            failure_mode=verification.failure_mode,
            length=len(context.steps),
            trajectory_ref=trajectory_path,
            meta={
                "criteria": verification.criteria,
                "memory_prompt": context.memory_prompt,
                **task.metadata,
            },
        )

        return EpisodeResult(
            record=record,
            trajectory=trajectory,
            trajectory_path=trajectory_path,
        )

    def run_batch(
        self,
        tasks: list[TaskSpec],
        episodes_per_task: int = 1,
        output_dir: Optional[str] = None,
    ) -> RecordStore:
        """Run multiple tasks and episodes, returning a single RecordStore."""
        store = RecordStore()
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        for task in tasks:
            for episode in range(episodes_per_task):
                result = self.run_episode(
                    task,
                    episode=episode,
                    output_dir=output_dir,
                )
                store.add(result.record)
        return store

    def _memory_prompt(self, task: TaskSpec) -> str:
        """Optionally retrieve memory for the task and return a prompt string."""
        if self.memory is None:
            return ""
        if TrajectoryMemory is None:
            return ""
        try:
            results = self.memory.retrieve_for_new_task(task.instruction, k=3)
            return self.memory.generate_retrieval_prompt(results, context="before_task")
        except Exception:
            return ""


def run_harness_episode(
    agent: Agent,
    executor: ToolExecutor,
    verifier: Verifier,
    task: TaskSpec,
    **harness_kwargs: Any,
) -> EpisodeResult:
    """Convenience entrypoint: create a harness and run one episode."""
    harness = AgentHarness(agent=agent, executor=executor, verifier=verifier, **harness_kwargs)
    return harness.run_episode(task)


def run_harness_batch(
    agent: Agent,
    executor: ToolExecutor,
    verifier: Verifier,
    tasks: list[TaskSpec],
    output_dir: Optional[str] = None,
    **harness_kwargs: Any,
) -> RecordStore:
    """Convenience entrypoint: run a batch of tasks and save records."""
    harness = AgentHarness(agent=agent, executor=executor, verifier=verifier, **harness_kwargs)
    store = harness.run_batch(tasks, output_dir=output_dir)
    if output_dir is not None:
        store.to_jsonl(os.path.join(output_dir, "records.jsonl"))
    return store
