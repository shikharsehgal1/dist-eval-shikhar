"""Tests for disteval.agent_harness."""
import json
import os
import tempfile
from typing import Optional

from disteval.agent_harness import (
    Agent,
    AgentContext,
    AgentHarness,
    BinaryVerifier,
    NoOpAgent,
    NoOpToolExecutor,
    Observation,
    Step,
    TaskSpec,
    ToolCall,
    ToolExecutor,
    Verifier,
    run_harness_batch,
    run_harness_episode,
)


class EchoExecutor(ToolExecutor):
    """Executor that echoes arguments back."""

    def execute(self, tool_call: ToolCall, context: AgentContext) -> Observation:
        return Observation(
            source_call_id=tool_call.tool_call_id or "echo",
            output=tool_call.arguments,
        )


class FixedVerifier(Verifier):
    """Verifier that returns a configured score."""

    def __init__(self, score: float = 0.75, success: bool = True, failure_mode: Optional[str] = None):
        self.score = score
        self.success = success
        self.failure_mode = failure_mode

    def verify(self, task: TaskSpec, context: AgentContext):
        from disteval.agent_harness import VerificationResult

        return VerificationResult(
            score=self.score,
            success=self.success,
            failure_mode=self.failure_mode,
        )


class TwoStepAgent(Agent):
    """Agent that takes two steps then stops."""

    def __init__(self, n_steps: int = 2):
        self.n_steps = n_steps
        self.count = 0

    def run_step(self, context: AgentContext) -> Step:
        self.count += 1
        step = Step(
            message=f"step {self.count}",
            tool_calls=[
                ToolCall(
                    function_name="read_file",
                    arguments={"file_path": f"/app/file_{self.count}.txt"},
                    tool_call_id=f"tc_{self.count}",
                )
            ],
        )
        if self.count >= self.n_steps:
            context.done = True
        return step

    def is_done(self, context: AgentContext) -> bool:
        return self.count >= self.n_steps


class TestAgentHarness:
    def test_run_episode_produces_record(self):
        harness = AgentHarness(
            agent=NoOpAgent(),
            executor=NoOpToolExecutor(),
            verifier=BinaryVerifier(score=1.0, success=True),
            agent_name="test-agent",
            run_id="run_001",
        )
        task = TaskSpec(id="task-1", instruction="do something", difficulty="easy")
        result = harness.run_episode(task, episode=0)

        assert result.record.task == "task-1"
        assert result.record.model == "test-agent"
        assert result.record.run_id == "run_001"
        assert result.record.score == 1.0
        assert result.record.success is True
        assert result.record.strata == {"difficulty": "easy"}
        assert result.record.length == 1
        assert result.record.trajectory_ref is None

    def test_run_episode_writes_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            harness = AgentHarness(
                agent=TwoStepAgent(n_steps=2),
                executor=EchoExecutor(),
                verifier=FixedVerifier(score=0.8, success=True),
                agent_name="agent",
                run_id="run_002",
            )
            task = TaskSpec(id="task-2", instruction="read files")
            result = harness.run_episode(task, episode=1, output_dir=d)

            assert result.trajectory_path == os.path.join(d, "traj_task-2_1.json")
            assert os.path.exists(result.trajectory_path)
            with open(result.trajectory_path) as f:
                data = json.load(f)
            assert data["task_id"] == "task-2"
            assert data["agent_name"] == "agent"
            assert data["run_id"] == "run_002"
            assert data["episode"] == 1
            assert len(data["steps"]) == 2
            assert data["steps"][0]["tool_calls"][0]["function_name"] == "read_file"
            assert data["steps"][0]["observation"]["results"][0]["source_call_id"] == "tc_1"
            assert data["metadata"]["n_steps"] == 2

    def test_run_batch_writes_records(self):
        with tempfile.TemporaryDirectory() as d:
            harness = AgentHarness(
                agent=TwoStepAgent(n_steps=1),
                executor=EchoExecutor(),
                verifier=FixedVerifier(score=0.6, success=False, failure_mode="partial"),
                agent_name="batch-agent",
                run_id="run_003",
            )
            tasks = [
                TaskSpec(id="a", instruction="task a"),
                TaskSpec(id="b", instruction="task b", difficulty="hard"),
            ]
            store = harness.run_batch(tasks, episodes_per_task=2, output_dir=d)

            assert len(store) == 4
            df = store.df()
            assert sorted(df["task"].tolist()) == ["a", "a", "b", "b"]
            assert df["score"].tolist() == [0.6, 0.6, 0.6, 0.6]
            assert df["success"].tolist() == [False, False, False, False]
            assert df["failure_mode"].tolist() == ["partial"] * 4
            assert set(df["s_difficulty"].dropna()) == {"hard"}

    def test_max_steps_caps_loop(self):
        class NeverDoneAgent(Agent):
            def run_step(self, context: AgentContext) -> Step:
                return Step(tool_calls=[ToolCall(function_name="noop")])

        harness = AgentHarness(
            agent=NeverDoneAgent(),
            executor=EchoExecutor(),
            verifier=FixedVerifier(),
            max_steps=3,
        )
        result = harness.run_episode(TaskSpec(id="task", instruction="loop"))
        assert len(result.trajectory.steps) == 3

    def test_memory_prompt_optional(self):
        harness = AgentHarness(
            agent=NoOpAgent(),
            executor=NoOpToolExecutor(),
            verifier=BinaryVerifier(),
            memory=None,
        )
        result = harness.run_episode(TaskSpec(id="task", instruction="test"))
        assert result.record.meta["memory_prompt"] == ""


class TestConvenienceEntrypoints:
    def test_run_harness_episode(self):
        result = run_harness_episode(
            agent=NoOpAgent(),
            executor=NoOpToolExecutor(),
            verifier=BinaryVerifier(score=0.9, success=True),
            task=TaskSpec(id="task", instruction="test"),
            agent_name="convenience-agent",
        )
        assert result.record.score == 0.9
        assert result.record.model == "convenience-agent"

    def test_run_harness_batch(self):
        with tempfile.TemporaryDirectory() as d:
            store = run_harness_batch(
                agent=TwoStepAgent(n_steps=1),
                executor=EchoExecutor(),
                verifier=FixedVerifier(),
                tasks=[TaskSpec(id="x", instruction="x")],
                output_dir=d,
                agent_name="batch-agent",
            )
            assert len(store) == 1
            assert os.path.exists(os.path.join(d, "records.jsonl"))
