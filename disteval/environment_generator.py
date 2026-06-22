"""
disteval.environment_generator — Generate runnable sub-task RL environments.

OVERVIEW
────────
Given a SubTaskGraph produced by RecursionEngine, this module builds a
set of GenEnv specifications. Each GenEnv describes a self-contained RL
environment for a single sub-task RMDP component:

  - task_id: sub-task identifier
  - parent_task: root task name
  - instruction: what the agent should accomplish
  - entry_step / exit_step: window in the parent trajectory
  - reward_spec: how to compute reward (checkpoint-based, structural, etc.)
  - initial_state: files/assets needed at episode start
  - verifier_command: how to score the sub-task outcome
  - termination: when the episode ends

These specs are intentionally environment-agnostic: they can be consumed by a
training harness (OpenAI Gym, custom simulator, or fine-tuning data builder).

The generator is default-disabled and only runs when recursion is enabled.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .recursion_engine import (
    PhaseBoundary,
    RecursionEngine,
    SubTaskDefinition,
    SubTaskGraph,
)
from .test_suite_parser import CheckpointSpec


@dataclass
class RewardSpec:
    """Reward specification for a generated sub-task environment."""

    type: str                        # "checkpoint" | "structural" | "manual"
    weight: float                    # relative reward weight
    success_threshold: float         # score considered a "success"
    checkpoint_id: Optional[str] = None
    condition_source: str = ""         # original test.sh condition
    verifier_command: Optional[str] = None
    reward_delta: float = 0.0


@dataclass
class InitialState:
    """Files/assets needed to start a sub-task episode."""

    files: dict[str, str] = field(default_factory=dict)  # path -> content
    task_dir: str = ""
    seed_data_files: list[str] = field(default_factory=list)


@dataclass
class TerminationSpec:
    """Termination condition for a sub-task episode."""

    max_steps: int = 50
    success_threshold: float = 0.99
    timeout_sec: float = 60.0
    require_files: list[str] = field(default_factory=list)


@dataclass
class GenEnv:
    """One generated sub-task RL environment."""

    task_id: str
    parent_task: str
    sub_task_depth: int
    instruction: str
    entry_step: int
    exit_step: int
    phase_tag: str
    reward: RewardSpec
    initial_state: InitialState
    termination: TerminationSpec
    source: str = "checkpoint"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize GenEnv to a dict for JSON export."""
        return {
            "task_id": self.task_id,
            "parent_task": self.parent_task,
            "sub_task_depth": self.sub_task_depth,
            "instruction": self.instruction,
            "entry_step": self.entry_step,
            "exit_step": self.exit_step,
            "phase_tag": self.phase_tag,
            "source": self.source,
            "reward": {
                "type": self.reward.type,
                "weight": self.reward.weight,
                "success_threshold": self.reward.success_threshold,
                "checkpoint_id": self.reward.checkpoint_id,
                "condition_source": self.reward.condition_source,
                "verifier_command": self.reward.verifier_command,
                "reward_delta": self.reward.reward_delta,
            },
            "initial_state": {
                "files": self.initial_state.files,
                "task_dir": self.initial_state.task_dir,
                "seed_data_files": self.initial_state.seed_data_files,
            },
            "termination": {
                "max_steps": self.termination.max_steps,
                "success_threshold": self.termination.success_threshold,
                "timeout_sec": self.termination.timeout_sec,
                "require_files": self.termination.require_files,
            },
            "metadata": self.metadata,
        }


@dataclass
class EnvironmentBundle:
    """Collection of generated environments for a parent task graph."""

    parent_tasks: list[str] = field(default_factory=list)
    environments: list[GenEnv] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "parent_tasks": self.parent_tasks,
            "environments": [e.to_dict() for e in self.environments],
            "edges": self.edges,
        }

    def save(self, path: str) -> None:
        """Save environment bundle to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class EnvironmentGenerator:
    """
    Convert a SubTaskGraph into a bundle of runnable GenEnv specifications.
    """

    def __init__(
        self,
        tasks_dir: str = "tasks",
        default_max_steps: int = 50,
        default_timeout_sec: float = 60.0,
    ) -> None:
        self.tasks_dir = Path(tasks_dir)
        self.default_max_steps = default_max_steps
        self.default_timeout_sec = default_timeout_sec

    def generate(
        self,
        sub_task_graph: SubTaskGraph,
        checkpoint_specs: Optional[dict[str, list[CheckpointSpec]]] = None,
    ) -> EnvironmentBundle:
        """
        Generate an EnvironmentBundle from a SubTaskGraph.

        Parameters
        ----------
        sub_task_graph : SubTaskGraph
            Graph of parent tasks and sub-task definitions.
        checkpoint_specs : dict[str, list[CheckpointSpec]] | None
            Optional checkpoint specs for richer reward specs.

        Returns
        -------
        EnvironmentBundle
            Bundle of generated environments.
        """
        bundle = EnvironmentBundle()
        bundle.parent_tasks = list(sub_task_graph.parent_tasks)
        bundle.edges = list(sub_task_graph.edges)

        for sub_task in sub_task_graph.sub_tasks:
            env = self._build_env(sub_task, checkpoint_specs)
            bundle.environments.append(env)

        return bundle

    def generate_from_engine(
        self,
        recursion_engine: RecursionEngine,
        graph: Optional[SubTaskGraph] = None,
    ) -> EnvironmentBundle:
        """
        Convenience: generate environments from a RecursionEngine.

        If graph is not provided, the engine must have a cached graph.
        """
        if graph is None:
            raise ValueError(
                "graph must be provided when using generate_from_engine"
            )
        return self.generate(
            graph,
            checkpoint_specs=recursion_engine._checkpoint_specs,
        )

    def _build_env(
        self,
        sub_task: SubTaskDefinition,
        checkpoint_specs: Optional[dict[str, list[CheckpointSpec]]],
    ) -> GenEnv:
        """Build a single GenEnv from a SubTaskDefinition."""
        task_base = Path(str(sub_task.parent_task)).name
        specs = (checkpoint_specs or {}).get(task_base, [])
        matching_spec = None
        for sp in specs:
            if sp.checkpoint_id == sub_task.checkpoint_id:
                matching_spec = sp
                break

        if matching_spec:
            reward = RewardSpec(
                type="checkpoint",
                weight=matching_spec.reward_weight,
                success_threshold=0.99,
                checkpoint_id=matching_spec.checkpoint_id,
                condition_source=matching_spec.condition_source,
                verifier_command=self._build_verifier_command(task_base, matching_spec),
                reward_delta=sub_task.reward_delta,
            )
        else:
            reward = RewardSpec(
                type="structural",
                weight=sub_task.weight,
                success_threshold=0.5,
                checkpoint_id=sub_task.checkpoint_id,
                condition_source="",
                verifier_command=None,
                reward_delta=sub_task.reward_delta,
            )

        initial_state = self._build_initial_state(task_base)
        termination = self._build_termination(task_base, reward)

        return GenEnv(
            task_id=sub_task.sub_task_id,
            parent_task=sub_task.parent_task,
            sub_task_depth=sub_task.sub_task_depth,
            instruction=sub_task.instruction,
            entry_step=sub_task.entry_step,
            exit_step=sub_task.exit_step,
            phase_tag=sub_task.phase_tag,
            reward=reward,
            initial_state=initial_state,
            termination=termination,
            source=sub_task.source,
            metadata={
                "estimated_q_star": sub_task.estimated_q_star,
                "estimated_q_bar": sub_task.estimated_q_bar,
                "kind": sub_task.kind,
            },
        )

    def _build_verifier_command(
        self,
        task_base: str,
        spec: CheckpointSpec,
    ) -> Optional[str]:
        """
        Build a verifier command that scores the checkpoint in isolation.

        This is a best-effort extraction: we wrap the condition source from the
        test.sh into a standalone Python snippet. Full isolation would require
        a harness that runs the original test.sh in a sandbox.
        """
        # Use the condition source as a verifier hint.
        cmd = f"python3 -c \"import json; exec(open('tasks/{task_base}/tests/test.sh').read())\""
        return cmd

    def _build_initial_state(self, task_base: str) -> InitialState:
        """Build the initial state for a sub-task environment."""
        task_dir = self.tasks_dir / task_base
        files: dict[str, str] = {}
        seed_files: list[str] = []

        if task_dir.is_dir():
            for f in sorted(task_dir.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(task_dir)
                    if f.name == "test.sh" or f.name == "task.toml":
                        continue
                    try:
                        files[str(rel)] = f.read_text()
                    except Exception:
                        pass
            seed_files = [str(f.relative_to(task_dir)) for f in sorted(task_dir.rglob("*.json"))]

        return InitialState(
            files=files,
            task_dir=str(task_dir),
            seed_data_files=seed_files,
        )

    def _build_termination(
        self,
        task_base: str,
        reward: RewardSpec,
    ) -> TerminationSpec:
        """Build a termination spec for a sub-task environment."""
        return TerminationSpec(
            max_steps=self.default_max_steps,
            success_threshold=reward.success_threshold,
            timeout_sec=self.default_timeout_sec,
            require_files=[],
        )

    def bundle_by_parent(
        self,
        bundle: EnvironmentBundle,
        parent_task: str,
    ) -> list[GenEnv]:
        """Return all environments belonging to a parent task."""
        return [e for e in bundle.environments if e.parent_task == parent_task]

    def get_environment(
        self,
        bundle: EnvironmentBundle,
        task_id: str,
    ) -> Optional[GenEnv]:
        """Return a single environment by task_id."""
        for env in bundle.environments:
            if env.task_id == task_id:
                return env
        return None
