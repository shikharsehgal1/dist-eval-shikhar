"""
disteval.environment_registry — Persistent registry of generated RL environments.

OVERVIEW
────────
Stores, indexes, and retrieves GenEnv specifications produced by
EnvironmentGenerator. The registry is backed by a JSON file and supports:

  - Register/unregister environments
  - Lookup by task_id, parent_task, or phase_tag
  - Save/load to JSON
  - Deduplication by task_id
  - Listing all registered environments

This is a lightweight persistence layer; the actual RL harness consumes the
GenEnv dicts and runs them in its own sandbox.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from .environment_generator import EnvironmentBundle, GenEnv


@dataclass
class EnvironmentRegistry:
    """
    Persistent registry of generated environments.

    Usage:
        registry = EnvironmentRegistry()
        registry.load("environments.json")
        registry.register(env)
        env = registry.get("medium-2::phase-1")
        registry.save("environments.json")
    """

    environments: dict[str, GenEnv] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    parent_tasks: list[str] = field(default_factory=list)

    def register(self, env: GenEnv) -> None:
        """Add or replace a GenEnv in the registry."""
        self.environments[env.task_id] = env
        if env.parent_task not in self.parent_tasks:
            self.parent_tasks.append(env.parent_task)
        edge = (env.parent_task, env.task_id)
        if edge not in self.edges:
            self.edges.append(edge)

    def register_bundle(self, bundle: EnvironmentBundle) -> None:
        """Register all environments from a bundle."""
        for env in bundle.environments:
            self.register(env)
        for edge in bundle.edges:
            if edge not in self.edges:
                self.edges.append(edge)
        for parent in bundle.parent_tasks:
            if parent not in self.parent_tasks:
                self.parent_tasks.append(parent)

    def unregister(self, task_id: str) -> bool:
        """Remove a GenEnv from the registry. Returns True if it existed."""
        return self.environments.pop(task_id, None) is not None

    def get(self, task_id: str) -> Optional[GenEnv]:
        """Return a single environment by task_id."""
        return self.environments.get(task_id)

    def get_by_parent(self, parent_task: str) -> list[GenEnv]:
        """Return all environments for a given parent task."""
        return [e for e in self.environments.values() if e.parent_task == parent_task]

    def get_by_phase_tag(self, phase_tag: str) -> list[GenEnv]:
        """Return all environments with a given phase tag."""
        return [e for e in self.environments.values() if e.phase_tag == phase_tag]

    def list_task_ids(self) -> list[str]:
        """Return all registered task IDs."""
        return list(self.environments.keys())

    def list_parent_tasks(self) -> list[str]:
        """Return all known parent tasks."""
        return list(self.parent_tasks)

    def to_bundle(self) -> EnvironmentBundle:
        """Export the registry as an EnvironmentBundle."""
        return EnvironmentBundle(
            parent_tasks=list(self.parent_tasks),
            environments=list(self.environments.values()),
            edges=list(self.edges),
        )

    def to_dict(self) -> dict:
        """Serialize the registry to a dict."""
        return self.to_bundle().to_dict()

    def save(self, path: str) -> None:
        """Save the registry to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str) -> "EnvironmentRegistry":
        """Load the registry from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.environments = {}
        self.edges = []
        self.parent_tasks = []

        for item in data.get("environments", []):
            env = self._dict_to_env(item)
            self.register(env)

        for edge in data.get("edges", []):
            edge = tuple(edge)
            if edge not in self.edges:
                self.edges.append(edge)

        for parent in data.get("parent_tasks", []):
            if parent not in self.parent_tasks:
                self.parent_tasks.append(parent)

        return self

    @staticmethod
    def _dict_to_env(item: dict) -> GenEnv:
        """Reconstruct a GenEnv from a serialized dict."""
        from .environment_generator import InitialState, RewardSpec, TerminationSpec

        reward = RewardSpec(**item["reward"])
        initial_state = InitialState(**item["initial_state"])
        termination = TerminationSpec(**item["termination"])
        return GenEnv(
            task_id=item["task_id"],
            parent_task=item["parent_task"],
            sub_task_depth=item["sub_task_depth"],
            instruction=item["instruction"],
            entry_step=item["entry_step"],
            exit_step=item["exit_step"],
            phase_tag=item["phase_tag"],
            reward=reward,
            initial_state=initial_state,
            termination=termination,
            source=item.get("source", "checkpoint"),
            metadata=item.get("metadata", {}),
        )

    def __len__(self) -> int:
        return len(self.environments)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self.environments
