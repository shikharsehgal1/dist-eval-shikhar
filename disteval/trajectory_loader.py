"""Standalone loader for Harbor job directories into TrajectoryRecords.

This module is intentionally thin and reusable: both `trajectory_monitor.py` and
`trajectory_memory.py` can import it to load real completed trials from the
filesystem.
"""
from __future__ import annotations

import json
import os


def load_trajectory_records(job_dir: str) -> list:
    """Load all completed trials from a Harbor jobs directory.

    Returns a list of `TrajectoryRecord` objects. This is intentionally not
    typed as `list[TrajectoryRecord]` to avoid a circular import with
    `trajectory_monitor.py`, but the elements are instances of that class.

    A trial is considered "completed" when it contains both
    `verifier/reward.txt` and `agent/trajectory.json`.
    """
    # Avoid circular import by importing at runtime.
    from .trajectory_monitor import TrajectoryRecord, TrajectoryFeaturizer

    featurizer = TrajectoryFeaturizer()
    records: list = []

    if not os.path.isdir(job_dir):
        return records

    for trial_name in sorted(os.listdir(job_dir)):
        trial_dir = os.path.join(job_dir, trial_name)
        if not os.path.isdir(trial_dir):
            continue

        traj_path = os.path.join(trial_dir, "agent", "trajectory.json")
        reward_path = os.path.join(trial_dir, "verifier", "reward.txt")
        config_path = os.path.join(trial_dir, "config.json")

        if not (os.path.exists(traj_path) and os.path.exists(reward_path)):
            continue

        try:
            with open(traj_path, "r", encoding="utf-8") as f:
                trajectory = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        try:
            with open(reward_path, "r", encoding="utf-8") as f:
                score = float(f.read().strip())
        except (ValueError, OSError):
            continue

        task_path = ""
        agent_name = ""
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                task_path = config.get("task", {}).get("path", "")
                agent_name = config.get("agent", {}).get("name", "")
            except (json.JSONDecodeError, OSError):
                pass

        if not agent_name:
            agent_name = trajectory.get("agent", {}).get("name", "unknown")

        steps = trajectory.get("steps", [])
        features = featurizer.featurize(steps)
        tool_sequence = featurizer.extract_tool_sequence(steps)

        records.append(
            TrajectoryRecord(
                trial_id=trial_name,
                task_path=task_path,
                agent_name=agent_name,
                score=score,
                features=features,
                tool_sequence=tool_sequence,
                traj_path=traj_path,
            )
        )

    return records
