"""disteval.adapters.swebench_adapter — Bridge SWE-bench/SWE-agent into disteval.

SWE-bench (https://www.swebench.com) is the standard real-world benchmark for
software engineering agents. SWE-agent (https://swe-agent.com) produces:

  - ``predictions.jsonl``: one JSON object per evaluated instance with keys
    ``instance_id``, ``model_name_or_path``, ``text``, and optionally a
    ``trajectory`` path.
  - ``results.json`` (from the SWE-bench evaluation harness): a dict keyed by
    ``instance_id`` whose value contains at least ``resolved`` (bool).

This adapter converts those artifacts into a disteval ``RecordStore`` and,
where trajectory files are available, into disteval ``TrajectoryRecord``
objects.

Design notes:
- ``score`` is binary: 1.0 if resolved, 0.0 otherwise.
- ``task`` maps to the SWE-bench ``instance_id``.
- ``model`` maps to the predicting agent name (e.g. ``swe-agent-gpt4``).
- SWE-agent uses a bash REPL; we map common bash patterns to disteval's tool
  taxonomy (``run_shell_command``, ``read_file``, ``write_file``, etc.) so the
  trajectory monitor and memory can consume them without modification.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..records import EpisodeRecord, RecordStore
from ..trajectory_monitor import TrajectoryRecord, TrajectoryFeatures


# SWE-agent bash actions -> disteval canonical tool names
_SWE_ACTION_TO_TOOL = {
    "read": "read_file",
    "view": "read_file",
    "edit": "write_file",
    "write": "write_file",
    "bash": "run_shell_command",
    "run": "run_shell_command",
    "execute": "run_shell_command",
    "test": "run_shell_command",
    "submit": "submit",
    "finish": "submit",
    "search": "grep_search",
    "find": "list_directory",
}


def load_swebench_results(results_path: str) -> dict[str, dict]:
    """Load the SWE-bench ``results.json`` evaluation harness output.

    Returns a dict mapping ``instance_id`` -> result record.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"SWE-bench results must be a dict, got {type(data).__name__}")
    return data


def load_swe_agent_predictions(predictions_path: str) -> list[dict[str, Any]]:
    """Load SWE-agent ``predictions.jsonl``.

    Returns a list of prediction dicts, one per line.
    """
    predictions = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            predictions.append(json.loads(line))
    return predictions


def load_swebench_predictions(
    predictions_path: str,
    results_path: str,
    *,
    agent_name: str = "swe-agent",
    run_id: str = "swe-run-0",
    trajectory_root: Optional[str] = None,
) -> RecordStore:
    """Convert SWE-agent predictions + SWE-bench results into a disteval RecordStore.

    Parameters
    ----------
    predictions_path: str
        Path to SWE-agent ``predictions.jsonl``.
    results_path: str
        Path to SWE-bench ``results.json``.
    agent_name: str
        Agent/policy name to attach to every record.
    run_id: str
        Run identifier for the eval repetition.
    trajectory_root: Optional[str]
        Directory under which per-instance trajectory files live. If a
        prediction contains a trajectory path, it is resolved relative to this
        root when it is not absolute.

    Returns
    -------
    RecordStore
    """
    results = load_swebench_results(results_path)
    predictions = load_swe_agent_predictions(predictions_path)

    store = RecordStore()
    for i, pred in enumerate(predictions):
        instance_id = pred.get("instance_id")
        if not instance_id:
            continue
        result = results.get(instance_id, {})
        resolved = bool(result.get("resolved", False))
        score = 1.0 if resolved else 0.0

        traj_ref = pred.get("trajectory")
        if traj_ref and trajectory_root and not os.path.isabs(traj_ref):
            traj_ref = os.path.join(trajectory_root, traj_ref)

        store.add(
            EpisodeRecord(
                run_id=run_id,
                model=agent_name,
                task=instance_id,
                episode=i,
                score=score,
                success=resolved,
                strata={"benchmark": "swebench", "domain": "software-engineering"},
                trajectory_ref=traj_ref,
                meta={"model_name_or_path": pred.get("model_name_or_path", "")},
            )
        )
    return store


def _extract_tool_sequence(steps: list[dict]) -> list[str]:
    """Map SWE-agent trajectory steps to disteval canonical tool names."""
    sequence: list[str] = []
    for step in steps:
        action = step.get("action", "") if isinstance(step, dict) else ""
        if isinstance(action, dict):
            action = action.get("action", "")
        action = str(action).lower().strip()
        tool = _SWE_ACTION_TO_TOOL.get(action)
        if tool is None:
            # Fallback: try to infer from the first word
            first = action.split()[0] if action else ""
            tool = _SWE_ACTION_TO_TOOL.get(first, "run_shell_command")
        sequence.append(tool)
    return sequence


def load_swe_agent_trajectory(
    traj_path: str,
    instance_id: str,
    agent_name: str,
    score: float = 0.0,
) -> Optional[TrajectoryRecord]:
    """Load a single SWE-agent trajectory file as a disteval TrajectoryRecord.

    SWE-agent trajectory files are typically JSON with a top-level ``steps``
    or ``trajectory`` list. Each step is a dict with an ``action`` field.
    """
    if not os.path.exists(traj_path):
        return None
    with open(traj_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps") or data.get("trajectory") or []
    if not isinstance(steps, list):
        return None

    sequence = _extract_tool_sequence(steps)
    # Minimal structural features: no per-step rich features available.
    n_steps = len(sequence)
    n_tool_calls = n_steps
    first_write_pos = next((i for i, t in enumerate(sequence) if t == "write_file"), n_steps)
    first_exec_pos = next((i for i, t in enumerate(sequence) if t == "run_shell_command"), n_steps)
    n_exec = sequence.count("run_shell_command")
    n_search = sequence.count("grep_search")
    n_reads = sequence.count("read_file")
    n_writes = sequence.count("write_file")
    search_ratio = n_search / max(n_tool_calls, 1)
    tool_diversity = len(set(sequence)) / max(n_tool_calls, 1) if sequence else 0.0
    features = TrajectoryFeatures(
        n_steps=n_steps,
        n_tool_calls=n_tool_calls,
        first_write_pos=first_write_pos,
        first_exec_pos=first_exec_pos,
        n_reads=n_reads,
        n_writes=n_writes,
        n_exec=n_exec,
        n_search=n_search,
        search_ratio=search_ratio,
        write_before_read=False,
        tool_diversity=tool_diversity,
        prefix_len=n_steps,
        is_prefix=False,
    )

    return TrajectoryRecord(
        trial_id=instance_id,
        task_path=instance_id,
        agent_name=agent_name,
        score=score,
        tool_sequence=sequence,
        traj_path=traj_path,
        features=features,
    )


def load_swebench_trajectories(
    predictions_path: str,
    results_path: str,
    trajectory_root: str,
    *,
    agent_name: str = "swe-agent",
) -> dict[str, TrajectoryRecord]:
    """Load all available SWE-agent trajectories as disteval TrajectoryRecords.

    Returns a dict mapping ``instance_id`` -> TrajectoryRecord.
    """
    results = load_swebench_results(results_path)
    predictions = load_swe_agent_predictions(predictions_path)
    records: dict[str, TrajectoryRecord] = {}
    for pred in predictions:
        instance_id = pred.get("instance_id")
        if not instance_id:
            continue
        traj_ref = pred.get("trajectory")
        if not traj_ref:
            continue
        traj_path = traj_ref if os.path.isabs(traj_ref) else os.path.join(trajectory_root, traj_ref)
        resolved = bool(results.get(instance_id, {}).get("resolved", False))
        score = 1.0 if resolved else 0.0
        rec = load_swe_agent_trajectory(traj_path, instance_id, agent_name, score=score)
        if rec is not None:
            records[instance_id] = rec
    return records
