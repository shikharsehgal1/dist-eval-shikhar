"""Generic adapter: any agent's JSON/JSONL records -> disteval RecordStore.

This adapter is independent of Harbor's file layout. It accepts a standardized
"disteval record" format and converts it into ``EpisodeRecord``/``RecordStore``
objects so any agent or benchmark can feed into disteval's analysis pipeline.

Standard record format (JSONL, one JSON object per line)::

    {
      "run_id": "run_001",
      "model": "my-agent-v2",
      "task": "my_task_name",
      "episode": 0,
      "score": 0.85,
      "success": true,
      "difficulty": "hard",
      "trajectory": "/path/to/trajectory.json",
      "metadata": {"duration_s": 42.1, "n_steps": 15}
    }

Field semantics:

- ``run_id`` (str, required): identifier for the repetition / eval run.
- ``model`` (str, required): agent or policy name.
- ``task`` (str, required): task or environment id.
- ``episode`` (int, default 0): index within ``(run_id, model, task)``.
- ``score`` (float, required): primary outcome in the range ``0.0``–``1.0``.
- ``success`` (bool, optional): binary success. Defaults to
  ``score >= success_threshold``.
- ``difficulty`` (str, optional): stored as a stratification key, e.g.
  ``"easy"`` / ``"medium"`` / ``"hard"``.
- ``trajectory`` (str, optional): path to trajectory file, stored as
  ``EpisodeRecord.trajectory_ref``.
- ``metadata`` (dict, optional): stored as ``EpisodeRecord.meta``.
- Any other top-level string/number keys are promoted to stratification keys
  in ``EpisodeRecord.strata``.

Top-level keys handled explicitly (and therefore not treated as generic strata):
``run_id``, ``model``, ``task``, ``episode``, ``score``, ``success``,
``trajectory``, ``metadata``, ``failure_mode``, ``length``.
``difficulty`` is also a standard field but is explicitly promoted to strata.

Typical usage::

    from disteval.adapters.generic import load_records, GenericAdapter

    store = load_records("my_results.jsonl")
    print(len(store))

    store = GenericAdapter(success_threshold=0.95).from_records(records)
"""
from __future__ import annotations

import json
import os
from typing import Any

from ..records import EpisodeRecord, RecordStore


# Top-level keys that have dedicated EpisodeRecord fields or are explicitly
# handled. Everything else that is a str/int/float is promoted to strata.
_RESERVED_KEYS = frozenset(
    {
        "run_id",
        "model",
        "task",
        "episode",
        "score",
        "success",
        "trajectory",
        "metadata",
        "failure_mode",
        "length",
    }
)


def validate_record(record: dict) -> list[str]:
    """Validate a raw disteval record.

    Returns a list of human-readable error messages. An empty list means the
    record is valid and can be converted with ``GenericAdapter.from_record``.
    """
    errors: list[str] = []

    if not isinstance(record, dict):
        errors.append("Record must be a JSON object (dict).")
        return errors

    required = [("run_id", str), ("model", str), ("task", str)]
    for key, expected in required:
        if key not in record:
            errors.append(f"Missing required field: {key!r}.")
        elif not isinstance(record[key], expected):
            errors.append(
                f"Field {key!r} must be a {expected.__name__}, got "
                f"{type(record[key]).__name__}."
            )

    if "score" not in record:
        errors.append("Missing required field: 'score'.")
    elif not isinstance(record["score"], (int, float)):
        errors.append(
            f"Field 'score' must be a number, got {type(record['score']).__name__}."
        )
    elif isinstance(record["score"], bool):
        errors.append("Field 'score' must be a number, not a bool.")
    else:
        score = float(record["score"])
        if score < 0.0 or score > 1.0:
            errors.append(f"Field 'score' must be between 0.0 and 1.0, got {score}.")

    if "episode" in record and not isinstance(record["episode"], int):
        errors.append(
            f"Field 'episode' must be an int, got {type(record['episode']).__name__}."
        )

    if "success" in record and not isinstance(record["success"], bool):
        errors.append(
            f"Field 'success' must be a bool, got {type(record['success']).__name__}."
        )

    if "metadata" in record and not isinstance(record["metadata"], dict):
        errors.append(
            f"Field 'metadata' must be a dict, got {type(record['metadata']).__name__}."
        )

    if "difficulty" in record and not isinstance(record["difficulty"], str):
        errors.append(
            f"Field 'difficulty' must be a string, got {type(record['difficulty']).__name__}."
        )

    if "trajectory" in record and not isinstance(record["trajectory"], str):
        errors.append(
            f"Field 'trajectory' must be a string, got {type(record['trajectory']).__name__}."
        )

    if "failure_mode" in record and not isinstance(record["failure_mode"], (str, type(None))):
        errors.append(
            f"Field 'failure_mode' must be a string or null, got "
            f"{type(record['failure_mode']).__name__}."
        )

    if "length" in record and not isinstance(record["length"], (int, type(None))):
        errors.append(
            f"Field 'length' must be an int or null, got "
            f"{type(record['length']).__name__}."
        )

    return errors


class GenericAdapter:
    """Convert standardized JSON/JSONL records into a ``RecordStore``."""

    def __init__(self, success_threshold: float = 0.99):
        """Create a generic adapter.

        Args:
            success_threshold: score cutoff for the default ``success`` value.
        """
        self.success_threshold = float(success_threshold)

    # ------------------------------------------------------------------
    # Single-record conversion
    # ------------------------------------------------------------------
    def from_record(self, record: dict) -> EpisodeRecord:
        """Convert one raw dict to an ``EpisodeRecord``.

        Raises:
            ValueError: if the record is missing required fields or has
                invalid types.
        """
        errors = validate_record(record)
        if errors:
            raise ValueError("Invalid record:\n" + "\n".join(f"  - {e}" for e in errors))

        run_id = str(record["run_id"])
        model = str(record["model"])
        task = str(record["task"])
        episode = int(record.get("episode", 0))
        score = float(record["score"])

        success = record.get("success")
        if success is None:
            success = score >= self.success_threshold
        success = bool(success)

        failure_mode = record.get("failure_mode")
        if failure_mode is not None:
            failure_mode = str(failure_mode)
        elif not success:
            # Provide a default failure-mode label when the record does not
            # supply one and success is False.
            failure_mode = "below_threshold"

        length = record.get("length")
        if length is not None:
            length = int(length)

        trajectory_ref = record.get("trajectory")
        if trajectory_ref is not None:
            trajectory_ref = str(trajectory_ref)

        meta = record.get("metadata") or {}

        # Build strata: the canonical `difficulty` field plus any other
        # top-level string/number fields are promoted to stratification keys.
        strata: dict[str, Any] = {}
        for key, value in record.items():
            if key in _RESERVED_KEYS:
                continue
            if isinstance(value, (str, int, float)):
                # Avoid treating booleans as stratification values.
                if isinstance(value, bool):
                    continue
                strata[key] = value
            # Lists/dicts are skipped; put structured data in metadata instead.

        return EpisodeRecord(
            run_id=run_id,
            model=model,
            task=task,
            episode=episode,
            score=score,
            success=success,
            strata=strata,
            failure_mode=failure_mode,
            length=length,
            trajectory_ref=trajectory_ref,
            meta=dict(meta),
        )

    # ------------------------------------------------------------------
    # Bulk conversion
    # ------------------------------------------------------------------
    def from_records(self, records: list[dict]) -> RecordStore:
        """Convert a list of dicts to a ``RecordStore``."""
        store = RecordStore()
        for record in records:
            store.add(self.from_record(record))
        return store

    def from_dicts(self, records: list[dict]) -> RecordStore:
        """Alias for ``from_records``."""
        return self.from_records(records)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    def from_jsonl(self, path: str) -> RecordStore:
        """Load a JSONL file of records."""
        records: list[dict] = []
        with open(path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}: {exc}"
                    ) from exc
        return self.from_records(records)

    def from_json(self, path: str) -> RecordStore:
        """Load a JSON file.

        Accepts either a top-level list of records, or a dict with a
        ``"records"`` key.
        """
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return self.from_records(data)
        if isinstance(data, dict):
            if "records" not in data:
                raise ValueError(
                    f"JSON file {path} must contain a list of records or an object "
                    f"with a 'records' key; got keys: {list(data.keys())}"
                )
            return self.from_records(data["records"])

        raise ValueError(
            f"JSON file {path} must contain a list or an object, got "
            f"{type(data).__name__}"
        )


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------
def load_records(path: str, success_threshold: float = 0.99) -> RecordStore:
    """Auto-detect ``.jsonl`` vs ``.json`` and load into a ``RecordStore``.

    This is the simplest possible entry point for loading disteval records
    from disk.
    """
    adapter = GenericAdapter(success_threshold=success_threshold)
    _, ext = os.path.splitext(path)
    if ext.lower() == ".jsonl":
        return adapter.from_jsonl(path)
    return adapter.from_json(path)


def records_from_dicts(records: list[dict], success_threshold: float = 0.99) -> RecordStore:
    """Build a ``RecordStore`` from a list of plain Python dicts."""
    return GenericAdapter(success_threshold=success_threshold).from_records(records)
