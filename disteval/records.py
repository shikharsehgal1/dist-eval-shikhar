"""Per-episode record store with first-class stratification.

This is the primitive the landscape review flagged as missing: a persisted,
queryable store with *one row per episode/sample* and arbitrary stratification
keys promoted to first-class columns (not metadata afterthoughts).

Design rule #1 from the review: never collapse early. We keep raw per-episode
values; every aggregate is derived on demand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Optional

import pandas as pd


@dataclass
class EpisodeRecord:
    """One episode / rollout / eval sample. The atomic unit we never throw away."""

    # --- identity ---
    run_id: str          # which *repetition* of the whole eval (see repeat.py) or training run
    model: str           # agent / policy under evaluation
    task: str            # task / environment id
    episode: int         # index within (run_id, model, task)

    # --- outcome ---
    score: float         # primary scalar outcome: return, reward, normalized score
    success: bool        # binary task success -> feeds pass@k / pass^k

    # --- stratification keys (arbitrary, promoted to columns) ---
    strata: dict = field(default_factory=dict)   # {"difficulty": "hard", "domain": "airline", "seed": 3}

    # --- failure tagging ---
    failure_mode: Optional[str] = None            # taxonomy label, populated when not success

    # --- provenance ---
    length: Optional[int] = None
    trajectory_ref: Optional[str] = None
    meta: dict = field(default_factory=dict)


class RecordStore:
    """Holds EpisodeRecords; exposes a tidy DataFrame view and stratified slicing.

    Persistence is JSONL (works everywhere) or Parquet (columnar, fast slicing).
    In a real deployment this is where you'd point at Inspect `.eval` logs — see
    adapters/inspect_log.py for the seam.
    """

    def __init__(self, records: Optional[Iterable[EpisodeRecord]] = None):
        self._records: list[EpisodeRecord] = list(records or [])

    # -- ingestion --
    def add(self, rec: EpisodeRecord) -> None:
        self._records.append(rec)

    def extend(self, recs: Iterable[EpisodeRecord]) -> None:
        self._records.extend(recs)

    def __len__(self) -> int:
        return len(self._records)

    # -- tidy view: strata flattened to s_<key> columns --
    def df(self) -> pd.DataFrame:
        rows = []
        for r in self._records:
            row = {
                "run_id": r.run_id,
                "model": r.model,
                "task": r.task,
                "episode": r.episode,
                "score": r.score,
                "success": r.success,
                "failure_mode": r.failure_mode,
                "length": r.length,
            }
            for k, v in r.strata.items():
                row[f"s_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def strata_cols(self) -> list[str]:
        return [c for c in self.df().columns if c.startswith("s_")]

    # -- slicing: the missing first-class primitive --
    def slice(self, **kv: Any) -> "RecordStore":
        """Return a sub-store filtered by stratification keys, model, or task.

        e.g. store.slice(model="A", difficulty="hard")
        """
        def keep(r: EpisodeRecord) -> bool:
            for k, want in kv.items():
                if k in ("model", "task", "run_id"):
                    if getattr(r, k) != want:
                        return False
                elif r.strata.get(k) != want:
                    return False
            return True

        return RecordStore([r for r in self._records if keep(r)])

    def scores(self) -> "pd.Series":
        return self.df()["score"]

    # -- persistence --
    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            for r in self._records:
                f.write(json.dumps(asdict(r)) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "RecordStore":
        recs = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    recs.append(EpisodeRecord(**json.loads(line)))
        return cls(recs)

    def to_parquet(self, path: str) -> None:
        self.df().to_parquet(path)

    @classmethod
    def from_parquet(cls, path: str) -> "RecordStore":
        import pyarrow.parquet as pq
        df = pd.read_parquet(path)
        store = cls()
        for _, row in df.iterrows():
            strata = {k[2:]: v for k, v in row.items() if k.startswith("s_") and pd.notna(v)}
            store.add(EpisodeRecord(
                run_id=row["run_id"], model=row["model"], task=row["task"],
                episode=int(row["episode"]), score=float(row["score"]),
                success=bool(row["success"]),
                strata=strata,
                failure_mode=row.get("failure_mode") if pd.notna(row.get("failure_mode")) else None,
                trajectory_ref=row.get("trajectory_ref") if pd.notna(row.get("trajectory_ref")) else None,
            ))
        return store

    @classmethod
    def merge(cls, *stores: "RecordStore") -> "RecordStore":
        """Merge multiple RecordStores into one."""
        merged = cls()
        for s in stores:
            merged.extend(s._records)
        return merged
