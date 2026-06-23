"""Tests for disteval adapters."""
import json
import os
import tempfile

import pytest

from disteval.adapters.generic import load_records, validate_record
from disteval.adapters.rliable_bridge import score_matrix, to_rliable_dict
from disteval.records import EpisodeRecord, RecordStore


class TestGenericAdapter:
    def test_load_records_from_jsonl(self):
        records = [
            {"run_id": "r1", "model": "m", "task": "t1", "episode": 0, "score": 0.9},
            {"run_id": "r1", "model": "m", "task": "t1", "episode": 1, "score": 0.5},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            path = f.name
        try:
            store = load_records(path)
            assert len(store) == 2
            assert store._records[0].score == pytest.approx(0.9)
            assert store._records[0].success is False  # default threshold is 0.95
        finally:
            os.unlink(path)

    def test_load_records_with_strata(self):
        records = [
            {"run_id": "r1", "model": "m", "task": "t1", "episode": 0,
             "score": 0.8, "difficulty": "hard", "domain": "finance"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            path = f.name
        try:
            store = load_records(path)
            rec = store._records[0]
            assert rec.strata["difficulty"] == "hard"
            assert rec.strata["domain"] == "finance"
        finally:
            os.unlink(path)

    def test_validate_record_missing_required(self):
        errors = validate_record({"score": 0.5})
        assert any("run_id" in e for e in errors)
        assert any("model" in e for e in errors)
        assert any("task" in e for e in errors)

    def test_validate_record_invalid_score_type(self):
        errors = validate_record({"run_id": "r", "model": "m", "task": "t", "score": "high"})
        assert any("number" in e for e in errors)


class TestRliableBridge:
    def test_score_matrix_shape(self):
        store = RecordStore([
            EpisodeRecord(run_id="r1", model="m", task="t1", episode=0, score=0.9, success=True),
            EpisodeRecord(run_id="r1", model="m", task="t2", episode=0, score=0.7, success=True),
            EpisodeRecord(run_id="r2", model="m", task="t1", episode=0, score=0.8, success=True),
            EpisodeRecord(run_id="r2", model="m", task="t2", episode=0, score=0.6, success=True),
        ])
        matrix, tasks = score_matrix(store)
        assert matrix.shape == (2, 2)
        assert set(tasks) == {"t1", "t2"}

    def test_to_rliable_dict(self):
        s1 = RecordStore([
            EpisodeRecord(run_id="r1", model="m", task="t1", episode=0, score=0.9, success=True),
            EpisodeRecord(run_id="r1", model="m", task="t2", episode=0, score=0.7, success=True),
        ])
        s2 = RecordStore([
            EpisodeRecord(run_id="r1", model="m", task="t1", episode=0, score=0.8, success=True),
            EpisodeRecord(run_id="r1", model="m", task="t2", episode=0, score=0.6, success=True),
        ])
        d = to_rliable_dict({"A": s1, "B": s2})
        assert "A" in d
        assert "B" in d
        assert d["A"].shape == (1, 2)
