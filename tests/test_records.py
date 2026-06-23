"""Tests for disteval.records – EpisodeRecord and RecordStore."""
import os
import tempfile

import pandas as pd
import pytest

from disteval.records import EpisodeRecord, RecordStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def basic_records():
    """Three records spread across two tasks and two models."""
    r1 = EpisodeRecord(run_id="run1", model="A", task="t1", episode=0,
                       score=0.75, success=True)
    r2 = EpisodeRecord(run_id="run1", model="A", task="t1", episode=1,
                       score=0.50, success=False, failure_mode="timeout")
    r3 = EpisodeRecord(run_id="run1", model="B", task="t2", episode=0,
                       score=1.00, success=True,
                       strata={"difficulty": "hard"})
    return [r1, r2, r3]


@pytest.fixture()
def store(basic_records):
    return RecordStore(basic_records)


# ---------------------------------------------------------------------------
# EpisodeRecord
# ---------------------------------------------------------------------------

class TestEpisodeRecord:
    def test_required_fields(self):
        r = EpisodeRecord(run_id="r0", model="m", task="t", episode=0,
                          score=1.0, success=True)
        assert r.run_id == "r0"
        assert r.model == "m"
        assert r.task == "t"
        assert r.episode == 0
        assert r.score == 1.0
        assert r.success is True

    def test_optional_defaults(self):
        r = EpisodeRecord(run_id="r0", model="m", task="t", episode=0,
                          score=0.0, success=False)
        assert r.failure_mode is None
        assert r.length is None
        assert r.trajectory_ref is None
        assert r.strata == {}
        assert r.meta == {}

    def test_failure_mode_set(self):
        r = EpisodeRecord(run_id="r0", model="m", task="t", episode=1,
                          score=0.0, success=False, failure_mode="crash")
        assert r.failure_mode == "crash"

    def test_strata_stored(self):
        r = EpisodeRecord(run_id="r0", model="m", task="t", episode=2,
                          score=0.5, success=True,
                          strata={"difficulty": "easy", "domain": "finance"})
        assert r.strata["difficulty"] == "easy"
        assert r.strata["domain"] == "finance"

    def test_length_and_meta(self):
        r = EpisodeRecord(run_id="r0", model="m", task="t", episode=3,
                          score=0.9, success=True,
                          length=42, meta={"seed": 7})
        assert r.length == 42
        assert r.meta["seed"] == 7


# ---------------------------------------------------------------------------
# RecordStore – ingestion
# ---------------------------------------------------------------------------

class TestRecordStoreIngestion:
    def test_empty_store(self):
        s = RecordStore()
        assert len(s) == 0

    def test_init_with_records(self, basic_records):
        s = RecordStore(basic_records)
        assert len(s) == 3

    def test_add_increments_length(self):
        s = RecordStore()
        r = EpisodeRecord(run_id="r", model="m", task="t", episode=0,
                          score=0.0, success=False)
        s.add(r)
        assert len(s) == 1

    def test_extend_multiple(self):
        s = RecordStore()
        recs = [
            EpisodeRecord(run_id="r", model="m", task="t", episode=i,
                          score=float(i), success=True)
            for i in range(5)
        ]
        s.extend(recs)
        assert len(s) == 5


# ---------------------------------------------------------------------------
# RecordStore – df()
# ---------------------------------------------------------------------------

class TestRecordStoreDF:
    def test_df_shape(self, store):
        df = store.df()
        assert df.shape[0] == 3          # 3 records
        assert "score" in df.columns
        assert "success" in df.columns
        assert "task" in df.columns
        assert "model" in df.columns
        assert "run_id" in df.columns
        assert "episode" in df.columns
        assert "failure_mode" in df.columns
        assert "length" in df.columns

    def test_df_strata_column_promoted(self, store):
        # r3 has strata={"difficulty": "hard"} → column s_difficulty must exist
        df = store.df()
        assert "s_difficulty" in df.columns

    def test_df_strata_value_correct(self, store):
        df = store.df()
        hard_rows = df[df["s_difficulty"] == "hard"]
        assert len(hard_rows) == 1
        assert hard_rows.iloc[0]["task"] == "t2"

    def test_df_scores_correct(self, store):
        df = store.df()
        assert list(df["score"]) == pytest.approx([0.75, 0.50, 1.00])

    def test_df_failure_mode_present(self, store):
        df = store.df()
        row = df[df["episode"] == 1].iloc[0]
        assert row["failure_mode"] == "timeout"

    def test_df_returns_dataframe(self, store):
        assert isinstance(store.df(), pd.DataFrame)


# ---------------------------------------------------------------------------
# RecordStore – scores()
# ---------------------------------------------------------------------------

class TestRecordStoreScores:
    def test_scores_length(self, store):
        assert len(store.scores()) == 3

    def test_scores_values(self, store):
        assert list(store.scores()) == pytest.approx([0.75, 0.50, 1.00])

    def test_scores_returns_series(self, store):
        assert isinstance(store.scores(), pd.Series)

    def test_scores_mean(self, store):
        assert store.scores().mean() == pytest.approx((0.75 + 0.50 + 1.00) / 3)


# ---------------------------------------------------------------------------
# RecordStore – strata_cols
# ---------------------------------------------------------------------------

class TestRecordStoreStrataCols:
    def test_strata_cols_detected(self, store):
        assert "s_difficulty" in store.strata_cols

    def test_no_strata_cols(self):
        s = RecordStore([
            EpisodeRecord(run_id="r", model="m", task="t", episode=0,
                          score=1.0, success=True)
        ])
        assert s.strata_cols == []


# ---------------------------------------------------------------------------
# RecordStore – slice()
# ---------------------------------------------------------------------------

class TestRecordStoreSlice:
    def test_slice_by_model(self, store):
        sub = store.slice(model="A")
        assert len(sub) == 2

    def test_slice_by_model_b(self, store):
        sub = store.slice(model="B")
        assert len(sub) == 1

    def test_slice_by_task(self, store):
        sub = store.slice(task="t1")
        assert len(sub) == 2

    def test_slice_by_strata(self, store):
        sub = store.slice(difficulty="hard")
        assert len(sub) == 1

    def test_slice_by_combined_model_and_task(self, store):
        sub = store.slice(model="A", task="t1")
        assert len(sub) == 2

    def test_slice_by_nonexistent_key_returns_empty(self, store):
        sub = store.slice(model="nonexistent")
        assert len(sub) == 0

    def test_slice_by_strata_no_match_returns_empty(self, store):
        sub = store.slice(difficulty="easy")
        assert len(sub) == 0

    def test_slice_returns_record_store(self, store):
        sub = store.slice(model="A")
        assert isinstance(sub, RecordStore)

    def test_slice_scores_correct(self, store):
        sub = store.slice(model="A")
        assert list(sub.scores()) == pytest.approx([0.75, 0.50])

    def test_slice_by_run_id(self, store):
        sub = store.slice(run_id="run1")
        assert len(sub) == 3

    def test_filter_by_predicate(self, store):
        sub = store.filter(lambda r: r.score >= 0.75)
        assert len(sub) == 2

    def test_query_by_score_range(self, store):
        sub = store.query(score_min=0.5, score_max=1.0)
        assert len(sub) == 3

    def test_query_by_success_and_model(self, store):
        sub = store.query(success=True, models=["A"])
        assert len(sub) == 1
        assert sub._records[0].task == "t1"

    def test_query_by_strata_list(self, store):
        sub = store.query(difficulty=["hard"])
        assert len(sub) == 1
        assert sub._records[0].task == "t2"

    def test_group_by_task(self, store):
        groups = store.group_by("task")
        assert len(groups) == 2
        assert len(groups[("t1",)]) == 2
        assert len(groups[("t2",)]) == 1


# ---------------------------------------------------------------------------
# RecordStore – Parquet round-trip
# ---------------------------------------------------------------------------

class TestRecordStoreParquet:
    def test_roundtrip_preserves_optional_columns(self, store):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            store.to_parquet(path)
            loaded = RecordStore.from_parquet(path)
            assert len(loaded) == len(store)
            for orig, rec in zip(store._records, loaded._records):
                assert orig.failure_mode == rec.failure_mode
                assert orig.trajectory_ref == rec.trajectory_ref
        finally:
            os.unlink(path)

    def test_from_parquet_missing_optional_columns(self):
        df = pd.DataFrame({
            "run_id": ["r1"],
            "model": ["m"],
            "task": ["t1"],
            "episode": [0],
            "score": [0.5],
            "success": [True],
        })
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            df.to_parquet(path)
            loaded = RecordStore.from_parquet(path)
            assert len(loaded) == 1
            assert loaded._records[0].failure_mode is None
            assert loaded._records[0].trajectory_ref is None
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# RecordStore – JSONL round-trip
# ---------------------------------------------------------------------------

class TestRecordStoreJSONL:
    def test_roundtrip_preserves_length(self, store):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            store.to_jsonl(path)
            loaded = RecordStore.from_jsonl(path)
            assert len(loaded) == len(store)
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_scores(self, store):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            store.to_jsonl(path)
            loaded = RecordStore.from_jsonl(path)
            assert list(loaded.scores()) == pytest.approx(list(store.scores()))
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_strata(self, store):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            store.to_jsonl(path)
            loaded = RecordStore.from_jsonl(path)
            hard_sub = loaded.slice(difficulty="hard")
            assert len(hard_sub) == 1
        finally:
            os.unlink(path)
