"""
Tests for disteval.trajectory_memory — cross-session memory keyed by trajectory outcome.

These tests cover:
  1. MemoryEntry           — fields, outcome_class, is_recoverable_high, summary
  2. TrajectoryMemory      — add, embed, retrieve, retrieve_for_new_task
  3. generate_retrieval_prompt — structure and content
  4. save/load persistence
  5. Integration with real job dirs
"""
import os
import json
import tempfile
import pytest
import numpy as np
from pathlib import Path

JOBS_ROOT = Path(__file__).parent.parent / "jobs"
JOB_DIRS = [
    str(JOBS_ROOT / "run_A" / "disteval-run-A"),
    str(JOBS_ROOT / "run_B" / "disteval-run-B"),
    str(JOBS_ROOT / "run_C" / "disteval-run-C"),
]
HAS_DATA = all(os.path.isdir(d) for d in JOB_DIRS)


def make_trajectory_record(
    tool_sequence: list[str],
    score: float,
    task_path: str = "tasks/easy-1",
    trial_id: str = "t0",
    agent_name: str = "test-agent",
    traj_path: str = "/fake/path.json",
):
    """Build a minimal TrajectoryRecord for testing."""
    from disteval.trajectory_memory import TrajectoryRecord
    n = len(tool_sequence)
    write_tools = {"write_file", "run_shell_command", "exec_command", "write_todos"}
    first_write = next((i for i, t in enumerate(tool_sequence) if t in write_tools), n)
    n_exec = sum(1 for t in tool_sequence if t in {"run_shell_command", "exec_command"})
    n_search = sum(1 for t in tool_sequence if "search" in t)
    return TrajectoryRecord(
        trial_id=trial_id,
        task_path=task_path,
        agent_name=agent_name,
        score=score,
        tool_sequence=tool_sequence,
        traj_path=traj_path,
        n_steps=n,
        first_write_pos=first_write,
        n_exec=n_exec,
        n_search=n_search,
    )


# ── MemoryEntry ───────────────────────────────────────────────────────────────

class TestMemoryEntry:
    def _entry(self, score, tools=None, is_recov_high=False):
        from disteval.trajectory_memory import MemoryEntry
        tools = tools or ["write_file", "run_shell_command"]
        rec = make_trajectory_record(tools, score)
        # outcome_class
        if score >= 0.75: cls = "high"
        elif score >= 0.25: cls = "medium"
        else: cls = "low"
        emb = np.array([1.0, 0.0, 0.5])
        norm = np.linalg.norm(emb)
        emb /= norm
        return MemoryEntry(
            record=rec,
            embedding=emb,
            outcome_class=cls,
            is_recoverable_high=is_recov_high,
            summary=f"[{cls}] 2 steps, score {score:.2f}",
        )

    def test_fields_accessible(self):
        entry = self._entry(1.0)
        assert entry.outcome_class == "high"
        assert entry.is_recoverable_high is False
        assert entry.embedding is not None

    def test_recoverable_high_flag(self):
        entry = self._entry(1.0, is_recov_high=True)
        assert entry.is_recoverable_high is True

    def test_outcome_class_low(self):
        entry = self._entry(0.0)
        assert entry.outcome_class == "low"

    def test_outcome_class_medium(self):
        entry = self._entry(0.5)
        assert entry.outcome_class == "medium"

    def test_embedding_normalized(self):
        entry = self._entry(1.0)
        norm = np.linalg.norm(entry.embedding)
        assert norm == pytest.approx(1.0, abs=1e-6)


# ── TrajectoryMemory ──────────────────────────────────────────────────────────

class TestTrajectoryMemory:
    @pytest.fixture
    def memory_with_records(self):
        from disteval.trajectory_memory import TrajectoryMemory
        mem = TrajectoryMemory()
        # High: write early, execute
        for i in range(3):
            rec = make_trajectory_record(
                ["write_file", "run_shell_command", "run_shell_command"],
                score=1.0, task_path="tasks/easy-1", trial_id=f"high-{i}"
            )
            mem.add(rec)
        # Low: search only
        for i in range(3):
            rec = make_trajectory_record(
                ["web_search_call"] * 10,
                score=0.0, task_path="tasks/easy-1", trial_id=f"low-{i}"
            )
            mem.add(rec)
        return mem

    def test_add_increases_count(self, memory_with_records):
        assert len(memory_with_records.entries) == 6

    def test_stats_high_low_count(self, memory_with_records):
        stats = memory_with_records.stats()
        assert stats["n_high"] == 3
        assert stats["n_low"] == 3
        assert stats["n_entries"] == 6

    def test_embedding_normalized(self, memory_with_records):
        for entry in memory_with_records.entries:
            norm = np.linalg.norm(entry.embedding)
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_retrieve_returns_k_results(self, memory_with_records):
        results = memory_with_records.retrieve(
            query_tool_sequence=["write_file", "run_shell_command"],
            k=3,
        )
        assert len(results) == 3

    def test_retrieve_sorted_by_score(self, memory_with_records):
        results = memory_with_records.retrieve(
            query_tool_sequence=["write_file", "run_shell_command"],
            k=6,
        )
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_outcome_filter_high(self, memory_with_records):
        results = memory_with_records.retrieve(
            query_tool_sequence=["write_file"],
            k=10,
            outcome_filter="high",
        )
        assert all(r.entry.outcome_class == "high" for r in results)

    def test_retrieve_outcome_filter_low(self, memory_with_records):
        results = memory_with_records.retrieve(
            query_tool_sequence=["web_search_call"] * 5,
            k=10,
            outcome_filter="low",
        )
        assert all(r.entry.outcome_class == "low" for r in results)

    def test_retrieve_for_new_task_returns_high(self, memory_with_records):
        results = memory_with_records.retrieve_for_new_task("easy-1 task", k=3)
        # Should return high-outcome entries
        assert len(results) > 0
        assert all(r.entry.outcome_class == "high" for r in results)

    def test_retrieve_result_has_rank(self, memory_with_records):
        results = memory_with_records.retrieve(
            query_tool_sequence=["write_file"],
            k=3,
        )
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_recoverable_high_flagged(self):
        """is_recoverable_high should be set when same task has both high and low scores."""
        from disteval.trajectory_memory import TrajectoryMemory
        mem = TrajectoryMemory()
        mem.add(make_trajectory_record(["write_file", "run_shell_command"], 1.0,
                                       task_path="tasks/easy-1", trial_id="h0"))
        mem.add(make_trajectory_record(["web_search_call"] * 5, 0.0,
                                       task_path="tasks/easy-1", trial_id="l0"))
        # After both are added, the high one should be is_recoverable_high=True
        # This requires the memory to recompute recoverable flags after all adds
        # OR to compute it at retrieval time. Check that at least one is flagged.
        high_entries = [e for e in mem.entries if e.outcome_class == "high"]
        assert any(e.is_recoverable_high for e in high_entries)

    def test_empty_memory_retrieve(self):
        from disteval.trajectory_memory import TrajectoryMemory
        mem = TrajectoryMemory()
        results = mem.retrieve(query_tool_sequence=["write_file"], k=3)
        assert results == []

    def test_stats_vocab_size(self, memory_with_records):
        stats = memory_with_records.stats()
        assert stats["vocab_size"] > 0


# ── generate_retrieval_prompt ─────────────────────────────────────────────────

class TestGenerateRetrievalPrompt:
    @pytest.fixture
    def memory_and_results(self):
        from disteval.trajectory_memory import TrajectoryMemory
        mem = TrajectoryMemory()
        for i in range(3):
            rec = make_trajectory_record(
                ["write_file", "run_shell_command"],
                score=1.0, task_path="tasks/easy-1", trial_id=f"t{i}"
            )
            mem.add(rec)
        results = mem.retrieve(query_tool_sequence=["write_file"], k=3)
        return mem, results

    def test_returns_string(self, memory_and_results):
        mem, results = memory_and_results
        prompt = mem.generate_retrieval_prompt(results)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_memory_header(self, memory_and_results):
        mem, results = memory_and_results
        prompt = mem.generate_retrieval_prompt(results)
        assert "MEMORY" in prompt.upper() or "memory" in prompt.lower()

    def test_prompt_contains_task_path(self, memory_and_results):
        mem, results = memory_and_results
        prompt = mem.generate_retrieval_prompt(results)
        assert "easy-1" in prompt

    def test_prompt_contains_score(self, memory_and_results):
        mem, results = memory_and_results
        prompt = mem.generate_retrieval_prompt(results)
        assert "1.0" in prompt or "1.00" in prompt

    def test_prompt_before_task_context(self, memory_and_results):
        mem, results = memory_and_results
        prompt = mem.generate_retrieval_prompt(results, context="before_task")
        assert isinstance(prompt, str)

    def test_prompt_recovery_context(self, memory_and_results):
        mem, results = memory_and_results
        prompt = mem.generate_retrieval_prompt(results, context="recovery")
        assert isinstance(prompt, str)

    def test_empty_results_prompt(self):
        from disteval.trajectory_memory import TrajectoryMemory
        mem = TrajectoryMemory()
        prompt = mem.generate_retrieval_prompt([])
        assert isinstance(prompt, str)


# ── save / load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load_roundtrip(self):
        from disteval.trajectory_memory import TrajectoryMemory
        mem = TrajectoryMemory()
        for i in range(4):
            rec = make_trajectory_record(
                ["write_file", "run_shell_command"] * (i + 1),
                score=float(i % 2), task_path=f"tasks/task-{i}", trial_id=f"t{i}"
            )
            mem.add(rec)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            mem.save(path)
            mem2 = TrajectoryMemory()
            mem2.load(path)
            assert len(mem2.entries) == len(mem.entries)
        finally:
            os.unlink(path)

    def test_load_preserves_embeddings(self):
        from disteval.trajectory_memory import TrajectoryMemory
        mem = TrajectoryMemory()
        rec = make_trajectory_record(["write_file", "run_shell_command"], 1.0)
        mem.add(rec)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            mem.save(path)
            mem2 = TrajectoryMemory()
            mem2.load(path)
            orig = mem.entries[0].embedding
            loaded = mem2.entries[0].embedding
            np.testing.assert_allclose(orig, loaded, atol=1e-6)
        finally:
            os.unlink(path)

    def test_load_preserves_recoverable_flag(self):
        from disteval.trajectory_memory import TrajectoryMemory
        mem = TrajectoryMemory()
        mem.add(make_trajectory_record(["write_file"], 1.0, task_path="tasks/t", trial_id="h"))
        mem.add(make_trajectory_record(["web_search_call"] * 5, 0.0, task_path="tasks/t", trial_id="l"))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            mem.save(path)
            mem2 = TrajectoryMemory()
            mem2.load(path)
            high_entries = [e for e in mem2.entries if e.outcome_class == "high"]
            assert any(e.is_recoverable_high for e in high_entries)
        finally:
            os.unlink(path)


# ── Integration: real job directories ────────────────────────────────────────

class TestIntegrationWithRealData:
    @pytest.fixture
    def loaded_memory(self):
        from disteval.trajectory_memory import TrajectoryMemory
        mem = TrajectoryMemory()
        mem.load_from_job_dirs(JOB_DIRS)
        return mem

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_loads_all_trajectories(self, loaded_memory):
        assert len(loaded_memory.entries) == 37

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_has_high_and_low_entries(self, loaded_memory):
        stats = loaded_memory.stats()
        assert stats["n_high"] > 0
        assert stats["n_low"] > 0

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_recoverable_high_entries_exist(self, loaded_memory):
        stats = loaded_memory.stats()
        assert stats["n_recoverable_high"] > 0

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_retrieve_for_word_count_task(self, loaded_memory):
        results = loaded_memory.retrieve_for_new_task("word count script", k=3)
        assert len(results) > 0

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_retrieval_prompt_nonempty(self, loaded_memory):
        results = loaded_memory.retrieve_for_new_task("fizzbuzz python", k=3)
        prompt = loaded_memory.generate_retrieval_prompt(results)
        assert len(prompt) > 100

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_high_outcome_retrieval_similarity(self, loaded_memory):
        """A 'write then execute' query should retrieve high-outcome trajectories."""
        results = loaded_memory.retrieve(
            query_tool_sequence=["write_file", "run_shell_command", "run_shell_command"],
            k=5,
            outcome_filter="high",
        )
        assert all(r.entry.outcome_class == "high" for r in results)

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_save_load_with_real_data(self, loaded_memory):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            loaded_memory.save(path)
            from disteval.trajectory_memory import TrajectoryMemory
            mem2 = TrajectoryMemory()
            mem2.load(path)
            assert len(mem2.entries) == len(loaded_memory.entries)
        finally:
            os.unlink(path)
