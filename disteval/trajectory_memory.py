"""disteval.trajectory_memory — cross-session memory for agent trajectories.

Core idea
─────────
An agent's memory should be indexed by (task-structural-type + outcome), not just
chronologically. When starting a new task, retrieve the highest-outcome
trajectories from structurally similar past situations. The right-tail insight:
RECOVERABLE tasks have at least one perfect trajectory — that trajectory is the
most valuable memory to retrieve.

This module is pure-numpy only. It builds a searchable bag-of-tools embedding
over completed trajectories and returns structured prompts that an agent can
consume before or during a task.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np

# The standalone loader exists in this package; reuse it. If for some reason it is
# unavailable, the memory module is still usable as long as records are supplied
# directly by callers.
try:
    from .trajectory_loader import load_trajectory_records as _load_trajectory_records
except Exception:  # pragma: no cover - guard against broken loader
    _load_trajectory_records = None


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TrajectoryRecord:
    """One completed trial with the exact fields the memory store expects."""

    trial_id: str
    task_path: str                # e.g. "tasks/medium-2"
    agent_name: str               # e.g. "claude-code"
    score: float
    tool_sequence: list[str]      # flat list of canonical tool call names
    traj_path: str                # path to trajectory.json
    n_steps: int
    first_write_pos: int          # index of first write/exec; len(seq) if never
    n_exec: int
    n_search: int

    # Recursive self-improvement extensions (optional, default-disabled)
    sub_task_slices: list[tuple[int, int, str]] = field(default_factory=list)


@dataclass
class MemoryEntry:
    """A single retrievable memory derived from a TrajectoryRecord."""

    record: TrajectoryRecord
    embedding: np.ndarray         # L2-normalized tool-frequency vector
    outcome_class: str            # "high" | "medium" | "low"
    is_recoverable_high: bool     # valuable: high outcome on a task with low attempts
    summary: str                  # human-readable 1-line summary


@dataclass
class RetrievalResult:
    """One ranked retrieval result."""

    entry: MemoryEntry
    similarity: float             # cosine structural similarity
    task_match: float             # 0-1 keyword overlap with task path
    final_score: float            # combined/ranked score
    rank: int


# ── Memory store ──────────────────────────────────────────────────────────────

class TrajectoryMemory:
    """
    Cross-session memory store for agent trajectories, indexed by outcome.

    Embedding: tool frequency vector (bag-of-tools), L2-normalized.
    Similarity: cosine similarity (dot product of normalized vectors).

    Key design: memory is not just a flat store. It maintains separate
    HIGH and LOW outcome indices, and flags RECOVERABLE-task high trajectories
    as especially valuable (they show the agent finding success where it usually
    fails).

    Usage:
        memory = TrajectoryMemory()
        memory.load_from_job_dirs([
            "jobs/run_A/disteval-run-A",
            "jobs/run_B/disteval-run-B",
            "jobs/run_C/disteval-run-C",
        ])

        # Before starting a new task, retrieve relevant memories:
        results = memory.retrieve_for_new_task("word count script", k=3)
        prompt  = memory.generate_retrieval_prompt(results)

        # After a run completes, add the new trajectory to memory:
        memory.add(new_trajectory_record)
        memory.save("memory.jsonl")
        memory.load("memory.jsonl")
    """

    def __init__(self):
        self.entries: list[MemoryEntry] = []
        self._tool_vocab: list[str] = []   # sorted list of all known tools
        self._tool_to_idx: dict[str, int] = {}
        self._embeddings: np.ndarray | None = None  # shape (N, vocab_size)

    # ── ingestion ─────────────────────────────────────────────────────────────

    def add(self, record: TrajectoryRecord) -> None:
        """Add a trajectory to memory, compute its embedding and metadata."""
        outcome_class = self._outcome_class(record.score)
        embedding = self._embed(record.tool_sequence)
        entry = MemoryEntry(
            record=record,
            embedding=embedding,
            outcome_class=outcome_class,
            is_recoverable_high=False,  # recomputed globally below
            summary=self._make_summary(record, outcome_class, False),
        )
        self.entries.append(entry)
        # Recompute recoverable flags so every entry on this task_path is correct.
        self._update_recoverable_flags()
        self._rebuild_index()

    def load_from_job_dirs(self, job_dirs: list[str]) -> "TrajectoryMemory":
        """Load all completed trials from Harbor job directories."""
        if _load_trajectory_records is None:
            raise RuntimeError(
                "trajectory_loader is not available and no standalone loader is defined"
            )

        raw_records: list[Any] = []
        for job_dir in job_dirs:
            raw_records.extend(_load_trajectory_records(job_dir))

        for raw in raw_records:
            self.add(self._convert_record(raw))
        return self

    @staticmethod
    def _convert_record(raw: Any) -> TrajectoryRecord:
        """Convert a record produced by trajectory_loader to our schema."""
        features = raw.features
        return TrajectoryRecord(
            trial_id=raw.trial_id,
            task_path=raw.task_path,
            agent_name=raw.agent_name,
            score=raw.score,
            tool_sequence=list(raw.tool_sequence),
            traj_path=raw.traj_path,
            n_steps=int(features.n_steps),
            first_write_pos=int(features.first_write_pos),
            n_exec=int(features.n_exec),
            n_search=int(features.n_search),
        )

    # ── indexing and embeddings ───────────────────────────────────────────────

    def _embed(self, tool_sequence: list[str]) -> np.ndarray:
        """Compute L2-normalized tool frequency vector. Handles unseen tools."""
        # Expand vocabulary if necessary.
        new_tools = sorted({t for t in tool_sequence if t not in self._tool_to_idx})
        if new_tools:
            for tool in new_tools:
                self._tool_to_idx[tool] = len(self._tool_vocab)
                self._tool_vocab.append(tool)
            self._rebuild_index()

        vec = np.zeros(len(self._tool_vocab), dtype=float)
        for tool in tool_sequence:
            idx = self._tool_to_idx.get(tool)
            if idx is not None:
                vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _rebuild_index(self) -> None:
        """Rebuild the embeddings matrix from current entries."""
        if not self.entries:
            self._embeddings = None
            return

        # Ensure every existing embedding has the current vocab size.
        n = len(self._tool_vocab)
        for entry in self.entries:
            if entry.embedding.shape[0] != n:
                entry.embedding = self._embed(entry.record.tool_sequence)

        self._embeddings = np.vstack([e.embedding for e in self.entries])

    def _update_recoverable_flags(self) -> None:
        """Recompute is_recoverable_high across all entries by task_path."""
        # Build task_path -> has any low (< 0.5) score.
        low_by_task: dict[str, bool] = {}
        for e in self.entries:
            if e.record.score < 0.5:
                low_by_task[e.record.task_path] = True

        for e in self.entries:
            is_rec = (
                e.outcome_class == "high"
                and low_by_task.get(e.record.task_path, False)
            )
            e.is_recoverable_high = is_rec
            e.summary = self._make_summary(e.record, e.outcome_class, is_rec)

    # ── retrieval ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_tool_sequence: list[str] | None = None,
        query_task_description: str | None = None,
        k: int = 5,
        outcome_filter: str | None = None,
        prefer_recoverable: bool = True,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k similar memories.

        If query_tool_sequence is given, uses structural similarity.
        If query_task_description is given, uses keyword overlap with task_path.
        If both given, combines (0.7 * structural + 0.3 * task_match).
        prefer_recoverable: multiplies similarity score by 1.3 for recoverable-high entries.
        """
        if not self.entries:
            return []

        use_tools = query_tool_sequence is not None
        use_task = query_task_description is not None and query_task_description.strip()

        if not use_tools and not use_task:
            return []

        query_vec = self._embed(query_tool_sequence) if use_tools else None
        query_tokens = self._tokenize(query_task_description) if use_task else set()

        candidates = self.entries
        if outcome_filter is not None:
            candidates = [e for e in candidates if e.outcome_class == outcome_filter]

        scored: list[tuple[float, float, float, MemoryEntry]] = []
        for entry in candidates:
            if use_tools and query_vec is not None:
                # Cosine similarity = dot product of normalized vectors.
                sim = float(query_vec @ entry.embedding)
            else:
                sim = 0.0

            if use_task:
                task_match = self._task_match(query_tokens, entry.record.task_path)
            else:
                task_match = 0.0

            if use_tools and use_task:
                final_score = 0.7 * sim + 0.3 * task_match
            elif use_tools:
                final_score = sim
            else:
                final_score = task_match

            if prefer_recoverable and entry.is_recoverable_high:
                final_score *= 1.3

            scored.append((final_score, sim, task_match, entry))

        # Sort by final score, then by score, then prefer recoverable-high entries.
        scored.sort(
            key=lambda x: (x[0], x[3].record.score, x[3].is_recoverable_high),
            reverse=True,
        )

        results: list[RetrievalResult] = []
        for rank, (final_score, sim, task_match, entry) in enumerate(scored[:k], start=1):
            results.append(
                RetrievalResult(
                    entry=entry,
                    similarity=sim,
                    task_match=task_match,
                    final_score=final_score,
                    rank=rank,
                )
            )
        return results

    def retrieve_for_new_task(
        self,
        task_description: str,
        k: int = 3,
    ) -> list[RetrievalResult]:
        """
        High-level retrieval: given a task description for a NEW task (no trajectory yet),
        find the k most relevant HIGH-outcome memories from similar past tasks.
        """
        return self.retrieve(
            query_tool_sequence=None,
            query_task_description=task_description,
            k=k,
            outcome_filter="high",
            prefer_recoverable=True,
        )

    def retrieve_for_sub_task(
        self,
        sub_task_description: str,
        entry_tool_sequence: list[str],
        k: int = 3,
    ) -> list[RetrievalResult]:
        """
        Retrieve memories for a specific sub-task window.

        Uses the tool sequence inside the sub-task window plus the sub-task
        description (e.g., "medium-2::phase-2 Engineering groupby") to find
        structurally similar successful demonstrations.
        """
        return self.retrieve(
            query_tool_sequence=entry_tool_sequence,
            query_task_description=sub_task_description,
            k=k,
            outcome_filter="high",
            prefer_recoverable=True,
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Extract lowercase alphanumeric tokens from a string."""
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _task_match(query_tokens: set[str], task_path: str) -> float:
        """Return fraction of query tokens that overlap with the task path.

        Overlap includes exact token matches or substring matches, so a query like
        "task" will match the task_path segment "tasks".
        """
        if not query_tokens:
            return 0.0
        task_lower = task_path.lower()
        matched = sum(1 for t in query_tokens if t in task_lower)
        return matched / len(query_tokens)

    # ── prompt generation ───────────────────────────────────────────────────────

    def generate_retrieval_prompt(
        self,
        results: list[RetrievalResult],
        context: str = "before_task",
    ) -> str:
        """
        Generate a structured prompt string that an agent can consume.

        context: "before_task" | "mid_task" | "recovery"
        """
        context_descriptions = {
            "before_task": (
                "Before you start a new task, here are the most relevant past "
                "successful approaches from similar situations."
            ),
            "mid_task": (
                "Your current approach resembles these past runs. Consider these "
                "patterns when deciding your next action."
            ),
            "recovery": (
                "You're on a low-outcome path. Here's what worked before on similar "
                "tasks — use these trajectories as a recovery template."
            ),
        }
        context_description = context_descriptions.get(
            context, context_descriptions["before_task"]
        )

        k = len(results)
        title = f"MEMORY RETRIEVAL — {k} relevant past trajectories"
        width = max(62, len(title) + 4)
        top_line = "╔" + "═" * width + "╗"
        title_line = "║" + title.center(width) + "║"
        bottom_line = "╚" + "═" * width + "╝"

        lines = [top_line, title_line, bottom_line, "", f"CONTEXT: {context_description}", ""]

        for r in results:
            entry = r.entry
            record = entry.record
            approach = " -> ".join(record.tool_sequence)
            if len(approach) > 80:
                approach = approach[:77] + "..."

            lines.append(
                f"  ━━ Memory #{r.rank} (similarity={r.similarity:.2f}, "
                f"outcome={entry.outcome_class}) ━━"
            )
            lines.append(f"  Task:     {record.task_path}")
            rec_tag = "★ RECOVERABLE-HIGH" if entry.is_recoverable_high else ""
            lines.append(
                f"  Score:    {record.score:.2f}  [{rec_tag}]"
            )
            lines.append(f"  Summary:  {entry.summary}")
            lines.append(f"  Approach: {approach}")
            lines.append("")
            lines.append(f"  KEY INSIGHT: {self._key_insight(record)}")
            lines.append("")

        lines.append("RECOMMENDATION:")
        lines.append(f"  {self._recommendation(results)}")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _key_insight(record: TrajectoryRecord) -> str:
        """Extract a concise structural insight from a record."""
        seq_len = len(record.tool_sequence)
        if record.first_write_pos >= seq_len:
            return "No write or execution actions; trajectory was purely exploratory."
        first_act = record.tool_sequence[record.first_write_pos]
        if record.n_exec > 0:
            return (
                f"First write/exec at step {record.first_write_pos + 1} "
                f"({first_act}), executed {record.n_exec}x across {record.n_steps} steps."
            )
        return (
            f"First write/exec at step {record.first_write_pos + 1} "
            f"({first_act}), but the run never executed code."
        )

    @staticmethod
    def _recommendation(results: list[RetrievalResult]) -> str:
        """Build a contrastive recommendation from high and low results."""
        high_entries = [r.entry for r in results if r.entry.outcome_class == "high"]
        low_entries = [r.entry for r in results if r.entry.outcome_class == "low"]

        parts: list[str] = []

        if high_entries:
            avg_first_write = sum(
                e.record.first_write_pos for e in high_entries
            ) / max(1, len(high_entries))
            avg_exec = sum(e.record.n_exec for e in high_entries) / max(1, len(high_entries))
            common = sorted(
                {
                    t
                    for e in high_entries
                    for t in e.record.tool_sequence
                    if t in {"write_file", "run_shell_command", "exec_command", "edit", "read_file"}
                }
            )
            pattern = (
                f"High-outcome runs typically write/execute early "
                f"(first write/exec at average index {avg_first_write:.1f}), "
                f"run code {avg_exec:.1f} times, and rely on tools like {', '.join(common) if common else 'core edit/exec tools'}."
            )
            parts.append(pattern)

        if low_entries:
            avg_first_write = sum(
                e.record.first_write_pos for e in low_entries
            ) / max(1, len(low_entries))
            parts.append(
                f"In contrast, low-outcome runs in this set delayed their first "
                f"write/exec until index {avg_first_write:.1f} — avoid that pattern."
            )

        if not parts:
            parts.append(
                "No clear high/low contrast in the retrieved memories. Proceed with "
                "your best judgement and test your code frequently."
            )

        return " ".join(parts)

    # ── persistence ─────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save memory to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                row = asdict(entry.record)
                row["embedding"] = entry.embedding.tolist()
                row["outcome_class"] = entry.outcome_class
                row["is_recoverable_high"] = entry.is_recoverable_high
                row["summary"] = entry.summary
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load(self, path: str) -> "TrajectoryMemory":
        """Load memory from JSONL file."""
        self.entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                record = TrajectoryRecord(
                    trial_id=row["trial_id"],
                    task_path=row["task_path"],
                    agent_name=row["agent_name"],
                    score=row["score"],
                    tool_sequence=list(row["tool_sequence"]),
                    traj_path=row["traj_path"],
                    n_steps=row["n_steps"],
                    first_write_pos=row["first_write_pos"],
                    n_exec=row["n_exec"],
                    n_search=row["n_search"],
                )
                entry = MemoryEntry(
                    record=record,
                    embedding=np.array(row["embedding"], dtype=float),
                    outcome_class=row["outcome_class"],
                    is_recoverable_high=row["is_recoverable_high"],
                    summary=row["summary"],
                )
                self.entries.append(entry)

        self._update_recoverable_flags()
        self._rebuild_index()
        return self

    # ── utilities ───────────────────────────────────────────────────────────────

    @staticmethod
    def _outcome_class(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.25:
            return "medium"
        return "low"

    @staticmethod
    def _make_summary(
        record: TrajectoryRecord,
        outcome_class: str,
        is_recoverable_high: bool,
    ) -> str:
        seq_len = len(record.tool_sequence)
        if record.first_write_pos < seq_len:
            first_act = record.tool_sequence[record.first_write_pos]
        else:
            first_act = "none"
        prefix = "★ RECOVERABLE-HIGH: " if is_recoverable_high else ""
        return (
            f"{prefix}[{outcome_class}] {record.n_steps} steps, "
            f"first action={first_act}, executed={record.n_exec}x — score {record.score:.2f}"
        )

    def stats(self) -> dict:
        """Return stats: n_entries, n_high, n_low, n_recoverable_high, vocab_size."""
        return {
            "n_entries": len(self.entries),
            "n_high": sum(1 for e in self.entries if e.outcome_class == "high"),
            "n_low": sum(1 for e in self.entries if e.outcome_class == "low"),
            "n_recoverable_high": sum(
                1 for e in self.entries if e.is_recoverable_high
            ),
            "vocab_size": len(self._tool_vocab),
        }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    job_dirs = [
        os.path.join(project_root, "jobs", "run_A", "disteval-run-A"),
        os.path.join(project_root, "jobs", "run_B", "disteval-run-B"),
        os.path.join(project_root, "jobs", "run_C", "disteval-run-C"),
    ]

    print("Loading trajectory memory from job directories...")
    memory = TrajectoryMemory().load_from_job_dirs(job_dirs)
    print(f"Loaded {len(memory.entries)} completed trials.")
    print("Stats:", memory.stats())

    print("\n--- Retrieval for a new 'word count task' ---")
    results = memory.retrieve_for_new_task("word count task", k=3)
    for r in results:
        print(
            f"#{r.rank}: {r.entry.record.task_path:<24} "
            f"score={r.entry.record.score:.2f} "
            f"sim={r.similarity:.2f} match={r.task_match:.2f} "
            f"final={r.final_score:.2f} "
            f"rec={r.entry.is_recoverable_high}"
        )

    print("\n--- Sample prompt (before_task) ---")
    prompt = memory.generate_retrieval_prompt(results, context="before_task")
    print(prompt)

    # Quick persistence sanity check.
    tmp_path = os.path.join(project_root, "memory_selftest.jsonl")
    memory.save(tmp_path)
    memory2 = TrajectoryMemory().load(tmp_path)
    print("\nPersistence sanity check:")
    print(f"  saved {len(memory.entries)} -> loaded {len(memory2.entries)} entries")
    print(f"  loaded stats: {memory2.stats()}")
    os.remove(tmp_path)
