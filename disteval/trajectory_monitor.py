"""disteval.trajectory_monitor — real-time trajectory pattern monitoring.

CORE IDEA
──────────
An agent's trajectory *structural signature* (the sequence of tool calls it
makes) predicts the eventual outcome. Historical data shows:

    high-outcome runs (score ≥ 1.0): first write/exec around tool call 2.4,
                                   ~2.9 execution calls
    low-outcome runs  (score < 0.5): first write/exec around tool call 24.7,
                                   ~0.2 execution calls

This module featurizes those sequences, trains a lightweight logistic-regression
classifier from scratch in numpy, and provides a live monitor that can warn
when a partial trajectory looks like a historically low-outcome pattern.

Usage
─────
    monitor = TrajectoryMonitor.from_job_dirs([
        "jobs/run_A/disteval-run-A",
        "jobs/run_B/disteval-run-B",
        "jobs/run_C/disteval-run-C",
    ])

    match = monitor.check(current_steps, prefix_n=len(current_steps))
    if match.prediction == "low":
        print(match.warning)
        print(match.recommendation)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class TrajectoryFeatures:
    """Feature vector describing a full or prefix trajectory."""

    n_steps: int
    n_tool_calls: int
    first_write_pos: int          # index of first write_file/exec/run_shell_command
    first_exec_pos: int           # index of first run_shell_command/exec_command
    n_reads: int                  # count of read_file / list_directory calls
    n_writes: int                 # count of write_file calls
    n_exec: int                   # count of run_shell_command / exec_command calls
    n_search: int                 # count of any *search* tool calls
    search_ratio: float           # n_search / max(n_tool_calls, 1)
    write_before_read: bool       # first write happens before first read
    tool_diversity: float         # unique tools / total tools
    prefix_len: int               # how many tool calls are in this prefix
    is_prefix: bool               # True if this is a partial trajectory


@dataclass
class TrajectoryRecord:
    """One completed trial with its trajectory features."""

    trial_id: str
    task_path: str                # e.g. "tasks/medium-2"
    agent_name: str               # e.g. "claude-code"
    score: float
    features: TrajectoryFeatures
    tool_sequence: list[str]
    traj_path: str

    # Recursive self-improvement extensions (optional, default-disabled)
    entry_step: int = 0           # first tool-call index of the sub-task window
    exit_step: int = -1          # last tool-call index of the sub-task window (-1 = full)


@dataclass
class PatternMatch:
    """Output of a live trajectory check."""

    prediction: str               # "high" | "low" | "uncertain"
    confidence: float             # 0.0-1.0
    p_high: float                 # P(high outcome)
    warning: Optional[str]        # human-readable warning if prediction="low"
    recommendation: Optional[str] # what to do differently
    similar_high: list[TrajectoryRecord]  # top-3 similar HIGH past trajectories
    similar_low: list[TrajectoryRecord]   # top-3 similar LOW past trajectories
    prefix_len: int
    features: TrajectoryFeatures

    # Recursive self-improvement extensions (optional, default-disabled)
    entry_step: int = 0           # sub-task window start
    exit_step: int = -1          # sub-task window end
    phase_tag: Optional[str] = None


# ── Featurizer ───────────────────────────────────────────────────────────────

class TrajectoryFeaturizer:
    """Extract a flat tool sequence and featurize full or prefix trajectories."""

    WRITE_TOOLS = {"write_file", "write_todos", "run_shell_command", "exec_command"}
    EXEC_TOOLS = {"run_shell_command", "exec_command"}
    READ_TOOLS = {"read_file", "list_directory"}

    # Map real-world tool names to canonical categories.
    _TOOL_ALIASES: dict[str, str] = {
        "write": "write_file",
        "write_file": "write_file",
        "write_todos": "write_todos",
        "create_file": "write_file",
        "edit_file": "write_file",
        "apply_edit": "write_file",
        "run_shell_command": "run_shell_command",
        "exec_command": "exec_command",
        "bash": "run_shell_command",
        "bash_command": "run_shell_command",
        "run_command": "run_shell_command",
        "execute_command": "run_shell_command",
        "execute": "exec_command",
        "read_file": "read_file",
        "read": "read_file",
        "view": "read_file",
        "list_directory": "list_directory",
        "list_files": "list_directory",
        "list_dir": "list_directory",
    }

    def _normalize(self, tool_name: str) -> str:
        """Map a raw tool name to a canonical category."""
        if not tool_name:
            return "unknown"
        key = tool_name.strip()
        # Direct alias first.
        if key in self._TOOL_ALIASES:
            return self._TOOL_ALIASES[key]
        # Case-insensitive alias.
        lower_key = key.lower()
        if lower_key in self._TOOL_ALIASES:
            return self._TOOL_ALIASES[lower_key]
        # Broad search heuristic.
        if re.search(r"search", lower_key):
            return "search_tool"
        return lower_key

    def extract_tool_sequence(self, steps: list[dict]) -> list[str]:
        """Extract flat list of tool call function names from trajectory steps."""
        sequence: list[str] = []
        for step in steps:
            for tc in step.get("tool_calls") or []:
                raw_name = tc.get("function_name") or tc.get("tool_use_name")
                if raw_name:
                    sequence.append(self._normalize(raw_name))
        return sequence

    def featurize(
        self,
        steps: list[dict],
        prefix_n: Optional[int] = None,
    ) -> TrajectoryFeatures:
        """Featurize full or prefix trajectory. prefix_n=None means use all steps."""
        full_sequence = self.extract_tool_sequence(steps)
        total_tool_calls = len(full_sequence)

        if prefix_n is None:
            sequence = full_sequence
            is_prefix = False
        else:
            prefix_n = max(0, int(prefix_n))
            sequence = full_sequence[:prefix_n]
            is_prefix = prefix_n < total_tool_calls

        prefix_len = len(sequence)
        n_steps = prefix_len  # in this context, "step" means tool call

        first_write_pos = prefix_len
        first_exec_pos = prefix_len
        first_read_pos = prefix_len

        n_reads = 0
        n_writes = 0
        n_exec = 0
        n_search = 0

        for idx, tool in enumerate(sequence):
            if tool in self.WRITE_TOOLS:
                if first_write_pos == prefix_len:
                    first_write_pos = idx
            if tool in self.EXEC_TOOLS:
                if first_exec_pos == prefix_len:
                    first_exec_pos = idx
            if tool in self.READ_TOOLS:
                if first_read_pos == prefix_len:
                    first_read_pos = idx

            if tool in self.READ_TOOLS:
                n_reads += 1
            if tool in self.WRITE_TOOLS:
                n_writes += 1
            if tool in self.EXEC_TOOLS:
                n_exec += 1
            if tool == "search_tool":
                n_search += 1

        search_ratio = n_search / max(prefix_len, 1)
        write_before_read = (
            first_write_pos < prefix_len
            and first_read_pos < prefix_len
            and first_write_pos < first_read_pos
        )
        tool_diversity = len(set(sequence)) / max(prefix_len, 1)

        return TrajectoryFeatures(
            n_steps=n_steps,
            n_tool_calls=prefix_len,
            first_write_pos=first_write_pos,
            first_exec_pos=first_exec_pos,
            n_reads=n_reads,
            n_writes=n_writes,
            n_exec=n_exec,
            n_search=n_search,
            search_ratio=search_ratio,
            write_before_read=write_before_read,
            tool_diversity=tool_diversity,
            prefix_len=prefix_len,
            is_prefix=is_prefix,
        )


# ── Logistic-regression predictor (numpy only) ─────────────────────────────────

class OutcomePredictor:
    """Lightweight classifier predicting high vs low outcome from features.

    Uses logistic regression implemented from scratch in numpy.
    """

    # Feature names used for the model (order matters).
    _FEATURE_NAMES = [
        "log1p_first_write_pos",
        "log1p_first_exec_pos",
        "n_exec",
        "n_search",
        "search_ratio",
        "n_writes",
        "n_reads",
        "write_before_read",
        "tool_diversity",
        "n_steps",
    ]

    def __init__(self):
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def _feature_vector(self, features: TrajectoryFeatures) -> np.ndarray:
        """Build a numeric feature vector from a TrajectoryFeatures object."""
        return np.array(
            [
                np.log1p(features.first_write_pos),
                np.log1p(features.first_exec_pos),
                features.n_exec,
                features.n_search,
                features.search_ratio,
                features.n_writes,
                features.n_reads,
                float(features.write_before_read),
                features.tool_diversity,
                features.n_steps,
            ],
            dtype=float,
        )

    def _standardize(self, X: np.ndarray, *, fit: bool = False) -> np.ndarray:
        """Z-score standardize the feature matrix."""
        if fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) + 1e-6
        return (X - self.mean) / self.std

    def fit(self, records: list[TrajectoryRecord]) -> "OutcomePredictor":
        """Train on a list of TrajectoryRecords."""
        if not records:
            self.weights = np.zeros(len(self._FEATURE_NAMES), dtype=float)
            self.bias = 0.0
            return self

        X = np.vstack([self._feature_vector(r.features) for r in records])
        y = np.array([1.0 if r.score >= 0.5 else 0.0 for r in records], dtype=float)

        Xs = self._standardize(X, fit=True)

        n_features = Xs.shape[1]
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        lr = 0.1
        n_iter = 50
        n_samples = Xs.shape[0]

        for _ in range(n_iter):
            z = Xs @ self.weights + self.bias
            p = self._sigmoid(z)
            error = p - y
            grad_w = (Xs.T @ error) / n_samples
            grad_b = error.mean()
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

        return self

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[neg])
        out[neg] = exp_z / (1.0 + exp_z)
        return out

    def predict_proba(self, features: TrajectoryFeatures) -> float:
        """Return P(high outcome) given features. 0.0-1.0."""
        if self.weights is None or self.mean is None:
            return 0.5
        x = self._feature_vector(features)
        xs = (x - self.mean) / self.std
        z = float(xs @ self.weights + self.bias)
        return float(self._sigmoid(np.array([z]))[0])

    def predict(self, features: TrajectoryFeatures, threshold: float = 0.5) -> str:
        """Return 'high' or 'low'."""
        return "high" if self.predict_proba(features) >= threshold else "low"

    def loo_accuracy(self, records: list[TrajectoryRecord]) -> float:
        """Leave-one-out cross-validation accuracy."""
        if len(records) < 2:
            return 0.0

        X = np.vstack([self._feature_vector(r.features) for r in records])
        y = np.array([1.0 if r.score >= 0.5 else 0.0 for r in records], dtype=float)
        mean_full = X.mean(axis=0)
        std_full = X.std(axis=0) + 1e-6

        correct = 0
        for i in range(len(records)):
            mask = np.ones(len(records), dtype=bool)
            mask[i] = False
            X_train = X[mask]
            y_train = y[mask]

            mean_i = X_train.mean(axis=0)
            std_i = X_train.std(axis=0) + 1e-6
            Xs_train = (X_train - mean_i) / std_i

            w = np.zeros(X.shape[1], dtype=float)
            b = 0.0
            lr = 0.1
            n_iter = 50
            n_samples = Xs_train.shape[0]
            for _ in range(n_iter):
                z = Xs_train @ w + b
                p = self._sigmoid(z)
                error = p - y_train
                grad_w = (Xs_train.T @ error) / n_samples
                grad_b = error.mean()
                w -= lr * grad_w
                b -= lr * grad_b

            x_test = (X[i] - mean_i) / std_i
            p = float(self._sigmoid(np.array([x_test @ w + b]))[0])
            pred = 1.0 if p >= 0.5 else 0.0
            if pred == y[i]:
                correct += 1

        return correct / len(records)


# ── Live monitor ─────────────────────────────────────────────────────────────

class TrajectoryMonitor:
    """Real-time trajectory monitor loaded with past TrajectoryRecords."""

    def __init__(self, records: list[TrajectoryRecord]):
        self.records = records
        self.featurizer = TrajectoryFeaturizer()
        self.predictor = OutcomePredictor().fit(records)

        # Build a vocabulary of all canonical tools for similarity vectors.
        all_tools: set[str] = set()
        for r in records:
            all_tools.update(r.tool_sequence)
        self._tool_vocab = sorted(all_tools)
        self._tool_to_idx = {t: i for i, t in enumerate(self._tool_vocab)}

    @classmethod
    def from_job_dirs(cls, job_dirs: list[str]) -> "TrajectoryMonitor":
        """Load all trajectory records from a list of Harbor job directories."""
        # Avoid a circular import by loading the loader at runtime.
        from .trajectory_loader import load_trajectory_records

        records: list[TrajectoryRecord] = []
        for job_dir in job_dirs:
            if not os.path.isdir(job_dir):
                continue

            recs = load_trajectory_records(job_dir)
            if recs:
                records.extend(recs)
                continue

            # Fallback: if the directory itself has no trial subdirs, look one
            # level deeper for a single Harbor run subdir.
            subdirs = [
                os.path.join(job_dir, d)
                for d in sorted(os.listdir(job_dir))
                if os.path.isdir(os.path.join(job_dir, d))
            ]
            for subdir in subdirs:
                records.extend(load_trajectory_records(subdir))

        return cls(records)

    def _tool_vector(self, tool_sequence: list[str]) -> np.ndarray:
        """Build a normalized bag-of-tools vector for a tool sequence."""
        vec = np.zeros(len(self._tool_vocab), dtype=float)
        if not tool_sequence:
            return vec
        for tool in tool_sequence:
            idx = self._tool_to_idx.get(tool)
            if idx is not None:
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _most_similar(
        self,
        query_sequence: list[str],
        outcome: str,
        k: int = 3,
    ) -> list[TrajectoryRecord]:
        """Return top-k similar records of a given outcome (high or low)."""
        query_vec = self._tool_vector(query_sequence)
        candidates = [
            r for r in self.records if ("high" if r.score >= 0.5 else "low") == outcome
        ]
        if not candidates:
            return []

        scores = []
        for r in candidates:
            r_vec = self._tool_vector(r.tool_sequence)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(r_vec)
            sim = float(query_vec @ r_vec) / denom if denom > 0 else 0.0
            scores.append((sim, r))

        scores.sort(key=lambda x: (-x[0], x[1].score))
        return [r for _, r in scores[:k]]

    def _warning_recommendation(
        self,
        features: TrajectoryFeatures,
        prediction: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Generate warning and recommendation for low-outcome predictions."""
        if prediction != "low":
            return None, None

        if features.n_search > 5 and features.n_exec == 0:
            return (
                "Searching extensively without executing any code. "
                "High-outcome runs typically execute within the first 10 tool calls.",
                "Stop searching. Write a minimal implementation and test it immediately.",
            )
        if features.first_write_pos > 10 and features.n_exec == 0:
            return (
                f"{features.first_write_pos} tool calls in and no code written or executed yet. "
                "Low-outcome runs show this pattern.",
                "Write a skeleton solution now, even if incomplete. Iterating on code beats planning.",
            )
        if features.n_steps > 20 and features.n_exec < 2:
            return (
                "Long trajectory with minimal execution. This matches the low-outcome pattern.",
                "Switch to implementation mode: write, execute, verify in a tight loop.",
            )

        return (
            "Trajectory pattern matches low-outcome historical runs.",
            "Consider a different approach — check similar successful trajectories.",
        )

    def check(
        self,
        steps: list[dict],
        prefix_n: Optional[int] = None,
    ) -> PatternMatch:
        """Check current (possibly partial) trajectory against learned patterns."""
        features = self.featurizer.featurize(steps, prefix_n=prefix_n)
        p_high = self.predictor.predict_proba(features)

        # Uncertainty band around 0.5.
        if 0.35 <= p_high < 0.65:
            prediction = "uncertain"
            confidence = 1.0 - abs(p_high - 0.5) * 2
        else:
            prediction = "high" if p_high >= 0.5 else "low"
            confidence = p_high if prediction == "high" else 1.0 - p_high

        warning, recommendation = self._warning_recommendation(features, prediction)

        query_sequence = self.featurizer.extract_tool_sequence(steps)
        if prefix_n is not None:
            query_sequence = query_sequence[:prefix_n]

        similar_high = self._most_similar(query_sequence, "high", k=3)
        similar_low = self._most_similar(query_sequence, "low", k=3)

        return PatternMatch(
            prediction=prediction,
            confidence=float(confidence),
            p_high=float(p_high),
            warning=warning,
            recommendation=recommendation,
            similar_high=similar_high,
            similar_low=similar_low,
            prefix_len=features.prefix_len,
            features=features,
        )

    def load_trajectory_steps(self, traj_path: str) -> list[dict]:
        """Load steps from a trajectory.json file."""
        with open(traj_path, "r", encoding="utf-8") as f:
            trajectory = json.load(f)
        return trajectory.get("steps", [])

    def check_at_step(self, traj_path: str, at_step: int) -> PatternMatch:
        """Load a trajectory file and check at a specific step index."""
        steps = self.load_trajectory_steps(traj_path)
        return self.check(steps, prefix_n=at_step)

    def find_phase_boundaries(
        self,
        traj_path: str,
        min_confidence: float = 0.70,
        max_boundaries: int = 5,
    ) -> list[dict]:
        """
        Propose structural phase boundaries for a completed trajectory.

        A boundary is a step index where the monitor's p_high crosses the
        confidence threshold (>= min_confidence for high, <= 1-min_confidence for
        low) or where the dominant tool category changes. This is a heuristic
        approximation of RMDP entry/exit points.

        Returns a list of dicts with keys:
            step_index, tool_name, p_high, phase_tag, confidence
        """
        steps = self.load_trajectory_steps(traj_path)
        sequence = self.featurizer.extract_tool_sequence(steps)
        n = len(sequence)
        if n == 0:
            return []

        boundaries = []
        prev_tag = None
        for i in range(1, n + 1):
            match = self.check(steps, prefix_n=i)
            tool = sequence[i - 1] if i <= len(sequence) else "unknown"
            tag = self._phase_tag(tool)

            # Boundary triggers
            high_conf = match.p_high >= min_confidence
            low_conf = match.p_high <= (1.0 - min_confidence)
            tag_changed = tag != prev_tag and prev_tag is not None

            if high_conf or low_conf or tag_changed:
                boundaries.append({
                    "step_index": i,
                    "tool_name": tool,
                    "p_high": match.p_high,
                    "phase_tag": tag,
                    "confidence": match.confidence,
                })
                prev_tag = tag
            else:
                prev_tag = tag

            if len(boundaries) >= max_boundaries:
                break

        return boundaries

    def _phase_tag(self, tool: str) -> str:
        """Map a canonical tool to a coarse phase tag."""
        if tool in ("write_file", "write_todos"):
            return "write"
        if tool in ("run_shell_command", "exec_command"):
            return "exec"
        if tool in ("read_file", "list_directory"):
            return "read"
        if tool == "search_tool":
            return "search"
        return "other"

    def divergence_steps(
        self,
        high_traj_path: str,
        low_traj_path: str,
        min_confidence: float = 0.70,
        max_boundaries: int = 5,
    ) -> list[int]:
        """
        Return step indices where a high-scoring and low-scoring trajectory
        first structurally diverge according to the monitor. Multiple divergence
        points are returned for multi-phase tasks.
        """
        high_steps = self.load_trajectory_steps(high_traj_path)
        low_steps = self.load_trajectory_steps(low_traj_path)
        high_seq = self.featurizer.extract_tool_sequence(high_steps)
        low_seq = self.featurizer.extract_tool_sequence(low_steps)
        max_len = min(len(high_seq), len(low_seq), max_boundaries * 3)

        divergences = []
        for i in range(1, max_len + 1):
            high_match = self.check(high_steps, prefix_n=i)
            low_match = self.check(low_steps, prefix_n=i)
            high_confident = high_match.p_high >= min_confidence
            low_confident = low_match.p_high <= (1.0 - min_confidence)
            if high_confident and low_confident:
                divergences.append(i)
                if len(divergences) >= max_boundaries:
                    break
        return divergences


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    job_dirs = [
        os.path.join(project_root, "jobs", "run_A", "disteval-run-A"),
        os.path.join(project_root, "jobs", "run_B", "disteval-run-B"),
        os.path.join(project_root, "jobs", "run_C", "disteval-run-C"),
    ]

    print("Loading trajectory records...")
    monitor = TrajectoryMonitor.from_job_dirs(job_dirs)
    print(f"Loaded {len(monitor.records)} completed trials.")

    if not monitor.records:
        print("No records found; exiting.")
        sys.exit(0)

    high_count = sum(1 for r in monitor.records if r.score >= 0.5)
    low_count = len(monitor.records) - high_count
    print(f"High outcomes (score >= 0.5): {high_count}")
    print(f"Low outcomes (score < 0.5):  {low_count}")

    loo_acc = monitor.predictor.loo_accuracy(monitor.records)
    print(f"\nLeave-one-out accuracy: {loo_acc:.2%}")

    # Sample predictions at prefix_n=5.
    print("\nSample predictions at prefix_n=5:")
    for r in monitor.records[:3]:
        match = monitor.check_at_step(r.traj_path, at_step=5)
        print(
            f"  {r.trial_id}: score={r.score:.2f}, "
            f"pred={match.prediction:<9}, p_high={match.p_high:.3f}, "
            f"prefix_len={match.prefix_len}"
        )
        if match.warning:
            print(f"    warning: {match.warning}")
        if match.recommendation:
            print(f"    recommendation: {match.recommendation}")

    print("\nSelf-test complete.")
