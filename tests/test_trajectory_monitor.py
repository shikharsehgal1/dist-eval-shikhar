"""
Tests for disteval.trajectory_monitor — real-time trajectory outcome prediction.

These tests cover:
  1. TrajectoryFeaturizer  — feature extraction from raw steps
  2. OutcomePredictor      — logistic regression, LOO accuracy, predict/predict_proba
  3. TrajectoryMonitor     — end-to-end from real job dirs
  4. PatternMatch          — structure and warning logic
"""
import os
import pytest
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

JOBS_ROOT = Path(__file__).parent.parent / "jobs"
JOB_DIRS = [
    str(JOBS_ROOT / "run_A" / "disteval-run-A"),
    str(JOBS_ROOT / "run_B" / "disteval-run-B"),
    str(JOBS_ROOT / "run_C" / "disteval-run-C"),
]
HAS_DATA = all(os.path.isdir(d) for d in JOB_DIRS)


def fake_steps(tool_names: list[str]) -> list[dict]:
    """Build minimal trajectory step dicts from a list of tool names."""
    steps = []
    for i, name in enumerate(tool_names):
        steps.append({
            "step_id": i + 1,
            "source": "agent",
            "tool_calls": [{"function_name": name, "arguments": {}}],
            "observation": {"results": []},
        })
    return steps


# ── TrajectoryFeaturizer ──────────────────────────────────────────────────────

class TestTrajectoryFeaturizer:
    @pytest.fixture
    def featurizer(self):
        from disteval.trajectory_monitor import TrajectoryFeaturizer
        return TrajectoryFeaturizer()

    def test_empty_steps(self, featurizer):
        feat = featurizer.featurize([])
        assert feat.n_tool_calls == 0
        assert feat.n_exec == 0
        assert feat.n_search == 0

    def test_tool_sequence_extraction(self, featurizer):
        steps = fake_steps(["read_file", "write_file", "run_shell_command"])
        seq = featurizer.extract_tool_sequence(steps)
        assert seq == ["read_file", "write_file", "run_shell_command"]

    def test_first_write_pos_detected(self, featurizer):
        # read_file x3, then write_file
        steps = fake_steps(["read_file", "read_file", "read_file", "write_file"])
        feat = featurizer.featurize(steps)
        assert feat.first_write_pos == 3

    def test_first_write_pos_never(self, featurizer):
        steps = fake_steps(["read_file", "list_directory", "grep_search"])
        feat = featurizer.featurize(steps)
        # Never wrote — should be len(seq) or similar sentinel
        assert feat.first_write_pos >= len(steps)

    def test_exec_counts(self, featurizer):
        steps = fake_steps(["run_shell_command", "run_shell_command", "write_file"])
        feat = featurizer.featurize(steps)
        assert feat.n_exec == 2

    def test_search_ratio(self, featurizer):
        # 3 searches out of 4 tools
        steps = fake_steps(["web_search_call", "web_search_call", "web_search_call", "write_file"])
        feat = featurizer.featurize(steps)
        assert feat.search_ratio == pytest.approx(3 / 4)

    def test_n_reads(self, featurizer):
        steps = fake_steps(["read_file", "list_directory", "write_file"])
        feat = featurizer.featurize(steps)
        assert feat.n_reads == 2

    def test_write_before_read(self, featurizer):
        steps = fake_steps(["write_file", "read_file"])
        feat = featurizer.featurize(steps)
        assert feat.write_before_read is True

    def test_read_before_write(self, featurizer):
        steps = fake_steps(["read_file", "write_file"])
        feat = featurizer.featurize(steps)
        assert feat.write_before_read is False

    def test_prefix_featurize(self, featurizer):
        steps = fake_steps(["read_file"] * 5 + ["write_file"] * 5)
        feat_prefix = featurizer.featurize(steps, prefix_n=5)
        assert feat_prefix.is_prefix is True
        assert feat_prefix.prefix_len == 5
        assert feat_prefix.n_writes == 0   # write_file not in first 5

    def test_tool_diversity(self, featurizer):
        # 2 unique tools out of 4 calls
        steps = fake_steps(["read_file", "write_file", "read_file", "write_file"])
        feat = featurizer.featurize(steps)
        assert feat.tool_diversity == pytest.approx(2 / 4)

    def test_all_same_tool(self, featurizer):
        steps = fake_steps(["web_search_call"] * 10)
        feat = featurizer.featurize(steps)
        assert feat.tool_diversity == pytest.approx(1 / 10)


# ── OutcomePredictor ──────────────────────────────────────────────────────────

class TestOutcomePredictor:
    @pytest.fixture
    def records_and_predictor(self):
        from disteval.trajectory_monitor import (
            TrajectoryFeaturizer, OutcomePredictor, TrajectoryRecord
        )
        featurizer = TrajectoryFeaturizer()

        def make_record(tools, score, task="task-a"):
            steps = fake_steps(tools)
            feat = featurizer.featurize(steps)
            return TrajectoryRecord(
                trial_id=f"trial-{score}",
                task_path=task,
                agent_name="test-agent",
                score=score,
                features=feat,
                tool_sequence=tools,
                traj_path="/fake/path",
            )

        # High-outcome: write early, execute
        high_records = [
            make_record(["write_file", "run_shell_command", "run_shell_command"], 1.0)
            for _ in range(5)
        ]
        # Low-outcome: search heavily, never execute
        low_records = [
            make_record(["web_search_call"] * 10, 0.0)
            for _ in range(5)
        ]
        records = high_records + low_records
        predictor = OutcomePredictor().fit(records)
        return records, predictor

    def test_fit_returns_self(self, records_and_predictor):
        records, predictor = records_and_predictor
        from disteval.trajectory_monitor import OutcomePredictor
        p = OutcomePredictor()
        result = p.fit(records)
        assert result is p

    def test_predict_high_for_high_pattern(self, records_and_predictor):
        from disteval.trajectory_monitor import TrajectoryFeaturizer
        records, predictor = records_and_predictor
        featurizer = TrajectoryFeaturizer()
        steps = fake_steps(["write_file", "run_shell_command", "run_shell_command"])
        feat = featurizer.featurize(steps)
        assert predictor.predict(feat) == "high"

    def test_predict_low_for_low_pattern(self, records_and_predictor):
        from disteval.trajectory_monitor import TrajectoryFeaturizer
        records, predictor = records_and_predictor
        featurizer = TrajectoryFeaturizer()
        steps = fake_steps(["web_search_call"] * 15)
        feat = featurizer.featurize(steps)
        assert predictor.predict(feat) == "low"

    def test_predict_proba_range(self, records_and_predictor):
        from disteval.trajectory_monitor import TrajectoryFeaturizer
        records, predictor = records_and_predictor
        featurizer = TrajectoryFeaturizer()
        steps = fake_steps(["write_file"])
        feat = featurizer.featurize(steps)
        p = predictor.predict_proba(feat)
        assert 0.0 <= p <= 1.0

    def test_loo_accuracy_above_chance(self, records_and_predictor):
        records, predictor = records_and_predictor
        acc = predictor.loo_accuracy(records)
        assert acc > 0.5   # must beat random on clearly separable data

    def test_loo_accuracy_range(self, records_and_predictor):
        records, predictor = records_and_predictor
        acc = predictor.loo_accuracy(records)
        assert 0.0 <= acc <= 1.0


# ── TrajectoryMonitor ─────────────────────────────────────────────────────────

class TestTrajectoryMonitor:
    @pytest.fixture
    def monitor(self):
        from disteval.trajectory_monitor import TrajectoryMonitor
        return TrajectoryMonitor.from_job_dirs(JOB_DIRS)

    def test_empty_records_raises(self):
        from disteval.trajectory_monitor import TrajectoryMonitor
        with pytest.raises(ValueError, match="at least one trajectory record"):
            TrajectoryMonitor([])

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_loads_records(self, monitor):
        assert len(monitor.records) > 0

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_loads_expected_count(self, monitor):
        # We know there are 37 trajectories
        assert len(monitor.records) == 37

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_check_returns_pattern_match(self, monitor):
        from disteval.trajectory_monitor import PatternMatch
        steps = fake_steps(["write_file", "run_shell_command"])
        result = monitor.check(steps)
        assert isinstance(result, PatternMatch)

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_check_prediction_is_valid(self, monitor):
        steps = fake_steps(["write_file", "run_shell_command"])
        result = monitor.check(steps)
        assert result.prediction in {"high", "low", "uncertain"}

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_check_confidence_range(self, monitor):
        steps = fake_steps(["web_search_call"] * 10)
        result = monitor.check(steps)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_low_pattern_has_warning(self, monitor):
        # Heavy search, no execution — known low pattern
        steps = fake_steps(["web_search_call"] * 15)
        result = monitor.check(steps)
        if result.prediction == "low":
            assert result.warning is not None
            assert result.recommendation is not None

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_check_has_similar_records(self, monitor):
        steps = fake_steps(["write_file", "run_shell_command"])
        result = monitor.check(steps)
        # similar_high and similar_low should be lists (possibly empty)
        assert isinstance(result.similar_high, list)
        assert isinstance(result.similar_low, list)

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_check_at_step_with_real_traj(self, monitor):
        # Use a real low-outcome trajectory
        traj_path = str(
            JOBS_ROOT / "run_B" / "disteval-run-B" /
            "easy-1__2bNtUEa" / "agent" / "trajectory.json"
        )
        if not os.path.exists(traj_path):
            pytest.skip("specific trajectory not present")
        result = monitor.check_at_step(traj_path, at_step=3)
        assert result.prefix_len <= 3

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_prefix_check_n5(self, monitor):
        # Should work with only 5 tool calls visible
        steps = fake_steps(["web_search_call"] * 5)
        result = monitor.check(steps, prefix_n=5)
        assert result.prefix_len == 5

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_loo_accuracy_above_60pct(self, monitor):
        """The predictor trained on real data should beat 60% LOO accuracy."""
        acc = monitor.predictor.loo_accuracy(monitor.records)
        assert acc >= 0.6, f"LOO accuracy {acc:.2f} below 60% — features may be broken"

    @pytest.mark.skipif(not HAS_DATA, reason="job dirs not present")
    def test_check_empty_steps(self, monitor):
        """Empty trajectory should not crash."""
        result = monitor.check([])
        assert result.prediction in {"high", "low", "uncertain"}


# ── PatternMatch ──────────────────────────────────────────────────────────────

class TestPatternMatch:
    def test_fields_exist(self):
        from disteval.trajectory_monitor import PatternMatch, TrajectoryFeaturizer
        featurizer = TrajectoryFeaturizer()
        feat = featurizer.featurize([])
        pm = PatternMatch(
            prediction="high",
            confidence=0.8,
            p_high=0.8,
            warning=None,
            recommendation=None,
            similar_high=[],
            similar_low=[],
            prefix_len=0,
            features=feat,
        )
        assert pm.prediction == "high"
        assert pm.confidence == pytest.approx(0.8)
        assert pm.similar_high == []
