"""Tests for disteval.adapters.swebench_adapter."""
import json
import os
import tempfile

import pytest

from disteval.adapters import swebench_adapter


class TestLoadSwebenchPredictions:
    def test_loads_predictions_and_results(self):
        predictions = [
            {"instance_id": "django-1", "model_name_or_path": "gpt-4"},
            {"instance_id": "django-2", "model_name_or_path": "gpt-4"},
        ]
        results = {
            "django-1": {"resolved": True},
            "django-2": {"resolved": False},
        }
        with tempfile.TemporaryDirectory() as d:
            pred_path = os.path.join(d, "predictions.jsonl")
            res_path = os.path.join(d, "results.json")
            with open(pred_path, "w") as f:
                for p in predictions:
                    f.write(json.dumps(p) + "\n")
            with open(res_path, "w") as f:
                json.dump(results, f)
            store = swebench_adapter.load_swebench_predictions(
                pred_path, res_path, agent_name="swe-agent"
            )
            assert len(store) == 2
            recs = {r.task: r for r in store._records}
            assert recs["django-1"].success is True
            assert recs["django-1"].score == pytest.approx(1.0)
            assert recs["django-2"].success is False
            assert recs["django-2"].score == pytest.approx(0.0)

    def test_missing_instance_skipped(self):
        predictions = [
            {"instance_id": "django-1"},
        ]
        results = {}
        with tempfile.TemporaryDirectory() as d:
            pred_path = os.path.join(d, "predictions.jsonl")
            res_path = os.path.join(d, "results.json")
            with open(pred_path, "w") as f:
                f.write(json.dumps(predictions[0]) + "\n")
            with open(res_path, "w") as f:
                json.dump(results, f)
            store = swebench_adapter.load_swebench_predictions(pred_path, res_path)
            assert len(store) == 1
            assert store._records[0].success is False


class TestToolSequenceExtraction:
    def test_maps_common_actions(self):
        steps = [
            {"action": "read"},
            {"action": "bash"},
            {"action": "edit"},
            {"action": "submit"},
        ]
        seq = swebench_adapter._extract_tool_sequence(steps)
        assert seq == ["read_file", "run_shell_command", "write_file", "submit"]

    def test_fallback_to_shell(self):
        steps = [{"action": "unknown"}]
        seq = swebench_adapter._extract_tool_sequence(steps)
        assert seq == ["run_shell_command"]


class TestLoadSweAgentTrajectory:
    def test_loads_trajectory_file(self):
        steps = [
            {"action": "read"},
            {"action": "bash"},
            {"action": "edit"},
        ]
        with tempfile.TemporaryDirectory() as d:
            traj_path = os.path.join(d, "traj.json")
            with open(traj_path, "w") as f:
                json.dump({"steps": steps}, f)
            rec = swebench_adapter.load_swe_agent_trajectory(
                traj_path, "django-1", "swe-agent", score=1.0
            )
            assert rec is not None
            assert rec.trial_id == "django-1"
            assert rec.score == pytest.approx(1.0)
            assert rec.tool_sequence == ["read_file", "run_shell_command", "write_file"]

    def test_missing_file_returns_none(self):
        rec = swebench_adapter.load_swe_agent_trajectory("/nonexistent", "t", "a")
        assert rec is None
